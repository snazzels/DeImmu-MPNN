import os, sys, re, json, types, warnings
warnings.filterwarnings("ignore")
import numpy as np

# --- shims for legacy vendored alphafold deps ---
from Bio.Data.PDBData import protein_letters_3to1_extended
shim = types.ModuleType("Bio.Data.SCOPData")
shim.protein_letters_3to1 = protein_letters_3to1_extended
sys.modules["Bio.Data.SCOPData"] = shim

REPO = os.path.join(os.environ["CAPE_ROOT"],
                    "CAPE_MPNN/external/repos/dl_binder_design/af2_initial_guess")
sys.path.insert(0, REPO)

import jax
from alphafold.common import residue_constants, protein, confidence
from alphafold.data import pipeline
from alphafold.model import data, config, model
sys.path.insert(0, REPO)  # af2_util imports pyrosetta at module level; avoid importing it, inline needed fns instead

DEST = os.environ.get("AF2IG_WORK", "/tmp/binder_panel2")
OUT  = os.path.dirname(os.path.abspath(__file__))
PANEL = {"8t5e": ("A", ["B"]), "9cc5": ("A", ["B"]), "9nzh": ("A", ["B"]),
         "9cce": ("A", ["D"]), "9nds": ("A", ["C", "D", "E"])}
TOPN = 24
NUM_RECYCLES = 3
MAX_AMIDE_DIST = 3.0

# ---- reimplemented af2_util functions (pyrosetta-free) ----
def generate_template_features(seq, all_atom_positions, all_atom_masks, residue_mask):
    L = len(seq)
    tpos = np.zeros((L, residue_constants.atom_type_num, 3))
    tmask = np.zeros((L, residue_constants.atom_type_num))
    tseq = ['-'] * L
    conf = [-1] * L
    for i in range(L):
        if residue_mask[i]:
            tpos[i] = all_atom_positions[i]
            tmask[i] = all_atom_masks[i]
            tseq[i] = seq[i]
            conf[i] = 9
    tseq = ''.join(tseq)
    taatype = residue_constants.sequence_to_onehot(tseq, residue_constants.HHBLITS_AA_TO_ID)
    return {
        'template_all_atom_positions': tpos[None].astype(np.float32),
        'template_all_atom_masks': tmask[None].astype(np.float32),
        'template_sequence': [tseq.encode()],
        'template_aatype': taatype[None].astype(np.float32),
        'template_confidence_scores': np.array(conf)[None],
        'template_domain_names': ['none'.encode()],
        'template_release_date': ['none'.encode()],
    }

def parse_initial_guess(all_atom_positions):
    import jax.numpy as jnp
    return jnp.array(all_atom_positions.astype(np.float32))

def check_residue_distances(all_positions, all_positions_mask, max_amide_distance):
    breaks = []
    c_i, n_i = residue_constants.atom_order['C'], residue_constants.atom_order['N']
    prev_unmasked, prev_c = False, None
    for i, (coords, mask) in enumerate(zip(all_positions, all_positions_mask)):
        this_unmasked = bool(mask[c_i]) and bool(mask[n_i])
        if this_unmasked:
            this_n = coords[n_i]
            if prev_unmasked:
                d = np.linalg.norm(this_n - prev_c)
                if d > max_amide_distance:
                    breaks.append(i)
            prev_c = coords[c_i]
        prev_unmasked = this_unmasked
    return breaks

def insert_truncations(residue_index, breaks):
    idx = residue_index.copy()
    for b in breaks:
        idx[b:] += 200
    return idx

# ---- pdb parsing (chain by chain, biotite-free: use vendored alphafold protein parser) ----
def parse_chain(pdb_path, chain_id):
    pdb_str = open(pdb_path).read()
    prot = protein.from_pdb_string(pdb_str, chain_id=chain_id)
    seq = ''.join(residue_constants.restypes_with_x[a] if a < 20 else 'X' for a in prot.aatype)
    return prot.atom_positions, prot.atom_mask, seq

def build_complex_arrays(pdb_path, binder_chain, target_chains):
    bpos, bmask, bseq = parse_chain(pdb_path, binder_chain)
    tpos_list, tmask_list, tseq = [], [], ''
    for tc in target_chains:
        p, m, s = parse_chain(pdb_path, tc)
        tpos_list.append(p); tmask_list.append(m); tseq += s
    all_pos = np.concatenate([bpos] + tpos_list, axis=0)
    all_mask = np.concatenate([bmask] + tmask_list, axis=0)
    binderlen = bpos.shape[0]
    return all_pos, all_mask, bseq, tseq, binderlen

def recs(fa):
    out = []; cur = None
    for l in open(fa):
        l = l.rstrip("\n")
        if l.startswith(">"):
            if cur: out.append(cur)
            cur = [l, ""]
        elif cur:
            cur[1] += l.strip()
    if cur: out.append(cur)
    return out

def binder_variants(pid):
    v = []
    base_recs = recs(f"{DEST}/redesign/{pid}_base/seqs/{pid}_cplx.fa")
    v.append(("native", base_recs[0][1].replace("X", "")))
    for mdl in ("base", "cape"):
        r = recs(f"{DEST}/redesign/{pid}_{mdl}/seqs/{pid}_cplx.fa")
        items = []
        for hdr, seq in r[1:]:
            m = re.search(r"global_score=([0-9.]+)", hdr)
            items.append((float(m.group(1)) if m else 999, seq.replace("X", "")))
        items.sort(key=lambda x: x[0])
        for k, (gs, seq) in enumerate(items[:TOPN]):
            v.append((f"{mdl}_{k+1}", seq))
    return v

def make_model():
    mc = config.model_config("model_1_ptm")
    mc.data.eval.num_ensemble = 1
    mc.data.common.num_recycle = NUM_RECYCLES
    mc.model.num_recycle = NUM_RECYCLES
    mc.model.embeddings_and_evoformer.initial_guess = True
    mc.data.common.max_extra_msa = 5
    mc.data.eval.max_msa_clusters = 5
    params = data.get_model_haiku_params(model_name="model_1_ptm", data_dir=os.path.join(REPO, "model_weights"))
    return model.RunModel(mc, params)

SANITIZED = {"9nds": os.path.join(OUT, "complex_sanitized", "9nds_cplx.pdb")}

def main(pids):
    runner = make_model()
    out_path = os.path.join(OUT, "af2ig_native_results.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for pid in pids:
        binder_chain, target_chains = PANEL[pid]
        cplx = SANITIZED.get(pid, f"{DEST}/complex/{pid}_cplx.pdb")
        all_pos, all_mask, native_binder_seq, target_seq, binderlen = build_complex_arrays(cplx, binder_chain, target_chains)
        targetlen = all_pos.shape[0] - binderlen
        residue_mask = [False] * binderlen + [True] * targetlen
        print(f"\n=== {pid}: binder_len={binderlen} target_len={targetlen} ===", flush=True)
        results[pid] = {}
        breaks = check_residue_distances(all_pos, all_mask, MAX_AMIDE_DIST)
        initial_guess = parse_initial_guess(all_pos)
        for label, bseq in binder_variants(pid):
            if len(bseq) != binderlen:
                print(f"  {label}: len {len(bseq)} != {binderlen}, skip", flush=True); continue
            seq = bseq + target_seq
            template_dict = generate_template_features(seq, all_pos, all_mask, residue_mask)
            feature_dict = {
                **pipeline.make_sequence_features(sequence=seq, description="none", num_res=len(seq)),
                **pipeline.make_msa_features(msas=[[seq]], deletion_matrices=[[[0]*len(seq)]]),
                **template_dict,
            }
            feature_dict['residue_index'] = insert_truncations(feature_dict['residue_index'], breaks)
            processed = runner.process_features(feature_dict, random_seed=0)
            result = runner.predict(processed, initial_guess=initial_guess)
            plddt = np.asarray(result['plddt'])
            pae = np.asarray(result['predicted_aligned_error'])
            pae_int = 0.5 * (pae[:binderlen, binderlen:].mean() + pae[binderlen:, :binderlen].mean())
            plddt_binder = float(plddt[:binderlen].mean())
            ptm = float(np.asarray(result['ptm']))
            results[pid][label] = dict(pae_interaction=round(float(pae_int), 2),
                                        plddt_binder=round(plddt_binder, 3), ptm=round(ptm, 3))
            print(f"  {label:9} pae_int={pae_int:6.2f} A  plddt_binder={plddt_binder:.3f}  ptm={ptm:.3f}", flush=True)
        json.dump(results, open(out_path, "w"), indent=2)
        print(f"  [checkpointed after {pid}]", flush=True)
    print("\nsaved af2ig_native_results.json")

if __name__ == "__main__":
    pids = sys.argv[1:] if len(sys.argv) > 1 else list(PANEL)
    main(pids)
