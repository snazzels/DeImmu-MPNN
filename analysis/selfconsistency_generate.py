#!/usr/bin/env python
"""Stage 1/3 of the ESMFold self-consistency benchmark (roadmap step P1.3).

WHY THIS EXISTS (read before reusing eval_deimmunisation.csv instead)
---------------------------------------------------------------------
Self-consistency (scTM/RMSD) requires the *exact backbone each sequence was
designed onto*. The Track 7 run (eval_deimmunisation.csv) stored the designed
sequences but NOT the backbones, and the PDB loader is non-deterministic
(get_data_loaders_pdb uses shuffle=True; loader_pdb samples a random chain and a
random biological assembly per cluster, and no seed was set). The backbones
behind those 2,000 sequences are therefore unrecoverable. So we regenerate a
fresh, seeded, matched sample and SAVE the backbone alongside every design.

This is the standard ProteinMPNN/RFdiffusion self-consistency protocol: design
sequence onto backbone X -> fold with a structure predictor -> measure TM/RMSD of
the prediction back to X. Here X is the native backbone ProteinMPNN redesigns.

SCOPE: single-chain proteins only, length-capped, because ESMFold2 (Stage 2) OOMs
on 24 GB above ~160 residues and complex/assembly alignment is deferred. 65% of
the test proteins are single-chain. Multi-chain assemblies are
future work.

Each design also gets its MHC-II presented-window count (same predictor as
Track 7), so Stage 3 can draw a foldability-vs-immunogenicity Pareto.

Outputs
-------
data/output/selfconsistency/designs.csv        one row per (protein, model, seq)
data/output/selfconsistency/refs/<name>.npz    native backbone: ca, bb(N,CA,C,O), mask, seq
The 'native' pseudo-model (seq_idx = -1) stores the native sequence as a positive
control: folding it should recover the native backbone at high TM.
"""

import os, sys, argparse, csv, time, random, hashlib
import numpy as np
import torch

PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # CAPE_MPNN/
sys.path.insert(0, os.path.join(PF, "libs"))

import kit
import kit.globals as G
from kit.data import DD, Split

G.ENV = DD()
G.ENV.PROJECT   = PF
G.ENV.INPUT     = os.path.join(PF, "data", "input")
G.ENV.ARTEFACTS = os.path.join(PF, "artefacts")
G.ENV.CONFIG    = os.path.join(PF, "configs")
G.JOB = DD(); G.JOB.ID = None
G.PROJECT_ENV = G.ENV

kit.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {kit.DEVICE}")

from CAPE.MPNN.model import CapeMPNN
from CAPE.MPNN.data.utils import get_data_loaders_pdb, get_pdbs, sample_dict_and_probs
from CAPE.MPNN.overwrite import tied_featurize
from CAPE.MPNN.data.aux import S_to_seqs
from kit.bioinf.immuno.mhc_2 import Mhc2PredictorPwm, MHC_2_PEPTIDE_LENGTHS

MODEL_HASH   = "057975d5"
CKPT         = "best"
BASE_MODEL   = "v_48_020"
# Resolve the PDB dataset portably — this tool must also run on the cluster, where the
# local path does not exist (same hardcoded-path trap that killed the sweep eval on
# 2026-08-18). LOCAL path first so a local run keeps its fast on-disk copy and its
# exact seed-based protein selection unchanged; then a node-local stage (PDB_CACHE_DIR);
# then the network-mount copy under data/input (the cluster fallback). All copies are
# byte-identical (md5-verified), so seed-based selection is reproducible across machines.
DATA_PATH    = next(
    (p for p in (os.environ.get("PDB_CACHE_DIR", ""),
                 os.environ.get("PDB_CACHE_DIR"),
                 os.path.join(PF, "data", "input", "CAPE-MPNN", "pdb_2021aug02"))
     if p and os.path.isdir(p)),
    os.path.join(PF, "data", "input", "CAPE-MPNN", "pdb_2021aug02"))
MHC2_ALLELES = "DRB1_0101+DRB1_0301+DRB1_0401+DRB1_0701+DRB1_1101+DRB1_1501"
PWM_DIR      = os.path.join(PF, "data", "input", "immuno", "mhc_2", "pwm")
TEMPERATURE  = 0.2
RESCUT       = 3.5


def score_mhc2(seq, predictor):
    presented = predictor.seq_presented(seq, alleles=MHC2_ALLELES, lengths=MHC_2_PEPTIDE_LENGTHS)
    return len(presented)


def main(a):
    t0 = time.time()
    OUT_DIR = a.outdir
    OUT_CSV = os.path.join(OUT_DIR, "designs.csv")
    REF_DIR = os.path.join(OUT_DIR, "refs")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    os.makedirs(REF_DIR, exist_ok=True)

    CapeMPNN.base_model_pt_dir_path   = os.path.join(G.ENV.INPUT, "CAPE-MPNN", "vanilla_model_weights")
    CapeMPNN.base_model_yaml_dir_path = os.path.join(G.ENV.INPUT, "CAPE-MPNN", "base_hparams")
    print("Loading base model…");   model_base = CapeMPNN.from_file(BASE_MODEL)
    ft_path = os.path.join(PF, "artefacts", "CAPE-MPNN", "models", a.model_hash, "ckpts", f"{a.ckpt}.pt")
    print(f"  fine-tuned = {a.model_hash}/{a.ckpt}.pt")
    print("Loading fine-tuned model…"); model_ft = CapeMPNN.from_file(ft_path)
    print("Loading MHC-II PWM predictor…"); predictor = Mhc2PredictorPwm(PWM_DIR)

    print(f"Building TEST loader ({DATA_PATH})…")
    dl = get_data_loaders_pdb(DATA_PATH, [Split.TEST], rescut=RESCUT, debug=False)
    print(f"Pulling a pool of up to {a.pool} structures (len <= {a.max_len})…")
    pool = get_pdbs(dl[Split.TEST], 1, a.max_len, a.pool)
    print(f"  got {len(pool)} structures; filtering to single-chain, {a.min_len}-{a.max_len} aa")

    # ── protein selection ─────────────────────────────────────────────────────
    # --protein_list pins the EXACT sample. Without it the draw depends on
    # get_pdbs()/DataLoader(shuffle=True) worker RNG, which --seed does NOT control:
    # every one of the 19 selfconsistency_boltz_sweep_* runs got a DISTINCT protein
    # set. Within-run base-vs-ft is still paired and valid; CROSS-run comparison is
    # not: never compare numbers across unmatched protein samples.
    if getattr(a, "protein_list", None):
        want = [ln.strip() for ln in open(a.protein_list) if ln.strip()
                and not ln.startswith("#")]
        by_name = {}
        for pd in pool:
            n_chains = len([k for k in pd if k.startswith("seq_chain_")])
            if n_chains == 1 and pd["name"] not in by_name:
                by_name[pd["name"]] = pd
        missing = [p for p in want if p not in by_name]
        if missing:
            sys.exit(
                f"--protein_list asks for {len(missing)} protein(s) not in the pool: "
                f"{missing[:8]}{'…' if len(missing) > 8 else ''}\n"
                f"  Pool held {len(by_name)} single-chain candidates. Raise --pool "
                f"(currently {a.pool}) or check --min_len/--max_len ({a.min_len}-{a.max_len}).\n"
                "  A pinned comparison must not silently drop proteins.")
        selected = [by_name[p] for p in want]
        print(f"  selected {len(selected)} single-chain proteins PINNED from {a.protein_list}")
    else:
        selected = []
        for pd in pool:
            n_chains = len([k for k in pd if k.startswith("seq_chain_")])
            L = len(pd["seq"])
            if n_chains == 1 and a.min_len <= L <= a.max_len:
                selected.append(pd)
            if len(selected) >= a.n_proteins:
                break
        print(f"  selected {len(selected)} single-chain proteins")
        print("  ⚠️  Sample NOT pinned: --seed does not control the draw here. This run's\n"
              "      base-vs-fine-tuned delta is paired and valid, but it is NOT comparable\n"
              "      to another model's run. Pass --protein_list for cross-model work.",
              flush=True)
    if not selected:
        sys.exit("No single-chain proteins in pool — raise --pool.")

    _pset_md5 = hashlib.md5("\n".join(sorted(p["name"] for p in selected)).encode()).hexdigest()[:12]
    print(f"  protein_set_md5 = {_pset_md5}  (n={len(selected)})")
    with open(os.path.join(OUT_DIR, "protein_set.txt"), "w") as fh:
        fh.write("\n".join(p["name"] for p in selected) + "\n")

    rows = []
    for pi, pd in enumerate(selected):
        name = pd["name"]
        batch = [pd]
        bs = list(tied_featurize(batch, kit.DEVICE, None))
        X, S, mask, chain_enc = bs[0], bs[1], bs[2], bs[5]
        L = int(mask[0].sum().item())               # valid residues
        Lfull = X.shape[1]
        # single chain -> residues are contiguous; take the first Lfull, keep mask
        bb = X[0].detach().cpu().numpy()             # (Lfull, 4, 3)  N,CA,C,O
        m  = mask[0].detach().cpu().numpy()          # (Lfull,)
        native_seq = S_to_seqs(S, chain_enc)[0]      # no '/', single chain
        np.savez(os.path.join(REF_DIR, f"{name}.npz"),
                 bb=bb.astype(np.float32), mask=m.astype(np.float32), seq=native_seq)

        # native control row
        rows.append(dict(design_id=f"{name}|native|-1", protein_name=name,
                         protein_length=len(native_seq), model="native", seq_idx=-1,
                         sequence=native_seq, mhc2_total=score_mhc2(native_seq, predictor)))
        # designed rows
        for model_name, model in [("base", model_base), ("finetuned", model_ft)]:
            for k in range(a.k_seqs):
                sample, _ = sample_dict_and_probs(model, model, bs, temperature=TEMPERATURE)
                seq = S_to_seqs(sample["S"], bs[5])[0]
                rows.append(dict(design_id=f"{name}|{model_name}|{k}", protein_name=name,
                                 protein_length=len(seq), model=model_name, seq_idx=k,
                                 sequence=seq, mhc2_total=score_mhc2(seq, predictor)))
        print(f"  [{pi+1}/{len(selected)}] {name}  L={len(native_seq)}  done", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    fields = ["design_id","protein_name","protein_length","model","seq_idx","sequence","mhc2_total"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {OUT_CSV}  ({len(rows)} rows, {len(selected)} proteins)")
    print(f"Refs in {REF_DIR}")
    print(f"Runtime {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_proteins", type=int, default=50)
    p.add_argument("--k_seqs",     type=int, default=3)
    p.add_argument("--pool",       type=int, default=400, help="structures to pull before single-chain filter")
    p.add_argument("--min_len",    type=int, default=40)
    p.add_argument("--max_len",    type=int, default=150, help="ESMFold2 OOMs above ~160 aa on 24GB")
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--outdir",     default=os.path.join(PF, "data", "output", "selfconsistency"))
    p.add_argument("--protein_list", default=None,
                   help="file with one protein name per line, pinning the EXACT sample. "
                        "REQUIRED for any run to be compared with another model's run: "
                        "--seed does NOT control this draw (get_pdbs/DataLoader shuffle=True "
                        "with worker RNG), which is why all 19 selfconsistency_boltz_sweep_* "
                        "runs landed on distinct protein sets.")
    p.add_argument("--model_hash", default=MODEL_HASH)
    p.add_argument("--ckpt",       default=CKPT)
    main(p.parse_args())
