#!/usr/bin/env python
"""Fixed-set self-consistency generation: select ONE protein set, then generate
designs for base + all three lambda models on that SAME set (so counts + proteins
match across lambda). Adapted from tools/selfconsistency_generate.py; PF hardcoded
so this can live outside the repo tree. Reads checkpoints only (no training)."""
import os, sys, argparse, csv, time, random
import numpy as np
import torch

PF = os.environ["CAPE_ROOT"] + "/CAPE_MPNN"
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
print(f"Device: {kit.DEVICE}", flush=True)

from CAPE.MPNN.model import CapeMPNN
from CAPE.MPNN.data.utils import get_data_loaders_pdb, get_pdbs, sample_dict_and_probs
from CAPE.MPNN.overwrite import tied_featurize
from CAPE.MPNN.data.aux import S_to_seqs
from kit.bioinf.immuno.mhc_2 import Mhc2PredictorPwm, MHC_2_PEPTIDE_LENGTHS

BASE_MODEL   = "v_48_020"
DATA_PATH    = os.environ.get("PDB_CACHE_DIR", "")
MHC2_ALLELES = "DRB1_0101+DRB1_0301+DRB1_0401+DRB1_0701+DRB1_1101+DRB1_1501"
PWM_DIR      = os.path.join(PF, "data", "input", "immuno", "mhc_2", "pwm")
TEMPERATURE  = 0.2
RESCUT       = 3.5
# label -> checkpoint hash  (mapping confirmed from eval_deimmunisation_*_summary.txt)
FT_MODELS = [("ft_l0", "54e1642c"), ("ft_l05", "057975d5"), ("ft_l10", "7e908919")]


def score_mhc2(seq, predictor):
    return len(predictor.seq_presented(seq, alleles=MHC2_ALLELES, lengths=MHC_2_PEPTIDE_LENGTHS))


def main(a):
    t0 = time.time()
    OUT_DIR = a.outdir
    REF_DIR = os.path.join(OUT_DIR, "refs")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    os.makedirs(REF_DIR, exist_ok=True)

    CapeMPNN.base_model_pt_dir_path   = os.path.join(G.ENV.INPUT, "CAPE-MPNN", "vanilla_model_weights")
    CapeMPNN.base_model_yaml_dir_path = os.path.join(G.ENV.INPUT, "CAPE-MPNN", "base_hparams")
    print("Loading base model…", flush=True); model_base = CapeMPNN.from_file(BASE_MODEL)
    ft = {}
    for label, h in FT_MODELS:
        p = os.path.join(PF, "artefacts", "CAPE-MPNN", "models", h, "ckpts", "best.pt")
        print(f"Loading {label} = {h}/best.pt", flush=True); ft[label] = CapeMPNN.from_file(p)
    print("Loading MHC-II PWM predictor…", flush=True); predictor = Mhc2PredictorPwm(PWM_DIR)

    print(f"Building TEST loader ({DATA_PATH})…", flush=True)
    dl = get_data_loaders_pdb(DATA_PATH, [Split.TEST], rescut=RESCUT, debug=False)
    print(f"Pulling a pool of up to {a.pool} structures (len <= {a.max_len})…", flush=True)
    pool = get_pdbs(dl[Split.TEST], 1, a.max_len, a.pool)
    selected = []
    for pd in pool:
        n_chains = len([k for k in pd if k.startswith("seq_chain_")])
        L = len(pd["seq"])
        if n_chains == 1 and a.min_len <= L <= a.max_len:
            selected.append(pd)
        if len(selected) >= a.n_proteins:
            break
    print(f"  selected {len(selected)} single-chain proteins ({a.min_len}-{a.max_len} aa)", flush=True)
    if not selected:
        sys.exit("No single-chain proteins in pool — raise --pool.")

    rows = []
    for pi, pd in enumerate(selected):
        name = pd["name"]
        bs = list(tied_featurize([pd], kit.DEVICE, None))
        X, S, chain_enc = bs[0], bs[1], bs[5]
        m = bs[2][0].detach().cpu().numpy()
        bb = X[0].detach().cpu().numpy()
        native_seq = S_to_seqs(S, chain_enc)[0]
        np.savez(os.path.join(REF_DIR, f"{name}.npz"),
                 bb=bb.astype(np.float32), mask=m.astype(np.float32), seq=native_seq)
        rows.append(dict(design_id=f"{name}|native|-1", protein_name=name,
                         protein_length=len(native_seq), model="native", seq_idx=-1,
                         sequence=native_seq, mhc2_total=score_mhc2(native_seq, predictor)))
        gen = [("base", model_base)] + [(lab, ft[lab]) for lab, _ in FT_MODELS]
        for model_name, model in gen:
            for k in range(a.k_seqs):
                sample, _ = sample_dict_and_probs(model, model, bs, temperature=TEMPERATURE)
                seq = S_to_seqs(sample["S"], bs[5])[0]
                rows.append(dict(design_id=f"{name}|{model_name}|{k}", protein_name=name,
                                 protein_length=len(seq), model=model_name, seq_idx=k,
                                 sequence=seq, mhc2_total=score_mhc2(seq, predictor)))
        print(f"  [{pi+1}/{len(selected)}] {name} L={len(native_seq)} done", flush=True)

    fields = ["design_id","protein_name","protein_length","model","seq_idx","sequence","mhc2_total"]
    with open(os.path.join(OUT_DIR, "designs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"\nWrote designs.csv ({len(rows)} rows, {len(selected)} proteins) to {OUT_DIR}", flush=True)
    print(f"Runtime {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_proteins", type=int, default=40)
    p.add_argument("--k_seqs",     type=int, default=3)
    p.add_argument("--pool",       type=int, default=600)
    p.add_argument("--min_len",    type=int, default=40)
    p.add_argument("--max_len",    type=int, default=150)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--outdir",     required=True)
    main(p.parse_args())
