#!/usr/bin/env python
"""Sequence-recovery re-run on the DEPOSITED lambda=0 retrain (54e1642c).

WHY THIS EXISTS
---------------
The original seq_recovery_summary.txt (-0.67 pp, 45 proteins) was computed on
2026-06-02 against DPO fine-tuned hash 057975d5 -- which at that time was the
lambda=0 model, but was later overwritten (2026-07-14) by the lambda=0.5 run.
That exact checkpoint no longer exists, so the number has no surviving
checkpoint. This regenerates it against the deposited lambda=0 retrain
54e1642c/best.pt (the same checkpoint behind the deposited 34.8% headline),
using a freshly seeded, within-run-matched sample (base v_48_020 vs 54e1642c on
identical backbones). No folding is involved -- recovery is a pure
sequence-identity metric.

Protocol mirrors the original: single-chain test proteins, K sequences per model
at temperature 0.2, recovery = fraction of valid residues where the sampled
amino acid matches the native. Per-protein means are compared with a paired
t-test and Wilcoxon signed-rank test.
"""

import os, sys, argparse, csv, time, random
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

MODEL_HASH  = "54e1642c"
CKPT        = "best"
BASE_MODEL  = "v_48_020"
# PDB dataset location — same resolution chain as evaluate_deimmunisation.py /
# cape-mpnn.py: PDB_CACHE_DIR (node-local stage) -> network-mount copy -> local path.
DATA_PATH   = next(
    (p for p in (os.environ.get("PDB_CACHE_DIR"),
                 os.path.join(PF, "data", "input", "CAPE-MPNN", "pdb_2021aug02"),
                 os.environ.get("PDB_CACHE_DIR", "")) if p and os.path.isdir(p)),
    os.path.join(PF, "data", "input", "CAPE-MPNN", "pdb_2021aug02"))
TEMPERATURE = 0.2
RESCUT      = 3.5


def recovery(sampled, native):
    """Fraction of positions where sampled AA == native AA (equal length)."""
    assert len(sampled) == len(native), (len(sampled), len(native))
    m = [a == b for a, b in zip(sampled, native)]
    return sum(m) / len(m)


def main(a):
    t0 = time.time()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    CapeMPNN.base_model_pt_dir_path   = os.path.join(G.ENV.INPUT, "CAPE-MPNN", "vanilla_model_weights")
    CapeMPNN.base_model_yaml_dir_path = os.path.join(G.ENV.INPUT, "CAPE-MPNN", "base_hparams")
    print("Loading base model...");   model_base = CapeMPNN.from_file(BASE_MODEL)
    ft_path = os.path.join(PF, "artefacts", "CAPE-MPNN", "models", a.model_hash, "ckpts", f"{a.ckpt}.pt")
    print(f"  fine-tuned = {a.model_hash}/{a.ckpt}.pt")
    print("Loading fine-tuned model..."); model_ft = CapeMPNN.from_file(ft_path)

    print(f"Building TEST loader ({DATA_PATH})...")
    dl = get_data_loaders_pdb(DATA_PATH, [Split.TEST], rescut=RESCUT, debug=False)
    print(f"Pulling a pool of up to {a.pool} structures (len <= {a.max_len})...")
    pool = get_pdbs(dl[Split.TEST], 1, a.max_len, a.pool)

    selected = []
    for pd in pool:
        n_chains = len([k for k in pd if k.startswith("seq_chain_")])
        L = len(pd["seq"])
        if n_chains == 1 and a.min_len <= L <= a.max_len:
            selected.append(pd)
        if len(selected) >= a.n_proteins:
            break
    print(f"  selected {len(selected)} single-chain proteins ({a.min_len}-{a.max_len} aa)")
    if not selected:
        sys.exit("No single-chain proteins in pool - raise --pool.")

    rows = []
    for pi, pd in enumerate(selected):
        name = pd["name"]
        batch = [pd]
        bs = list(tied_featurize(batch, kit.DEVICE, None))
        native_seq = S_to_seqs(bs[1], bs[5])[0]
        for model_name, model in [("base", model_base), ("finetuned", model_ft)]:
            for k in range(a.k_seqs):
                sample, _ = sample_dict_and_probs(model, model, bs, temperature=TEMPERATURE)
                seq = S_to_seqs(sample["S"], bs[5])[0]
                rows.append(dict(protein_name=name, protein_length=len(seq),
                                 model=model_name, seq_idx=k,
                                 seq_recovery=recovery(seq, native_seq)))
        print(f"  [{pi+1}/{len(selected)}] {name}  L={len(native_seq)}  done", flush=True)

    os.makedirs(a.outdir, exist_ok=True)
    out_csv = os.path.join(a.outdir, "seq_recovery.csv")
    fields = ["protein_name", "protein_length", "model", "seq_idx", "seq_recovery"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    # ---- aggregate + stats (per-protein means) ----
    import pandas as pd
    from scipy import stats
    df = pd.DataFrame(rows)
    per = df.groupby(["protein_name", "model"])["seq_recovery"].mean().unstack()
    per = per.dropna(subset=["base", "finetuned"])
    b, ft = per["base"].values, per["finetuned"].values
    t, p = stats.ttest_rel(b, ft)
    W, pw = stats.wilcoxon(b, ft)
    all_b = df[df.model == "base"]["seq_recovery"].values
    all_ft = df[df.model == "finetuned"]["seq_recovery"].values

    lines = []
    def emit(s=""):
        print(s); lines.append(s)
    emit("=" * 65)
    emit("SEQUENCE RECOVERY CHECK  (re-run on deposited lambda=0 retrain)")
    emit(f"Base {BASE_MODEL} vs DPO fine-tuned {a.model_hash}/{a.ckpt}  (temperature={TEMPERATURE})")
    emit("=" * 65)
    emit(f"Proteins matched: {len(per)}")
    emit(f"Sequences/model:  {a.k_seqs}")
    emit("")
    emit("-- Per-protein mean recovery -----------------------------------")
    emit(f"  Base:       {100*b.mean():.2f}% +/- {100*b.std():.2f}%  (median {100*np.median(b):.2f}%)")
    emit(f"  Fine-tuned: {100*ft.mean():.2f}% +/- {100*ft.std():.2f}%  (median {100*np.median(ft):.2f}%)")
    emit(f"  Change:     {100*(ft.mean()-b.mean()):+.2f} pp")
    emit(f"  Paired t-test: t={t:.3f}, p={p:.5f}")
    emit(f"  Wilcoxon:      W={W:.0f}, p={pw:.5f}")
    emit(f"  Proteins where FT >= base: {int((ft>=b).sum())}/{len(per)}")
    emit("")
    emit("-- All sequences (not per-protein) -----------------------------")
    emit(f"  Base:       {100*all_b.mean():.2f}% +/- {100*all_b.std():.2f}%")
    emit(f"  Fine-tuned: {100*all_ft.mean():.2f}% +/- {100*all_ft.std():.2f}%")
    emit("=" * 65)

    with open(os.path.join(a.outdir, "seq_recovery_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {out_csv} and seq_recovery_summary.txt to {a.outdir}")
    print(f"Runtime {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_proteins", type=int, default=45)
    p.add_argument("--k_seqs",     type=int, default=10)
    p.add_argument("--pool",       type=int, default=400)
    p.add_argument("--min_len",    type=int, default=40)
    p.add_argument("--max_len",    type=int, default=400, help="original recovery set spanned up to 366 aa")
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--model_hash", default=MODEL_HASH)
    p.add_argument("--ckpt",       default=CKPT)
    p.add_argument("--outdir",     default=os.path.join(PF, "data", "output", "seq_recovery_54e1642c"))
    main(p.parse_args())
