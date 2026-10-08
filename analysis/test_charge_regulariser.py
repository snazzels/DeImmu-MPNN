#!/usr/bin/env python
"""Prototype/calibration for the DPO charge regulariser (curbs the glutamate crutch).

Imports the actual reward-side penalty (CAPE.MPNN.data.preference_pair.charge_penalty)
and shows, on the real base vs fine-tuned designs, (1) that it correctly targets the
acidic fine-tuned sequences and barely touches base, and (2) how the penalty magnitude
compares to the MHC-II epitope reduction — which is what sets a sensible lambda.
No training or GPU needed.
"""
import os, sys
import numpy as np, pandas as pd

PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PF, "libs"))
from CAPE.MPNN.data.preference_pair import charge_penalty

EVAL = os.path.join(PF, "data", "output", "eval_deimmunisation.csv")
TARGET = 0.15
LAMBDAS = [0.25, 0.5, 1.0, 2.0]


def acidic_frac(seq):
    s = seq.replace("/", "")
    return (s.count("D") + s.count("E")) / len(s)


def main():
    df = pd.read_csv(EVAL)
    df["acidic"] = df.sequence.apply(acidic_frac)
    df["len"] = df.sequence.str.replace("/", "", regex=False).str.len()
    b = df[df.model == "base"]; f = df[df.model == "finetuned"]

    print("="*72); print("CHARGE REGULARISER — behaviour & lambda calibration"); print("="*72)
    print(f"acidic (D+E) fraction:  base {b.acidic.mean():.3f}   fine-tuned {f.acidic.mean():.3f}"
          f"   (target {TARGET})")
    print(f"MHC-II windows/seq:     base {b.n_presented_total.mean():.1f}   "
          f"fine-tuned {f.n_presented_total.mean():.1f}   "
          f"(reduction {b.n_presented_total.mean()-f.n_presented_total.mean():.1f})")
    print()
    print(f"{'lambda':>7}  {'penalty base':>13}  {'penalty ft':>11}  {'ft penalty as % of':>20}")
    print(f"{'':>7}  {'(windows-eq)':>13}  {'(windows-eq)':>11}  {'MHC-II reduction':>20}")
    print("-"*60)
    red = b.n_presented_total.mean() - f.n_presented_total.mean()
    for lam in LAMBDAS:
        cfg = {"weight": lam, "target": TARGET}
        pb = b.apply(lambda r: charge_penalty(r.sequence, cfg), axis=1).mean()
        pf = f.apply(lambda r: charge_penalty(r.sequence, cfg), axis=1).mean()
        print(f"{lam:>7.2f}  {pb:>13.2f}  {pf:>11.2f}  {100*pf/red:>19.0f}%")
    print()
    print("Reading it: 'penalty' is in the same units as the reward (presented windows).")
    print("At a given lambda, a fine-tuned-style acidic design pays ~penalty_ft, which offsets")
    print("that fraction of the epitope reduction it bought via charge. base pays ~0 (it is")
    print("already near the natural acidic target), so the regulariser does not distort the")
    print("base distribution — it only pushes back on ADDED acidic content.")
    print()
    print("Suggested sweep: lambda in {0, 0.25, 0.5, 1.0}. lambda=0 reproduces the current")
    print("model; increasing lambda trades epitope reduction for a smaller charge shift. The")
    print("v2 run should re-measure BOTH the MHC-II reduction (Track 7) and the charge shift")
    print("(quantify_charge_shift.py) across the sweep to find the knee.")
    print("="*72)


if __name__ == "__main__":
    main()
