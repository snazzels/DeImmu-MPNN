#!/usr/bin/env python
"""Quantify the 'glutamate crutch': how much does DPO de-immunisation shift charge?

Track 17's mechanism analysis found de-immunisation swaps alanine for glutamate
(acidic content 14%->22%) and that this drives the DQ reduction. This script asks
whether that is a biophysical liability: it computes net charge, isoelectric point
(pI), GRAVY hydropathy, and acidic/basic fractions for base vs fine-tuned designs,
and — using the self-consistency set, which also stores the native sequence — the
full native -> base -> fine-tuned trajectory (is it the fine-tuning that adds charge,
or was base ProteinMPNN already shifted?).

Inputs
  data/output/eval_deimmunisation.csv        base + finetuned, 100 proteins (primary)
  data/output/selfconsistency/designs.csv    native + base + finetuned, 37 proteins (context)

Outputs (data/output/)
  charge_shift_summary.txt / .json
  charge_shift_figure.png
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats
from Bio.SeqUtils.ProtParam import ProteinAnalysis

PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PF, "data", "output")
STD = set("ACDEFGHIKLMNPQRSTVWY")


def clean(seq):
    return "".join(c for c in seq.replace("/", "") if c in STD)


def props(seq):
    s = clean(seq)
    pa = ProteinAnalysis(s)
    n = len(s)
    nD = s.count("D"); nE = s.count("E"); nK = s.count("K"); nR = s.count("R")
    return dict(
        length=n,
        net_charge_pH74=pa.charge_at_pH(7.4),
        net_charge_per100=100.0 * (nK + nR - nD - nE) / n,
        pI=pa.isoelectric_point(),
        gravy=pa.gravy(),
        frac_acidic=100.0 * (nD + nE) / n,
        frac_basic=100.0 * (nK + nR) / n,
    )


def enrich(df):
    p = df["sequence"].apply(props).apply(pd.Series)
    return pd.concat([df[["protein_name", "model"]], p], axis=1)


COLS = ["net_charge_pH74", "net_charge_per100", "pI", "gravy", "frac_acidic", "frac_basic"]
LABELS = {"net_charge_pH74":"net charge @pH7.4", "net_charge_per100":"net charge /100 res",
          "pI":"isoelectric point", "gravy":"GRAVY hydropathy",
          "frac_acidic":"% acidic (D+E)", "frac_basic":"% basic (K+R)"}


def main(args):
    tag      = args.out_tag if args.out_tag else "default"
    sum_txt  = os.path.join(OUT, f"charge_shift_{tag}_summary.txt")
    sum_json = os.path.join(OUT, f"charge_shift_{tag}_summary.json")
    fig_png  = os.path.join(OUT, f"charge_shift_{tag}_figure.png")
    for pth in (sum_txt, sum_json, fig_png):
        assert not os.path.exists(pth), f"Refusing to overwrite {pth}; use a distinct --out_tag."

    ev = enrich(pd.read_csv(args.eval_csv))
    sc = None
    if args.selfconsistency_csv and os.path.exists(args.selfconsistency_csv):
        sc = enrich(pd.read_csv(args.selfconsistency_csv))

    lines, J = [], {}
    def emit(s=""): print(s); lines.append(s)
    emit("=" * 76); emit(f"CHARGE / BIOPHYSICS SHIFT FROM DE-IMMUNISATION  [tag: {tag}]"); emit("=" * 76)
    emit(f"eval_csv: {args.eval_csv}")

    # ── Primary: base vs finetuned on eval_deimmunisation (paired per protein) ──
    emit(f"\nPRIMARY  (eval_deimmunisation.csv: {ev.protein_name.nunique()} proteins, "
         f"{len(ev)} sequences)")
    emit(f"{'metric':<22}{'base':>12}{'fine-tuned':>14}{'Δ':>10}{'paired p':>12}")
    emit("-" * 70)
    pm = ev.groupby(["protein_name", "model"])[COLS].mean()
    b = pm.xs("base", level="model"); f = pm.xs("finetuned", level="model")
    J["primary"] = {}
    for c in COLS:
        bv, fv = b[c], f[c]
        t, p = stats.ttest_rel(bv, fv)
        emit(f"{LABELS[c]:<22}{bv.mean():>12.2f}{fv.mean():>14.2f}{fv.mean()-bv.mean():>+10.2f}{p:>12.1g}")
        J["primary"][c] = dict(base=float(bv.mean()), ft=float(fv.mean()),
                               delta=float(fv.mean()-bv.mean()), p=float(p))
    # how many designs flip sign of net charge?
    ev_b = ev[ev.model=="base"]; ev_f = ev[ev.model=="finetuned"]
    emit("")
    emit(f"Net charge @pH7.4: base mean {ev_b.net_charge_pH74.mean():+.1f}, "
         f"fine-tuned {ev_f.net_charge_pH74.mean():+.1f}")
    emit(f"Sequences net-negative: base {100*(ev_b.net_charge_pH74<0).mean():.0f}%  ->  "
         f"fine-tuned {100*(ev_f.net_charge_pH74<0).mean():.0f}%")
    emit(f"pI < 5 (strongly acidic): base {100*(ev_b.pI<5).mean():.0f}%  ->  "
         f"fine-tuned {100*(ev_f.pI<5).mean():.0f}%")

    # ── Context: native -> base -> finetuned (selfconsistency), optional ──
    if sc is not None:
        emit(f"\nTRAJECTORY  (selfconsistency: native vs base vs fine-tuned, "
             f"{sc.protein_name.nunique()} proteins)")
        emit(f"{'metric':<22}{'native':>10}{'base':>10}{'fine-tuned':>13}")
        emit("-" * 56)
        J["trajectory"] = {}
        for c in COLS:
            vals = {m: sc[sc.model==m][c].mean() for m in ["native","base","finetuned"]}
            emit(f"{LABELS[c]:<22}{vals['native']:>10.2f}{vals['base']:>10.2f}{vals['finetuned']:>13.2f}")
            J["trajectory"][c] = {k: float(v) for k, v in vals.items()}
        emit("")
        emit("Reading the trajectory: compare native->base (ProteinMPNN's own bias) with")
        emit("base->fine-tuned (the de-immunisation's added shift).")
    else:
        emit("\nTRAJECTORY  skipped (no --selfconsistency_csv supplied for this run).")

    # ── data-driven summary (numbers only, no interpretation) ──
    dacid = J["primary"]["frac_acidic"]["delta"]
    dpI   = J["primary"]["pI"]["delta"]
    nc_b  = J["primary"]["net_charge_pH74"]["base"]; nc_f = J["primary"]["net_charge_pH74"]["ft"]
    ev_b  = ev[ev.model=="base"]; ev_f = ev[ev.model=="finetuned"]
    pI5_b = 100*(ev_b.pI<5).mean(); pI5_f = 100*(ev_f.pI<5).mean()
    J["primary"]["pI_lt5_pct"] = dict(base=float(pI5_b), ft=float(pI5_f))
    emit("\n" + "-" * 76)
    emit(f"CHARGE SHIFT (data-driven, tag {tag})")
    emit(f"  Base -> fine-tuned: acidic {dacid:+.1f} pp, net charge @pH7.4 {nc_b:+.1f} -> {nc_f:+.1f}, "
         f"pI {dpI:+.1f}.")
    emit(f"  Designs below pI 5: base {pI5_b:.0f}% -> fine-tuned {pI5_f:.0f}%.")
    if "trajectory" in J:
        traj = J["trajectory"]["net_charge_per100"]
        emit(f"  Trajectory (net charge/100 res): native {traj['native']:+.1f} -> base "
             f"{traj['base']:+.1f} -> fine-tuned {traj['finetuned']:+.1f}.")
    emit("=" * 76)

    open(sum_txt, "w").write("\n".join(lines) + "\n")
    json.dump(J, open(sum_json, "w"), indent=2)
    print(f"\nWrote {os.path.basename(sum_txt)} / .json")
    make_fig(ev, sc, fig_png)


def make_fig(ev, sc, out_png):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"base":"#4C72B0", "finetuned":"#DD8452", "native":"#555555"}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    panels = [("net_charge_pH74","net charge @ pH 7.4"),
              ("pI","isoelectric point (pI)"),
              ("frac_acidic","% acidic residues (D+E)")]
    for a, (col, lab) in zip(ax, panels):
        for m in ["base","finetuned"]:
            v = ev[ev.model==m][col]
            a.hist(v, bins=30, alpha=0.55, color=C[m], label=f"{m} (μ={v.mean():.1f})", density=True)
        a.axvline(ev[ev.model=="base"][col].mean(), color=C["base"], ls="--", lw=1)
        a.axvline(ev[ev.model=="finetuned"][col].mean(), color=C["finetuned"], ls="--", lw=1)
        a.set_xlabel(lab); a.set_ylabel("density"); a.legend(fontsize=8)
        a.spines[["top","right"]].set_visible(False)
    ax[0].set_title("De-immunisation shifts designs toward net-negative charge", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(out_png, dpi=300)
    print(f"Wrote {os.path.basename(out_png)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Quantify charge/pI shift from de-immunisation")
    ap.add_argument("--eval_csv", default=os.path.join(OUT, "eval_deimmunisation.csv"),
                    help="CSV with columns protein_name, model (base|finetuned), sequence")
    ap.add_argument("--selfconsistency_csv", default=None,
                    help="Optional CSV with native/base/finetuned rows for the trajectory panel")
    ap.add_argument("--out_tag", default="default",
                    help="Suffix for output files (charge_shift_<tag>_*)")
    main(ap.parse_args())
