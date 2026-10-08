#!/usr/bin/env python
"""Grouped allele-generalization figure: unregularized (lambda=0) vs deployed
charge-regularized (lambda=1.0), side by side per allele group.

Read-only: pulls per-group reductions from the two frozen netMHCIIpan-direct
summary JSONs (never re-scores, never touches any .csv/.json/.pt). Style follows
the single-model replot (blue/orange palette, edge lines, top/right spines off).
Writes a NEW versioned PNG under figures/ -- does not overwrite any prior figure.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(PF, "CAPE_MPNN", "data", "output")
J_L0 = json.load(open(os.path.join(OUTDIR, "allele_generalisation_dual_lambda0_seed42_summary.json")))
J_L1 = json.load(open(os.path.join(OUTDIR, "allele_generalisation_v3_creg1.0_seed42_summary.json")))
J_DEP = json.load(open(os.path.join(OUTDIR, "allele_generalisation_deployed_f16b51e6_summary.json")))

GROUPS = ["trained DRB1", "held-out DRB1", "DRB3/4/5", "HLA-DQ", "HLA-DP"]
red0 = [J_L0["groups"][g]["reduction_pct"] for g in GROUPS]
red1 = [J_L1["groups"][g]["reduction_pct"] for g in GROUPS]
print("lambda=0  :", [f"{r:.1f}" for r in red0])
print("lambda=1.0:", [f"{r:.1f}" for r in red1])
redD=[J_DEP["groups"][g]["reduction_pct"] for g in GROUPS]
print("deployed  :", [f"{r:.1f}" for r in redD])

x = np.arange(len(GROUPS))
w = 0.27
fig, ax = plt.subplots(figsize=(9.5, 4.8))
b0 = ax.bar(x - w, red0, w, color="#4C72B0", edgecolor="k", linewidth=0.4,
            label=r"$\lambda=0$ (unregularized)")
b1 = ax.bar(x, red1, w, color="#DD8452", edgecolor="k", linewidth=0.4,
            label=r"$\lambda=1.0$, lr $3\times10^{-7}$")
bD = ax.bar(x + w, redD, w, color="#D55E00", edgecolor="#222222", linewidth=0.8,
            label=r"deployed ($\lambda=1.0$, lr $1\times10^{-6}$)")

# trained-DRB1 (lambda=0) reference line, as in the single-model figure
ref = red0[0]
ax.axhline(ref, ls="--", c="#4C72B0", lw=1, alpha=0.8,
           label=f"trained reference ({ref:.0f}%)")
ax.axhline(0, c="k", lw=0.6)

def label(bars, reds):
    for rect, r in zip(bars, reds):
        cx = rect.get_x() + rect.get_width()/2
        if r >= 0:
            ax.text(cx, r + 0.8, f"{r:.0f}%", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(cx, r/2, f"{r:.0f}%", ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold")
label(b0, red0)
label(b1, red1)
label(bD, redD)          # was missing: the deployed bars rendered unlabelled

def ticklabel(g):
    """The two DRB1 group names already encode trained/held-out, so appending the
    suffix verbatim produced "trained DRB1\n(trained)". Render that pair in a
    parallel "DRB1 / (state)" form instead and mark the rest held-out."""
    if g.endswith("DRB1"):
        return "DRB1\n(%s)" % ("trained" if g.startswith("trained") else "held-out")
    return g + "\n(held-out)"

ax.set_xticks(x)
ax.set_xticklabels([ticklabel(g) for g in GROUPS], fontsize=8)
ax.set_ylabel("MHC-II presentation reduction\n(base → fine-tuned, netMHCIIpan)")
# Legend above the axes: at loc="upper right" it collided with the tall deployed
# HLA-DQ bar. Outside the data area it cannot overlap any bar at any value.
ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.005),
          ncol=4, frameon=False, borderaxespad=0.0, columnspacing=1.4,
          handlelength=1.6)
ax.set_ylim(top=max(red0 + red1 + redD) * 1.10)   # headroom for the value labels
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

# Versioned output + staged generic copy; project rule: never overwrite a figure.
# The manuscript's \includegraphics points at the generic staged name.
OUT = os.path.join(PF, "figures", "allele_generalization_reanchored_v3_ticks.png")
if os.path.exists(OUT):
    raise SystemExit(f"Refusing to overwrite {OUT}; bump the version.")
fig.savefig(OUT, dpi=300)
print("Wrote", OUT)
STAGED = os.path.join(PF, "figures", "allele_generalization_reanchored.png")
fig.savefig(STAGED, dpi=300)
print("Staged", STAGED)
