#!/usr/bin/env python
"""Publication-quality SI panel for the beta x learning-rate sweep.
Replaces the diagnostic sweep_pareto.png emitted by slurm/03_pareto.py (which had a
transposed-annotation bug and no front/colorbar). Encodes beta by colour, learning rate
by marker shape, draws the Pareto front, and highlights the deployed model."""
import os, csv
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OD = os.environ["CAPE_ROOT"] + "/CAPE_MPNN/data/output"
FIG = os.environ["CAPE_ROOT"] + "/figures"
DEPLOYED = "f16b51e6"

rows = []
for r in csv.DictReader(open(os.path.join(OD, "sweep_pareto_table.csv"))):
    if not r["netmhc_reduction_pct"] or not r["seq_recovery_ft_pct"]:
        continue
    rows.append(dict(mid=r["model_id"], beta=float(r["beta"]), lr=float(r["lr"]),
                     net=float(r["netmhc_reduction_pct"]), rec=float(r["seq_recovery_ft_pct"])))

# Pareto front: no other config has both higher reduction and higher recovery
def dominated(p, others):
    return any(o["net"] >= p["net"] and o["rec"] >= p["rec"] and
               (o["net"] > p["net"] or o["rec"] > p["rec"]) for o in others)
front = sorted([p for p in rows if not dominated(p, rows)], key=lambda p: p["rec"])

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.linewidth": 0.8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
LR_MARKER = {1e-7: "o", 3e-7: "s", 1e-6: "^"}
LR_LABEL = {1e-7: r"$1\times10^{-7}$", 3e-7: r"$3\times10^{-7}$", 1e-6: r"$1\times10^{-6}$"}
norm = mpl.colors.LogNorm(vmin=0.01, vmax=0.2)
cmap = mpl.cm.viridis

fig, ax = plt.subplots(figsize=(4.6, 3.6))

# Pareto front line first, so markers sit on top
ax.plot([p["rec"] for p in front], [p["net"] for p in front], "-", lw=1.0,
        color="0.55", zorder=1, label="Pareto front")

for p in rows:
    ax.scatter(p["rec"], p["net"], marker=LR_MARKER[p["lr"]], s=52,
               c=[cmap(norm(p["beta"]))], edgecolor="k", linewidth=0.5, zorder=3)

# highlight the deployed configuration
dep = next(p for p in rows if p["mid"] == DEPLOYED)
ax.scatter(dep["rec"], dep["net"], marker="o", s=210, facecolor="none",
           edgecolor="#D55E00", linewidth=1.6, zorder=4)
ax.annotate("deployed\n($\\beta$0.05, lr $1\\times10^{-6}$)", (dep["rec"], dep["net"]),
            textcoords="offset points", xytext=(-14, 26), ha="center", fontsize=6.8,
            color="#8a3b00",
            arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#D55E00",
                            connectionstyle="arc3,rad=0.25"))

ax.set_xlabel("sequence recovery (%)")
ax.set_ylabel("netMHCIIpan MHC class II reduction (%)")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(33.5, 51)
ax.set_ylim(0, 102)

sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
cb.set_label(r"DPO $\beta$ (KL strength)", fontsize=8)
cb.set_ticks([0.01, 0.02, 0.05, 0.1, 0.2])
cb.ax.set_yticklabels(["0.01", "0.02", "0.05", "0.10", "0.20"], fontsize=7)
cb.outline.set_linewidth(0.6)

handles = [Line2D([], [], marker=LR_MARKER[lr], color="0.35", linestyle="none",
                  markersize=5.5, markeredgecolor="k", markeredgewidth=0.5,
                  label=f"lr {LR_LABEL[lr]}") for lr in (1e-7, 3e-7, 1e-6)]
handles.append(Line2D([], [], color="0.55", lw=1.0, label="Pareto front"))
ax.legend(handles=handles, frameon=False, loc="upper right", handletextpad=0.5,
          borderpad=0.2, labelspacing=0.35)

fig.tight_layout()
for ext in ("png", "pdf"):
    op = os.path.join(FIG, f"sweep_pareto_si.{ext}")
    fig.savefig(op, dpi=400, bbox_inches="tight")
print("wrote sweep_pareto_si.png / .pdf")
print("front (recovery, reduction):", [(p["rec"], p["net"]) for p in front])
print(f"n configs plotted: {len(rows)}; front size: {len(front)}")
