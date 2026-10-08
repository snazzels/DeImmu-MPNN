#!/usr/bin/env python
"""MHC-II-only ablation figure: the double dissociation.

Grouped bars of per-protein percent reduction in predicted presented windows for
the two fine-tuned models, by MHC class. The MHC-II-only model (7777d75f) reduces
MHC class II strongly while barely touching MHC class I; the dual-objective model
(057975d5) reduces both. This attributes the MHC-II reduction to the MHC-II term.

Numbers are the matched-seed evaluation (both best.pt, --seed 42, same 100 held-out
proteins), from:
  data/output/eval_deimmunisation_{mhc2only,dual}_seed42_summary.txt   (MHC-II)
  data/output/eval_mhc1_arm_{mhc2only,dual}_seed42_summary.txt          (MHC-I)

Output: figures/mhc2only_ablation_figure.png  (300 dpi)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── House palette: gray base/neutral, blue fine-tuned/MHC-II, orange MHC-I.
# Here colour encodes the MODEL: the MHC-II-only model in house blue, the
# dual-objective model in neutral gray.
BLUE = "#2c7fb8"   # MHC-II-only model
GRAY = "#7f7f7f"   # dual-objective model
FG   = "#222222"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": FG,
    "axes.linewidth": 0.8,
    "svg.fonttype": "none",
})

# ── Matched-seed results (mean ± sd of per-protein % reduction) ────────────────
#              MHC class II            MHC class I
#            mean    sd            mean    sd
DATA = {
    "MHC-II-only\n(7777d75f)": {"II": (42.6, 9.9), "I": (6.3, 10.4), "c": BLUE},
    "Dual-objective\n(057975d5)": {"II": (35.5, 12.2), "I": (27.6, 8.4), "c": GRAY},
}
CLASSES = ["MHC class II", "MHC class I"]
CLASS_KEY = {"MHC class II": "II", "MHC class I": "I"}

fig, ax = plt.subplots(figsize=(5.6, 4.2))

x = np.arange(len(CLASSES))         # group positions (by MHC class)
width = 0.38
models = list(DATA.keys())

for j, m in enumerate(models):
    means = [DATA[m][CLASS_KEY[c]][0] for c in CLASSES]
    sds   = [DATA[m][CLASS_KEY[c]][1] for c in CLASSES]
    offset = (j - 0.5) * width
    bars = ax.bar(x + offset, means, width, yerr=sds, capsize=3.5,
                  color=DATA[m]["c"], edgecolor=FG, linewidth=0.8,
                  error_kw=dict(ecolor=FG, lw=0.9), label=m.replace("\n", " "))
    for b, mn in zip(bars, means):
        ax.annotate(f"{mn:.1f}%", (b.get_x() + b.get_width() / 2, mn),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold", color=FG)

ax.set_xticks(x)
ax.set_xticklabels(CLASSES, fontsize=11)
ax.set_ylabel("Per-protein reduction in\npredicted presented windows (%)", fontsize=11)
ax.set_ylim(0, 60)
ax.axhline(0, color=FG, lw=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.set_major_locator(plt.MultipleLocator(10))
ax.tick_params(length=3)

# Legend keyed by model colour
legend_handles = [Patch(facecolor=DATA[m]["c"], edgecolor=FG, label=m.replace("\n", " "))
                  for m in models]
ax.legend(handles=legend_handles, frameon=False, fontsize=9.5, loc="upper right",
          title="Fine-tuned model", title_fontsize=9.5)

fig.tight_layout()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mhc2only_ablation_figure.png")
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Wrote {OUT}")
