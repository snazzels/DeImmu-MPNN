#!/usr/bin/env python
"""Regenerate figures/charge_regularizer_knee.png (and British-spelled sibling).

The original figure was made ad hoc and no generator was saved. This script
reconstructs it faithfully from the authoritative cached values for
charge-model v1/v2/v3 (i.e. charge-regulariser weight lambda =
0 / 0.5 / 1.0). Panel titles are the bare letters A/B/C (the descriptive
titles used previously have been removed by request); everything else
(axis labels, legend, lines/markers, annotations, dual y-axis in panel B,
colours) matches the previous figure.

Data are hard-coded literals; this script reads/writes no result files.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- data
lam = [0.0, 0.5, 1.0]                       # charge-regulariser weight lambda

# Panel A -- MHC-II presentation reduction (%), base -> fine-tuned
epitope_reduction = [39.0, 35.7, 32.7]

# Panel B -- charge liability (dual axis)
net_charge_shift = [-9.0, -5.0, -2.86]      # left axis, units
below_pi5 = [67.0, 57.0, 42.0]              # right axis, %
base_below_pi5 = 32.0                       # base-model reference (~32%)

# Panel C -- self-consistency TM-score
tm_base = [0.895, 0.882, 0.857]             # base ProteinMPNN
tm_ft = [0.895, 0.895, 0.869]               # fine-tuned

# ---------------------------------------------------------------- colours
BLUE = "#1f77b4"
GRAY = "#999999"
RED = "#c0392b"
ORANGE = "#e2871a"

plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.0})

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 5))

# ============================================================ Panel A
axA.plot(lam, epitope_reduction, "-o", color=BLUE, lw=2.2, ms=9)
for x, y in zip(lam, epitope_reduction):
    axA.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=11)
axA.set_ylim(0, 45)
axA.set_xticks(lam)
axA.set_xlabel(r"Charge-regulariser weight $\lambda$")
axA.set_ylabel("MHC-II presentation reduction (%)\n(base $\\rightarrow$ fine-tuned)")
axA.set_title("A", loc="left", fontweight="bold")
axA.spines["top"].set_visible(False)
axA.spines["right"].set_visible(False)

# ============================================================ Panel B (dual axis)
# left axis: net-charge shift
lB1, = axB.plot(lam, net_charge_shift, "-o", color=RED, lw=2.2, ms=9,
                label="net-charge shift (units)")
axB.set_ylim(-9.3, -2.6)
axB.set_yticks([-9, -8, -7, -6, -5, -4, -3])
axB.set_xticks(lam)
axB.set_xlabel(r"Charge-regulariser weight $\lambda$")
axB.set_ylabel("Net-charge shift (units)", color=RED)
axB.tick_params(axis="y", colors=RED)
axB.spines["top"].set_visible(False)
axB.spines["left"].set_color(RED)
axB.set_title("B", loc="left", fontweight="bold")

# right axis: designs below pI 5 (%)
axB2 = axB.twinx()
lB2, = axB2.plot(lam, below_pi5, "--s", color=ORANGE, lw=2.2, ms=9,
                 label="designs below pI 5 (%)")
axB2.axhline(base_below_pi5, ls=":", color=ORANGE, lw=1.5)
axB2.annotate("base model ( $\\approx$32%)", (0.5, base_below_pi5),
              textcoords="offset points", xytext=(0, 6), ha="center",
              color=ORANGE, fontstyle="italic", fontsize=11)
axB2.set_ylim(31, 68)
axB2.set_yticks([35, 40, 45, 50, 55, 60, 65])
axB2.set_ylabel("Designs below pI 5 (%)", color=ORANGE)
axB2.tick_params(axis="y", colors=ORANGE)
axB2.spines["top"].set_visible(False)
axB2.spines["right"].set_color(ORANGE)
axB2.spines["left"].set_visible(False)

axB.legend(handles=[lB1, lB2], loc="upper right", frameon=False, fontsize=10)

# ============================================================ Panel C
import numpy as np
x = np.arange(len(lam))
w = 0.38
axC.bar(x - w / 2, tm_base, w, color=GRAY, label="base ProteinMPNN")
axC.bar(x + w / 2, tm_ft, w, color=BLUE, label="fine-tuned")
axC.set_ylim(0, 1.0)
axC.set_xticks(x)
axC.set_xticklabels([r"$\lambda$=0", r"$\lambda$=0.5", r"$\lambda$=1.0"])
axC.set_ylabel("Self-consistency TM-score")
axC.set_title("C", loc="left", fontweight="bold")
axC.legend(loc="lower right", frameon=False, fontsize=10)
axC.spines["top"].set_visible(False)
axC.spines["right"].set_visible(False)

fig.tight_layout()

OUT = os.environ["CAPE_ROOT"] + "/figures"
for name in ("charge_regularizer_knee.png", "charge_regulariser_knee.png"):
    fig.savefig(f"{OUT}/{name}", dpi=161, bbox_inches="tight")
    print("wrote", name)
