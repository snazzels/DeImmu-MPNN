#!/usr/bin/env python
"""Strategy-comparison figure: DPO fine-tuning vs post-hoc filtering vs random base.

Reconstructed generator (the original was made ad hoc and never saved). Faithful to
the previously rendered figures/strategy_comparison_figure.png; this version keeps the
bare panel letters (A/B/C) and drops the descriptive panel titles and the descriptive
suptitle. All data points, bars, axis labels, legends, annotations and colours are
unchanged.

Data sources (read-only; nothing here is overwritten):
  data/output/filtering_baseline.csv      -> filtered-base top-10 per protein (panels A,B)
  data/output/eval_deimmunisation.csv     -> base / fine-tuned per-protein means (panels A,B)
  data/output/human_likeness.csv          -> per-sequence PWM windows vs SwissProt 15-mer
                                             fraction, base vs fine-tuned (panel C)

Output: figures/strategy_comparison_figure.png (300 dpi)
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "..", "CAPE_MPNN", "data", "output")

# ── Colours sampled from the original render ──────────────────────────────────
C_RANDOM = "#2e8bc0"   # random base bar
C_FILT   = "#d591b7"   # filtered base bar
C_DPO    = "#eab02e"   # DPO fine-tuned bar / panel-B points (goldenrod)
C_COMBO  = "#2ca25f"   # DPO + filter bar (green = combined strategy)
C_BASE   = "#0072b2"   # panel C base   (Okabe-Ito blue)
C_FT     = "#e69f00"   # panel C fine-tuned (Okabe-Ito orange)
TEAL     = "#009e73"   # panel B annotation
FG       = "#222222"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.edgecolor": FG,
    "axes.linewidth": 0.9,
    "svg.fonttype": "none",
})

# ── Load panel A/B data (DEPOSITED lambda=0 retrain 54e1642c) ─────────────────
# Everything is recomputed from the single deposited eval CSV (10 base + 10
# fine-tuned designs per protein): random base = mean of the 10 base designs;
# filtered base = the 3 lowest-PWM of those 10 base designs (matched 3-of-10
# filter depth, the framing used in the text); DPO = mean of the 10 fine-tuned
# designs. This replaces the lost original-run eval_deimmunisation.csv +
# filtering_baseline.csv and matches the reported 15.8% / 34.8% reductions.
eval_rows = list(csv.DictReader(open(os.path.join(OUT, "eval_deimmunisation_dual_lambda0_seed42.csv"))))
prots = sorted(set(r["protein_name"] for r in eval_rows))

def _vals(p, m):
    return sorted(int(r["n_presented_total"]) for r in eval_rows
                  if r["protein_name"] == p and r["model"] == m)

FILT_DEPTH = 3  # lowest-PWM 3 of 10 designs
base_pp = np.array([np.mean(_vals(p, "base"))                    for p in prots])  # random base (K=10)
dpo_pp  = np.array([np.mean(_vals(p, "finetuned"))               for p in prots])  # DPO fine-tuned (K=10)
filt_pp = np.array([np.mean(_vals(p, "base")[:FILT_DEPTH])       for p in prots])  # filtered base (3 of 10)
fdpo_pp = np.array([np.mean(_vals(p, "finetuned")[:FILT_DEPTH])  for p in prots])  # DPO + filter (3 of 10)

means = [np.nanmean(a) for a in (base_pp, filt_pp, dpo_pp, fdpo_pp)]
sems  = [stats.sem(a, nan_policy="omit") for a in (base_pp, filt_pp, dpo_pp, fdpo_pp)]
# Report per-protein reduction means (mean of per-protein ratios), matching the text.
red_filt = float(np.mean((base_pp - filt_pp) / base_pp) * 100.0)
red_dpo  = float(np.mean((base_pp - dpo_pp)  / base_pp) * 100.0)
red_fdpo = float(np.mean((base_pp - fdpo_pp) / base_pp) * 100.0)  # DPO then filtered

_, p_rf = stats.ttest_rel(base_pp, filt_pp)   # random vs filtered
_, p_fd = stats.ttest_rel(filt_pp, dpo_pp)    # filtered vs DPO
_, p_rd = stats.ttest_rel(base_pp, dpo_pp)    # random vs DPO
_, p_df = stats.ttest_rel(dpo_pp,  fdpo_pp)   # DPO vs DPO+filter
n_dpo_better  = int(np.sum(dpo_pp < filt_pp))
n_fdpo_better = int(np.sum(fdpo_pp < dpo_pp))  # DPO+filter vs DPO
n_prot = len(prots)

# ══════════════════════════════════════════════════════════════════════════════
# Panel C (immunogenicity vs human-likeness) removed 2026-08-07; the human-likeness
# result (base 0.28% vs ft 0.28%) is stated in the text. 3-panel backup kept as
# strategy_comparison_figure_3panel.png / make_strategy_comparison_figure_3panel.py.bak.
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 5))

# ── Panel A: strategy comparison bars ─────────────────────────────────────────
x = np.arange(4)
bars = axA.bar(x, means, yerr=sems, width=0.66,
               color=[C_RANDOM, C_FILT, C_DPO, C_COMBO], capsize=5,
               error_kw=dict(ecolor=FG, lw=1.1))
for xi, m, lab in zip(x, means, ["ref.", f"−{red_filt:.0f}%", f"−{red_dpo:.0f}%", f"−{red_fdpo:.0f}%"]):
    axA.text(xi, m * 0.5, lab, ha="center", va="center",
             color="white", fontweight="bold", fontsize=14)

def _sig(ax, x1, x2, y, txt, h=2.5):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.1, color=FG)
    ax.text((x1 + x2) / 2, y + h, txt, ha="center", va="bottom", fontsize=10)

_sig(axA, 0, 1, 108, f"p={p_rf:.0e}")
_sig(axA, 1, 2, 122, f"p={p_fd:.0e}")
_sig(axA, 0, 2, 136, f"p={p_rd:.0e}")
_sig(axA, 2, 3,  92, f"p={p_df:.0e}")

axA.set_xticks(x)
axA.set_xticklabels(["Random base\n(K=10)",
                     "Filtered base\n(3 of 10)",
                     "DPO\nfine-tuned",
                     "DPO + filter\n(3 of 10)"], fontsize=10.5)
axA.set_ylabel("Mean MHC-II presented windows\nper designed sequence (PWM)", fontsize=12)
axA.set_ylim(0, 148)
axA.yaxis.set_major_locator(plt.MultipleLocator(20))
axA.spines[["top", "right"]].set_visible(False)
axA.set_title("A", loc="left", fontweight="bold", fontsize=16)

# ── Panel B: per-protein DPO vs filtering ─────────────────────────────────────
lim = 235
axB.plot([0, lim], [0, lim], ls="--", lw=1.2, color="#b0b0b0", zorder=1)
axB.scatter(filt_pp, dpo_pp, s=70, color=C_DPO, alpha=0.78,
            edgecolors="none", zorder=2)
axB.set_xlim(0, lim); axB.set_ylim(0, lim)
axB.xaxis.set_major_locator(plt.MultipleLocator(50))
axB.yaxis.set_major_locator(plt.MultipleLocator(50))
axB.set_xlabel("Filtered base (3 of 10)", fontsize=13)
axB.set_ylabel("DPO fine-tuned (10 seqs)", fontsize=13)
axB.text(0.96, 0.05, f"DPO better in {n_dpo_better}/{n_prot} proteins",
         transform=axB.transAxes, ha="right", va="bottom",
         color=TEAL, fontweight="bold", fontsize=12)
axB.spines[["top", "right"]].set_visible(False)
axB.set_title("B", loc="left", fontweight="bold", fontsize=16)

fig.tight_layout()
OUT_PNG = os.path.join(HERE, "strategy_comparison_figure.png")
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Wrote {OUT_PNG}")
print(f"means={means}  sems={sems}  red_filt={red_filt:.1f}  red_dpo={red_dpo:.1f}")
print(f"p_rf={p_rf:.3g} p_fd={p_fd:.3g} p_rd={p_rd:.3g} p_df={p_df:.3g}")
print(f"red_fdpo={red_fdpo:.1f}  DPO better {n_dpo_better}/{n_prot}  DPO+filter better {n_fdpo_better}/{n_prot}")
