#!/usr/bin/env python
"""Fig 3 (attribution + strategy), Panel A RE-ANCHORED to the deployed operating point.

Supersedes make_fig_attribution_strategy_v7_f16b51e6.py, which is NOT edited or deleted
(it is the record of the published panel). That version deliberately held Panel A at
lr=3e-7 -- class-II-only 7777d75f vs dual lambda=0 54e1642c vs Gasser -- so the three bars
were matched on learning rate. Two things made that the wrong trade:

  1. The paper's subject is the DEPLOYED model, and the Discussion's double-dissociation
     argument now rests on deployed (f16b51e6) vs class-II-only (abec4d2c). Panel A showing
     a different pair told a different dissociation story than the text.
  2. The lr=3e-7 matching was never the binding constraint anyway: the Gasser arm sits on a
     different PROTEIN SAMPLE (100 proteins, 44 overlapping the pinned 98), so the panel
     already contained an unmatched arm while the caption claimed the comparison was matched.

Panel A now plots, for the two in-house arms:
    abec4d2c   class-II-only at the deployed operating point (lr 1e-6)
    f16b51e6   DEPLOYED, dual objective (lr 1e-6)
These differ in the class-I predictor ALONE -- the cleanest available single-objective
dissociation -- and are computed here on the clean 94 with a base arm that is verified
identical design-by-design within each metric.

The Gasser class-I-only arm cannot be re-anchored (external published checkpoint, lr 3e-7,
its own 100-protein sample). It is kept, HATCHED to mark it as the unmatched arm, and the
caption states its status plainly. Re-running it on the pinned 98 is the clean fix and remains
an open item.

Panel B is unchanged from v7: the deployed model's strategy comparison.

Read-only with respect to data. Writes a NEW versioned PNG plus the generic staged name.
"""
import os, re, sys, csv
from collections import defaultdict
from statistics import mean, stdev

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

OUT = os.environ["CAPE_ROOT"] + "/CAPE_MPNN/data/output"
FIG = os.environ["CAPE_ROOT"] + "/figures"
BLUE = "#2c7fb8"; ORANGE = "#e6550d"; GRAY = "#7f7f7f"
C_RANDOM = "#2e8bc0"; C_FILT = "#d591b7"; C_DPO = "#eab02e"; C_COMBO = "#2ca25f"; FG = "#222222"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 15, "axes.edgecolor": FG,
                     "axes.linewidth": 0.9, "svg.fonttype": "none"})

# the four assembly-mismatched chains: base and fine-tuned are scored on different
# molecules, so their per-protein percentage reduction is meaningless
EXCLUDE = {"2pt5_B", "3hmt_A", "4i4y_C", "4nc7_B"}

# Minimum-base guard (2026-08-25). A per-protein PERCENTAGE reduction is undefined at
# base = 0 and noise-dominated just above it. The unguarded convention let 7m10_A
# (mean base 0.1 windows) contribute -300% to the headline. See
# CAPE_MPNN/tools/validate_with_netmhciipan.py for the full rationale.
MIN_BASE = 1.0


def per_protein(fn, col):
    """-> {protein: (base_mean, ft_mean)} averaging the k sequences within each arm."""
    acc = defaultdict(lambda: defaultdict(list))
    with open(os.path.join(OUT, fn)) as fh:
        for r in csv.DictReader(fh):
            acc[r["protein_name"]][r["model"]].append(float(r[col]))
    return {p: (mean(d["base"]), mean(d["finetuned"])) for p, d in acc.items()}


def reduction(fn, col):
    """Per-protein mean reduction and sd on the clean 94, plus the base vector."""
    d = per_protein(fn, col)
    names = sorted(p for p in set(d) - EXCLUDE if d[p][0] > MIN_BASE)
    b = [d[n][0] for n in names]
    f = [d[n][1] for n in names]
    per = [100.0 * (x - y) / x for x, y in zip(b, f)]
    return mean(per), stdev(per), len(per), b, names


# ---------------------------------------------------------------- Panel A, matched arms
# class II and class I come from different validators; within EACH metric the two arms
# must share a base arm, and that is asserted rather than assumed. (Across metrics they
# need not match -- the panel never compares class II to class I within a model.)
II_ABEC = "validate_netmhciipan_netmhcmatch_pwmtraj_abec4d2c_ep200.csv"
II_DEP  = "validate_netmhciipan_ckptrob_traj_ep200.csv"
I_ABEC  = "validate_netmhcpan_mhc1_netmhcmatch_pwmtraj_abec4d2c_ep200.csv"
I_DEP   = "validate_netmhcpan_mhc1_netmhcmatch_pwmtraj_f16b51e6_ep200.csv"

ii_abec = reduction(II_ABEC, "netmhc_total")
ii_dep  = reduction(II_DEP,  "netmhc_total")
i_abec  = reduction(I_ABEC,  "netmhc_total")
i_dep   = reduction(I_DEP,   "netmhc_total")

for tag, a, b in (("class II", ii_abec, ii_dep), ("class I", i_abec, i_dep)):
    assert a[4] == b[4], f"{tag}: protein sets differ between the two arms"
    worst = max(abs(x - y) for x, y in zip(a[3], b[3]))
    if worst > 1e-9:
        raise SystemExit(
            f"{tag}: the two arms do NOT share a base arm (max per-protein |diff| = {worst:.6f}). "
            "Refusing to plot a cross-model comparison on different base designs -- this is the "
            "defect that put a different design draw into the netMHC table."
        )
    print(f"{tag}: base arm identical design-by-design across both arms (n={a[2]})")


def _parse(path, pat, what):
    if not os.path.exists(path): sys.exit(f"missing {what}: {path}")
    m = re.search(pat, open(path).read())
    if not m: sys.exit(f"cannot parse {what} from {path}")
    return float(m.group(1)), float(m.group(2))


# The external, UNMATCHED arm: published Gasser class-I-only checkpoint, lr 3e-7, 100
# proteins. Read from the GUARDED recomputes so this bar uses the same min-base
# convention as the two matched bars (tools/recompute_guarded_reduction.py); the
# original summaries were written under the unguarded convention.
g_ii = _parse(f"{OUT}/validate_netmhciipan_fullset_mhcionly_k10_guarded_summary.txt",
              r"Reduction:\s*mean\s*(-?[\d.]+)%\s*±\s*([\d.]+)%", "II gasser")
g_i  = _parse(f"{OUT}/validate_netmhcpan_mhc1_fullset_mhcionly_k5_guarded_summary.txt",
              r"Reduction:\s*mean\s*(-?[\d.]+)%\s*±\s*([\d.]+)%", "I gasser")

N_MATCHED = ii_abec[2]
N_GASSER = 99   # one protein dropped by the min-base guard on the class-II arm
DATA = {
    "MHC-II-only (abec4d2c)": dict(
        II=(ii_abec[0], ii_abec[1] / np.sqrt(N_MATCHED)),
        I=(i_abec[0],  i_abec[1] / np.sqrt(N_MATCHED)), c=BLUE, hatch=""),
    "Dual-objective (deployed)": dict(
        II=(ii_dep[0], ii_dep[1] / np.sqrt(N_MATCHED)),
        I=(i_dep[0],  i_dep[1] / np.sqrt(N_MATCHED)), c=GRAY, hatch=""),
    "MHC-I-only (precursor, unmatched)": dict(
        II=(g_ii[0], g_ii[1] / np.sqrt(N_GASSER)),
        I=(g_i[0],  g_i[1] / np.sqrt(N_GASSER)), c=ORANGE, hatch="///"),
}

# ---------------------------------------------------------------- Panel B, unchanged
rows = list(csv.DictReader(open(f"{OUT}/validate_netmhciipan_sweep_f16b51e6.csv")))
prots = sorted(set(r["protein_name"] for r in rows))
def _basemean(p):
    v = [int(r["netmhc_total"]) for r in rows if r["protein_name"] == p and r["model"] == "base"]
    return sum(v) / len(v) if v else 0.0
prots = [p for p in prots if _basemean(p) > MIN_BASE]   # same guard as panel A
def _pairs(p, m): return [(int(r["pwm_total"]), int(r["netmhc_total"]))
                          for r in rows if r["protein_name"] == p and r["model"] == m]
FD = 3
base_pp = np.array([np.mean([n for _, n in _pairs(p, "base")]) for p in prots])
dpo_pp  = np.array([np.mean([n for _, n in _pairs(p, "finetuned")]) for p in prots])
filt_pp = np.array([np.mean([n for _, n in sorted(_pairs(p, "base"), key=lambda t: t[0])[:FD]]) for p in prots])
fdpo_pp = np.array([np.mean([n for _, n in sorted(_pairs(p, "finetuned"), key=lambda t: t[0])[:FD]]) for p in prots])
means = [np.nanmean(a) for a in (base_pp, filt_pp, dpo_pp, fdpo_pp)]
sems  = [stats.sem(a, nan_policy="omit") for a in (base_pp, filt_pp, dpo_pp, fdpo_pp)]
red_filt = float(np.mean((base_pp - filt_pp) / base_pp) * 100)
red_dpo  = float(np.mean((base_pp - dpo_pp) / base_pp) * 100)
red_fdpo = float(np.mean((base_pp - fdpo_pp) / base_pp) * 100)
_, p_rf = stats.ttest_rel(base_pp, filt_pp); _, p_fd = stats.ttest_rel(filt_pp, dpo_pp)
_, p_rd = stats.ttest_rel(base_pp, dpo_pp);  _, p_df = stats.ttest_rel(dpo_pp, fdpo_pp)

# ---------------------------------------------------------------- draw
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw=dict(width_ratios=[1.0, 1.2]))
CLASSES = ["MHC class II", "MHC class I"]; CKEY = {"MHC class II": "II", "MHC class I": "I"}
xA = np.arange(2); width = 0.26
for j, lab in enumerate(DATA):
    mv = [DATA[lab][CKEY[c]][0] for c in CLASSES]
    sem = [DATA[lab][CKEY[c]][1] for c in CLASSES]
    bars = axA.bar(xA + (j - 1) * width, mv, width, yerr=sem, capsize=3.0,
                   color=DATA[lab]["c"], edgecolor=FG, linewidth=0.8,
                   hatch=DATA[lab]["hatch"], error_kw=dict(ecolor=FG, lw=0.9))
    for b, mn in zip(bars, mv):
        # Vertical stagger on alternate bars: the class-II values for the two matched arms
        # are 42.0 and 41.3, so at equal label heights the text overlaps.
        axA.annotate(f"{mn:.1f}%", (b.get_x() + b.get_width() / 2, mn),
                     xytext=(0, 4 + (j % 2) * 13),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=11.5, fontweight="bold", color=FG)
axA.set_xticks(xA); axA.set_xticklabels(CLASSES, fontsize=15)
axA.set_ylabel("Per-protein reduction in\nnetMHC-predicted presented windows (%)", fontsize=14)
axA.set_ylim(0, 60); axA.axhline(0, color=FG, lw=0.8); axA.spines[["top", "right"]].set_visible(False)
axA.yaxis.set_major_locator(plt.MultipleLocator(10)); axA.tick_params(length=4, labelsize=13.5)
axA.legend(handles=[Patch(facecolor=DATA[l]["c"], edgecolor=FG, hatch=DATA[l]["hatch"], label=l)
                    for l in DATA],
           frameon=False, fontsize=11.5, loc="upper center", bbox_to_anchor=(0.5, -0.13),
           ncol=2, title="Fine-tuned model", title_fontsize=13.5,
           columnspacing=1.2, handlelength=1.3)
axA.set_title("A", loc="left", fontweight="bold", fontsize=21)

xB = np.arange(4)
axB.bar(xB, means, yerr=sems, width=0.66, color=[C_RANDOM, C_FILT, C_DPO, C_COMBO],
        capsize=5, error_kw=dict(ecolor=FG, lw=1.1))
for xi, m, lab in zip(xB, means, ["ref.", f"−{red_filt:.0f}%", f"−{red_dpo:.0f}%", f"−{red_fdpo:.0f}%"]):
    axB.text(xi, m * 0.5, lab, ha="center", va="center", color="white", fontweight="bold", fontsize=16)
base = means[0]; H = base * 0.05
def _sig(x1, x2, y, txt):
    axB.plot([x1, x1, x2, x2], [y, y + H, y + H, y], lw=1.1, color=FG)
    axB.text((x1 + x2) / 2, y + H, txt, ha="center", va="bottom", fontsize=13)
_sig(0, 1, base * 1.10, f"p={p_rf:.0e}"); _sig(1, 2, base * 1.26, f"p={p_fd:.0e}")
_sig(0, 2, base * 1.42, f"p={p_rd:.0e}"); _sig(2, 3, base * 0.95, f"p={p_df:.0e}")
axB.set_xticks(xB)
axB.set_xticklabels(["Random base (K=10)", "Filtered base (3 of 10)",
                     "DPO fine-tuned (deployed)", "DPO (deployed) + filter"],
                    fontsize=12.5, rotation=20, ha="right", rotation_mode="anchor")
axB.set_xlim(-0.65, 3.65)
axB.set_ylabel("Mean MHC-II presented windows\nper designed sequence (netMHCIIpan)", fontsize=14)
axB.set_ylim(0, base * 1.62); axB.yaxis.set_major_locator(plt.MultipleLocator(10))
axB.spines[["top", "right"]].set_visible(False); axB.tick_params(labelsize=13.5)
axB.set_title("B", loc="left", fontweight="bold", fontsize=21)
fig.tight_layout(w_pad=3.0)

op = f"{FIG}/fig_attribution_strategy_pinned_v4.png"
assert not os.path.exists(op), f"refuse overwrite {op}"
fig.savefig(op, dpi=300, bbox_inches="tight", facecolor="white")
print("\nWrote", op)
staged = f"{FIG}/fig_attribution_strategy_pinned.png"
fig.savefig(staged, dpi=300, bbox_inches="tight", facecolor="white")
print("Staged", staged)
print("Panel A:", {l: (round(DATA[l]["II"][0], 1), round(DATA[l]["I"][0], 1)) for l in DATA})
print(f"Panel B means={[round(m,1) for m in means]}  red_filt={red_filt:.1f} "
      f"red_dpo={red_dpo:.1f} red_fdpo={red_fdpo:.1f}")
