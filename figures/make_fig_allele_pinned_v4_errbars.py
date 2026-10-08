#!/usr/bin/env python
"""TWO-model allele-generalization figure on the PINNED 24-protein sample (main-text Figure 4).

v4, 2026-09-30. Identical to make_fig_allele_pinned_v3_twoarm.py in every plotted
VALUE; the only addition is a bootstrap confidence interval on each bar. The v3
generator is NOT deleted or edited -- it is the record of the figure as first
submitted.

WHY A BOOTSTRAP AND NOT A STANDARD ERROR OF THE MEAN
----------------------------------------------------
The quantity on each bar is a POOLED RATIO:

    reduction_pct = (sum_over_proteins base_windows - sum ft_windows)
                    / sum_over_proteins base_windows * 100

i.e. every window in the sample is summed first and divided once. It is NOT the
mean of the 24 per-protein reductions, and the two differ materially -- 42.6 %
against 41.1 % at trained DRB1 for the deployed model. Attaching a per-protein
s.e.m. to a pooled bar would therefore centre the interval on a number that is
not the bar, mixing two estimators in one graphic.

A bootstrap RESAMPLED OVER PROTEINS puts an interval on exactly the estimator
that is plotted, so no published value changes. The protein is the resampling
unit because the 4 designs drawn per protein are not independent of each other;
resampling designs would understate the uncertainty.

WHY ONE STANDARD ERROR AND NOT A 95 % INTERVAL (revised 2026-09-30)
--------------------------------------------------------------------
The first version of this script drew a 95 % percentile CI. That was internally
correct but INCONSISTENT WITH FIGURE 3, whose whisker is +/- 1 s.e.m. (~68 %
coverage). Side by side, Figure 4's whiskers then looked about twice as long as
Figure 3's for noise that is in fact comparable -- the deployed trained-DRB1
bootstrap SE is 2.71, against 2.47 s.e.m. for the deployed class-II bar of
Figure 3. A reader flipping between the two figures would have read this one as
far noisier than it is.

The whisker is therefore ONE BOOTSTRAP STANDARD ERROR, drawn symmetrically: the
standard deviation of the bootstrap replicate distribution. That is the same
quantity Figure 3 shows -- a standard error of the plotted estimator -- computed
by bootstrap only because the estimator here is a ratio rather than a mean. The
two figures' whiskers now carry identical meaning and coverage.

Symmetry is not an assumption: the replicate distributions are near-symmetric
(upper-arm / lower-arm ratio 0.92-1.04 across the five groups), so drawing
+/- SE rather than a percentile interval discards nothing.

Supersedes allele_generalization_pinned24_v4_errbars.png, which carried the 95 %
interval and was never uploaded. The manuscript points at the generic STAGED
name, so nothing in the .tex needed to move for this change.

THE HLA-DP CAVEAT, WHICH IS REAL AND IS ADDRESSED IN THE CAPTION
----------------------------------------------------------------
For the deployed model the HLA-DP interval is [1.9, 24.5] and excludes zero,
while the bar is hatched not-significant on a paired Wilcoxon test (p=0.078,
13/24 improved). These do not contradict each other: the pooled ratio is
burden-weighted, so the high-burden proteins dominate it, whereas the Wilcoxon
is rank-based and weights all 24 proteins equally. Both captions state the
error-bar definition and name this divergence explicitly rather than leaving a
reader to find an interval clearing zero next to an "n.s." hatch.

Protein-set matching and the common base arm are verified at run time exactly as
in v3; the script refuses to plot if either check fails.

Read-only with respect to data: reads the frozen summary JSONs for the point
estimates and the frozen per-protein CSVs for the bootstrap. Never re-scores,
never writes any .csv/.json/.pt.
"""
import os
import csv
import glob
import json
import subprocess
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Input resolution (deposition copy, 2026-10-05).
#
# This generator must run from two trees that shelve the same files differently:
#   * the RELEASE repo       -- inputs under data/<section>/, protein-set checker
#                               at analysis/check_protein_match.py
#   * the original WORKING tree -- inputs under CAPE_MPNN/data/output/, checker
#                               at CAPE_MPNN/tools/
# Inputs are therefore located BY BASENAME against candidate roots, release
# first, with $CAPE_ROOT (see environment.md) honoured as a final fallback.
# A miss is fatal and names the file: a figure drawn from a silently absent
# input is precisely the failure this tree versions its outputs to avoid.
# ---------------------------------------------------------------------------
_ROOTS = [os.path.join(PF, "data"),                         # release tree
          os.path.join(PF, "CAPE_MPNN", "data", "output")]  # working tree
if os.environ.get("CAPE_ROOT"):
    _ROOTS.append(os.path.join(os.environ["CAPE_ROOT"],
                               "CAPE_MPNN", "data", "output"))


def datafile(basename):
    """Absolute path to a deposited input, searched release tree first."""
    for root in _ROOTS:
        direct = os.path.join(root, basename)
        if os.path.isfile(direct):
            return direct
        hit = sorted(glob.glob(os.path.join(root, "*", basename)))
        if hit:
            return hit[0]
    raise SystemExit(
        "cannot locate input %r; searched %s. Run from the release repo, or "
        "set CAPE_ROOT (environment.md)." % (basename, _ROOTS))


def _checker():
    cands = [os.path.join(PF, "analysis", "check_protein_match.py"),
             os.path.join(PF, "CAPE_MPNN", "tools", "check_protein_match.py")]
    if os.environ.get("CAPE_ROOT"):
        cands.append(os.path.join(os.environ["CAPE_ROOT"], "CAPE_MPNN",
                                  "tools", "check_protein_match.py"))
    for c in cands:
        if os.path.isfile(c):
            return c
    raise SystemExit("cannot locate check_protein_match.py; searched %s." % cands)


CHECKER = _checker()

# (legend label, summary-JSON basename, face colour)
# Palette is Okabe-Ito: colourblind-distinguishable, and the deployed model keeps
# the vermillion it carries in the other figures.
ARMS = [
    (r"deployed ($\lambda=1.0$, lr $1\times10^{-6}$)",
                                            "allele_generalisation_pinned24_deployed",  "#D55E00"),
    ("class-II-only (abec4d2c)",            "allele_generalisation_pinned24_mhc2only",  "#009E73"),
]

GROUPS = ["trained DRB1", "held-out DRB1", "DRB3/4/5", "HLA-DQ", "HLA-DP"]
ALPHA = 0.05          # bars at or above this p are hatched as not significant
N_BOOT = 10000        # bootstrap replicates
BOOT_SEED = 20260930  # fixed so the published intervals are reproducible


def verify_match(paths):
    """Refuse to plot unless check_protein_match.py exits 0 on every input."""
    print("verifying protein-set match ...")
    r = subprocess.run([sys.executable, CHECKER] + paths,
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.stderr:
        sys.stderr.write(r.stderr)
    if r.returncode != 0:
        raise SystemExit(
            f"check_protein_match.py exited {r.returncode} -- the arms are NOT on "
            "the same proteins. Refusing to plot a cross-model comparison. This is the "
            "exact defect this generator exists to prevent."
        )
    print("protein-set match verified (exit 0)\n")


def per_protein_totals(csv_path):
    """{group: {protein: (base_windows, ft_windows)}}, summed over that protein's designs."""
    acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            slot = 0 if row["model"] == "base" else 1
            for g in GROUPS:
                acc[g][row["protein_name"]][slot] += float(row["grp_" + g])
    return {g: {p: tuple(v) for p, v in d.items()} for g, d in acc.items()}


def bootstrap_se(pairs, n_boot=N_BOOT, seed=BOOT_SEED):
    """Bootstrap standard error of the pooled reduction, resampling PROTEINS.

    `pairs` is one (base_windows, ft_windows) tuple per protein. Returns
    (point, se, skew) where `point` is the pooled estimate on the observed sample
    and must reproduce the frozen summary JSON. `skew` is the upper/lower arm
    ratio of the 95 % percentile interval, reported only so the symmetric +/- SE
    whisker can be shown to discard nothing; it is not plotted.
    """
    arr = np.asarray(pairs, dtype=float)          # (n_proteins, 2)
    base_tot, ft_tot = arr.sum(axis=0)
    point = (base_tot - ft_tot) / base_tot * 100.0

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    sums = arr[idx].sum(axis=1)                   # (n_boot, 2)
    reps = (sums[:, 0] - sums[:, 1]) / sums[:, 0] * 100.0
    se = float(reps.std(ddof=1))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    skew = (hi - point) / (point - lo) if point > lo else float("nan")
    return point, se, skew


paths = [datafile(base + "_summary.json") for _, base, _ in ARMS]
verify_match(paths)

data = [json.load(open(p)) for p in paths]

# The base arm must be common: every model is scored against base ProteinMPNN on
# the same proteins, so the per-group base burden has to agree across the arms.
for g in GROUPS:
    bases = [d["groups"][g]["base_per_allele"] for d in data]
    if max(bases) - min(bases) > 1e-9:
        raise SystemExit(f"base burden differs across arms at {g}: {bases}")
print("base arm common across all arms (per-group base burden identical)\n")

reds = [[d["groups"][g]["reduction_pct"] for g in GROUPS] for d in data]
pvals = [[d["groups"][g]["wilcoxon_p"] for g in GROUPS] for d in data]

# Bootstrap intervals, and a hard check that the CSV aggregation reproduces the
# frozen JSON point estimate. If these ever disagree the CSV and the summary have
# drifted apart and the interval would sit on a different quantity than the bar.
ses = []
for (label, base, _), r in zip(ARMS, reds):
    totals = per_protein_totals(datafile(base + ".csv"))
    se_row = []
    print(f"{label}")
    for g, published in zip(GROUPS, r):
        pairs = list(totals[g].values())
        point, se, skew = bootstrap_se(pairs)
        if abs(point - published) > 1e-6:
            raise SystemExit(
                f"{base} / {g}: bootstrap point estimate {point:.6f} does not "
                f"reproduce summary reduction_pct {published:.6f}. The CSV and the "
                "summary JSON disagree; refusing to draw an interval that does not "
                "belong to the plotted bar."
            )
        # A symmetric whisker is only honest while the replicate distribution is
        # near-symmetric. Warn rather than silently misrepresent if that changes.
        if not (0.8 <= skew <= 1.25):
            print(f"    WARNING: {g} replicate distribution is skewed "
                  f"(upper/lower arm {skew:.2f}); a symmetric +/- SE whisker "
                  f"misrepresents it.")
        se_row.append(se)
        print(f"    {g:16s} {published:6.1f}  +/- {se:5.2f} SE   "
              f"(symmetry {skew:.2f})  n={len(pairs)}")
    ses.append(se_row)
print(f"\npoint estimates reproduce every frozen summary value "
      f"({N_BOOT} replicates, seed {BOOT_SEED})\n")

x = np.arange(len(GROUPS))
n = len(ARMS)
w = 0.34
offs = (np.arange(n) - (n - 1) / 2) * w

fig, ax = plt.subplots(figsize=(10.5, 5.0))
bars = []
for (label, _, colour), r, p, se_row, off in zip(ARMS, reds, pvals, ses, offs):
    # Not-significant bars are hatched rather than recoloured, so the model's
    # identity stays readable while the reader can see which effects are real.
    hatches = ["///" if pi >= ALPHA else "" for pi in p]
    # Symmetric +/- 1 bootstrap SE, matching the s.e.m. whisker of Figure 3.
    bb = ax.bar(x + off, r, w, color=colour, edgecolor="#222222", linewidth=0.6,
                label=label, yerr=se_row, capsize=2.5,
                error_kw=dict(ecolor="#222222", elinewidth=0.9, capthick=0.9, zorder=3))
    for rect, h in zip(bb, hatches):
        if h:
            rect.set_hatch(h)
            rect.set_edgecolor("#555555")
    bars.append(bb)

# Reference: the DEPLOYED model's own trained-panel reduction. Each arm has its own
# trained-panel reference (its leftmost bar), so an unlabelled dashed line would be
# ambiguous; this one is tied explicitly to the deployed arm and coloured to match.
ref = reds[0][0]
ax.axhline(ref, ls="--", c="#D55E00", lw=1.0, alpha=0.85, zorder=0)
ax.axhline(0, c="k", lw=0.6)
# Annotated inline rather than in the legend: as a fifth legend entry it broke the
# models out of their left-to-right order. The HLA-DP group is the one place with
# clear space above the bars.
ax.annotate(f"deployed trained-panel\nreference ({ref:.0f}%)",
            xy=(len(GROUPS) - 1.32, ref), xytext=(len(GROUPS) - 1.32, ref + 4.5),
            fontsize=8.5, color="#D55E00", ha="left", va="bottom", linespacing=1.25)

# Value labels clear the whisker, not the bar, or they collide with the cap.
for bb, r, se_row in zip(bars, reds, ses):
    for rect, v, se in zip(bb, r, se_row):
        cx = rect.get_x() + rect.get_width() / 2
        if v >= 0:
            ax.text(cx, v + se + 0.9, f"{v:.0f}", ha="center", va="bottom", fontsize=9.0)
        else:
            ax.text(cx, v - se - 0.9, f"{v:.0f}", ha="center", va="top", fontsize=9.0)


def ticklabel(g):
    """The two DRB1 group names already encode trained/held-out, so appending a
    held-out suffix verbatim would render "trained DRB1\n(trained)"."""
    if g.endswith("DRB1"):
        return "DRB1\n(%s)" % ("trained" if g.startswith("trained") else "held-out")
    return g + "\n(held-out)"


ax.set_xticks(x)
ax.set_xticklabels([ticklabel(g) for g in GROUPS], fontsize=9.5)
ax.set_ylabel("MHC-II presentation reduction, %\n(base → fine-tuned, netMHCIIpan)")
ax.legend(fontsize=9.5, loc="lower center", bbox_to_anchor=(0.5, 1.005),
          ncol=2, frameon=False, borderaxespad=0.0, columnspacing=1.6,
          handlelength=1.6)

# Limits must accommodate the whiskers and the labels above them, not just the bars.
flat_lo = [v - s for row, srow in zip(reds, ses) for v, s in zip(row, srow)]
flat_hi = [v + s for row, srow in zip(reds, ses) for v, s in zip(row, srow)]
ax.set_ylim(min(0, min(flat_lo) * 1.25), max(flat_hi) * 1.16)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

OUT = os.path.join(PF, "figures", "allele_generalization_pinned24_v5_se.png")
if os.path.exists(OUT):
    raise SystemExit(f"Refusing to overwrite {OUT}; bump the version.")
fig.savefig(OUT, dpi=300)
print("Wrote", OUT)

STAGED = os.path.join(PF, "figures", "allele_generalization_pinned.png")
fig.savefig(STAGED, dpi=300)
print("Staged", STAGED)
