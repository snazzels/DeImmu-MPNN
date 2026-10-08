#!/usr/bin/env python
"""Four-model allele-generalization figure on the PINNED 24-protein sample.

Replaces make_fig_allele_reanchored.py, which is NOT deleted or edited: it is
kept as the record of the superseded figure. That generator read three summary
JSONs whose protein samples were DISJOINT (overlaps 0/24, 5/24, 1/24; nothing
common to all three), so every cross-model statement its figure supported was an
unmatched comparison.

This generator reads the four allele_generalisation_pinned24_* summaries, all of
which were produced with an explicit --protein_list and therefore share protein
set fingerprint 9cae3da74828. That match is VERIFIED at run time by shelling out
to CAPE_MPNN/tools/check_protein_match.py -- the script refuses to plot on a
non-zero exit, so the defect cannot silently recur.

Arms, in the left-to-right order the bars appear:
    54e1642c  lambda=0,   lr 3e-7   -- unregularized earlier operating point
    7e908919  lambda=1.0, lr 3e-7   -- regularized earlier operating point
    f16b51e6  lambda=1.0, lr 1e-6   -- DEPLOYED
    abec4d2c  class-II-only, lr 1e-6 -- re-anchored MHC class-II-only ablation

Read-only with respect to data: pulls per-group reductions from frozen summary
JSONs, never re-scores, never touches any .csv/.json/.pt. Writes a NEW versioned
PNG plus the generic staged name the manuscript's \\includegraphics points at
(the user strips _vN suffixes on Overleaf upload).
"""
import os
import json
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(PF, "CAPE_MPNN", "data", "output")
CHECKER = os.path.join(PF, "CAPE_MPNN", "tools", "check_protein_match.py")

# (legend label, summary-JSON basename, face colour)
# Palette is Okabe-Ito: colourblind-distinguishable, and the deployed model keeps
# the vermillion it carries in the other figures. The earlier lambda=1.0 arm uses
# amber rather than the previous light orange because at four bars a light-orange
# / vermillion pair sits adjacent and is hard to separate.
ARMS = [
    (r"$\lambda=0$, lr $3\times10^{-7}$",   "allele_generalisation_pinned24_lambda0",   "#4C72B0"),
    (r"$\lambda=1.0$, lr $3\times10^{-7}$", "allele_generalisation_pinned24_lambda1",   "#E69F00"),
    (r"deployed ($\lambda=1.0$, lr $1\times10^{-6}$)",
                                            "allele_generalisation_pinned24_deployed",  "#D55E00"),
    ("class-II-only (abec4d2c)",            "allele_generalisation_pinned24_mhc2only",  "#009E73"),
]

GROUPS = ["trained DRB1", "held-out DRB1", "DRB3/4/5", "HLA-DQ", "HLA-DP"]
ALPHA = 0.05          # bars at or above this p are hatched as not significant


def verify_match(paths):
    """Refuse to plot unless check_protein_match.py exits 0 on all four inputs."""
    print("verifying protein-set match ...")
    r = subprocess.run([sys.executable, CHECKER] + paths,
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.stderr:
        sys.stderr.write(r.stderr)
    if r.returncode != 0:
        raise SystemExit(
            f"check_protein_match.py exited {r.returncode} -- the four arms are NOT on "
            "the same proteins. Refusing to plot a cross-model comparison. This is the "
            "exact defect this generator exists to prevent."
        )
    print("protein-set match verified (exit 0)\n")


paths = [os.path.join(OUTDIR, base + "_summary.json") for _, base, _ in ARMS]
verify_match(paths)

data = [json.load(open(p)) for p in paths]

# The base arm must be common: every model is scored against base ProteinMPNN on
# the same proteins, so the per-group base burden has to agree across the four.
for g in GROUPS:
    bases = [d["groups"][g]["base_per_allele"] for d in data]
    if max(bases) - min(bases) > 1e-9:
        raise SystemExit(f"base burden differs across arms at {g}: {bases}")
print("base arm common across all four (per-group base burden identical)\n")

reds = [[d["groups"][g]["reduction_pct"] for g in GROUPS] for d in data]
pvals = [[d["groups"][g]["wilcoxon_p"] for g in GROUPS] for d in data]
for (label, _, _), r in zip(ARMS, reds):
    print(f"{label:48s}", " ".join(f"{v:6.1f}" for v in r))

x = np.arange(len(GROUPS))
n = len(ARMS)
w = 0.20
offs = (np.arange(n) - (n - 1) / 2) * w

fig, ax = plt.subplots(figsize=(10.5, 5.0))
bars = []
for (label, _, colour), r, p, off in zip(ARMS, reds, pvals, offs):
    # Not-significant bars are hatched rather than recoloured, so the model's
    # identity stays readable while the reader can see which effects are real.
    hatches = ["///" if pi >= ALPHA else "" for pi in p]
    bb = ax.bar(x + off, r, w, color=colour, edgecolor="#222222", linewidth=0.6,
                label=label)
    for rect, h in zip(bb, hatches):
        if h:
            rect.set_hatch(h)
            rect.set_edgecolor("#555555")
    bars.append(bb)

# Reference: the DEPLOYED model's own trained-panel reduction. With four models a
# single unlabelled dashed line would be ambiguous -- each arm has its own
# trained-panel reference (its leftmost bar) -- so this one is tied explicitly to
# the deployed arm and coloured to match it.
ref = reds[2][0]
ax.axhline(ref, ls="--", c="#D55E00", lw=1.0, alpha=0.85, zorder=0)
ax.axhline(0, c="k", lw=0.6)
# Annotated inline rather than in the legend: as a fifth legend entry it broke the
# four models out of their left-to-right order. The HLA-DP group is the one place
# with clear space above the bars.
ax.annotate(f"deployed trained-panel\nreference ({ref:.0f}%)",
            xy=(len(GROUPS) - 1.32, ref), xytext=(len(GROUPS) - 1.32, ref + 4.5),
            fontsize=7.5, color="#D55E00", ha="left", va="bottom", linespacing=1.25)

for bb, r in zip(bars, reds):
    for rect, v in zip(bb, r):
        cx = rect.get_x() + rect.get_width() / 2
        if v >= 0:
            ax.text(cx, v + 0.8, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
        else:
            ax.text(cx, v - 0.8, f"{v:.0f}", ha="center", va="top", fontsize=7.5)


def ticklabel(g):
    """The two DRB1 group names already encode trained/held-out, so appending a
    held-out suffix verbatim would render "trained DRB1\n(trained)"."""
    if g.endswith("DRB1"):
        return "DRB1\n(%s)" % ("trained" if g.startswith("trained") else "held-out")
    return g + "\n(held-out)"


ax.set_xticks(x)
ax.set_xticklabels([ticklabel(g) for g in GROUPS], fontsize=8.5)
ax.set_ylabel("MHC-II presentation reduction, %\n(base → fine-tuned, netMHCIIpan)")
ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.005),
          ncol=4, frameon=False, borderaxespad=0.0, columnspacing=1.6,
          handlelength=1.6)

flat = [v for r in reds for v in r]
ax.set_ylim(min(0, min(flat) * 1.25), max(flat) * 1.14)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

OUT = os.path.join(PF, "figures", "allele_generalization_pinned24_v2.png")
if os.path.exists(OUT):
    raise SystemExit(f"Refusing to overwrite {OUT}; bump the version.")
fig.savefig(OUT, dpi=300)
print("\nWrote", OUT)

STAGED = os.path.join(PF, "figures", "allele_generalization_pinned.png")
fig.savefig(STAGED, dpi=300)
print("Staged", STAGED)
