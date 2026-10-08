#!/usr/bin/env python
"""Plot the human-proteome reference point against the two design arms.

Reads the frozen outputs of human_proteome_reference.py and the published design
eval CSV; computes nothing new except the between-population tests, so the figure
cannot drift from the numbers in the summary file.

The density axis (presented windows per 100 residues) is the one plotted, because
raw counts scale with length and the populations are only length-matched in one
stratum. Refuses to overwrite.
"""
import os, sys, csv, argparse, collections
import statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OD = os.path.join(PF, "data", "output")
DESIGN_CSV = os.path.join(OD, "validate_netmhciipan_sweep_f16b51e6.csv")
EXCLUDE = {"4b6d_C", "7m10_A"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--out_tag", default="v1")
    args = ap.parse_args()

    src = os.path.join(OD, f"human_proteome_reference_{args.tag}.csv")
    out = os.path.join(OD, f"human_proteome_reference_{args.out_tag}_figure.png")
    if os.path.exists(out):
        sys.exit(f"REFUSING TO OVERWRITE existing output: {out}")

    H = list(csv.DictReader(open(src)))
    lm = np.array([float(r["per_100_residues"]) for r in H
                   if "length_matched" in r["stratum"]])

    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    L = {}
    for r in csv.DictReader(open(DESIGN_CSV)):
        if r["protein_name"] in EXCLUDE:
            continue
        acc[r["protein_name"]][r["model"]].append(float(r["netmhc_total"]))
        L[r["protein_name"]] = int(r["protein_length"])
    arm = {a: np.array([100 * st.mean(m[a]) / L[n] for n, m in acc.items()])
           for a in ("base", "finetuned")}

    # Label spelling follows the manuscript's American house style ("de-immunized"),
    # not the -ised spelling used in this repository's filenames.
    groups = [("Base\nProteinMPNN", arm["base"], "#C1666B"),
              ("De-immunized\n(this work)", arm["finetuned"], "#48A9A6"),
              ("Human proteome\n(length-matched)", lm, "#8A8FA3")]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    rng = np.random.default_rng(20260928)
    for i, (lab, v, c) in enumerate(groups):
        parts = ax.violinplot([v], positions=[i], widths=0.72,
                              showextrema=False, showmedians=False)
        for b in parts["bodies"]:
            b.set_facecolor(c); b.set_alpha(0.35); b.set_edgecolor(c)
            b.set_linewidth(1.2)
        x = i + rng.normal(0, 0.055, size=len(v))
        ax.scatter(x, v, s=7, color=c, alpha=0.55, linewidths=0, zorder=3)
        m = np.median(v)
        ax.hlines(m, i - 0.30, i + 0.30, color=c, lw=2.6, zorder=4)
        ax.text(i + 0.36, m, f"{m:.1f}", va="center", ha="left",
                fontsize=9, color=c, fontweight="semibold")

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_ylabel("Predicted presented windows per 100 residues", fontsize=9.5)
    ax.set_ylim(0, max(arm["base"].max(), lm.max()) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)

    def star(a, b):
        p = stats.mannwhitneyu(a, b, alternative="two-sided")[1]
        return (f"p = {p:.3g}" if p >= 1e-3 else f"p = {p:.1e}"), p

    y = ax.get_ylim()[1]
    for (i, j, yy) in ((0, 2, y * 0.965), (1, 2, y * 0.865)):
        txt, _ = star(groups[i][1], groups[j][1])
        ax.plot([i, i, j, j], [yy - y * 0.018, yy, yy, yy - y * 0.018],
                lw=0.9, color="0.35")
        ax.text((i + j) / 2, yy + y * 0.008, txt, ha="center", va="bottom",
                fontsize=8.2, color="0.25")

    n = [len(g[1]) for g in groups]
    ax.set_title("Predicted class II burden of designs against ordinary human proteins\n"
                 f"netMHCIIpan, six DRB1 alleles, %Rank$_{{EL}}$ ≤ 2  "
                 f"(n = {n[0]}, {n[1]}, {n[2]})",
                 fontsize=9.5, pad=10)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print("wrote", out)
    for i, j in ((0, 2), (1, 2)):
        txt, p = star(groups[i][1], groups[j][1])
        u = stats.mannwhitneyu(groups[i][1], groups[j][1])[0]
        r = 2 * u / (len(groups[i][1]) * len(groups[j][1])) - 1
        print(f"  {groups[i][0]!r} vs {groups[j][0]!r}: {txt}, rank-biserial {r:+.3f}")


if __name__ == "__main__":
    main()
