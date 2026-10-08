#!/usr/bin/env python
"""SI figure: the beta x learning-rate sweep, as ONE two-panel figure.

  Panel A  reduction-vs-recovery Pareto front over all 15 configs
  Panel B  structural self-consistency across the same front (the foldability cliff)

Replaces the two separate SI figures (sweep_pareto_si.png + foldability_cliff.png),
which the manuscript previously placed side by side with \\includegraphics.

Outputs (all versioned; the generic staged name is refreshed separately):
  fig_si_sweep_combined.png   400 dpi raster for the manuscript
  fig_si_sweep_combined.pdf   vector
  fig_si_sweep_combined.svg   vector with LIVE TEXT (svg.fonttype='none'), so the
                              labels stay editable in Inkscape/Illustrator

Two deliberate choices, both about keeping the SVG editable and the panels clean:
  * No mathtext ($...$) anywhere. Matplotlib renders mathtext as many separately
    positioned glyphs, which in an SVG becomes a pile of one-character <text>
    elements that are painful to edit. Unicode (beta, Delta, x10^-7) stays as one
    editable string per label.
  * Floating annotations were cut in favour of legend entries. The previous
    foldability figure carried four free-floating text blocks, one of which
    collided with the y-axis tick labels.

Data (read-only; nothing here re-scores anything):
  sweep_pareto_table.csv                      panel A, and the x axis of panel B
  sweep_boltz_selfconsist_table.csv           panel B, Boltz-2 full-length arm
  selfconsistency_sweep_<id>/selfconsistency_summary.json   panel B, ESMFold2 arm
"""
import os, csv, json, glob
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OD = os.path.join(PF, "CAPE_MPNN", "data", "output")
FIG = os.path.join(PF, "figures")
DEPLOYED = "f16b51e6"          # re-anchored 2026-08-19; NOT 7e908919
STEM = "fig_si_sweep_combined"

# --------------------------------------------------------------------- style
mpl.rcParams.update({
    "font.family": "sans-serif",
    # DejaVu Sans first, not Liberation Sans: Liberation lacks U+207B SUPERSCRIPT
    # MINUS, so "1×10⁻⁷" rendered as a missing-glyph box. DejaVu covers the whole
    # set (×, ⁻, β, Δ, ≤) which is what lets the labels stay single editable
    # strings instead of mathtext glyph piles.
    "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
    "font.size": 8, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.linewidth": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none",     # <- keeps SVG text as editable text
})
BLUE, ORANGE = "#0072B2", "#D55E00"
LR_MARKER = {1e-7: "o", 3e-7: "s", 1e-6: "^"}
LR_LABEL = {1e-7: "1×10⁻⁷", 3e-7: "3×10⁻⁷",
            1e-6: "1×10⁻⁶"}


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""


# ---------------------------------------------------------------------- data
rows = []
for r in csv.DictReader(open(os.path.join(OD, "sweep_pareto_table.csv"))):
    if r["netmhc_reduction_pct"] and r["seq_recovery_ft_pct"]:
        rows.append(dict(mid=r["model_id"], beta=float(r["beta"]), lr=float(r["lr"]),
                         net=float(r["netmhc_reduction_pct"]),
                         rec=float(r["seq_recovery_ft_pct"])))

boltz = {}
for r in csv.DictReader(open(os.path.join(OD, "sweep_boltz_selfconsist_table.csv"))):
    boltz[r["model_id"]] = dict(x=float(r["mhc2_reduction_pct"]),
                                d=float(r["delta_ft_minus_base"]),
                                sd=float(r["delta_std"]), n=int(r["n_prot"]),
                                p=float(r["paired_t_p"]))

esm = {}
for d in sorted(glob.glob(os.path.join(OD, "selfconsistency_sweep_*"))):
    mid = os.path.basename(d).split("_")[-1]
    try:
        j = json.load(open(os.path.join(d, "selfconsistency_summary.json")))["paired"]
    except (OSError, KeyError):
        continue
    esm[mid] = dict(d=j["delta"], sd=j["delta_std"], n=j["n"], p=j["t_p"])

# x axis of panel B is the measured MHC-II reduction, taken from the Boltz table
def series(store):
    pts = [(boltz[m]["x"], v["d"], v["sd"] / np.sqrt(v["n"]), v["p"])
           for m, v in store.items() if m in boltz]
    return sorted(pts)

esm_pts, boltz_pts = series(esm), series({k: v for k, v in boltz.items()})
print(f"panel A: {len(rows)} configs | panel B: ESMFold {len(esm_pts)}, "
      f"Boltz {len(boltz_pts)}")

# ------------------------------------------------------------------- figure
fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 3.15))

# ---- Panel A: Pareto front -------------------------------------------------
def dominated(p, others):
    return any(o["net"] >= p["net"] and o["rec"] >= p["rec"] and
               (o["net"] > p["net"] or o["rec"] > p["rec"]) for o in others)

front = sorted([p for p in rows if not dominated(p, rows)], key=lambda p: p["rec"])
norm = mpl.colors.LogNorm(vmin=0.01, vmax=0.2)
cmap = mpl.cm.viridis

axA.plot([p["rec"] for p in front], [p["net"] for p in front], "-", lw=1.0,
         color="0.55", zorder=1)
for p in rows:
    axA.scatter(p["rec"], p["net"], marker=LR_MARKER[p["lr"]], s=46,
                c=[cmap(norm(p["beta"]))], edgecolor="k", linewidth=0.5, zorder=3)

dep = next(p for p in rows if p["mid"] == DEPLOYED)
axA.scatter(dep["rec"], dep["net"], marker="o", s=200, facecolor="none",
            edgecolor=ORANGE, linewidth=1.5, zorder=4)
# One word: the configuration itself is already encoded by colour and marker,
# and spelled out in the caption.
axA.annotate("deployed", (dep["rec"], dep["net"]), textcoords="offset points",
             xytext=(-30, 30), ha="center", fontsize=7.5, color=ORANGE,
             arrowprops=dict(arrowstyle="-|>", lw=0.9, color=ORANGE,
                             connectionstyle="arc3,rad=0.25"))

axA.set_xlabel("sequence recovery (%)")
axA.set_ylabel("netMHCIIpan MHC class II reduction (%)")
axA.set_xlim(33.5, 51)
axA.set_ylim(0, 102)
axA.set_xticks([35, 40, 45, 50])          # integers; the 37.5/42.5 ticks were noise
axA.spines[["top", "right"]].set_visible(False)

sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, ax=axA, pad=0.03, fraction=0.055)
cb.set_label("DPO β (KL strength)", fontsize=8)
cb.set_ticks([0.01, 0.02, 0.05, 0.1, 0.2])
cb.ax.set_yticklabels(["0.01", "0.02", "0.05", "0.10", "0.20"], fontsize=7)
cb.outline.set_linewidth(0.6)

axA.legend(handles=[Line2D([], [], marker=LR_MARKER[lr], color="0.35",
                           linestyle="none", markersize=5.2, markeredgecolor="k",
                           markeredgewidth=0.5, label="lr " + LR_LABEL[lr])
                    for lr in (1e-7, 3e-7, 1e-6)]
                   + [Line2D([], [], color="0.55", lw=1.0, label="Pareto front")],
           frameon=False, loc="upper right", handletextpad=0.5, borderpad=0.2,
           labelspacing=0.3)

# ---- Panel B: foldability cliff -------------------------------------------
axB.axhspan(-0.05, 0.05, color="0.88", zorder=0)          # no-material-change band
axB.axhline(0, color="0.35", ls="--", lw=0.8, zorder=1)

for pts, col, mk, ls, above in ((esm_pts, BLUE, "o", "-", True),
                                (boltz_pts, ORANGE, "s", "--", False)):
    x, y, e = [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]
    axB.errorbar(x, y, yerr=e, color=col, marker=mk, ms=4.5, lw=1.2, ls=ls,
                 capsize=2, elinewidth=0.9, mec="k", mew=0.4, zorder=3)
    # Significance markers. Two things would otherwise collide: the two series
    # where their curves cross (~77-81%), fixed by putting ESMFold above its
    # points and Boltz below; and neighbouring significant points within one
    # series (76.9% and 81.0% sit ~4 apart on x), fixed by alternating the
    # offset so consecutive stars sit at different heights.
    sig = [p for p in pts if stars(p[3])]
    for j, (xi, yi, ei, pi) in enumerate(sig):
        far = j % 2 == 1
        dy = (10 if far else 6) if above else (-19 if far else -13)
        axB.annotate(stars(pi), (xi, yi + (ei if above else -ei)),
                     textcoords="offset points", xytext=(0, dy), ha="center",
                     fontsize=7, color=col, zorder=4)

# A vertical guide, not an arrow to one marker. Two reasons: the deployed config
# has a point in BOTH series (ESMFold +0.022, Boltz -0.013), so an arrow to
# either one misrepresents it; and an ORANGE callout here would read as labelling
# the orange Boltz-2 curve. Neutral grey marks the operating point for both arms.
xd = boltz[DEPLOYED]["x"]
axB.axvline(xd, color="0.45", lw=0.8, ls=":", zorder=1)
axB.annotate("deployed", (xd, 0.092), xytext=(3, 0), textcoords="offset points",
             ha="left", va="top", fontsize=7.5, color="0.30")

axB.set_xlabel("MHC class II epitope reduction (%)")
axB.set_ylabel("Δ TM (fine-tuned − base)")
axB.set_xlim(5, 100)
axB.set_ylim(-0.66, 0.10)
axB.spines[["top", "right"]].set_visible(False)
axB.legend(handles=[
    Line2D([], [], color=BLUE, marker="o", ms=4.5, lw=1.2, label="ESMFold2 (≤150 aa)"),
    Line2D([], [], color=ORANGE, marker="s", ms=4.5, lw=1.2, ls="--",
           label="Boltz-2 (full length)"),
    Patch(facecolor="0.88", label="no material change (±0.05)")],
    frameon=False, loc="lower left", handletextpad=0.6, borderpad=0.2,
    labelspacing=0.3)

fig.tight_layout(w_pad=2.6)
# Panel letters in FIGURE coordinates, placed after tight_layout: in axes
# coordinates the "B" landed on top of panel B's y-axis label.
for xf, lab in ((0.005, "A"), (0.505, "B")):
    fig.text(xf, 0.98, lab, fontsize=11, fontweight="bold", va="top", ha="left")
for ext in ("png", "pdf", "svg"):
    op = os.path.join(FIG, f"{STEM}.{ext}")
    if os.path.exists(op):
        raise SystemExit(f"Refusing to overwrite {op}; bump STEM or move it aside.")
    fig.savefig(op, dpi=400, bbox_inches="tight")
    print("wrote", os.path.basename(op))
