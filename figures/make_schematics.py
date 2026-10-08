import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
from matplotlib.lines import Line2D

# House palette (matching binder_redesign_v1/scripts/make_figure.py)
GRAY = "#9e9e9e"      # base / neutral
BLUE = "#2b7bba"      # fine-tuned / MHC-II pathway accent
RED = "#c0392b"       # highlight / ADA outcome
BLACK = "#111111"     # native / emphasis
ORANGE = "#e08214"    # MHC-I pathway accent
LIGHTBLUE = "#dbe9f6"
LIGHTORANGE = "#fbe5cf"
LIGHTGRAY = "#eeeeee"

plt.rcParams["font.family"] = "DejaVu Sans"


def box(ax, xy, w, h, text, fc="white", ec="black", fontsize=9, lw=1.3, weight="normal", zorder=3, textcolor="black"):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                        fc=fc, ec=ec, lw=lw, zorder=zorder)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color=textcolor, zorder=zorder + 1, linespacing=1.25)
    return p


def arrow(ax, start, end, color="black", lw=1.6, style="-|>", connectionstyle="arc3,rad=0.0", zorder=2):
    a = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=13,
                         color=color, lw=lw, connectionstyle=connectionstyle, zorder=zorder)
    ax.add_patch(a)


def peptide_chain(ax, x, y, n, r=0.035, dx=0.075, color=BLUE):
    for i in range(n):
        ax.add_patch(Circle((x + i * dx, y), r, fc=color, ec="black", lw=0.8, zorder=4))


def y_shape(ax, x, y, scale=0.09, color="black", lw=1.8):
    ax.plot([x, x], [y, y + scale * 0.9], color=color, lw=lw, zorder=4)
    ax.plot([x, x - scale * 0.7], [y + scale * 0.9, y + scale * 1.7], color=color, lw=lw, zorder=4)
    ax.plot([x, x + scale * 0.7], [y + scale * 0.9, y + scale * 1.7], color=color, lw=lw, zorder=4)


# =============================================================================
# FIGURE 1 — MHC class I vs class II antigen presentation and the ADA response
# =============================================================================
fig, ax = plt.subplots(figsize=(11.5, 6.4))
ax.set_xlim(0, 11.5)
ax.set_ylim(0, 6.4)
ax.axis("off")

# --- Therapeutic protein entering the cell ---
box(ax, (0.15, 5.35), 1.7, 0.55, "de novo designed\ntherapeutic protein", fc=LIGHTGRAY, fontsize=8.2)
arrow(ax, (1.85, 5.5), (2.55, 5.15), color=BLACK)

# --- Antigen-presenting cell outline ---
apc = FancyBboxPatch((2.5, 0.35), 6.6, 5.5, boxstyle="round,pad=0.02,rounding_size=0.35",
                      fc="white", ec=BLACK, lw=2.0, zorder=0.5)
ax.add_patch(apc)
ax.text(2.75, 5.68, "antigen-presenting cell", fontsize=9.5, fontweight="bold", color=BLACK, ha="left")

# --- MHC class I (cytosolic) pathway, left half of the cell ---
ax.text(4.05, 5.25, "MHC class I pathway (cytosolic)", fontsize=8.6, fontweight="bold",
        color=ORANGE, ha="center")
box(ax, (3.1, 4.35), 1.9, 0.6, "proteasome\n(cytosolic degradation)", fc=LIGHTORANGE, fontsize=7.6)
arrow(ax, (4.05, 4.35), (4.05, 3.75), color=ORANGE)
peptide_chain(ax, 3.55, 3.55, 3, color=ORANGE)
ax.text(4.35, 3.55, "8–10-mer", fontsize=7.2, va="center", color=ORANGE)
arrow(ax, (4.05, 3.42), (4.05, 2.85), color=ORANGE)
box(ax, (3.1, 2.35), 1.9, 0.55, "TAP transport into ER;\nloaded onto MHC class I", fc=LIGHTORANGE, fontsize=7.3)
arrow(ax, (4.05, 2.35), (4.05, 1.75), color=ORANGE)

# --- MHC class II (endosomal) pathway, right half of the cell ---
ax.text(7.35, 5.25, "MHC class II pathway (endosomal)", fontsize=8.6, fontweight="bold",
        color=BLUE, ha="center")
box(ax, (6.4, 4.35), 1.9, 0.6, "endosome / lysosome\n(cathepsin degradation)", fc=LIGHTBLUE, fontsize=7.6)
arrow(ax, (7.35, 4.35), (7.35, 3.75), color=BLUE)
peptide_chain(ax, 6.75, 3.55, 5, dx=0.065, color=BLUE)
ax.text(7.6, 3.55, "13–25-mer", fontsize=7.2, va="center", color=BLUE)
arrow(ax, (7.35, 3.42), (7.35, 2.85), color=BLUE)
box(ax, (6.4, 2.35), 1.9, 0.55, "loaded onto MHC class II\nin the MIIC compartment", fc=LIGHTBLUE, fontsize=7.3)
arrow(ax, (7.35, 2.35), (7.35, 1.75), color=BLUE)

# --- Cell surface presentation + T-cell recognition ---
ax.plot([2.5, 9.1], [1.55, 1.55], color=BLACK, lw=1.4, ls=(0, (4, 2)), zorder=0.6)
ax.text(2.6, 1.6, "cell surface", fontsize=7, color=BLACK, style="italic")

# MHC-I / peptide displayed, CD8 T cell
box(ax, (3.35, 1.05), 1.4, 0.42, "MHC-I + peptide", fc="white", ec=ORANGE, fontsize=7.2)
arrow(ax, (4.05, 1.05), (4.05, 0.75), color=BLACK, style="-")
box(ax, (3.15, 0.15), 1.8, 0.5, "CD8$^+$ cytotoxic T cell", fc=LIGHTORANGE, ec=ORANGE, fontsize=7.6)
ax.text(4.05, -0.32, "cytotoxic killing of\ninfected/aberrant cells", fontsize=7, ha="center", color=ORANGE)

# MHC-II / peptide displayed, CD4 T-helper cell
box(ax, (6.65, 1.05), 1.4, 0.42, "MHC-II + peptide", fc="white", ec=BLUE, fontsize=7.2)
arrow(ax, (7.35, 1.05), (7.35, 0.75), color=BLACK, style="-")
box(ax, (6.45, 0.15), 1.8, 0.5, "CD4$^+$ T-helper cell", fc=LIGHTBLUE, ec=BLUE, fontsize=7.6)

# --- Downstream ADA cascade (only from MHC-II / CD4 side) ---
arrow(ax, (8.25, 0.4), (9.25, 0.4), color=BLUE, connectionstyle="arc3,rad=-0.25")
ax.text(8.75, 0.75, "cytokines,\nCD40L–CD40", fontsize=6.6, ha="center", color=BLUE)
box(ax, (9.3, 0.15), 1.6, 0.5, "B cell $\\rightarrow$ plasma cell", fc=LIGHTBLUE, ec=BLUE, fontsize=7.4)
arrow(ax, (10.1, 0.65), (10.1, 1.15), color=RED)
y_shape(ax, 10.1, 1.2, scale=0.11, color=RED)
y_shape(ax, 10.45, 1.15, scale=0.09, color=RED)
ax.text(10.65, 1.7, "anti-drug\nantibodies\n(ADA)", fontsize=7.4, fontweight="bold", color=RED, ha="center")
ax.text(10.65, 2.55, "ADA neutralize the\ntherapeutic and\nreduce its half-life", fontsize=7.2, color=RED,
        ha="center", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=RED, lw=0.8))
# Route the ADA feedback arrow over the top of the cell, not through the middle.
ax.plot([10.1, 10.1], [3.15, 6.05], color=RED, lw=1.5, zorder=2)
ax.plot([10.1, 1.0], [6.05, 6.05], color=RED, lw=1.5, zorder=2)
arrow(ax, (1.0, 6.05), (1.0, 5.92), color=RED, lw=1.5)

ax.set_title("MHC class I versus class II antigen presentation and the anti-drug antibody (ADA) response",
              fontsize=10.5, fontweight="bold", pad=10)

plt.tight_layout()
# NOTE: Figure 1 (the MHC mechanism) is now produced by make_mhc_figure.py using
# Bioicons glyphs; this box-based version is superseded. Kept only as the archived
# fallback so re-running this script does NOT clobber the icon figure the manuscript uses.
plt.savefig("mhc_mechanism_figure_boxes_v1.png", dpi=300, bbox_inches="tight")
print("saved mhc_mechanism_figure_boxes_v1.png (Figure 1 is now make_mhc_figure.py)")
plt.close(fig)


# =============================================================================
# FIGURE 2 — DPO fine-tuning pipeline and evaluation overview
# =============================================================================
fig, ax = plt.subplots(figsize=(11.8, 9.4))
ax.set_xlim(0, 11.8)
ax.set_ylim(0, 9.4)
ax.axis("off")

DIVIDER_Y = 4.15
ax.plot([0.1, 11.6], [DIVIDER_Y, DIVIDER_Y], color=BLACK, lw=0.8, ls=(0, (5, 3)), zorder=0.5)
ax.text(0.05, 9.1, "A", fontsize=13, fontweight="bold")
ax.text(0.05, 3.95, "B", fontsize=13, fontweight="bold")

# --- Panel A: DPO fine-tuning loop (rows top -> bottom, all y >= DIVIDER_Y) ---
r1_top, r1_bot = 8.75, 9.35   # base model + sampling
r2_top, r2_bot = 7.35, 7.95   # three parallel scoring boxes
r3_top, r3_bot = 6.15, 6.75   # combined preference score + DPO loss
r4_top, r4_bot = 4.85, 5.4    # update weights / fine-tuned model

box(ax, (0.6, r1_top), 2.7, r1_bot - r1_top, "base ProteinMPNN\n(v\\_48\\_020)", fc=LIGHTGRAY, fontsize=8.8, weight="bold")
arrow(ax, (3.3, (r1_top + r1_bot) / 2), (4.1, (r1_top + r1_bot) / 2), color=BLACK)
box(ax, (4.15, r1_top), 3.3, r1_bot - r1_top, "sample candidate sequence pairs\nfor a PDB backbone", fc="white", fontsize=8.2)
arrow(ax, (5.0, r1_top), (2.2, r2_bot + 0.05), color=ORANGE, connectionstyle="arc3,rad=-0.2")
arrow(ax, (5.8, r1_top), (5.8, r2_bot + 0.05), color=BLUE)
arrow(ax, (6.6, r1_top), (9.5, r2_bot + 0.05), color=GRAY, connectionstyle="arc3,rad=0.2")

box(ax, (0.9, r2_top), 2.6, r2_bot - r2_top, "MHC-I PWM score\n(6 HLA-A/B/C alleles)", fc=LIGHTORANGE, ec=ORANGE, fontsize=7.8)
box(ax, (4.5, r2_top), 2.6, r2_bot - r2_top, "MHC-II PWM score\n(6 HLA-DRB1 alleles)", fc=LIGHTBLUE, ec=BLUE, fontsize=7.8)
box(ax, (8.1, r2_top), 2.8, r2_bot - r2_top, "acidic-composition\npenalty ($\\lambda\\cdot$excess D+E)", fc=LIGHTGRAY, fontsize=7.8)

box(ax, (3.6, r3_top), 4.6, r3_bot - r3_top,
    "preference score $=-$(MHC-I + MHC-II windows) $-\\lambda\\cdot$penalty\n$\\downarrow$ DPO loss (Rafailov et al. 2023)",
    fc="white", fontsize=7.9)
arrow(ax, (2.2, r2_top), (5.3, r3_bot - 0.03), color=ORANGE, connectionstyle="arc3,rad=0.25")
arrow(ax, (5.8, r2_top), (5.9, r3_bot - 0.03), color=BLUE)
arrow(ax, (9.5, r2_top), (6.5, r3_bot - 0.03), color=GRAY, connectionstyle="arc3,rad=-0.25")
arrow(ax, (5.9, r3_top), (5.9, r4_bot + 0.05), color=BLACK)

box(ax, (4.2, r4_top), 3.4, r4_bot - r4_top, "update ProteinMPNN weights", fc="white", fontsize=8.2)
arrow(ax, (8.6, (r4_top + r4_bot) / 2), (9.3, (r4_top + r4_bot) / 2), color=BLACK)
box(ax, (9.35, r4_top), 2.15, r4_bot - r4_top, "fine-tuned\nCAPE-MPNN-II", fc=LIGHTBLUE, ec=BLUE, fontsize=8.4, weight="bold")

# training-loop back-edge, routed down the left margin
loop_x = 0.35
arrow(ax, (4.15, (r4_top + r4_bot) / 2), (loop_x + 0.15, (r4_top + r4_bot) / 2), color=BLACK)
ax.plot([loop_x, loop_x], [(r4_top + r4_bot) / 2, (r1_top + r1_bot) / 2], color=BLACK, lw=1.6, zorder=2)
arrow(ax, (loop_x, (r1_top + r1_bot) / 2), (0.55, (r1_top + r1_bot) / 2), color=BLACK)
ax.text(loop_x - 0.18, (r1_top + r4_bot) / 2, "200 epochs; preference pairs\nregenerated every 2 epochs",
        fontsize=7.2, rotation=90, va="center", ha="center", color=BLACK)

# --- Panel B: application and evaluation (all y <= DIVIDER_Y) ---
b1_top, b1_bot = 3.15, 3.8   # two application boxes
b2_top, b2_bot = 1.15, 2.0   # two evaluation boxes

arrow(ax, (9.9, r4_top), (2.6, b1_bot + 0.05), color=BLACK, connectionstyle="arc3,rad=0.2")
arrow(ax, (10.3, r4_top), (9.1, b1_bot + 0.05), color=BLACK, connectionstyle="arc3,rad=-0.15")

box(ax, (0.6, b1_top), 4.0, b1_bot - b1_top, "100 held-out monomer\ntest proteins", fc="white", fontsize=8.4)
box(ax, (6.9, b1_top), 4.3, b1_bot - b1_top, "11 de novo binder\N{EN DASH}target complexes\n(binder redesigned, target fixed)", fc="white", fontsize=8.0)

arrow(ax, (2.6, b1_top), (2.6, b2_bot + 0.05), color=BLACK)
arrow(ax, (9.05, b1_top), (9.05, b2_bot + 0.05), color=BLACK)

box(ax, (0.2, b2_top), 4.8, b2_bot - b2_top, "evaluation: PWM $\\cdot$ netMHCIIpan\n(37.7\\% reduction, 20/20 proteins)", fc=LIGHTORANGE, ec=ORANGE, fontsize=7.8)
box(ax, (6.7, b2_top), 4.9, b2_bot - b2_top, "evaluation: PWM $\\cdot$ Boltz-2 $\\cdot$\nAF2 initial-guess (no binding cost)", fc=LIGHTBLUE, ec=BLUE, fontsize=7.8)

ax.set_title("Overview of the DPO fine-tuning pipeline (A) and its evaluation (B)",
              fontsize=11, fontweight="bold", pad=12)

plt.tight_layout()
# NOTE: Figure 2 (pipeline overview) is now produced by make_pipeline_flowchart.py
# (SVG flowchart rasterized with Inkscape); this matplotlib version is superseded and
# writes to a fallback name so re-running this script does NOT clobber the flowchart.
plt.savefig("pipeline_overview_figure_v1.png", dpi=300, bbox_inches="tight")
print("saved pipeline_overview_figure_v1.png (Figure 2 is now make_pipeline_flowchart.py)")
plt.close(fig)
