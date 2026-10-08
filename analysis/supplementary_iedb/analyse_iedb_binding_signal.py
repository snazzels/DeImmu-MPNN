#!/usr/bin/env python
"""Does netMHCIIpan-predicted presentation predict EXPERIMENTAL immunogenicity?

Follow-up to mine_IEDB_negatives.py (roadmap P1.2). Reads the netMHCIIpan-scored
IEDB positives and negatives and asks the question the whole project rests on.

The DPO objective minimises netMHCIIpan-predicted presented windows. That is only
a valid immunogenicity objective if presentation actually predicts immunogenicity.
This script measures how well it does, on experimentally-labelled IEDB peptides.

Two scoring regimes, and they give very different answers:

  (a) min %Rank over the 6-allele DRB1 panel -- what the design pipeline does,
      because the patient's HLA type is unknown.
  (b) %Rank on the peptide's ACTUAL restricting allele, as recorded in IEDB --
      only available for the minority of records that name a specific allele.

No netMHCIIpan calls; reads cached scores. Runs in ~1 minute (IEDB CSV load).

Outputs
-------
data/output/iedb_binding_signal.txt
data/output/iedb_binding_signal_figure.png
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

PF = os.environ.get("PF", os.path.join(os.path.dirname(__file__), ".."))

IEDB_DIR = os.path.join(PF, "data", "input", "immuno", "mhc_2", "iedb")
CACHE = os.path.join(IEDB_DIR, "cache", "tcell_full_v3.csv")
OUT_TXT = os.path.join(PF, "data", "output", "iedb_binding_signal.txt")
OUT_FIG = os.path.join(PF, "data", "output", "iedb_binding_signal_figure.png")

ALLELES = ["DRB1_0101", "DRB1_0301", "DRB1_0401",
           "DRB1_0701", "DRB1_1101", "DRB1_1501"]
PANEL_MAP = {"HLA-DRB1*01:01": "DRB1_0101", "HLA-DRB1*03:01": "DRB1_0301",
             "HLA-DRB1*04:01": "DRB1_0401", "HLA-DRB1*07:01": "DRB1_0701",
             "HLA-DRB1*11:01": "DRB1_1101", "HLA-DRB1*15:01": "DRB1_1501"}
EFFECTOR = {"IFNg release", "proliferation", "cytotoxicity", "TNFa release",
            "TNF release", "activation", "degranulation", "granzyme B release",
            "antibody help", "T cell help", "IFNa release", "IFNb release"}
STD_AA = set("ACDEFGHIKLMNPQRSTVWY")
THRESHOLDS = [0.5, 1, 2, 5, 10, 20]


def auc(pos_scores, neg_scores):
    """AUROC via Mann-Whitney U. Higher score => predicted immunogenic."""
    p, n = np.asarray(pos_scores, float), np.asarray(neg_scores, float)
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    u, _ = mannwhitneyu(p, n, alternative="two-sided")
    return u / (len(p) * len(n))


def valid(s):
    s = str(s).strip().upper()
    return len(s) == 15 and set(s) <= STD_AA


def restricting_allele_subset(pos, neg):
    """Rows whose IEDB-recorded restricting allele is in our panel, with that allele's %Rank."""
    print(f"Loading {CACHE} for restricting-allele annotations…")
    df = pd.read_csv(CACHE, low_memory=False, header=[0, 1])
    df.columns = [f"{a}|{b}" if not str(b).startswith("Unnamed") else str(a)
                  for a, b in df.columns]
    ce, ch, ca = "Epitope|Name", "Host|Name", "MHC Restriction|Name"
    cc, cq, cr = ("MHC Restriction|Class", "Assay|Qualitative Measurement",
                  "Assay|Response measured")

    s = df[df[ch].str.contains("Homo sapiens", na=False, case=False)
           & df[cc].astype(str).str.contains("II", na=False)]
    s = s[s[cr].isin(EFFECTOR)].copy()
    s["pep"] = s[ce].astype(str).str.strip().str.upper()
    s = s[s.pep.map(valid)]

    n_total = len(s)
    s["panel"] = s[ca].map(PANEL_MAP)
    n_panel = s.panel.notna().sum()
    s = s[s.panel.notna()]

    ranks = pd.concat([pos, neg]).set_index("peptide")
    s = s[s.pep.isin(ranks.index)]
    s["rank_true"] = [ranks.loc[p, a] for p, a in zip(s.pep, s.panel)]
    s = s.dropna(subset=["rank_true"])

    s["outcome"] = np.where(s[cq].str.startswith("Positive", na=False), 1,
                            np.where(s[cq].eq("Negative"), 0, np.nan))
    s = s.dropna(subset=["outcome"]).drop_duplicates(["pep", "panel", "outcome"])
    # drop peptide/allele pairs tested both ways
    amb = s.groupby(["pep", "panel"]).outcome.nunique()
    s = s[~s.set_index(["pep", "panel"]).index.isin(amb[amb > 1].index)]
    return s, n_total, n_panel


def main(args):
    for f in (os.path.join(IEDB_DIR, "immunogenic_binding.csv"),
              os.path.join(IEDB_DIR, "nonimmunogenic.csv")):
        if not os.path.exists(f):
            sys.exit(f"Missing {f}. Run mine_IEDB_negatives.py first.")

    pos = pd.read_csv(os.path.join(IEDB_DIR, "immunogenic_binding.csv"))
    neg = pd.read_csv(os.path.join(IEDB_DIR, "nonimmunogenic.csv"))

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit("DOES netMHCIIpan PRESENTATION PREDICT EXPERIMENTAL IMMUNOGENICITY?")
    emit("=" * 78)
    emit("Positives : IEDB effector-assay POSITIVE 15-mers")
    emit("Negatives : IEDB effector-assay NEGATIVE 15-mers (never observed positive)")
    emit(f"n_pos = {len(pos):,}   n_neg = {len(neg):,}")
    emit("")
    emit("Score convention: lower %Rank = stronger predicted binder. We test whether")
    emit("-%Rank predicts the experimental label. AUC 0.5 = no better than chance.")
    emit("")

    # ── Regime (a): min over panel -- what the design pipeline actually does ──
    emit("-" * 78)
    emit("(a) min %Rank over the 6-allele DRB1 panel")
    emit("    This is what the DPO pipeline does: the patient's HLA type is unknown,")
    emit("    so a designed sequence is scored against the whole panel.")
    emit("-" * 78)
    a_best = auc(-pos.best_rank, -neg.best_rank)
    a_nall = auc(pos.n_alleles_presented, neg.n_alleles_presented)
    emit(f"AUC, best %Rank over panel     : {a_best:.4f}")
    emit(f"AUC, n alleles presented       : {a_nall:.4f}")
    emit("")
    emit("Per-allele AUC:")
    for al in ALLELES:
        emit(f"  {al:<12} {auc(-pos[al], -neg[al]):.4f}")
    emit("")
    emit(f"% predicted binders (<=2% rank): immunogenic {100*pos.is_binder.mean():.1f}%   "
         f"non-immunogenic {100*neg.is_binder.mean():.1f}%")
    emit(f"median best %Rank              : immunogenic {pos.best_rank.median():.2f}   "
         f"non-immunogenic {neg.best_rank.median():.2f}")
    emit("")
    emit(">>> In the deployment regime, predicted presentation carries essentially NO")
    emit(">>> information about experimental immunogenicity.")
    emit("")

    # ── Regime (b): the true restricting allele ──────────────────────────────
    sub, n_total, n_panel = restricting_allele_subset(pos, neg)
    P = sub[sub.outcome == 1].rank_true
    N = sub[sub.outcome == 0].rank_true
    a_true = auc(-P, -N)
    _, p_mw = mannwhitneyu(P, N, alternative="less")

    emit("-" * 78)
    emit("(b) %Rank on the peptide's ACTUAL restricting allele (IEDB-recorded)")
    emit("-" * 78)
    emit(f"Effector-assay MHC-II 15-mer rows          : {n_total:,}")
    emit(f"  ... naming a specific allele in our panel: {n_panel:,} "
         f"({100*n_panel/max(n_total,1):.1f}%)")
    emit("  The overwhelming majority of IEDB MHC-II records give the restriction")
    emit("  only as 'HLA class II', so this subset is small and biased toward")
    emit("  well-characterised epitopes.")
    emit("")
    emit(f"n_pos = {len(P):,}   n_neg = {len(N):,}")
    emit(f"AUC (-%Rank predicts immunogenicity) : {a_true:.4f}")
    emit(f"Mann-Whitney (positives bind better) : p = {p_mw:.3g}")
    emit(f"median %Rank : immunogenic {P.median():.2f}   non-immunogenic {N.median():.2f}")
    emit("")
    emit(">>> Given the correct allele, presentation carries a real but WEAK signal.")
    emit(">>> Taking the min over an allele panel destroys it.")
    emit("")

    # ── Threshold sweep ──────────────────────────────────────────────────────
    emit("-" * 78)
    emit("Sensitivity / specificity of the %Rank_EL threshold (true restricting allele)")
    emit("-" * 78)
    emit(f"{'threshold':>10}{'sens (TPR)':>13}{'FPR':>10}{'enrichment':>13}")
    rows = []
    for th in THRESHOLDS:
        tpr = (P <= th).mean()
        fpr = (N <= th).mean()
        enr = tpr / max(fpr, 1e-9)
        rows.append((th, tpr, fpr, enr))
        mark = "   <- used in this project" if th == 2 else ""
        emit(f"{th:>9.1f}%{100*tpr:>12.1f}%{100*fpr:>9.1f}%{enr:>12.2f}x{mark}")
    emit("")
    emit(f"At the 2% cutoff, netMHCIIpan misses {100*(1-(P<=2).mean()):.0f}% of")
    emit("experimentally-confirmed immunogenic epitopes, and flags "
         f"{100*(N<=2).mean():.0f}% of confirmed")
    emit("non-immunogenic peptides. Enrichment is only ~1.5x at every threshold.")
    emit("")
    emit("NOTE: paper_outline.md 2.2 specifies a 10% rank cutoff ('field standard for")
    emit("MHC-II weak binders'); the implementation uses 2%. At 10% sensitivity rises")
    emit("to %.0f%% but FPR rises to %.0f%% -- enrichment is unchanged. The threshold is"
         % (100*(P <= 10).mean(), 100*(N <= 10).mean()))
    emit("not the problem; the predictor's immunogenicity signal is.")
    emit("")

    # ── Consequences ─────────────────────────────────────────────────────────
    emit("=" * 78)
    emit("WHAT THIS MEANS -- read carefully, and note the caveats")
    emit("=" * 78)
    emit("1. The DPO objective minimises predicted presented windows, scored as the min")
    emit("   over an allele panel. On experimentally-labelled peptides that quantity has")
    emit(f"   AUC = {a_best:.3f} for immunogenicity. The reduction we report is real, but it is")
    emit("   a reduction in PREDICTED PRESENTATION, not demonstrably in immunogenicity.")
    emit("")
    emit("2. This is not a bug in our pipeline. It is a property of netMHCIIpan, and")
    emit("   Gasser et al. state it explicitly: 'Being visible on the cell's surface is")
    emit("   a necessary but not a sufficient condition for a CTL reaction (it does not")
    emit("   imply immunogenicity).' Presentation is necessary, not sufficient.")
    emit("")
    emit("3. CAVEATS that limit how strongly this can be claimed:")
    emit("   - Selection bias: IEDB negatives are peptides someone chose to test. They")
    emit("     are not random peptides, and are plausibly enriched for predicted binders.")
    emit("     Against a random background, presentation WOULD separate strongly -- but")
    emit("     that separation is binding, not immunogenicity.")
    emit("   - Allele mismatch: most IEDB MHC-II records do not name the restricting")
    emit("     allele, and the true allele is often outside our 6-DRB1 panel.")
    emit("   - Only exact 15-mers are used; IEDB MHC-II epitopes range 13-25 aa.")
    emit("   - Immunogenicity depends on donor T-cell repertoire, tolerance and antigen")
    emit("     processing -- none of which any binding predictor models.")
    emit("")
    emit("4. The defensible framing for the manuscript is therefore NOT 'netMHCIIpan is")
    emit("   useless', but: 'presentation is necessary but not sufficient; a presentation-")
    emit("   only objective cannot be assumed to reduce immunogenicity; we therefore")
    emit("   validate on an independent, experimentally-grounded IEDB axis.'")
    emit("   This is precisely the argument for the IEDB training signal that the funded")
    emit("   proposal demands, and it is now supported by our own measurement rather than")
    emit("   asserted.")
    emit("")
    emit("5. It also raises the value of the binder-matched dataset: within peptides that")
    emit("   ARE predicted binders, presentation cannot separate the classes at all, so a")
    emit("   classifier trained there must learn genuine immunogenicity features.")
    emit("=" * 78)

    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_TXT}")

    # ── Figure ───────────────────────────────────────────────────────────────
    if not args.no_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
        C_POS, C_NEG = "#C44E52", "#4C72B0"

        # (A) %Rank distributions, panel-min regime
        ax = axes[0]
        bins = np.linspace(0, 40, 60)
        ax.hist(pos.best_rank, bins=bins, alpha=0.6, density=True, color=C_POS,
                label=f"immunogenic (n={len(pos):,})")
        ax.hist(neg.best_rank, bins=bins, alpha=0.6, density=True, color=C_NEG,
                label=f"non-immunogenic (n={len(neg):,})")
        ax.axvline(2, color="k", ls="--", lw=1, label="2% cutoff")
        ax.set_xlabel("best %Rank_EL over 6-allele panel")
        ax.set_ylabel("density")
        ax.set_title("(A)", loc="left")
        ax.legend(fontsize=7)

        # (B) true restricting allele
        ax = axes[1]
        ax.hist(P, bins=bins, alpha=0.6, density=True, color=C_POS,
                label=f"immunogenic (n={len(P):,})")
        ax.hist(N, bins=bins, alpha=0.6, density=True, color=C_NEG,
                label=f"non-immunogenic (n={len(N):,})")
        ax.axvline(2, color="k", ls="--", lw=1)
        ax.set_xlabel("%Rank_EL on true restricting allele")
        ax.set_title("(B)", loc="left")
        ax.legend(fontsize=7)

        # (C) enrichment vs threshold
        ax = axes[2]
        th = [r[0] for r in rows]
        ax.plot(th, [100*r[1] for r in rows], "o-", color=C_POS, label="sensitivity (TPR)")
        ax.plot(th, [100*r[2] for r in rows], "s-", color=C_NEG, label="FPR")
        ax.set_xscale("log")
        ax.set_xticks(th)
        ax.set_xticklabels([str(t) for t in th])
        ax.set_xlabel("%Rank_EL threshold")
        ax.set_ylabel("%")
        ax.axvline(2, color="k", ls="--", lw=1, label="2% (used here)")
        ax2 = ax.twinx()
        ax2.plot(th, [r[3] for r in rows], "^--", color="gray", label="enrichment")
        ax2.set_ylabel("enrichment (TPR/FPR)", color="gray")
        ax2.set_ylim(0, 3)
        ax.set_title("(C)", loc="left")
        ax.legend(fontsize=7, loc="upper left")

        for a in axes:
            a.spines[["top"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT_FIG, dpi=300)
        print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no_figure", action="store_true")
    main(ap.parse_args())
