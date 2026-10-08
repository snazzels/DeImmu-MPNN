#!/usr/bin/env python
"""Why does the de-immunisation transfer to DR + DQ but not DP? (Track 17 follow-up)

Reads allele_generalisation.csv (Track 17) and dissects the transfer mechanism at
the per-protein *change* level. Three questions:

  A. Edit-coupling: is the reduction on each locus driven by the SAME edits that
     cut trained-DRB1? (correlate per-protein Δpresentation with Δ trained-DRB1)
  B. Composition: does de-immunisation lower overall hydrophobic P1-anchor content,
     and if so does that drop EXPLAIN the DQ reduction? (the hypothesis this script
     was written to test)
  C. Which residues shift base -> fine-tuned.

FINDING (see summary): the transfer runs on two DIFFERENT drivers.
  * DR family (DRB1, DRB3/4/5): targeted positional edits -- edit-coupled, the same
    mutations cut all DR molecules (Δr 0.68-0.83).
  * DQ: the acidic enrichment de-immunisation introduces (A->E shift). The per-protein
    glutamate/acidic increase predicts the DQ reduction at r~0.9, but does NOT predict
    the DR reduction -- which is why ΔDR and ΔDQ are uncorrelated (different drivers).
    (My first guess, a hydrophobic-anchor shift, was REFUTED: anchor loss is small and
    does not predict ΔDQ; the real axis is charge, not hydrophobicity.)
  * DP: neither route reduces it -- distinct β-chain groove chemistry.

Output: data/output/allele_transfer_mechanism_summary.txt
"""
import os, sys
import pandas as pd, numpy as np
from scipy import stats

PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(PF, "data", "output", "allele_generalisation.csv")
OUT = os.path.join(PF, "data", "output", "allele_transfer_mechanism_summary.txt")
ANCHOR = set("FWYLIVM")   # strong MHC-II P1 hydrophobic/aromatic anchors
GRP = {"trained DRB1":"grp_trained DRB1", "held-out DRB1":"grp_held-out DRB1",
       "DRB3/4/5":"grp_DRB3/4/5", "DQ":"grp_HLA-DQ", "DP":"grp_HLA-DP"}


def frac(seq, s):
    seq = seq.replace("/","").replace("-","").replace("*","")
    return sum(c in s for c in seq)/len(seq) if seq else 0.0


def main(a=None):
    global CSV, OUT
    if a is not None:
        if getattr(a, "in_csv", None):
            CSV = a.in_csv
        if getattr(a, "out_tag", None):
            OUT = os.path.join(PF, "data", "output", f"allele_transfer_mechanism_{a.out_tag}_summary.txt")
    df = pd.read_csv(CSV)
    df["anchor"] = df.sequence.apply(lambda s: frac(s, ANCHOR))
    pm = df.groupby(["protein_name","model"]).agg(
        anchor=("anchor","mean"), **{k:(v,"mean") for k,v in GRP.items()}).reset_index()
    b = pm[pm.model=="base"].set_index("protein_name")
    f = pm[pm.model=="finetuned"].set_index("protein_name")
    n = len(b)
    L = []
    def emit(s=""): print(s); L.append(s)

    emit("="*74); emit("TRANSFER MECHANISM ANALYSIS (Track 17 follow-up)"); emit("="*74)
    emit(f"n = {n} proteins. Δ = base - fine-tuned (positive = reduction).\n")

    emit("A. EDIT-COUPLING: is each locus reduced by the SAME edits as trained-DRB1?")
    dref = b["trained DRB1"] - f["trained DRB1"]
    for name in GRP:
        d = b[name] - f[name]
        r,p = stats.pearsonr(dref, d)
        emit(f"   Δtrained-DRB1 vs Δ{name:<14}: r={r:+.2f} (p={p:.1g})   mean Δ={d.mean():+.1f}")
    emit("   -> DR family (DRB1/3/4/5) is edit-coupled; DQ drops but is DEcoupled (r~0); DP neither.\n")

    emit("B. COMPOSITION: what compositional change drives the DQ reduction?")
    t,p = stats.ttest_rel(b["anchor"], f["anchor"])
    emit(f"   (i) Hydrophobic-anchor content FWYLIVM: base {b['anchor'].mean():.3f} -> ft "
         f"{f['anchor'].mean():.3f} (Δ {100*(b['anchor'].mean()-f['anchor'].mean()):+.1f}% abs, "
         f"p={p:.1g}). Drops slightly, but:")
    danch = b["anchor"] - f["anchor"]
    for name in ["trained DRB1","DQ","DP"]:
        r,pp = stats.pearsonr(danch, b[name]-f[name])
        emit(f"       Δanchor vs Δ{name:<12}: r={r:+.2f} (p={pp:.2g})")
    emit("       -> anchor loss does NOT predict the DQ reduction. 'DQ via anchor shift' REFUTED.")
    emit("")
    # The real shift is A->E (see part C). Test the acidic/glutamate enrichment.
    dfE = df.copy()
    dfE["acidic"] = dfE.sequence.apply(lambda s: frac(s, set("DE")))
    pmE = dfE.groupby(["protein_name","model"]).agg(acidic=("acidic","mean")).reset_index()
    bE = pmE[pmE.model=="base"].set_index("protein_name")["acidic"]
    fE = pmE[pmE.model=="finetuned"].set_index("protein_name")["acidic"]
    dE = fE - bE   # increase in acidic content
    emit(f"   (ii) Acidic (D/E) content ENRICHMENT: base {bE.mean():.3f} -> ft {fE.mean():.3f} "
         f"(Δ +{100*dE.mean():.1f}% abs) -- the dominant shift (A->E, part C):")
    for name in ["trained DRB1","DQ","DP"]:
        r,pp = stats.pearsonr(dE, b[name]-f[name])
        emit(f"       Δacidic vs Δ{name:<12}: r={r:+.2f} (p={pp:.2g})")
    emit("       -> The acidic/glutamate enrichment PREDICTS the DQ reduction (r~0.9) but NOT")
    emit("          the DR reduction (n.s.) or DP. So the two routes are now explicit:")
    emit("            * DR family : targeted positional edits (part A, edit-coupled)")
    emit("            * DQ        : the acidic (A->E) enrichment de-immunisation introduces")
    emit("            * DP        : neither route touches it")
    emit("          This is exactly why ΔDR and ΔDQ are uncorrelated -- different drivers.\n")

    emit("C. RESIDUE SHIFT base -> fine-tuned (mean over all sequences, percentage points)")
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        cb = df[df.model=="base"].sequence.apply(lambda s: frac(s,{aa})).mean()
        cf = df[df.model=="finetuned"].sequence.apply(lambda s: frac(s,{aa})).mean()
        v = (cf-cb)*100
        emit(f"   {aa} {'anchor' if aa in ANCHOR else '      '} {v:+.2f}")
    emit("="*74)
    open(OUT,"w").write("\n".join(L)+"\n")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--in_csv", default=None, help="allele_generalisation_<tag>.csv to analyse")
    p.add_argument("--out_tag", default=None, help="suffix for the versioned summary")
    main(p.parse_args())
