#!/usr/bin/env python
"""P1-3 / R3-W2 -- aggregation and PTM liability panel, base vs fine-tuned.

THE CHALLENGE THIS ANSWERS. The developability panel in the manuscript reports net
charge, pI, GRAVY and acidic/basic fractions, and omits aggregation propensity and
post-translational-modification liability. The pharma-perspective seat noted that
aggregation is itself an ADA driver, so an objective that lowered predicted epitope
burden while raising aggregation risk could invert the paper's thesis. That is a
real possibility and it is cheap to test.

WHAT THIS IS, AND WHAT IT IS NOT. This is a SEQUENCE-LEVEL SCREEN FOR CHANGE, not
an absolute aggregation prediction. CamSol, Aggrescan3D and SAP were not run: the
first two are web services and SAP needs a structure and a force field. What is
computed here instead:

  * PTM liability motifs, which are exact string patterns and need no model at all.
    N-glycosylation sequon N-X-[ST] with X != P; deamidation sites NG/NS/NT (NG is
    the fast one); aspartate isomerisation DG/DS; and free-cysteine, methionine and
    tryptophan counts for oxidation and disulfide-scrambling risk.

  * A hydrophobic-patch proxy for aggregation propensity: the number of windows
    whose mean Kyte-Doolittle hydropathy exceeds a threshold, and the single most
    hydrophobic window. Kyte-Doolittle is the same scale the manuscript already
    uses for GRAVY, so no new and unverifiable numeric table is introduced.

The paired design is what makes this informative despite the crudeness of the
proxy. Base and fine-tuned designs come from identical backbones, so a proxy that
is a poor absolute predictor can still detect a RELATIVE shift, and "no detectable
shift" is the claim the thesis actually needs. An absolute aggregation risk is not
claimed and would not be supported by these quantities.

MULTIPLICITY. Ten metrics are tested on one sample, so an uncorrected 0.05 would be
expected to produce a false positive here roughly half the time. Holm-Bonferroni is
applied across the whole panel and both raw and adjusted p-values are reported. The
direction of each significant shift is stated, because for this question a
significant IMPROVEMENT and a significant DETERIORATION are not interchangeable.

Refuses to overwrite existing output.
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

try:
    from Bio.SeqUtils.ProtParamData import kd as KD
except ImportError:
    sys.exit("FATAL: Biopython required (Bio.SeqUtils.ProtParamData.kd).")

SEQUON = re.compile(r"N[^P][ST]")
WINDOW = 7
PATCH_THRESHOLD = 1.5   # mean KD over the window; ~the 90th percentile of natural 7-mers


def hydrophobic_patches(seq, w=WINDOW, thr=PATCH_THRESHOLD):
    """(count of windows above threshold, most hydrophobic window mean)."""
    vals = [KD.get(a, 0.0) for a in seq]
    if len(vals) < w:
        return 0, float("nan")
    means = np.convolve(vals, np.ones(w) / w, mode="valid")
    return int((means > thr).sum()), float(means.max())


def count_overlapping(seq, pairs):
    return sum(seq.count(p) for p in pairs)


def metrics(seq):
    seq = str(seq).upper()
    n_patch, max_patch = hydrophobic_patches(seq)
    L = max(len(seq), 1)
    return {
        "n_glyc_sequon":  len(SEQUON.findall(seq)),
        "deamid_NG":      seq.count("NG"),
        "deamid_NS_NT":   count_overlapping(seq, ("NS", "NT")),
        "isomer_DG_DS":   count_overlapping(seq, ("DG", "DS")),
        "cysteine":       seq.count("C"),
        "methionine":     seq.count("M"),
        "tryptophan":     seq.count("W"),
        "hydrophobic_patches":     n_patch,
        "hydrophobic_patches_per100": 100.0 * n_patch / L,
        "max_patch_hydropathy":    max_patch,
    }


# Lower is better for every metric here: each is a liability count or a
# hydrophobicity measure. Used only to phrase the direction of a shift.
METRIC_ORDER = ["n_glyc_sequon", "deamid_NG", "deamid_NS_NT", "isomer_DG_DS",
                "cysteine", "methionine", "tryptophan",
                "hydrophobic_patches", "hydrophobic_patches_per100",
                "max_patch_hydropathy"]


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="per-design CSV with a 'sequence' column")
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--out_dir", default="CAPE_MPNN/data/output")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    out_csv = os.path.join(args.out_dir, f"{args.out_tag}.csv")
    out_txt = os.path.join(args.out_dir, f"{args.out_tag}_summary.txt")
    for p in (out_csv, out_txt):
        if os.path.exists(p):
            sys.exit(f"REFUSING TO OVERWRITE {p} -- use a new --out_tag.")

    d = pd.read_csv(args.csv)
    for col in ("protein_name", "model", "sequence"):
        if col not in d.columns:
            sys.exit(f"FATAL: column '{col}' not in {args.csv}")

    recs = []
    for _, row in d.iterrows():
        m = metrics(row.sequence)
        m.update(protein_name=row.protein_name, model=row.model)
        recs.append(m)
    per_design = pd.DataFrame(recs)

    # Collapse to one value per protein per arm, then pair on protein.
    per_prot = per_design.groupby(["protein_name", "model"])[METRIC_ORDER].mean().reset_index()
    base = per_prot[per_prot.model == "base"].set_index("protein_name")
    fine = per_prot[per_prot.model == "finetuned"].set_index("protein_name")
    common = sorted(set(base.index) & set(fine.index))
    if not common:
        sys.exit("FATAL: no proteins present in both arms.")
    dropped = (len(set(base.index) | set(fine.index)) - len(common))
    base, fine = base.loc[common], fine.loc[common]

    out = pd.DataFrame({"protein_name": common})
    for m in METRIC_ORDER:
        out[f"base_{m}"] = base[m].to_numpy()
        out[f"fine_{m}"] = fine[m].to_numpy()
        out[f"delta_{m}"] = fine[m].to_numpy() - base[m].to_numpy()
    out.to_csv(out_csv, index=False)

    rows, praw = [], []
    for m in METRIC_ORDER:
        b, f = base[m].to_numpy(float), fine[m].to_numpy(float)
        diff = f - b
        if np.allclose(diff, 0):
            p_t = p_w = 1.0
        else:
            p_t = stats.ttest_rel(f, b).pvalue
            try:
                p_w = stats.wilcoxon(f, b).pvalue
            except ValueError:
                p_w = 1.0
        rows.append([m, b.mean(), f.mean(), diff.mean(), p_t, p_w])
        praw.append(p_t)
    adj = holm(np.array(praw))

    L = []
    A = L.append
    A("AGGREGATION AND PTM LIABILITY PANEL (P1-3 / R3-W2)")
    A("=" * 84)
    A(f"source CSV : {os.path.basename(args.csv)}")
    A(f"proteins   : {len(common)} paired" + (f"  ({dropped} present in only one arm, dropped)" if dropped else ""))
    A(f"designs    : per-protein means over all designs in each arm, then paired on protein")
    A("")
    A("SCOPE. These are sequence-level proxies computed for CHANGE between two arms")
    A("designed from identical backbones. CamSol, Aggrescan3D and SAP were NOT run.")
    A("No absolute aggregation risk is claimed; what is tested is whether the")
    A("de-immunisation objective moved any liability axis, in either direction.")
    A("")
    A(f"{'metric':<30}{'base':>10}{'fine-tuned':>12}{'delta':>10}{'p (raw)':>11}{'p (Holm)':>11}  direction")
    A("-" * 96)
    worse, better = [], []
    for (m, bm, fm, dm, p_t, p_w), pa in zip(rows, adj):
        if pa < args.alpha:
            direction = "WORSE (higher liability)" if dm > 0 else "better (lower liability)"
            (worse if dm > 0 else better).append(m)
        else:
            direction = "no detectable change"
        A(f"{m:<30}{bm:>10.3f}{fm:>12.3f}{dm:>+10.3f}{p_t:>11.2e}{pa:>11.2e}  {direction}")
    A("")
    A(f"Holm-Bonferroni across all {len(rows)} metrics at alpha={args.alpha}. Wilcoxon p-values")
    A("are in the CSV alongside the per-protein values; the table above tests the")
    A("paired t on per-protein means, with Holm applied to those.")
    A("")
    A("READING:")
    if worse:
        A(f"  ** {len(worse)} metric(s) moved toward HIGHER liability: {', '.join(worse)}.")
        A("     This is the direction that could offset the epitope reduction and must be")
        A("     reported as such, with effect sizes, not only as a p-value.")
    else:
        A("  No metric moved toward higher liability at the corrected threshold.")
    if better:
        A(f"  {len(better)} metric(s) moved toward LOWER liability: {', '.join(better)}.")
    if not worse and not better:
        A("  The panel is flat: the objective did not detectably move any liability axis.")
    A("")
    A("  A null here is a bounded negative result, not proof of equivalence: with this")
    A("  sample a small shift would not be detected, and the proxies are crude by")
    A("  construction. What it excludes is a large shift on these axes.")
    A("=" * 84)

    with open(out_txt, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[out] {out_csv}\n[out] {out_txt}")


if __name__ == "__main__":
    main()
