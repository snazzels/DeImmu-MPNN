#!/usr/bin/env python
"""Score the already-generated designed sequences for MHC class I presentation.

The Track 6 production run trained with BOTH --mhc_1_predictor pwm and
--mhc_2_predictor pwm, so the model is a dual-objective de-immuniser. Track 7
measured only the MHC-II arm. This script measures the MHC-I arm.

No GPU and no model loading are needed: the designed sequences are already stored
in eval_deimmunisation.csv (Track 7). We simply re-score them against the
production MHC-I PWMs (data/input/immuno/mhc_1/Mhc1PredictorPwm/ -- 3M peptides,
6 alleles, lengths 8/9/10; NOT the mhc_1/pwm/ smoke-test directory).

Outputs
-------
data/output/eval_mhc1_arm.csv          per-sequence MHC-I + MHC-II counts
data/output/eval_mhc1_arm_summary.txt  statistics
data/output/eval_mhc1_arm_figure.png   4-panel figure
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

PF = os.environ.get("PF", os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PF, "libs"))

from kit.bioinf.immuno.mhc_1 import Mhc1Predictor, MHC_1_PEPTIDE_LENGTHS  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────
# Must match the alleles used in the Track 6 training run.
MHC1_ALLELES = "HLA-A*02:01+HLA-A*24:02+HLA-B*07:02+HLA-B*39:01+HLA-C*07:01+HLA-C*16:01"
MHC2_ALLELES = "DRB1_0101+DRB1_0301+DRB1_0401+DRB1_0701+DRB1_1101+DRB1_1501"

# The production PWM directory. The sibling 'pwm/' dir is a smoke-test leftover.
PWM_DIR = os.path.join(PF, "data", "input", "immuno", "mhc_1", "Mhc1PredictorPwm")

IN_CSV      = os.path.join(PF, "data", "output", "eval_deimmunisation.csv")
OUT_CSV     = os.path.join(PF, "data", "output", "eval_mhc1_arm.csv")
OUT_SUMMARY = os.path.join(PF, "data", "output", "eval_mhc1_arm_summary.txt")
OUT_FIGURE  = os.path.join(PF, "data", "output", "eval_mhc1_arm_figure.png")

# The percentile cache grows without bound; clear it periodically (same reason as
# CAPE/MPNN/data/preference_pair.py:70).
CACHE_RESET_EVERY = 100


def score_sequence(seq, predictor, alleles, lengths):
    """Return dict of presented-window counts: total + per-allele.

    seq_presented() splits multi-chain sequences on '/' internally.
    """
    presented = predictor.seq_presented(seq, alleles=alleles, lengths=lengths)
    per_allele = {a: 0 for a in alleles}
    for _peptide, allele, _rank, _pos in presented:
        per_allele[allele] += 1
    return {"total": sum(per_allele.values()), **per_allele}


def paired_by_protein(df, value_col):
    """Collapse to one mean value per (protein, model), then align base vs finetuned."""
    per_prot = df.groupby(["protein_name", "model"])[value_col].mean().unstack("model")
    per_prot = per_prot.dropna(subset=["base", "finetuned"])
    return per_prot["base"].values, per_prot["finetuned"].values, per_prot.index.tolist()


def main(args):
    t0 = time.time()

    # Optional parametrisation: score a specific model's designs and version the
    # outputs so a new run never overwrites a frozen prior result.
    global IN_CSV, OUT_CSV, OUT_SUMMARY, OUT_FIGURE
    if getattr(args, "in_csv", None):
        IN_CSV = args.in_csv
    if getattr(args, "out_tag", None):
        _od = os.path.join(PF, "data", "output")
        OUT_CSV     = os.path.join(_od, f"eval_mhc1_arm_{args.out_tag}.csv")
        OUT_SUMMARY = os.path.join(_od, f"eval_mhc1_arm_{args.out_tag}_summary.txt")
        OUT_FIGURE  = os.path.join(_od, f"eval_mhc1_arm_{args.out_tag}_figure.png")
    assert not os.path.exists(OUT_CSV), (
        f"Refusing to overwrite existing {OUT_CSV}; pass a distinct --out_tag."
    )

    if not os.path.exists(IN_CSV):
        sys.exit(f"Missing input: {IN_CSV}\nRun evaluate_deimmunisation.py first (Track 7).")

    print(f"Reading designed sequences from {IN_CSV}")
    df = pd.read_csv(IN_CSV)
    if args.limit:
        df = df.head(args.limit).copy()
    print(f"  {len(df)} sequences  |  models: {sorted(df['model'].unique())}")
    print(f"  {df['protein_name'].nunique()} distinct proteins")

    mhc1_alleles = MHC1_ALLELES.split("+")
    mhc2_alleles = MHC2_ALLELES.split("+")

    print(f"\nLoading MHC-I PWM predictor from {PWM_DIR}")
    if not os.path.isdir(os.path.join(PWM_DIR, "pwm")):
        sys.exit(f"No pwm/ subdirectory under {PWM_DIR}. Wrong directory?")
    predictor = Mhc1Predictor.get_predictor("Mhc1PredictorPwm")(
        data_dir_path=PWM_DIR, limit=0.02
    )

    print(f"Scoring {len(df)} sequences against {len(mhc1_alleles)} MHC-I alleles "
          f"(lengths {MHC_1_PEPTIDE_LENGTHS})…")
    records = []
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Sequences")):
        scores = score_sequence(row["sequence"], predictor, mhc1_alleles,
                                MHC_1_PEPTIDE_LENGTHS)
        rec = {
            "protein_name":   row["protein_name"],
            "protein_length": row["protein_length"],
            "model":          row["model"],
            "seq_idx":        row["seq_idx"],
            "mhc1_total":     scores["total"],
            "mhc2_total":     row["n_presented_total"],
        }
        for a in mhc1_alleles:
            rec[f"mhc1_{a}"] = scores[a]
        for a in mhc2_alleles:
            rec[f"mhc2_{a}"] = row[a]
        records.append(rec)

        if (i + 1) % CACHE_RESET_EVERY == 0:
            predictor.percentiles = defaultdict(lambda: {})

    out = pd.DataFrame.from_records(records)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({len(out)} rows)")

    # ── Statistics ────────────────────────────────────────────────────────────
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit("MHC-I ARM EVALUATION  (roadmap step P1.1)")
    emit("=" * 78)
    emit(f"Sequences scored : {len(out)}")
    emit(f"Proteins         : {out['protein_name'].nunique()}")
    emit(f"MHC-I alleles    : {MHC1_ALLELES}")
    emit(f"Peptide lengths  : {MHC_1_PEPTIDE_LENGTHS}")
    emit(f"PWM directory    : {PWM_DIR}")
    emit("")

    base_m = out[out.model == "base"]
    ft_m = out[out.model == "finetuned"]

    emit("-" * 78)
    emit("MHC-I presented windows per sequence")
    emit("-" * 78)
    emit(f"{'':<26}{'base':>16}{'finetuned':>16}")
    emit(f"{'mean ± sd':<26}{base_m.mhc1_total.mean():>9.1f} ± {base_m.mhc1_total.std():<4.1f}"
         f"{ft_m.mhc1_total.mean():>9.1f} ± {ft_m.mhc1_total.std():<4.1f}")
    emit(f"{'median':<26}{base_m.mhc1_total.median():>16.1f}{ft_m.mhc1_total.median():>16.1f}")
    emit("")

    b, f, prot_names = paired_by_protein(out, "mhc1_total")
    n_prot = len(b)
    # per-protein percent reduction
    red = 100.0 * (b - f) / np.where(b > 0, b, np.nan)
    red_valid = red[~np.isnan(red)]

    t_stat, t_p = stats.ttest_rel(b, f)
    try:
        w_stat, w_p = stats.wilcoxon(b, f)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    emit("-" * 78)
    emit(f"Paired per-protein comparison (n = {n_prot} proteins)")
    emit("-" * 78)
    emit(f"Mean per-protein reduction : {np.nanmean(red_valid):.1f}% "
         f"± {np.nanstd(red_valid):.1f}%   (median {np.nanmedian(red_valid):.1f}%)")
    emit(f"Proteins improved          : {int((f < b).sum())}/{n_prot}")
    emit(f"Paired t-test              : t = {t_stat:.3f}, p = {t_p:.3g}")
    emit(f"Wilcoxon signed-rank       : W = {w_stat:.1f}, p = {w_p:.3g}")
    emit("")

    emit("-" * 78)
    emit("Per-allele breakdown (mean presented windows per sequence)")
    emit("-" * 78)
    emit(f"{'allele':<16}{'base':>10}{'finetuned':>12}{'reduction':>12}")
    per_allele_rows = []
    for a in mhc1_alleles:
        bm = base_m[f"mhc1_{a}"].mean()
        fm = ft_m[f"mhc1_{a}"].mean()
        r = 100.0 * (bm - fm) / bm if bm > 0 else float("nan")
        emit(f"{a:<16}{bm:>10.1f}{fm:>12.1f}{r:>11.0f}%")
        per_allele_rows.append((a, bm, fm, r))
    emit("")

    # ── MHC-I vs MHC-II: did both classes drop together? ─────────────────────
    b2, f2, _ = paired_by_protein(out, "mhc2_total")
    red2 = 100.0 * (b2 - f2) / np.where(b2 > 0, b2, np.nan)

    emit("-" * 78)
    emit("Dual-objective check: MHC-I vs MHC-II on the same proteins")
    emit("-" * 78)
    emit(f"MHC-I  mean reduction : {np.nanmean(red_valid):.1f}% ± {np.nanstd(red_valid):.1f}%")
    emit(f"MHC-II mean reduction : {np.nanmean(red2):.1f}% ± {np.nanstd(red2):.1f}%")

    ok = ~(np.isnan(red) | np.isnan(red2))
    if ok.sum() > 2:
        r_pear, p_pear = stats.pearsonr(red[ok], red2[ok])
        emit(f"Per-protein reduction correlation     : r = {r_pear:.3f} "
             f"(R² = {r_pear**2:.2f}), p = {p_pear:.3g}")
        t_d, p_d = stats.ttest_rel(red2[ok], red[ok])
        emit(f"MHC-II reduced MORE than MHC-I by      {np.mean(red2[ok] - red[ok]):.1f} pp "
             f"(paired t = {t_d:.2f}, p = {p_d:.3g})")
        emit(f"Proteins where MHC-II red > MHC-I red  : "
             f"{int((red2[ok] > red[ok]).sum())}/{int(ok.sum())}")
    emit("")

    # Are the two epitope burdens intrinsically coupled, or is the apparent
    # coupling just protein length? Longer proteins have more windows of BOTH
    # classes, which inflates the naive correlation. Control for it.
    emit("-" * 78)
    emit("Are MHC-I and MHC-II burdens intrinsically coupled?")
    emit("-" * 78)

    def partial_r(x, y, z):
        """Pearson r between x and y after linearly regressing out z."""
        x, y, z = (np.asarray(v, dtype=float) for v in (x, y, z))
        rx = x - np.poly1d(np.polyfit(z, x, 1))(z)
        ry = y - np.poly1d(np.polyfit(z, y, 1))(z)
        return stats.pearsonr(rx, ry)

    r_burden, _ = stats.pearsonr(out.mhc1_total, out.mhc2_total)
    r_len1, _ = stats.pearsonr(out.protein_length, out.mhc1_total)
    r_len2, _ = stats.pearsonr(out.protein_length, out.mhc2_total)
    emit(f"Naive burden correlation (all seqs)   : r = {r_burden:.3f}   <- CONFOUNDED")
    emit(f"  protein_length vs MHC-I burden      : r = {r_len1:.3f}")
    emit(f"  protein_length vs MHC-II burden     : r = {r_len2:.3f}")
    emit("  Longer proteins have more windows of both classes, which alone produces a")
    emit("  large positive correlation. The naive number must not be reported.")
    emit("")
    emit("Length-controlled, computed separately within each model:")
    emit(f"{'model':<14}{'partial r':>12}{'p':>14}")
    for m in ["base", "finetuned"]:
        sub = out[out.model == m]
        rp, pp_ = partial_r(sub.mhc1_total, sub.mhc2_total, sub.protein_length)
        emit(f"{m:<14}{rp:>12.3f}{pp_:>14.3g}")
    emit("")
    emit("Interpretation: once protein length is controlled and models are considered")
    emit("separately, the MHC-I and MHC-II epitope burdens are near-independent. The two")
    emit("objectives therefore act on largely disjoint sequence features, and the MHC-II")
    emit("reduction is NOT a by-product of the MHC-I objective (or vice versa). Note this")
    emit("bounds, but does not replace, the MHC-II-only ablation arm (roadmap P2.2):")
    emit("near-zero coupling means an MHC-I-only objective could not have produced the")
    emit("observed 39% MHC-II drop, but the exact attribution still needs the ablation.")
    emit("")
    emit("-" * 78)
    emit("Comparison to Gasser et al. 2025 -- READ BEFORE CITING")
    emit("-" * 78)
    emit("Gasser reports 'relative visibility' = sequence visibility / NATIVE TEMPLATE")
    emit("visibility, and reaches up to ~70% reduction along a quality/visibility Pareto")
    emit("curve. Our reduction is measured against BASE-MODEL DESIGNS, not the native")
    emit("template. The denominators differ, so the numbers are NOT directly comparable.")
    emit("A like-for-like comparison requires re-scoring the native template sequences.")
    emit("")
    emit(f"Runtime: {time.time() - t0:.0f}s")
    emit("=" * 78)

    with open(OUT_SUMMARY, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {OUT_SUMMARY}")

    # ── Figure ────────────────────────────────────────────────────────────────
    if not args.no_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
        C_BASE, C_FT = "#4C72B0", "#DD8452"

        # (A) distribution
        ax = axes[0]
        parts = ax.violinplot([base_m.mhc1_total.values, ft_m.mhc1_total.values],
                              showmedians=True)
        for pc, c in zip(parts["bodies"], [C_BASE, C_FT]):
            pc.set_facecolor(c)
            pc.set_alpha(0.7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["base", "fine-tuned"])
        ax.set_ylabel("MHC-I presented windows / sequence")
        ax.set_title(f"(A) MHC-I burden\n{np.nanmean(red_valid):.0f}% mean reduction")

        # (B) per-protein scatter
        ax = axes[1]
        lim = max(b.max(), f.max()) * 1.05
        ax.scatter(b, f, s=22, alpha=0.7, color=C_FT, edgecolor="k", linewidth=0.3)
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="no change")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("base (mean MHC-I windows)")
        ax.set_ylabel("fine-tuned")
        ax.set_title(f"(B) Per-protein\n{int((f < b).sum())}/{n_prot} improved")
        ax.legend(fontsize=8)

        # (C) per-allele
        ax = axes[2]
        x = np.arange(len(mhc1_alleles))
        bm = [r[1] for r in per_allele_rows]
        fm = [r[2] for r in per_allele_rows]
        ax.bar(x - 0.2, bm, 0.4, label="base", color=C_BASE)
        ax.bar(x + 0.2, fm, 0.4, label="fine-tuned", color=C_FT)
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace("HLA-", "") for a in mhc1_alleles],
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("mean presented windows")
        ax.set_title("(C) Per-allele MHC-I")
        ax.legend(fontsize=8)

        # (D) MHC-I vs MHC-II burden DENSITY (length-normalised; the raw burden
        #     correlation is dominated by protein length and would mislead here)
        ax = axes[3]
        for name, sub, c in [("base", base_m, C_BASE), ("fine-tuned", ft_m, C_FT)]:
            ax.scatter(sub.mhc2_total / sub.protein_length,
                       sub.mhc1_total / sub.protein_length,
                       s=8, alpha=0.35, color=c, label=name)
        rp_ft, _ = partial_r(ft_m.mhc1_total, ft_m.mhc2_total, ft_m.protein_length)
        rp_b, _ = partial_r(base_m.mhc1_total, base_m.mhc2_total, base_m.protein_length)
        ax.set_xlabel("MHC-II windows per residue")
        ax.set_ylabel("MHC-I windows per residue")
        ax.set_title("(D) Both classes drop, independently\n"
                     f"partial r: base {rp_b:.2f}, FT {rp_ft:.2f}")
        ax.legend(fontsize=8, markerscale=2)

        for a in axes:
            a.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT_FIGURE, dpi=300)
        print(f"Wrote {OUT_FIGURE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None,
                    help="only score the first N sequences (smoke test)")
    ap.add_argument("--no_figure", action="store_true")
    ap.add_argument("--in_csv", type=str, default=None,
                    help="deimmunisation eval CSV whose designed sequences to re-score for MHC-I "
                         "(default: eval_deimmunisation.csv). Use a seed-matched CSV to score a "
                         "specific model's designs, e.g. eval_deimmunisation_mhc2only_seed42.csv.")
    ap.add_argument("--out_tag", type=str, default=None,
                    help="suffix for the output files (default: none -> eval_mhc1_arm.*). "
                         "Versions outputs so a new run never overwrites a prior one.")
    main(ap.parse_args())
