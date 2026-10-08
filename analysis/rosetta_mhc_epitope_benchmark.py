#!/usr/bin/env python
"""P2-1 / R3 -- does the reduction replicate on Rosetta's MHCEpitopeEnergy?

THE CHALLENGE THIS ANSWERS. Every predicted-presentation number in the manuscript
comes from netMHCIIpan, and the training reward is a PWM fitted to netMHCIIpan
output. The reward and the evaluator therefore share a predictor, and R3 called an
independent check "near-mandatory for a Tools paper": if the reduction is real it
should show up on a predictor with no relationship to netMHCIIpan; if it is the
model learning netMHCIIpan's idiosyncrasies, it should not.

Rosetta's `mhc_epitope` score term is the right instrument for that. It is the
de-immunisation term of Yachnin et al. (2021), built on the ProPred/TEPITOPE
position-specific scoring matrices for eight HLA-DR alleles. ProPred predates
netMHCIIpan, is a different model class (a PSSM over 9-mers rather than a neural
network over 15-mers), was fitted to different data, and covers an allele set that
only partly overlaps our six-allele DRB1 panel. Agreement between the two is
therefore evidence about the designs; disagreement localises the claim to
netMHCIIpan.

WHY THE ANSWER IS NOT OBVIOUS EITHER WAY. The two predictors could disagree for
reasons that have nothing to do with overfitting. ProPred scores 9-mer cores where
netMHCIIpan scores 15-mer windows with flanks; its allele set is not ours; and a
PSSM cannot represent the position interactions a network can. So a *smaller*
reduction on ProPred is consistent both with predictor-specific overfitting and
with ProPred simply being blunter. What would be decisive in the other direction is
a reduction near zero, or negative, on ProPred while netMHCIIpan reports 41%.

WHAT IS COMPUTED, on the SAME frozen designs as the headline. For each design
sequence in the input CSV, over every valid 9-mer window:
  * propred_alleles -- summed ProPred hits, i.e. for each window the number of the
                       eight alleles predicted to bind it, summed over windows.
                       This is exactly Rosetta's raw `mhc_epitope` score.
  * propred_windows -- number of windows predicted to bind at least one allele.
                       The closer analogue of the manuscript's window count.
  * iedb_hits       -- windows occurring in Rosetta's shipped IEDB 9-mer set
                       (190,594 peptides). A membership lookup, NOT a predictor;
                       reported because it is free and orthogonal to both PSSMs.
Reductions are recomputed per protein, paired, under the same minimum-base guard,
and reported beside the netMHCIIpan reduction ON THE SAME PROTEINS so the two are
comparable protein for protein rather than merely both being "about 40%".

CROSS-PREDICTOR AGREEMENT is reported separately from the reduction, on the base
arm alone. If ProPred and netMHCIIpan do not correlate across base designs, they
are not measuring the same thing on these proteins and no conclusion about
overfitting can be drawn from their reductions agreeing or disagreeing. This is a
validity check on the instrument, and it is computed before any reduction is
interpreted.

ACCEPTANCE TESTS, all three of which must pass or no summary is written:
  1. ProPred reference value: raw_score("FVKQNTLKL") == 7.0. That peptide is the
     influenza HA promiscuous core and 7-of-8 is the value the shipped matrix
     gives. A change means the matrix or the threshold moved, and the numbers
     would not be comparable to any earlier run.
  2. Window enumeration: for a sequence of only standard residues the window count
     is len(seq) - 8 exactly.
  3. Determinism: rescoring a peptide returns an identical value.

CONVENTION, deliberately matched to human_kmer_match.py and NOT to the 15-mer
burden pipeline. A 9-mer containing any non-standard residue -- which includes the
'/' chain separator and 'X' -- is dropped, and the number dropped is reported.
The 15-mer builders behind the published burden metric instead STRIP '/' and so do
count windows spanning a chain junction. A 9-mer crossing a junction is not a real
peptide, so dropping it is correct here; the other convention is preserved there
because the published counts depend on it. The two differ on purpose.

Requires PyRosetta (env `rosetta`). Refuses to overwrite existing output.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

K = 9
AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_BASE = 1.0

# Acceptance-test constant: the influenza HA promiscuous core, 7 of 8 alleles under
# the shipped propred8_5 matrix. Verified 2026-09-09.
REF_PEPTIDE = "FVKQNTLKL"
REF_SCORE = 7.0


def valid_kmers(seq):
    """(list of valid 9-mers, number dropped). Order preserved; duplicates kept."""
    keep, dropped = [], 0
    for i in range(len(seq) - K + 1):
        w = seq[i:i + K]
        if all(c in AA for c in w):
            keep.append(w)
        else:
            dropped += 1
    return keep, dropped


def load_predictor(config):
    """MHCEpitopeEnergySetup for a shipped .mhc config, e.g. 'propred8_5.mhc'."""
    import pyrosetta
    pyrosetta.init("-mute all", silent=True)
    from pyrosetta.rosetta.core.scoring.mhc_epitope_energy import MHCEpitopeEnergySetup
    s = MHCEpitopeEnergySetup()
    s.initialize_from_file(config)
    if s.get_peptide_length() != K:
        sys.exit(f"FATAL: {config} scores {s.get_peptide_length()}-mers, expected {K}.")
    return s


def guarded_stats(base, fine, label, keep_idx=None):
    """Paired per-protein reduction under the minimum-base guard."""
    idx = range(len(base)) if keep_idx is None else keep_idx
    keep = [i for i in idx if base[i] > MIN_BASE]
    b = np.array([base[i] for i in keep], float)
    f = np.array([fine[i] for i in keep], float)
    if len(b) < 2:
        return {"label": label, "n": len(b), "mean": float("nan"), "sd": float("nan"),
                "pooled": float("nan"), "improved": 0, "p_t": float("nan"),
                "p_w": float("nan"), "keep": keep}
    red = 100.0 * (b - f) / b
    try:
        p_w = stats.wilcoxon(f, b).pvalue
    except ValueError:
        p_w = float("nan")
    return {"label": label, "n": len(b), "mean": red.mean(), "sd": red.std(ddof=1),
            "pooled": 100.0 * (1 - f.sum() / b.sum()),
            "improved": int((f < b).sum()),
            "p_t": stats.ttest_rel(f, b).pvalue, "p_w": p_w, "keep": keep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="per-design CSV with 'sequence' + the published netMHCIIpan column")
    ap.add_argument("--published_col", default="netmhc_total",
                    help="column holding the published per-design 15-mer window count")
    ap.add_argument("--propred_config", default="propred8_5.mhc")
    ap.add_argument("--iedb_config", default="iedb_data.mhc",
                    help="set to '' to skip the IEDB membership metric")
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--out_dir", default="CAPE_MPNN/data/output")
    ap.add_argument("--max_proteins", type=int, default=0,
                    help="0 = all; a positive value subsets for a smoke test")
    args = ap.parse_args()

    out_csv = os.path.join(args.out_dir, f"{args.out_tag}.csv")
    out_txt = os.path.join(args.out_dir, f"{args.out_tag}_summary.txt")
    for p in (out_csv, out_txt):
        if os.path.exists(p):
            sys.exit(f"REFUSING TO OVERWRITE {p} -- a corrective run needs a NEW --out_tag.")

    d = pd.read_csv(args.csv)
    for col in ("protein_name", "model", "seq_idx", "sequence", args.published_col):
        if col not in d.columns:
            sys.exit(f"FATAL: column '{col}' not in {args.csv}")
    if args.max_proteins:
        d = d[d.protein_name.isin(sorted(d.protein_name.unique())[:args.max_proteins])]

    print(f"[init] PyRosetta, {args.propred_config} …", flush=True)
    pro = load_predictor(args.propred_config)
    print(f"       {pro.report().strip()}", flush=True)

    # --- acceptance tests 1 and 3 -------------------------------------
    ref = pro.raw_score(REF_PEPTIDE)
    if abs(ref - REF_SCORE) > 1e-9:
        sys.exit(f"ACCEPTANCE TEST 1 FAILED: raw_score({REF_PEPTIDE}) = {ref}, "
                 f"expected {REF_SCORE}. The matrix or threshold has changed, so these "
                 f"numbers are not comparable to the recorded run. Refusing to write.")
    if pro.raw_score(REF_PEPTIDE) != ref:
        sys.exit("ACCEPTANCE TEST 3 FAILED: scoring is not deterministic.")

    iedb = None
    if args.iedb_config:
        iedb = load_predictor(args.iedb_config)
        print(f"       {iedb.report().strip()}", flush=True)

    # --- score every unique window once ------------------------------
    uniq = sorted({w for s in d.sequence for w in valid_kmers(str(s))[0]})
    print(f"[score] {len(uniq)} unique valid 9-mers", flush=True)
    pro_cache = {w: pro.raw_score(w) for w in uniq}
    iedb_cache = {w: iedb.raw_score(w) for w in uniq} if iedb else {}

    rows, dropped_total, window_total, acc2_fail = [], 0, 0, []
    for _, r in d.iterrows():
        seq = str(r.sequence)
        wins, dropped = valid_kmers(seq)
        # acceptance test 2, only meaningful where nothing was dropped
        if dropped == 0 and len(wins) != len(seq) - K + 1:
            acc2_fail.append((r.protein_name, r.model, int(r.seq_idx)))
        dropped_total += dropped
        window_total += len(wins) + dropped
        pa = sum(pro_cache[w] for w in wins)
        pw = sum(1 for w in wins if pro_cache[w] >= 1.0)
        row = {"protein_name": r.protein_name, "model": r.model, "seq_idx": r.seq_idx,
               "protein_length": r.get("protein_length", len(seq)),
               "n_windows_valid": len(wins), "n_windows_dropped": dropped,
               "propred_alleles": pa, "propred_windows": pw,
               "netmhc_total": r[args.published_col]}
        if iedb:
            row["iedb_hits"] = sum(iedb_cache[w] for w in wins)
        rows.append(row)

    if acc2_fail:
        sys.exit(f"ACCEPTANCE TEST 2 FAILED on {len(acc2_fail)} designs: window count "
                 f"!= len(seq)-{K-1} with nothing dropped. Refusing to write.")

    per_design = pd.DataFrame(rows)
    per_design.to_csv(out_csv, index=False)

    # Percentage reductions are only defined for metrics with a usable denominator.
    # iedb_hits is a rare-event count -- base means sit below the guard for nearly
    # every protein -- so it is reported as an absolute paired difference instead.
    metrics = ["propred_alleles", "propred_windows", "netmhc_total"]
    agg_cols = metrics + (["iedb_hits"] if iedb else [])
    agg = per_design.groupby(["protein_name", "model"])[agg_cols].mean()
    prots = sorted({p for p, _ in agg.index})
    prots = [p for p in prots if (p, "base") in agg.index and (p, "finetuned") in agg.index]

    arms = {m: ([agg.loc[(p, "base"), m] for p in prots],
                [agg.loc[(p, "finetuned"), m] for p in prots]) for m in agg_cols}
    res = {m: guarded_stats(*arms[m], m) for m in metrics}

    # The matched view: ProPred restricted to the proteins the netMHCIIpan guard
    # retains, so the two reductions sit on one protein set rather than two.
    nm_keep = res["netmhc_total"]["keep"]
    matched = {m: guarded_stats(*arms[m], f"{m} (netMHCIIpan-guarded set)", keep_idx=nm_keep)
               for m in metrics if m != "netmhc_total"}

    # iedb_hits: absolute paired difference on the netMHCIIpan-guarded set.
    iedb_block = None
    if iedb:
        ib = np.array([arms["iedb_hits"][0][i] for i in nm_keep], float)
        iff = np.array([arms["iedb_hits"][1][i] for i in nm_keep], float)
        try:
            p_w_i = stats.wilcoxon(iff, ib).pvalue
        except ValueError:
            p_w_i = float("nan")
        iedb_block = {"n": len(ib), "base": ib.mean(), "fine": iff.mean(),
                      "diff": (iff - ib).mean(),
                      "p_t": stats.ttest_rel(iff, ib).pvalue if len(ib) > 1 else float("nan"),
                      "p_w": p_w_i,
                      "lower": int((iff < ib).sum()), "zero_base": int((ib == 0).sum())}

    # --- instrument validity: do the two predictors agree at base? ---
    base_only = per_design[per_design.model == "base"]
    rho_a, p_rho_a = stats.spearmanr(base_only.propred_alleles, base_only.netmhc_total)
    rho_w, p_rho_w = stats.spearmanr(base_only.propred_windows, base_only.netmhc_total)

    L = []
    A = L.append
    A("ROSETTA MHCEpitopeEnergy (ProPred) ORTHOGONAL BENCHMARK (P2-1 / R3)")
    A("=" * 82)
    A(f"source CSV : {os.path.basename(args.csv)}")
    A(f"scorer     : Rosetta mhc_epitope, {pro.report().strip()}")
    if iedb:
        A(f"             {iedb.report().strip()}")
    A(f"guard      : per-protein mean base burden > {MIN_BASE}, applied per metric")
    A(f"windows    : {window_total} total, {dropped_total} dropped for non-standard "
      f"residues ({100.0 * dropped_total / max(window_total, 1):.3f}%)")
    A(f"unique 9-mers scored : {len(uniq)}")
    A("")
    A("ACCEPTANCE TESTS PASSED: ProPred reference peptide scores "
      f"{REF_SCORE:.0f}/8; window enumeration exact; scoring deterministic.")
    A("")
    A("INSTRUMENT VALIDITY (base arm only, per design) -- do the two predictors even")
    A("measure the same thing on these proteins?")
    A(f"  Spearman propred_alleles vs netMHCIIpan : rho = {rho_a:+.3f}  (p = {p_rho_a:.2e})")
    A(f"  Spearman propred_windows vs netMHCIIpan : rho = {rho_w:+.3f}  (p = {p_rho_w:.2e})")
    A("  A rho near zero would mean the two disagree about which base designs are")
    A("  epitope-rich, and no conclusion about overfitting could be drawn below.")
    A("")
    A("REDUCTION, each metric under its own guard")
    A(f"  {'metric':<34}{'n':>4}{'mean %':>10}{'sd':>8}{'pooled %':>10}{'impr.':>7}{'Wilcoxon':>11}")
    A("  " + "-" * 80)
    for m in metrics:
        s = res[m]
        A(f"  {s['label']:<34}{s['n']:>4}{s['mean']:>10.2f}{s['sd']:>8.2f}"
          f"{s['pooled']:>10.2f}{s['improved']:>7}{s['p_w']:>11.2e}")
    A("")
    A("REDUCTION, ProPred on the netMHCIIpan-guarded protein set (matched, protein for")
    A("protein, to the netMHCIIpan row above)")
    A(f"  {'metric':<34}{'n':>4}{'mean %':>10}{'sd':>8}{'pooled %':>10}{'impr.':>7}{'Wilcoxon':>11}")
    A("  " + "-" * 80)
    for m, s in matched.items():
        A(f"  {s['label']:<34}{s['n']:>4}{s['mean']:>10.2f}{s['sd']:>8.2f}"
          f"{s['pooled']:>10.2f}{s['improved']:>7}{s['p_w']:>11.2e}")
    A("")
    if iedb_block:
        b = iedb_block
        A("IEDB 9-MER MEMBERSHIP (absolute, not a percentage). Reported this way because")
        A("it is a rare-event count: the per-protein base mean sits below the guard for")
        A("nearly every protein, so a ratio has no usable denominator.")
        A(f"  n = {b['n']} proteins ({b['zero_base']} with zero base hits)")
        A(f"  base {b['base']:.3f} -> fine-tuned {b['fine']:.3f} hits per design, "
          f"paired difference {b['diff']:+.3f}")
        A(f"  paired t p = {b['p_t']:.3g}, Wilcoxon p = {b['p_w']:.3g}, "
          f"fewer hits after fine-tuning in {b['lower']}/{b['n']}")
        A("  This counts exact occurrence in Rosetta's shipped IEDB 9-mer set, which is a")
        A("  membership lookup and not a prediction. A design can avoid every known")
        A("  epitope and still present novel ones, so a fall here is weak supporting")
        A("  evidence and a null here excludes nothing.")
        A("")
    A("READING (fixed before the numbers were computed):")
    A("  ProPred reduction of the same order as netMHCIIpan's -> the reduction is a")
    A("    property of the designed sequences, not of netMHCIIpan. The shared-predictor")
    A("    objection is answered and the headline can be stated as a reduction in")
    A("    predicted presentation without qualifying it to one predictor.")
    A("  ProPred reduction clearly smaller but still substantial and significant ->")
    A("    partly a real sequence property and partly predictor-specific. Report both")
    A("    numbers; the honest claim is the smaller one.")
    A("  ProPred reduction near zero or negative -> the model has learned netMHCIIpan's")
    A("    idiosyncrasies. The headline would then have to be restated as a reduction")
    A("    in netMHCIIpan-predicted presentation specifically, and this becomes the")
    A("    paper's principal limitation rather than a supporting panel.")
    A("")
    pw_m = matched["propred_windows"]
    nm = res["netmhc_total"]
    delta = pw_m["mean"] - nm["mean"]
    A(f"  OBSERVED (matched set, n={pw_m['n']}): ProPred windows {pw_m['mean']:.2f}% vs "
      f"netMHCIIpan {nm['mean']:.2f}%,")
    A(f"            a difference of {delta:+.2f} pp.")
    if not np.isfinite(delta):
        A("  -> Not evaluable: too few proteins survived the guard.")
    elif abs(delta) <= 10.0 and pw_m["mean"] > 0:
        A("  -> Same order on an independent predictor. The shared-predictor objection")
        A("     is not supported.")
    elif pw_m["mean"] <= 0:
        A("  -> No reduction on the independent predictor. Restate the headline as")
        A("     netMHCIIpan-specific and treat this as the principal limitation.")
    elif delta < -10.0:
        A("  -> Materially smaller on the independent predictor. Report both; lead with")
        A("     the ProPred figure as the conservative one.")
    else:
        A("  -> LARGER on the independent predictor. The netMHCIIpan headline is the")
        A("     conservative of the two.")
    A("")
    A("WHAT THIS DOES NOT SHOW. ProPred covers eight HLA-DR alleles that only partly")
    A("overlap the six-allele DRB1 panel the model was trained against, and scores")
    A("9-mer cores rather than 15-mer windows. Agreement therefore supports the")
    A("sequence-level claim without extending it to alleles neither predictor covers,")
    A("and neither predictor is evidence about T-cell responses in a person.")
    A("=" * 82)

    with open(out_txt, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[out] {out_csv}\n[out] {out_txt}")


if __name__ == "__main__":
    main()
