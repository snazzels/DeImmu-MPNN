#!/usr/bin/env python
"""P1-2 / DA-M5 -- filtering-vs-fine-tuning as a function of ensemble size.

THE CHALLENGE THIS ANSWERS. The manuscript compares fine-tuning against a
post-hoc filtering baseline frozen at "keep the best 3 of 10" base-model designs.
Filtering improves monotonically with ensemble size, so "fine-tuning beats
filtering" as stated is a claim about one arbitrary budget. The Devil's Advocate
seat asked for the reduction-vs-ensemble-size curve with the crossing point
marked. That is what this produces.

WHY NO RE-SCORING IS NEEDED. The per-design netMHCIIpan CSV already holds every
number required: for each protein, 10 base designs and 10 fine-tuned designs, each
with its own presented-window total. Filtering at budget k is then a pure
re-analysis -- draw k of the 10 base designs and keep the best -- and the
fine-tuned arm is unchanged. Re-scoring would introduce sampling noise for nothing.

THE ESTIMATOR, AND WHY IT IS NOT "SORT AND TAKE THE FIRST k". For a budget of k
designs the quantity wanted is the EXPECTED burden of the best of k draws, not the
burden of the best design in one particular subset of size k. With n=10 designs
per protein the expectation over all C(n,k) subsets has an exact closed form: if
the designs are ranked worst-to-best as x_(1) >= ... >= x_(n), then

    E[min of k draws] = sum_i x_(i) * C(i-1, k-1) / C(n, k)

i.e. design i is the minimum of its subset exactly when the other k-1 members are
drawn from the i-1 designs worse than it. This is exact, deterministic, and needs
no bootstrap. It is also why the curve is smooth rather than jagged.

TIES. Ranking uses a stable sort, so tied burdens are resolved by design index.
Ties only occur between designs with identical integer window counts, where the
choice is immaterial to the estimate.

WHAT "CROSSING POINT" MEANS HERE. The reported crossing is the smallest k at
which filtering's mean per-protein reduction reaches the fine-tuned model's. Two
honest caveats are printed with it: (a) the fine-tuned arm is itself an ensemble
of 10, so the fair like-for-like comparison at budget k is filtering-at-k against
FINE-TUNING-AT-1 (one forward pass, no selection), which is also tabulated; and
(b) filtering at budget k costs k netMHCIIpan calls per protein at design time,
which the fine-tuned model does not, so equal reduction is not equal cost.

GUARDS. The minimum-base guard of the manuscript is applied on the per-protein
mean base burden over ALL n designs (never on a filtered subset -- filtering
changes the numerator, and letting it change the denominator too would silently
compare different protein sets across k). Proteins are pinned to the intersection
of both arms. The script refuses to overwrite an existing output.
"""

import argparse
import os
import sys
from math import comb

import numpy as np
import pandas as pd

MIN_BASE = 1.0


def expected_best_of_k(values, k, keep=1):
    """Exact E[ mean of the `keep` smallest of a uniformly drawn k-subset ].

    Sort ascending as b_(1) <= ... <= b_(n). The design at ascending rank r is both
    drawn and among the `keep` smallest of its subset exactly when it is drawn and
    fewer than `keep` of the other k-1 members are smaller than it:

        P(r) = sum_{j=0}^{keep-1} C(r-1, j) * C(n-r, k-1-j) / C(n, k)

    where j counts drawn designs better than r. E[sum of the keep smallest] is then
    sum_r b_(r) P(r), and the mean divides by `keep`. keep=1 reduces to E[min].

    Ties are handled by the ascending sort, which fixes an arbitrary but consistent
    order among equal values; the expectation is unaffected because equal values
    contribute equally whichever way they are ordered.
    """
    n = len(values)
    if k > n:
        raise ValueError(f"budget k={k} exceeds ensemble size n={n}")
    if keep > k:
        raise ValueError(f"keep={keep} exceeds budget k={k}")
    b = np.sort(np.asarray(values, dtype=float))
    denom = comb(n, k)
    total = 0.0
    for r in range(1, n + 1):
        p = sum(comb(r - 1, j) * comb(n - r, k - 1 - j) for j in range(0, keep))
        total += b[r - 1] * p
    return float(total / denom / keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="per-design netMHCIIpan CSV")
    ap.add_argument("--column", default="netmhc_total")
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--out_dir", default="CAPE_MPNN/data/output")
    ap.add_argument("--min_base", type=float, default=MIN_BASE)
    ap.add_argument("--keep", type=int, default=3,
                    help="designs kept per protein in the manuscript's baseline rule")
    ap.add_argument("--expect", type=float, default=None,
                    help="published unguarded keep-of-n reduction, as an acceptance test")
    args = ap.parse_args()

    out_csv = os.path.join(args.out_dir, f"{args.out_tag}.csv")
    out_txt = os.path.join(args.out_dir, f"{args.out_tag}_summary.txt")
    for p in (out_csv, out_txt):
        if os.path.exists(p):
            sys.exit(f"REFUSING TO OVERWRITE {p} -- use a new --out_tag.")

    d = pd.read_csv(args.csv)
    for col in ("protein_name", "model", "seq_idx", args.column):
        if col not in d.columns:
            sys.exit(f"FATAL: column '{col}' not in {args.csv}")

    base = d[d.model == "base"]
    fine = d[d.model == "finetuned"]
    if base.empty or fine.empty:
        sys.exit("FATAL: need both 'base' and 'finetuned' rows.")

    common = sorted(set(base.protein_name) & set(fine.protein_name))
    n_base_only = len(set(base.protein_name) - set(common))
    n_fine_only = len(set(fine.protein_name) - set(common))

    sizes = {len(g) for _, g in base.groupby("protein_name")}
    sizes |= {len(g) for _, g in fine.groupby("protein_name")}
    if len(sizes) != 1:
        sys.exit(f"FATAL: ragged ensembles, sizes={sorted(sizes)}. "
                 "The exact estimator assumes one ensemble size.")
    n_ens = sizes.pop()

    rows, excluded = [], []
    for prot in common:
        b = base[base.protein_name == prot].sort_values("seq_idx")[args.column].to_numpy(float)
        f = fine[fine.protein_name == prot].sort_values("seq_idx")[args.column].to_numpy(float)
        # Guard on the FULL base ensemble mean -- never on a filtered subset.
        if b.mean() <= args.min_base:
            excluded.append((prot, b.mean()))
            continue
        rec = {"protein_name": prot, "base_mean": b.mean(), "fine_mean": f.mean()}
        for k in range(1, n_ens + 1):
            rec[f"filter_k{k}"] = expected_best_of_k(b, k, keep=1)
            if k >= args.keep:
                rec[f"keep{args.keep}_k{k}"] = expected_best_of_k(b, k, keep=args.keep)
                rec[f"ft_keep{args.keep}_k{k}"] = expected_best_of_k(f, k, keep=args.keep)
        rows.append(rec)

    # Unguarded acceptance value: reproduces the manuscript's published baseline,
    # which was computed over all proteins with no minimum-base guard.
    acc = []
    for prot in common:
        b = base[base.protein_name == prot].sort_values("seq_idx")[args.column].to_numpy(float)
        if b.mean() <= 0:
            continue
        acc.append(100.0 * (b.mean() - expected_best_of_k(b, n_ens, keep=args.keep)) / b.mean())
    acc_mean = float(np.mean(acc)) if acc else float("nan")

    if not rows:
        sys.exit("FATAL: no proteins survived the guard.")
    r = pd.DataFrame(rows)

    # Per-protein percentage reductions against the unfiltered base mean.
    K = args.keep
    for k in range(1, n_ens + 1):
        r[f"red_filter_k{k}"] = 100.0 * (r.base_mean - r[f"filter_k{k}"]) / r.base_mean
        if k >= K:
            r[f"red_keep{K}_k{k}"] = 100.0 * (r.base_mean - r[f"keep{K}_k{k}"]) / r.base_mean
            r[f"red_ft_keep{K}_k{k}"] = 100.0 * (r.base_mean - r[f"ft_keep{K}_k{k}"]) / r.base_mean
    r["red_fine"] = 100.0 * (r.base_mean - r.fine_mean) / r.base_mean
    r.to_csv(out_csv, index=False)

    fine_red = r.red_fine.mean()
    fine_pooled = 100.0 * (1 - r.fine_mean.sum() / r.base_mean.sum())
    curve = [(k, r[f"red_filter_k{k}"].mean(),
              100.0 * (1 - r[f"filter_k{k}"].sum() / r.base_mean.sum()))
             for k in range(1, n_ens + 1)]

    reached = [k for k, m, _ in curve if m >= fine_red]
    crossing = reached[0] if reached else None

    L = []
    A = L.append
    A("FILTERING vs FINE-TUNING AS A FUNCTION OF ENSEMBLE SIZE (P1-2 / DA-M5)")
    A("=" * 78)
    A(f"source CSV : {os.path.basename(args.csv)}")
    A(f"column     : {args.column}")
    A(f"ensemble   : {n_ens} designs per protein per arm")
    A(f"guard      : per-protein mean base burden > {args.min_base} over ALL {n_ens} designs")
    A(f"proteins   : {len(r)} retained, {len(excluded)} excluded"
      + (f" ({', '.join(f'{p} [{v:.2f}]' for p, v in excluded)})" if excluded else ""))
    if n_base_only or n_fine_only:
        A(f"⚠ arm-only proteins dropped: base-only {n_base_only}, fine-only {n_fine_only}")
    A("")
    A("ESTIMATOR: exact expectation of the best of k designs drawn from the")
    A("ensemble of %d, E[min of k] = sum_i x_(i) C(i-1,k-1)/C(n,k). Not a bootstrap." % n_ens)
    A("")
    A("Two selection rules are tabulated because they answer different questions.")
    A(f"  best-of-k : generate k, keep the single best. The natural ensemble-size curve.")
    A(f"  keep-{K}-of-k : generate k, keep the best {K} and average them. THIS is the rule")
    A(f"            behind the manuscript's published filtering baseline.")
    A("")
    A(f"  k    best-of-k %   keep-{K}-of-k %   fine-tuned keep-{K}-of-k %   best-of-k vs FT")
    A("  " + "-" * 74)
    for k, m, pooled in curve:
        kk = r[f"red_keep{K}_k{k}"].mean() if k >= K else float("nan")
        ff = r[f"red_ft_keep{K}_k{k}"].mean() if k >= K else float("nan")
        s_kk = f"{kk:>11.2f}" if k >= K else f"{'--':>11}"
        s_ff = f"{ff:>21.2f}" if k >= K else f"{'--':>21}"
        mark = "  <--" if crossing == k else ""
        A(f"  {k:<3} {m:>11.2f}   {s_kk}   {s_ff}        {m - fine_red:+7.2f} pp{mark}")
    A("")
    if args.expect is not None:
        delta = acc_mean - args.expect
        ok = abs(delta) <= 1.0
        A(f"ACCEPTANCE TEST (unguarded, all {len(acc)} proteins, keep-{K}-of-{n_ens}):")
        A(f"  recomputed {acc_mean:.2f}%  vs published {args.expect:.2f}%  (delta {delta:+.2f} pp)"
          + ("  PASS" if ok else "  ** FAIL **"))
        if not ok:
            A("  The selection rule or the protein set does not match the published run.")
            A("  Do NOT quote the curve above against the published baseline until this passes.")
        A("")
    else:
        A(f"(unguarded keep-{K}-of-{n_ens} over all {len(acc)} proteins: {acc_mean:.2f}% "
          f"-- pass --expect to assert this against the published baseline)")
        A("")
    A(f"  fine-tuned (ensemble of {n_ens}) : mean {fine_red:.2f}%   pooled {fine_pooled:.2f}%")
    A(f"  fine-tuned, ONE design, no selection : mean {fine_red:.2f}%  (a single draw's")
    A( "      expected burden is the ensemble mean, so no netMHCIIpan call is involved)")
    A("")
    if crossing is None:
        A(f"RESULT: filtering does NOT reach the fine-tuned reduction at any budget up to")
        A(f"        k={n_ens}. The largest budget tested reaches {curve[-1][1]:.2f}% against")
        A(f"        {fine_red:.2f}%, a shortfall of {fine_red - curve[-1][1]:.2f} pp.")
        A("        The curve is monotone, so a crossing at some larger k is not excluded;")
        A("        what is excluded is a crossing within the budget range tested.")
    else:
        A(f"RESULT: filtering reaches the fine-tuned reduction at k={crossing}.")
    A("")
    A("READ WITH TWO CAVEATS, both of which favour fine-tuning and neither of which")
    A("is captured by the crossing point alone:")
    A("  (a) COST IS NOT EQUAL AT EQUAL REDUCTION. Filtering at budget k requires k")
    A("      netMHCIIpan calls per protein at design time. The fine-tuned model")
    A("      requires none -- the objective is already in the weights.")
    A("  (b) THE ARMS ARE NOT LIKE FOR LIKE. The fine-tuned figure above is its own")
    A("      %d-design ensemble mean; a single fine-tuned design (no selection at all)" % n_ens)
    A("      is the row marked k=1 in the fine-tuned line. The two strategies also")
    A("      compose -- filtering the FINE-TUNED ensemble is reported separately in")
    A("      the manuscript and exceeds either alone.")
    A("=" * 78)

    with open(out_txt, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[out] {out_csv}\n[out] {out_txt}")


if __name__ == "__main__":
    main()
