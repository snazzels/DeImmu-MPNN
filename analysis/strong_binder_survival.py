#!/usr/bin/env python
"""P1-10 / DA-m4 -- does the reduction reach the STRONG binders, or only the weak ones?

THE CHALLENGE THIS ANSWERS. The paper reports aggregate presented-window counts at
%Rank_EL <= 2. The Devil's Advocate seat pointed out that the ADA argument the paper
invokes is about immunodominance, not total burden: a response is typically driven by
a few strong epitopes. A method that shaved many marginal binders while leaving the
strongest ones intact would move the headline number without moving the risk. The
unexamined premise behind the whole paper is that total predicted presented-window
count is monotone in ADA risk, and this is the cheapest available probe of it.

WHAT IS COMPUTED, AND FROM WHAT. The %Rank sensitivity run retained predicted ranks
and recounted presented windows at cutoffs of 1, 2, 5 and 10%. netMHCIIpan's own
strong-binder threshold for class II is 1%, so the 1% row IS the strong-binder count.
This tool re-reads that per-protein table and asks three questions the published
summary does not:
  (1) Is the reduction at the strong threshold as large as at the reported one?
  (2) Is it significant on its own, paired per protein?
  (3) How many proteins are left with no predicted strong binder at all?

WHAT IT CANNOT ANSWER WITHOUT A NEW SCORING PASS, stated so nobody assumes otherwise:
the change in each protein's SINGLE strongest binder (min %Rank_EL), and per-DESIGN
rather than per-protein counts of zero remaining strong binders. Both need the
per-window ranks, which the published tables aggregate away. This tool deliberately
does not estimate them.

DIRECTION OF THE READING, fixed in advance. A strong-binder reduction that matches
or exceeds the 2% reduction supports the premise that the aggregate metric tracks
what matters. A strong-binder reduction materially SMALLER than the 2% reduction
means the headline is carried by marginal binders, and the immunodominance objection
stands.

Refuses to overwrite existing output.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

MIN_BASE = 1.0
STRONG = 1.0
REPORTED = 2.0


def paired(sub, label):
    b = sub.base_mean.to_numpy(float)
    f = sub.finetuned_mean.to_numpy(float)
    keep = b > MIN_BASE
    b, f = b[keep], f[keep]
    red = 100.0 * (b - f) / b
    try:
        p_w = stats.wilcoxon(f, b).pvalue
    except ValueError:
        p_w = float("nan")
    return {"label": label, "n": len(b), "excluded": int((~keep).sum()),
            "base": b.mean(), "fine": f.mean(), "mean": red.mean(),
            "sd": red.std(ddof=1) if len(red) > 1 else float("nan"),
            "pooled": 100.0 * (1 - f.sum() / b.sum()),
            "improved": int((f < b).sum()),
            "p_t": stats.ttest_rel(f, b).pvalue, "p_w": p_w,
            "zero_base": int((b == 0).sum()), "zero_fine": int((f == 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="rank-sensitivity per-protein CSV")
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--out_dir", default="CAPE_MPNN/data/output")
    args = ap.parse_args()

    out_txt = os.path.join(args.out_dir, f"{args.out_tag}_summary.txt")
    out_csv = os.path.join(args.out_dir, f"{args.out_tag}.csv")
    for p in (out_csv, out_txt):
        if os.path.exists(p):
            sys.exit(f"REFUSING TO OVERWRITE {p} -- use a new --out_tag.")

    d = pd.read_csv(args.csv)
    for col in ("cutoff", "protein_name", "base_mean", "finetuned_mean"):
        if col not in d.columns:
            sys.exit(f"FATAL: column '{col}' not in {args.csv}")
    cutoffs = sorted(d.cutoff.unique())
    for need in (STRONG, REPORTED):
        if need not in cutoffs:
            sys.exit(f"FATAL: cutoff {need} absent; found {cutoffs}. "
                     "This tool needs both the strong and the reported threshold.")

    # The protein set must be identical across cutoffs, or the rows are not comparable.
    sets = {c: set(d[d.cutoff == c].protein_name) for c in cutoffs}
    if len({frozenset(s) for s in sets.values()}) != 1:
        sys.exit("FATAL: the cutoffs do not share one protein set. Rows are not comparable.")

    res = [paired(d[d.cutoff == c].sort_values("protein_name"), f"%Rank <= {c:g}")
           for c in cutoffs]
    pd.DataFrame(res).to_csv(out_csv, index=False)

    strong = next(r for r, c in zip(res, cutoffs) if c == STRONG)
    reported = next(r for r, c in zip(res, cutoffs) if c == REPORTED)
    gap = reported["mean"] - strong["mean"]

    L, A = [], None
    A = L.append
    A("STRONG-BINDER SURVIVAL (P1-10 / DA-m4)")
    A("=" * 82)
    A(f"source CSV : {os.path.basename(args.csv)}")
    A(f"guard      : per-protein mean base burden > {MIN_BASE}, applied within each cutoff")
    A(f"note       : netMHCIIpan's class II strong-binder threshold is %Rank_EL <= 1,")
    A(f"             so the first row below is the strong-binder count, not a sensitivity row.")
    A("")
    A(f"  {'threshold':<16}{'n':>4}{'base':>9}{'fine':>9}{'mean %':>9}{'pooled %':>10}"
      f"{'impr.':>7}{'Wilcoxon':>11}{'zero ft':>9}")
    A("  " + "-" * 80)
    for r in res:
        A(f"  {r['label']:<16}{r['n']:>4}{r['base']:>9.2f}{r['fine']:>9.2f}{r['mean']:>9.2f}"
          f"{r['pooled']:>10.2f}{r['improved']:>7}{r['p_w']:>11.2e}{r['zero_fine']:>9}")
    A("")
    A("READING (fixed before the numbers were computed):")
    A("  strong-binder reduction >= the reported-threshold reduction -> the aggregate")
    A("    metric tracks the immunodominant end too, and the premise that total")
    A("    presented-window count is monotone in risk survives this probe.")
    A("  strong-binder reduction materially SMALLER -> the headline is carried by")
    A("    marginal binders and the immunodominance objection stands.")
    A("")
    A(f"  OBSERVED: strong (<=1%) {strong['mean']:.2f}% vs reported (<=2%) "
      f"{reported['mean']:.2f}%, a difference of {gap:+.2f} pp.")
    if gap <= 1.0:
        A("  -> The reduction reaches the strong binders at least as well as the marginal")
        A("     ones. The immunodominance objection is not supported by this probe.")
    elif gap <= 5.0:
        A("  -> Slightly weaker at the strong end. Report both thresholds; the difference")
        A("     is small but should not be left out.")
    else:
        A("  -> Materially weaker at the strong end. The headline is disproportionately")
        A("     marginal binders and must be reported alongside the strong-binder figure.")
    A("")
    A(f"  Proteins left with NO predicted strong binder: {strong['zero_fine']} of {strong['n']} "
      f"fine-tuned, against {strong['zero_base']} at base.")
    A("")
    A("  NOT ANSWERED HERE, and not estimated: the change in each protein's single")
    A("  strongest binder (min %Rank_EL), and per-DESIGN counts of zero remaining strong")
    A("  binders. Both need per-window ranks, which these tables aggregate away, so both")
    A("  require a further scoring pass.")
    A("=" * 82)

    with open(out_txt, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[out] {out_csv}\n[out] {out_txt}")


if __name__ == "__main__":
    main()
