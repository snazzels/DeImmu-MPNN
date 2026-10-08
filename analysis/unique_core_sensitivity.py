#!/usr/bin/env python
"""P1-1 / R2-W4 -- is the reduction an artefact of counting overlapping windows?

THE CHALLENGE THIS ANSWERS. The manuscript's burden metric counts presented 15-mer
windows. Consecutive windows that share one binding core are counted separately, so
a single epitope contributes up to seven times. The manuscript calls this
"conservative" and asserts it without support. R2 asked for a sensitivity row on
unique (allele, binding-core) pairs instead.

WHAT R2 EXPLICITLY DID *NOT* CLAIM, and this must not be garbled in the write-up:
the paired tests use per-protein means with n = proteins, so window correlation
does NOT inflate the degrees of freedom. This is a metric-interpretation question,
not a statistical-inflation one. The reduction could be perfectly well estimated
and still be describing a quantity nobody cares about.

WHY THE ANSWER IS NOT OBVIOUS EITHER WAY. If de-immunisation removes whole
epitopes, window counts and unique-core counts fall together and the metric choice
is immaterial. If instead it nibbles at the edges of epitopes -- shortening the run
of windows that pass threshold without removing the core -- then the window count
falls much faster than the unique-core count, and the headline is inflated by the
metric. The permutation control already showed the effect is compositional rather
than positional, which makes the second possibility a live one rather than a
formality.

WHAT IS COMPUTED. netMHCIIpan-4.3 reports the binding Core for every peptide. For
each design and each allele, over the windows passing %Rank_EL <= threshold:
  * n_windows      -- the published metric, reproduced exactly (acceptance test)
  * n_unique_cores -- distinct Core strings for that allele
  * n_unique_pairs -- distinct (allele, Core) pairs, summed over alleles
Reductions are then recomputed on each, under the same minimum-base guard, on the
same proteins, with the same pairing. The deflation ratio (windows per unique core)
is reported per arm, because a change in THAT ratio between base and fine-tuned is
precisely the edge-nibbling signature.

ACCEPTANCE TEST. With cores ignored, every per-design window count must reproduce
the published per-design counts from the source CSV exactly. If it does not, the
scoring conventions have drifted and no summary is written. Conventions replicated
deliberately: windows as a LIST (duplicate 15-mers count more than once),
%Rank_EL <= 2.0, and */-// stripped -- the same three that dp_orientation.py had to
match for the same reason.

Refuses to overwrite existing output.
"""

import argparse
import collections
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy import stats

PEPTIDE_LEN = 15
THRESHOLD = 2.0
MIN_BASE = 1.0
ALLELES = ["DRB1_0101", "DRB1_0301", "DRB1_0401", "DRB1_0701", "DRB1_1101", "DRB1_1501"]


def get_15mers(seq):
    """LIST, not set -- duplicates count more than once, matching the published pipeline."""
    seq = seq.replace("*", "").replace("-", "").replace("/", "")
    return [seq[i:i + PEPTIDE_LEN] for i in range(len(seq) - PEPTIDE_LEN + 1)]


def run_netmhciipan(peptides, allele):
    """{peptide: (best_rank, core_of_that_row)}.

    The core must travel with the row whose rank is kept; taking a core from an
    arbitrary row for the same peptide would silently mix registers.
    """
    out = {}
    peptides = sorted(set(peptides))
    with tempfile.NamedTemporaryFile("w", suffix=".pep", delete=False) as fh:
        fh.write("\n".join(peptides) + "\n")
        pep_file = fh.name
    try:
        res = subprocess.run(
            ["netMHCIIpan", "-inptype", "1", "-f", pep_file, "-a", allele],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
        for line in res.stdout.decode().splitlines():
            p = line.split()
            if len(p) < 10:
                continue
            try:
                int(p[0])
                rank = float(p[9])
            except ValueError:
                continue          # header, separator, or the trailing summary line
            pep, core = p[2], p[4]
            if pep not in out or rank < out[pep][0]:
                out[pep] = (rank, core)
    finally:
        os.unlink(pep_file)
    if not out:
        sys.exit(f"FATAL: netMHCIIpan returned zero parseable ranks for {allele}. On this "
                 f"cluster that means the wrapper did not run (it execs tcsh). Not a result.")
    return out


def design_counts(seq, scored):
    """(windows, unique (allele,core) pairs, per-allele unique-core counts)."""
    windows = get_15mers(seq)
    n_win = 0
    pairs = set()
    per_allele = collections.Counter()
    for al in ALLELES:
        cores = set()
        table = scored[al]
        for w in windows:
            hit = table.get(w)
            if hit is not None and hit[0] <= THRESHOLD:
                n_win += 1
                cores.add(hit[1])
        per_allele[al] = len(cores)
        pairs |= {(al, c) for c in cores}
    return n_win, len(pairs), per_allele


def guarded_stats(base, fine, label):
    """Paired per-protein reduction with the minimum-base guard. Returns dict."""
    keep = [i for i in range(len(base)) if base[i] > MIN_BASE]
    b = np.array([base[i] for i in keep], float)
    f = np.array([fine[i] for i in keep], float)
    red = 100.0 * (b - f) / b
    p_t = stats.ttest_rel(f, b).pvalue if len(b) > 1 else float("nan")
    try:
        p_w = stats.wilcoxon(f, b).pvalue
    except ValueError:
        p_w = float("nan")
    return {"label": label, "n": len(b), "mean": red.mean(), "sd": red.std(ddof=1),
            "pooled": 100.0 * (1 - f.sum() / b.sum()),
            "improved": int((f < b).sum()), "p_t": p_t, "p_w": p_w,
            "excluded": len(base) - len(keep)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="per-design CSV with sequence + published counts")
    ap.add_argument("--published_col", default="netmhc_total",
                    help="column holding the published per-design window count")
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--out_dir", default="data/output")
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
        keep = sorted(d.protein_name.unique())[:args.max_proteins]
        d = d[d.protein_name.isin(keep)]

    kmers = sorted({w for s in d.sequence for w in get_15mers(str(s))})
    print(f"[pep] {len(kmers)} unique 15-mers -> {len(ALLELES)} netMHCIIpan passes", flush=True)
    scored = {}
    for al in ALLELES:
        print(f"  netMHCIIpan {al} …", flush=True)
        scored[al] = run_netmhciipan(kmers, al)

    rows, mismatches = [], []
    for _, r in d.iterrows():
        n_win, n_pairs, per_allele = design_counts(str(r.sequence), scored)
        if n_win != int(r[args.published_col]):
            mismatches.append((r.protein_name, r.model, int(r.seq_idx),
                               int(r[args.published_col]), n_win))
        rows.append({"protein_name": r.protein_name, "model": r.model, "seq_idx": r.seq_idx,
                     "n_windows": n_win, "n_unique_pairs": n_pairs,
                     **{f"cores_{a}": per_allele[a] for a in ALLELES}})

    if mismatches:
        print("\nACCEPTANCE TEST FAILED — refusing to write a summary.", file=sys.stderr)
        print(f"{len(mismatches)} of {len(rows)} designs disagree with '{args.published_col}'.",
              file=sys.stderr)
        for m in mismatches[:10]:
            print(f"  {m[0]} {m[1]} seq{m[2]}: published {m[3]}, recomputed {m[4]}", file=sys.stderr)
        print("\nThe window count is reproduced from the same peptides and threshold, so a\n"
              "disagreement means a scoring convention has drifted (windows-as-set,\n"
              "a different %Rank cutoff, or different character stripping). The\n"
              "unique-core numbers would sit on a different footing than the published\n"
              "ones and must not be compared with them.", file=sys.stderr)
        sys.exit(1)

    per_design = pd.DataFrame(rows)
    per_design.to_csv(out_csv, index=False)

    agg = per_design.groupby(["protein_name", "model"])[["n_windows", "n_unique_pairs"]].mean()
    prots = sorted({p for p, _ in agg.index})
    prots = [p for p in prots if (p, "base") in agg.index and (p, "finetuned") in agg.index]

    res = {}
    for metric in ("n_windows", "n_unique_pairs"):
        b = [agg.loc[(p, "base"), metric] for p in prots]
        f = [agg.loc[(p, "finetuned"), metric] for p in prots]
        res[metric] = guarded_stats(b, f, metric)

    ratio_b = (sum(agg.loc[(p, "base"), "n_windows"] for p in prots)
               / max(sum(agg.loc[(p, "base"), "n_unique_pairs"] for p in prots), 1e-9))
    ratio_f = (sum(agg.loc[(p, "finetuned"), "n_windows"] for p in prots)
               / max(sum(agg.loc[(p, "finetuned"), "n_unique_pairs"] for p in prots), 1e-9))

    w, u = res["n_windows"], res["n_unique_pairs"]
    delta = w["mean"] - u["mean"]

    L, A = [], None
    A = L.append
    A("UNIQUE-BINDING-CORE SENSITIVITY (P1-1 / R2-W4)")
    A("=" * 78)
    A(f"source CSV : {os.path.basename(args.csv)}")
    A(f"scorer     : netMHCIIpan-4.3, %Rank_EL <= {THRESHOLD}, {len(ALLELES)} HLA-DRB1 alleles")
    A(f"guard      : per-protein mean base burden > {MIN_BASE}")
    A(f"unique 15-mers scored : {len(kmers)}")
    A("")
    A("ACCEPTANCE TEST PASSED: every per-design window count reproduces the published")
    A(f"'{args.published_col}' exactly, so the core-based counts sit on the same footing.")
    A("")
    A(f"  {'metric':<26}{'n':>4}{'mean %':>10}{'sd':>8}{'pooled %':>10}{'impr.':>7}{'Wilcoxon':>11}")
    A("  " + "-" * 76)
    for m in ("n_windows", "n_unique_pairs"):
        s = res[m]
        A(f"  {s['label']:<26}{s['n']:>4}{s['mean']:>10.2f}{s['sd']:>8.2f}"
          f"{s['pooled']:>10.2f}{s['improved']:>7}{s['p_w']:>11.2e}")
    A("")
    A(f"  windows per unique (allele, core) pair:  base {ratio_b:.2f}   fine-tuned {ratio_f:.2f}")
    A("")
    A("READING (fixed before the numbers were computed):")
    A("  The window metric and the unique-core metric agreeing to within ~2 pp means the")
    A("  reduction is not an artefact of overlap counting, and 'conservative' is then")
    A("  supported rather than asserted.")
    A("  The window reduction exceeding the unique-core reduction by a wide margin means")
    A("  de-immunisation is shortening runs of presented windows more than it is removing")
    A("  distinct binding cores, and the headline is inflated by the metric. In that case")
    A("  the unique-core figure, not the window figure, is the one to lead with.")
    A("")
    A(f"  OBSERVED: window {w['mean']:.2f}% vs unique-core {u['mean']:.2f}%, "
      f"a difference of {delta:+.2f} pp.")
    if abs(delta) <= 2.0:
        A("  -> The two metrics agree. The overlap-counting objection does not bite.")
    elif delta > 0:
        A("  -> The window metric is the more favourable one. Report both, and do not")
        A("     quote the window figure without the unique-core figure beside it.")
    else:
        A("  -> The unique-core metric is the MORE favourable one, so overlap counting was")
        A("     genuinely conservative, as the manuscript claimed.")
    A("=" * 78)

    with open(out_txt, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[out] {out_csv}\n[out] {out_txt}")


if __name__ == "__main__":
    main()
