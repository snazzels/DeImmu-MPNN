#!/usr/bin/env python
"""Recompute per-protein reduction statistics from a FROZEN validator CSV, applying
the minimum-base guard, without re-running netMHCIIpan / netMHCpan.

WHY THIS EXISTS. validate_with_netmhciipan.py computed the per-protein percentage
reduction as (base - ft) / (base + 1e-9) * 100 with no guard: the 1e-9 only avoided a
ZeroDivisionError. A protein whose MEAN base burden is a fraction of a window therefore
contributed an enormous negative percentage to the mean. On the deployed model's headline
run, 7m10_A had a mean base burden of 0.1 windows (one presented window across ten base
designs) and a per-protein reduction of -300%; that single protein pulled the reported
headline from 41.4% to 37.9% and inflated the sd from ~19 to 39.1.

The guard is now in validate_with_netmhciipan.py for future runs. This tool applies the
same correction to results that already exist, because re-scoring costs ~6 h of
netMHCIIpan per arm and the stored CSVs already contain every per-sequence count needed.

Never overwrites: writes <stem>_guarded_summary.txt and refuses if it exists.

Usage:
    python tools/recompute_guarded_reduction.py <validator.csv> [...] \
        [--min_base 1.0] [--exclude_assembly] [--col netmhc_total] [--no_write]
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
from scipy import stats

# base and fine-tuned are scored on different molecules for these chains, so their
# paired per-protein reduction is meaningless regardless of denominator size
ASSEMBLY_MISMATCH = ["2pt5_B", "3hmt_A", "4i4y_C", "4nc7_B"]


def per_protein(path, col):
    acc = defaultdict(lambda: defaultdict(list))
    with open(path) as fh:
        rd = csv.DictReader(fh)
        if col not in (rd.fieldnames or []):
            raise SystemExit(f"{os.path.basename(path)}: no column {col!r} "
                             f"(has {rd.fieldnames})")
        for r in rd:
            acc[r["protein_name"]][r["model"]].append(float(r[col]))
    out = {}
    for p, v in acc.items():
        if "base" in v and "finetuned" in v:
            out[p] = (float(np.mean(v["base"])), float(np.mean(v["finetuned"])))
    return out


def analyse(d, min_base, exclude):
    names = sorted(p for p in d if p not in exclude)
    dropped_assembly = sorted(p for p in d if p in exclude)
    kept = [p for p in names if d[p][0] > min_base]
    dropped_low = [(p, d[p][0]) for p in names if d[p][0] <= min_base]
    b = np.array([d[p][0] for p in kept])
    f = np.array([d[p][1] for p in kept])
    red = (b - f) / b * 100.0
    return dict(
        n=len(kept), mean=red.mean(), sd=red.std(ddof=1), median=float(np.median(red)),
        pooled=100.0 * (b.sum() - f.sum()) / b.sum(),
        improved=int((f < b).sum()),
        t_p=stats.ttest_rel(b, f).pvalue, w_p=stats.wilcoxon(b, f).pvalue,
        base_mean=b.mean(), ft_mean=f.mean(),
        dropped_low=dropped_low, dropped_assembly=dropped_assembly)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--min_base", type=float, default=1.0)
    ap.add_argument("--col", default="netmhc_total")
    ap.add_argument("--exclude_assembly", action="store_true",
                    help="also drop the four assembly-mismatched chains")
    ap.add_argument("--no_write", action="store_true")
    a = ap.parse_args()

    excl = set(ASSEMBLY_MISMATCH) if a.exclude_assembly else set()
    print(f"min_base = {a.min_base}   column = {a.col}   "
          f"assembly exclusion = {'on' if a.exclude_assembly else 'off'}\n")
    hdr = (f"{'arm':52s} {'n':>4s} {'old mean':>9s} {'NEW mean':>9s} {'sd':>6s} "
           f"{'median':>7s} {'pooled':>7s} {'impr':>7s}")
    print(hdr); print("-" * len(hdr))

    for path in a.csvs:
        d = per_protein(path, a.col)
        old = analyse(d, -1.0, excl)      # the historical, unguarded convention
        new = analyse(d, a.min_base, excl)
        stem = os.path.basename(path).replace(".csv", "")
        print(f"{stem[:52]:52s} {new['n']:4d} {old['mean']:8.2f}% {new['mean']:8.2f}% "
              f"{new['sd']:6.1f} {new['median']:6.1f}% {new['pooled']:6.1f}% "
              f"{new['improved']:3d}/{new['n']:<3d}")
        for p, bb in new["dropped_low"]:
            print(f"{'':52s}   dropped (base={bb:.2f}): {p}")

        if a.no_write:
            continue
        out = path.replace(".csv", "_guarded_summary.txt")
        if os.path.exists(out):
            print(f"{'':52s}   ! exists, not overwriting: {os.path.basename(out)}")
            continue
        L = [
            "=" * 70,
            f"GUARDED RECOMPUTE — {stem}",
            "=" * 70,
            f"Source CSV (unmodified): {os.path.basename(path)}",
            f"Column: {a.col}",
            f"Minimum-base guard: base > {a.min_base} mean predicted windows",
            f"Assembly-mismatch exclusion: {'on' if a.exclude_assembly else 'off'}",
            "",
            f"Proteins retained: {new['n']}",
            f"  excluded, low base : {len(new['dropped_low'])}"
            + (f"  ({', '.join(f'{p} [{bb:.2f}]' for p, bb in new['dropped_low'])})"
               if new["dropped_low"] else ""),
            f"  excluded, assembly : {len(new['dropped_assembly'])}"
            + (f"  ({', '.join(new['dropped_assembly'])})"
               if new["dropped_assembly"] else ""),
            "",
            "── Per-protein means (guarded set) ───────────────────────────",
            f"  Base:       {new['base_mean']:.2f}",
            f"  Fine-tuned: {new['ft_mean']:.2f}",
            f"  Reduction:  mean {new['mean']:.2f}% ± {new['sd']:.2f}%  "
            f"(median {new['median']:.2f}%)",
            f"  Pooled reduction (denominator-free): {new['pooled']:.2f}%",
            f"  Improved:   {new['improved']}/{new['n']} proteins",
            "",
            "── Statistical tests (guarded set) ───────────────────────────",
            f"  Paired t-test: p={new['t_p']:.4g}",
            f"  Wilcoxon:      p={new['w_p']:.4g}",
            "",
            "── For comparison: the historical UNGUARDED convention ───────",
            f"  Reduction:  mean {old['mean']:.2f}% ± {old['sd']:.2f}%  "
            f"(median {old['median']:.2f}%)   n={old['n']}",
            f"  Shift from applying the guard: {new['mean'] - old['mean']:+.2f} pp",
            "=" * 70,
        ]
        with open(out, "w") as fh:
            fh.write("\n".join(L) + "\n")
        print(f"{'':52s}   wrote {os.path.basename(out)}")


if __name__ == "__main__":
    main()
