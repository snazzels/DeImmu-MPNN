#!/usr/bin/env python
"""%Rank threshold sensitivity for ANY model's stored designs.

Parameterised rewrite of data/output/rank_sensitivity_f16b51e6.py, which hardcoded the
deployed model's input and output paths. That script is left untouched — it produced the
published 43.5/37.9/37.8/35.1% figures answering reviewer R2 P2-7, and results are
immutable. This version takes --in_csv/--out_tag so the same analysis can be
run for the class-II-only ablation and any future checkpoint.

The efficiency trick is preserved exactly: netMHCIIpan is run ONCE per allele over the
deduplicated 15-mer set with the per-peptide %Rank_EL retained, so all four cutoffs are
counted from a single pass instead of four. validate_with_netmhciipan.py cannot be reused
for this because it hardcodes THRESHOLD=2.0 and stores only counts.

Scoring logic is byte-for-byte the same as the original: per-allele best rank, per-protein
means over designs, proteins with zero base-model presented windows excluded from the
percentage reduction (undefined), and the improved count taken over the same set.

  python tools/rank_sensitivity.py \
      --in_csv data/output/eval_deimmunisation_pwmtraj_abec4d2c_ep200.csv \
      --out_tag reanchor_mhc2only
"""
import os, csv, subprocess, tempfile, collections, argparse
import statistics as st

# Derived from __file__ and deliberately NOT from $PF, matching
# validate_with_netmhciipan.py and test_allele_generalisation.py. The sbatch scripts set
# PF to the PROJECT ROOT, one level above CAPE_MPNN, so an earlier version of this file
# that honoured $PF resolved data/output to <project>/data/output and died with
# "missing input" (job 517851 task 3). The one tool that does read $PF,
# evaluate_mhc1_arm.py, is why 14_mhc1_rescore.sbatch has to pass PF="$PF/CAPE_MPNN" --
# an override worth not replicating.
PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OD = os.path.join(PF, "data", "output")

ALLELES = ["DRB1_0101", "DRB1_0301", "DRB1_0401", "DRB1_0701", "DRB1_1101", "DRB1_1501"]
CUTOFFS = [1.0, 2.0, 5.0, 10.0]
# see validate_with_netmhciipan.py for the full rationale
MIN_BASE = 1.0
PEP = 15
BATCH = 50000


def kmers(seq):
    s = seq.replace("*", "").replace("-", "").replace("/", "").replace("X", "")
    return [s[i:i + PEP] for i in range(len(s) - PEP + 1)]


def run_allele(peps, allele):
    """Return {peptide: best %Rank_EL} for one allele."""
    ranks = {}
    for i in range(0, len(peps), BATCH):
        batch = peps[i:i + BATCH]
        with tempfile.NamedTemporaryFile("w", suffix=".pep", delete=False) as f:
            f.write("\n".join(batch)); pf = f.name
        try:
            r = subprocess.run(["netMHCIIpan", "-inptype", "1", "-f", pf, "-a", allele],
                               capture_output=True, text=True)
        except FileNotFoundError:
            os.unlink(pf)
            raise SystemExit("FATAL: netMHCIIpan not on PATH. Add "
                             "CAPE_MPNN/external/programs/bin to PATH (and note the tool "
                             "needs tcsh, so it cannot run on a machine without it).")
        os.unlink(pf)
        for line in r.stdout.splitlines():
            p = line.split()
            if len(p) < 10:
                continue
            try: int(p[0])
            except ValueError: continue
            pep = p[2]
            try: rank = float(p[9])
            except ValueError: continue
            if pep not in ranks or rank < ranks[pep]:
                ranks[pep] = rank
    return ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True,
                    help="eval_deimmunisation_*.csv holding base+finetuned designs")
    ap.add_argument("--out_tag", required=True,
                    help="output goes to validate_netmhciipan_rank_sensitivity_<tag>_summary.txt")
    ap.add_argument("--label", default=None, help="human label for the report header")
    a = ap.parse_args()

    in_csv = a.in_csv if os.path.isabs(a.in_csv) else os.path.join(PF, a.in_csv)
    out = os.path.join(OD, f"validate_netmhciipan_rank_sensitivity_{a.out_tag}_summary.txt")
    # Per-protein table (2026-08-26). The summary-only output was the reason the min-base
    # guard could NOT be applied retrospectively to this analysis, unlike every other
    # validator arm, which tools/recompute_guarded_reduction.py fixed offline from its CSV.
    # A ~6 h netMHCIIpan re-run was the price. Persist the per-protein numbers so any future
    # change to the exclusion rule or the aggregation is an offline recompute, not a re-score.
    out_csv = os.path.join(OD, f"validate_netmhciipan_rank_sensitivity_{a.out_tag}.csv")
    # Same guard as the original: never overwrite a frozen result.
    assert not os.path.exists(out), f"refusing to overwrite {out}"
    assert not os.path.exists(out_csv), f"refusing to overwrite {out_csv}"
    assert os.path.exists(in_csv), f"missing input {in_csv}"
    label = a.label or a.out_tag

    rows = list(csv.DictReader(open(in_csv)))
    # Only the two arms this analysis compares; a stray model label would corrupt the means.
    rows = [r for r in rows if r.get("model") in ("base", "finetuned")]
    uniq = sorted({k for r in rows for k in kmers(r["sequence"])})
    print(f"{len(rows)} designs; {len(uniq)} unique 15-mers; one netMHCIIpan pass per allele",
          flush=True)

    rankmap = {}
    for al in ALLELES:
        rankmap[al] = run_allele(uniq, al)
        print(f"  scored {al}: {len(rankmap[al])} peptides", flush=True)
    # Fail loudly and early if the predictor produced nothing. Without this the run
    # proceeds, every base total is 0, every protein is excluded as undefined, and the
    # first cutoff dies on `statistics.mean` of an empty list -- a confusing crash a long
    # way from the cause. Discovered locally, where netMHCIIpan cannot run at all: its
    # wrapper `exec`s tcsh, which is absent, so every allele silently returned 0 peptides.
    empty = [al for al, rm in rankmap.items() if not rm]
    if empty:
        raise SystemExit(
            f"FATAL: netMHCIIpan returned no parseable ranks for {len(empty)}/{len(ALLELES)} "
            f"alleles ({', '.join(empty)}). Check the binary runs at all -- on a machine "
            f"without tcsh the wrapper fails with 'exec: tcsh: not found' and produces "
            f"empty output rather than an error exit.")

    lines = [f"%RANK THRESHOLD SENSITIVITY -- {label}",
             "=" * 78,
             f"designs: {len(rows)} ({os.path.basename(in_csv)});  unique 15-mers: {len(uniq)}",
             "One netMHCIIpan pass per allele; all cutoffs counted from the retained ranks.",
             f"Per-protein reduction excludes proteins whose mean base burden is <= {MIN_BASE}",
             "window(s): the percentage is undefined at 0 and noise-dominated just above it.", "",
             f"{'cutoff':>8}{'base':>10}{'finetuned':>11}{'reduction':>11}{'improved':>11}"]

    # Pre-compute each design's k-mers once; the original recomputed them inside the
    # allele loop for every cutoff, which is 4 x 6 redundant passes over every sequence.
    # Indexed by position, NOT by id(row): id() keys are only valid while every row object
    # stays alive, which is true here but is a trap waiting for the next edit.
    kmer_cache = [kmers(r["sequence"]) for r in rows]

    per_protein_rows, unguarded = [], []
    for cut in CUTOFFS:
        per = collections.defaultdict(lambda: {"base": [], "finetuned": []})
        for ri, r in enumerate(rows):
            ks = kmer_cache[ri]
            tot = 0
            for al in ALLELES:
                rm = rankmap[al]
                tot += sum(1 for k in ks if rm.get(k, 999) <= cut)
            per[r["protein_name"]][r["model"]].append(tot)
        # bs/fs are the GUARDED set, so the base and finetuned columns describe the same
        # proteins as the reduction column beside them (2026-08-26). The original appended
        # them before the guard, which left one row of the table mixing a 100-protein mean
        # with a 98-protein percentage -- the published 2% row reads base 45.79 (n=100)
        # against a reduction over n=98, and so disagrees with
        # validate_netmhciipan_sweep_f16b51e6_guarded_summary.txt's base of 46.70 on the
        # same designs. bs_all/fs_all keep the old convention for continuity, reported
        # separately in the footer rather than silently inside the table.
        reds, bs, fs, bs_all, fs_all, dropped = [], [], [], [], [], []
        for p, d in per.items():
            if not d["base"] or not d["finetuned"]:
                continue
            b = st.mean(d["base"]); f = st.mean(d["finetuned"])
            bs_all.append(b); fs_all.append(f)
            # Written for every protein at every cutoff, INCLUDING the ones the guard drops,
            # with the guard decision recorded rather than applied. That is what makes an
            # offline recompute under a different rule possible.
            per_protein_rows.append({
                "cutoff": cut, "protein_name": p,
                "n_base": len(d["base"]), "n_finetuned": len(d["finetuned"]),
                "base_mean": f"{b:.6f}", "finetuned_mean": f"{f:.6f}",
                "reduction_pct": f"{100 * (b - f) / b:.6f}" if b > 0 else "",
                "excluded_low_base": int(b <= MIN_BASE), "min_base": MIN_BASE,
            })
            # Minimum-base guard (2026-08-25). Excluding only b == 0 is too weak: a mean
            # base burden of a fraction of a window still yields a huge percentage. On the
            # deployed model's headline run a protein with base = 0.1 contributed -300%,
            # moving the reported mean by 3.4 pp. Aligned with validate_with_netmhciipan.py.
            if b <= MIN_BASE:
                dropped.append((p, b))
                continue
            bs.append(b); fs.append(f)
            reds.append(100 * (b - f) / b)
        lines.append(f"{cut:>7.0f}%{st.mean(bs):>10.2f}{st.mean(fs):>11.2f}"
                     f"{st.mean(reds):>10.1f}%{sum(x > 0 for x in reds):>8}/{len(reds)}"
                     + (f"   [{len(dropped)} excluded, base<={MIN_BASE}]" if dropped else ""))
        print(lines[-1], flush=True)
        unguarded.append((cut, st.mean(bs_all), st.mean(fs_all), len(bs_all)))

    lines += ["", "Base and fine-tuned columns are means over the SAME guarded proteins as the",
              "reduction beside them. For continuity with the pre-guard convention, the",
              "unguarded means over all proteins with both arms were:"]
    lines += [f"  {c:>3.0f}%  base {b:>7.2f}  finetuned {f:>7.2f}  (n={n})"
              for c, b, f, n in unguarded]
    lines += ["", "Interpretation: if the reduction holds across cutoffs, the effect is not an",
              "artifact of the field-standard 2% strong/weak-binder threshold (R2 P2-7)."]
    open(out, "w").write("\n".join(lines) + "\n")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_protein_rows[0].keys()))
        w.writeheader(); w.writerows(per_protein_rows)
    print("\nwrote", out)
    print("wrote", out_csv, f"({len(per_protein_rows)} rows)")


if __name__ == "__main__":
    main()
