#!/usr/bin/env python
"""
P0-5 step 2 -- the HLA-DP inverted-binding confound, resolved by forward-only re-scoring.

WHY THIS EXISTS
    The paper's entire HLA-DP story turns on ONE heterodimer, DPA1*02:01-DPB1*01:01:
    it carries ~2x the base burden of the other two, dominates the DP group statistic,
    is the one the deployed model fails to reduce (3.1% pooled), and is the one the
    class-II-only checkpoint reduces by 51.7% -- which is the sole basis for the
    recommendation "where HLA-DP coverage matters, use the class-II-only checkpoint".

    DPA1*02:01-DPB1*01:01 is also the field's canonical INVERTED-BINDER molecule, and
    netMHCIIpan-4.3 applies inversion BY DEFAULT FOR DP ONLY.  Step 1 (job 531434)
    measured it on our own designs: 22.8% of probe 15-mers are called inverted at this
    heterodimer, against 6.7% and 8.8% at the other two.  Raised by the domain seat of
    the 2026-09-05 panel; the manuscript never mentions inversion.

    THE RIVAL EXPLANATION.  A training signal built from FORWARD-register 15-mer DRB1
    PWMs has no purchase on ligands scored in the reverse orientation.  That would
    explain BOTH the elevated base burden at this heterodimer AND the deployed model's
    failure on it, without invoking the class I objective at all.

THE TEST
    Re-score and recompute the base-vs-fine-tuned DP reduction on FORWARD-ORIENTATION
    windows only.

        gap SURVIVES forward-only  -> the objective-based explanation is strengthened
                                      and the paper gains a result.
        gap does NOT survive       -> the locus-level deployment recommendation rests on
                                      a scoring-register artefact and must be reworded.

WHY A RE-SCORE AND NOT A RE-PARSE
    validate_with_netmhciipan.py and test_allele_generalisation.py both keep only field
    10 (%Rank_EL) and discard Core and Inverted, so orientation has never existed on
    disk.  It must be regenerated.

COLUMN LAYOUT (confirmed twice over, independently)
    1 Pos  2 MHC  3 Peptide  4 Of  5 Core  6 Core_Rel  7 Inverted  8 Identity
    9 Score_EL  10 %Rank_EL  11 Exp_Bind  12 BindLevel(optional)
    (a) the vendor's shipped reference outputs carry this header identically across all
        five output modes; (b) on real DP output every Inverted=1 row's Core is the
        REVERSE of a forward substring of the peptide.  This script re-verifies (b) at
        runtime rather than trusting the index -- see _check_orientation_semantics.

ACCEPTANCE TEST -- ENFORCED, NOT ADVISORY
    With orientation ignored, this script must reproduce the published per-design counts
    EXACTLY, design by design, from allele_generalisation_pinned24_*.csv (column
    n_<allele>).  It replicates that pipeline's conventions deliberately: windows as a
    LIST (duplicate 15-mers count more than once), %Rank_EL <= 2.0, and */-// stripped.
    If the all-windows arm does not reproduce, the forward-only split is not
    interpretable and the run aborts rather than emitting a plausible, wrong table.

USAGE
    python tools/dp_orientation.py \
        --arm deployed=data/output/allele_generalisation_pinned24_deployed.csv \
        --arm mhc2only=data/output/allele_generalisation_pinned24_mhc2only.csv \
        --out_tag dp_orientation_pinned24 --orient_col 6
"""
import os, sys, csv, argparse, collections, subprocess, tempfile
import numpy as np

PF = os.environ.get("CAPE_PF", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OD = os.path.join(PF, "data", "output")

DP_ALLELES = ["HLA-DPA10201-DPB10101", "HLA-DPA10103-DPB10401", "HLA-DPA10103-DPB10201"]
THRESHOLD   = 2.0     # %Rank_EL <= 2% = presented; identical to test_allele_generalisation.py
PEPTIDE_LEN = 15
MIN_BASE    = 1.0     # per-protein guard, identical to every other arm in this project
SEMANTIC_MIN_FRAC = 0.90   # fraction of Inverted=1 rows whose reversed Core must be found


# ── window handling: replicate test_allele_generalisation.py EXACTLY ──────────
def get_15mers(seq):
    """NOTE: returns a LIST, not a set. Duplicate 15-mers count more than once, which is
    what the published pipeline does; deduplicating here would silently change the
    counts and break the acceptance test."""
    seq = seq.replace("*", "").replace("-", "").replace("/", "")
    return [seq[i:i + PEPTIDE_LEN] for i in range(len(seq) - PEPTIDE_LEN + 1)]


# ── netMHCIIpan, retaining orientation ───────────────────────────────────────
def run_netmhciipan(peptides, allele, orient_col):
    """{peptide: (best_rank, inverted, core)}. Keeps the row with the LOWEST %Rank_EL,
    together with THAT row's orientation -- the orientation must travel with the score
    it belongs to, not be taken from an arbitrary row for the same peptide."""
    out, n_rows, sem_ok, sem_tot = {}, 0, 0, 0
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
                int(p[0])                      # Pos; rejects header and the trailing
                rank = float(p[9])             # "Number of strong binders:" summary line
            except ValueError:
                continue
            pep, core, inv = p[2], p[4], p[orient_col]
            if inv not in ("0", "1"):
                sys.exit(f"FATAL: column {orient_col} holds {inv!r}, not 0/1, for {allele}.\n"
                         f"       That is not the Inverted column. Refusing to guess.\n"
                         f"       Offending line: {line.strip()}")
            n_rows += 1
            if inv == "1":                     # semantic check, accumulated per allele
                sem_tot += 1
                if core[::-1] in pep:
                    sem_ok += 1
            if pep not in out or rank < out[pep][0]:
                out[pep] = (rank, int(inv), core)
    finally:
        os.unlink(pep_file)

    if not out:
        sys.exit(f"FATAL: netMHCIIpan returned zero parseable ranks for {allele}. On this "
                 f"cluster that means the wrapper did not run (it execs tcsh). Not a result.")
    _check_orientation_semantics(allele, sem_ok, sem_tot, n_rows)
    return out


def _check_orientation_semantics(allele, ok, tot, n_rows):
    """An Inverted=1 row should report a Core that is the REVERSE of a forward substring
    of its peptide. This verifies the column MEANS what we think rather than merely
    sitting where we think -- the failure mode this two-step design exists to prevent is
    a plausible table built from a wrong field, and a header label alone cannot rule
    that out."""
    if tot == 0:
        print(f"    [orient] {allele}: 0 inverted rows of {n_rows} "
              f"(semantic check not exercised)")
        return
    frac = ok / tot
    print(f"    [orient] {allele}: {tot}/{n_rows} inverted; reversed-core verified "
          f"{ok}/{tot} ({100*frac:.1f}%)")
    if frac < SEMANTIC_MIN_FRAC:
        sys.exit(f"FATAL: only {100*frac:.1f}% of Inverted=1 rows at {allele} have a Core "
                 f"that reverses into the peptide (need >= {100*SEMANTIC_MIN_FRAC:.0f}%).\n"
                 f"       Either the orientation column is wrong or the Core semantics "
                 f"differ from what step 1 established. Refusing to emit a table.")


# ── counting ─────────────────────────────────────────────────────────────────
def counts_for(seq, scored):
    """(all, forward, inverted) presented-window counts for one design at one allele."""
    a = f = i = 0
    for k in get_15mers(seq):
        v = scored.get(k)
        if v is None or v[0] > THRESHOLD:
            continue
        a += 1
        if v[1] == 1:
            i += 1
        else:
            f += 1
    return a, f, i


def pooled(per_protein):
    """Mean-over-proteins of mean-over-designs, matching the published convention."""
    b = np.mean([np.mean(v["base"]) for v in per_protein.values()])
    f = np.mean([np.mean(v["finetuned"]) for v in per_protein.values()])
    return b, f, (100.0 * (b - f) / b if b else float("nan"))


def guarded(per_protein):
    pct, excl = [], 0
    for v in per_protein.values():
        b, f = np.mean(v["base"]), np.mean(v["finetuned"])
        if b <= MIN_BASE:
            excl += 1
            continue
        pct.append(100.0 * (b - f) / b)
    pct = np.array(pct) if pct else np.array([np.nan])
    return pct, excl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    metavar="LABEL=PATH", help="repeatable; e.g. deployed=data/output/x.csv")
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--orient_col", type=int, required=True,
                    help="0-based index of the Inverted column (6 for netMHCIIpan-4.3)")
    a = ap.parse_args()

    out_csv = os.path.join(OD, f"{a.out_tag}.csv")
    out_txt = os.path.join(OD, f"{a.out_tag}_summary.txt")
    for p in (out_csv, out_txt):
        if os.path.exists(p):
            sys.exit(f"REFUSING TO OVERWRITE {p}\nGive the corrective run a new --out_tag.")

    # ── load arms ────────────────────────────────────────────────────────────
    arms = {}
    for spec in a.arm:
        if "=" not in spec:
            sys.exit(f"FATAL: --arm needs LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        src = path if os.path.isabs(path) else os.path.join(PF, path)
        arms[label] = list(csv.DictReader(open(src)))
        print(f"[in ] {label}: {os.path.basename(src)} ({len(arms[label])} rows)")

    # ── HARD RULE: the arms must be matched, on the protein set AND the base arm ──
    labels = list(arms)
    psets = {l: {r["protein_name"] for r in arms[l]} for l in labels}
    ref = psets[labels[0]]
    for l in labels[1:]:
        if psets[l] != ref:
            sys.exit(f"FATAL: protein sets differ ({labels[0]} n={len(ref)} vs {l} "
                     f"n={len(psets[l])}; symmetric difference "
                     f"{sorted(ref ^ psets[l])}).\n"
                     "       Cross-arm numbers on different samples are not comparable.")
    def basemap(rows):
        return {(r["protein_name"], r["seq_idx"]): r["sequence"]
                for r in rows if r["model"] == "base"}
    bref = basemap(arms[labels[0]])
    for l in labels[1:]:
        bl = basemap(arms[l])
        if set(bl) != set(bref) or any(bl[k] != bref[k] for k in bref):
            n_diff = sum(1 for k in bref if bl.get(k) != bref[k])
            sys.exit(f"FATAL: the BASE arm differs between {labels[0]} and {l} "
                     f"({n_diff} of {len(bref)} designs).\n"
                     "       Same protein names is NOT the same base arm.")
    print(f"[chk] {len(labels)} arms matched: {len(ref)} proteins, "
          f"{len(bref)} base designs byte-identical")

    # ── score ────────────────────────────────────────────────────────────────
    kmers = {k for rows in arms.values() for r in rows for k in get_15mers(r["sequence"])}
    print(f"[pep] {len(kmers)} unique 15-mers -> {len(DP_ALLELES)} netMHCIIpan passes")
    scored = {}
    for al in DP_ALLELES:
        print(f"  netMHCIIpan {al} …", flush=True)
        scored[al] = run_netmhciipan(kmers, al, a.orient_col)

    # ── aggregate + acceptance test ──────────────────────────────────────────
    # per_protein[(arm, allele, kind)][protein][model] = [counts per design]
    agg = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(list)))
    mismatches = []
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "allele", "protein_name", "model", "seq_idx",
                    "n_all", "n_forward", "n_inverted", "n_published"])
        for label, rows in arms.items():
            for r in rows:
                for al in DP_ALLELES:
                    n_all, n_fwd, n_inv = counts_for(r["sequence"], scored[al])
                    pub = r.get(f"n_{al}")
                    pub_i = int(float(pub)) if pub not in (None, "") else None
                    if pub_i is not None and pub_i != n_all:
                        mismatches.append((label, al, r["protein_name"],
                                           r["model"], r["seq_idx"], n_all, pub_i))
                    w.writerow([label, al, r["protein_name"], r["model"],
                                r["seq_idx"], n_all, n_fwd, n_inv,
                                "" if pub_i is None else pub_i])
                    for kind, n in (("all", n_all), ("forward", n_fwd), ("inverted", n_inv)):
                        agg[(label, al, kind)][r["protein_name"]][r["model"]].append(n)

    if mismatches:
        head = "\n".join(f"        {m[0]} {m[1]} {m[2]} {m[3]} idx{m[4]}: "
                         f"recomputed {m[5]} vs published {m[6]}" for m in mismatches[:12])
        sys.exit(f"FATAL: ACCEPTANCE TEST FAILED. {len(mismatches)} design-level "
                 f"disagreements between the all-windows recomputation and the published "
                 f"n_<allele> counts.\n{head}\n"
                 "       Something other than orientation differs, so the forward-only "
                 "split is not interpretable. Refusing to write a summary.")
    print(f"[chk] ACCEPTANCE TEST PASSED: all-windows counts reproduce the published "
          f"n_<allele> exactly, design by design")

    # ── report ───────────────────────────────────────────────────────────────
    L = ["HLA-DP ORIENTATION / INVERTED-BINDING RE-SCORE (P0-5 step 2)",
         "=" * 78,
         f"arms      : {', '.join(labels)}  (matched: {len(ref)} proteins, shared base arm)",
         f"scorer    : netMHCIIpan-4.3, %Rank_EL <= {THRESHOLD}, Inverted = column "
         f"{a.orient_col} (0-based)",
         f"guard     : per-protein reduction excludes mean base burden <= {MIN_BASE}",
         f"unique 15-mers scored : {len(kmers)}",
         "",
         "ACCEPTANCE TEST PASSED: with orientation ignored, every per-design count "
         "reproduces",
         "the published n_<allele> exactly. Orientation is therefore the only new variable.",
         ""]

    # (a) inverted fraction OF PRESENTED WINDOWS, per allele per arm per model
    L += ["-" * 78,
          "(a) INVERTED FRACTION OF PRESENTED WINDOWS (%Rank_EL <= 2)",
          "    Step 1's 22.8/6.7/8.8% were over ALL probe 15-mers, unthresholded and",
          "    arm-pooled. These are the presented-window fractions, split by arm and",
          "    model -- these are the reportable ones.",
          "",
          f"    {'allele':24s} {'arm':10s} {'base':>14s} {'fine-tuned':>14s}"]
    for al in DP_ALLELES:
        for label in labels:
            row = []
            for model in ("base", "finetuned"):
                tot = sum(sum(v[model]) for v in agg[(label, al, "all")].values())
                inv = sum(sum(v[model]) for v in agg[(label, al, "inverted")].values())
                row.append(f"{inv}/{tot} ({100*inv/tot:.1f}%)" if tot else "0/0 (n/a)")
            L.append(f"    {al:24s} {label:10s} {row[0]:>14s} {row[1]:>14s}")
    L.append("")

    # (b) the load-bearing comparison
    L += ["-" * 78,
          "(b) BASE-vs-FINE-TUNED DP REDUCTION: ALL WINDOWS vs FORWARD ONLY",
          "    'all' must match the published pooled numbers (it is the acceptance test).",
          "    'forward' is the confound-free comparison.",
          "",
          f"    {'allele':24s} {'arm':10s} {'all %':>9s} {'forward %':>11s} "
          f"{'delta':>8s} {'fwd n':>6s} {'excl':>5s}"]
    verdict_rows = []
    for al in DP_ALLELES:
        for label in labels:
            _, _, r_all = pooled(agg[(label, al, "all")])
            pp_fwd = agg[(label, al, "forward")]
            bf, ff, r_fwd = pooled(pp_fwd)
            pct_fwd, excl = guarded(pp_fwd)
            L.append(f"    {al:24s} {label:10s} {r_all:>8.2f} {r_fwd:>10.2f} "
                     f"{r_fwd - r_all:>+8.2f} {len(pct_fwd):>6d} {excl:>5d}")
            verdict_rows.append((al, label, r_all, r_fwd, bf, ff))
    L.append("")

    # per-protein paired test on the forward-only arm, per allele per arm
    L += ["-" * 78,
          "(c) FORWARD-ONLY, PER-PROTEIN GUARDED MEAN + PAIRED TEST",
          "",
          f"    {'allele':24s} {'arm':10s} {'mean %':>9s} {'sd':>7s} {'n':>4s} "
          f"{'improved':>9s} {'Wilcoxon p':>12s}"]
    for al in DP_ALLELES:
        for label in labels:
            pp = agg[(label, al, "forward")]
            pct, excl = guarded(pp)
            p_str = "n/a"
            try:
                from scipy import stats
                b = np.array([np.mean(v["base"]) for v in pp.values()])
                f = np.array([np.mean(v["finetuned"]) for v in pp.values()])
                keep = b > MIN_BASE
                if keep.sum() > 1 and np.any(b[keep] != f[keep]):
                    p_str = f"{stats.wilcoxon(b[keep], f[keep]).pvalue:.2e}"
            except Exception:                                  # noqa: BLE001
                pass
            L.append(f"    {al:24s} {label:10s} {np.nanmean(pct):>8.2f} "
                     f"{np.nanstd(pct, ddof=1) if len(pct) > 1 else float('nan'):>7.2f} "
                     f"{len(pct):>4d} {int(np.nansum(pct > 0)):>9d} {p_str:>12s}")
    L.append("")

    # ── the interpretation, fixed in advance ─────────────────────────────────
    key = [v for v in verdict_rows if v[0] == "HLA-DPA10201-DPB10101"]
    L += ["-" * 78,
          "READING (fixed before the numbers were computed):",
          "  The claim at risk is the locus-level recommendation 'where HLA-DP coverage",
          "  matters, use the class-II-only checkpoint', which rests entirely on",
          "  DPA1*02:01-DPB1*01:01 -- the canonical inverted binder.",
          "",
          "  If the deployed-vs-class-II-only GAP at that heterodimer SURVIVES forward-only",
          "  scoring -> inversion is not the explanation; the objective-based account is",
          "  strengthened and the paper gains a result.",
          "  If the gap COLLAPSES -> the recommendation rests on a scoring-register",
          "  artefact and must be reworded.",
          ""]
    if len(key) == 2:
        by = {k[1]: k for k in key}
        if set(by) == {"deployed", "mhc2only"}:
            g_all = by["mhc2only"][2] - by["deployed"][2]
            g_fwd = by["mhc2only"][3] - by["deployed"][3]
            L += [f"  AT DPA1*02:01-DPB1*01:01:",
                  f"    gap (class-II-only - deployed), all windows   : {g_all:+.2f} pp",
                  f"    gap (class-II-only - deployed), forward only  : {g_fwd:+.2f} pp"]
            # A ratio is only meaningful when both gaps point the same way. Printing
            # "retains -93%" for a sign flip would read as "almost all of it".
            if abs(g_all) < 1e-9:
                L.append("    all-windows gap is ~zero; a retention ratio is undefined")
            elif g_fwd * g_all < 0:
                L.append(f"    ⚠ THE GAP CHANGED SIGN under forward-only scoring "
                         f"({g_all:+.2f} -> {g_fwd:+.2f} pp). A retention ratio is "
                         f"meaningless here; read the two numbers.")
            else:
                L.append(f"    the gap retains {100*g_fwd/g_all:.0f}% of its "
                         f"all-windows size")
            L.append("")
    L += ["  Do not round either outcome into the other. Report the split.",
          ""]

    open(out_txt, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"[out] {out_csv}\n[out] {out_txt}")


if __name__ == "__main__":
    main()
