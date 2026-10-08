#!/usr/bin/env python
"""Refuse the unmatched-sample trap: check that two or more result files are
protein-for-protein comparable BEFORE their numbers are compared.

WHY THIS EXISTS
---------------
Nearly every validator in this project draws its own protein sample. Two runs on
DIFFERENT protein sets differ for two reasons at once -- the model, and the sample --
and the two cannot be separated after the fact. This has bitten the project four
times:

  * Figure 3's MHC-I-only arm: 100 proteins, only 44 overlapping the pinned 98.
  * All 19 selfconsistency_boltz_sweep_* dirs: 19 DISTINCT protein sets.
  * Every orthogonal validator existed for the deployed model on a different
    sample than the ablation (netMHCpan, MixMHC2pred, allele gen, rank sensitivity).
  * allele_generalisation: deployed f16b51e6 vs abec4d2c shared only 15 of 24
    proteins -- despite a seeded RNG, because the length quartiles that drive the
    stratified draw are computed from each input CSV's own protein population.

A seeded RNG is NOT sufficient. Only an explicit pinned list is.

USAGE
-----
  python tools/check_protein_match.py FILE [FILE ...]

Accepts any mix of .csv (needs a protein_name / protein / name column) and .json
(reads protein_set, else any list-of-str value under a *protein* key). Prints the
per-file fingerprint, the pairwise overlap matrix, and a verdict.

  exit 0  -> all files share an identical protein set; cross-file comparison is sound
  exit 1  -> sets differ; cross-file comparison is NOT valid as-is
  exit 2  -> could not determine a protein set for some file

Intersecting to the common subset and recomputing is a legitimate rescue for
paired per-protein statistics; re-running with --protein_list is the clean fix.
Relabelling is NOT a fix.
"""
import os, sys, csv, json, hashlib, argparse, itertools

CSV_KEYS  = ("protein_name", "protein", "name", "pdb", "pdb_chain")


def from_csv(path):
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        if rd.fieldnames is None:
            return None, "empty csv"
        key = next((k for k in CSV_KEYS if k in rd.fieldnames), None)
        if key is None:
            return None, f"no protein column (looked for {CSV_KEYS}, have {rd.fieldnames[:8]})"
        return sorted({r[key] for r in rd if r.get(key)}), None


def from_json(path):
    try:
        J = json.load(open(path))
    except Exception as e:
        return None, f"unreadable json: {e}"
    if isinstance(J, dict):
        if isinstance(J.get("protein_set"), list):
            return sorted(set(map(str, J["protein_set"]))), None
        for k, v in J.items():
            if "protein" in k.lower() and isinstance(v, list) and v and isinstance(v[0], str):
                return sorted(set(v)), None
    return None, "no protein_set (or *protein* list) key"


def load(path):
    if not os.path.exists(path):
        return None, "file not found"
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return from_csv(path)
    if ext == ".json":
        return from_json(path)
    return None, f"unsupported extension {ext!r} (want .csv or .json)"


def md5(names):
    return hashlib.md5("\n".join(names).encode()).hexdigest()[:12]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Verify two or more result files are protein-for-protein comparable.")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--write_common", metavar="PATH", default=None,
                    help="write the intersection to PATH, usable as --protein_list for a clean re-run")
    a = ap.parse_args(argv)

    if len(a.files) < 2:
        print("Need at least two files to compare.", file=sys.stderr)
        return 2

    sets, bad = {}, False
    print(f"{'file':<58} {'n':>5}  fingerprint")
    print("-" * 84)
    for f in a.files:
        names, err = load(f)
        if names is None:
            print(f"{os.path.basename(f):<58} {'--':>5}  ERROR: {err}")
            bad = True
            continue
        sets[f] = names
        print(f"{os.path.basename(f):<58} {len(names):>5}  {md5(names)}")

    if bad:
        print("\nVERDICT: ⚠️  could not read a protein set from every file — cannot certify.")
        return 2

    fingerprints = {md5(v) for v in sets.values()}
    if len(fingerprints) == 1:
        n = len(next(iter(sets.values())))
        print(f"\nVERDICT: ✅ MATCHED — all {len(sets)} files use the same {n} proteins.")
        print("Cross-file comparison is protein-for-protein sound.")
        return 0

    print("\nPairwise overlap:")
    print(f"  {'A':<26} {'B':<26} {'common':>7} {'A only':>7} {'B only':>7}")
    for x, y in itertools.combinations(sets, 2):
        sx, sy = set(sets[x]), set(sets[y])
        print(f"  {os.path.basename(x)[:26]:<26} {os.path.basename(y)[:26]:<26} "
              f"{len(sx & sy):>7} {len(sx - sy):>7} {len(sy - sx):>7}")

    common = set.intersection(*(set(v) for v in sets.values()))
    smallest = min(len(v) for v in sets.values())
    print(f"\nCommon to all {len(sets)} files: {len(common)} proteins "
          f"({100*len(common)/smallest:.0f}% of the smallest set).")

    if a.write_common:
        if os.path.exists(a.write_common):
            print(f"\nRefusing to overwrite {a.write_common}.", file=sys.stderr)
            return 2
        with open(a.write_common, "w") as fh:
            fh.write("\n".join(sorted(common)) + "\n")
        print(f"Wrote {len(common)} common proteins -> {a.write_common}")

    print("\nVERDICT: ❌ NOT MATCHED — these files do NOT share a protein set.")
    print("Any cross-file difference confounds the model effect with the sample effect.")
    print("Fix: re-run with --protein_list (clean), or intersect to the common subset and")
    print("recompute paired per-protein statistics (rescue). Relabelling is not a fix.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
