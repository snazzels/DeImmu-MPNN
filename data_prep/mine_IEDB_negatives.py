#!/usr/bin/env python
"""Mine IEDB for experimentally-tested NON-immunogenic MHC-II peptides.

Prerequisite for breaking the netMHCIIpan
circularity.

WHY THIS EXISTS
---------------
mine_IEDB_MHC2.py keeps only `Positive` assay records. Its "non-immunogenic"
class (human_nonim.csv) is *random human 15-mers* selected only for the absence
of an IEDB immunogenic record. A classifier trained to separate those from the
immunogenic set learns "is this peptide human, and does it look like an MHC
binder?" -- not "is this peptide immunogenic?". It would post a high AUC that
means nothing.

IEDB also contains 360k `Negative` assay records: peptides that were
experimentally tested and found NOT to elicit a T-cell response. Those are the
correct negatives. This script extracts them.

TWO NEGATIVE SETS ARE PRODUCED
------------------------------
1. `nonimmunogenic.csv` -- clean negatives: tested negative on a canonical
   effector readout (the same readouts used to define the positive set), never
   observed positive anywhere in IEDB.

2. `nonimmunogenic_binders.csv` -- the subset of (1) that netMHCIIpan predicts
   to be a STRONG BINDER to at least one HLA-DR allele (%Rank_EL <= 2).

Set (2) is the scientifically important one. Its peptides bind MHC-II but do
not trigger a T-cell response, so a classifier trained on
{immunogenic vs binder-matched negatives} cannot succeed by learning MHC
binding -- it must learn immunogenicity itself. That distinction is what the
funded proposal is really asking for, and most de-immunisation tools conflate
the two.

Outputs
-------
<output>/nonimmunogenic.csv           clean negatives + netMHCIIpan binding
<output>/nonimmunogenic_binders.csv   binder-matched subset (the useful one)
<output>/immunogenic_binding.csv      positives + netMHCIIpan binding (for comparison)
<output>/negatives_stats.txt          statistics, incl. the binding-confound check
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

PF = os.environ.get("PF", os.path.join(os.path.dirname(__file__), ".."))

# ── Config ────────────────────────────────────────────────────────────────────
PEPTIDE_LENGTH = 15
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
BATCH_SIZE = 5000

DRB1_ALLELES = ["DRB1_0101", "DRB1_0301", "DRB1_0401",
                "DRB1_0701", "DRB1_1101", "DRB1_1501"]

# %Rank_EL threshold for "presented" -- matches the definition used throughout
# the project (MHC-II_rank_peptides.py, Track 8).
RANK_THRESHOLD = 2.0

# Must match mine_IEDB_MHC2.py exactly, so positives and negatives are drawn
# from the same assay universe and differ ONLY in outcome.
EFFECTOR_RESPONSES = {
    "IFNg release", "proliferation", "cytotoxicity", "TNFa release", "TNF release",
    "activation", "degranulation", "granzyme B release", "antibody help",
    "T cell help", "IFNa release", "IFNb release",
}
POSITIVE_QUALITATIVE = {"Positive", "Positive-High", "Positive-Intermediate",
                        "Positive-Low"}
NEGATIVE_QUALITATIVE = {"Negative"}


def is_valid_peptide(seq) -> bool:
    s = str(seq).strip().upper()
    return len(s) == PEPTIDE_LENGTH and set(s) <= STANDARD_AA


# ── netMHCIIpan (same wrappers as validate_with_netmhciipan.py) ───────────────

def _parse_netmhciipan(lines):
    ranks = {}
    for line in lines:
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            int(parts[0])
        except ValueError:
            continue
        peptide, rank_el = parts[2], float(parts[9])
        if peptide not in ranks or rank_el < ranks[peptide]:
            ranks[peptide] = rank_el
    return ranks


def run_netmhciipan(peptides, allele):
    ranks = {}
    batches = [peptides[i:i + BATCH_SIZE] for i in range(0, len(peptides), BATCH_SIZE)]
    for batch in tqdm(batches, desc=f"  {allele}", leave=False):
        with tempfile.TemporaryDirectory() as tmpdir:
            pep_file = os.path.join(tmpdir, "input.pep")
            with open(pep_file, "w") as fh:
                fh.write("\n".join(batch))
            res = subprocess.run(
                ["netMHCIIpan", "-inptype", "1", "-f", pep_file, "-a", allele],
                capture_output=True, check=True,
            )
            ranks.update(_parse_netmhciipan(res.stdout.decode().splitlines()))
    return ranks


def score_peptides(peptides, alleles):
    """Return DataFrame: peptide, one %Rank_EL column per allele, best_rank, n_alleles_presented."""
    peptides = sorted(set(peptides))
    print(f"Scoring {len(peptides):,} peptides x {len(alleles)} alleles with netMHCIIpan…")
    df = pd.DataFrame({"peptide": peptides})
    for allele in alleles:
        t0 = time.time()
        ranks = run_netmhciipan(peptides, allele)
        df[allele] = df.peptide.map(ranks)
        miss = df[allele].isna().sum()
        print(f"  {allele}: done in {time.time()-t0:.0f}s"
              + (f"  ({miss} unscored)" if miss else ""))
    df["best_rank"] = df[alleles].min(axis=1)
    df["n_alleles_presented"] = (df[alleles] <= RANK_THRESHOLD).sum(axis=1)
    df["is_binder"] = df.n_alleles_presented > 0
    return df


# ── IEDB extraction ───────────────────────────────────────────────────────────

def load_iedb(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        sys.exit(f"Missing IEDB cache: {cache_path}\nRun mine_IEDB_MHC2.py first.")
    print(f"Reading {cache_path} (this takes a minute; ~1.3 GB)…")
    df = pd.read_csv(cache_path, low_memory=False, header=[0, 1])
    df.columns = [f"{a}|{b}" if not str(b).startswith("Unnamed") else str(a)
                  for a, b in df.columns]
    return df


def find_col(df, *keywords):
    for k in keywords:
        for c in df.columns:
            if k.lower() in c.lower():
                return c
    return None


def extract_sets(df):
    """Return (positives, clean_negatives, n_ambiguous) as sets of 15-mer strings."""
    col_ep = find_col(df, "Epitope|Name", "epitope")
    col_host = find_col(df, "Host|Name", "host")
    col_class = find_col(df, "MHC Restriction|Class", "class")
    col_qual = find_col(df, "Qualitative")
    col_resp = find_col(df, "Response measured", "response_measured")
    for name, c in [("epitope", col_ep), ("host", col_host), ("class", col_class),
                    ("qualitative", col_qual), ("response", col_resp)]:
        if c is None:
            sys.exit(f"ERROR: could not locate '{name}' column.")
    print(f"Columns: {col_ep} | {col_host} | {col_class} | {col_qual} | {col_resp}")

    mask = (df[col_host].str.contains("Homo sapiens", na=False, case=False)
            & df[col_class].astype(str).str.contains("II", na=False))
    sub = df[mask]
    print(f"MHC-II + human host: {len(sub):,} assay rows")

    # Restrict BOTH classes to the same canonical effector readouts, so the only
    # difference between positive and negative is the measured outcome.
    eff = sub[col_resp].isin(EFFECTOR_RESPONSES)
    pos_rows = sub[eff & sub[col_qual].isin(POSITIVE_QUALITATIVE)]
    neg_rows = sub[eff & sub[col_qual].isin(NEGATIVE_QUALITATIVE)]
    print(f"  effector-assay positive rows: {len(pos_rows):,}")
    print(f"  effector-assay negative rows: {len(neg_rows):,}")

    def peps(rows):
        s = rows[col_ep].astype(str).str.strip().str.upper()
        return {p for p in s if is_valid_peptide(p)}

    positives = peps(pos_rows)
    negatives_raw = peps(neg_rows)

    # A peptide observed positive in ANY study is not a usable negative, even if
    # another study found it negative (different donor, HLA background, context).
    ambiguous = negatives_raw & positives
    clean_negatives = negatives_raw - positives

    print(f"\nunique valid 15-mers  positive: {len(positives):,}"
          f"   negative: {len(negatives_raw):,}")
    print(f"  ambiguous (tested both ways, discarded): {len(ambiguous):,}"
          f" ({100*len(ambiguous)/max(len(negatives_raw),1):.1f}% of negatives)")
    print(f"  clean negatives: {len(clean_negatives):,}")
    return positives, clean_negatives, ambiguous


def main(args):
    t0 = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)

    df = load_iedb(cache)
    positives, negatives, ambiguous = extract_sets(df)
    del df

    if args.limit:
        positives = set(sorted(positives)[:args.limit])
        negatives = set(sorted(negatives)[:args.limit])
        print(f"\n[--limit {args.limit}] positives={len(positives)} negatives={len(negatives)}")

    # ── netMHCIIpan on both classes ──────────────────────────────────────────
    print("\n" + "=" * 78)
    print("NEGATIVES")
    print("=" * 78)
    neg_df = score_peptides(sorted(negatives), DRB1_ALLELES)
    neg_df["label"] = "non_immunogenic"

    print("\n" + "=" * 78)
    print("POSITIVES (for the binding-confound comparison)")
    print("=" * 78)
    pos_df = score_peptides(sorted(positives), DRB1_ALLELES)
    pos_df["label"] = "immunogenic"

    binders = neg_df[neg_df.is_binder].copy()
    # For a genuinely binding-matched comparison BOTH classes must be binders.
    # Positives that netMHCIIpan calls non-binders are predictor misses; keeping
    # them would reintroduce a binding signal (in the opposite direction) and add
    # label noise. The matched pair below is the set to train the classifier on.
    pos_binders = pos_df[pos_df.is_binder].copy()

    neg_df.to_csv(out / "nonimmunogenic.csv", index=False)
    binders.to_csv(out / "nonimmunogenic_binders.csv", index=False)
    pos_df.to_csv(out / "immunogenic_binding.csv", index=False)
    pos_binders.to_csv(out / "immunogenic_binders.csv", index=False)

    # The matched training set: every peptide binds MHC-II; only the outcome differs.
    matched = pd.concat([
        pos_binders.assign(label="immunogenic"),
        binders.assign(label="non_immunogenic"),
    ], ignore_index=True)
    matched.to_csv(out / "binder_matched_dataset.csv", index=False)

    # ── Statistics ───────────────────────────────────────────────────────────
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    pos_bind = 100 * pos_df.is_binder.mean()
    neg_bind = 100 * neg_df.is_binder.mean()

    emit("=" * 78)
    emit("IEDB NEGATIVE ASSAY RECORDS  (roadmap step P1.2)")
    emit("=" * 78)
    emit(f"Immunogenic (effector-positive) 15-mers : {len(pos_df):,}")
    emit(f"Clean negatives (effector-negative)     : {len(neg_df):,}")
    emit(f"Discarded as ambiguous (both outcomes)  : {len(ambiguous):,}")
    emit(f"Binder-matched negatives (%Rank<={RANK_THRESHOLD})    : {len(binders):,}")
    emit("")
    emit("Both classes are drawn from the SAME effector-assay universe")
    emit("(IFN-g release, proliferation, cytotoxicity, activation, ...), so they")
    emit("differ only in the measured outcome.")
    emit("")

    emit("-" * 78)
    emit("THE BINDING CONFOUND -- why the binder-matched set is necessary")
    emit("-" * 78)
    emit(f"{'class':<28}{'n':>10}{'% predicted binders':>22}{'median best %Rank':>20}")
    emit(f"{'immunogenic':<28}{len(pos_df):>10,}{pos_bind:>21.1f}%{pos_df.best_rank.median():>20.2f}")
    emit(f"{'non-immunogenic (all)':<28}{len(neg_df):>10,}{neg_bind:>21.1f}%{neg_df.best_rank.median():>20.2f}")
    emit(f"{'non-immunogenic (binders)':<28}{len(binders):>10,}{100.0:>21.1f}%{binders.best_rank.median():>20.2f}")
    emit("")
    emit(f"Immunogenic peptides are {pos_bind/max(neg_bind,1e-9):.2f}x more likely to be")
    emit("predicted MHC-II binders than the full negative set. A classifier trained on")
    emit("{immunogenic vs all negatives} can therefore score well by learning MHC")
    emit("BINDING rather than IMMUNOGENICITY -- the exact conflation the proposal")
    emit("warns against.")
    emit("")
    emit("The binder-matched negatives are 100% predicted binders by construction.")
    emit("Training on {immunogenic vs binder-matched negatives} removes the binding")
    emit("shortcut: every peptide in both classes binds MHC-II, so the only remaining")
    emit("signal is whether it elicits a T-cell response.")
    emit("")

    emit("-" * 78)
    emit("Per-allele presentation rate (%Rank_EL <= 2)")
    emit("-" * 78)
    emit(f"{'allele':<14}{'immunogenic':>14}{'negatives':>14}")
    for a in DRB1_ALLELES:
        emit(f"{a:<14}{100*(pos_df[a]<=RANK_THRESHOLD).mean():>13.1f}%"
             f"{100*(neg_df[a]<=RANK_THRESHOLD).mean():>13.1f}%")
    emit("")

    emit("-" * 78)
    emit("THE BINDER-MATCHED DATASET  (binder_matched_dataset.csv)")
    emit("-" * 78)
    emit("Both classes restricted to predicted MHC-II binders. Every peptide binds;")
    emit("only the experimental T-cell outcome differs. This is the set to train the")
    emit("independent classifier on.")
    emit("")
    emit(f"immunogenic binders      : {len(pos_binders):,}"
         f"   ({100*len(pos_binders)/max(len(pos_df),1):.1f}% of positives retained)")
    emit(f"non-immunogenic binders  : {len(binders):,}"
         f"   ({100*len(binders)/max(len(neg_df),1):.1f}% of negatives retained)")
    ratio = len(binders) / max(len(pos_binders), 1)
    emit(f"ratio (neg:pos)          : {ratio:.2f}:1")
    if ratio < 0.5 or ratio > 2.0:
        emit("NOTE: imbalanced. Use class weighting or subsampling when training.")
    else:
        emit("Reasonably balanced -- no special handling required.")
    emit("")
    emit("Sanity check -- binding must NOT separate the matched classes:")
    emit(f"  median best %Rank, immunogenic binders     : {pos_binders.best_rank.median():.3f}")
    emit(f"  median best %Rank, non-immunogenic binders : {binders.best_rank.median():.3f}")
    emit(f"  mean n_alleles presented, immunogenic      : {pos_binders.n_alleles_presented.mean():.2f}")
    emit(f"  mean n_alleles presented, non-immunogenic  : {binders.n_alleles_presented.mean():.2f}")
    emit("  If these differ substantially, residual binding signal remains and the")
    emit("  classifier could still exploit it. Consider %Rank-stratified matching.")
    emit("")
    emit("NEXT (P1.4): cluster both classes with MMseqs2 at 30% identity, split")
    emit("train/val/test by cluster (no leakage), train the classifier, verify")
    emit("held-out AUC, and only then score base vs DPO designs.")
    emit("")
    emit(f"Runtime: {time.time()-t0:.0f}s")
    emit("=" * 78)

    (out / "negatives_stats.txt").write_text("\n".join(lines) + "\n")
    print(f"\nWrote:\n  {out/'nonimmunogenic.csv'}\n  {out/'nonimmunogenic_binders.csv'}"
          f"\n  {out/'immunogenic_binding.csv'}\n  {out/'negatives_stats.txt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache",
                    default=os.path.join(PF, "data", "input", "immuno", "mhc_2",
                                         "iedb", "cache", "tcell_full_v3.csv"))
    ap.add_argument("--output",
                    default=os.path.join(PF, "data", "input", "immuno", "mhc_2", "iedb"))
    ap.add_argument("--limit", type=int, default=None,
                    help="cap each class at N peptides (smoke test)")
    main(ap.parse_args())
