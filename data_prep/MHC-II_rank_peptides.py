#!/usr/bin/env python
"""Build MHC-II PWMs from random peptides ranked by netMHCIIpan 4.3.

Equivalent of MHC-I_rank_peptides.py but for MHC class II (15-mer peptides,
netMHCIIpan instead of netMHCpan, DRB1_0101-style allele names).

Usage example:
    MHC-II_rank_peptides.py \
        --alleles DRB1_0101+DRB1_0301+DRB1_0401+DRB1_0701+DRB1_1101+DRB1_1501 \
        --peptides_per_length 1000000 \
        --output ${PF}/data/input/immuno/mhc_2/pwm
"""

import os
import re
import sys
import argparse
import traceback
import subprocess
import tempfile

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from kit.path import join
from kit.log import setup_logger
from kit.bioinf import generate_random_aa_seq
from kit.data import str_to_file, file_to_str
from kit.bioinf.pwm import count_amino_acid_occurrences, calc_PWMs, save_PWMs


LIMIT_RANK = 2.0   # %Rank_EL; peptides below this are considered MHC-II presented
BATCH_SIZE = 5000  # peptides per netMHCIIpan call; balances progress granularity vs overhead

# Percentile of random-peptide PWM scores used as the presentation threshold.
# 98.0 → top 2% of random peptides are flagged as potentially presented (matches LIMIT_RANK).
THRESHOLD_PERCENTILE = 100.0 - LIMIT_RANK


def run_netmhciipan(peptides, allele):
    """Run netMHCIIpan on peptides for one allele in batches. Returns {peptide: %Rank_EL}."""
    ranks = {}
    batches = [peptides[i:i + BATCH_SIZE] for i in range(0, len(peptides), BATCH_SIZE)]
    for batch in tqdm(batches, desc=f"  netMHCIIpan {allele}", unit="batch", leave=False):
        with tempfile.TemporaryDirectory() as tmpdir:
            pep_file = os.path.join(tmpdir, "input.pep")
            with open(pep_file, "w") as fh:
                fh.write("\n".join(batch))
            result = subprocess.run(
                ["netMHCIIpan", "-inptype", "1", "-f", pep_file, "-a", allele],
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"netMHCIIpan failed for allele {allele}:\n"
                    + result.stderr.decode("utf-8")
                )
            ranks.update(_parse_output(result.stdout.decode("utf-8").splitlines()))
    return ranks


def _parse_output(lines):
    """Parse netMHCIIpan stdout lines → {peptide: best_%Rank_EL}.

    Data lines look like (space-delimited):
      Pos  MHC  Peptide  Of  Core  Core_Rel  Inverted  Identity  Score_EL  %Rank_EL  Exp_Bind  [BindLevel]
        1  DRB1_0101  AAAGAEAGKATTE  1  AAGAEAGKA  0.740  0  Sequence  0.000143  72.71  0.000
    """
    ranks = {}
    for line in lines:
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            int(parts[0])
        except ValueError:
            continue
        peptide = parts[2]
        rank_el = float(parts[9])
        if peptide not in ranks or rank_el < ranks[peptide]:
            ranks[peptide] = rank_el
    return ranks


def _score_peptides_vectorised(peptides, log_pwm_df):
    """Score all peptides against a log-PWM DataFrame. Returns a numpy array of scores.

    Uses numpy advanced indexing: O(N*L) time, O(N*L) memory. Fast for 1M peptides.
    Unknown amino acids receive a score of -1e9 per position.
    """
    aa_to_idx = {aa: i for i, aa in enumerate(log_pwm_df.index)}
    pwm = log_pwm_df.values  # (20, L)
    n, L = len(peptides), pwm.shape[1]

    pep_idx = np.array([[aa_to_idx.get(c, -1) for c in p] for p in peptides], dtype=np.int32)
    unknown = pep_idx < 0
    pep_idx_safe = np.where(unknown, 0, pep_idx)

    col_idx = np.broadcast_to(np.arange(L, dtype=np.int32), (n, L))
    scores = pwm[pep_idx_safe, col_idx].sum(axis=1)
    scores -= unknown.sum(axis=1) * 1e9  # penalise unknown AAs
    return scores


def _save_score_thresholds(allele, peptides, log_pwms, folder):
    """Compute PWM score threshold as the THRESHOLD_PERCENTILE-th percentile across all peptides.

    Saves one threshold file per length to pwm/{allele}/threshold-{allele}-{length}.txt.
    The threshold is the score that exactly THRESHOLD_PERCENTILE % of random peptides fall below,
    so only the top LIMIT_RANK % are considered "presented."
    """
    allele_for_path = allele.replace("*", "_")
    for length, log_pwm_df in tqdm(log_pwms.items(), desc=f"  thresholds {allele}", leave=False):
        length_peptides = [p for p in peptides if len(p) == length]
        scores = _score_peptides_vectorised(length_peptides, log_pwm_df)
        threshold = float(np.percentile(scores, THRESHOLD_PERCENTILE))
        threshold_file = os.path.join(
            folder, "pwm", allele_for_path, f"threshold-{allele_for_path}-{length}.txt"
        )
        with open(threshold_file, "w") as fh:
            fh.write(str(threshold))
        print(f"  {allele} length={length}: threshold={threshold:.4f} "
              f"(top {100 - THRESHOLD_PERCENTILE:.1f}% of {len(length_peptides)} peptides)")


def _process_presented_peptides(allele, df_ranks, folder, compute_threshold=True):
    peptides_presented = [
        p
        for p, row in tqdm(df_ranks.iterrows(), "get presented peptides", leave=False)
        if pd.notna(row[allele]) and row[allele] <= LIMIT_RANK
    ]
    if len(peptides_presented) < 10:
        print(f"WARNING: only {len(peptides_presented)} presented peptides for {allele} — PWM may be unreliable")

    counts = count_amino_acid_occurrences(peptides_presented)
    pwms, log_pwms = calc_PWMs(counts)
    save_PWMs(allele, pwms, log_pwms, folder)

    if compute_threshold:
        _save_score_thresholds(allele, list(df_ranks.index), log_pwms, folder)


def main(_args):
    lengths = [int(l) for l in _args.lengths.split("+")]
    alleles = _args.alleles.split("+")
    tasks = _args.tasks.split("+")

    if _args.output:
        folder = join(_args.output)
    else:
        folder = join(
            os.environ["DATA"],
            "processed", "MHC_class_II", "random_peptides",
            _args.lengths, f"random_{_args.peptides_per_length}",
        )

    join(folder, "pwm")
    finished_alleles_file = os.path.join(folder, "finished_alleles.txt")
    finished_alleles = set(file_to_str(finished_alleles_file).split("\n"))

    skip_finished = tasks not in [["stats"], ["threshold"]]
    eval_alleles = (
        [a for a in alleles if a not in finished_alleles]
        if skip_finished
        else alleles
    )

    random_peptides_file = os.path.join(folder, "random_peptides.txt")
    if os.path.exists(random_peptides_file):
        print("Read existing random_peptides from disk")
        random_peptides = file_to_str(random_peptides_file).split("\n")
    else:
        print("Generate new random_peptides")
        random_peptides = []
        for length in tqdm(lengths, "Generate random peptides"):
            random_peptides += [
                generate_random_aa_seq(length) for _ in range(_args.peptides_per_length)
            ]
        str_to_file("\n".join(random_peptides), random_peptides_file)

    df_ranks = pd.DataFrame(index=random_peptides, columns=["length"])
    df_ranks.index.name = "peptide"
    df_ranks["length"] = [len(p) for p in random_peptides]

    pbar = tqdm(eval_alleles, "Alleles")
    for allele in pbar:
        allele_for_path = allele.replace("*", "_")
        rank_file = join(folder, "ranks", f"{allele_for_path}.csv")

        if "rank" in tasks:
            if os.path.exists(rank_file):
                choice = input(f"{rank_file} exists. Overwrite? (y/n) ")
                if choice.strip().lower() != "y":
                    print(f"Skipping rank for {allele}")
                    continue

            pbar.set_description(f"{allele} - rank")
            ranks = run_netmhciipan(random_peptides, allele)
            df_ranks[allele] = [ranks.get(p, np.nan) for p in random_peptides]
            df_ranks[[allele]].sort_index().to_csv(rank_file)

        if "pwm" in tasks:
            pbar.set_description(f"{allele} - pwm")
            if allele not in df_ranks.columns:
                df_ranks = df_ranks.join(
                    pd.read_csv(rank_file).set_index("peptide"), how="left"
                )
            _process_presented_peptides(allele, df_ranks, folder, compute_threshold="threshold" in tasks)
            str_to_file(f"{allele}\n", finished_alleles_file, append=True)

        if "threshold" in tasks and "pwm" not in tasks:
            # Run threshold computation on already-built PWMs without rebuilding them.
            pbar.set_description(f"{allele} - threshold")
            allele_for_path = allele.replace("*", "_")
            pwm_subdir = os.path.join(folder, "pwm", allele_for_path)
            log_pwms = {}
            for length in lengths:
                log_pwm_file = os.path.join(pwm_subdir, f"{allele_for_path}-{length}_log.csv")
                if not os.path.exists(log_pwm_file):
                    print(f"WARNING: {log_pwm_file} not found — skipping length {length}")
                    continue
                df = pd.read_csv(log_pwm_file).set_index("AA")
                df.columns = [int(c) for c in df.columns]
                log_pwms[length] = df
            if log_pwms:
                _save_score_thresholds(allele, random_peptides, log_pwms, folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--lengths", type=str, default="15",
        help="peptide lengths to evaluate, e.g. 15 or 13+15+17",
    )
    parser.add_argument(
        "--peptides_per_length", type=int, default=100,
        help="number of random peptides per length (use 1000000 for production)",
    )
    parser.add_argument(
        "--alleles", type=str, default="",
        help="alleles in DRB1_0101 format, e.g. DRB1_0101+DRB1_0301+DRB1_0401",
    )
    parser.add_argument(
        "--tasks", type=str, default="rank+pwm+threshold",
        help="rank: call netMHCIIpan; pwm: build PWMs from top presenters; threshold: save score percentile cutoff",
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="output directory (defaults to $DATA/processed/MHC_class_II/...)",
    )
    args = parser.parse_args()

    try:
        setup_logger()
        main(args)
    except Exception:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        import pdb
        pdb.post_mortem(tb)
