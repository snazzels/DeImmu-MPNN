#!/usr/bin/env python
"""Build PRODUCTION register-anchored 9-mer MHC-II PWMs for DPO training.

This is the training-format counterpart of the validation-only build_register_pwm.py.
It produces, per allele, a 20x9 log-PWM over netMHCIIpan's
reported binding CORE plus a calibrated presentation threshold, in the exact directory
layout Mhc2PredictorPwm expects (pwm/{allele}/{allele}-9_log.csv + threshold-{allele}-9.txt).
Passing lengths=[9] to Mhc2PredictorPwm then scores every 9-mer window of a sequence
against the core PWM -- i.e. register-anchored scoring (each window is a candidate core).

Method (deliberately identical to MHC-II_rank_peptides.py, only the input peptides differ):
  1. Recover each allele's ~9-13k presenters (%Rank_EL <= 2) from the cached
     ranks/{allele}.csv produced by the original 15-mer build (no re-scoring of 1M peptides).
  2. Re-run netMHCIIpan on just those presenters to recover their 9-mer Core (the offset
     was computed but discarded by the original run). Forward cores only (inverted binding
     is ~0% on DR, Track 18).
  3. Build the PWM with the SAME kit.bioinf.pwm routines (plain log-frequency, no pseudocount)
     used for the 15-mer PWM, over the aligned 9-mer cores.
  4. Calibrate the threshold as the 98th percentile of 9-mer-window scores over the SAME
     cached random peptides (windowed to 9-mers), matching the 2%-rank presentation definition.

Output: data/input/immuno/mhc_2/pwm_register/pwm/{allele}/  (a NEW dir; nothing overwritten)
"""
import os, sys, subprocess, tempfile, argparse
import numpy as np
import pandas as pd

PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # CAPE_MPNN/
sys.path.insert(0, os.path.join(PF, "libs"))
from kit.bioinf.pwm import count_amino_acid_occurrences, calc_PWMs, save_PWMs

ALLELES = "DRB1_0101+DRB1_0301+DRB1_0401+DRB1_0701+DRB1_1101+DRB1_1501".split("+")
RANKS_DIR = os.path.join(PF, "data", "input", "immuno", "mhc_2", "pwm", "ranks")
OUT_DIR   = os.path.join(PF, "data", "input", "immuno", "mhc_2", "pwm_register")
LIMIT_RANK = 2.0
THRESHOLD_PERCENTILE = 100.0 - LIMIT_RANK    # 98 -> top 2%
CORE_LEN = 9
BATCH = 5000


def netmhciipan_cores(peptides, allele):
    """Return {peptide: (rank, core, inverted)} keeping the best (lowest-rank) line."""
    res = {}
    for i in range(0, len(peptides), BATCH):
        batch = peptides[i:i + BATCH]
        with tempfile.TemporaryDirectory() as td:
            pf = os.path.join(td, "in.pep"); open(pf, "w").write("\n".join(batch))
            out = subprocess.run(["netMHCIIpan", "-inptype", "1", "-f", pf, "-a", allele],
                                 capture_output=True, check=True)
            for line in out.stdout.decode().splitlines():
                p = line.split()
                if len(p) < 10:
                    continue
                try:
                    int(p[0])
                except ValueError:
                    continue
                pep, core, inv, rank = p[2], p[4], int(p[6]), float(p[9])
                if pep not in res or rank < res[pep][0]:
                    res[pep] = (rank, core, inv)
        print(f"    netMHCIIpan {min(i + BATCH, len(peptides))}/{len(peptides)}", end="\r", flush=True)
    print()
    return res


def score_windows_9mer(peptides15, log_pwm_df):
    """Score every 9-mer window of each 15-mer against the 9-PWM; return all window scores."""
    aa_to_idx = {aa: i for i, aa in enumerate(log_pwm_df.index)}
    pwm = log_pwm_df.values                     # (20, 9)
    windows = []
    for p in peptides15:
        for o in range(len(p) - CORE_LEN + 1):
            windows.append(p[o:o + CORE_LEN])
    idx = np.array([[aa_to_idx.get(c, -1) for c in w] for w in windows], dtype=np.int32)
    unknown = idx < 0
    idx_safe = np.where(unknown, 0, idx)
    col = np.broadcast_to(np.arange(CORE_LEN, dtype=np.int32), idx.shape)
    scores = pwm[idx_safe, col].sum(axis=1) - unknown.sum(axis=1) * 1e9
    return scores


def main(a):
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    for allele in ALLELES:
        print(f"\n=== {allele} ===")
        ranks = pd.read_csv(os.path.join(RANKS_DIR, f"{allele}.csv"))
        col = [c for c in ranks.columns if c != "peptide"][0]
        presenters = ranks.loc[ranks[col] <= LIMIT_RANK, "peptide"].tolist()
        print(f"  {len(presenters)} presenters (<= {LIMIT_RANK}% rank)")

        sc = netmhciipan_cores(presenters, allele)
        cores = [core for (_, core, inv) in sc.values() if inv == 0 and len(core) == CORE_LEN]
        print(f"  {len(cores)} forward 9-mer cores ({len(sc) - len(cores)} dropped: inverted/malformed)")

        counts = count_amino_acid_occurrences(cores)          # {9: DataFrame(20x9)}
        pwms, log_pwms = calc_PWMs(counts)
        save_PWMs(allele, pwms, log_pwms, OUT_DIR)            # writes pwm/{allele}/{allele}-9[_log].csv

        # info-content sanity (bits above uniform) — expect peaks at P1/P4/P6/P9
        freq = pwms[CORE_LEN].values
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.nansum(np.where(freq > 0, freq * np.log2(freq), 0), axis=0)
        ic = np.log2(20) - ent
        print(f"  info content (bits/pos): " + " ".join(f"{v:.2f}" for v in ic) +
              f"  | total {ic.sum():.2f}, peak at P{int(np.argmax(ic)) + 1}")

        # threshold: 98th percentile of 9-mer-window scores over a sample of the cached random peptides
        sample = ranks["peptide"].sample(n=min(300000, len(ranks)), random_state=0).tolist()
        wscores = score_windows_9mer(sample, log_pwms[CORE_LEN])
        threshold = float(np.percentile(wscores, THRESHOLD_PERCENTILE))
        tfile = os.path.join(OUT_DIR, "pwm", allele, f"threshold-{allele}-{CORE_LEN}.txt")
        with open(tfile, "w") as fh:
            fh.write(str(threshold))
        frac = float((wscores > threshold).mean())
        print(f"  threshold(9) = {threshold:.4f}  ({100*frac:.2f}% of {len(wscores)} random 9-mer windows above)")

    print(f"\nDone. Register PWMs written under {OUT_DIR}/pwm/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    main(ap.parse_args())
