#!/bin/bash
#SBATCH --job-name=allelepin2
#SBATCH --array=1-2%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=logs/allelepin2_%A_%a.out
#SBATCH --error=logs/allelepin2_%A_%a.err
#
# ============================================================================
# The two EARLIER-OPERATING-POINT arms of Figure fig:allele, on the SAME pinned
# 24 as job 19. This completes the figure's protein-set matching.
# ============================================================================
#
# WHY THIS EXISTS -- a defect in a figure that is already in the manuscript
# ------------------------------------------------------------------------
# Figure fig:allele (figures/allele_generalization_reanchored.png, generator
# figures/make_fig_allele_reanchored.py) plots THREE models side by side and its
# caption makes explicit cross-model claims: that the two earlier models "track
# each other closely", that the deployed model "transfers more strongly to every
# group", and that HLA-DP "shifts from a net increase under the two earlier models
# (-6% and -3%) to a 21% reduction under the deployed model".
#
# Those three bars come from three DISJOINT protein samples. Measured 2026-08-25
# with tools/check_protein_match.py on the three source CSVs:
#
#   allele_generalisation_dual_lambda0_seed42.csv    n=24  fp d6dea7e3d85b
#   allele_generalisation_v3_creg1.0_seed42.csv      n=24  fp 70824a180824
#   allele_generalisation_deployed_f16b51e6.csv      n=24  fp 410bf179a262
#
#   lambda0 vs lambda1.0 : 0 of 24 in common
#   lambda0 vs deployed  : 5 of 24
#   lambda1.0 vs deployed: 1 of 24
#   common to all three  : 0 proteins            -> exit 1, NOT MATCHED
#
# Verified not to be a naming-convention artifact: all three files use the same
# `1zx4_C` chain-suffix form, and the sets are genuinely different draws.
#
# ROOT CAUSE is the documented one (see 19_allele_gen_pinned.sbatch): each of the
# three runs let select_proteins() re-derive its own stratified draw, and that draw
# depends on the INPUT CSV's own protein-length population, not only on the seed.
# The three arms were scored from three separately sampled eval runs, so seed 42
# produced three different samples. This is the FIFTH instance of the unmatched-
# sample defect in this project and the first one found inside a live figure.
#
# WHAT SURVIVES AND WHAT DOES NOT. Within each file, base-vs-fine-tuned is paired
# on one sample and remains valid -- so each model's own per-group reduction is a
# sound measurement of that model on that sample. What is NOT supported is every
# cross-model statement in the caption and in Section sec:results-allele. The
# pairwise overlaps (0, 5, 1) are far too small to rescue by intersection, so the
# only fix is to re-measure both earlier arms on the pinned set.
#
# THE PINNED SET -- identical to job 19, deliberately
# --------------------------------------------------
#   data/input/protein_lists/allele_gen_matched24.txt   (n=24, file md5
#   76b488e337905b3e32a789eb45bc4d59; protein_set fingerprint 9cae3da74828 as
#   reported by check_protein_match.py on the job 19 outputs)
#
# Pre-flight already done for BOTH inputs below: all 24 pinned proteins present in
# each CSV, no protein carrying two different protein_length values within the list
# (so no base/fine-tuned assembly mismatch inside the sample), and no leakage of the
# four excluded proteins (2pt5_B, 3hmt_A, 4i4y_C, 4nc7_B).
#
# TASK MAP
#   1  54e1642c  lambda=0,   lr 3e-7  -> allele_generalisation_pinned24_lambda0
#   2  7e908919  lambda=1.0, lr 3e-7  -> allele_generalisation_pinned24_lambda1
#
#   Submit:   sbatch slurm/20_allele_gen_pinned_earlier.sbatch
#   Dry run:  CAPE_DRYRUN=1 bash slurm/20_allele_gen_pinned_earlier.sbatch
#
# check_train_flags.py does NOT apply here: this trains nothing, so there is no
# hashed hyperparameter to pass explicitly.
#
# RUNTIME. Same shape as job 19 (21 alleles x 24 proteins x 2 arms x 4 sequences),
# which measured ~75-105 min/task. 16 h is generous headroom.
#
# AFTERWARDS -- do not skip, this is the entire point of the job:
#   python CAPE_MPNN/tools/check_protein_match.py \
#     CAPE_MPNN/data/output/allele_generalisation_pinned24_lambda0.csv \
#     CAPE_MPNN/data/output/allele_generalisation_pinned24_lambda1.csv \
#     CAPE_MPNN/data/output/allele_generalisation_pinned24_mhc2only.csv \
#     CAPE_MPNN/data/output/allele_generalisation_pinned24_deployed.csv
#   Exit 0 across all four = Figure fig:allele can be redrawn as a genuinely
#   matched four-model comparison. Anything else = do NOT compare the summaries.
#
# NOTE the redrawn figure gains a fourth arm (abec4d2c, the class-II-only
# checkpoint) whose HLA-DP result is the reason any of this was looked at. The
# generator make_fig_allele_reanchored.py reads three hardcoded summary JSONs and
# will need a new version -- do not edit it in place, per project convention.
# ============================================================================
# FIRST executable line, before `set -e` can abort anything -- if a future failure
# still yields an empty .out, the task never started (scheduler/node level) and the
# script body is not the suspect. See the activate_env.sh nvidia-smi fix.
echo "[preamble] task=${SLURM_ARRAY_TASK_ID:-none} job=${SLURM_JOB_ID:-none} node=$(hostname) pwd=$(pwd) date=$(date)"

set -euo pipefail

: "${PF:=${CAPE_ROOT:?set CAPE_ROOT to the project root (see README)}}"
export PF
export PATH="$PF/CAPE_MPNN/external/programs/bin:$PATH"

# No --gres=gpu here, so activate_env.sh must not assert CUDA.
export CAPE_REQUIRE_GPU=0

TASK="${SLURM_ARRAY_TASK_ID:-1}"
case "$TASK" in
  1) HASH=54e1642c; TAG=pinned24_lambda0 ;;
  2) HASH=7e908919; TAG=pinned24_lambda1 ;;
  *) echo "FATAL: bad array id $TASK" >&2; exit 1 ;;
esac

EP=200
PLIST="data/input/protein_lists/allele_gen_matched24.txt"
IN="data/output/$(printf 'eval_deimmunisation_pwmtraj_%s_ep%03d.csv' "$HASH" "$EP")"

if [ -n "${CAPE_DRYRUN:-}" ]; then
  for t in 1 2; do
    case "$t" in
      1) h=54e1642c; g=pinned24_lambda0 ;;
      2) h=7e908919; g=pinned24_lambda1 ;;
    esac
    i="data/output/$(printf 'eval_deimmunisation_pwmtraj_%s_ep%03d.csv' "$h" "$EP")"
    printf "task %s  %-9s tag=%-22s in=%s %s\n" "$t" "$h" "$g" "$i" \
      "$([ -e "$PF/CAPE_MPNN/$i" ] && echo OK || echo MISSING)"
  done
  echo "protein_list: $PLIST $([ -e "$PF/CAPE_MPNN/$PLIST" ] && echo OK || echo MISSING)"
  exit 0
fi

source "$PF/slurm/activate_env.sh"
cd "$PF/CAPE_MPNN"; mkdir -p "$PF/slurm/logs"

echo "[$(date)] task ${TASK}: ${HASH} allele-gen PINNED  in=${IN}  tag=${TAG}"

if [ -e "data/output/allele_generalisation_${TAG}.csv" ]; then
  echo "  [skip] allele_generalisation_${TAG}.csv present"; exit 0
fi

# --protein_list is the whole point: it pins the sample instead of re-deriving it
# from this CSV's own length distribution. The tool asserts every requested protein
# is present rather than silently dropping any, and writes a protein_set_md5 into
# its summary and json so the match is verifiable after the fact.
PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/test_allele_generalisation.py \
    --in_csv "${IN}" --out_tag "${TAG}" --protein_list "${PLIST}" --k_seqs 4

echo "[$(date)] done: task ${TASK} ${HASH} allele-gen pinned"
