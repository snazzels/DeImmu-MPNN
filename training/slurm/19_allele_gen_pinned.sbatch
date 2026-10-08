#!/bin/bash
#SBATCH --job-name=allelepin
#SBATCH --array=1-2%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=logs/allelepin_%A_%a.out
#SBATCH --error=logs/allelepin_%A_%a.err
#
# RESOURCES: deliberately identical to 16_netmhc_matched_ablation.sbatch and
# 18_ablation_coverage_threshold.sbatch, both of which run this same tool on this
# same cluster without trouble. The first submission of this script (519073) asked
# for 4 CPUs / 16 G / 8 h and carried --exclude=t38cn047, and both tasks produced
# ZERO-BYTE .out AND .err -- i.e. they died before activate_env.sh could echo its
# first line, so no part of the body ran. Rather than debug a bespoke resource
# request blind, this now matches the known-good jobs exactly. --exclude is dropped
# for the same reason: it was the only other structural difference from them.
#
# ============================================================================
# MATCHED allele-generalisation re-run on an EXPLICITLY PINNED protein set.
# ============================================================================
#
# WHY THIS EXISTS
# ---------------
# Job 18 (517851) tasks 1 and 2 were intended as a matched pair: the same 24
# proteins, one model each. They were not matched. Both read the same pinned-98
# eval CSVs and both passed --n_proteins 24, but the draws shared only 18 of 24
# (fingerprints 1c979f94074d vs 920364c70e3a, tools/check_protein_match.py exit 1).
#
# ROOT CAUSE: test_allele_generalisation.py's select_proteins() builds its
# name->length map by dict comprehension over rows, so the LAST row wins. 3hmt_A's
# last row is a 202-residue homodimer in the abec4d2c CSV but a 101-residue monomer
# in the f16b51e6 one. That single value moves the length quartiles, which moves
# every bucket boundary, which rewrites the entire stratified draw. A seeded RNG
# does not protect against this -- only an explicit list does.
#
# The 18-protein intersection was rescued by hand and already gives the answer
# (HLA-DP 37% vs 11%, paired Wilcoxon p=0.0016, 16/18, survives Bonferroni across
# the 5 groups and leave-one-out). This job is NOT needed for that result to hold.
# It exists so the manuscript can say "both models on the same pinned 24" instead
# of "we intersected post hoc to 18", which is the version a reviewer accepts.
#
# THE PINNED SET
# --------------
#   data/input/protein_lists/allele_gen_matched24.txt   (md5 9cae3da74828, n=24)
# Drawn from the 94 of the pinned-98 proteins that are FREE of the base/fine-tuned
# assembly mismatch -- 2pt5_B, 3hmt_A, 4i4y_C and 4nc7_B are excluded, because for
# those the two arms are scored on different molecules (monomer vs '/'-joined
# homodimer) and their per-protein reductions are artifacts. Stratified by monomer
# length into quartiles, 6 per quartile, seed 42.
#
# TASK MAP
#   1  abec4d2c  (class-II-only checkpoint)  -> allele_generalisation_pinned24_mhc2only
#   2  f16b51e6  (deployed)                  -> allele_generalisation_pinned24_deployed
#
#   Submit:   sbatch --array=1-2 slurm/19_allele_gen_pinned.sbatch
#   Dry run:  CAPE_DRYRUN=1 bash slurm/19_allele_gen_pinned.sbatch
#
# RUNTIME. 21 alleles over 24 proteins x 2 arms x 4 sequences. Job 18 measured
# 190-265 s/allele, i.e. ~75-105 min/task; 8 h is generous headroom.
#
# AFTERWARDS -- this is the whole point, so do not skip it:
#   python CAPE_MPNN/tools/check_protein_match.py \
#     CAPE_MPNN/data/output/allele_generalisation_pinned24_mhc2only.csv \
#     CAPE_MPNN/data/output/allele_generalisation_pinned24_deployed.csv
#   Exit 0 = matched. Anything else means do NOT compare the two summaries.
# ============================================================================
# FIRST executable line, before `set -e` can abort anything and before any variable
# is dereferenced. If a future failure still yields an empty .out, the task never
# started at all (scheduler/node level) and the script itself is not the suspect --
# that distinction is exactly what submission 519073 could not be diagnosed without.
echo "[preamble] task=${SLURM_ARRAY_TASK_ID:-none} job=${SLURM_JOB_ID:-none} node=$(hostname) pwd=$(pwd) date=$(date)"

set -euo pipefail

: "${PF:=${CAPE_ROOT:?set CAPE_ROOT to the project root (see README)}}"
export PF
export PATH="$PF/CAPE_MPNN/external/programs/bin:$PATH"

# No --gres=gpu here, so activate_env.sh must not assert CUDA (job 517191 lost
# 17/19 tasks to exactly that).
export CAPE_REQUIRE_GPU=0

TASK="${SLURM_ARRAY_TASK_ID:-1}"
case "$TASK" in
  1) HASH=abec4d2c; TAG=pinned24_mhc2only ;;
  2) HASH=f16b51e6; TAG=pinned24_deployed ;;
  *) echo "FATAL: bad array id $TASK" >&2; exit 1 ;;
esac

EP=200
PLIST="data/input/protein_lists/allele_gen_matched24.txt"
IN="data/output/$(printf 'eval_deimmunisation_pwmtraj_%s_ep%03d.csv' "$HASH" "$EP")"

if [ -n "${CAPE_DRYRUN:-}" ]; then
  for t in 1 2; do
    case "$t" in
      1) h=abec4d2c; g=pinned24_mhc2only ;;
      2) h=f16b51e6; g=pinned24_deployed ;;
    esac
    i="data/output/$(printf 'eval_deimmunisation_pwmtraj_%s_ep%03d.csv' "$h" "$EP")"
    printf "task %s  %-9s tag=%-20s in=%s %s\n" "$t" "$h" "$g" "$i" \
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
