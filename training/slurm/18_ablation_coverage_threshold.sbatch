#!/bin/bash
#SBATCH --job-name=abl_cov_thr
#SBATCH --array=1-4%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=logs/ablcovthr_%A_%a.out
#SBATCH --error=logs/ablcovthr_%A_%a.err
# =========================== EDIT FOR YOUR CLUSTER ===========================
# CPU-ONLY. Both analyses re-score sequences that ALREADY EXIST in the epoch-200 sweep
# CSVs. No GPU, no design generation, no training.
# ----------------------------------------------------------------------------
# ⚠️ SUBMIT FROM THE PROJECT ROOT (--output=logs/... is relative).
#   cd ${CAPE_ROOT} && sbatch slurm/18_ablation_coverage_threshold.sbatch
#
# ---------------------------------------------------------------------------
# TWO REMAINING QUESTIONS ABOUT THE CHECKPOINT v35 RECOMMENDS.
# manuscript v35 names the MHC-II-only checkpoint as the targeted option for
# anti-drug-antibody-limited applications. It now has PWM trajectories, netMHCIIpan and
# netMHCpan (job 16), MixMHC2pred (29.5% +- 19.2, 94/98), charge/developability, and
# Boltz-2 self-consistency (dTM -0.023, p=0.411). Two properties the paper claims for the
# DEPLOYED model have never been measured for it:
#
#   (A) ALLELE COVERAGE. The paper's coverage result -- transfer to unseen HLA-DRB1, the
#       DRB3/4/5 paralogs and HLA-DQ, with HLA-DP the one genuine gap -- is a headline
#       feature. Whether it survives when the class-I objective is removed is unknown.
#   (B) %RANK THRESHOLD ROBUSTNESS. Reviewer R2 P2-7 was answered for the deployed model
#       (43.5/37.9/37.8/35.1% at cutoffs 1/2/5/10%). If the ablation is recommended, the
#       same question applies to it.
#
# ---------------------------------------------------------------------------
# WHY THE DEPLOYED MODEL IS RE-RUN HERE TOO (tasks 2 and 4).
# This is the same protein-set mismatch that has now bitten on every orthogonal
# validator. The existing deployed-model results were computed on DIFFERENT protein
# samples than the ablation's:
#   * allele generalisation -> allele_generalisation_v3_creg1.0_seed42 / _deployed_f16b51e6
#   * rank sensitivity      -> validate_netmhciipan_rank_sensitivity_f16b51e6 (built from
#                              eval_deimmunisation_sweep_f16b51e6.csv)
# Comparing the ablation against those would repeat the error that made task 2 of job 16
# necessary on the class-I axis and tasks 11-12 necessary on MixMHC2pred. Both arms are
# therefore scored here from their epoch-200 PINNED-98 CSVs.
#
# TASK MAP
#   1  abec4d2c  allele generalisation   <-- the key one
#   2  f16b51e6  allele generalisation   matched baseline on the pinned 98
#   3  abec4d2c  %Rank sensitivity       <-- the key one
#   4  f16b51e6  %Rank sensitivity       matched baseline (OPTIONAL: the published
#                                        deployed answer already exists and stands on its
#                                        own; this only makes the CROSS-MODEL delta clean.
#                                        Use --array=1-3 to skip it and save ~6 h CPU.)
#
# RUNTIME. netMHCIIpan dominates both. Allele generalisation scores 21 alleles over a
# 24-protein length-stratified subsample at 4 designs per arm (~48k unique 15-mers,
# measured ~2 h for the deployed run). Rank sensitivity scores 6 alleles over the full
# 98-protein set (~470k unique 15-mers) but in ONE pass per allele with ranks retained,
# so all four cutoffs come from that single pass -- budget ~6-8 h. All four run at once.
#
#   Core only:  sbatch --array=1-3 18_ablation_coverage_threshold.sbatch
#   Dry run:    CAPE_DRYRUN=1 bash slurm/18_ablation_coverage_threshold.sbatch
set -euo pipefail

: "${PF:=${CAPE_ROOT:?set CAPE_ROOT to the project root (see README)}}"
export PF
export PATH="$PF/CAPE_MPNN/external/programs/bin:$PATH"
# activate_env.sh asserts CUDA and is SOURCED, so without this a CPU-only task dies on a
# GPU-less node (job 517191, 17/19 tasks).
export CAPE_REQUIRE_GPU=0

TASK="${SLURM_ARRAY_TASK_ID:-1}"
case "$TASK" in
  1) HASH=abec4d2c; KIND=allele ; LABEL=reanchor_mhc2only ;;
  2) HASH=f16b51e6; KIND=allele ; LABEL=deployed          ;;
  3) HASH=abec4d2c; KIND=rank   ; LABEL=reanchor_mhc2only ;;
  4) HASH=f16b51e6; KIND=rank   ; LABEL=deployed          ;;
  *) echo "FATAL: bad array id $TASK" >&2; exit 1 ;;
esac
EP=200
IN="data/output/$(printf 'eval_deimmunisation_pwmtraj_%s_ep%03d.csv' "$HASH" "$EP")"
# Tag carries the protein set so these can never be confused with the differently-sampled
# prior results they are meant to replace for comparison purposes.
TAG="${LABEL}_pinned98_ep${EP}"

if [ -n "${CAPE_DRYRUN:-}" ]; then
  echo "task $TASK: ${HASH} (${LABEL}) ${KIND}"
  echo "  in : $PF/CAPE_MPNN/${IN}"
  ls -la "$PF/CAPE_MPNN/${IN}" >/dev/null 2>&1 && echo "  input OK" || echo "  *** INPUT MISSING ***"
  echo "  tag: ${TAG}"
  exit 0
fi

_ok=0
for _t in 1 2 3; do
  if source "$PF/slurm/activate_env.sh"; then _ok=1; break; fi
  echo "[ablcovthr] activate_env attempt ${_t} failed; retry in 60 s" >&2; sleep 60
done
[ "$_ok" = "1" ] || { echo "[ablcovthr] FATAL: env activation failed 3x" >&2; exit 1; }

cd "$PF/CAPE_MPNN"; mkdir -p "$PF/slurm/logs"
# netMHCIIpan's vendor script falls back to $NMHOME/tmp when TMPDIR is unset, and $NMHOME
# is on the NFS mount, which intermittently EPERMs under concurrent write load.
export TMPDIR="${SLURM_TMPDIR:-/tmp}/ablcovthr_${SLURM_JOB_ID:-0}_${TASK}"
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT

[ -f "$IN" ] || { echo "FATAL: missing input $IN" >&2; exit 1; }
L=$(wc -l < "$IN")
[ "$L" -eq 1961 ] || { echo "FATAL: $IN has $L lines, expected 1961" >&2; exit 1; }
command -v netMHCIIpan >/dev/null 2>&1 || { echo "FATAL: netMHCIIpan not on PATH" >&2; exit 1; }

echo "[$(date)] task ${TASK}: ${HASH} (${LABEL}) ${KIND}  in=${IN}  tag=${TAG}"

if [ "$KIND" = "allele" ]; then
  if [ -e "data/output/allele_generalisation_${TAG}.csv" ]; then
    echo "  [skip] allele_generalisation_${TAG}.csv present"; exit 0
  fi
  # --n_proteins 24 / --k_seqs 4 match the deployed run this is compared against
  # (allele_generalisation_v3_creg1.0_seed42: "24 proteins selected", "4/model/protein").
  # The tool refuses to overwrite an existing out_tag, so the tag must stay distinct.
  PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/test_allele_generalisation.py \
      --in_csv "${IN}" --out_tag "${TAG}" --n_proteins 24 --k_seqs 4
else
  if [ -e "data/output/validate_netmhciipan_rank_sensitivity_${TAG}_summary.txt" ]; then
    echo "  [skip] rank sensitivity ${TAG} present"; exit 0
  fi
  # tools/rank_sensitivity.py is the parameterised rewrite of the model-specific scratch
  # script data/output/rank_sensitivity_f16b51e6.py, which is left untouched because it
  # produced the published figures. The rewrite was verified against the original's
  # aggregation on stubbed predictor output: identical to 6 decimal places.
  PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/rank_sensitivity.py \
      --in_csv "${IN}" --out_tag "${TAG}" --label "${LABEL} (pinned 98, epoch ${EP})"
fi

echo "[$(date)] done: task ${TASK} ${HASH} ${KIND}"
