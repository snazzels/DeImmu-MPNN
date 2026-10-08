#!/bin/bash
#SBATCH --job-name=matched_eval
#SBATCH --array=1-5%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/matched_%A_%a.out
#SBATCH --error=logs/matched_%A_%a.err
# =========================== EDIT FOR YOUR CLUSTER ===========================
#SBATCH --constraint="rtx5090|rtx4090|rtx6000|rtxa5000"
# ----------------------------------------------------------------------------
# Re-evaluate the charge-regulariser arms on ONE FIXED protein set.
#
# WHY. --seed does not determinise which proteins get evaluated: the set is
# whatever the first n_proteins of the test loader happen to be, and something in
# that path is unseeded. Empirically, all 15 configs of 02_sweep_eval.sbatch ran
# with --seed 42 and still drew 15 DIFFERENT samples from a pool of >=426
# proteins (pairwise overlap 33-45/100). The lambda=0 and lambda=1.0 arms of
# Table tab:sweep share only 19 of 100 proteins, so their reductions are not
# comparable arm-to-arm and no paired test is possible.
#
# FIX. evaluate_deimmunisation.py --protein_list pins the set by NAME, loading each
# requested chain directly via loader_pdb() (every chain has its own file), and
# ABORTS if any cannot be loaded -- a short match can never be silently reported
# as "the matched set". Filtering a sampled pool cannot work instead: one pass
# yields only ~205 chains out of ~1553 test clusters, and which ones varies.
#
# THE SET. 98 chains: the deployed model's own evaluated chains minus two
# (1y4m_A, 5jkq_A) that load unstably, because loader_pdb() picks at random among
# a PDB entry's assemblies so a borderline protein's length sometimes exceeds
# MAX_LEN=500. Verified 98/98 on three consecutive passes.
#
#   task 1  lambda=0    54e1642c
#   task 2  lambda=0.5  057975d5   (weights verified md5-identical to the
#                                   057975d5_v2_creg0.5/ backup, so the hash loads
#                                   the lambda=0.5 run despite its stale YAML)
#   task 3  lambda=1.0  7e908919
#   task 4  MHC class II-only ablation 7777d75f -- repairs Sec 3.1's claim that the
#           deployed 37.9% "exceeds" this model's 31.7%, also a cross-sample compare
#   task 5  the DEPLOYED model f16b51e6 -- required, not optional: its own 100 were
#           themselves a random draw, so it must be re-scored on the pinned set for
#           any arm-to-deployed comparison to be paired.
#   Subset:  sbatch --array=1-3 10_matched_eval.sbatch
set -euo pipefail

export PF="${CAPE_ROOT:?set CAPE_ROOT to the project root (see README)}"
export MM2BIN=$PF/CAPE_MPNN/external/programs/MixMHC2pred/MixMHC2pred_unix
export PATH="$PF/CAPE_MPNN/external/programs/bin:$PATH"

# Same GPU-visibility guard as 09_reanchor_train: a node that admits more
# --gres=gpu:1 tasks than it has usable GPUs otherwise kills us in the preamble.
if ! nvidia-smi -L 2>/dev/null | grep -q '^GPU 0'; then
  echo "[matched] no usable GPU on $(hostname) (restart ${SLURM_RESTART_COUNT:-0})" >&2
  if [ "${SLURM_RESTART_COUNT:-0}" -lt 3 ]; then
    scontrol requeue "$SLURM_JOB_ID" && sleep 60 && exit 0
  fi
  exit 1
fi

source "$PF/slurm/activate_env.sh"
cd "$PF/CAPE_MPNN"; mkdir -p "$PF/slurm/logs"

# Node-local TMPDIR, required for netMHCpan specifically. Its vendor script falls
# back to `$NMHOME/tmp` when TMPDIR is unset -- and $NMHOME lives on the NFS mount,
# so four concurrent array tasks would share one temp directory on a filesystem
# that intermittently EPERMs under write load. netMHCIIpan is unaffected (it falls
# back to /tmp), which is why this only shows up once MHC-I scoring is added.
export TMPDIR="${SLURM_TMPDIR:-/tmp}/matched_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
echo "[matched] TMPDIR=$TMPDIR"

LIST="$PF/CAPE_MPNN/data/output/matched_eval_protein_list.txt"
[ -f "$LIST" ] || { echo "missing protein list $LIST"; exit 1; }

case "${SLURM_ARRAY_TASK_ID}" in
  1) ID=54e1642c; LABEL=lam0    ;;
  2) ID=057975d5; LABEL=lam05   ;;
  3) ID=7e908919; LABEL=lam10   ;;
  4) ID=7777d75f; LABEL=mhc2only;;
  5) ID=f16b51e6; LABEL=deployed;;
  *) echo "bad array id"; exit 1 ;;
esac
# CHECKPOINT: the manuscript reports a MIXED policy, and the artefacts confirm it
# (eval_deimmunisation_sweep_f16b51e6 records epoch_200; the lambda-arm and ablation
# summaries record /best):
#   deployed model + the 15-config grid -> epoch_200 (matched on training budget)
#   earlier lambda sweep + single-objective ablations -> best (lowest DPO val loss)
# So reproducing the paper's arms means best for tasks 1-4 and epoch_200 for task 5.
# Defaulting every arm to epoch_200 would silently change published numbers.
if [ -z "${CKPT:-}" ]; then
  if [ "${SLURM_ARRAY_TASK_ID}" = "5" ]; then CKPT=epoch_200; else CKPT=best; fi
fi
# Tag carries a non-default checkpoint so a robustness pass at the other checkpoint
# (sbatch --export=ALL,CKPT=epoch_200 --array=1-4) cannot collide with these outputs.
TAG="matched_${LABEL}"
if { [ "${SLURM_ARRAY_TASK_ID}" = "5" ] && [ "$CKPT" != "epoch_200" ]; } ||
   { [ "${SLURM_ARRAY_TASK_ID}" != "5" ] && [ "$CKPT" != "best" ]; }; then
  TAG="${TAG}_${CKPT}"
fi
echo "[$(date)] matched eval: ${LABEL} id=${ID} ckpt=${CKPT} list=$(grep -cvE "^\s*(#|$)" "$LIST") proteins"

# 1) designs on the PINNED protein set + PWM burden (GPU)
PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/evaluate_deimmunisation.py \
    --model_hash "${ID}" --ckpt "${CKPT}" --protein_list "${LIST}" \
    --k_seqs 10 --seed 42 --out_tag "${TAG}"
CSV="data/output/eval_deimmunisation_${TAG}.csv"

# 2) netMHCIIpan-direct MHC class II reduction (CPU).
#    --n_proteins 1000, not 100: select_proteins() sub-samples stratified by length
#    quartile (per_q = n//4, k = min(per_q, len(bucket))), so a value far above the
#    set size makes it take EVERY protein instead of a stratified subset. Scoring a
#    subset would reintroduce exactly the sampling mismatch this job exists to fix.
PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/validate_with_netmhciipan.py \
    --in_csv "${CSV}" --n_proteins 1000 --k_seqs 10 --out_tag "${TAG}"

# 3) MixMHC2pred, the orthogonal MHC class II predictor (CPU).
#    Needed for the same reason as netMHCIIpan: without it, any cross-arm
#    orthogonal comparison inherits the unmatched-sample problem.
#    (no --k_seqs: validate_with_mixmhc2pred.py does not define it, unlike the
#    netMHCIIpan scorer -- passing it aborts the task.)
#    --bin is REQUIRED (argparse required=True) and --threshold 2.0 matches the working
#    reference 02_sweep_eval.sbatch. This script previously omitted --bin, which aborts
#    the task instantly -- confirmed the hard way on job 512943 tasks 1-2, which died
#    here after 6 h of netMHCIIpan and, under `set -e`, took the netMHCpan step with them.
PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/validate_with_mixmhc2pred.py \
    --in_csv "${CSV}" --bin "${MM2BIN}" --n_proteins 0 --threshold 2.0 \
    --out_tag "${TAG}"   # --n_proteins 0 = all proteins

# 4) netMHCpan MHC class I (CPU). This should just work on the cluster with no
#    setup: external/programs/bin/netMHCpan is the same self-locating wrapper as
#    netMHCIIpan, and the vendor script derives NMHOME from its own location
#    (both patched together on 2026-08-14), so the NFS path prefix is irrelevant.
#    The guard stays as a safety net -- if the binary or tcsh is somehow absent the
#    task still completes its MHC-II work and prints how to finish MHC-I later,
#    which only needs the CSV written above.
if command -v netMHCpan >/dev/null 2>&1; then
  PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/validate_with_netmhcpan.py \
      --in_csv "${CSV}" --out_tag "${TAG}" --k 5
else
  echo "[matched] netMHCpan not on PATH -- MHC-I step SKIPPED for ${LABEL}."
  echo "          Run later:  python tools/validate_with_netmhcpan.py \\"
  echo "            --in_csv ${CSV} --out_tag ${TAG} --k 5"
fi

echo "[$(date)] done: ${LABEL}"
