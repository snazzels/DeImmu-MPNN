#!/bin/bash
#SBATCH --job-name=pwm_epoch_sweep
#SBATCH --array=1-19%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/pwmsweep_%A_%a.out
#SBATCH --error=logs/pwmsweep_%A_%a.err
# =========================== EDIT FOR YOUR CLUSTER ===========================
#SBATCH --constraint="rtx5090|rtx4090|rtx6000|rtxa5000"
# ----------------------------------------------------------------------------
# PWM reward trajectory for EVERY fully-trained model. Eval-only, no training.
#
# ONE TASK PER MODEL (19 tasks), each walking that model's own epoch list from
# slurm/pwm_epoch_sweep_configs.txt: the 25-epoch grid (25..200) plus its
# lowest-DPO-val-loss epoch plus its best.pt epoch, deduplicated. 179 points total.
#
# ⚠️ THIS GROUPING WAS THE WRONG CALL — measured 2026-08-21, keep for the record.
# The original justification was that 179 per-point tasks would each re-pay the
# env-activation AND model-load cost. Only the first half is true: every point is
# already a separate `python tools/evaluate_deimmunisation.py` invocation inside the
# loop below, so model load + PWM-predictor init + the 98-chain read are paid PER
# POINT either way (measured: 6.6 min/point wall = 5.6 compute + 1.0 setup on a
# 5090; 18.7 = 13.4 + 5.3 on an A5000). Grouping therefore saves only the SLURM slot
# and one micromamba activate -- a minute or two per task.
# What it COSTS is larger: parallelism is capped at 19 and each model's points are
# serialised, so a 10-point model on a slow card takes ~3.1 h however many GPUs are
# idle. ONE TASK PER POINT is the better shape whenever GPUs are free -- use that
# for the re-anchor follow-up below.
#
# ⚠️ DO NOT run two overlapping arrays over the same points. The skip-if-exists
# check further down is NOT atomic: it runs once when a point starts, so a second
# array can miss it for a point the first array is mid-way through, recompute it,
# and write the same path with open(...,"w"). That risks a reader seeing a truncated
# CSV and it violates the never-overwrite rule. To parallelise further, raise
# ArrayTaskThrottle on THIS job (scontrol update JobId=<id> ArrayTaskThrottle=N),
# or scancel only tasks that have not started yet and resubmit those.
#
# WHY THIS EXISTS. The checkpoint policy adopted 2026-08-21 reports epoch_200 for
# every arm. That is defensible only if the reward-vs-fidelity curve is flat near
# 200 for the models we actually report, not just for the deployed one. This array
# measures the curve for all 19, and puts the two rival checkpoint choices --
# lowest val loss, and what `--ckpt best` really loads -- on the same axes.
#
# WHAT IS PINNED, AND WHAT IS NOT:
#   * protein set  -- IDENTICAL for all 179 points via --protein_list (98 chains,
#     pinned by name, hard abort if any fails to load). Verified byte-identical
#     across the 8 existing ckptrob_* CSVs.
#   * base arm     -- IDENTICAL for all 179 points via --reuse_base_csv. The base
#     (v_48_020) designs do not depend on the checkpoint, so regenerating them 179
#     times would burn half the compute AND inject noise: independently drawn base
#     arms differ by ~0.25 recovery points, which is the same order as the
#     epoch-to-epoch differences we are trying to resolve on the plateau.
#   * fine-tuned designs -- NOT pinned. Sampling is stochastic even under --seed 42
#     (the same non-determinism that made --protein_list necessary). Measured paired
#     SE of a reduction difference between adjacent epochs is 1.06 pp today, so two
#     epochs are distinguishable only if they differ by >~2 pp. Expect curve SHAPE
#     (~23% -> 50%) to be far above noise and plateau distinctions often not to be.
#
# PWM ONLY -- no netMHCIIpan, no MixMHC2pred. That is what makes 179 points
# affordable, and it means these are REWARD trajectories: the PWM is the training
# signal, so any headline claim still needs orthogonal validation at whichever
# checkpoints end up selected. Do not quote a number from this array as a
# de-immunisation result.
#
#   Single model:   sbatch --array=9 13_pwm_epoch_sweep.sbatch      # f16b51e6
#   Sweep grid only:sbatch --array=1-15 13_pwm_epoch_sweep.sbatch
#
# TODO (deferred by the author 2026-08-21): once 12_reanchor_resume.sbatch has taken
# dce9dd92 / 1c908bb6 / abec4d2c / 58a9b5ae to epoch 200, append them to the config
# file and run --array=20-23. That is what makes this cover EVERY model we trained.
set -euo pipefail

export PF="${CAPE_ROOT:?set CAPE_ROOT to the project root (see README)}"
export PATH="$PF/CAPE_MPNN/external/programs/bin:$PATH"

# Same GPU-visibility guard as 09/10/11 (t38cn047 oversubscribes gres/gpu).
if ! nvidia-smi -L 2>/dev/null | grep -q '^GPU 0'; then
  echo "[pwmsweep] no usable GPU on $(hostname) (restart ${SLURM_RESTART_COUNT:-0})" >&2
  if [ "${SLURM_RESTART_COUNT:-0}" -lt 3 ]; then
    scontrol requeue "$SLURM_JOB_ID" && sleep 60 && exit 0
  fi
  exit 1
fi

sleep $(( (SLURM_ARRAY_TASK_ID - 1) * 7 ))     # spread env activation a little
source "$PF/slurm/activate_env.sh"
cd "$PF/CAPE_MPNN"; mkdir -p "$PF/slurm/logs"

export TMPDIR="${SLURM_TMPDIR:-/tmp}/pwmsweep_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

LIST="$PF/CAPE_MPNN/data/output/matched_eval_protein_list.txt"
[ -f "$LIST" ] || { echo "FATAL: missing protein list $LIST" >&2; exit 1; }

# The cached base arm. This particular CSV is the canonical base reference for the
# whole sweep: 98 proteins x 10 base sequences, header verified to match the
# evaluator's own fieldnames exactly. evaluate_deimmunisation.py re-validates the
# protein set, the per-protein sequence count, the recovery column and every allele
# column, and aborts on any mismatch -- so a wrong file here fails fast, it does not
# silently produce a mismatched comparison.
BASECSV="$PF/CAPE_MPNN/data/output/eval_deimmunisation_ckptrob_traj_ep200.csv"
if [ ! -f "$BASECSV" ]; then
  echo "FATAL: base-arm CSV $BASECSV not found. Either restore it, or drop" >&2
  echo "       --reuse_base_csv below and accept ~2x runtime plus base-resampling noise." >&2
  exit 1
fi

CFG="$PF/slurm/pwm_epoch_sweep_configs.txt"
[ -f "$CFG" ] || { echo "FATAL: missing $CFG" >&2; exit 1; }
# Nth non-comment, non-blank line -> this task's model.
read -r HASH EPSPEC LABEL < <(grep -vE '^[[:space:]]*(#|$)' "$CFG" | sed -n "${SLURM_ARRAY_TASK_ID}p")
[ -n "${HASH:-}" ] || { echo "FATAL: no config line ${SLURM_ARRAY_TASK_ID} in $CFG" >&2; exit 1; }

NPTS=$(awk -F, '{print NF}' <<< "$EPSPEC")
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID}: ${HASH} (${LABEL}) — ${NPTS} checkpoints"
echo "           epochs: ${EPSPEC}"
echo "           proteins: $(grep -cvE '^[[:space:]]*(#|$)' "$LIST")  base: $(basename "$BASECSV")"

DONE=0; RAN=0
IFS=',' read -ra POINTS <<< "$EPSPEC"
for PT in "${POINTS[@]}"; do
  EP="${PT%%:*}"; ROLE="${PT##*:}"
  CKPT="epoch_${EP}"
  CKPT_FILE="$PF/CAPE_MPNN/artefacts/CAPE-MPNN/models/${HASH}/ckpts/${CKPT}.pt"
  if [ ! -f "$CKPT_FILE" ]; then
    echo "FATAL: missing checkpoint $CKPT_FILE" >&2; exit 1
  fi
  TAG=$(printf "pwmtraj_%s_ep%03d" "$HASH" "$EP")
  CSV="data/output/eval_deimmunisation_${TAG}.csv"

  # SKIP, not fail, if this point is already done. Unlike 11_ckpt_robustness (one
  # point per task, where an existing CSV means a mistake), a task here is 10 points
  # long, so a task that died partway must be resumable without redoing the rest.
  # This never overwrites: an existing CSV is left exactly as it is.
  # NOT ATOMIC -- see the warning in the header. Safe for resuming a dead task,
  # unsafe for two arrays running over the same points concurrently.
  if [ -e "$CSV" ]; then
    echo "  [skip] ${TAG} (role=${ROLE}) already present"
    DONE=$((DONE+1)); continue
  fi

  echo "  [$(date +%H:%M:%S)] ${TAG}  role=${ROLE}"
  PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/evaluate_deimmunisation.py \
      --model_hash "${HASH}" --ckpt "${CKPT}" --protein_list "${LIST}" \
      --reuse_base_csv "${BASECSV}" \
      --k_seqs 10 --seed 42 --out_tag "${TAG}"
  RAN=$((RAN+1))
done

echo "[$(date)] done: ${HASH} (${LABEL}) — ${RAN} evaluated, ${DONE} already present"
