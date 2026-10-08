#!/bin/bash
#SBATCH --job-name=reanchor_resume
#SBATCH --array=1-4%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/reanchor_resume_%A_%a.out
#SBATCH --error=logs/reanchor_resume_%A_%a.err
# =========================== EDIT FOR YOUR CLUSTER ===========================
#SBATCH --constraint="rtx5090|rtx4090|rtx6000|rtxa5000"
# rtxa5000 is left IN on purpose here (unlike 09_reanchor_train, where it is the
# reason the 48 h wall is missed): a resume needs only ~20 epochs, which is ~5 h
# even at the A5000 rate of 16 min/epoch. Waiting for a faster card costs more
# than running on a slow one.
# ----------------------------------------------------------------------------
# Finish the four re-anchor runs of job 512539 (09_reanchor_train.sbatch), which
# wall at 48 h on 2026-08-22 ~11:19 with tasks 1/2/4 at ~epoch 178-181 of 200.
#
# WHY THIS EXISTS AT ALL. Truncation used to be survivable because the paper
# reported `best.pt`. As of 2026-08-21 the policy is `epoch_200` for EVERY arm,
# so a run that stops at epoch 180 has no reportable checkpoint.
#
# WHY NOT 04_sweep_resume.sbatch. That script maps an array index -> model hash
# through find_model_id.py, which matches on (beta, lr, charge_reg_weight) only.
# All four re-anchor runs share beta=0.05 and lr=1e-6, and three of them
# (abec4d2c, 58a9b5ae, f16b51e6) additionally share charge_reg_weight=1.0 --
# they differ only in mhc_1_predictor and seed, neither of which find_model_id.py
# looks at. It would return whichever directory sorts first (58a9b5ae) for all
# three. The hashes are therefore HARDCODED below and verified against each
# model's own dpo_hparams.yaml before training starts.
#
#   task 1  dce9dd92  lambda=0    @ lr 1e-6   (TODO 4: isolates the regulariser)
#   task 2  1c908bb6  lambda=0.5  @ lr 1e-6   (charge knee 0 / 0.5 / 1.0)
#   task 3  abec4d2c  MHC-II-only @ lr 1e-6   (matched attribution control)
#   task 4  58a9b5ae  replicate, seed 1       (run-to-run variance, R1 P1.1)
#
# SELF-SELECTING: any task whose model already has epoch_200.pt exits 0 without
# training, so the whole array is safe to submit. Task 3 was projected to reach
# epoch 200 on its own at ~04:10 on 2026-08-22 and should skip itself.
#
# SUBMIT (do NOT submit while 512539 is still running -- both would write the
# same ckpts/ directory):
#     sbatch --dependency=afterany:512539 slurm/12_reanchor_resume.sbatch
# The squeue preflight below refuses to start if any task of PARENT_JOB is still
# queued or running, so a mis-timed bare `sbatch` fails fast instead of
# corrupting a run. Override with PARENT_JOB=none once 512539 has left the queue.
#
# NO OVERWRITE (project hard rule). Checkpoint names are 1-indexed against the
# 0-indexed training loop: loop epoch e writes epoch_{e+1}.pt (cape-mpnn.py:161).
# ModelManager sets last_dpo_epoch = int("epoch_86".split("_")[1]) = 86
# (libs/CAPE/MPNN/utils.py:103) and the loop runs range(86, 200), so the first
# thing written is epoch_87.pt. No existing epoch_*.pt is touched. Verified
# against the 04_sweep_resume runs of 2026-08-17.
set -euo pipefail

export PF="${CAPE_ROOT:?set CAPE_ROOT to the project root (see README)}"

# ---- do not race the parent array -------------------------------------------
PARENT_JOB="${PARENT_JOB:-512539}"
if [ "$PARENT_JOB" != "none" ]; then
  STILL=$(squeue -j "$PARENT_JOB" -h -o '%i %T' 2>/dev/null || true)
  if [ -n "$STILL" ]; then
    echo "ABORT: parent job $PARENT_JOB is still in the queue:" >&2
    echo "$STILL" >&2
    echo "       Two processes writing one ckpts/ directory corrupts the run." >&2
    echo "       Resubmit with --dependency=afterany:$PARENT_JOB, or once 512539" >&2
    echo "       has fully exited, with PARENT_JOB=none." >&2
    exit 1
  fi
fi

# ---- GPU visibility guard (same as 09/10: t38cn047 oversubscribes gres/gpu) ---
if ! nvidia-smi -L 2>/dev/null | grep -q '^GPU 0'; then
  echo "[resume] no usable GPU on $(hostname) (restart ${SLURM_RESTART_COUNT:-0})" >&2
  if [ "${SLURM_RESTART_COUNT:-0}" -lt 3 ]; then
    echo "[resume] requeueing to land on another node" >&2
    scontrol requeue "$SLURM_JOB_ID" && sleep 60 && exit 0
  fi
  echo "[resume] FATAL: no GPU after ${SLURM_RESTART_COUNT:-0} requeues" >&2
  exit 1
fi

sleep $(( (SLURM_ARRAY_TASK_ID - 1) * 45 ))
_act_ok=0
for _try in 1 2 3; do
  if source "$PF/slurm/activate_env.sh"; then _act_ok=1; break; fi
  echo "[resume] activate_env attempt ${_try} failed; retrying in 60 s" >&2
  sleep 60
done
[ "$_act_ok" = "1" ] || { echo "[resume] FATAL: env activation failed 3x" >&2; exit 1; }

cd "$PF/CAPE_MPNN"; mkdir -p "$PF/slurm/logs"

# ---- per-task identity -------------------------------------------------------
# CREG_W / CREG_T / SEED_ARG are the values that must be RE-PASSED on the command
# line. Everything else (beta, lr, alleles, both predictors, num_epochs=200,
# batch_size, max_protein_length, rescut, temperature, reload_data_every_n_epochs)
# is loaded from the model's saved dpo_hparams.yaml by --cont_ckpt_id and must NOT
# be re-passed. See the "READ FROM ARGS" note below for why these three are the
# exception.
SEED_ARG=""
case "${SLURM_ARRAY_TASK_ID}" in
  1) HASH=dce9dd92; CREG_W=0.0; LABEL="lambda0 @ lr1e-6"        ;;
  2) HASH=1c908bb6; CREG_W=0.5; LABEL="lambda0.5 @ lr1e-6"      ;;
  3) HASH=abec4d2c; CREG_W=1.0; LABEL="MHC-II-only @ lr1e-6"    ;;
  4) HASH=58a9b5ae; CREG_W=1.0; LABEL="replicate seed 1"
     SEED_ARG="--seed 1" ;;
  *) echo "bad array id ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac
CREG_T=0.15

MDIR="$PF/CAPE_MPNN/artefacts/CAPE-MPNN/models/$HASH"
YAML="$MDIR/dpo_hparams.yaml"
CKPTS="$MDIR/ckpts"
[ -f "$YAML" ] || { echo "FATAL: no such model $HASH ($YAML missing)" >&2; exit 1; }

if [ -f "$CKPTS/epoch_200.pt" ]; then
  echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} ($HASH, ${LABEL}): epoch_200.pt already present -- complete, skipping."
  exit 0
fi
LAST=$(ls "$CKPTS"/epoch_*.pt 2>/dev/null | sed 's/.*epoch_\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
[ -n "${LAST:-}" ] || { echo "FATAL: $HASH has no epoch_*.pt to resume from" >&2; exit 1; }

# ---- READ FROM ARGS, NOT FROM THE MODEL: verify before spending GPU hours -----
# Three settings are taken from argparse at RUNTIME even on a resume, so omitting
# them silently changes the objective for the remaining epochs:
#   * charge_reg_weight / charge_reg_target -- cape-mpnn.py:328 reads args, not
#     model.dpo_hparams. Omitting them trains the tail with the regulariser OFF
#     (default 0.0). This is the same class of bug as 09's missing
#     --mhc_2_predictor, one layer out: eval/resume flags are not hashed, so
#     check_train_flags.py cannot catch them.
#   * seed -- cape-mpnn.py:183 seeds the RNGs from args.seed. The saved hparam
#     (seed: '1') only fixes the model id; it does NOT re-seed a resumed run.
#     Task 4 must pass --seed 1 or its tail is unseeded.
#   * save_model_every_n_epochs -- cape-mpnn.py:158 reads args. Default is 1 and
#     matches all four models, but it is passed explicitly for the same reason.
# The check below compares what we are about to pass against what the run was
# actually started with, so a typo in the table above aborts in seconds.
yval() {
  local v; v=$(sed -n "s/^$2: *//p" "$1" | tr -d "\"'")
  # An absent key must not read as empty: awk coerces "" to 0, which would let a
  # yaml missing charge_reg_weight pass the lambda=0 check silently.
  [ -n "$v" ] || { echo "FATAL: key '$2' absent from $1" >&2; exit 1; }
  printf '%s' "$v"
}
same_num() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a+0==b+0)}'; }

Y_CREG_W=$(yval "$YAML" charge_reg_weight)
Y_CREG_T=$(yval "$YAML" charge_reg_target)
Y_SEED=$(yval "$YAML" seed)
Y_MHC2=$(yval "$YAML" mhc_2_predictor)
Y_MHC1=$(yval "$YAML" mhc_1_predictor)
Y_EPOCHS=$(yval "$YAML" num_epochs)

same_num "$CREG_W" "$Y_CREG_W" || { echo "FATAL: $HASH was trained with charge_reg_weight=$Y_CREG_W but this script would pass $CREG_W" >&2; exit 1; }
same_num "$CREG_T" "$Y_CREG_T" || { echo "FATAL: $HASH was trained with charge_reg_target=$Y_CREG_T but this script would pass $CREG_T" >&2; exit 1; }
if [ -n "$SEED_ARG" ]; then
  same_num "1" "$Y_SEED" || { echo "FATAL: task 4 passes --seed 1 but $HASH records seed=$Y_SEED" >&2; exit 1; }
else
  [ "$Y_SEED" = "None" ] || { echo "FATAL: $HASH records seed=$Y_SEED but this task passes no --seed, so its tail would be unseeded" >&2; exit 1; }
fi
# The 09 bug, re-checked at the model level: every arm here must have the MHC-II
# objective on, and only task 3 may have MHC-I off.
[ "$Y_MHC2" = "pwm" ] || { echo "FATAL: $HASH has mhc_2_predictor=$Y_MHC2; this script only resumes MHC-II models" >&2; exit 1; }
if [ "${SLURM_ARRAY_TASK_ID}" = "3" ]; then
  [ "$Y_MHC1" = "none" ] || { echo "FATAL: task 3 is the MHC-II-only arm but $HASH has mhc_1_predictor=$Y_MHC1" >&2; exit 1; }
else
  [ "$Y_MHC1" = "pwm" ] || { echo "FATAL: $HASH has mhc_1_predictor=$Y_MHC1, expected pwm" >&2; exit 1; }
fi
[ "$Y_EPOCHS" = "200" ] || { echo "FATAL: $HASH records num_epochs=$Y_EPOCHS; the resume target would not be 200" >&2; exit 1; }

echo "[$(date)] task ${SLURM_ARRAY_TASK_ID}: resume $HASH (${LABEL}) from epoch $LAST -> 200"
echo "          charge_reg=${CREG_W}/${CREG_T} ${SEED_ARG:-(unseeded)}" \
     "mhc_1=${Y_MHC1} mhc_2=${Y_MHC2}  [$((200 - LAST)) epochs left]"

# ---- node-local PDB staging (the loader still reloads every 2 epochs) ---------
TAR="$PF/CAPE_MPNN/data/input/CAPE-MPNN/pdb_2021aug02.tar"
if [ -f "$TAR" ]; then
  STAGE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/cape_pdb_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  mkdir -p "$STAGE"; tar -xf "$TAR" -C "$STAGE"
  export PDB_CACHE_DIR="$STAGE/pdb_2021aug02"
  trap 'rm -rf "$STAGE"' EXIT
  echo "staged PDB dataset -> $PDB_CACHE_DIR"
else
  echo "WARN: $TAR missing; falling back to NFS reads (slower)"
fi

PYTHONPATH="$PF/CAPE_MPNN/libs" python CAPE-MPNN/cape-mpnn.py \
  --project_path "$PF/CAPE_MPNN" \
  --job "j${SLURM_JOB_ID}" \
  --cont_ckpt_id "${HASH}:epoch_${LAST}" \
  --charge_reg_weight "${CREG_W}" --charge_reg_target "${CREG_T}" \
  --save_model_every_n_epochs 1 ${SEED_ARG}

REACHED=$(ls "$CKPTS"/epoch_*.pt | sed 's/.*epoch_\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
echo "[$(date)] done: $HASH (${LABEL}) reached epoch ${REACHED}"
[ "$REACHED" = "200" ] || echo "WARNING: $HASH stopped at epoch ${REACHED}, not 200 -- resubmit this task."

# ---- AFTERWARDS --------------------------------------------------------------
# Two artefacts of resuming, both harmless under the epoch_200 policy but worth
# knowing before reading metrics/losses.tsv:
#   * the `reload` column restarts at 0, so it is no longer a running count.
#   * best.pt is recomputed over the RESUMED SEGMENT ONLY -- model.dpo_losses
#     starts empty in memory, so the first saved epoch cannot be "best" and any
#     later improvement within the tail overwrites the global best.pt with a
#     worse checkpoint. best.pt is not comparable across a resume boundary. This
#     is an additional, independent reason the paper reports epoch_200.
# Then evaluate with 10_matched_eval.sbatch (these four are not in its table yet).
