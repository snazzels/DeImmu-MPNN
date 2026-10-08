#!/bin/bash
#SBATCH --job-name=dpo_sweep_resume
#SBATCH --array=1-14%20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/resume_%A_%a.out
#SBATCH --error=logs/resume_%A_%a.err
#SBATCH --constraint="rtx5090|rtx4090|rtx6000|rtxa5000"
# NOTE: the original 01 array has fully exited (all 13 tails walled 2026-08-16/17), so there
# is no running task to collide with -> submit immediately, no --begin hold needed. (Earlier
# revisions carried #SBATCH --begin=now+36hours to stagger behind still-running 01 tasks;
# removed 2026-08-17 as it's now pure dead time.)
# Resume any sweep config that the 48 h wall limit cut off before epoch 200.
# Each config's model dir already exists (the fresh run wrote ckpts up to the kill point);
# --cont_ckpt_id loads the SAVED hparams (beta/lr/alleles/num_epochs=200) + last_dpo_epoch
# and trains to 200. The array is 1-14 and SELF-SELECTING: configs already at epoch_200.pt
# are skipped, so it's safe to submit the whole array. Run this AFTER the original 01 array
# has fully exited (otherwise a still-running task and its resume would both write the dir).
# ~15 leftover epochs ≈ 3-4 h on an A5000; faster cards finish sooner.
set -euo pipefail

export PF="${CAPE_ROOT:?set CAPE_ROOT to the project root (see README)}"
source "$PF/slurm/activate_env.sh"
cd "$PF/CAPE_MPNN"
mkdir -p "$PF/slurm/logs"

read BETA LR < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PF/slurm/sweep_configs.txt")
HASH=$(PF=$PF python "$PF/slurm/find_model_id.py" "$BETA" "$LR" | tail -1)
if [ -z "$HASH" ]; then
  echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} (beta=$BETA lr=$LR): no model dir found — nothing to resume (train it fresh with 01). Skipping."
  exit 0
fi
CKPTS="$PF/CAPE_MPNN/artefacts/CAPE-MPNN/models/$HASH/ckpts"
if [ -f "$CKPTS/epoch_200.pt" ]; then
  echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} ($HASH): already complete (epoch_200.pt present). Skipping."
  exit 0
fi
LAST=$(ls "$CKPTS"/epoch_*.pt 2>/dev/null | sed 's/.*epoch_\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
if [ -z "${LAST:-}" ]; then
  echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} ($HASH): no epoch checkpoint to resume from — use 01 for a fresh run. Skipping."
  exit 0
fi
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID}: resume $HASH (beta=$BETA lr=$LR) from epoch $LAST -> 200"

# --- stage the PDB dataset to node-local disk (continued training still reloads every 2 epochs) ---
PDB_TAR="$PF/CAPE_MPNN/data/input/CAPE-MPNN/pdb_2021aug02.tar"
STAGE_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/cape_pdb_${SLURM_JOB_ID}"
if [ -f "$PDB_TAR" ]; then
  mkdir -p "$STAGE_ROOT"
  trap 'rm -rf "$STAGE_ROOT"' EXIT
  echo "[$(date)] staging PDB dataset -> $STAGE_ROOT"
  tar -xf "$PDB_TAR" -C "$STAGE_ROOT"
  export PDB_CACHE_DIR="$STAGE_ROOT/pdb_2021aug02"
else
  echo "[$(date)] WARNING: $PDB_TAR not found; falling back to NFS reads (run slurm/00_stage_pdb_tar.sh first)"
fi

# Resume. hparams (beta/lr/alleles/num_epochs=200) load from the saved model via --cont_ckpt_id;
# do NOT re-pass them. BUT the charge regulariser is read from args at runtime (cape-mpnn.py:306),
# NOT from the saved model, so it MUST be re-passed here or the remaining epochs would train with
# it OFF (default weight 0.0) — inconsistent with the first ~185 epochs. Keep it identical to 01.
PYTHONPATH="$PF/CAPE_MPNN/libs" python CAPE-MPNN/cape-mpnn.py \
  --project_path "$PF/CAPE_MPNN" \
  --job "j${SLURM_JOB_ID}" \
  --cont_ckpt_id "$HASH:epoch_$LAST" \
  --charge_reg_weight 1.0 --charge_reg_target 0.15

echo "[$(date)] done resume $HASH -> reached epoch $(ls "$CKPTS"/epoch_*.pt | sed 's/.*epoch_\([0-9]*\)\.pt/\1/' | sort -n | tail -1)"
