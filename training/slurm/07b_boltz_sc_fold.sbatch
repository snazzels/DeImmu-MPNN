#!/bin/bash
#SBATCH --job-name=boltzsc_fold
#SBATCH --array=1-120%120
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/boltzsc_fold_%A_%a.out
#SBATCH --error=logs/boltzsc_fold_%A_%a.err
# =========================== EDIT FOR YOUR CLUSTER ===========================
#SBATCH --constraint="rtxa5000|rtx5090"    # A5000 fleet + 5090
# #SBATCH --account=YOUR_ACCOUNT
# ----------------------------------------------------------------------------
# PHASE 2/3: Boltz-2 fold, SHARDED. 5 configs x 8 shards = 40 tasks. Each task folds
# one strided 1/8 slice (~28 structures) of one config's designs.csv on one GPU into
# the shared per-config preds/ dir, writing its own fold_metrics_shard file. Full-length
# single-sequence (msa: empty). Submit with --dependency=afterok:<07a jobid>.
set -euo pipefail
export PF="${CAPE_ROOT:?set CAPE_ROOT to the project root (see README)}"
BOLTZ_BIN=${BOLTZ_BIN:?set BOLTZ_BIN to the Boltz-2 env bin dir}

NSHARDS=${NSHARDS:-8}
SAMPLING_STEPS=${SAMPLING_STEPS:-200}
RECYCLING=${RECYCLING:-3}

# task -> (config index 1..15, shard 0..NSHARDS-1); (beta,lr) = line CFG of the config file
CFG=$(( (SLURM_ARRAY_TASK_ID - 1) / NSHARDS + 1 ))
SHARD=$(( (SLURM_ARRAY_TASK_ID - 1) % NSHARDS ))
read BETA LR < <(sed -n "${CFG}p" "$PF/slurm/selfconsist_configs.txt")
[ -z "${BETA:-}" ] && { echo "no config at line ${CFG}"; exit 1; }

source "$PF/slurm/activate_env.sh"          # provides find_model_id python; boltz uses its own bin below
cd "$PF/CAPE_MPNN"; mkdir -p "$PF/slurm/logs"

ID=$(PF=$PF python "$PF/slurm/find_model_id.py" "${BETA}" "${LR}" | tail -1)
[ -z "${ID:-}" ] && { echo "no model for beta=${BETA} lr=${LR}"; exit 1; }
OUT="data/output/selfconsistency_boltz_sweep_${ID}"
WORK="${TMPDIR:-/tmp}/boltzsc_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
[ -f "${OUT}/designs.csv" ] || { echo "missing ${OUT}/designs.csv — run 07a first"; exit 1; }
echo "[$(date)] fold id=${ID} config=${CFG} shard=${SHARD}/${NSHARDS} -> ${OUT}"

# Boltz env python + its `boltz` binary on PATH; PYTHONPATH cleared so the leaked amber
# mpi4py can't crash torch. Intermediates node-local; final PDBs -> mount.
PYTHONPATH= PATH="${BOLTZ_BIN}:$PATH" "${BOLTZ_BIN}/python" tools/selfconsistency_fold_boltz.py \
    --outdir "${OUT}" --workroot "${WORK}" --nshards "${NSHARDS}" --shard "${SHARD}" \
    --sampling_steps "${SAMPLING_STEPS}" --recycling "${RECYCLING}"

rm -rf "${WORK}" 2>/dev/null || true
echo "[$(date)] fold done id=${ID} shard=${SHARD}"
