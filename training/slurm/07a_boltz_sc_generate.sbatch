#!/bin/bash
#SBATCH --job-name=boltzsc_gen
#SBATCH --array=1-15%15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/boltzsc_gen_%A_%a.out
#SBATCH --error=logs/boltzsc_gen_%A_%a.err
# =========================== EDIT FOR YOUR CLUSTER ===========================
#SBATCH --constraint="rtxa5000|rtx5090"    # A5000 fleet + 5090; cu128 env runs on both
# #SBATCH --account=YOUR_ACCOUNT
# ----------------------------------------------------------------------------
# PHASE 1/3 of Boltz-2 full-length self-consistency. One job per config: generate
# seeded backbones + base/ft designs (full max_len), writing designs.csv + refs/ to
# the per-config mount dir. Fast (ProteinMPNN sampling, light NFS read). Phase 2
# (07b) then folds slices of designs.csv in parallel; phase 3 (07c) scores.
set -euo pipefail
export PF="${CAPE_ROOT:?set CAPE_ROOT to the project root (see README)}"

SEED=${SEED:-0}
N_PROTEINS=${N_PROTEINS:-32}
K_SEQS=${K_SEQS:-3}
MIN_LEN=${MIN_LEN:-40}
MAX_LEN=${MAX_LEN:-400}
CKPT=${CKPT:-epoch_200}

# (beta, lr) for this config = line N of selfconsist_configs.txt (all 15 sweep configs)
read BETA LR < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PF/slurm/selfconsist_configs.txt")
[ -z "${BETA:-}" ] && { echo "no config at line ${SLURM_ARRAY_TASK_ID}"; exit 1; }

source "$PF/slurm/activate_env.sh"
cd "$PF/CAPE_MPNN"; mkdir -p "$PF/slurm/logs"

ID=$(PF=$PF python "$PF/slurm/find_model_id.py" "${BETA}" "${LR}" | tail -1)
[ -z "${ID:-}" ] && { echo "no model for beta=${BETA} lr=${LR}"; exit 1; }
OUT="data/output/selfconsistency_boltz_sweep_${ID}"
echo "[$(date)] gen beta=${BETA} lr=${LR} id=${ID} -> ${OUT}"
if [ -f "${OUT}/designs.csv" ]; then echo "designs.csv exists; skip"; exit 0; fi

PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/selfconsistency_generate.py \
    --outdir "${OUT}" --model_hash "${ID}" --ckpt "${CKPT}" --seed "${SEED}" \
    --n_proteins "${N_PROTEINS}" --k_seqs "${K_SEQS}" --min_len "${MIN_LEN}" --max_len "${MAX_LEN}"
echo "[$(date)] gen done: ${ID}"
