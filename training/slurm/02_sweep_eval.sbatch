#!/bin/bash
#SBATCH --job-name=dpo_sweep_eval
#SBATCH --array=1-15%20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
# =========================== EDIT FOR YOUR CLUSTER ===========================
#SBATCH --constraint="rtx5090"     # 5090-first; cu128 env runs on all cards, widen to add more
# #SBATCH --account=YOUR_ACCOUNT
# ----------------------------------------------------------------------------
# Evaluate each swept model (run AFTER 01_sweep_train finishes):
#   generate 100×10 designs (GPU) -> netMHCIIpan reduction (CPU) ->
#   MixMHC2pred orthogonal reduction (CPU) -> sequence recovery.
# Task 1-14 = sweep_configs.txt; task 15 = the already-trained centre (β0.05/lr3e-7 = 7e908919).
set -euo pipefail

export PF="${CAPE_ROOT:?set CAPE_ROOT to the project root (see README)}"                 # HPC project path
export MM2BIN=$PF/CAPE_MPNN/external/programs/MixMHC2pred/MixMHC2pred_unix   # MixMHC2pred v2.1 (vpan_v29), statically linked; PWMdef+nnetsDef bundled, found via exec: relative to binary
export PATH="$PF/CAPE_MPNN/external/programs/bin:$PATH"                      # netMHCIIpan wrapper (self-locating; no path edit needed)

# activate the env (cu128 everywhere: cape_mpnn_cu128 runs on all our GPUs)
source "$PF/slurm/activate_env.sh"
cd "$PF/CAPE_MPNN"; mkdir -p "$PF/slurm/logs"

if [ "${SLURM_ARRAY_TASK_ID}" -le 14 ]; then
  read BETA LR < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PF/slurm/sweep_configs.txt")
else
  BETA=0.05; LR=3e-7
fi
ID=$(PF=$PF python "$PF/slurm/find_model_id.py" "${BETA}" "${LR}")
[ -z "${ID:-}" ] && { echo "no trained model for beta=${BETA} lr=${LR}; run 01_sweep_train first"; exit 1; }
TAG="sweep_${ID}"
CKPT="${CKPT:-epoch_200}"   # checkpoint evaluated for EVERY sweep model: 'epoch_200' (final; apples-to-apples, LOCKED 2026-08-17) — every model trained to a fixed 200 epochs, so the final ckpt is the clean cross-sweep comparison. Override: sbatch --export=ALL,CKPT=best 02_sweep_eval.sbatch
echo "[$(date)] eval beta=${BETA} lr=${LR} id=${ID} ckpt=${CKPT}"

# 1) generate designs + PWM burden (GPU)
PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/evaluate_deimmunisation.py --model_hash "${ID}" --ckpt "${CKPT}" \
    --n_proteins 100 --k_seqs 10 --seed 42 --out_tag "${TAG}"
CSV="data/output/eval_deimmunisation_${TAG}.csv"

# 2) netMHCIIpan-direct reduction (CPU)
PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/validate_with_netmhciipan.py --in_csv "${CSV}" --n_proteins 100 --k_seqs 10 --out_tag "${TAG}"

# 3) MixMHC2pred orthogonal reduction (CPU)
PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/validate_with_mixmhc2pred.py --in_csv "${CSV}" --bin "${MM2BIN}" --n_proteins 0 --threshold 2.0 --out_tag "${TAG}"

# 4) sequence recovery  (writes <outdir>/seq_recovery_summary.txt + seq_recovery.csv; 03_pareto.py parses that path)
PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/seq_recovery_rerun.py --model_hash "${ID}" --ckpt "${CKPT}" \
    --outdir "data/output/seq_recovery_${TAG}" || echo "seq_recovery FAILED for ${ID}"
echo "[$(date)] eval done: ${TAG}"
