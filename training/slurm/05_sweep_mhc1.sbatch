#!/bin/bash
#SBATCH --job-name=dpo_sweep_mhc1
#SBATCH --array=1-15%20
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mhc1_%A_%a.out
#SBATCH --error=logs/mhc1_%A_%a.err
# =========================== EDIT FOR YOUR CLUSTER ===========================
# CPU-ONLY on purpose: MHC-I scoring is pure PWM lookups, no GPU. Do not add
# --gres=gpu. If your CPU partition needs an explicit name/account, add it here
# (e.g. #SBATCH --partition=cpu  /  #SBATCH --account=YOUR_ACCOUNT).
# ----------------------------------------------------------------------------
# Score the MHC class *I* burden of the SAME designs 02_sweep_eval already
# generated. No GPU and no regeneration: it re-scores the sequences stored in
# eval_deimmunisation_sweep_<hash>.csv against the production MHC-I PWMs
# (6 HLA alleles, lengths 8/9/10). This is the separate MHC-I arm of the sweep,
# complementing 02's MHC-II axis (netMHCIIpan + MixMHC2pred).
#
# Submit AFTER 02_sweep_eval has produced the per-config CSVs. To auto-chain it
# so it starts only once 02 succeeds:
#   sbatch --dependency=afterok:<02_array_jobid> 05_sweep_mhc1.sbatch
# Task 1-14 = sweep_configs.txt; task 15 = the centre (beta0.05/lr3e-7 = 7e908919).
set -euo pipefail

export PF="${CAPE_ROOT:?set CAPE_ROOT to the project root (see README)}"   # top-level project (find_model_id.py appends CAPE_MPNN/...)

# Activate the env WITHOUT slurm/activate_env.sh's GPU gate: that helper does
# `exit 1` if torch cannot see a GPU, which would kill this CPU-only job on a
# CPU node. The env itself (cape_mpnn_cu128) imports fine on a GPU-less node.
CAPE_ENV="${CAPE_ENV:-cape_mpnn_cu128}"
eval "$(micromamba shell hook --shell bash)" 2>/dev/null
micromamba activate "$CAPE_ENV"
echo "[mhc1] node=$(hostname)  env=${CAPE_ENV}  task=${SLURM_ARRAY_TASK_ID}"

cd "$PF/CAPE_MPNN"; mkdir -p "$PF/slurm/logs"

if [ "${SLURM_ARRAY_TASK_ID}" -le 14 ]; then
  read BETA LR < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PF/slurm/sweep_configs.txt")
else
  BETA=0.05; LR=3e-7
fi
ID=$(PF=$PF python "$PF/slurm/find_model_id.py" "${BETA}" "${LR}")
[ -z "${ID:-}" ] && { echo "no trained model for beta=${BETA} lr=${LR}; run 01_sweep_train first"; exit 1; }
TAG="sweep_${ID}"
CSV="$PF/CAPE_MPNN/data/output/eval_deimmunisation_${TAG}.csv"
[ -f "$CSV" ] || { echo "missing ${CSV}; run 02_sweep_eval for this config first"; exit 1; }
echo "[$(date)] mhc1 beta=${BETA} lr=${LR} id=${ID} csv=${CSV}"

# Score MHC-I on the already-generated designs. PF is overridden to the CAPE_MPNN
# root for THIS tool because evaluate_mhc1_arm.py reads os.environ["PF"] as that
# root (PWM_DIR / IN_CSV / OUT paths), unlike the other eval tools which derive it
# from __file__. --out_tag versions all three outputs; --no_figure keeps it lean.
PF="$PF/CAPE_MPNN" PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/evaluate_mhc1_arm.py \
    --in_csv "${CSV}" --out_tag "${TAG}" --no_figure
echo "[$(date)] mhc1 done: ${TAG}  -> data/output/eval_mhc1_arm_${TAG}_summary.txt"
