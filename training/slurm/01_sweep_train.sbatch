#!/bin/bash
#SBATCH --job-name=dpo_sweep_train
#SBATCH --array=1-14%20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
# =========================== EDIT FOR YOUR CLUSTER ===========================
#SBATCH --constraint="rtx5090|rtx4090|rtx6000|rtxa5000"                             # 5090-first; widen to add more cards
# cu128 env runs on all our GPUs, so no per-card env work is needed to widen this.
# To use more cards, add their feature tokens, e.g. "rtx5090|rtx6000|rtx4090|rtxa5000".
# ----------------------------------------------------------------------------
# β×lr hyperparameter sweep, deployed λ=1.0 config. 14 configs, ≤20 concurrent.
# Each run ≈ 20 h on 1 GPU (measured from the deployed 7e908919 run, 200 epochs).
# The grid centre (β=0.05, lr=3e-7) is ALREADY 7e908919 — not retrained here.
# Each config differs in a hashed hyperparameter → a distinct model id → the
# never-overwrite guard is satisfied automatically (no --seed patch needed).
set -euo pipefail

export PF="${CAPE_ROOT:?set CAPE_ROOT to the project root (see README)}"   # HPC project path

# --- activate the env (cu128 everywhere: cape_mpnn_cu128 runs on all our GPUs) ---
# One env/torch version across the whole fleet (Turing..Blackwell); one submission
# can span mixed node types. (micromamba; for conda/module load edit activate_env.sh.)
source "$PF/slurm/activate_env.sh"

cd "$PF/CAPE_MPNN"
mkdir -p "$PF/slurm/logs"

# --- stage the PDB dataset to node-local disk (avoids per-epoch small-file NFS reads) ---
# The 15 GB / ~200k-file dataset is reloaded every 2 epochs; over NFS each reload is ~11 min
# (200k round-trips) and dominates wall-clock. We extract a single pre-built tarball once
# (sequential read) onto the node's local ext4, and point training there via PDB_CACHE_DIR.
# Build the tarball once with slurm/00_stage_pdb_tar.sh BEFORE submitting. If it is missing
# we fall back to NFS reads (slow but correct). STAGE_ROOT is unique per array task and is
# removed on exit so concurrent tasks on one node don't collide or leak node-local disk.
PDB_TAR="$PF/CAPE_MPNN/data/input/CAPE-MPNN/pdb_2021aug02.tar"
STAGE_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/cape_pdb_${SLURM_JOB_ID}"
if [ -f "$PDB_TAR" ]; then
  mkdir -p "$STAGE_ROOT"
  trap 'rm -rf "$STAGE_ROOT"' EXIT
  echo "[$(date)] staging PDB dataset -> $STAGE_ROOT"
  tar -xf "$PDB_TAR" -C "$STAGE_ROOT"
  export PDB_CACHE_DIR="$STAGE_ROOT/pdb_2021aug02"
  echo "[$(date)] staged $(find "$PDB_CACHE_DIR/pdb" -maxdepth 1 -type d 2>/dev/null | wc -l) shards to node-local disk"
else
  echo "[$(date)] WARNING: $PDB_TAR not found; falling back to NFS reads (run slurm/00_stage_pdb_tar.sh first)"
fi

read BETA LR < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PF/slurm/sweep_configs.txt")
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID}  beta=${BETA}  lr=${LR}"

# PYTHONPATH is set to the CAPE support libs (kit/CAPE live in CAPE_MPNN/libs) AND
# thereby overrides the leaked broken amber mpi4py on the login shell (see README).
PYTHONPATH="$PF/CAPE_MPNN/libs" python CAPE-MPNN/cape-mpnn.py \
  --project_path "$PF/CAPE_MPNN" \
  --job "j${SLURM_JOB_ID}" \
  --base_model_name v_48_020 \
  --mhc_1_predictor pwm \
  --mhc_1_alleles "HLA-A*02:01+HLA-A*24:02+HLA-B*07:02+HLA-B*39:01+HLA-C*07:01+HLA-C*16:01" \
  --mhc_2_predictor pwm \
  --mhc_2_alleles "DRB1_0101+DRB1_0301+DRB1_0401+DRB1_0701+DRB1_1101+DRB1_1501" \
  --charge_reg_weight 1.0 --charge_reg_target 0.15 \
  --num_epochs 200 --num_examples_per_epoch 1000000 --batch_size 2000 \
  --temperature 0.2 --reload_data_every_n_epochs 2 \
  --max_protein_length 500 --rescut 3.5 \
  --beta "${BETA}" --lr "${LR}"

echo "[$(date)] done beta=${BETA} lr=${LR} -> model id: $(PF=$PF python "$PF/slurm/find_model_id.py" "${BETA}" "${LR}")"
