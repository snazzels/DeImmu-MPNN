#!/bin/bash
#SBATCH --job-name=reanchor_train
#SBATCH --array=1-4%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/reanchor_%A_%a.out
#SBATCH --error=logs/reanchor_%A_%a.err
# =========================== EDIT FOR YOUR CLUSTER ===========================
#SBATCH --constraint="rtx5090|rtx4090|rtx6000|rtxa5000"   # avoid rtxa5000: ~15-17 min/epoch -> over the 48 h wall
# #SBATCH --account=YOUR_ACCOUNT
# ----------------------------------------------------------------------------
# Training runs needed to complete the re-anchor onto the deployed model
# f16b51e6 (beta=0.05, lr=1e-6, lambda=1.0). Each run is ~20 h on a fast card
# (200 epochs). All four are INDEPENDENT -- submit as an array, any subset via --array.
#
#   task 1  lambda=0   @ lr 1e-6   -> TODO 4: isolates the acidic regularizer's effect AT the
#                                     deployed learning rate. Currently Sec 3.3 compares two
#                                     lambda=1.0 models differing in lr, which is weaker.
#   task 2  lambda=0.5 @ lr 1e-6   -> completes the charge knee (0 / 0.5 / 1.0) at the deployed
#                                     lr, so Table tab:sweep can be re-anchored wholesale
#                                     instead of carrying an lr=3e-7 caveat.
#   task 3  MHC-II-only @ lr 1e-6  -> matched attribution control at the deployed lr. The current
#                                     ablation is at lr=3e-7, which is why the deployed model
#                                     (37.9%) now EXCEEDS its 31.7% "ceiling" and Figure 3 panels
#                                     A and B describe different operating points.
#   task 4  seeded replicate of the deployed config -> quantifies run-to-run variance, the one
#                                     unquantified caveat on the whole re-anchor (reviewer R1 P1.1).
#                                     REQUIRES the --seed patch; see the guard below.
#
# SAFETY (project rule: never overwrite weights):
#   * charge_reg_weight/charge_reg_target ARE part of the model hash (verified in
#     libs/CAPE/MPNN/utils.py: d_for_hash keeps every non-None hparam), so tasks 1 and 2 get
#     ids distinct from f16b51e6 -- no collision with the deployed model. Empirical proof:
#     54e1642c (lambda=0) and 7e908919 (lambda=1.0) share beta/lr yet differ in id.
#   * cape-mpnn.py additionally REFUSES any fresh run whose ckpts/ is non-empty.
#   * 'seed' is NOT in dpo_hparams_names, so without the patch every replicate would collide
#     onto the deployed id. Task 4 therefore aborts unless the patch is present.
set -euo pipefail

export PF="${CAPE_ROOT:?set CAPE_ROOT to the project root (see README)}"

# ============================ DIAGNOSIS CORRECTED (2026-08-20) ===============
# The 511251 deaths were NOT a concurrent-`micromamba activate` race. That was a
# misdiagnosis, and the 45 s stagger below could never have fixed it.
#
# Real cause, from the logs of the four tasks that failed on resubmission
# (511262_1, 511262_3, 511280_4, 511281_4): every one landed on t38cn047 and
# logged `gpu=No devices were found`, then activate_env.sh correctly refused
# because torch cannot see a GPU. The node accepts more --gres=gpu:1 tasks than
# it has usable GPUs, so whichever task wins takes the card and the rest start
# with an empty CUDA_VISIBLE_DEVICES and die in the preamble -- which is exactly
# the observed exit 6 at MaxRSS ~17 MB. 511251_2 is the task that won that race
# on 2026-08-19 and is still training on t38cn047 (hash 6fa73cbf).
#
# Retrying cannot help: the winner holds the card for ~43 h. The fix is to
# requeue onto a different node. SLURM_RESTART_COUNT bounds the requeues so a
# scheduler that keeps re-placing us on the same bad node cannot loop forever.
# For an immediate resubmit while 511251_2 still occupies that GPU, also pass
#   sbatch --exclude=t38cn047 09_reanchor_train.sbatch
if ! nvidia-smi -L 2>/dev/null | grep -q '^GPU 0'; then
  echo "[reanchor] no usable GPU on $(hostname) (restart ${SLURM_RESTART_COUNT:-0})" >&2
  if [ "${SLURM_RESTART_COUNT:-0}" -lt 3 ]; then
    echo "[reanchor] requeueing to land on another node" >&2
    scontrol requeue "$SLURM_JOB_ID" && sleep 60 && exit 0
  fi
  echo "[reanchor] FATAL: no GPU after ${SLURM_RESTART_COUNT:-0} requeues" >&2
  exit 1
fi

# Stagger the env activation anyway: harmless, and it keeps four simultaneous
# micromamba activations off one filesystem. The retry below covers transient
# activation failures, not the no-GPU case handled above.
sleep $(( (SLURM_ARRAY_TASK_ID - 1) * 45 ))
_act_ok=0
for _try in 1 2 3; do
  if source "$PF/slurm/activate_env.sh"; then _act_ok=1; break; fi
  echo "[reanchor] activate_env attempt ${_try} failed; retrying in 60 s" >&2
  sleep 60
done
[ "$_act_ok" = "1" ] || { echo "[reanchor] FATAL: env activation failed 3x" >&2; exit 1; }

cd "$PF/CAPE_MPNN"; mkdir -p "$PF/slurm/logs"

# node-local PDB staging (same as 01_sweep_train)
TAR="$PF/CAPE_MPNN/data/input/CAPE-MPNN/pdb_2021aug02.tar"
if [ -f "$TAR" ]; then
  STAGE="${TMPDIR:-/tmp}/cape_pdb_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  mkdir -p "$STAGE"; tar -xf "$TAR" -C "$STAGE"
  export PDB_CACHE_DIR="$STAGE/pdb_2021aug02"
  trap 'rm -rf "$STAGE"' EXIT
  echo "staged PDB dataset -> $PDB_CACHE_DIR"
else
  echo "WARN: $TAR missing; falling back to NFS reads (slower)"
fi

BETA=0.05
LR=1e-6
CREG_W=1.0
CREG_T=0.15
MHC1=pwm
# ===================== BUG FIX (2026-08-20): MHC-II WAS NEVER ON ==============
# This script previously passed only --mhc_1_predictor. cape-mpnn.py defaults
# mhc_2_predictor to None, so EVERY run launched from here trained an
# MHC-I-only objective -- the exact opposite of the paper's subject. It failed
# silently for tasks 1/2/4 (MHC-I alone is a valid objective, so training just
# proceeded on the wrong one) and fatally for task 3, where --mhc_1_predictor
# none left no objective at all and tripped the assertion in cape-mpnn.py:335.
# That assertion is the only reason the bug surfaced.
# 01_sweep_train.sbatch, which trained the deployed model f16b51e6, passes all
# three flags; this script must match it exactly.
MHC2=pwm
MHC2_ALLELES="DRB1_0101+DRB1_0301+DRB1_0401+DRB1_0701+DRB1_1101+DRB1_1501"
# Second instance of the same bug, found 2026-08-20 while auditing every flag
# against 01_sweep_train.sbatch: --mhc_1_alleles was ALSO never passed, so it
# defaulted to the single allele 'HLA-A*02:01' instead of the six-allele HLA-A/B/C
# panel that every real model used (verified in the stored dpo_hparams.yaml of
# f16b51e6 / 7e908919 / 54e1642c / 7777d75f). It is a hashed hparam, so this
# silently trained a different model. Pass every hashed hparam EXPLICITLY --
# never rely on an argparse default for anything that enters the model hash.
MHC1_ALLELES="HLA-A*02:01+HLA-A*24:02+HLA-B*07:02+HLA-B*39:01+HLA-C*07:01+HLA-C*16:01"
BASE_MODEL=v_48_020
MAXLEN=500
RESCUT=3.5
EXTRA=""
LABEL=""

case "${SLURM_ARRAY_TASK_ID}" in
  1) CREG_W=0.0; LABEL="lambda0 @ lr1e-6 (TODO 4)" ;;
  2) CREG_W=0.5; LABEL="lambda0.5 @ lr1e-6 (charge knee)" ;;
  3) MHC1=none;  LABEL="MHC-II-only ablation @ lr1e-6 (matched attribution)" ;;
  4) SEED_VAL=${SEED_VAL:-1}
     LABEL="deployed-config replicate, seed ${SEED_VAL} (variance)"
     if ! grep -q "'seed'" "$PF/CAPE_MPNN/CAPE-MPNN/cape-mpnn.py"; then
       echo "ABORT: 'seed' is not in dpo_hparams_names, so this replicate would hash to the"
       echo "       DEPLOYED model id (f16b51e6) and the overwrite guard would (correctly) stop it."
       echo "       Apply the --seed patch first: add 'seed' to ModelManager.dpo_hparams_names in"
       echo "       CAPE-MPNN/cape-mpnn.py and expose --seed. Because d_for_hash drops None values,"
       echo "       existing seed-less models keep their current ids; only seeded runs get new ones."
       echo "       Runbook: replicate_dpo_runbook_2026-08-13.md"
       exit 1
     fi
     EXTRA="--seed ${SEED_VAL}" ;;
  *) echo "bad array id ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac

echo "[$(date)] task ${SLURM_ARRAY_TASK_ID}: ${LABEL}"
echo "         beta=${BETA} lr=${LR} charge_reg=${CREG_W}/${CREG_T}" \
     "mhc_1_predictor=${MHC1} mhc_2_predictor=${MHC2} ${EXTRA}"

# Objective preflight: MHC-II must be on for every task here (task 3 is the
# MHC-II-*only* ablation, so it needs it most). Catching this in the shell beats
# discovering it 18 h into a run from a Namespace dump.
[ "${MHC2}" = "none" ] && { echo "FATAL: MHC-II objective off; this script trains MHC-II models" >&2; exit 1; }

PYTHONPATH="$PF/CAPE_MPNN/libs" python CAPE-MPNN/cape-mpnn.py \
  --project_path "$PF/CAPE_MPNN" \
  --job "j${SLURM_JOB_ID}" \
  --base_model_name "${BASE_MODEL}" \
  --mhc_1_predictor "${MHC1}" \
  --mhc_1_alleles "${MHC1_ALLELES}" \
  --mhc_2_predictor "${MHC2}" \
  --mhc_2_alleles "${MHC2_ALLELES}" \
  --charge_reg_weight "${CREG_W}" --charge_reg_target "${CREG_T}" \
  --num_epochs 200 --num_examples_per_epoch 1000000 --batch_size 2000 \
  --temperature 0.2 --reload_data_every_n_epochs 2 \
  --max_protein_length "${MAXLEN}" --rescut "${RESCUT}" \
  --beta "${BETA}" --lr "${LR}" ${EXTRA}

echo "[$(date)] done: ${LABEL}"
echo "resolve the new model id with: CREG=${CREG_W} PF=\$PF python \$PF/slurm/find_model_id.py ${BETA} ${LR}"
