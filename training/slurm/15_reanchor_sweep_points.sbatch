#!/bin/bash
#SBATCH --job-name=reanchor_sweep
#SBATCH --array=1-38%12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/reanchorsweep_%A_%a.out
#SBATCH --error=logs/reanchorsweep_%A_%a.err
# =========================== EDIT FOR YOUR CLUSTER ===========================
#SBATCH --constraint="rtx5090|rtx4090|rtx6000|rtxa5000"
#SBATCH --exclude=t38cn047
# ----------------------------------------------------------------------------
# ⚠️ --exclude=t38cn047 IS LOAD-BEARING, added 2026-08-24 after job 517539 lost
# 29 of 38 tasks to that node in ~7 minutes. t38cn047 advertises more gres/gpu than
# it has usable GPUs, so tasks land there, find no GPU, and die in the preamble.
# The nvidia-smi guard below DETECTED this correctly on all 29 — but its recovery
# path is dead on this cluster: `scontrol requeue` returns
#     Requested operation is presently disabled for job <id>
# because requeue is administratively disabled here. So the guard degrades to a
# clean abort, and with 12 tasks cycling through a bad node the array drains fast.
# Excluding the node is the only fix that actually keeps tasks alive; the same dead
# requeue path is inherited verbatim by 09/10/11/13 and should get the same
# treatment if they ever draw that node.
# ----------------------------------------------------------------------------
# Extends the PWM epoch sweep to the four RE-ANCHOR models — the deferred TODO in
# 13_pwm_epoch_sweep.sbatch. Eval-only, no training, no hashed hyperparameters.
#
# ⚠️ SUBMIT FROM THE PROJECT ROOT, NOT FROM slurm/.
#   cd ${CAPE_ROOT} && sbatch slurm/15_reanchor_sweep_points.sbatch
# `#SBATCH --output=logs/...` is a RELATIVE path, so logs follow the submission
# directory. On 2026-08-21 one job's logs got split across ./logs/ and slurm/logs/
# because two submissions used different cwds, and that alone made a healthy job
# look dead. Job 517473 and 517525 both wrote to ./logs/ — keep it that way.
#
# ---------------------------------------------------------------------------
# ONE TASK PER POINT (38 tasks), which is the correction 13_pwm_epoch_sweep.sbatch's
# own header asks for. 13 used one task per MODEL and its header now records why that
# was wrong: every point is already a separate `python` invocation, so model load +
# PWM-predictor init + the 98-chain read are paid per point either way (measured 6.6
# min/point on a 5090, 18.7 on an A5000). Grouping saved only the SLURM slot and one
# micromamba activate, while capping parallelism at the model count and serialising
# each model's points. With 38 points and 12 concurrent tasks this finishes in ~3
# rounds instead of 10 serialised points on whatever card the model happened to draw.
#
# A useful side effect: the skip-if-exists check below is NOT atomic (see 13's
# header), but with one point per task each output path is owned by exactly one task,
# so within this job the race cannot occur. It is still unsafe to run this job
# CONCURRENTLY with 13 over the same points — it is not, since 13 covers config
# lines 1-19 and this covers 20-23.
#
# ---------------------------------------------------------------------------
# WHAT THESE FOUR MODELS ARE, AND WHY THEY ARE WORTH 38 POINTS.
# All hyperparameters below were read back from each model's own dpo_hparams.yaml on
# 2026-08-24, not taken from the job table:
#
#   dce9dd92  beta0.05 lr1e-06 charge_reg 0.0  mhc_1=pwm   mhc_2=pwm   seed None
#   1c908bb6  beta0.05 lr1e-06 charge_reg 0.5  mhc_1=pwm   mhc_2=pwm   seed None
#   abec4d2c  beta0.05 lr1e-06 charge_reg 1.0  mhc_1=none  mhc_2=pwm   seed None
#   58a9b5ae  beta0.05 lr1e-06 charge_reg 1.0  mhc_1=pwm   mhc_2=pwm   seed 1
#   f16b51e6  beta0.05 lr1e-06 charge_reg 1.0  mhc_1=pwm   mhc_2=pwm   (DEPLOYED)
#
# The point is the lr. The lambda arms and the MHC-II-only ablation already in the
# sweep (54e1642c, 057975d5, 7777d75f, 8da989f4) are all at **lr 3e-07**, while the
# deployed model is at **lr 1e-06**. So none of them is hyperparameter-matched to the
# model the paper reports. These four are the same arms re-run at the deployed lr:
#
#   * abec4d2c is the EXACT MHC-II-only counterpart of the deployed model — it differs
#     from f16b51e6 in mhc_1_predictor and nothing else. That makes it the matched
#     drift control the attribution argument actually wants. The old 7777d75f differs
#     from the deployed model in BOTH lr and charge_reg_weight, so it is an
#     independent instance, not a matched control. DO NOT describe abec4d2c as a
#     "replicate of 7777d75f" — agreement between them is evidence the ~6% incidental
#     MHC-I drift generalises across configs, which is a different and weaker claim
#     than a same-config error bar.
#   * 58a9b5ae is f16b51e6 with seed 1 and is otherwise identical — a true replicate,
#     so it measures run-to-run variance on the deployed config directly instead of
#     inferring it from the 1.06 pp adjacent-epoch noise floor.
#   * dce9dd92 / 1c908bb6 put lambda=0 / lambda=0.5 on the deployed lr, so the charge
#     regulariser knee can be drawn at the lr that is actually reported.
#
# ---------------------------------------------------------------------------
# WORK LIST: lines 20-23 of slurm/pwm_epoch_sweep_configs.txt, flattened to one point
# per line. Those four config lines were appended 2026-08-24 and carry their own
# derivation notes. Roles were NOT guessed: val-min is the lowest val loss over
# epoch-END rows only (non-empty train column), and each best.pt was md5-confirmed
# byte-identical to its epoch_N.pt so it can be evaluated by epoch number and
# deduplicated against the 25-grid. 9+10+9+10 = 38, not 40 — dce9dd92 and abec4d2c
# each have val-min == best.pt and so are run once at that epoch.
#
# PINNED THE SAME WAY AS THE FIRST 179 POINTS, which is what makes the new panels
# comparable to the old ones rather than merely adjacent to them:
#   * protein set -- the same 98 chains via --protein_list, hard abort on any failure
#   * base arm    -- the same 980 base sequences via --reuse_base_csv, so the base is
#                    not redrawn and contributes no noise
#   * fine-tuned designs -- NOT pinned; sampling is stochastic even under --seed 42.
#
# PWM ONLY. These are REWARD trajectories, not de-immunisation results. Add the MHC-I
# axis afterwards with `sbatch --array=20-23 slurm/14_mhc1_rescore.sbatch` (CPU-only,
# ~50 s-3 min per point) — that script reads the same config file and picks up lines
# 20-23 with no edit needed.
#
#   Single model: sbatch --array=1-9   15_...   # dce9dd92 (see the map it prints)
#   Dry run:      CAPE_DRYRUN=1 bash slurm/15_reanchor_sweep_points.sbatch
set -euo pipefail

# Default is the cluster path, but allow an override so CAPE_DRYRUN=1 can be run from
# the NFS-mounted copy on a workstation (where the /zfs path does not exist). SLURM
# never sets PF, so on the cluster this is exactly the old unconditional assignment.
: "${PF:=${CAPE_ROOT:?set CAPE_ROOT to the project root (see README)}}"
export PF
export PATH="$PF/CAPE_MPNN/external/programs/bin:$PATH"

CFG="$PF/slurm/pwm_epoch_sweep_configs.txt"
[ -f "$CFG" ] || { echo "FATAL: missing $CFG" >&2; exit 1; }

# Flatten config lines 20-23 into "hash epoch role label", one point per line.
# CFG_FIRST/CFG_LAST are the re-anchor block; lines 1-19 belong to 13_pwm_epoch_sweep.
CFG_FIRST=20; CFG_LAST=23; EXPECT_POINTS=38
POINTS=$(grep -vE '^[[:space:]]*(#|$)' "$CFG" | sed -n "${CFG_FIRST},${CFG_LAST}p" \
         | awk '{n=split($2,a,","); for(i=1;i<=n;i++){split(a[i],b,":"); print $1, b[1], b[2], $3}}')
NP=$(printf '%s\n' "$POINTS" | grep -c .)
# Guard against a config edit silently changing the work list under a fixed --array.
if [ "$NP" -ne "$EXPECT_POINTS" ]; then
  echo "FATAL: flattened $NP points from $CFG lines ${CFG_FIRST}-${CFG_LAST}, expected ${EXPECT_POINTS}." >&2
  echo "       The config file changed. Fix --array=1-${NP} and EXPECT_POINTS together." >&2
  exit 1
fi

# Dry run: print the whole array map and exit before requesting any GPU work.
if [ -n "${CAPE_DRYRUN:-}" ]; then
  echo "flattened ${NP} points from ${CFG} lines ${CFG_FIRST}-${CFG_LAST}:"
  printf '%s\n' "$POINTS" | nl -ba -w4 -s'  task -> '
  exit 0
fi

# Same GPU-visibility guard as 09/10/11/13 (t38cn047 admits more gres/gpu than it has
# usable GPUs, and the loser dies in the preamble).
if ! nvidia-smi -L 2>/dev/null | grep -q '^GPU 0'; then
  echo "[reanchorsweep] no usable GPU on $(hostname) (restart ${SLURM_RESTART_COUNT:-0})" >&2
  # Try requeue, but do NOT assume it works: on this cluster `scontrol requeue` is
  # administratively disabled ("Requested operation is presently disabled"), which is
  # how job 517539 lost 29/38 tasks. `|| true` keeps set -e from masking the real
  # message, and the explicit hint below is the actionable part -- resubmitting is
  # always safe here because each point skips if its CSV already exists.
  if [ "${SLURM_RESTART_COUNT:-0}" -lt 3 ] && scontrol requeue "$SLURM_JOB_ID" 2>/dev/null; then
    sleep 60; exit 0
  fi
  echo "[reanchorsweep] requeue unavailable or exhausted -- aborting task ${SLURM_ARRAY_TASK_ID}." >&2
  echo "                Bad node should be in --exclude. Just resubmit: finished points skip." >&2
  exit 1
fi

sleep $(( (SLURM_ARRAY_TASK_ID - 1) % 12 * 7 ))    # spread env activation a little
source "$PF/slurm/activate_env.sh"
cd "$PF/CAPE_MPNN"; mkdir -p "$PF/slurm/logs"

export TMPDIR="${SLURM_TMPDIR:-/tmp}/reanchorsweep_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

LIST="$PF/CAPE_MPNN/data/output/matched_eval_protein_list.txt"
[ -f "$LIST" ] || { echo "FATAL: missing protein list $LIST" >&2; exit 1; }

# The canonical cached base arm for the whole sweep — the SAME file the first 179
# points used, which is what makes the new panels comparable to the old ones.
# evaluate_deimmunisation.py re-validates the protein set, the per-protein sequence
# count, the recovery column and every allele column, and aborts on any mismatch.
BASECSV="$PF/CAPE_MPNN/data/output/eval_deimmunisation_ckptrob_traj_ep200.csv"
[ -f "$BASECSV" ] || { echo "FATAL: base-arm CSV $BASECSV not found." >&2; exit 1; }

read -r HASH EP ROLE LABEL < <(printf '%s\n' "$POINTS" | sed -n "${SLURM_ARRAY_TASK_ID}p")
[ -n "${HASH:-}" ] || { echo "FATAL: no point ${SLURM_ARRAY_TASK_ID} in the flattened list" >&2; exit 1; }

CKPT="epoch_${EP}"
CKPT_FILE="$PF/CAPE_MPNN/artefacts/CAPE-MPNN/models/${HASH}/ckpts/${CKPT}.pt"
[ -f "$CKPT_FILE" ] || { echo "FATAL: missing checkpoint $CKPT_FILE" >&2; exit 1; }

# Same tag scheme as the first 179 points (pwmtraj_<hash>_ep<NNN>) so the existing
# figure generators and 14_mhc1_rescore.sbatch pick these up with no changes.
TAG=$(printf "pwmtraj_%s_ep%03d" "$HASH" "$EP")
CSV="data/output/eval_deimmunisation_${TAG}.csv"

echo "[$(date)] task ${SLURM_ARRAY_TASK_ID}/${NP}: ${HASH} (${LABEL}) ${CKPT} role=${ROLE}"
echo "           proteins: $(grep -cvE '^[[:space:]]*(#|$)' "$LIST")  base: $(basename "$BASECSV")"

# SKIP, not fail, so a requeued or resubmitted task is safe. Never overwrites.
if [ -e "$CSV" ]; then
  echo "  [skip] ${TAG} already present — leaving it untouched"
  exit 0
fi

PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/evaluate_deimmunisation.py \
    --model_hash "${HASH}" --ckpt "${CKPT}" --protein_list "${LIST}" \
    --reuse_base_csv "${BASECSV}" \
    --k_seqs 10 --seed 42 --out_tag "${TAG}"

echo "[$(date)] done: ${TAG}"
