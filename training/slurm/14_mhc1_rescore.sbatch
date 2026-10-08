#!/bin/bash
#SBATCH --job-name=mhc1_rescore
#SBATCH --array=1-19%19
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/mhc1res_%A_%a.out
#SBATCH --error=logs/mhc1res_%A_%a.err
# =========================== EDIT FOR YOUR CLUSTER ===========================
# CPU-ONLY on purpose: MHC-I scoring is pure PWM lookups, no GPU. Do NOT add
# --gres=gpu -- doing so would make this queue behind the GPU jobs for no reason.
# If your CPU partition needs an explicit name/account, add it here
# (e.g. #SBATCH --partition=cpu  /  #SBATCH --account=YOUR_ACCOUNT).
# ----------------------------------------------------------------------------
# Add the MHC class *I* axis to every point of the PWM trajectory sweep
# (13_pwm_epoch_sweep / jobs 514056+514120), which scored MHC class II ONLY.
#
# NO GPU AND NO REGENERATION. evaluate_mhc1_arm.py re-scores the sequences already
# stored in eval_deimmunisation_pwmtraj_<hash>_ep<NNN>.csv against the production
# MHC-I PWMs (6 HLA-A/B/C alleles, lengths 8/9/10). Measured 50 s per point, so all
# 179 points is ~2.5 CPU-hours total. Same shape as 05_sweep_mhc1.sbatch, which did
# this for the epoch-200 sweep.
#
# WHY IT IS WORTH DOING -- it supplies the drift control the attribution argument
# needs, from our OWN models:
#   * Most models here are DUAL-objective (mhc_1_predictor=pwm AND mhc_2_predictor=pwm),
#     so both axes are optimised and neither is a control. f16b51e6 at epoch 200 gives
#     MHC-II 49.7% / MHC-I 36.0% -- both trained, so their r=0.635 per-protein
#     correlation says nothing about drift.
#   * 7777d75f is MHC-II-ONLY (--mhc_1_predictor none). Its MHC-I reduction is therefore
#     PURE INCIDENTAL DRIFT. Its 10-point trajectory is an internal, drift-matched
#     control: if incidental MHC-I reduction climbs in step with MHC-II reduction as
#     drift grows, the double dissociation is mostly drift; if MHC-II climbs steeply
#     while MHC-I stays flat, the objective is doing the work. The single-epoch version
#     already known (MHC-II -42.6% vs MHC-I -6.3%) points to the latter, but one drift
#     level is not a curve.
#   * 54e1642c / 057975d5 / f16b51e6 etc. give the dual-objective reference curves.
# Pairs with the external control (Gasser 458340e4/e32b8ed0, MHC-I-only, on the MHC-II
# axis) to give a drift-matched dissociation on BOTH axes -- what Figure 3 needs.
#
# ORDERING. Run AFTER 13_pwm_epoch_sweep has produced the CSVs. It is safe to run
# while the sweep is still going: each point is gated on its input CSV being COMPLETE
# (exactly 1961 lines = 1 header + 2 arms x 98 proteins x 10 sequences -- verified
# identical across every finished CSV), so a half-written file is skipped rather than
# half-scored, and re-running picks up whatever has landed since. Do NOT rely on the
# presence of a sweep _summary.txt as the completeness test: 15 valid CSVs from job
# 514056 have no summary, because that job crashed after writing the CSV but before
# the summary.
#
# ⚠️ Like the sweep, the skip-if-exists check is NOT atomic -- safe for resuming,
# unsafe for two overlapping arrays over the same points.
set -euo pipefail

export PF="${CAPE_ROOT:?set CAPE_ROOT to the project root (see README)}"

# ===== WHY THIS JOB NEEDS CAPE_REQUIRE_GPU=0 (job 514191 post-mortem) ==========
# activate_env.sh asserts torch.cuda.is_available() and, being SOURCED, its `exit 1`
# kills the calling task outright. This job requests NO --gres=gpu (correctly: MHC-I
# scoring is pure PWM lookups), so SLURM places it on GPU-less nodes and every such
# task died instantly. On 2026-08-21 that killed 17 of 19 tasks of job 514191; the
# only 2 survivors were the ones SLURM happened to put on GTX 1080 Ti nodes, which
# passed the CUDA check by accident. The 0-byte .out files initially looked like
# activation contention -- they were not; the error goes to .err and the task was
# already dead. activate_env.sh now takes CAPE_REQUIRE_GPU=0 to skip only the CUDA
# assertion (default unchanged, so no GPU script is affected).
export CAPE_REQUIRE_GPU=0

# Modest stagger: 19 simultaneous micromamba activations on the NFS mount is load
# worth spreading regardless, and the retry loop covers a transient failure. This is
# NOT the fix for the crash above -- CAPE_REQUIRE_GPU=0 is.
sleep $(( (SLURM_ARRAY_TASK_ID - 1) * 12 ))
_act_ok=0
for _try in 1 2 3; do
  if source "$PF/slurm/activate_env.sh"; then _act_ok=1; break; fi
  echo "[mhc1res] activate_env attempt ${_try} failed; retrying in 60 s" >&2
  sleep 60
done
[ "$_act_ok" = "1" ] || { echo "[mhc1res] FATAL: env activation failed 3x" >&2; exit 1; }

cd "$PF/CAPE_MPNN"; mkdir -p "$PF/slurm/logs"

export TMPDIR="${SLURM_TMPDIR:-/tmp}/mhc1res_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

CFG="$PF/slurm/pwm_epoch_sweep_configs.txt"
[ -f "$CFG" ] || { echo "FATAL: missing $CFG" >&2; exit 1; }
read -r HASH EPSPEC LABEL < <(grep -vE '^[[:space:]]*(#|$)' "$CFG" | sed -n "${SLURM_ARRAY_TASK_ID}p")
[ -n "${HASH:-}" ] || { echo "FATAL: no config line ${SLURM_ARRAY_TASK_ID} in $CFG" >&2; exit 1; }

EXPECT_LINES=1961    # 1 header + 2 * 98 proteins * 10 seqs

echo "[$(date)] task ${SLURM_ARRAY_TASK_ID}: MHC-I re-score ${HASH} (${LABEL})"
echo "           epochs: ${EPSPEC}"

DONE=0; RAN=0; WAIT=0
IFS=',' read -ra POINTS <<< "$EPSPEC"
for PT in "${POINTS[@]}"; do
  EP="${PT%%:*}"; ROLE="${PT##*:}"
  SRC_TAG=$(printf "pwmtraj_%s_ep%03d" "$HASH" "$EP")
  IN="data/output/eval_deimmunisation_${SRC_TAG}.csv"
  OUT="data/output/eval_mhc1_arm_mhc1_${SRC_TAG}.csv"

  if [ -e "$OUT" ]; then
    echo "  [skip] ${SRC_TAG} (role=${ROLE}) already scored"
    DONE=$((DONE+1)); continue
  fi
  if [ ! -f "$IN" ]; then
    echo "  [wait] ${SRC_TAG}: sweep has not produced it yet"
    WAIT=$((WAIT+1)); continue
  fi
  # Completeness gate: a CSV still being written would be scored short and the
  # resulting MHC-I numbers would silently rest on fewer sequences.
  L=$(wc -l < "$IN")
  if [ "$L" -ne "$EXPECT_LINES" ]; then
    echo "  [wait] ${SRC_TAG}: ${L} lines, expected ${EXPECT_LINES} (still being written?)"
    WAIT=$((WAIT+1)); continue
  fi

  echo "  [$(date +%H:%M:%S)] ${SRC_TAG}  role=${ROLE}"
  # --no_figure: 179 per-point scatter PNGs would be clutter; the cross-model figure
  # is built by figures/make_fig_pwm_*.py from the CSVs instead.
  PF="$PF/CAPE_MPNN" PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/evaluate_mhc1_arm.py \
      --in_csv "${IN}" --out_tag "mhc1_${SRC_TAG}" --no_figure
  RAN=$((RAN+1))
done

echo "[$(date)] done: ${HASH} (${LABEL}) — ${RAN} scored, ${DONE} already present, ${WAIT} awaiting sweep"
[ "$WAIT" -gt 0 ] && echo "  -> re-run this task once the sweep finishes to pick up the remaining ${WAIT}."
exit 0
