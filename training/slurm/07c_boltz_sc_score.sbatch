#!/bin/bash
#SBATCH --job-name=boltzsc_score
#SBATCH --array=1-15%15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/boltzsc_score_%A_%a.out
#SBATCH --error=logs/boltzsc_score_%A_%a.err
# =========================== EDIT FOR YOUR CLUSTER ===========================
#SBATCH --constraint="rtxa5000|rtx5090"    # GPU only used if straggler healing is needed
# #SBATCH --account=YOUR_ACCOUNT
# ----------------------------------------------------------------------------
# PHASE 3/3, SELF-HEALING: fold any designs left unfolded by preempted 07b shards
# (resume-safe, folds only the missing), then score TM/RMSD vs native. One job/config.
# Submit with --dependency=afterANY:<07b jobid> (NOT afterok) so a preempted fold shard
# can never strand this phase. If 07b completed cleanly, no folding happens — this just
# merges the per-shard metrics and scores (the GPU sits idle).
set -euo pipefail
export PF="${CAPE_ROOT:?set CAPE_ROOT to the project root (see README)}"
BOLTZ_BIN=${BOLTZ_BIN:?set BOLTZ_BIN to the Boltz-2 env bin dir}
SAMPLING_STEPS=${SAMPLING_STEPS:-200}
RECYCLING=${RECYCLING:-3}

read BETA LR < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PF/slurm/selfconsist_configs.txt")
[ -z "${BETA:-}" ] && { echo "no config at line ${SLURM_ARRAY_TASK_ID}"; exit 1; }

source "$PF/slurm/activate_env.sh"
cd "$PF/CAPE_MPNN"; mkdir -p "$PF/slurm/logs"

ID=$(PF=$PF python "$PF/slurm/find_model_id.py" "${BETA}" "${LR}" | tail -1)
[ -z "${ID:-}" ] && { echo "no model for beta=${BETA} lr=${LR}"; exit 1; }
OUT="data/output/selfconsistency_boltz_sweep_${ID}"
# already scored (e.g. the 5 configs done earlier) -> leave untouched, resume-safe
[ -f "${OUT}/selfconsistency_summary.txt" ] && { echo "already scored ${ID}; skip"; exit 0; }
[ -f "${OUT}/designs.csv" ] || { echo "missing ${OUT}/designs.csv — run 07a first"; exit 1; }

NDES=$(($(wc -l < "${OUT}/designs.csv") - 1))
NPRED=$(/usr/bin/find "${OUT}/preds" -maxdepth 1 -name '*.pdb' 2>/dev/null | wc -l)
echo "[$(date)] score id=${ID}: ${NPRED}/${NDES} preds present"

if [ "${NPRED}" -lt "${NDES}" ]; then
  # SELF-HEAL: fold only the missing designs (fold script skips existing PDBs) and harvest
  # ALL of them into a COMPLETE fold_metrics.csv. nshards=1 (default) => slice = all designs.
  echo "[$(date)] healing $((NDES - NPRED)) unfolded designs before scoring"
  WORK="${TMPDIR:-/tmp}/boltzsc_heal_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  PYTHONPATH= PATH="${BOLTZ_BIN}:$PATH" "${BOLTZ_BIN}/python" tools/selfconsistency_fold_boltz.py \
      --outdir "${OUT}" --workroot "${WORK}" \
      --sampling_steps "${SAMPLING_STEPS}" --recycling "${RECYCLING}"
  rm -rf "${WORK}" 2>/dev/null || true
else
  # already complete: merge the per-shard metrics (preserves plddt/ptm the shards recorded)
  PYTHONPATH="$PF/CAPE_MPNN/libs" python - "${OUT}" <<'PY'
import sys, glob, os, pandas as pd
out = sys.argv[1]
sh = sorted(glob.glob(os.path.join(out, "fold_metrics_shard*of*.csv")))
if sh:
    df = pd.concat([pd.read_csv(s) for s in sh], ignore_index=True).drop_duplicates("design_id")
    df.to_csv(os.path.join(out, "fold_metrics.csv"), index=False)
    print(f"merged {len(sh)} shard metrics -> fold_metrics.csv ({len(df)} rows)")
elif not os.path.exists(os.path.join(out, "fold_metrics.csv")):
    sys.exit("no fold_metrics (shard or merged) found")
PY
fi

PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/selfconsistency_score.py --outdir "${OUT}" --no_figure
echo "[$(date)] score done: ${ID}"
