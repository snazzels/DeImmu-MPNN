#!/bin/bash
#SBATCH --job-name=rankguard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rankguard_%j.out
#SBATCH --error=logs/rankguard_%j.err
# =========================== EDIT FOR YOUR CLUSTER ===========================
# CPU-ONLY. Re-scores sequences that ALREADY EXIST. No GPU, no design generation,
# no training. Do NOT add --gres=gpu.
# ----------------------------------------------------------------------------
# ⚠️ SUBMIT FROM THE PROJECT ROOT (--output=logs/... is relative).
#   cd ${CAPE_ROOT} && sbatch slurm/21_rank_sensitivity_guarded.sbatch
#
# ---------------------------------------------------------------------------
# WHY THIS SCRIPT EXISTS INSTEAD OF `sbatch --array=4 18_ablation_coverage_threshold.sbatch`
#
# The v41 handoff named task 4 of job 18 as the fix for the one manuscript quantity the
# min-base guard could not reach: the %Rank cutoff series in §3.1 (43.5 / 37.9 / 37.8 /
# 35.1). Task 4 will NOT fix it, for two independent reasons.
#
#   (1) IT WOULD NOT RUN AT ALL. Task 4 already ran on 2026-08-25 16:43 and its output
#       validate_netmhciipan_rank_sensitivity_deployed_pinned98_ep200_summary.txt is on
#       disk. Job 18 line 121 skips when that file exists, and rank_sensitivity.py asserts
#       against overwriting it. The job would exit 0 having done nothing -- and note that
#       run predates the MIN_BASE patch (the tool was patched at 17:53 the same day), so
#       its numbers are unguarded.
#
#   (2) IT IS THE WRONG BASE ARM. Task 4 reads
#       eval_deimmunisation_pwmtraj_f16b51e6_ep200.csv (1960 designs, pinned 98). The
#       §3.1 series and the 41.4% headline both come from
#       eval_deimmunisation_sweep_f16b51e6.csv (2000 designs, 100 proteins) -- the source
#       of validate_netmhciipan_sweep_f16b51e6.csv, which is what
#       tools/recompute_guarded_reduction.py guarded to 41.36%. The two CSVs share 98
#       protein names but ZERO identical base design sets: checked design by design, all
#       98 common proteins differ. This is the handoff's own second matching axis -- same
#       protein set, different base arm -- which check_protein_match.py cannot see.
#
# The load-bearing property of the §3.1 row is that its 2% entry reproduces the headline
# exactly. Only the sweep CSV can do that. On the pinned-98 CSV the 2% entry is 40.1%,
# which would leave the row still contradicting the headline, just less obviously.
#
# EXPECTED RESULT. The 2% row should land on 41.4% (matching
# validate_netmhciipan_sweep_f16b51e6_guarded_summary.txt to rounding) and the other three
# should rise ~3 pp from 43.5 / 37.8 / 35.1. Guard drops 2 proteins: 4b6d_C (base 0.00)
# and 7m10_A (base 0.10). If the 2% row does NOT reproduce 41.4%, do not fold the series
# into the manuscript -- something differs beyond the guard, and the arithmetic cross-check
# below shows exactly what to look at.
#
#   ARITHMETIC CROSS-CHECK, already verified against the pre-patch summary:
#   unguarded base mean 45.79 over 100 proteins, minus the two dropped (0.00 + 0.10),
#   renormalised to 98 -> 46.72, versus the guarded recompute's 46.70. The two pipelines
#   agree, so the sweep CSV is the right input and the guard is the only difference.
#
# This run also writes a PER-PROTEIN CSV for the first time
# (validate_netmhciipan_rank_sensitivity_<tag>.csv, one row per protein per cutoff, with
# the guard decision recorded rather than applied). The absence of that table is precisely
# why this ~6 h re-score is needed at all; every other arm was corrected offline.
#
# RUNTIME. Six alleles over ~476k unique 15-mers, ONE netMHCIIpan pass per allele with
# ranks retained, so all four cutoffs come from that single pass. Budget ~6-8 h.
#
#   Dry run: CAPE_DRYRUN=1 bash slurm/21_rank_sensitivity_guarded.sbatch
set -euo pipefail

: "${PF:=${CAPE_ROOT:?set CAPE_ROOT to the project root (see README)}}"
export PF
export PATH="$PF/CAPE_MPNN/external/programs/bin:$PATH"
# activate_env.sh asserts CUDA and is SOURCED, so without this a CPU-only task dies on a
# GPU-less node (job 517191, 17/19 tasks).
export CAPE_REQUIRE_GPU=0

IN="data/output/eval_deimmunisation_sweep_f16b51e6.csv"
# Tag names the base arm, not just the model: the pinned-98 tag is already taken by a
# different design draw of the same checkpoint, and conflating them is the failure this
# script exists to avoid.
TAG="deployed_sweep_guarded"
LABEL="deployed model f16b51e6 (sweep base arm, min-base guarded)"

if [ -n "${CAPE_DRYRUN:-}" ]; then
  echo "in : $PF/CAPE_MPNN/${IN}"
  ls -la "$PF/CAPE_MPNN/${IN}" >/dev/null 2>&1 && echo "  input OK" || echo "  *** INPUT MISSING ***"
  echo "tag: ${TAG}"
  exit 0
fi

_ok=0
for _t in 1 2 3; do
  if source "$PF/slurm/activate_env.sh"; then _ok=1; break; fi
  echo "[rankguard] activate_env attempt ${_t} failed; retry in 60 s" >&2; sleep 60
done
[ "$_ok" = "1" ] || { echo "[rankguard] FATAL: env activation failed 3x" >&2; exit 1; }

cd "$PF/CAPE_MPNN"; mkdir -p "$PF/slurm/logs"
# netMHCIIpan's vendor script falls back to $NMHOME/tmp when TMPDIR is unset, and $NMHOME
# is on the NFS mount, which intermittently EPERMs under concurrent write load.
export TMPDIR="${SLURM_TMPDIR:-/tmp}/rankguard_${SLURM_JOB_ID:-0}"
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT

[ -f "$IN" ] || { echo "FATAL: missing input $IN" >&2; exit 1; }
L=$(wc -l < "$IN")
[ "$L" -eq 2001 ] || { echo "FATAL: $IN has $L lines, expected 2001" >&2; exit 1; }
command -v netMHCIIpan >/dev/null 2>&1 || { echo "FATAL: netMHCIIpan not on PATH" >&2; exit 1; }
# The summary this must reproduce at the 2% cutoff. Its absence means the guarded
# recompute was moved or renamed, and the cross-check above is no longer anchored.
REF="data/output/validate_netmhciipan_sweep_f16b51e6_guarded_summary.txt"
[ -f "$REF" ] || echo "WARNING: reference $REF missing; cannot cross-check the 2% row" >&2

echo "[$(date)] rank sensitivity, guarded: in=${IN} tag=${TAG}"

PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/rank_sensitivity.py \
    --in_csv "${IN}" --out_tag "${TAG}" --label "${LABEL}"

echo
echo "=== cross-check: the 2% row must match this guarded recompute ==="
grep -E "Reduction|Base:|Improved" "$REF" 2>/dev/null || true
echo "[$(date)] done"
