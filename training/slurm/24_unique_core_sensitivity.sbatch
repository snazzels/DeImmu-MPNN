#!/bin/bash
#SBATCH --job-name=coresens
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=logs/coresens_%j.out
#SBATCH --error=logs/coresens_%j.err
# =========================== EDIT FOR YOUR CLUSTER ===========================
# CPU-ONLY. Re-scores sequences that ALREADY EXIST. No GPU, no design generation,
# no training. Do NOT add --gres=gpu.
# ----------------------------------------------------------------------------
# ⚠️ SUBMIT FROM THE PROJECT ROOT (--output=logs/... is relative).
#   cd ${CAPE_ROOT} && sbatch slurm/24_unique_core_sensitivity.sbatch
#
# ---------------------------------------------------------------------------
# P1-1 / R2-W4 -- unique-binding-core sensitivity.
#
# WHAT IT ANSWERS. The burden metric counts presented 15-mer windows, so windows
# sharing one binding core are counted separately and one epitope can contribute up
# to seven. The manuscript calls this "conservative" and never supports it. This
# recomputes the reduction on distinct (allele, binding core) pairs instead.
#
# WHAT R2 DID NOT CLAIM, and the write-up must not garble it: the paired tests use
# per-protein means with n = proteins, so window correlation does NOT inflate the
# degrees of freedom. This is metric interpretation, not statistical inflation.
#
# WHY IT IS NOT A FORMALITY. The permutation control (§3.3) showed the reduction is
# compositional rather than positional. A compositional shift could plausibly shorten
# runs of presented windows around a core that still binds, which would deflate the
# window count faster than the unique-core count and inflate the headline. This
# measures which of the two happened.
#
# INPUT AND WHY THIS ONE. data/output/validate_netmhciipan_sweep_f16b51e6.csv is the
# arm behind the 41.4% headline -- 2000 designs over 100 proteins, base and
# fine-tuned, with the per-design netmhc_total the acceptance test reproduces.
# Do NOT substitute the pinned-98 CSV: it shares 98 protein names with this one but
# has a DIFFERENT base design set, which is the second matching axis
# check_protein_match.py cannot see (see slurm/21_rank_sensitivity_guarded.sbatch).
#
# ACCEPTANCE TEST, enforced not advisory. Every per-design window count recomputed
# here must reproduce that CSV's netmhc_total EXACTLY. On any disagreement the tool
# writes no summary and exits non-zero, because the unique-core numbers would then
# sit on a different footing than the published ones.
#
# RUNTIME. ~1.0M unique 15-mers over 6 alleles, one netMHCIIpan pass per allele.
# The shuffle control (job 529925) did 993,546 x 6 in 8 h 56 m on this partition,
# so budget ~9-12 h. --time is set to 16 h for headroom.
#
# SMOKE-TESTED 2026-09-08 against a deterministic stub (netMHCIIpan cannot run on the
# local box -- its wrapper execs tcsh). Verified: happy path end to end; the
# acceptance test fires and writes NO summary on mismatched counts; the overwrite
# guard refuses an existing tag.
#
#   Dry run: CAPE_DRYRUN=1 bash slurm/24_unique_core_sensitivity.sbatch
set -euo pipefail

: "${PF:=${CAPE_ROOT:?set CAPE_ROOT to the project root (see README)}}"
export PF
export PATH="$PF/CAPE_MPNN/external/programs/bin:$PATH"
# activate_env.sh asserts CUDA and is SOURCED, so without this a CPU-only task dies on a
# GPU-less node.
export CAPE_REQUIRE_GPU=0

IN="data/output/validate_netmhciipan_sweep_f16b51e6.csv"
TAG="unique_core_sweep_f16b51e6"

if [ -n "${CAPE_DRYRUN:-}" ]; then
  echo "in : $PF/CAPE_MPNN/${IN}"
  ls -la "$PF/CAPE_MPNN/${IN}" >/dev/null 2>&1 && echo "  input OK" || echo "  *** INPUT MISSING ***"
  echo "tag: ${TAG}"
  echo "out: $PF/CAPE_MPNN/data/output/${TAG}{.csv,_summary.txt}"
  exit 0
fi

# ⚠️ ORDERING. Every guard that depends on the environment MUST come after this block.
# A fail-fast check placed above its own setup turns a working job into an instant,
# confident-looking failure (jobs 529062/529063 died in <1 s on "netMHCIIpan not on
# PATH", which was true and useless).
_ok=0
for _t in 1 2 3; do
  if source "$PF/slurm/activate_env.sh"; then _ok=1; break; fi
  echo "[coresens] activate_env attempt ${_t} failed; retry in 60 s" >&2; sleep 60
done
[ "$_ok" = "1" ] || { echo "[coresens] FATAL: env activation failed 3x" >&2; exit 1; }

cd "$PF/CAPE_MPNN"; mkdir -p "$PF/slurm/logs"
export PYTHONPATH="$PF/CAPE_MPNN/libs"
# netMHCIIpan's vendor script falls back to $NMHOME/tmp when TMPDIR is unset, and $NMHOME
# is on the NFS mount, which intermittently EPERMs under concurrent write load.
export TMPDIR="${SLURM_TMPDIR:-/tmp}/coresens_${SLURM_JOB_ID:-0}"
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT

[ -f "$IN" ] || { echo "FATAL: missing input $IN" >&2; exit 1; }
L=$(wc -l < "$IN")
[ "$L" -eq 2001 ] || { echo "FATAL: $IN has $L lines, expected 2001" >&2; exit 1; }
command -v netMHCIIpan >/dev/null 2>&1 || { echo "FATAL: netMHCIIpan not on PATH" >&2; exit 1; }

OUT="data/output/${TAG}_summary.txt"
if [ -f "$OUT" ]; then
  echo "[coresens] $OUT already exists — nothing to do."
  echo "           A corrective re-run needs a NEW tag, never a deletion."; exit 0
fi

echo "[$(date)] unique-core sensitivity: in=${IN} tag=${TAG}"
python tools/unique_core_sensitivity.py \
  --csv "$IN" \
  --published_col netmhc_total \
  --out_tag "${TAG}" \
  --out_dir data/output

echo
echo "=== HOW TO READ THE RESULT ==="
echo "  The published window-metric reduction on this arm is 41.36% (93/98 improved),"
echo "  in validate_netmhciipan_sweep_f16b51e6_guarded_summary.txt. The summary's"
echo "  n_windows row must reproduce it — that is the acceptance test made visible."
echo "  Then compare the n_unique_pairs row:"
echo "    within ~2 pp  -> overlap counting is not inflating the headline;"
echo "                     'conservative' becomes supported rather than asserted."
echo "    windows much higher -> the reduction shortens runs of presented windows more"
echo "                     than it removes distinct cores; lead with the unique-core"
echo "                     figure and report both."
echo "    unique-core higher  -> overlap counting was genuinely conservative."
echo "[$(date)] done"
