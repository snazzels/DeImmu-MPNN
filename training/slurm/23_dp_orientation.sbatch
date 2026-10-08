#!/bin/bash
#SBATCH --job-name=dporient
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dporient_%j.out
#SBATCH --error=logs/dporient_%j.err
# =========================== EDIT FOR YOUR CLUSTER ===========================
# CPU-ONLY. Re-scores sequences that ALREADY EXIST. No GPU. Do NOT add --gres=gpu.
# ----------------------------------------------------------------------------
# ⚠️ SUBMIT FROM THE PROJECT ROOT (--output=logs/... is relative).
#   cd ${CAPE_ROOT} && sbatch slurm/23_dp_orientation.sbatch
#
# ---------------------------------------------------------------------------
# P0-5 -- THE HLA-DP INVERTED-BINDING CONFOUND
#
# Raised by the domain (computational immunology) seat of the 2026-09-05 panel, and
# it is the most substantive NEW scientific challenge in that review.
#
# THE PROBLEM. The paper's entire HLA-DP story turns on ONE heterodimer,
# DPA1*02:01-DPB1*01:01: it carries ~2x the base burden of the other two, dominates
# the group statistic, is the one the deployed model fails to reduce (3.1%, p=0.99),
# and is the one the class-II-only checkpoint reduces by 51.7% -- which is what drives
# the recommendation "where HLA-DP coverage matters, use the class-II-only checkpoint".
#
# DPA1*02:01-DPB1*01:01 is also the field's canonical INVERTED-BINDER molecule. Peptide
# inversion (C->N binding) happens almost exclusively for HLA-DP; every DP molecule with
# >=5% inverted ligands carries DPA1*02:01/*02:02 as alpha chain; and netMHCIIpan-4.3
# applies inversion BY DEFAULT FOR DP ONLY. The manuscript never mentions inversion.
#
# WHY THAT MATTERS. It supplies a parsimonious rival to the paper's objective-based
# explanation. A training signal built from FORWARD-register 15-mer DRB1 PWMs has no
# purchase on ligands scored in the reverse orientation -- which would explain BOTH the
# elevated base burden at this heterodimer AND the deployed model's total failure on it,
# without invoking the class I objective at all.
#
#   If the DP gap SURVIVES forward-only scoring -> the objective-based explanation is
#     considerably STRENGTHENED, and the paper gains a result rather than losing one.
#   If it does NOT survive -> the locus-level deployment recommendation needs rewording,
#     because it would rest on a scoring-register artefact.
#
# WHY THIS NEEDS A RE-SCORE AND NOT A RE-PARSE. Checked: validate_with_netmhciipan.py
# keeps ONLY field 9 (%Rank_EL) and discards Core and orientation
# (CAPE_MPNN/tools/validate_with_netmhciipan.py, _parse_netmhciipan). The orientation
# column was never retained anywhere in this project, so the information does not exist
# on disk and must be regenerated. That is the whole reason this is a job and not an
# afternoon of pandas.
#
# ---------------------------------------------------------------------------
# ⚠️ COLUMN LAYOUT IS NOT ASSUMED. netMHCIIpan-4.3's DP output carries the binding-core
# offset and an inversion indicator, but this project has never parsed them. This script
# therefore does TWO things in order:
#
#   STEP 1 dumps the RAW netMHCIIpan output (header included) for a small probe set and
#          prints it. Read that header, confirm which column carries the orientation /
#          inverted flag, and set ORIENT_COL below.
#   STEP 2 only runs once ORIENT_COL is set to something other than AUTO.
#
# This is deliberate. Guessing a column index and silently producing a plausible
# inverted-fraction table is exactly the failure mode this project has been bitten by
# before (a valid-looking number from a wrong field). Two-step is slower and honest.
#
# ---------------------------------------------------------------------------
# LAYOUT AS READ FROM THE VENDOR'S OWN SHIPPED REFERENCE OUTPUT (2026-09-06).
#
# The installed release ships worked examples under
# CAPE_MPNN/external/programs/netMHCIIpan/netMHCIIpan-4.3/test/. Their table header is
# IDENTICAL across all five output modes present (peptide, FASTA, peptide-MHC,
# context, hlaseq A+B):
#
#   1 Pos   2 MHC   3 Peptide   4 Of   5 Core   6 Core_Rel   7 Inverted
#   8 Identity      9 Score_EL  10 %Rank_EL    11 Exp_Bind   12 BindLevel(optional)
#
#   => Inverted is field 7 (1-based)  ==  index 6 (0-based)  ==  the ORIENT_COL wanted.
#   => %Rank_EL is field 10 (1-based) ==  index 9 (0-based), which independently CONFIRMS
#      the parts[9] used by validate_with_netmhciipan.py and shuffle_control.py.
#
# This is a primary source, not an inference, and it is a stronger one than the probe:
# it covers five modes and includes a DP heterodimer (HLA-DPA10103-DPB11501).
#
# ⚠️ WHAT IT DOES *NOT* ESTABLISH, which is why STEP 1 still runs. Every data row in
# every shipped example has Inverted=0 (0 of 506 rows across all eight .out files), so the
# column position is confirmed but the column has never been OBSERVED taking value 1. The
# probe on real DP ligands is what shows that this install actually emits inverted calls
# for the three heterodimers we care about -- and if it emits none, the whole confound is
# moot and the DP result stands as-is. Do not skip STEP 1 on the strength of the header;
# it now answers a different and better question than "which column".
#
# Note also: `-inv_all 0` is the documented default, i.e. inversion is considered for DP
# only. Adding -BA would insert affinity columns and shift everything after Score_EL, so
# do not add it without re-reading the header.
#
# ---------------------------------------------------------------------------
# PREAMBLE PROVENANCE. Copied verbatim from slurm/21_rank_sensitivity_guarded.sbatch
# (job 519997, ran clean 2026-08-26), changing only the log tag. The hand-composed version
# died in under a second on job 529063 with "FATAL: netMHCIIpan not on PATH" -- a guard
# placed ABOVE the environment setup it depends on, so it could never pass. See the same
# note in 22_shuffle_control.sbatch for the full three-fault breakdown.
# ---------------------------------------------------------------------------
set -euo pipefail

: "${PF:=${CAPE_ROOT:?set CAPE_ROOT to the project root (see README)}}"
export PF
export PATH="$PF/CAPE_MPNN/external/programs/bin:$PATH"
# activate_env.sh asserts CUDA and is SOURCED; without this a CPU-only task dies on a
# GPU-less node.
export CAPE_REQUIRE_GPU=0

[ -f "$PF/CAPE_MPNN/tools/validate_with_netmhciipan.py" ] || {
  echo "FATAL: \$PF does not look like the project root ($PF)" >&2; exit 1; }

_ok=0
for _t in 1 2 3; do
  if source "$PF/slurm/activate_env.sh"; then _ok=1; break; fi
  echo "[dporient] activate_env attempt ${_t} failed; retry in 60 s" >&2; sleep 60
done
[ "$_ok" = "1" ] || { echo "[dporient] FATAL: env activation failed 3x" >&2; exit 1; }

cd "$PF/CAPE_MPNN"
# netMHCIIpan's vendor script falls back to $NMHOME/tmp when TMPDIR is unset, and $NMHOME
# is on the NFS mount, which intermittently EPERMs under concurrent write load.
export TMPDIR="${SLURM_TMPDIR:-/tmp}/dporient_${SLURM_JOB_ID:-0}"
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT

command -v netMHCIIpan >/dev/null 2>&1 || { echo "FATAL: netMHCIIpan not on PATH" >&2; exit 1; }

DP_ALLELES="HLA-DPA10201-DPB10101 HLA-DPA10103-DPB10401 HLA-DPA10103-DPB10201"
# _v2: the v1 probe (job 529926) is kept. It exited 0 but captured only netMHCIIpan's
# ~50-line parameter preamble, because `head -40` cut off ABOVE the data table -- the
# `Pos MHC Peptide ...` header sits at line ~56. Anchoring on a line count was the bug;
# this version anchors on the table header itself and cannot truncate that way again.
PROBE="data/output/dp_orientation_probe_v2.txt"
SRC="data/output/allele_generalisation_pinned24_deployed.csv"
N_PROBE_PEP=200      # cheap (seconds); 40 was too few to estimate a ~10-20% inverted rate

# ---- STEP 1: probe the output format -------------------------------------
if [ ! -f "$PROBE" ]; then
  echo "=== STEP 1: probing netMHCIIpan DP output format ==="
  [ -f "$SRC" ] || { echo "FATAL: missing $SRC" >&2; exit 1; }

  # real 15-mers from the pinned-24 designs
  PYTHONPATH="$PF/CAPE_MPNN/libs" python - "$SRC" "$TMPDIR/probe.pep" "$N_PROBE_PEP" <<'PY'
import csv, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
seen=[]
for r in csv.DictReader(open(src)):
    s=r["sequence"].replace("/","")
    seen += [s[i:i+15] for i in range(0, min(len(s)-15, 60), 15)]
    if len(seen) >= n: break
peps = sorted(set(seen))[:n]
assert peps, "no 15-mers extracted from " + src
open(dst,"w").write("\n".join(peps)+"\n")
print(f"[probe] {len(peps)} unique 15-mers from {src.split('/')[-1]}")
PY

  # Build into $TMPDIR and move into place only once ALL alleles have been probed.
  # Appending straight to $PROBE would leave a partial file that the `[ ! -f ]` guard
  # above then accepts as complete on the next submit.
  : > "$TMPDIR/probe.out"
  for AL in $DP_ALLELES; do
    echo "--- $AL" >> "$TMPDIR/probe.out"
    # Do NOT pipe netMHCIIpan directly into `head`: head closing the pipe SIGPIPEs the
    # producer, and `set -o pipefail` would turn that into a job abort. Capture, then head.
    # stderr is left unredirected so a real vendor-script error reaches the .err log
    # instead of /dev/null.
    netMHCIIpan -inptype 1 -f "$TMPDIR/probe.pep" -a "$AL" > "$TMPDIR/raw.txt"
    # Anchor on the table header, NEVER on a line count. netMHCIIpan's parameter preamble
    # is ~50 lines, so `head -40` (v1, job 529926) captured settings and no data at all.
    # Print from the "Pos MHC Peptide" header through the first 12 data rows.
    awk '/^ *Pos +MHC +Peptide/{f=1} f{print; if(++n>=15) exit}' "$TMPDIR/raw.txt" \
      >> "$TMPDIR/probe.out"
    # A DP pass that produced no data rows is the tcsh-less-node failure mode: it fails by
    # returning nothing, not by erroring. Stop rather than freeze an empty probe.
    # "Data row" is defined exactly as _parse_netmhciipan() defines it: >=10 whitespace
    # fields, an INTEGER in field 1 (Pos), and a numeric %Rank_EL in field 10.
    # The field-1 test is load-bearing, not decoration: netMHCIIpan's trailing summary line
    #   "Number of strong binders: 2 Number of weak binders: 0"
    # has 11 fields and a numeric field 10, so without it a run that predicted NOTHING
    # would still satisfy this guard -- the exact false negative the guard exists to catch.
    awk 'NF>=10 && ($1+0==$1) && $1!="" && ($10+0==$10) && $10!="" {n++} END{exit(n>0?0:1)}' "$TMPDIR/raw.txt" || {
      echo "FATAL: netMHCIIpan returned no parseable data rows for $AL. Not a result." >&2
      echo "       On this cluster that means the wrapper did not run (it execs tcsh)." >&2
      exit 1; }

    # INDICATIVE inverted count, read at field 7 (Inverted) per the vendor header above.
    # This is NOT the reported statistic -- it is 200 probe 15-mers, not the design set --
    # but it answers the only question that gates whether step 2 is worth writing at all:
    # does THIS install emit inverted calls for THIS heterodimer? The table header is
    # printed directly above it, so the column mapping is checkable by eye, not hidden.
    awk -v al="$AL" '
      NF>=10 && ($1+0==$1) && $1!="" && ($10+0==$10) && $10!="" { t++; if ($7==1) i++ }
      END { printf "[indicative] %-24s inverted %d / %d rows (%.1f%%)\n", al, i+0, t+0, (t?100*i/t:0) }
    ' "$TMPDIR/raw.txt" | tee -a "$TMPDIR/probe.out"
  done
  mv "$TMPDIR/probe.out" "$PROBE"

  echo "Wrote $PF/CAPE_MPNN/$PROBE"
  echo
  cat "$PROBE"
  echo
  echo "============================================================"
  echo "ACTION REQUIRED: read the header above, identify the column"
  echo "carrying the binding-core offset / inverted indicator, set"
  echo "ORIENT_COL in this script, and resubmit."
  echo "============================================================"
  exit 0
fi

# ---- STEP 2: the real run -------------------------------------------------
# ORIENT_COL is now SET, on evidence, not guessed. Two independent confirmations:
#   (1) the vendor's shipped reference outputs under external/programs/netMHCIIpan/
#       netMHCIIpan-4.3/test/ carry `Inverted` as field 7 (1-based) in an identical
#       header across all five output modes -> index 6 (0-based);
#   (2) step 1 on real DP output (job 531434) showed every Inverted=1 row reports a Core
#       that is the REVERSE of a forward substring of its peptide, e.g.
#       AANSLEDLAAKLAKI -> core KAALDELSN, reversed NSLEDLAAK = positions 3-11.
# dp_orientation.py RE-VERIFIES (2) at runtime and aborts if <90% of inverted rows hold,
# so a wrong-but-binary column cannot silently produce a table. Verified on the real
# probe output: 5/5 inverted rows pass (100%).
ORIENT_COL=6

echo "=== STEP 2: DP orientation re-score, ORIENT_COL=$ORIENT_COL ==="
echo "Arms: deployed (f16b51e6) and class-II-only (abec4d2c), pinned 24-protein set."
echo

DEPLOYED="data/output/allele_generalisation_pinned24_deployed.csv"
MHC2ONLY="data/output/allele_generalisation_pinned24_mhc2only.csv"
OUT_TAG="dp_orientation_pinned24"
OUT="data/output/${OUT_TAG}_summary.txt"

for f in "$DEPLOYED" "$MHC2ONLY"; do
  [ -f "$f" ] || { echo "FATAL: missing input $f" >&2; exit 1; }
done
if [ -f "$OUT" ]; then
  echo "[skip] $OUT already exists -- refusing to overwrite a frozen result."
  echo "       A corrective re-run needs a NEW --out_tag, never a deletion."; exit 0
fi

# The class-II-only arm is the `_mhc2only` file, NOT a filename containing abec4d2c.
# Both carry protein_set_md5 9cae3da74828 and their 96 base designs are byte-identical,
# so this is a genuinely matched pair -- the tool asserts both properties and aborts
# otherwise, because "same protein names" is not "same base arm".
#
# COST: ~72k unique 15-mers over 3 heterodimers (~217k peptide-allele evaluations).
# Job 22 sustained ~660k/h, so budget ~20-30 min. The 8 h wall is headroom.
#
# ACCEPTANCE TEST (enforced in the tool, not advisory): with orientation ignored, every
# per-design count must reproduce the published n_<allele> column EXACTLY, design by
# design. If it does not, something other than orientation differs and the tool refuses
# to write a summary rather than emit a plausible, wrong forward-only table.
PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/dp_orientation.py \
    --arm "deployed=${DEPLOYED}" \
    --arm "mhc2only=${MHC2ONLY}" \
    --out_tag "${OUT_TAG}" \
    --orient_col "${ORIENT_COL}"

echo
echo "=== REFERENCE: the published all-windows pooled DP reductions ==="
echo "  these are what section (b)'s 'all %' column must reproduce"
echo "    DPA1*02:01-DPB1*01:01   deployed  3.13%   class-II-only 51.65%"
echo "    DPA1*01:03-DPB1*04:01   deployed 29.87%   class-II-only 16.17%"
echo "    DPA1*01:03-DPB1*02:01   deployed 15.86%   class-II-only 25.85%"
echo "[$(date)] done"
