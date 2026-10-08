#!/bin/bash
#SBATCH --job-name=netmhc_matched
#SBATCH --array=1-7%7
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=logs/netmhcmatch_%A_%a.out
#SBATCH --error=logs/netmhcmatch_%A_%a.err
# =========================== EDIT FOR YOUR CLUSTER ===========================
# CPU-ONLY on purpose. Both validators are CPU tools run over sequences that ALREADY
# EXIST in the sweep CSVs -- no designs are generated, no GPU is touched. Do NOT add
# --gres=gpu; it would queue this behind the GPU jobs for nothing, and it would hold a
# card idle for hours (exactly what jobs 517535-517538 are doing right now).
# ----------------------------------------------------------------------------
# ⚠️ SUBMIT FROM THE PROJECT ROOT.  --output=logs/... is a RELATIVE path.
#   cd ${CAPE_ROOT} && sbatch slurm/16_netmhc_matched_ablation.sbatch
#
# ---------------------------------------------------------------------------
# WHY THIS EXISTS. The matched MHC-II-only ablation result now carried in the
# Discussion (manuscript v35) is measured on the PWM TRAINING REWARD:
#
#   epoch 200, differing in mhc_1_predictor alone, same lr 1e-6, same lambda 1.0
#     f16b51e6  dual (deployed)     MHC-II 49.66 +- 1.69   MHC-I  +36.00 +- 1.37   1.89 pt
#     abec4d2c  MHC-II-only         MHC-II 50.72 +- 1.54   MHC-I   -0.79 +- 2.51   1.76 pt
#     58a9b5ae  dual, seed 1        MHC-II 49.51 +- 1.68   MHC-I  +37.57 +- 1.48   1.97 pt
#
# But the manuscript's own stated policy (Methods, "Evaluation metrics") is that every
# REPORTED reduction is measured with the netMHC tools rather than the PWM, "since the
# PWMs are the training signal and a PWM-based reduction would be partly self-referential".
# So those three rows cannot be promoted to headline numbers as they stand. This job
# re-scores the SAME stored designs with the two independent-lineage predictors, which
# is what puts them on the same footing as every other reduction in the paper.
#
# It also closes a gap that predates the sweep: the deployed model has netMHCIIpan on the
# pinned 98 chains (validate_netmhciipan_ckptrob_traj_ep200) but has NEVER had netMHCpan
# on that same set -- the existing class-I numbers are on other protein samples
# (validate_netmhcpan_mhc1_fullset_*, validate_netmhcpan_monomer_f16b51e6). A class-I
# comparison between the dual model and the ablation therefore has no matched baseline
# until task 2 below runs.
#
# WHAT IS ALREADY DONE AND DELIBERATELY NOT REPEATED:
#   f16b51e6 ep200 netMHCIIpan  -> validate_netmhciipan_ckptrob_traj_ep200.*  (exists)
# That output was produced by 11_ckpt_robustness task 8 from
# eval_deimmunisation_ckptrob_traj_ep200.csv, which is the SAME model/epoch/protein set
# as eval_deimmunisation_pwmtraj_f16b51e6_ep200.csv but a different design draw. Re-using
# it keeps us from overwriting a result file, and the two draws differ only by sampling
# noise (~1 pp, and 1.9 pp between seeds).
#
# ---------------------------------------------------------------------------
# TASK MAP.  One task per (model, validator) so the seven run concurrently rather than
# serialising ~6 h of netMHCIIpan four times over.
#
#   1  abec4d2c  netMHCIIpan   the matched MHC-II-only ablation, class II   <-- the key one
#   2  f16b51e6  netMHCpan     the deployed model's MISSING class-I baseline on this set
#   3  abec4d2c  netMHCpan     the ablation's class I: expected ~0 if PWM is right
#   4  58a9b5ae  netMHCIIpan   seed replicate, class II -> between-run error bar (R-2)
#   5  58a9b5ae  netMHCpan     seed replicate, class I
#   6  7777d75f  netMHCIIpan   the OLD unmatched control, on the same protein set
#   7  7777d75f  netMHCpan     ditto class I -- lets old-vs-matched control be compared
#                              without the protein-set confound
#
# Tasks 6-7 are included because the old control's netMHC numbers live on a different
# protein sample, and the whole point of the matched ablation is to remove exactly that
# kind of mismatch. Dropping them would leave the old-vs-new control contrast resting on
# PWM only.
#
# RUNTIME. netMHCIIpan dominates: budget ~6-8 h for tasks 1/4/6 (98 proteins x 2 arms x
# 10 sequences x 6 alleles). netMHCpan tasks are shorter (5 sequences per arm, not 10).
# All seven run at once, so wall time is roughly one netMHCIIpan pass.
#
#   Class II only:  sbatch --array=1,4,6 16_netmhc_matched_ablation.sbatch
#   Class I only:   sbatch --array=2,3,5,7 16_netmhc_matched_ablation.sbatch
#   Dry run:        CAPE_DRYRUN=1 bash slurm/16_netmhc_matched_ablation.sbatch
set -euo pipefail

: "${PF:=${CAPE_ROOT:?set CAPE_ROOT to the project root (see README)}}"
export PF
export PATH="$PF/CAPE_MPNN/external/programs/bin:$PATH"

# activate_env.sh asserts torch.cuda.is_available() and is SOURCED, so its `exit 1` would
# kill this task outright on a GPU-less node -- which is where a job with no --gres=gpu
# belongs. This is the flag added after job 517191 lost 17/19 tasks to exactly that.
export CAPE_REQUIRE_GPU=0

TASK="${SLURM_ARRAY_TASK_ID:-1}"
case "$TASK" in
  1) HASH=abec4d2c; VAL=ii ; LABEL=reanchor_mhc2only     ;;
  2) HASH=f16b51e6; VAL=i  ; LABEL=deployed              ;;
  3) HASH=abec4d2c; VAL=i  ; LABEL=reanchor_mhc2only     ;;
  4) HASH=58a9b5ae; VAL=ii ; LABEL=replicate_seed1       ;;
  5) HASH=58a9b5ae; VAL=i  ; LABEL=replicate_seed1       ;;
  6) HASH=7777d75f; VAL=ii ; LABEL=old_mhc2only          ;;
  7) HASH=7777d75f; VAL=i  ; LABEL=old_mhc2only          ;;
  # ---- added 2026-08-24, submit as --array=8-10 -----------------------------------
  # 8: MixMHC2pred is the INDEPENDENT-LINEAGE class-II check. The manuscript treats it
  #    as a requirement for the deployed model ("re-scored ... with predictors that were
  #    not optimized during training"), so the checkpoint v35 recommends as the targeted
  #    alternative has to clear the same bar.
  # 9,10: netMHCIIpan for the two re-anchor lambda arms. This is what allows the CHARGE
  #    KNEE FIGURE to be redrawn at the deployed operating point: the charge half is
  #    already done locally and free (charge_shift_pwmtraj_reanchor_lambda0 / _lambda05,
  #    2026-08-24), but its epitope axis is currently PWM-only. The manuscript's knee is
  #    built on the lr-3e-7 series, so without these two the redraw cannot use netMHCIIpan.
  8)  HASH=abec4d2c; VAL=mm ; LABEL=reanchor_mhc2only     ;;
  9)  HASH=dce9dd92; VAL=ii ; LABEL=reanchor_lambda0      ;;
  10) HASH=1c908bb6; VAL=ii ; LABEL=reanchor_lambda05     ;;
  # ---- added 2026-08-24 after task 8 landed, submit as --array=11,12 --------------
  # Task 8 gave abec4d2c a MixMHC2pred reduction of 29.5% +- 19.2 (94/98) on the PINNED
  # 98 chains. The deployed model's MixMHC2pred figure (33.3% +- 17.2, 95/100 improved,
  # validate_mixmhc2pred_sweep_f16b51e6) is on a **100-protein** sample, so the two are
  # NOT comparable -- the same protein-set mismatch that task 2 exists to fix on the
  # class-I axis. 11 supplies the matched deployed baseline; 12 adds the seed replicate
  # so MixMHC2pred also gets a between-run error bar, as netMHCIIpan does via task 4.
  # This mismatch pattern has now appeared on all three orthogonal validators: every one
  # exists for the deployed model but on a different protein sample than the ablation.
  11) HASH=f16b51e6; VAL=mm ; LABEL=deployed              ;;
  12) HASH=58a9b5ae; VAL=mm ; LABEL=replicate_seed1       ;;
  *) echo "FATAL: bad array id $TASK" >&2; exit 1 ;;
esac
EP=200
SRC_TAG=$(printf "pwmtraj_%s_ep%03d" "$HASH" "$EP")
IN="data/output/eval_deimmunisation_${SRC_TAG}.csv"
OUT_TAG="netmhcmatch_${SRC_TAG}"

if [ -n "${CAPE_DRYRUN:-}" ]; then
  echo "task $TASK: ${HASH} (${LABEL}) ep${EP} validator=${VAL}"
  echo "  in : $PF/CAPE_MPNN/${IN}"
  ls -la "$PF/CAPE_MPNN/${IN}" 2>/dev/null || echo "  *** INPUT MISSING ***"
  echo "  out tag: ${OUT_TAG}"
  exit 0
fi

_act_ok=0
for _try in 1 2 3; do
  if source "$PF/slurm/activate_env.sh"; then _act_ok=1; break; fi
  echo "[netmhcmatch] activate_env attempt ${_try} failed; retrying in 60 s" >&2
  sleep 60
done
[ "$_act_ok" = "1" ] || { echo "[netmhcmatch] FATAL: env activation failed 3x" >&2; exit 1; }

cd "$PF/CAPE_MPNN"; mkdir -p "$PF/slurm/logs"

# Node-local TMPDIR is REQUIRED for netMHCpan specifically: its vendor script falls back
# to $NMHOME/tmp when TMPDIR is unset, and $NMHOME sits on the NFS mount, which
# intermittently EPERMs under concurrent write load (see 11_ckpt_robustness).
export TMPDIR="${SLURM_TMPDIR:-/tmp}/netmhcmatch_${SLURM_JOB_ID:-0}_${TASK}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

[ -f "$IN" ] || { echo "FATAL: missing input $IN" >&2; exit 1; }
# The sweep CSVs are all exactly 1 header + 2 arms x 98 proteins x 10 sequences. A short
# file would be silently scored on fewer sequences.
L=$(wc -l < "$IN")
[ "$L" -eq 1961 ] || { echo "FATAL: $IN has $L lines, expected 1961" >&2; exit 1; }

echo "[$(date)] task ${TASK}: ${HASH} (${LABEL}) ep${EP}  validator=${VAL}"
echo "           in=${IN}  out_tag=${OUT_TAG}"

if [ "$VAL" = "ii" ]; then
  # Results are immutable: never overwrite an existing validator output.
  if [ -e "data/output/validate_netmhciipan_${OUT_TAG}.csv" ]; then
    echo "  [skip] validate_netmhciipan_${OUT_TAG}.csv already present"; exit 0
  fi
  # --n_proteins 1000, not 98: select_proteins() sub-samples stratified by length
  # quartile, so a value far above the set size makes it take EVERY protein. A subset
  # would reintroduce the sampling mismatch this whole job exists to remove.
  # --k_seqs 10 matches the class-II convention used throughout (5 is the argparse default).
  PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/validate_with_netmhciipan.py \
      --in_csv "${IN}" --n_proteins 1000 --k_seqs 10 --out_tag "${OUT_TAG}"
elif [ "$VAL" = "mm" ]; then
  if [ -e "data/output/validate_mixmhc2pred_${OUT_TAG}.csv" ]; then
    echo "  [skip] validate_mixmhc2pred_${OUT_TAG}.csv already present"; exit 0
  fi
  MM2BIN="$PF/CAPE_MPNN/external/programs/MixMHC2pred/MixMHC2pred_unix"
  [ -x "$MM2BIN" ] || { echo "FATAL: MixMHC2pred_unix not executable at $MM2BIN" >&2; exit 1; }
  # --bin is argparse required=True. Omitting it is what killed tasks 1-2 of job 512943
  # AFTER 6 h of netMHCIIpan had already completed, and under `set -e` it took the
  # netMHCpan step down with it -- which is why those two tags still have no
  # MixMHC2pred/netMHCpan output today. --n_proteins 0 means all; --threshold 2.0
  # mirrors the netMHCIIpan %Rank cutoff used everywhere else. Do NOT pass --k_seqs:
  # this script does not define it and aborts if given it.
  PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/validate_with_mixmhc2pred.py \
      --in_csv "${IN}" --bin "${MM2BIN}" --n_proteins 0 --threshold 2.0 \
      --out_tag "${OUT_TAG}"
else
  if [ -e "data/output/validate_netmhcpan_${OUT_TAG}.csv" ]; then
    echo "  [skip] validate_netmhcpan_${OUT_TAG}.csv already present"; exit 0
  fi
  command -v netMHCpan >/dev/null 2>&1 || {
    echo "FATAL: netMHCpan not on PATH -- check external/programs/bin" >&2; exit 1; }
  # --k 5: five designs per arm for class I, the convention used for every class-I
  # number in this work (class II uses ten).
  PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/validate_with_netmhcpan.py \
      --in_csv "${IN}" --out_tag "${OUT_TAG}" --k 5
fi

echo "[$(date)] done: task ${TASK}  ${HASH} (${LABEL}) validator=${VAL}"
