#!/bin/bash
#SBATCH --job-name=ckpt_robust
#SBATCH --array=1-8%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ckptrob_%A_%a.out
#SBATCH --error=logs/ckptrob_%A_%a.err
# =========================== EDIT FOR YOUR CLUSTER ===========================
#SBATCH --constraint="rtx5090|rtx4090|rtx6000|rtxa5000"
# ----------------------------------------------------------------------------
# Checkpoint robustness: (A) benchmark the precursor at ITS OWN reported operating
# point, and (B) show where our reported checkpoint sits on the reward/quality curve.
#
# NO TRAINING. Every checkpoint below already exists on disk (verified 2026-08-20);
# save_model_every_n_epochs=1 means every epoch of every model was kept.
#
# ---------------------------------------------------------------------------
# WHY (A): Figure 3's MHC-I-only arm currently evaluates Gasser's models at
# `epoch_2` -- their EARLIEST shipped checkpoint, a tenth of their training -- while
# we report ourselves at epoch_200. Their repo ships exactly epoch_2/epoch_10/epoch_20
# and NO best.pt, and their paper's headline designs are `458340e4:epoch_20` and
# `e32b8ed0:epoch_20`, selected by the kink in mean TM-score. So epoch_20 is their
# reported operating point and epoch_2 is not. The ep2 numbers (MHC-II 2.8% and -0.2%)
# are the ones that make our double-dissociation X-pattern crisp, which is exactly why
# this has to be re-measured rather than left alone.
#
# Equal epoch counts across the two papers are impossible (their models exist only at
# 2/10/20, ours at 1..200) and are not the right target anyway. The rule adopted is:
# COMPARE EACH MODEL AT THE OPERATING POINT ITS OWN AUTHORS REPORT -- epoch_20 for
# Gasser, epoch_200 for us.
#
# Expect the number to move: at epoch 2 the model has barely drifted from base, so a
# ~0% incidental MHC-II effect is unsurprising, and MHC-I/MHC-II cross-talk is real in
# our own data (our MHC-II-only model still cuts MHC-I by 6.3%). A non-zero MHC-II
# reduction at their epoch_20 would NOT overturn the claim that MHC-II must be targeted
# explicitly -- it is directionally safe -- but it shrinks the margin, and the honest
# number is the one that belongs in the figure.
#
# ---------------------------------------------------------------------------
# WHY (B): the paper reports the final checkpoint. That is a stated rule rather than a
# search, which is the point -- `best.pt` is chosen by lowest DPO validation loss, and
# that quantity (i) is a ranking score over self-generated preference pairs, not a
# measure of the designs, (ii) moves mostly with which proteins got re-drawn at each
# 2-epoch reload (all 20 of the 20 largest jumps land on a reload boundary, p=5e-5),
# and (iii) is not even a well-defined function of the training run: `best.pt` is
# recomputed from an EMPTY in-memory loss list on every resume, so for a model that hit
# the walltime it is selected over the resumed segment alone. Verified on f16b51e6 by
# md5: best.pt == epoch_189.pt, whose val loss (0.4795) is WORSE than epoch 137's
# (0.4677) from before the epoch-176 resume. Selecting on it would mean reporting the
# luckiest draw of whichever segment happened to be last.
#   (An earlier revision of this comment claimed `best` is ANTI-correlated with the
#   reported objective, from 7e908919's 32.7% at best vs 36.7% at epoch_200. DO NOT
#   re-assert that: on matched proteins the gap does not survive -- ep137 48.6% vs
#   ep200 49.8%, a 1.2 pp difference against an SE-of-difference of ~2.4 pp. The
#   honest claim is that the checkpoint choice barely matters on the plateau, which
#   is a stronger argument than anti-correlation because this array measures it.)
#
# What the endpoint rule does need is evidence it is not PAST the point where designs
# degrade -- the real DPO failure mode, which Gasser hit visibly (checkpoints emitting
# constant chains). This array supplies it as a curve rather than an argument: epitope
# reduction AND sequence recovery at 25/50/100/137/150/200, all on the SAME pinned 98
# chains, so the val-loss-optimal epoch appears on the curve as a labelled point and
# the choice becomes visible instead of arguable.
#
# ⚠️ CORRECTED 2026-08-21: epoch 137 is f16b51e6's lowest-val-loss epoch, NOT its
# `best.pt`. The file `--ckpt best` actually loads is byte-identical to epoch_189.pt
# (md5 e894a7d4), because the 2026-08-17 resume at epoch 176 re-selected `best` over
# epochs 177-200 only. Label the 137 point "val-loss minimum" in any figure; to put
# the real `best.pt` on the curve, add an epoch_189 task (eval-only, ~1 h).
#
# THIS IS A ROBUSTNESS DISPLAY, NOT A RE-SELECTION. Reading the curve and then picking
# the best-looking epoch would put us straight back to choosing on the outcome we
# report. The reported checkpoint stays epoch_200 whatever the curve looks like. If the
# curve is still RISING at 200, that is a limitation to state (we stopped arbitrarily
# and more was available), not a licence to move the goalposts.
#
# Both axes come from one pass per checkpoint because evaluate_deimmunisation.py now
# emits a `recovery` column (added 2026-08-20). seq_recovery_rerun.py was NOT used: it
# draws its own protein sample, which would reintroduce across-checkpoint sampling
# mismatch -- the very bug --protein_list exists to remove.
#
#   task 1  458340e4 epoch_20   Gasser cfg A at their reported point   full validators
#   task 2  e32b8ed0 epoch_20   Gasser cfg B at their reported point   full validators
#   task 3  f16b51e6 epoch_25   trajectory                             PWM + recovery
#   task 4  f16b51e6 epoch_50   trajectory                             PWM + recovery
#   task 5  f16b51e6 epoch_100  trajectory + netMHCIIpan cross-check
#   task 6  f16b51e6 epoch_137  trajectory  (val-loss minimum; NOT best.pt)  PWM + recovery
#   task 7  f16b51e6 epoch_150  trajectory                             PWM + recovery
#   task 8  f16b51e6 epoch_200  trajectory endpoint + netMHCIIpan      (anchors the
#                               curve to the reported 37.9% headline)
#   task 9  458340e4 epoch_2    Gasser cfg A, low drift    PWM + recovery + netMHCIIpan
#   task 10 e32b8ed0 epoch_2    Gasser cfg B, low drift    PWM + recovery + netMHCIIpan
#   task 11 458340e4 epoch_10   Gasser cfg A, mid drift    PWM + recovery + netMHCIIpan
#   task 12 e32b8ed0 epoch_10   Gasser cfg B, mid drift    PWM + recovery + netMHCIIpan
#
# ⚠️ TASKS 1-8 ARE ALL COMPLETE (job 512943). A bare `sbatch` of this file re-runs
# --array=1-8 and every task will correctly abort on the never-overwrite guard —
# that is what happened to job 517525 on 2026-08-24, which looked like a crash and
# was not. The outstanding work is tasks 9-12.
#
#   Gasser low/mid drift: sbatch --array=9-12 11_ckpt_robustness.sbatch   <-- OUTSTANDING
#   Gasser arm only:      sbatch --array=1-2 11_ckpt_robustness.sbatch    (done)
#   Trajectory only:      sbatch --array=3-8 11_ckpt_robustness.sbatch    (done)
#
# ---------------------------------------------------------------------------
# WHY TASKS 9-12 EXIST (added 2026-08-24) — TWO reasons, and the second is the one
# that was not previously understood.
#
# (i) DRIFT-MATCHING THE EXTERNAL CONTROL. Gasser's models are the MHC-I-only arm of
# Figure 3, i.e. the external drift control on the MHC-II axis. Their epoch_20 sits at
# ~8 points of recovery cost, whereas we report ourselves at ~1.9. A control measured
# at 4x our drift cannot separate "MHC-II must be targeted explicitly" from "they had
# simply drifted further", and their incidental MHC-II reduction at ep20 is large. The
# ep2/ep10 points bracket OUR operating point, so the comparison becomes drift-matched
# in the same way the internal mhc2only control now is.
#
# (ii) THE EXISTING ep2 NUMBERS ARE ON THE WRONG PROTEIN SET. Figure 3's MHC-I-only
# arm uses eval_deimmunisation_mhci_only_{458340e4,e32b8ed0}_ep2_seed42 (MHC-II 2.8%
# and -0.2%) — and those CSVs contain **100 proteins, of which only 44 are in the
# pinned 98** (measured 2026-08-24: 56 present only there, 54 present only in the
# pinned list). The ckptrob_gasser*_ep20 CSVs match the pinned list exactly, 0
# differences. So the figure currently compares our 98-chain numbers against a
# 44-chain overlap, and no amount of re-labelling fixes that — the ep2 point has to be
# re-measured on --protein_list to be paired protein-for-protein with anything else we
# report. This is a comparability defect, not just an unequal-budget one.
#
# VALIDATORS: NETMHC2=1, deliberately NOT FULL=1. Checked on disk 2026-08-24: the
# ep20 tags have eval_* and validate_netmhciipan_* outputs and NOTHING ELSE — no
# validate_mixmhc2pred_* and no validate_netmhcpan_* were ever produced for them
# (steps 3-4 died on the missing --bin documented below and could not be re-run,
# because the never-overwrite guard then blocked the task). netMHCIIpan is therefore
# the only validator on which ep2/ep10/ep20 can be compared on equal footing, and it
# is the MHC-II axis the incidental-reduction claim lives on. Setting FULL=1 here
# would produce two validators the ep20 reference lacks and still not make the set
# uniform. Runtime is dominated by netMHCIIpan: budget ~6-8 h per task, NOT the ~30
# min a PWM-only point takes. All four run concurrently.
set -euo pipefail

export PF="${CAPE_ROOT:?set CAPE_ROOT to the project root (see README)}"
export MM2BIN=$PF/CAPE_MPNN/external/programs/MixMHC2pred/MixMHC2pred_unix
export PATH="$PF/CAPE_MPNN/external/programs/bin:$PATH"

# Same GPU-visibility guard as 09/10: a node that admits more --gres=gpu:1 tasks than
# it has usable GPUs otherwise kills us in the preamble (t38cn047).
if ! nvidia-smi -L 2>/dev/null | grep -q '^GPU 0'; then
  echo "[ckptrob] no usable GPU on $(hostname) (restart ${SLURM_RESTART_COUNT:-0})" >&2
  if [ "${SLURM_RESTART_COUNT:-0}" -lt 3 ]; then
    scontrol requeue "$SLURM_JOB_ID" && sleep 60 && exit 0
  fi
  exit 1
fi

source "$PF/slurm/activate_env.sh"
cd "$PF/CAPE_MPNN"; mkdir -p "$PF/slurm/logs"

# Node-local TMPDIR, required for netMHCpan specifically: its vendor script falls back
# to $NMHOME/tmp when TMPDIR is unset, and $NMHOME is on the NFS mount, which
# intermittently EPERMs under concurrent write load.
export TMPDIR="${SLURM_TMPDIR:-/tmp}/ckptrob_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

# The SAME pinned set as 10_matched_eval.sbatch. Sharing it is the point: every number
# this array produces is then paired with the matched-eval numbers protein-for-protein.
LIST="$PF/CAPE_MPNN/data/output/matched_eval_protein_list.txt"
[ -f "$LIST" ] || { echo "missing protein list $LIST"; exit 1; }

# ID / CKPT / LABEL, plus which validators to run.
#   FULL=1 -> netMHCIIpan + MixMHC2pred + netMHCpan (needed to reproduce both Fig 3
#             columns for the Gasser arm: MHC-II via netMHCIIpan, MHC-I via netMHCpan)
#   FULL=0 -> PWM + recovery only; NETMHC2=1 adds just the netMHCIIpan cross-check
FULL=0; NETMHC2=0
case "${SLURM_ARRAY_TASK_ID}" in
  1) ID=458340e4; CKPT=epoch_20;  LABEL=gasser458_ep20;  FULL=1 ;;
  2) ID=e32b8ed0; CKPT=epoch_20;  LABEL=gasser32b_ep20;  FULL=1 ;;
  3) ID=f16b51e6; CKPT=epoch_25;  LABEL=traj_ep025 ;;
  4) ID=f16b51e6; CKPT=epoch_50;  LABEL=traj_ep050 ;;
  5) ID=f16b51e6; CKPT=epoch_100; LABEL=traj_ep100; NETMHC2=1 ;;
  6) ID=f16b51e6; CKPT=epoch_137; LABEL=traj_ep137 ;;   # val-loss minimum (best.pt is ep189)
  7) ID=f16b51e6; CKPT=epoch_150; LABEL=traj_ep150 ;;
  8) ID=f16b51e6; CKPT=epoch_200; LABEL=traj_ep200; NETMHC2=1 ;;
  # Gasser at low/mid drift, on the PINNED list. See the header for both reasons.
  # Tags carry ep02/ep10 rather than ep2 so they sort next to ep20 and never collide
  # with the old off-list mhci_only_*_ep2_seed42 files, which are left untouched.
  9)  ID=458340e4; CKPT=epoch_2;  LABEL=gasser458_ep02; NETMHC2=1 ;;
  10) ID=e32b8ed0; CKPT=epoch_2;  LABEL=gasser32b_ep02; NETMHC2=1 ;;
  11) ID=458340e4; CKPT=epoch_10; LABEL=gasser458_ep10; NETMHC2=1 ;;
  12) ID=e32b8ed0; CKPT=epoch_10; LABEL=gasser32b_ep10; NETMHC2=1 ;;
  *) echo "bad array id"; exit 1 ;;
esac
TAG="ckptrob_${LABEL}"

CKPT_FILE="$PF/CAPE_MPNN/artefacts/CAPE-MPNN/models/${ID}/ckpts/${CKPT}.pt"
[ -f "$CKPT_FILE" ] || { echo "FATAL: missing checkpoint $CKPT_FILE"; exit 1; }

# Refuse to clobber a previous run of this task (results are immutable).
CSV="data/output/eval_deimmunisation_${TAG}.csv"
# Written as an explicit if, NOT `[ -e ] && { exit 1; }`: a leading-test-fails `&&` list
# under `set -e` is the footgun already hit in 09_reanchor_train.sbatch.
if [ -e "$CSV" ]; then
  echo "FATAL: $CSV exists -- refusing to overwrite. Move it aside or change TAG." >&2
  exit 1
fi

echo "[$(date)] ckpt robustness: ${LABEL}  id=${ID} ckpt=${CKPT} full=${FULL} netmhc2=${NETMHC2}"
echo "           list=$(grep -cvE '^[[:space:]]*(#|$)' "$LIST") proteins  tag=${TAG}"

# 1) designs on the PINNED protein set + PWM burden + sequence recovery (GPU).
#    --seed 42 and --k_seqs 10 match 10_matched_eval.sbatch so the two jobs' numbers
#    are directly comparable.
PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/evaluate_deimmunisation.py \
    --model_hash "${ID}" --ckpt "${CKPT}" --protein_list "${LIST}" \
    --k_seqs 10 --seed 42 --out_tag "${TAG}"

# 2) netMHCIIpan-direct MHC class II (CPU). --n_proteins 1000, not 100: select_proteins()
#    sub-samples stratified by length quartile, so a value far above the set size makes
#    it take EVERY protein. A subset would reintroduce the sampling mismatch.
if [ "$FULL" = "1" ] || [ "$NETMHC2" = "1" ]; then
  PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/validate_with_netmhciipan.py \
      --in_csv "${CSV}" --n_proteins 1000 --k_seqs 10 --out_tag "${TAG}"
fi

if [ "$FULL" = "1" ]; then
  # 3) MixMHC2pred, orthogonal MHC-II predictor (CPU).
  #    No --k_seqs: validate_with_mixmhc2pred.py does not define it and aborts if passed.
  #    --n_proteins 0 means "all".
  #    --bin is REQUIRED (argparse required=True) and --threshold 2.0 matches the
  #    working reference 02_sweep_eval.sbatch. Omitting --bin is what killed tasks 1-2
  #    of job 512943 after 6 h of netMHCIIpan, and under `set -e` it took the netMHCpan
  #    step down with it.
  PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/validate_with_mixmhc2pred.py \
      --in_csv "${CSV}" --bin "${MM2BIN}" --n_proteins 0 --threshold 2.0 \
      --out_tag "${TAG}"

  # 4) netMHCpan MHC class I (CPU) -- this is the axis on which the Gasser models are
  #    supposed to be strong, so it is the half of Fig 3 that makes their arm a valid
  #    MHC-I-only representative rather than just a weak model.
  if command -v netMHCpan >/dev/null 2>&1; then
    PYTHONPATH="$PF/CAPE_MPNN/libs" python tools/validate_with_netmhcpan.py \
        --in_csv "${CSV}" --out_tag "${TAG}" --k 5
  else
    echo "[ckptrob] netMHCpan not on PATH -- MHC-I step SKIPPED for ${LABEL}."
    echo "          Run later:  python tools/validate_with_netmhcpan.py \\"
    echo "            --in_csv ${CSV} --out_tag ${TAG} --k 5"
  fi
fi

echo "[$(date)] done: ${LABEL}"
