#!/usr/bin/env bash
#
# fetch_weights.sh -- download the trained checkpoints from Zenodo into models/.
#
# The released checkpoints are hosted on Zenodo (not committed to git, because they
# are large and because trained weights are immutable release artefacts). Fill in
# ZENODO_RECORD once the deposition exists.
#
# NINE checkpoints are released. Every hyperparameter quoted below was read from
# that model's own dpo_hparams.yaml (see models/README.md); beta=0.05, 200 epochs,
# 1e6 examples/epoch and sampling temperature 0.2 are common to all nine.
#
#   -- deployed operating point (learning rate 1e-6) --
#
#   deimmu_deployed.pt              THE DEPLOYED MODEL -- every headline number, the
#                                   11-complex binder panel, allele transfer.
#                                   lambda=1.0, dual class I + II objective.
#   deimmu_mhc2only.pt              class II-only ablation at the SAME operating point
#                                   (mhc_1_predictor=none); the Figure 3A attribution
#                                   control. lambda=1.0. Differs from the deployed
#                                   model in that one setting alone.
#   deimmu_lambda0_lr1e-6.pt        lambda=0 arm of the acidic-composition regulariser
#                                   series at the deployed learning rate (Table 3).
#   deimmu_lambda05_lr1e-6.pt       lambda=0.5 arm of the same series (Table 3).
#   deimmu_deployed_seed_replicate.pt
#                                   seed replicate of the deployed model (seed=1);
#                                   supplies the between-run scale (1.5 points)
#                                   against which every small cross-checkpoint
#                                   difference in the paper is read.
#
#   -- earlier operating point (learning rate 3e-7) --
#
#   base_lambda0.pt                 lambda=0 arm of the charge-regulariser sweep.
#                                   A legacy name and a slight misnomer: it is the
#                                   lambda=0 FINE-TUNE, not the base model. The base
#                                   model is stock ProteinMPNN v_48_020, not
#                                   redistributed here.
#   cape_creg0.5.pt                 lambda=0.5 arm of the sweep.
#   cape_creg1.0.pt                 lambda=1.0 arm of the sweep. Same lambda as
#                                   deployed but a lower learning rate, so it is NOT
#                                   the deployed model -- use deimmu_deployed.pt for
#                                   the binder panel and every headline result.
#   deimmu_mhc2only_lr3e-7_lambda0.pt
#                                   class II-only ablation at the earlier learning
#                                   rate. NOTE lambda=0 here, NOT 1.0: this file
#                                   differs from deimmu_mhc2only.pt in BOTH learning
#                                   rate and lambda, which is why lambda is in the
#                                   name. Superseded for Figure 3A by
#                                   deimmu_mhc2only.pt; deposited because the
#                                   manuscript names it.
#
# Updated 2026-08-26: this script previously fetched only the three sweep arms, so it
# would have produced a working directory without the model the paper reports.
# Updated 2026-09-09: extended from six to nine. The three deployed-learning-rate
# regulariser checkpoints (deimmu_lambda0_lr1e-6, deimmu_lambda05_lr1e-6,
# deimmu_deployed_seed_replicate) back a main-text table and were missing, which is
# the same defect as the 2026-08-26 note above -- the third recurrence. Run
# ./check_manifest_sync.py to stop it happening a fourth time. The class II-only
# lr 3e-7 file was also renamed to carry its lambda (see the note on it above).
#
# After download, point application/redesign.sh at the weights via WEIGHTS_CAPE
# (see application/README.md and models/README.md).

set -euo pipefail

ZENODO_RECORD=${ZENODO_RECORD:-"REPLACE_WITH_ZENODO_RECORD_ID"}
DEST=$(cd "$(dirname "$0")" && pwd)

if [ "$ZENODO_RECORD" = "REPLACE_WITH_ZENODO_RECORD_ID" ]; then
  echo "ERROR: set ZENODO_RECORD to the Zenodo record id first (see models/README.md)." >&2
  echo "       e.g.  ZENODO_RECORD=1234567 ./fetch_weights.sh" >&2
  exit 1
fi

# The single source of truth for the released file set. check_manifest_sync.py
# asserts that this list, ZENODO_WEIGHTS.md and README.md all name the same nine
# files; keep the three in step or that check fails.
WEIGHTS=(
  deimmu_deployed.pt
  deimmu_mhc2only.pt
  deimmu_lambda0_lr1e-6.pt
  deimmu_lambda05_lr1e-6.pt
  deimmu_deployed_seed_replicate.pt
  base_lambda0.pt
  cape_creg0.5.pt
  cape_creg1.0.pt
  deimmu_mhc2only_lr3e-7_lambda0.pt
)

BASE_URL="https://zenodo.org/records/${ZENODO_RECORD}/files"
for f in "${WEIGHTS[@]}"; do
  echo "downloading $f ..."
  curl -fL --retry 3 -o "$DEST/$f" "${BASE_URL}/${f}?download=1"
done

# Refuse to report success on a partial download: a working directory missing one
# checkpoint silently reproduces a subset of the paper.
missing=()
for f in "${WEIGHTS[@]}"; do
  [ -s "$DEST/$f" ] || missing+=("$f")
done
if [ ${#missing[@]} -ne 0 ]; then
  echo "ERROR: ${#missing[@]} checkpoint(s) missing or empty after download:" >&2
  printf '       %s\n' "${missing[@]}" >&2
  exit 1
fi

echo "done. ${#WEIGHTS[@]} checkpoints in $DEST"
echo "verify against the SHA256SUMS in models/README.md before use."
