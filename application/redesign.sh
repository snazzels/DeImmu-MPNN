#!/usr/bin/env bash
#
# redesign.sh -- apply the de-immunised (MHC-II-aware) ProteinMPNN model to a
# protein or protein-binder complex, holding chosen chains fixed as context.
#
# This is the primary "apply the fine-tuned model" entrypoint. It is a thin
# wrapper around stock ProteinMPNN (protein_mpnn_run.py): the de-immunisation
# lives entirely in the weights, not in a modified sampler. Running the base
# model vs. the fine-tuned model on the same input, with the same seed, is the
# apples-to-apples comparison reported throughout the manuscript.
#
# Usage:
#   ./redesign.sh <complex.pdb> <chains_to_redesign> <out_dir> [base|cape] [n_seq] [temp] [seed]
#
# Example (redesign binder chain C of a complex, target chains held fixed):
#   ./redesign.sh mycomplex.pdb C out/mycomplex_cape cape 24 0.1 37
#
# Notes:
#   * <chains_to_redesign> is a space- or comma-free chain string, e.g. "C" or "AC".
#     Every other chain in the PDB is automatically held FIXED as structural
#     context (stock ProteinMPNN behaviour -- verified in code). For a monomer,
#     pass its single chain id.
#   * "base" uses vanilla ProteinMPNN v_48_020; "cape" uses the fine-tuned,
#     de-immunised weights. See ../models/README.md to fetch the weights first.
#   * Set MHC2AWARE_ENV / PMPNN_DIR / WEIGHTS_* below or via environment.

set -euo pipefail

# --- configuration (override via environment) ---------------------------------
PY=${MHC2AWARE_PY:-python}                       # python with ProteinMPNN deps
PMPNN=${PMPNN_DIR:?set PMPNN_DIR to your ProteinMPNN checkout}/protein_mpnn_run.py
BASE_WEIGHTS=${WEIGHTS_BASE:-../models/vanilla}  # dir holding v_48_020.pt
CAPE_WEIGHTS=${WEIGHTS_CAPE:-../models/cape}     # dir holding cape_v3.pt (fetched)
BASE_MODEL=${BASE_MODEL:-v_48_020}
CAPE_MODEL=${CAPE_MODEL:-cape_v3}

# --- arguments ----------------------------------------------------------------
PDB=${1:?usage: redesign.sh <complex.pdb> <chains> <out_dir> [base|cape] [n] [T] [seed]}
CHAINS=${2:?missing chains-to-redesign}
OUT=${3:?missing out_dir}
WHICH=${4:-cape}
NSEQ=${5:-24}
TEMP=${6:-0.1}
SEED=${7:-37}

if [ "$WHICH" = "cape" ]; then
  WPATH=$CAPE_WEIGHTS; WMODEL=$CAPE_MODEL
elif [ "$WHICH" = "base" ]; then
  WPATH=$BASE_WEIGHTS; WMODEL=$BASE_MODEL
else
  echo "4th arg must be 'base' or 'cape', got '$WHICH'" >&2; exit 2
fi

mkdir -p "$OUT"
"$PY" "$PMPNN" \
  --pdb_path "$PDB" \
  --pdb_path_chains "$CHAINS" \
  --path_to_model_weights "$WPATH" \
  --model_name "$WMODEL" \
  --out_folder "$OUT" \
  --num_seq_per_target "$NSEQ" \
  --sampling_temp "$TEMP" \
  --seed "$SEED" \
  --batch_size 1

echo "done: $WHICH redesign of $PDB (redesigned chains '$CHAINS', $NSEQ seqs) -> $OUT"
echo "next: score MHC-II burden with  python score_burden.py  (see application/README.md)"
