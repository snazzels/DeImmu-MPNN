# Application — apply the de-immunised model to your own design

This is the focus of the release: taking the fine-tuned model and using it to
redesign a protein or a binder–target complex so that predicted MHC class II
epitope burden drops, without destroying structure or binding.

The de-immunisation lives **entirely in the weights**. There is no modified
sampler: you run stock ProteinMPNN (`protein_mpnn_run.py`) with the fine-tuned
checkpoint, holding the chains you want to preserve (e.g. the target of a binder)
fixed as context. Running the base model and the de-immunised model on the same
input with the same seed is the apples-to-apples comparison used throughout the
paper.

## What runs on arbitrary inputs vs. what is paper-specific

Turnkey CLIs (run on any input, `--help` for options):
- **`redesign.sh`** — run base or de-immunised ProteinMPNN on any PDB, choosing
  the chain(s) to redesign. Start here.
- **`build_complex.py`** — assemble a clean minimal binder+target complex from
  any PDB deposition.
- **`score_burden.py`** — MHC-II burden + interface/core/surface breakdown for
  any complex and any set of labelled design FASTAs.

Released **as run for the paper** (contain the 11-complex `PANEL` dict and/or
absolute paths; adapt before use): `contacts.py`, `boltz_gen_yaml.py`,
`boltz_parse.py`, `consolidate_panel.py`, and `af2_initial_guess/`. They are
provided for exact reproducibility and as templates.

`make_panel_figure.py` is self-contained: it reads the three shipped result JSONs
next to it (`burden_results.json`, `iptm_full24.json`,
`af2_initial_guess/af2ig_native_results.json`) and regenerates the manuscript's
11-complex `panel11_figure.png` (plus `panel11_results.csv` / `panel11_summary.txt`)
byte-for-byte with no editing.

## Pipeline

### 1. Build a clean minimal complex (binders only)
`build_complex.py` reads a PDB deposition and writes a minimal complex keeping
only the binder chain and its true target chain(s): standard amino acids only,
altloc A, single model, HETATM waters/ligands dropped.
```bash
python build_complex.py --pdb 9nds.pdb --binder A --targets C D E \
    --out complex/9nds_cplx.pdb
```
Use `contacts.py` first if you are unsure which chains actually form the
interface — several PDB depositions pair non-adjacent chains (e.g. 9cce pairs
A–D, not A–C).

### 2. Redesign (the core step)
```bash
export PMPNN_DIR=/path/to/ProteinMPNN
export MHC2AWARE_PY=/path/to/env/bin/python
export WEIGHTS_CAPE=/path/to/DeImmu-MPNN/models   # see ../models/README.md

# only chain C is rewritten; all other chains are held fixed as context
./redesign.sh  complex/mycomplex_cplx.pdb  C  out/mycomplex_cape  cape  24 0.1 37
./redesign.sh  complex/mycomplex_cplx.pdb  C  out/mycomplex_base  base  24 0.1 37
```
`--num_seq_per_target 24`, `--sampling_temp 0.1`, `--seed 37` are the paper's
settings. For a monomer, pass its single chain id and skip steps 4–5.

### 3. Score predicted MHC-II burden
`score_burden.py` counts predicted MHC-II presented 15-mer windows per sequence
with `Mhc2PredictorPwm` over the six-allele HLA-DRB1 panel
(`DRB1_0101/0301/0401/0701/1101/1501`), and classifies each redesigned binder
residue as interface / core / surface (biotite ΔSASA) so you can check the
reduction is not concentrated at the binding interface.
```bash
python score_burden.py \
    --complex complex/9nds_cplx.pdb --binder A \
    --pwm $MHC2_PWM --libs /path/to/CAPE_MPNN/libs \
    --designs base:out/9nds_base/seqs/9nds_cplx.fa \
              cape:out/9nds_cape/seqs/9nds_cplx.fa \
    --out 9nds_burden.json
```
`--pwm`/`--libs` fall back to the `MHC2_PWM`/`CAPE_LIBS` env vars. For a monomer
there is no interface, so residues fall into core/surface only.

### 4. Validate binding — Boltz-2 (optional but recommended)
`boltz_gen_yaml.py` emits Boltz-2 YAML inputs for every design (natural target
folded with an MMseqs2-server MSA, de novo binder as single sequence — this
combination matters; a single-sequence target inflates ipTM). Run Boltz-2, then
`boltz_parse.py` extracts interface ipTM. Confident interface ≈ ipTM > 0.5.

### 5. Validate binding — AF2 initial-guess (optional, second validator)
`af2_initial_guess/` is a no-PyRosetta reimplementation of the Bennett et al.
(2023) "initial guess" protocol (AlphaFold2 params required). Pass criterion is
`pae_interaction < 10 Å`. Two methodologically unrelated validators agreeing is
the evidence used in the paper; each also flags its own known weak spots
(large / no-MSA targets), so treat single-validator floors accordingly.

### 6. Consolidate + figure
`consolidate_panel.py` merges burden + Boltz + AF2-ig into the panel table;
`make_panel_figure.py` renders the multi-complex figure.

## Reference numbers (paper's 11-complex panel)

Mean epitope-burden reduction **+39.7%** (11/11 positive); Boltz-2 ipTM base
0.866 vs cape 0.862; AF2-ig pass rate base 40.5% vs cape 42.8%. A per-complex
breakdown is in the manuscript's panel table.
