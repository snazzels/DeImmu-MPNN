# De-immunising de novo protein designs: MHC class II–aware ProteinMPNN

Code and models for **"DeImmu-MPNN: Reducing Immunogenicity Burden in
De Novo Protein Design"**
([manuscript reference / DOI to be added on publication]).

**Archived releases.** Code: [`10.5281/zenodo.23239981`](https://doi.org/10.5281/zenodo.23239981) · Trained
checkpoints and evaluation data: [`10.5281/zenodo.23239266`](https://doi.org/10.5281/zenodo.23239266)

This repository releases a fine-tuned ProteinMPNN that redesigns a protein
sequence to **reduce predicted MHC class II (T-helper) epitope burden** while
preserving structure and, for binders, predicted binding. It is intended to be
dropped into an existing de novo design workflow: given a backbone (a monomer or
a binder–target complex), it proposes sequences that a therapeutic-protein
developer would expect to carry a lower anti-drug-antibody (ADA) risk than the
sequences a naive design tool produces.

The model is a direct-preference-optimization (DPO) fine-tune of
[ProteinMPNN](https://github.com/dauparas/ProteinMPNN), extending
[CAPE-MPNN (Gasser et al., 2025)](https://github.com/hcgasser/CAPE_MPNN) from
MHC class I to the ADA-relevant MHC class II pathway. See
[Relationship to CAPE-MPNN](#relationship-to-cape-mpnn) and [License](#license).

> **Scope of the claim.** The objective reduces *predicted* MHC class II
> presentation, a mechanistically necessary precondition for a T-helper
> response. Whether this translates to reduced clinical immunogenicity is not
> established in this work, which is computational throughout; establishing it
> would require an experimental readout rather than a further predictor. See the
> manuscript's Supplementary Section S1 and
> [`analysis/supplementary_iedb/`](analysis/supplementary_iedb/).
>
> **What the model does, mechanistically.** A composition-matched permutation
> control settles this and it is worth knowing before you use the weights.
> Permuting the residues of every design — which preserves amino-acid composition
> exactly and destroys every binding register — retains **97.5%** of the reduction
> (41.36% original vs 40.34% permuted, paired difference +1.02 pp, 95% CI
> [−2.76, +4.81], n=98, *p*=0.59). **The reduction is compositional, not
> positional**: the model has learned an amino-acid usage prior that lowers
> netMHCIIpan's presented-window count, and it does *not* locate and rewrite
> individual epitopes. Do not expect it to remove a particular predicted epitope
> from a particular sequence.

## Headline result

On 100 held-out proteins the deployed model (`f16b51e6`) reduces netMHCIIpan-predicted
MHC class II presentation by **41.4 ± 18.9%** (93 of 98 evaluable proteins improved;
two present at most one base-model window on average, so their percentage reduction
is dominated by sampling noise and is excluded — see the note below), agreeing
with a second predictor (**MixMHC2pred, 33.3%**) and accompanied by a
**33.0%** netMHCpan MHC class I reduction (100/100). MixMHC2pred uses a different
architecture, but the two draw on partially overlapping mass-spectrometry ligand
data, and a compositional reduction (see the box above) is the kind of change a
second presentation predictor would be expected to register as well — so read the
agreement as consistent with the reduction rather than as independent
corroboration of it. There is no measurable cost to
self-consistency structure quality under either ESMFold2 or Boltz-2, and a 2.1-point
sequence-recovery change. It **exceeds** netMHCIIpan-based post-hoc filtering of
base-model ensembles (41.4% vs 26.3%), far exceeds a fast PWM-based filter (10.9%),
and stacks with filtering (48.1% when its own ensemble is PWM-filtered). Applied to
**11 published de novo binder–target complexes** (redesigning only the binder chain,
target held fixed) it cuts predicted MHC-II burden by **mean 50.4%** (11/11;
MixMHC2pred 40.4%, netMHCpan MHC-I 30.2%), while two independent structure
predictors — **Boltz-2** (ipTM base 0.866 vs 0.857, *p*=0.08) and **AF2
initial-guess** (interface pass rate 40.5% vs 44.3%) — show no systematic binding cost.

> **On the minimum-base guard (numbers updated 2026-08-26).** A per-protein
> percentage reduction needs a usable denominator. Proteins whose base designs
> present a mean of ≤1.0 windows are therefore excluded from the mean: previously
> only exactly-zero proteins were, so one protein averaging 0.10 windows
> contributed −300% and depressed the headline from 41.4% to 37.9%. Every number
> above is the guarded value. In the data tree, prefer `*_guarded_summary.txt`
> over the `*_summary.txt` beside it; `analysis/recompute_guarded_reduction.py`
> derives the former from the unchanged CSVs, so the correction is auditable
> rather than a re-scoring. The one arm not yet corrected is the %Rank cutoff
> series, which is flagged in `data/MANIFEST.md`.

## Repository layout

```
application/          THE FOCUS: apply the fine-tuned model to your own design
  redesign.sh           run base or de-immunised ProteinMPNN on a PDB (target chains fixed)
  build_complex.py      assemble a clean minimal binder+target complex from a PDB
  contacts.py           pairwise chain-contact analysis (identify the true binder/target pairing)
  score_burden.py       predicted MHC-II epitope burden per sequence (Mhc2PredictorPwm)
  boltz_gen_yaml.py     emit Boltz-2 inputs (target MSA, binder single-seq) for all designs
  boltz_parse.py        parse Boltz-2 ipTM confidence
  af2_initial_guess/    AF2 initial-guess binding validator (no-PyRosetta reimpl of Bennett 2023)
  consolidate_panel.py  merge burden + Boltz + AF2-ig into the panel table
  make_panel_figure.py  the multi-complex panel figure

training/             DPO fine-tuning (extends CAPE-MPNN)
  cape-mpnn.py          training entrypoint (MHC-I + MHC-II reward + acidic-composition regulariser)

analysis/             evaluation reported in the main text
  selfconsistency_*.py       ESMFold/Boltz-2 self-consistency (structure fidelity)
  test_allele_generalisation.py, analyse_transfer_mechanism.py   held-out allele / DQ / DP transfer
  quantify_charge_shift.py, test_charge_regulariser.py           the electrostatic-shortcut control
  build_register_pwm.py      register-anchored PWM (sharper netMHCIIpan approximation)
  evaluate_deimmunisation.py, evaluate_mhc1_arm.py               headline burden evaluation
  filtering_baseline.py      post-hoc ensemble-filtering baseline (strategy 2)
  validate_with_netmhciipan.py   re-score with netMHCIIpan itself (not the training PWM)
  supplementary_iedb/        SI S1: the negative validation result
    analyse_iedb_binding_signal.py   netMHCIIpan vs measured immunogenicity (AUC 0.492)
    train_iedb_classifier.py         IEDB T-cell-assay classifier (organism-holdout collapse)
    validate_iedb_knn_signal.py      k-NN signal, same collapse (Phase-0 gate)

data_prep/            IEDB mining + peptide ranking (derives training/eval datasets)
  mine_IEDB_MHC2.py, mine_IEDB_negatives.py, MHC-II_rank_peptides.py, MHC-I_rank_peptides.py

notebooks/            Colab notebook for the sequence-generation step (no local install)

figures/              plotting scripts for every data-driven manuscript figure
  make_fig_si_sweep_combined.py            SI figure: Pareto front + foldability cliff
  make_fig_attribution_strategy_v7_*.py    Fig 3: objective attribution + strategy comparison
  make_fig_allele_reanchored.py            Fig 4: held-out allele transfer
  README_FIGURES.md                        figure -> generator map; which figures are
                                           hand-drawn schematics with no script

training/slurm/       cluster job scripts for every training and evaluation run
  09_reanchor_train.sbatch, 01_sweep_train.sbatch, 02_sweep_eval.sbatch, ...
  check_train_flags.py  MANDATORY preflight: asserts every hashed hyperparameter is
                        passed explicitly (two silent wrong-objective runs came from
                        relying on an argparse default)

data/                 ALL data behind every number in the paper -- see data/MANIFEST.md
  monomer_eval/         100 held-out proteins: netMHCIIpan, MixMHC2pred, netMHCpan,
                        %Rank sensitivity, and the Fig 3A attribution arms
  binder_panel/         the 11 binder complexes: burden, Boltz-2 ipTM, AF2-ig
  sweep/                all 15 beta x learning-rate configurations
  selfconsistency/      ESMFold2 and Boltz-2 paired self-consistency across the front
  allele_generalisation/  held-out DRB1, DRB3/4/5, HLA-DQ, HLA-DP
  charge/               net charge / pI / acidic-fraction shifts
  headroom/             per-design base vs fine-tuned burden for all 111 entities

models/               how to fetch the trained checkpoints (Zenodo) -- see models/README.md
```

`data/MANIFEST.md` maps each directory to the specific paper claim it supports and
the script that produced it. It is regenerated by `tools_build_deposition.py` in the
working tree.

## Try it in Colab (no install)

The sequence-generation step runs in the browser with no local setup:
[**Open in Colab**](https://colab.research.google.com/github/snazzels/DeImmu-MPNN/blob/main/notebooks/deimmunize_proteinmpnn_colab.ipynb)
([`notebooks/deimmunize_proteinmpnn_colab.ipynb`](notebooks/deimmunize_proteinmpnn_colab.ipynb)).
Give it a PDB (by ID or upload) and the chain(s) to redesign; it fetches the released weights,
runs the de-immunised model and base ProteinMPNN, returns the sequences, and can optionally
validate a design with **Boltz-2** (interface ipTM for a binder, or pTM/pLDDT for a monomer).
Quantifying the burden reduction is done locally with `application/score_burden.py`. (The Colab
link activates once the repository is public.)

## Quickstart — apply the de-immunised model

Applying the model is deliberately simple: the de-immunisation lives in the
weights, so you run stock ProteinMPNN with the fine-tuned checkpoint. See
[`application/README.md`](application/README.md) for the full walk-through.

```bash
# 0. one-time: set up the environment and fetch weights
#    (see "Installation" below and models/README.md)
export PMPNN_DIR=/path/to/ProteinMPNN
export MHC2AWARE_PY=/path/to/env/bin/python

# 1. (binders only) build a clean minimal complex from a PDB deposition
python application/build_complex.py --pdb mytarget.pdb --binder C --targets A B \
    --out complex/mycomplex_cplx.pdb

# 2. redesign -- target chains held fixed automatically; only chain C is rewritten
./application/redesign.sh  complex/mycomplex_cplx.pdb  C  out/mycomplex_cape  cape  24 0.1 37
./application/redesign.sh  complex/mycomplex_cplx.pdb  C  out/mycomplex_base  base  24 0.1 37

# 3. score predicted MHC-II epitope burden (base vs de-immunised)
python application/score_burden.py --complex complex/mycomplex_cplx.pdb --binder C \
    --pwm $MHC2_PWM --libs /path/to/CAPE_MPNN/libs \
    --designs base:out/mycomplex_base/seqs/mycomplex_cplx.fa \
              cape:out/mycomplex_cape/seqs/mycomplex_cplx.fa

# 4. (optional) confirm binding is retained with an independent structure predictor
#    Boltz-2 (application/boltz_gen_yaml.py) or AF2 initial-guess (application/af2_initial_guess/)
```

For a **monomer** (no target), pass its single chain id at step 2 and skip the
binding validation.

## Installation

The pipeline spans several tools, each with its own environment. Exact versions
and setup notes are in [`environment.md`](environment.md). In brief:

- **ProteinMPNN + de-immunised weights** — a PyTorch env (`cape_mpnn`); the only
  requirement for the core `redesign.sh` step.
- **MHC-II burden scoring** — `Mhc2PredictorPwm` (position-weight-matrix
  approximation to netMHCIIpan) and, for the exact re-scoring in the paper,
  a local install of [netMHCIIpan-4.3](https://services.healthtech.dtu.dk/) (DTU
  academic licence; not redistributable).
- **Binding validators** — [Boltz-2](https://github.com/jwohlwend/boltz) and the
  AF2 initial-guess reimplementation (AlphaFold2 params required; no PyRosetta).

## Reproducing the paper

- **Train** the model: `training/cape-mpnn.py` (200-epoch DPO run; joint MHC-I +
  MHC-II reward with the acidic-composition regulariser). See
  [`training/README.md`](training/README.md). The exact cluster invocations are in
  `training/slurm/` — run `training/slurm/check_train_flags.py` before submitting
  anything, for the reason given in that file's docstring.
- **Main-text evaluation**: the `analysis/` scripts, one per result.
- **Figures**: the `figures/` scripts; see `figures/README_FIGURES.md` for which
  figure each one produces and which figures are hand-drawn schematics instead.
- **Verify without recomputing**: every number in the paper has its underlying
  file in `data/`, indexed by `data/MANIFEST.md`. Recomputing the evaluations needs
  netMHCIIpan-4.3, netMHCpan-4.1 and MixMHC2pred, which are licensed separately
  and not redistributed here.
- **Supplementary S1 (negative result)**: `analysis/supplementary_iedb/`.
- **Derived datasets**: `data_prep/` regenerates the training/eval peptide sets
  from a fresh IEDB export (raw IEDB not redistributed here; see below).

## Models

Checkpoints are deposited on Zenodo and fetched with a script rather than committed
to git; see [`models/README.md`](models/README.md). Base is stock ProteinMPNN
`v_48_020`, not a released checkpoint here.

**Two checkpoints are meant for use, and they are not ordered by quality.** They
differ in one training term — whether the co-trained MHC class I objective is
present — and therefore in which presentation pathways they address:

| | deployed `f16b51e6` | class II-only `abec4d2c` |
|---|---|---|
| MHC class II reduction | 40.7% | 41.4% |
| MHC class I reduction | 32.9% | 2.2% |
| HLA-DP (highest-burden heterodimer) | 3.1%, n.s. | 51.7% |
| HLA-DQ / DRB3-4-5 / held-out DRB1 | covered | covered |
| sequence-recovery cost | 1.89 pp | 1.76 pp |
| structural self-consistency | no measurable cost | no measurable cost |
| net-charge shift | −1.52 units | −3.25 units |
| designs below pI 5 | 33% (base 31%) | 41% (base 31%) |
| **use it when** | CD8 epitope load also matters | HLA-DP coverage is required |

All figures above are on one matched 98-protein sample against one base arm.
**Use `f16b51e6` by default.** Prefer `abec4d2c` if your HLA coverage must include
HLA-DP, and check the isoelectric point of the resulting designs if you do — it
reaches its charge shift by stripping basic residues, a route the acidic-composition
regulariser does not constrain.

**The remaining checkpoints support specific results and are not design tools**:
`dce9dd92` (λ=0) and `1c908bb6` (λ=0.5) are the regulariser series at the deployed
learning rate; `58a9b5ae` is a seed replicate of the deployed model that supplies the
between-run scale (1.5 points); `7777d75f` is the class II-only ablation at the sweep
learning rate; and the earlier λ series at learning rate 3×10⁻⁷ underlies the
charge-regulariser sweep table. See [`models/README.md`](models/README.md) for the
full manifest, which records what each id is a function of and which checkpoint of it
each result was produced from.

## Compatibility and runtime

The de-immunisation lives entirely in the weights. The released checkpoint is a
ProteinMPNN checkpoint in the stock format — 118 tensors, 1,660,485 parameters,
with the same tensor names, shapes, `num_edges=48` and `noise_level=0.2` as
`v_48_020`. Every tensor differs from base (it is a full fine-tune, not an adapter)
but nothing about the interface does, so running it is running stock ProteinMPNN
with `--path_to_model_weights` pointed at the released file.

| target | loads | status |
|---|---|---|
| ProteinMPNN, stock `v_48_020` | yes | verified; the reference configuration |
| soluble-MPNN | yes | same architecture; interaction **untested** here |
| **LigandMPNN** | **no** | different architecture; parameters not interchangeable |
| BindCraft | yes | uses ProteinMPNN for sequence design; unmodified here |
| RFdiffusion | n/a | operates on backbones, before sequence design |

**Runtime.** On one NVIDIA RTX 4090, designing a 107-residue chain takes
**12.3 ms per sequence** with either the base or the de-immunised checkpoint
(measured as the slope from 32 to 512 sequences, excluding ≈1.4 s of interpreter
start-up and model loading). The two are identical within run-to-run variation, as
they must be — same architecture, same parameter count, same sampling procedure.
24 designs take about 0.3 s of GPU time. **There is no inference-time cost to
using the de-immunised weights.**

The cost asymmetry that motivates the PWM surrogate is on the *reward* side. In
this work's runs netMHCIIpan-4.3 scored 5.96×10⁶ peptide–allele pairs in 8 h 56 min
on 8 CPU cores (≈185 s⁻¹, including input preparation and parsing), while the PWM
approximation scores ≈5.5×10⁵ s⁻¹ single-threaded — roughly three orders of
magnitude. Since DPO scores every candidate in every preference pair at every step,
netMHCIIpan in the training loop is a different order of computation, not merely a
slower run. It is used for final re-scoring instead.

## Data

- **IEDB** peptide data were obtained under IEDB's terms of use from
  <https://www.iedb.org/database_export_v3.php> and are **not redistributed**
  here. `data_prep/` reproduces the derived datasets from a fresh export.
- The **11 binder–target complexes** are public PDB depositions (accession codes
  in the manuscript's panel table), drawn from the
  [Protein Design Archive](https://pragmaticproteindesign.bio.ed.ac.uk/pda/).
- All **derived evaluation data** — the per-sequence and per-protein predicted-burden
  tables, confidence scores, self-consistency metrics and summary statistics behind
  every figure, table and quoted number — **are included** under `data/`, organised by
  the claim each supports (`data/MANIFEST.md`). This is derived data: it contains
  designed sequences and predictor outputs, not redistributed IEDB records.
- The **MHC-II PWM training inputs are not redistributed.** They are derived directly
  from netMHCIIpan output, which DTU licenses for academic use and does not permit
  redistributing. This follows upstream CAPE-MPNN, which likewise ships no
  netMHCpan-derived content. They are fully regenerable — see below.

### Regenerating the MHC-II PWMs

The PWM is the fast surrogate the DPO objective is trained against, because
netMHCIIpan is too slow to call inside the training loop. Rebuilding it needs a
local netMHCIIpan install (DTU academic licence) and roughly a day of CPU time.

```bash
data_prep/MHC-II_rank_peptides.py \
    --alleles DRB1_0101+DRB1_0301+DRB1_0401+DRB1_0701+DRB1_1101+DRB1_1501 \
    --peptides_per_length 1000000 \
    --output $CAPE_ROOT/CAPE_MPNN/data/input/immuno/mhc_2/pwm
```

That reproduces the published matrices exactly, with these parameters:

| parameter | value |
|---|---|
| predictor | **netMHCIIpan-4.3** |
| alleles | the six-allele HLA-DRB1 panel above (~85.6% population coverage) |
| peptide length | 15-mers only (a single length, unlike the class I build) |
| peptides scored | **1,000,000 per allele** |
| presentation threshold | `%Rank_EL <= 2.0` |
| matrix | plain log-frequency over presented peptides, no pseudocount |
| score threshold | 98th percentile of random-peptide PWM scores, matching the 2% rank |

The build writes `random_peptides.txt` (the scored peptide set),
`ranks/<allele>.csv` (one `%Rank_EL` per peptide), and
`pwm/<allele>/<allele>-15[_log].csv`. Keeping `ranks/` matters: the
register-anchored variant (`analysis/build_register_pwm_production.py`) recovers
each allele's presenters from that cache rather than re-scoring a million
peptides.

Because the peptide set is drawn at random, a rebuild reproduces the matrices to
sampling noise rather than bit-identically. To compare a rebuilt PWM against the
published one, check the per-position amino-acid frequencies rather than file
checksums.

## Relationship to CAPE-MPNN

This work extends [CAPE-MPNN](https://github.com/hcgasser/CAPE_MPNN) (Gasser et
al., 2025), which introduced DPO fine-tuning of ProteinMPNN against predicted
MHC class **I** presentation. Our contribution is the extension to MHC class
**II** (the ADA-relevant pathway), a population-representative HLA-DRB1 panel,
the acidic-composition regulariser, the de novo binder-redesign application, and
the IEDB validation analysis. `training/cape-mpnn.py` is adapted from the
CAPE-MPNN training code; the CAPE-MPNN copyright and MIT licence are retained in
[`LICENSE`](LICENSE).

## Citation

If you use this code or the models, please cite both this work and CAPE-MPNN.
A `CITATION.cff` is provided; its DOI, repository URL and release date are
completed on publication.

## License

MIT, retaining the original CAPE-MPNN copyright (Hans-Christof Gasser, 2025)
alongside this work's additions. See [`LICENSE`](LICENSE).
