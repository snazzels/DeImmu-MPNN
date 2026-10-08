# Zenodo deposition — trained model weights

This file is the **authoring draft** for the separate Zenodo deposition that
hosts the **nine** trained checkpoints (the code is archived in its own deposition;
see `.zenodo.json` at the repo root). Once deposited, put the record id into
`fetch_weights.sh` and the DOI into `.zenodo.json` and `CITATION.cff`.

**Before uploading, run `./check_manifest_sync.py`.** It asserts that this file,
`fetch_weights.sh` and `README.md` name the same nine files. The three have drifted
apart three times (see the corrections at the foot of this file and of `README.md`),
each time shipping a deposition without a model the manuscript reports.

## Deposition metadata

- **Title:** DeImmu-MPNN: Reducing Immunogenicity Burden in De Novo Protein Design — trained checkpoints
- **Upload type:** Model (`upload_type: model`)
- **Access:** Open
- **License:** MIT (retaining original CAPE-MPNN copyright, Gasser 2025)
- **Version:** 1.0.0
- **Creators:** [complete author list + order — matches the paper]; Niklas
  Halbwedl (Technical University of Munich), …
- **Keywords:** ProteinMPNN, MHC class II, immunogenicity, de-immunisation,
  direct preference optimization, de novo protein design, model weights

### Related identifiers
| Relation | Identifier |
|---|---|
| `isSupplementTo` | [paper DOI — REPLACE] |
| `isSupplementedBy` | [code Zenodo DOI / GitHub release — REPLACE] |
| `isDerivedFrom` | https://github.com/hcgasser/CAPE_MPNN |
| `isDerivedFrom` | https://github.com/dauparas/ProteinMPNN |

## Description (paste into Zenodo)

> Nine ProteinMPNN checkpoints fine-tuned by direct preference optimization to
> reduce predicted MHC class II (T-helper) epitope presentation, accompanying the
> paper *"DeImmu-MPNN: Reducing Immunogenicity Burden in De Novo
> Protein Design"*. The models extend
> CAPE-MPNN (Gasser et al., 2025) from MHC class I to
> the anti-drug-antibody-relevant class II pathway, scored against a
> population-representative six-allele HLA-DRB1 panel
> (DRB1\*01:01/03:01/04:01/07:01/11:01/15:01).
>
> `deimmu_deployed.pt` is the deployed model and the one to use by default: a dual
> class I + class II objective at λ=1.0 and learning rate 1×10⁻⁶, reducing
> netMHCIIpan-predicted class II presentation by 41.4% on 98 held-out test proteins
> (93 of 98 improved). `deimmu_mhc2only.pt` is an ablation identical to it except
> that the class I term is removed, provided for applications whose limiting risk is
> the anti-drug-antibody response; it also serves as the objective-attribution
> control in the paper.
>
> Three further checkpoints sit at the same deployed learning rate and span the
> weight λ of an acidic-composition regulariser that suppresses a generic
> electrostatic shortcut — λ=0 (`deimmu_lambda0_lr1e-6.pt`), λ=0.5
> (`deimmu_lambda05_lr1e-6.pt`) and λ=1.0 (the deployed model itself) — accompanied
> by a seed replicate of the deployed model
> (`deimmu_deployed_seed_replicate.pt`) that supplies the run-to-run scale against
> which the λ trend should be read. The remaining four checkpoints repeat that λ
> series at an earlier operating point (learning rate 3×10⁻⁷), plus the class II-only
> ablation at that learning rate.
>
> Load with stock ProteinMPNN; usage and a fetch script are in the code repository.
> Fine-tuned from ProteinMPNN v_48_020. IEDB training-signal source data are not
> redistributed (IEDB terms of use); the code repository regenerates all derived
> datasets from a fresh IEDB export.

## Upload manifest

Every λ, learning rate and predictor setting below was **read from that model's own
`dpo_hparams.yaml`** on 2026-09-09, not inferred from a directory name — except
`057975d5`, whose λ is not recoverable that way (see the note after the table).

| File to upload | id | objective | λ | lr | seed | Source in working tree (`artefacts/CAPE-MPNN/models/`) | Notes |
|---|---|---|---|---|---|---|---|
| `deimmu_deployed.pt` | `f16b51e6` | dual | 1.0 | 1×10⁻⁶ | — | `f16b51e6/ckpts/epoch_200.pt` | **The deployed model.** Every headline number, the 11-complex binder panel, allele transfer. |
| `deimmu_mhc2only.pt` | `abec4d2c` | class II only | 1.0 | 1×10⁻⁶ | `None` | `abec4d2c/ckpts/epoch_200.pt` | Figure 3A attribution control; differs from the deployed model in `mhc_1_predictor` alone. |
| `deimmu_lambda0_lr1e-6.pt` | `dce9dd92` | dual | 0 | 1×10⁻⁶ | `None` | `dce9dd92/ckpts/epoch_200.pt` | λ=0 arm of the regulariser series at the deployed learning rate (`tab:knee`). |
| `deimmu_lambda05_lr1e-6.pt` | `1c908bb6` | dual | 0.5 | 1×10⁻⁶ | `None` | `1c908bb6/ckpts/epoch_200.pt` | λ=0.5 arm of the same series (`tab:knee`). |
| `deimmu_deployed_seed_replicate.pt` | `58a9b5ae` | dual | 1.0 | 1×10⁻⁶ | `1` | `58a9b5ae/ckpts/epoch_200.pt` | Seed replicate of the deployed model; supplies the between-run scale (1.5 points). Differs from `f16b51e6` in seed alone. |
| `base_lambda0.pt` | `54e1642c` | dual | 0 | 3×10⁻⁷ | — | `54e1642c/ckpts/best.pt` | λ=0 sweep arm. Legacy name: it is the λ=0 *fine-tune*, not the base model. |
| `cape_creg0.5.pt` | `057975d5` | dual | 0.5 | 3×10⁻⁷ | — | `057975d5_v2_creg0.5/best.pt` | λ=0.5 sweep arm. Half the charge shift, most of the reduction kept. |
| `cape_creg1.0.pt` | `7e908919` | dual | 1.0 | 3×10⁻⁷ | — | `7e908919/ckpts/best.pt` | λ=1.0 sweep arm. **Not** the deployed model — same λ, lower learning rate. |
| `deimmu_mhc2only_lr3e-7_lambda0.pt` | `7777d75f` | class II only | **0** | 3×10⁻⁷ | — | `7777d75f/ckpts/epoch_200.pt` | Earlier-learning-rate class II-only ablation; **λ=0, so it differs from `deimmu_mhc2only.pt` in two settings, not one.** Checkpoint resolved 2026-09-09, see below. |

A dash in the seed column means the key is absent from that model's YAML (a run
predating `seed` joining the hashed hyperparameter list), not that a seed of zero
was used.

**`057975d5` is the one λ that cannot be read from its own YAML.** Its
`dpo_hparams.yaml` records `lr`, `mhc_1_predictor` and `mhc_2_predictor` but **no
`charge_reg_weight` key at all**, because the run predates that parameter joining
`ModelManager.dpo_hparams_names` on 2026-07-15. Its λ=0.5 assignment rests on the
provenance of the surviving weights (the `_v2_creg0.5` backup copy) rather than on
the model's own record — which is precisely why that directory has a λ in its name.
Treat this one arm's λ as documented-by-provenance and the other eight as
documented-by-YAML.

Optional to include for provenance: each model's own `dpo_hparams.yaml` (the
authoritative record of its configuration) and a short `MODEL_CARD.md`. Note that
the nine checkpoints do **not** share one training config: β=0.05, 200 epochs,
10⁶ examples/epoch and T=0.2 throughout, but the learning rate is 1×10⁻⁶ for five
of them and 3×10⁻⁷ for the other four, so a single config file would misdescribe
half the deposition.

### `7777d75f`: checkpoint resolved 2026-09-09 — upload `epoch_200.pt`

This cell read "checkpoint unresolved" from 2026-08-20 to 2026-09-09, because
`README.md` recorded `best.pt` while the matched attribution data on disk is
`*_pwmtraj_7777d75f_ep200.*`. Resolved by looking at what exists:

1. **`7777d75f/ckpts/best.pt` is byte-identical to `epoch_168.pt`** (md5
   `34d07c48518e97fdc8210b9d6a4a156a`; checked against epochs 168, 173, 175, 199 and
   200, and only 168 matches). So "best" and an epoch file are not competing
   objects here — the running-best checkpoint for this run *is* epoch 168.
2. **No result file in `data/output` was produced from `best.pt`.** Every
   `7777d75f` output is from the epoch trajectory (`ep025`…`ep200`), and every
   netMHCIIpan- and netMHCpan-validated output for it exists at `ep200` only
   (`validate_netmhciipan_netmhcmatch_pwmtraj_7777d75f_ep200*`,
   `validate_netmhcpan_mhc1_netmhcmatch_pwmtraj_7777d75f_ep200*`).
3. `tools_build_deposition.py` deposits
   `eval_deimmunisation_pwmtraj_7777d75f_ep200.csv` for this id.

**Upload `epoch_200.pt`**, which is the checkpoint the deposited data for this id
was produced from — the rule stated in `README.md`. One file suffices; `best.pt`
need not be uploaded separately, since a user wanting it can take `epoch_168.pt`,
and no reported result rests on it.

> **Related manuscript bookkeeping, flagged not fixed.** `manuscript_edits_myself_v55.tex`
> places `7777d75f` inside the set of models "reported at the running-best
> checkpoint" (Methods) and describes it as retained "because the regularizer sweep
> reports it" (checkpoint decision section). Neither is quite right: `tab:sweep`
> has no `7777d75f` column, `tab:knee` and `tab:checkpoints` do not either, and no
> number anywhere in v55 is attributable to this id — it appears only in provenance
> prose, always as superseded. This does not change any result or any uploaded file;
> it is two sentences of manuscript wording for the author to decide on.

### Corrected 2026-08-26 — this table was wrong in three ways

1. **It listed `base_lambda0.pt` (λ=0) as `7e908919`.** That model's stored
   `charge_reg_weight` is **1.0**, so it is the λ=1.0 arm; λ=0 is `54e1642c`. This
   is the same error corrected in `README.md` on 2026-08-20 and never propagated
   here, so the two files in this directory contradicted each other for six days.
2. **It omitted the deployed model and both ablations** — half the deposition, and
   including the checkpoint behind every number in the paper.
3. **It described `base_lambda0.pt` as the source of the "headline monomer
   results"** and `cape_creg1.0.pt` as "used for the 11-complex binder panel".
   Both belong to `deimmu_deployed.pt`; the binder panel was re-run on the deployed
   model (`binder_redesign_v2_f16b51e6`).

Two source paths were also reconciled rather than guessed at: `README.md` gave
`7e908919/ckpts/best.pt` for the λ=1.0 arm where this file gave
`7e908919_v3_creg1.0/best.pt`. **They are byte-identical** (md5 `24cac8adf0ed`,
both 6,674,029 bytes), as are `057975d5/ckpts/best.pt` and
`057975d5_v2_creg0.5/best.pt` (md5 `252e0b462aba`). Either path yields the correct
weights; the `ckpts/` form is used above for consistency except for λ=0.5, where
the suffixed backup remains the safer reference.

> **The hash-collision warning that used to sit here is obsolete and has been
> removed.** `charge_reg_weight` and `charge_reg_target` joined
> `ModelManager.dpo_hparams_names` on 2026-07-15, so λ variants now get distinct
> ids — `54e1642c` (λ=0) and `7e908919` (λ=1.0) share β and learning rate yet
> differ, which is the proof. The λ=0.5 backup directory exists because of a
> collision that happened *before* that fix. **Still verify each uploaded file is
> the intended variant** — but read the id from the model's own
> `dpo_hparams.yaml`, and do not try to predict an id by recomputing the hash.

### Corrected 2026-09-09 — six to nine, and the same defect a third time

**Three checkpoints behind a main-text table were missing from this manifest**
(`dce9dd92`, `1c908bb6`, `58a9b5ae`). `README.md` had listed all nine since
2026-09-08 and carried a warning that this file and `fetch_weights.sh` still listed
six; a deposition built from the scripts as they stood would have shipped without
the models behind `tab:knee`. Both are now extended to nine.

This is the **third** recurrence of one failure mode: the manifest not carrying a
model the manuscript reports (2026-08-20, 2026-08-26, and now). The pattern is that
the model list lives in three places — this file, `fetch_weights.sh`, `README.md` —
and a new checkpoint gets added to one. Two changes address that directly:

- `fetch_weights.sh` now holds its file set in a single `WEIGHTS` array with the
  drift warning attached, and refuses to report success if any file is missing or
  empty after download.
- **`check_manifest_sync.py` (new)** parses all three files and asserts they name
  the same set, exiting non-zero on any mismatch. **Run it before uploading.**

One release name was also changed: `deimmu_mhc2only_lr3e-7.pt` →
`deimmu_mhc2only_lr3e-7_lambda0.pt`. The old name implied the learning rate was the
only difference from `deimmu_mhc2only.pt`, when λ differs too (0 vs 1.0) — the same
class of trap as the corrections above. Safe to rename: the release names appear
nowhere outside this directory, and the manuscript cites model hashes, not
filenames.

## After deposition

1. Record the Zenodo record id in `fetch_weights.sh` (`ZENODO_RECORD`).
2. Record the DOI in the repo-root `.zenodo.json` and in `CITATION.cff`.
3. Compute and paste `sha256sum *.pt` into the SHA256SUMS block in `README.md`.
