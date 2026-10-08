# Trained checkpoints

The checkpoints accompanying the paper are **not committed to git** (large, and
trained weights are immutable release artefacts); they are deposited on Zenodo and
fetched with [`fetch_weights.sh`](fetch_weights.sh). The deposition authoring draft
(description, metadata, upload manifest) is in
[`ZENODO_WEIGHTS.md`](ZENODO_WEIGHTS.md).

| Release name (Zenodo) | Model hash | λ | learning rate | Role |
|---|---|---|---|---|
| `deimmu_deployed.pt` | `f16b51e6` | 1.0 | 1×10⁻⁶ | **The deployed model.** Every headline number in the paper: 41.4% monomer reduction, the 11-complex binder panel, allele transfer. |
| `deimmu_mhc2only.pt` | `abec4d2c` | 1.0 | 1×10⁻⁶ | **MHC class II-only ablation at the deployed operating point** (`--mhc_1_predictor none`); the attribution control in **Figure 3A**, and the class II-only arm of the allele-generalisation and charge comparisons. |
| `deimmu_mhc2only_lr3e-7_lambda0.pt` | `7777d75f` | 0 | 3×10⁻⁷ | MHC class II-only ablation at the **earlier** learning rate. Superseded for Figure 3A by `abec4d2c`; deposited because the manuscript names it. **λ=0, so it differs from `deimmu_mhc2only.pt` in both learning rate and λ** — hence the λ in the release name (renamed 2026-09-09). |
| `base_lambda0.pt` | `54e1642c` | 0 | 3×10⁻⁷ | λ=0 arm of the charge-regulariser sweep (regulariser off). |
| `cape_creg0.5.pt` | `057975d5` | 0.5 | 3×10⁻⁷ | λ=0.5 arm of the sweep. Weights live in the backup directory `057975d5_v2_creg0.5/` (see below). |
| `cape_creg1.0.pt` | `7e908919` | 1.0 | 3×10⁻⁷ | λ=1.0 arm of the sweep (same λ as deployed, lower learning rate). |
| `deimmu_lambda0_lr1e-6.pt` | `dce9dd92` | 0 | 1×10⁻⁶ | λ=0 arm of the regulariser series **at the deployed learning rate**. Added 2026-09-08 — see the correction below. |
| `deimmu_lambda05_lr1e-6.pt` | `1c908bb6` | 0.5 | 1×10⁻⁶ | λ=0.5 arm of the same series. Added 2026-09-08. |
| `deimmu_deployed_seed_replicate.pt` | `58a9b5ae` | 1.0 | 1×10⁻⁶ | Seed replicate of the deployed model; supplies the between-run scale (1.5 points) against which every small cross-checkpoint difference in the paper is read. Added 2026-09-08. |

`abec4d2c` differs from `f16b51e6` in **`mhc_1_predictor` alone** (`none` vs
`pwm`) — same λ, β and learning rate, both verified from their stored
`dpo_hparams.yaml`. That single-variable difference is what makes Figure 3A a
direct attribution rather than an inference, so the pair must be deposited
together.

`base_lambda0.pt` is a legacy release name and is a slight misnomer: it is the
**λ=0 fine-tune**, not the base model. The base model is stock ProteinMPNN
`v_48_020`, which is not redistributed here.

## Provenance

Source artefacts under `artefacts/CAPE-MPNN/models/` in the working tree. Each
λ below was **read from that model's own `dpo_hparams.yaml`**, not inferred from a
directory name — with one exception. **`057975d5`'s YAML has no
`charge_reg_weight` key at all**, because the run predates that parameter joining
`ModelManager.dpo_hparams_names` on 2026-07-15; its λ=0.5 rests on the provenance
of the surviving weights (the `_v2_creg0.5` backup copy described below), not on the
model's own record. The other eight are YAML-documented, re-verified 2026-09-09.

The checkpoint differs by role, matching what the manuscript reports (Methods,
"Every model in the grid ... reported at the epoch-200 checkpoint. For the earlier
single-point models ... the reported checkpoint is instead the *best checkpoint*"):
the deployed model and the sweep grid are evaluated at **epoch 200** so
cross-configuration comparisons are matched on training budget, while the earlier
λ sweep and the single-objective ablations use the **best** (lowest DPO validation
loss) checkpoint. Deposit the checkpoint each result was produced from:

| λ / role | source file | checkpoint |
|---|---|---|
| deployed, λ=1.0 @ lr 1×10⁻⁶ | `f16b51e6/ckpts/epoch_200.pt` | epoch 200 |
| class II-only @ lr 1×10⁻⁶ | `abec4d2c/ckpts/epoch_200.pt` | epoch 200 |
| λ=0 | `54e1642c/ckpts/best.pt` | best |
| λ=0.5 | `057975d5_v2_creg0.5/best.pt` | best |
| λ=1.0 @ lr 3×10⁻⁷ | `7e908919/ckpts/best.pt` | best |
| class II-only @ lr 3×10⁻⁷ | `7777d75f/ckpts/epoch_200.pt` | epoch 200 — **resolved 2026-09-09, see below** |
| λ=0 @ lr 1×10⁻⁶ | `dce9dd92/ckpts/epoch_200.pt` | epoch 200 |
| λ=0.5 @ lr 1×10⁻⁶ | `1c908bb6/ckpts/epoch_200.pt` | epoch 200 |
| deployed seed replicate | `58a9b5ae/ckpts/epoch_200.pt` | epoch 200 |

Verified against the evaluation summaries: `eval_deimmunisation_sweep_f16b51e6`
records `epoch_200`, while the λ-arm summaries record `/best`. `abec4d2c` is
epoch 200: every attribution file for it — `*_netmhcmatch_pwmtraj_abec4d2c_ep200.*`
— is scored at that checkpoint, matching the deployed model's training budget,
which is the point of the comparison.

> **✅ Resolved 2026-09-09: deposit `7777d75f/ckpts/epoch_200.pt`.** This file said
> `best.pt` from 2026-08-20, against matched attribution data named
> `*_pwmtraj_7777d75f_ep200.*`. The two turned out not to be competing options:
>
> 1. **`best.pt` is byte-identical to `epoch_168.pt`** (md5
>    `34d07c48518e97fdc8210b9d6a4a156a`; epochs 168, 173, 175, 199 and 200 checked,
>    only 168 matches). The running-best checkpoint for this run *is* epoch 168,
>    which is also why an `ep168` entry appears in its trajectory.
> 2. **No result file was produced from `best.pt`.** Every `7777d75f` output in
>    `data/output` comes from the epoch trajectory, and every netMHCIIpan- and
>    netMHCpan-validated output for it exists at `ep200` alone.
>
> Per the rule stated above — deposit the checkpoint each result was produced from —
> the file to upload is `epoch_200.pt`, and one file suffices. Anyone wanting the
> running-best weights can take `epoch_168.pt`; no reported result rests on them.
>
> Note for the manuscript, not for this deposition: v55's Methods groups `7777d75f`
> with the models "reported at the running-best checkpoint", and the checkpoint
> decision section says it is retained "because the regularizer sweep reports it".
> No table in v55 has a `7777d75f` column and no number in v55 is attributable to
> it — it appears only in provenance prose. Wording for the author to settle; it
> changes nothing here.

### Correction (2026-08-20)

An earlier revision of this file mapped **λ=0 to `7e908919`**. That was wrong:
`7e908919/dpo_hparams.yaml` records `charge_reg_weight: 1.0`, so it is the **λ=1.0**
model, and the λ=0 model is **`54e1642c`** (`charge_reg_weight: 0.0`, same β and
learning rate). The earlier revision also omitted the deployed model `f16b51e6`
and the ablation `7777d75f` entirely, although the manuscript's data-availability
statement promises both. Corrected here; the manuscript and this file now agree.

### Correction (2026-08-26)

Two further defects, both of the same kind as the one above — the manifest not
carrying the model behind a headline figure:

1. **`abec4d2c` was missing entirely.** Figure 3A was re-anchored on 2026-08-25 so
   that its class II-only bar is `abec4d2c` (deployed operating point) rather than
   `7777d75f` (earlier learning rate), keeping the pair at one operating point. The
   manuscript reports `abec4d2c` by name in the supplementary statistics table, but
   this file did not list it and `7777d75f` was still described as "the attribution
   control in Figure 3A". Both fixed.
2. **The deployed model's headline was stale at 37.9%.** The correct figure is
   **41.4%**: the per-protein reduction had been computed without a minimum-base
   guard, so a protein presenting 0.10 windows on average contributed −300%. See
   `data/MANIFEST.md` and `analysis/recompute_guarded_reduction.py`.

**Note for whoever writes the next manuscript revision:** the data-availability
statement in `manuscript_edits_myself_v41.tex` still attributes Figure 3A to
`7777d75f`. Per the figure's own generator
(`figures/make_fig_attribution_strategy_pinned.py`) the bar is `abec4d2c`. The
manuscript, not this file, is the one that needs correcting.

### Correction (2026-09-08)

Same kind of defect a third time, and this one had a second failure mode attached.

**Three checkpoints behind a main-text table were missing from this manifest.**
The manuscript now reports the acidic-composition regulariser series *at the
deployed learning rate* (`dce9dd92` λ=0, `1c908bb6` λ=0.5, `f16b51e6` λ=1.0, with
`58a9b5ae` as the seed replicate), all scored on one matched 98-protein sample
against one base arm. Only `f16b51e6` was listed here. Added above.

**The second failure mode: the manuscript had been carrying a `RE-ANCHOR TODO`
saying the λ=0 run at lr 1×10⁻⁶ did not exist and would cost ~20 h of GPU time to
produce.** `dce9dd92` had been on disk since 2026-08-22 with 41 evaluation files
beside it, and `training/slurm/16_netmhc_matched_ablation.sbatch` states in a
comment that tasks 9 and 10 exist precisely so the charge knee can be redrawn at
the deployed operating point. A full `dpo_hparams.yaml` diff against the deployed
model returns one substantive line, `charge_reg_weight: 1.0` vs `0.0`. **A TODO is
a claim about the world and decays like any other; check whether the artifact
exists before quoting one as open work.**

> **✅ Closed 2026-09-09: `fetch_weights.sh` and `ZENODO_WEIGHTS.md` now list all
> nine.** Both were extended, and because this was the third recurrence of one
> failure mode — the manifest not carrying a model the manuscript reports — two
> guards were added rather than just the fix:
>
> - `fetch_weights.sh` holds its file set in a single `WEIGHTS` array and now fails
>   loudly if any checkpoint is missing or empty after download, instead of
>   reporting success on a partial fetch.
> - **`check_manifest_sync.py` (new) parses this file, `fetch_weights.sh` and
>   `ZENODO_WEIGHTS.md` and asserts all three name the same set.** Run it before
>   any upload; exit 0 means the three agree.

### Why a suffixed backup directory exists for λ=0.5

Until 2026-07-15 the model hash was derived from base model + alleles + core
hyperparameters and did **not** include `charge_reg_weight`, so two runs differing
only in λ collided on the same hash directory. That is how the original λ=0 weights
in `057975d5` were destroyed: the λ=0.5 run wrote into the same directory, epoch by
epoch. The surviving λ=0.5 weights were copied aside to
`057975d5_v2_creg0.5/` on 2026-07-14, which is why that arm alone is referenced
through a suffixed directory rather than a `ckpts/` path.

`charge_reg_weight` and `charge_reg_target` have been part of
`ModelManager.dpo_hparams_names` since 2026-07-15, so λ variants now receive
distinct hashes and cannot collide — `54e1642c` (λ=0) and `7e908919` (λ=1.0) share
β and learning rate yet have different ids, which is the empirical proof. Runs
started after that date, including every model in the table above except the
λ=0.5 backup, are unaffected.

> **Do not predict a hash by recomputing it.** Recomputation is unreliable here
> (the stored YAML holds strings where training used floats), so read the id from
> the model's own `dpo_hparams.yaml`, or resolve it with `slurm/find_model_id.py`.

## Fetching

```bash
ZENODO_RECORD=<record-id> ./fetch_weights.sh
```

Then point the redesign wrapper at the fetched weights:

```bash
export WEIGHTS_CAPE=/abs/path/to/DeImmu-MPNN/models   # dir containing the .pt
# redesign.sh uses CAPE_MODEL (default 'cape_v3') as the ProteinMPNN --model_name;
# set CAPE_MODEL / the filename to match how the checkpoint is loaded.
```

`fetch_weights.sh` and `ZENODO_WEIGHTS.md` were extended to three checkpoints
(pre-2026-08-26), then six (2026-08-26), then **nine (2026-09-09)**, each time
because the manifest had fallen behind the manuscript. Run
`./check_manifest_sync.py` before uploading to confirm all three files still agree.

SHA256 of each released file, so a download can be verified:

```
# sha256sum *.pt   (filled 2026-10-08 from the staged upload set)
deimmu_deployed.pt                  4b50100ae76c90862a437ebefde589cd3ea5e3727e0a4878f4b1379d206e06d3
deimmu_mhc2only.pt                  53f844b13c6fdbc7555a2f22907ad0057ab3428de434bd870f30a2bb1cb2fae1
deimmu_lambda0_lr1e-6.pt            35330e48620fd4875266027627e6c7579c4171b74742436794866c9c366759c0
deimmu_lambda05_lr1e-6.pt           04ae75c4ff17c765213afac3f8fdd79fb5a10c6d03b9b4cb4827ada7271ecdd1
deimmu_deployed_seed_replicate.pt   fcfe6b53d5782b46a011fcb6fb5cb4ae1379093eebc75c4884a19c2298b84a72
base_lambda0.pt                     6290487223c1f94c91bd930338236409d30f7d7035dc00a79a37e2263f06d5b2
cape_creg0.5.pt                     1b93fbc58e0429fd29406dfbe594fcd6ddd02b2b5af16f50df85ff29fba00bfd
cape_creg1.0.pt                     f2e8731b2abb33d8f7313e60b775867028e7c2022b810143ef5f0adc10805c80
deimmu_mhc2only_lr3e-7_lambda0.pt   5addb41453372b6d4f2b0f6b70b2516a13df7de0ad24d08d4b84509b9782251b
```

Verify a download with `sha256sum -c` against the `SHA256SUMS` file shipped
alongside the weights in the Zenodo record.

> **Provenance of these hashes.** They were taken from the staged upload set, and on
> 2026-10-08 every one of the nine staged files was confirmed byte-identical to the
> source checkpoint named for it in `ZENODO_WEIGHTS.md`. Each model's
> objective, λ and learning rate were also re-read from its own `dpo_hparams.yaml`
> and matched the manifest table — including the documented exception, `057975d5`,
> whose YAML carries no `charge_reg_weight` key, so its λ=0.5 rests on the
> provenance of the `_v2_creg0.5` backup directory rather than on its own record.
