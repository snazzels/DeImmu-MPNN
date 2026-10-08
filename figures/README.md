# Figure generators

Scripts that produce the manuscript figures. Each is self-contained; run with the
project environment active (`source ../CAPE_MPNN/tools/set_ENV.sh` in the working
tree, or the equivalent `cape_mpnn` conda env with matplotlib). Figures use a common
house palette (gray = base/neutral, blue = fine-tuned / MHC class II, orange = MHC
class I, red = highlight). In-plot panel headers are bare letters (`A`/`B`/`C`); the
descriptive text for each panel lives in the manuscript caption, which references the
panels.

| Manuscript figure | PNG | Generator | Data source |
|---|---|---|---|
| MHC-I/II mechanism schematic | `mhc_mechanism_figure.png` | `make_mhc_figure.py` (+ `icons/`, `ICON_ATTRIBUTION.md`) | none (schematic); Bioicons glyphs, rasterized to `icons/*.png` |
| DPO pipeline overview | `pipeline_overview_figure.png` | `make_pipeline_flowchart.py` → `pipeline_flowchart.svg`, then rasterize with Inkscape (command in the script header) | none (schematic) |
| MHC-II-only ablation (double dissociation) | `mhc2only_ablation_figure.png` | `make_ablation_figure.py` | matched-seed eval summaries `eval_deimmunisation_{mhc2only,dual}_seed42` + `eval_mhc1_arm_*` (values hard-coded in the script) |
| Strategy comparison (filtering vs DPO, human-likeness) | `strategy_comparison_figure.png` | `make_strategy_comparison_figure.py` | `data/output/filtering_baseline.csv`, `eval_deimmunisation.csv`, `human_likeness.csv` |
| Charge-regularizer knee (λ sweep) | `charge_regularizer_knee.png` | `make_charge_knee_figure.py` | λ-sweep summary values (v1/v2/v3 charge-shift + eval), hard-coded |
| Allele generalization | `allele_generalization_figure.png` | `../analysis/test_allele_generalisation.py` (full run) — plot re-generable from `data/output/allele_generalisation.csv` | netMHCIIpan over held-out designs (cached CSV) |
| Register-anchored vs smeared PWM | `register_pwm_figure.png` | `../analysis/build_register_pwm.py` (deterministic, `random.seed(0)`) | netMHCIIpan on random 15-mers |
| De novo binder panel (11 complexes) | `panel11_figure.png` | `../application/make_panel_figure.py` (self-contained; reads the three shipped JSONs beside it) | burden + Boltz-2 ipTM (all-24) + AF2-initial-guess JSONs in `../application/` |
| S1 — netMHCIIpan vs experimental immunogenicity | `iedb_binding_signal_figure.png` | `../analysis/supplementary_iedb/analyse_iedb_binding_signal.py` | cached netMHCIIpan-scored IEDB peptides (no live netMHCIIpan calls) |
| S1 — IEDB classifier organism-holdout | `iedb_classifier_figure.png` | `../analysis/supplementary_iedb/train_iedb_classifier.py` — bars re-generable from `iedb_classifier_summary.json` | IEDB effector labels + features |

Notes
- `make_schematics.py` is the earlier matplotlib version of the two schematics
  (Figures for mechanism and pipeline); it is superseded by `make_mhc_figure.py`
  (icon-based) and `make_pipeline_flowchart.py` (SVG flowchart), and now writes
  `*_v1.png` fallback names so it cannot clobber the current figures.
- Icon attribution and licenses (CC-BY / CC0 only) are in `ICON_ATTRIBUTION.md`.
