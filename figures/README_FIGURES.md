# Figure manifest — what generates what

Figures fall into two classes, and the reproducibility requirement differs:

- **Data-driven figures** (Fig 3, Fig 4, SI) plot measured results. Every one of these
  **must** have a surviving generator in *this directory* — that is a publication
  requirement. It has already been violated once: the foldability-cliff generator lived
  in `scratchpad/`, which no longer exists, so the figure had to be rebuilt from data
  on 2026-08-20. **Never put a data-driven figure's generator in `scratchpad/`.**
- **Schematics** (Fig 1, Fig 2) are drawn, not computed. They carry no data, so they
  need no generator script for the publication (author's call, 2026-08-20). Their
  source of truth is the **hand-edited SVG**, and any matplotlib script that seeded
  them is historical scaffolding, not a reproduction path.

For a schematic, the thing that must not be lost is the SVG. For a data-driven figure,
it is the script.

## The staging convention (and its failure mode)

Generators write a **versioned** file (`..._v3_ticks.png`, `..._v7b_f16b51e6.png`).
A copy is then staged under the **generic** name that `\includegraphics` points at.
Versioned renders are immutable; only the staged copy is refreshed.

The failure mode to watch: **a newer versioned render exists but the staged copy was
never refreshed**, so the manuscript silently embeds a superseded figure. This has
happened twice — Figure 2 carried the wrong model name and pre-re-anchor numbers for
nine days, and Figure 1 is currently suspect (see below). After regenerating, always
re-stage, and check the staged file's timestamp is not older than the newest version.

## Current figures (manuscript v129, updated 2026-10-05)

Verified against the `\includegraphics` list in `new_edits_v129.tex`.

| # | `\includegraphics` target | class | source of truth | versioned output |
|---|---|---|---|---|
| 1 | `mhc_mechanism_figure.png` | schematic | **`mhc_mechanism_figure_v6.svg`** (hand-edited) | `mhc_mechanism_figure_v6.png` |
| 2 | `pipeline_overview_figure.png` | schematic | **`pipeline_flowchart.svg`** (hand-edited) | `pipeline_overview_figure_v3_reanchored.png` |
| 3 | `fig_attribution_strategy_pinned.png` | data | `make_fig_attribution_strategy_pinned_v5.py` | `fig_attribution_strategy_pinned_v5.png` |
| 4 | `allele_generalization_pinned.png` | data | `make_fig_allele_pinned_v4_errbars.py` | `allele_generalization_pinned24_v5_se.png` |
| SI | `fig_si_sweep_combined.png` | data | `make_fig_si_sweep_combined.py` | `fig_si_sweep_combined.{png,pdf,svg}` |
| SI | `allele_generalization_four_model.png` | data | `make_fig_allele_four_model_v2_errbars.py` | `allele_generalization_four_model_v3_se.png` |
| SI | `epitope_structure_2wjr.png` | data | `epitope_structure_figure.py` | `epitope_structure_2wjr_A_<tag>/` |

> **⚠️ Do not infer the current generator from its filename. Read it out of the
> table above.** Three of these names actively mislead:
>
> - **`make_fig_allele_pinned.py` is not the Figure 4 generator.** Despite the name
>   it draws the *four-model* panel, which is now an SI figure, and it draws it at
>   its 2026-08-25 state. Figure 4 became a two-arm comparison on 2026-09-20.
> - **`make_fig_allele_pinned_v4_errbars.py` emits the `_v5_se` render**, not a
>   `_v4` one. The whisker changed from a 95% percentile interval to one bootstrap
>   SE on 2026-09-30 and the output was versioned while the script was not renamed.
> - **`make_fig_attribution_strategy_pinned.py` (no suffix) is superseded** by the
>   `_v5` file beside it, which re-anchored Panel A onto the deployed model.
>
> Everything not named in the table is retained deliberately as the provenance of
> values reported in earlier manuscript versions — `make_fig_allele_reanchored.py`,
> `make_fig_attribution_strategy_v7_f16b51e6.py`, `make_fig_allele_pinned.py` and
> `make_fig_attribution_strategy_pinned.py` all still run and all produce numbers
> that are no longer the paper's.
>
> The current generators are not merely newer — they *check* things their
> predecessors did not. Each verifies at run time that the arms it compares share a
> protein set **and a base arm that is identical design by design**, and aborts
> rather than plot otherwise. Two runs on the same protein *names* can still hold
> different base designs, which no protein-name check can detect and which silently
> corrupted a cross-model comparison before these guards existed. Figure 3
> additionally reads its external MHC class I-only arm from `*_guarded_summary.txt`
> files, so that all three bars use one minimum-base convention. Figure 4 and the SI
> four-model panel additionally assert that the bootstrap point estimate reproduces
> the frozen summary JSON, so a whisker can never be drawn around a value that is
> not the bar.

### Running these from this repository

The four data-driven generators resolve their inputs **by basename**, searching
`data/` in this repo first and the original working-tree layout
(`CAPE_MPNN/data/output/`) second, with `$CAPE_ROOT` as a final fallback. So from a
clean checkout, with no environment set:

```bash
python figures/make_fig_attribution_strategy_pinned_v5.py   # Fig 3
python figures/make_fig_allele_pinned_v4_errbars.py         # Fig 4
python figures/make_fig_allele_four_model_v2_errbars.py     # SI four-model
```

`epitope_structure_figure.py` is the exception: it needs netMHCIIpan, PyMOL
(`$PYMOL_BIN`) and the ProteinMPNN `pdb_2021aug02` backbone set (`$PDB_CACHE_DIR`),
none of which are redistributable. It fails with that instruction rather than
silently resolving to something else.

**Verified 2026-10-05** by running all three from this tree with `CAPE_ROOT` unset:
Figure 4 and the SI four-model panel reproduce the published PNGs **byte for byte**;
Figure 3 reproduces every published value but not the exact bytes — see the
matplotlib note under Conventions.

**Figure 1 — v6 is current** (author-confirmed 2026-08-20). The staged copy had been
stuck at a 2026-08-03 render for a week; re-staged from `mhc_mechanism_figure_v6.png`,
with the stale copy kept as `mhc_mechanism_figure_v0_staged_0803.png`.

⚠️ **Figure 1 open item: the PNG may lag its own SVG.** `mhc_mechanism_figure_v6.svg`
(2026-08-10 13:32) is *newer* than `mhc_mechanism_figure_v6.png` (11:14) and contains
Inkscape namespace markers, i.e. it was hand-edited after the PNG was rendered — the
same pattern that left Figure 2 stale for nine days. If those later SVG edits are
wanted, re-render:

```bash
inkscape mhc_mechanism_figure_v6.svg --export-type=png \
  --export-filename=mhc_mechanism_figure_v7_fromsvg.png \
  --export-width=<match current> --export-background=white
```

Note both schematics are matplotlib-seeded then hand-finished, so their `make_*.py`
files do **not** reproduce the shipped figure and must not be re-run over the SVG.

### Figure 2 — do not run the generator

`make_pipeline_flowchart.py` is **stale and marked DO NOT RUN**. The SVG has since been
hand-edited in Inkscape and those edits were never folded back, so running it would
overwrite `pipeline_flowchart.svg` and revert both the model name and the panel-B
numbers. Edit the SVG, then:

```bash
inkscape pipeline_flowchart.svg --export-type=png \
  --export-filename=pipeline_overview_figure_vN_<label>.png \
  --export-width=4375 --export-background=white
cp pipeline_overview_figure_vN_<label>.png pipeline_overview_figure.png
```

## Conventions

- **Environment:** matplotlib figures build with `PYTHONPATH=` cleared (a leaked
  `mpi4py` from `/home/t38guest/amber22` otherwise breaks imports).
- **⚠️ matplotlib version is load-bearing for Figure 3 only, and the published render
  did not come from the env this file used to name.** Figure 3 is the one generator
  that saves with `bbox_inches="tight"`, and tight-bbox rounding changed between
  matplotlib 3.8 and 3.10: the same script, on the same data, yields 4763×1724 px
  under 3.10.9 (the published figure) and 4766×1725 px under 3.8.4. **Every plotted
  value is identical** — this is canvas cropping, not content. Figures 4 and the SI
  four-model panel do not use tight bbox and are byte-reproducible across both.
  The published Fig 3 PNG records `Matplotlib version3.10.9` in its `tEXt` chunk;
  read that chunk rather than trusting any env name here, since the env that
  produced it no longer carries that version:

  ```bash
  python -c "import struct,sys; d=open(sys.argv[1],'rb').read(); i=8
  while i<len(d):
      n=struct.unpack('>I',d[i:i+4])[0]; t=d[i+4:i+8].decode()
      print(d[i+8:i+8+n][:60]) if t=='tEXt' else None; i+=12+n" FIG.png
  ```
- **Overwrite guards:** generators refuse to overwrite their versioned output. Bump the
  version rather than deleting — prior renders are the only record of a prior state.
- **Editable SVGs:** set `svg.fonttype = "none"` so text stays live text, and avoid
  matplotlib mathtext (`$...$`) in labels — mathtext becomes a pile of one-character
  `<text>` elements. Use Unicode (`β`, `Δ`, `≤`, `1×10⁻⁶`) so each label is one
  editable string. Note **Liberation Sans lacks U+207B superscript minus**; DejaVu Sans
  covers the full set.
- **Fonts:** Arial/Helvetica are not installed. Inkscape substitutes wider DejaVu Sans,
  which overflows fixed-width boxes in the hand-drawn schematics.

## Superseded, kept for the record

`sweep_pareto_si.*` and `foldability_cliff.png` were merged into
`fig_si_sweep_combined` on 2026-08-20 (one two-panel figure, decluttered). Their
predecessor `make_sweep_pareto_si.py` is retained; the combined generator supersedes it.
**Added 2026-08-26:** `make_fig_attribution_strategy_v7_f16b51e6.py` (Figure 3) and
`make_fig_allele_reanchored.py` (Figure 4) are now themselves superseded, by the
`_pinned` generators listed in the table above. They are retained, not pruned,
because they produced the figures embedded in manuscript versions up to v36.

**Added 2026-10-05 — and the lesson is the reason this file exists.** The `_pinned`
pair named directly above was *itself* superseded during September, and this README
was not updated, so between 2026-09-20 and 2026-10-05 the deposited tree shipped
generators for Figures 3 and 4 that no longer produced the figures in the paper —
while this file asserted that they did. The current generators
(`make_fig_attribution_strategy_pinned_v5.py`, `make_fig_allele_pinned_v4_errbars.py`,
`make_fig_allele_four_model_v2_errbars.py`, `epitope_structure_figure.py`) had never
been deposited at all.

Two things let that survive five weeks, both worth guarding against:

1. **The staging convention hides generator drift.** `\includegraphics` points at a
   stable generic name, so a figure can change generator — twice, here — without one
   character of the manuscript changing. Nothing in a `.tex` diff can reveal it.
2. **A superseding generator is often a *new file*, not an edit.** Every freshness
   check in this project compares a deposited file against its source. None of them
   can see a source that was never listed. The check that does work is the one run on
   2026-10-05: take the `\includegraphics` list from the current manuscript, resolve
   each target to the generator that last wrote it, and confirm *that* file is
   deposited. Do this whenever a figure is regenerated.

`fig_attribution_strategy*.py` v1–v6, `make_mhc_mechanism_figure*.py` v1–v6,
`make_ablation_figure*.py`, `make_charge_knee_figure.py`,
`make_strategy_comparison_figure*.py`, `make_human_likeness_deposited.py` and
`make_fig_allele_grouped.py` correspond to earlier figure versions or to analyses cut
from the current draft. Retained deliberately — do not prune without checking the
manuscript's current `\includegraphics` list.
