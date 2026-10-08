# Icon attribution — Figure 1 (mhc_mechanism_figure.png)

Cell and molecule icons are from [Bioicons](https://bioicons.com) (curated by
Simon Dürr). Sources and licenses of the icons used in `make_mhc_figure.py`
(SVG sources and rasterized PNGs are in `figures/icons/`):

| Icon file | Depicts | Author | License |
|---|---|---|---|
| `dendritic-cell-1` | antigen-presenting cell | Servier | CC BY 3.0 |
| `t-lymphocyte`      | CD8⁺ / CD4⁺ T cell     | Servier | CC BY 3.0 |
| `b-lymphocyte`      | B cell                 | Servier | CC BY 3.0 |
| `immunoglobulin`    | antibody (ADA)         | Servier | CC BY 3.0 |
| `mhc1`              | MHC class I molecule   | Helicase_11 | CC BY 4.0 |
| `MHC2`              | MHC class II molecule  | Helicase_11 | CC BY 4.0 |
| `plasma_cell`       | plasma cell            | El-Jayawant | CC BY 4.0 |
| `proteasome`        | proteasome             | jaiganesh | CC0 |

All icons are CC BY or CC0 (no share-alike or non-commercial terms), so they can
be used in the publication with attribution. The attribution above is reproduced
in the Figure 1 caption of the manuscript.

Servier Medical Art icons: "Servier Medical Art by Servier, licensed under
CC BY 3.0 (https://creativecommons.org/licenses/by/3.0/)."

## Regenerating the figure

Icons were downloaded as SVG from bioicons.com and rasterized to transparent PNG
(height 400 px) with Inkscape:

```
inkscape icon.svg --export-type=png --export-filename=icon.png \
    --export-height=400 --export-background-opacity=0
```

Then `python make_mhc_figure.py` composes them into `mhc_mechanism_figure.png`
(300 dpi). The previous, icon-free version is kept as
`mhc_mechanism_figure_boxes_v1.png`.
