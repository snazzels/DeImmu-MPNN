# Data preparation — IEDB mining and peptide ranking

These scripts regenerate the derived datasets used for training and evaluation
from a fresh **IEDB** export. Raw IEDB data are obtained under IEDB's terms of
use from <https://www.iedb.org/database_export_v3.php> and are **not
redistributed** in this repository.

| Script | Purpose |
|---|---|
| `mine_IEDB_MHC2.py` | Mine positive (immunogenic) MHC class II T-cell-assay records. |
| `mine_IEDB_negatives.py` | Mine `Negative`-assay records filtered to the same effector-readout universe; yields the clean experimentally-negative set and the binder-matched subset used in Supplementary S1. |
| `MHC-II_rank_peptides.py` | Rank peptides / build the MHC class II PWM inputs over the HLA-DRB1 panel. |
| `MHC-I_rank_peptides.py` | The MHC class I equivalent (for the joint objective). |

The peptide-ranking scripts call a local **netMHCIIpan-4.3** / **netMHCpan-4.1**
install (DTU academic licence; not redistributable). See
[`../environment.md`](../environment.md).
