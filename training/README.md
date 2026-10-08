# Training — DPO fine-tuning

`cape-mpnn.py` fine-tunes ProteinMPNN by direct preference optimization against a
joint reward: the negative sum of predicted MHC class I **and** MHC class II
presented windows, plus an optional acidic-composition regulariser (weight `λ`,
`--charge_reg_weight`) that suppresses the electrostatic shortcut. It is adapted
from the CAPE-MPNN training code (Gasser et al., 2025); the MHC class II reward,
the HLA-DRB1 panel, and the acidic-composition regulariser are this work's
additions.

Paper settings: 200 epochs, 1M examples/epoch, batch size 2000, sampling
temperature 0.2, DPO β=0.05, learning rate 3×10⁻⁷, preference pairs regenerated
from the current policy every 2 epochs, same PDB backbone training set as the
original ProteinMPNN. A single run is a substantial GPU commitment (comparable to
each charge-regulariser sweep point), not a quick job.

> **Do not overwrite checkpoints.** The model-hash directory is derived from base
> model + alleles + core hyperparameters and does **not** include
> `charge_reg_weight`, so two runs differing only in λ collide on the same hash
> directory and the second silently overwrites the first. Give each run a
> distinct output id and confirm the target `ckpts/` is empty before launching.
> See `models/README.md`.

Dependencies (the `kit` / `CAPE.MPNN` libraries, ProteinMPNN, the MHC predictors)
come from the CAPE-MPNN environment; see [`../environment.md`](../environment.md).
This file is the entrypoint only, not a standalone package.
