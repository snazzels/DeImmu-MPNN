# Environments and external dependencies

The pipeline spans several tools, each with its own environment. This project
used `micromamba`. Nothing here is a single pinned `environment.yml` because the
components have mutually incompatible dependency stacks (JAX vs PyTorch vs Boltz);
they are run as separate environments.

## Core — ProteinMPNN + de-immunised weights (`cape_mpnn`)
The only environment needed for the core `application/redesign.sh` step and for
MHC-II burden scoring.
- PyTorch (CUDA), NumPy, biotite (SASA / interface classification).
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) checkout (`protein_mpnn_run.py`); set `PMPNN_DIR`.
- The `kit` / `CAPE.MPNN` support libraries (from the CAPE-MPNN tree); the paper
  scripts `sys.path`-insert `.../CAPE_MPNN/libs` — adapt to your checkout.
- The `Mhc2PredictorPwm` PWMs (six-allele HLA-DRB1 panel); set the PWM path in
  `application/score_burden.py`.

## MHC binding predictors (for exact re-scoring / data prep)
- [netMHCIIpan-4.3](https://services.healthtech.dtu.dk/) and netMHCpan-4.1 (DTU
  academic licence; **not redistributable**). Used by `data_prep/` and
  `analysis/validate_with_netmhciipan.py`. The PWM surrogate approximates
  netMHCIIpan so training does not call it in the inner loop.

## Binding validators
- **Boltz-2** — [github.com/jwohlwend/boltz](https://github.com/jwohlwend/boltz),
  own env (`new_boltz`). Target folded with an MMseqs2-server MSA; binder single
  sequence.
- **AF2 initial-guess** — AlphaFold2 model params required. The reimplementation
  in `application/af2_initial_guess/` removes the PyRosetta dependency of the
  original [dl_binder_design](https://github.com/nrbennet/dl_binder_design)
  (Bennett et al., 2023). Known env fixes that were needed: a `scipy.linalg`
  `tril`/`triu` shim for newer scipy, a ptxas ≥ 11.8 on PATH for jaxlib, a
  `Bio.Data.SCOPData` shim for modern biopython, and `tensorflow-cpu` for the
  legacy `tf.data` feature pipeline.

## Folding for self-consistency
- ESMFold (`esm_biohub`, ≤ ~160 aa) and Boltz-2 (larger proteins) for the
  `analysis/selfconsistency_*` scripts.

## Environment variables the released scripts expect

The scripts carry no absolute paths. Every machine-specific location is read
from an environment variable, so the release runs unmodified once these are
set. Only `CAPE_ROOT` is needed for the common path; the rest are required only
by the specific tools named.

| variable | required for | meaning |
|---|---|---|
| `CAPE_ROOT` | nearly everything | the project root — the directory **containing** `CAPE_MPNN/`. Shell scripts abort with a message if it is unset; Python scripts raise `KeyError: 'CAPE_ROOT'`. |
| `CAPE_PY` | cluster job scripts | Python interpreter of the `cape_mpnn` environment. Defaults to `python` on `PATH`. |
| `CAPE_ENV` | a few job scripts | the `cape_mpnn` environment directory itself. |
| `ESM_PY` | ESMFold self-consistency | Python interpreter of the `esm_biohub` environment. Defaults to `python`. |
| `ESM_WD` | ESMFold self-consistency | ESMFold working directory (model cache). |
| `BOLTZ_BIN` | Boltz-2 folding | `bin/` directory of the Boltz-2 environment. |
| `PDB_CACHE_DIR` | training / evaluation | local copy of the ProteinMPNN `pdb_2021aug02` training set. |
| `PYMOL_BIN` | a few figure scripts | PyMOL executable. Defaults to `pymol`. |
| `AF2IG_WORK` | `af2_initial_guess/run_af2ig_native_panel2.py` | scratch working directory. Defaults to `/tmp/binder_panel2`. |
| `MHC2_PWM`, `CAPE_LIBS`, `PMPNN_DIR` | `application/` CLIs | PWM directory, `CAPE_MPNN/libs`, and the ProteinMPNN checkout. Each also has a command-line flag. |

A variable with a default degrades gracefully; one without (`CAPE_ROOT`,
`BOLTZ_BIN`, `PDB_CACHE_DIR`, `ESM_WD`) fails loudly rather than silently
writing to the wrong place.

> Provenance paths recorded **inside** files under `data/` are the absolute
> paths of the machine that produced them and were deliberately left as
> generated. Those are frozen result artefacts; rewriting their recorded
> provenance would edit a deposited result.
