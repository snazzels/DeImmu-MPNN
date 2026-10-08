# Notebooks

**`deimmunize_proteinmpnn_colab.ipynb`** — a Google Colab notebook that: clones stock
ProteinMPNN, fetches the released de-immunised weights from Zenodo, takes a PDB (by
ID or upload), redesigns the chosen chain(s) with the de-immunised model and base
ProteinMPNN for comparison, downloads the sequences, and **optionally validates a
design with Boltz-2** (interface ipTM for a binder — natural target with an
MMseqs2-server MSA, de novo binder single-sequence, per the paper's protocol — or
pTM/pLDDT for a monomer). The de-immunisation is entirely in the weights, so the
generation step is stock ProteinMPNN with the released checkpoint.

The Boltz-2 cell is optional and heavy (multi-GB weight download on first run, GPU
required, minutes per prediction; defaults to one design; large complexes may exceed
free-Colab memory). To quantify the MHC-II burden reduction on the generated
sequences, use [`../application/score_burden.py`](../application/score_burden.py); the
second, independent binding validator (AF2 initial-guess) is in
[`../application/af2_initial_guess/`](../application/af2_initial_guess/).

**Before the Colab badge works**, the repository must be public. The badge path in
the top-level README is set to `snazzels/DeImmu-MPNN`. The weight-download cell
needs the Zenodo record id from the deposited checkpoints (see
[`../models/README.md`](../models/README.md)).

> Not yet run end-to-end in Colab (the released weights are not deposited yet). The
> code cells parse and follow the same ProteinMPNN invocation as the paper's
> `application/redesign.sh`, but the notebook should be executed once against the
> live Zenodo deposit before the repository is published.
