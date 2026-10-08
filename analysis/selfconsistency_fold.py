#!/usr/bin/env python
"""Stage 2/3 of the ESMFold self-consistency benchmark (roadmap step P1.3).

Folds every sequence in designs.csv (native controls + base + finetuned designs)
with ESMFold2 and writes one predicted PDB per design. Stage 3 computes TM/RMSD
of each prediction back to the native backbone saved by Stage 1.

RUN IN THE esm_biohub ENV (NOT cape_mpnn):
    micromamba run -n esm_biohub python tools/selfconsistency_fold.py
ESMFold2 = ESMC-6B backbone + folding head; ~13.6 GB resident, peaks ~22 GB at
L=150 on a 24 GB card, so this needs the GPU to itself (run after Stage 1 exits).
It is the original Meta ESMFold v1 that the ProteinMPNN/RFdiffusion scTM
literature uses; v1 is unavailable in this env (fair-esm absent; transformers
ESMFold needs torch>=2.1 while cape_mpnn is pinned at 2.0.1). ESMFold2 (biohub) is
an independent single-sequence structure predictor and serves the same role; the
substitution is noted for the paper.

Resumable: skips designs whose PDB already exists. OOM on a sequence is caught and
recorded as failed rather than aborting the run.

Outputs
-------
data/output/selfconsistency/preds/<design_id>.pdb   predicted structure
data/output/selfconsistency/fold_metrics.csv        design_id, plddt, ptm, n_res, status
"""
import os, sys, csv, time, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR   = os.path.join(PF, "data", "output", "selfconsistency")
IN_CSV    = os.path.join(OUT_DIR, "designs.csv")
PRED_DIR  = os.path.join(OUT_DIR, "preds")
METRICS   = os.path.join(OUT_DIR, "fold_metrics.csv")
ESM_WD    = os.environ.get("ESM_WD", "")


def safe(design_id):
    return design_id.replace("|", "__").replace("/", "_")


def main(a):
    os.makedirs(PRED_DIR, exist_ok=True)
    import pandas as pd
    df = pd.read_csv(IN_CSV)
    if a.limit:
        df = df.head(a.limit)
    print(f"{len(df)} sequences to fold from {IN_CSV}")

    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    print("Loading ESMFold2…")
    m = ESMFold2Model.from_pretrained(f"{ESM_WD}/esmfold2", load_esmc=False).to("cuda").eval()
    m.load_esmc(f"{ESM_WD}/esmc_6b", precision="bf16")
    print(f"  VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    # resume: keep already-written metrics
    done = {}
    if os.path.exists(METRICS):
        with open(METRICS) as f:
            for r in csv.DictReader(f):
                done[r["design_id"]] = r
    fields = ["design_id", "protein_name", "model", "seq_idx", "plddt", "ptm", "n_res", "status"]

    t0 = time.time()
    with open(METRICS, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for i, row in enumerate(df.itertuples(index=False)):
            did = row.design_id
            pdb_path = os.path.join(PRED_DIR, safe(did) + ".pdb")
            if did in done and os.path.exists(pdb_path) and done[did].get("status") == "ok":
                w.writerow({k: done[did].get(k, "") for k in fields}); fh.flush(); continue
            seq = row.sequence
            rec = dict(design_id=did, protein_name=row.protein_name, model=row.model,
                       seq_idx=row.seq_idx, plddt="", ptm="", n_res=len(seq), status="ok")
            try:
                t = time.time()
                with torch.no_grad():
                    out = m.infer_protein(seq, num_loops=a.num_loops, num_sampling_steps=a.num_steps)
                open(pdb_path, "w").write(m.output_to_pdb(out))
                rec["plddt"] = round(float(out["plddt"].mean()), 4)
                rec["ptm"]   = round(float(out["ptm"].mean()), 4)
                dt = time.time() - t
                del out
                print(f"  [{i+1}/{len(df)}] {did} L={len(seq)} {dt:.1f}s pLDDT {rec['plddt']}", flush=True)
            except torch.cuda.OutOfMemoryError:
                rec["status"] = "oom"
                print(f"  [{i+1}/{len(df)}] {did} L={len(seq)} OOM", flush=True)
            except Exception as e:
                rec["status"] = f"err:{type(e).__name__}"
                print(f"  [{i+1}/{len(df)}] {did} ERROR {e}", flush=True)
            torch.cuda.empty_cache()
            w.writerow(rec); fh.flush()
    print(f"\nWrote {METRICS} and PDBs to {PRED_DIR}. Runtime {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num_loops", type=int, default=4)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--outdir", default=OUT_DIR)
    a = p.parse_args()
    OUT_DIR = a.outdir
    IN_CSV = os.path.join(OUT_DIR, "designs.csv")
    PRED_DIR = os.path.join(OUT_DIR, "preds")
    METRICS = os.path.join(OUT_DIR, "fold_metrics.csv")
    main(a)
