#!/usr/bin/env python
"""Monomer self-consistency of the REDESIGNED BINDER CHAINS (reviewer R-4 / ST-9).

The panel reports interface metrics but not whether each redesigned binder's own
fold is preserved. Here we fold the top-3-by-global_score binder designs (base and
cape) plus the native binder as a monomer with ESMFold2, and TM-score the prediction
against the binder's native backbone (fixed 1:1 correspondence, gap-aware). Base vs
cape TM answers whether de-immunization degrades the binder's own fold.

Restricted to binder chains <= 160 aa (ESMFold2's practical ceiling here); 9cce (197)
and 9nds (248) are skipped.

Run:  micromamba run -n esm_biohub python tools/selfconsistency_binder_chains.py
Output: data/output/selfconsistency_binder/  (preds/, results.txt)
"""
import os, sys, re, time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np

PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ESM_WD = os.environ.get("ESM_WD", "")
BR = os.path.join(PF, "data", "output", "binder_redesign_v2_f16b51e6")
OUT = os.path.join(PF, "data", "output", "selfconsistency_binder_v2_f16b51e6")
PRED = os.path.join(OUT, "preds")
os.makedirs(PRED, exist_ok=True)

# binder chain id per complex (from the contact analysis) and binder length
BINDER = {"9kku": "C", "9ju1": "C", "8upa": "A", "5vli": "C", "4oyd": "B",
          "7jzl": "E", "8t5e": "A", "9cc5": "A", "9nzh": "A",
          "9cce": "A", "9nds": "A"}   # 9cce=197, 9nds=248 aa (test the ESMFold2 VRAM ceiling)
TOPK = 3


def binder_ca_by_resseq(pdb_path, chain):
    """Return dict {resseq: (x,y,z)} of Cα for the binder chain, in file order."""
    ca = {}
    order = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain:
                continue
            if line[12:16].strip() != "CA":
                continue
            alt = line[16]
            if alt not in (" ", "A"):
                continue
            resseq = int(line[22:26])
            if resseq in ca:
                continue
            ca[resseq] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            order.append(resseq)
    return ca, order


def read_pred_ca(path):
    ca = []
    with open(path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                ca.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(ca, dtype=np.float64)


def kabsch_tm(P, Q, L_norm):
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    V, _, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1, 1, d]) @ Wt
    di = np.linalg.norm((Pc @ R) - Qc, axis=1)
    rmsd = float(np.sqrt((di ** 2).mean()))
    d0 = 1.24 * (max(L_norm, 19) - 15) ** (1.0 / 3.0) - 1.8
    tm = float((1.0 / L_norm) * np.sum(1.0 / (1.0 + (di / d0) ** 2)))
    return rmsd, tm


def top_designs(pid, model, k):
    fa = os.path.join(BR, "redesign_fastas", f"{pid}_{model}.fa")
    items, native = [], None
    cur = None
    for line in open(fa):
        line = line.rstrip("\n")
        if line.startswith(">"):
            cur = line
        elif cur is not None:
            seq = line.strip().replace("X", "")
            if "sample=" in cur:                       # a design entry
                m = re.search(r"global_score=([0-9.]+)", cur)
                if m:
                    items.append((float(m.group(1)), seq))
            elif native is None:                        # first entry = native/original
                native = seq
            cur = None
    items.sort(key=lambda x: x[0])
    return native, [s for _, s in items[:k]]


def main():
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    import torch
    print("Loading ESMFold2…", flush=True)
    m = ESMFold2Model.from_pretrained(f"{ESM_WD}/esmfold2", load_esmc=False).to("cuda").eval()
    m.load_esmc(f"{ESM_WD}/esmc_6b", precision="bf16")

    def fold(seq, tag):
        p = os.path.join(PRED, tag + ".pdb")
        if not os.path.exists(p):
            with torch.no_grad():
                out = m.infer_protein(seq, num_loops=4, num_sampling_steps=50)
            open(p, "w").write(m.output_to_pdb(out))
            del out
        return read_pred_ca(p)

    rows = []
    for pid, chain in BINDER.items():
        cplx = os.path.join(BR, "complexes", f"{pid}_cplx.pdb")
        ca, order = binder_ca_by_resseq(cplx, chain)
        if not ca:
            print(f"{pid}: no Cα for chain {chain}, skip", flush=True); continue
        first = min(order)
        native_seq, _ = top_designs(pid, "base", TOPK)  # native from base fasta

        def tm_for(seq, tag):
            L = len(seq)
            nat = np.full((L, 3), np.nan)
            for rs, xyz in ca.items():
                pos = rs - first
                if 0 <= pos < L:
                    nat[pos] = xyz
            valid = ~np.isnan(nat[:, 0])
            pred = fold(seq, tag)
            if len(pred) != L:
                return None
            if valid.sum() < 15:
                return None
            _, tm = kabsch_tm(pred[valid], nat[valid], int(valid.sum()))
            return tm

        rec = {"pid": pid, "L": len(native_seq) if native_seq else len(order)}
        try:
            if native_seq:
                rec["native"] = tm_for(native_seq, f"{pid}_native")
            for model in ("base", "cape"):
                _, seqs = top_designs(pid, model, TOPK)
                tms = [tm_for(s, f"{pid}_{model}_{i+1}") for i, s in enumerate(seqs)]
                tms = [t for t in tms if t is not None]
                rec[model] = float(np.mean(tms)) if tms else None
                rec[model + "_n"] = len(tms)
        except Exception as e:
            torch.cuda.empty_cache()
            print(f"  {pid}: FAILED at L={rec['L']} ({type(e).__name__}: {str(e)[:100]})", flush=True)
            rows.append(rec); continue
        rows.append(rec)
        print(f"  {pid}: native {rec.get('native')}, base {rec.get('base')}, cape {rec.get('cape')}", flush=True)

    # summary
    L = ["=" * 70, "BINDER-CHAIN MONOMER SELF-CONSISTENCY (ESMFold2, top-3 by global_score)",
         "TM to native binder backbone; fixed 1:1 correspondence, gap-aware.", "=" * 70,
         f"{'pid':6}{'L':>5}{'native':>9}{'base':>9}{'cape':>9}{'cape-base':>11}"]
    bb, cc = [], []
    for r in rows:
        nat = f"{r['native']:.3f}" if r.get('native') is not None else "  -  "
        b = r.get('base'); c = r.get('cape')
        if b is not None and c is not None:
            bb.append(b); cc.append(c)
            L.append(f"{r['pid']:6}{r['L']:>5}{nat:>9}{b:>9.3f}{c:>9.3f}{c-b:>+11.3f}")
    if bb:
        L.append("-" * 70)
        L.append(f"{'MEAN':6}{'':>5}{'':>9}{np.mean(bb):>9.3f}{np.mean(cc):>9.3f}{np.mean(cc)-np.mean(bb):>+11.3f}")
        import scipy.stats as st
        try:
            t, p = st.wilcoxon(bb, cc)
            L.append(f"Wilcoxon base vs cape: p={p:.3g}  (n={len(bb)} complexes)")
        except Exception:
            pass
    L.append("Interpretation: base ≈ cape TM => de-immunization preserves the binder's own monomer fold.")
    txt = "\n".join(L)
    print("\n" + txt, flush=True)
    open(os.path.join(OUT, "results.txt"), "w").write(txt + "\n")
    print(f"\nWrote {os.path.join(OUT, 'results.txt')}", flush=True)


if __name__ == "__main__":
    main()
