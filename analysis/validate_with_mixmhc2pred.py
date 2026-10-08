#!/usr/bin/env python
"""P0-1: orthogonal-predictor validation of the MHC-II reduction.

Re-scores the SAME deployed-model designs the netMHCIIpan headline used, but with
MixMHC2pred (Gfeller lab) — an MHC-II presentation predictor of independent lineage
from netMHCIIpan (different lab, method, and training data). If the base->fine-tuned
reduction survives MixMHC2pred, the effect is not a netMHCIIpan idiosyncrasy — the
core circular-validation critique (reviewer P0-1) is answered.

Read-only on the frozen eval CSV; writes only new versioned outputs.

Install MixMHC2pred v2 (standalone, no netMHC dependency):
    git clone https://github.com/GfellerLab/MixMHC2pred
    # follow its README; the executable is MixMHC2pred_unix (Linux). Point --bin at it.
Then:
    python tools/validate_with_mixmhc2pred.py \
        --in_csv data/output/eval_deimmunisation_dual_lambda0_seed42.csv \
        --bin /path/to/MixMHC2pred_unix --n_proteins 0 --out_tag lambda0
"""
import os, sys, csv, subprocess, tempfile, argparse, time
import numpy as np
from collections import defaultdict

PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 6 trained DRB1 alleles in MixMHC2pred v2 nomenclature (DRB1*01:01 -> DRB1_01_01)
ALLELES = ["DRB1_01_01","DRB1_03_01","DRB1_04_01","DRB1_07_01","DRB1_11_01","DRB1_15_01"]
PEPTIDE_LEN = 15
BATCH = 20000

def get_15mers(seq):
    seq = seq.replace("*","").replace("-","").replace("/","")
    return [seq[i:i+PEPTIDE_LEN] for i in range(len(seq)-PEPTIDE_LEN+1)]

def run_mixmhc2pred(peptides, binp, alleles):
    """Return {peptide: {'best': rank, allele: rank, ...}}.
    Robust to v2 output layout: uses '%Rank_best' + per-allele '%Rank_<allele>' columns."""
    ranks = {}
    for i in range(0, len(peptides), BATCH):
        batch = peptides[i:i+BATCH]
        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td,"pep.txt"); outp = os.path.join(td,"out.txt")
            # v2 accepts a plain peptide list; some builds want a 'Peptide' header — include it, tool ignores if not needed
            open(inp,"w").write("\n".join(batch)+"\n")
            cmd = [binp,"--input",inp,"--output",outp,"--alleles"]+list(alleles)+["--no_context"]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0:
                sys.stderr.write(r.stderr.decode()[-2000:]); r.check_returncode()
            ranks.update(_parse(outp, alleles))
    return ranks

def _parse(path, alleles):
    out = {}
    lines = [l for l in open(path) if not l.startswith("#")]
    if not lines: return out
    hdr = lines[0].rstrip("\n").split("\t")
    def col(name):
        for j,h in enumerate(hdr):
            if h.strip().lower()==name.lower(): return j
        return None
    ci_pep = col("Peptide")
    ci_best = col("%Rank_best")
    per = {a: col(f"%Rank_{a}") for a in alleles}
    for l in lines[1:]:
        p = l.rstrip("\n").split("\t")
        if ci_pep is None or ci_pep>=len(p): continue
        pep = p[ci_pep]
        d = {}
        if ci_best is not None and ci_best<len(p):
            try: d["best"]=float(p[ci_best])
            except ValueError: pass
        for a,cix in per.items():
            if cix is not None and cix<len(p):
                try: d[a]=float(p[cix])
                except ValueError: pass
        out[pep]=d
    return out

def presented(seq, ranks, alleles, thr, mode):
    ks = get_15mers(seq)
    if mode=="per_allele":   # sum over alleles of windows presented by that allele (matches netMHCIIpan counting)
        return sum(sum(1 for k in ks if ranks.get(k,{}).get(a,999)<=thr) for a in alleles)
    return sum(1 for k in ks if ranks.get(k,{}).get("best",999)<=thr)  # windows presented by >=1 allele

def main(a):
    t0=time.time()
    rows=list(csv.DictReader(open(a.in_csv)))
    proteins=sorted(set(r["protein_name"] for r in rows))
    if a.n_proteins: proteins=proteins[:a.n_proteins]
    prset=set(proteins); rows=[r for r in rows if r["protein_name"] in prset]
    print(f"{len(proteins)} proteins, {len(rows)} sequences",flush=True)
    uniq=sorted({k for r in rows for k in get_15mers(r["sequence"])})
    print(f"{len(uniq)} unique 15-mers; MixMHC2pred on {len(ALLELES)} alleles…",flush=True)
    ranks=run_mixmhc2pred(uniq, a.bin, ALLELES)
    have_per = any(any(al in v for al in ALLELES) for v in ranks.values())
    mode = "best" if a.best_only or not have_per else "per_allele"  # per-allele sum matches netMHCIIpan counting
    per=defaultdict(lambda: defaultdict(list))
    for r in rows:
        per[r["protein_name"]][r["model"]].append(presented(r["sequence"],ranks,ALLELES,a.threshold,mode))
    reds=[]; base=[]; ft=[]
    for p in proteins:
        b=per[p].get("base"); f=per[p].get("finetuned")
        if not b or not f: continue
        bm,fm=np.mean(b),np.mean(f); base.append(bm); ft.append(fm)
        if bm>0: reds.append(100*(bm-fm)/bm)
    reds=np.array(reds)
    summary=["="*60,f"MixMHC2pred ORTHOGONAL VALIDATION — {a.out_tag}","="*60,
             f"predictor: MixMHC2pred (independent of netMHCIIpan)   counting mode: {mode}",
             f"%Rank threshold (presented): {a.threshold}   proteins: {len(reds)}",
             f"  base mean windows: {np.mean(base):.2f}   fine-tuned: {np.mean(ft):.2f}",
             f"  per-protein reduction: {reds.mean():.1f}% ± {reds.std(ddof=1):.1f}   "
             f"({int((reds>0).sum())}/{len(reds)} improved)",
             "", "Compare to netMHCIIpan on the SAME designs (headline 28.1%): if this is",
             "substantially positive, the reduction is not a netMHCIIpan idiosyncrasy (P0-1).",
             f"runtime {(time.time()-t0)/60:.1f} min"]
    od=os.path.join(PF,"data","output")
    open(os.path.join(od,f"validate_mixmhc2pred_{a.out_tag}_summary.txt"),"w").write("\n".join(summary))
    with open(os.path.join(od,f"validate_mixmhc2pred_{a.out_tag}.csv"),"w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["protein","base_windows","ft_windows","reduction_pct"])
        for p in proteins:
            b=per[p].get("base"); f=per[p].get("finetuned")
            if not b or not f: continue
            bm,fm=np.mean(b),np.mean(f); w.writerow([p,round(bm,3),round(fm,3),round(100*(bm-fm)/bm,2) if bm>0 else ""])
    print("\n".join(summary),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_csv",default=os.path.join(PF,"data/output/eval_deimmunisation_dual_lambda0_seed42.csv"),
                    help="eval CSV with a 'sequence' and 'model' (base/finetuned) column")
    ap.add_argument("--bin",required=True,help="path to MixMHC2pred_unix executable")
    ap.add_argument("--n_proteins",type=int,default=0,help="0 = all")
    ap.add_argument("--threshold",type=float,default=2.0,help="%%Rank presented cutoff (mirrors netMHCIIpan 2%%; sweep 2/5/10)")
    ap.add_argument("--best_only",action="store_true",help="count windows presented by >=1 allele (%%Rank_best) instead of per-allele sum")
    ap.add_argument("--out_tag",default="lambda0")
    main(ap.parse_args())
