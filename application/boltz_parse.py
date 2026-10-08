import os,json,glob
import numpy as np
WORK=os.path.dirname(os.path.abspath(__file__))
DEST=os.environ["CAPE_ROOT"] + "/CAPE_MPNN/data/output/binder_redesign_v1"
cs=json.load(open(f"{DEST}/chain_seqs.json"))
manifest=json.load(open(f"{WORK}/manifest_full24.json"))
OUT=os.path.join(WORK,"out_msa")

def conf_path(name):
    hits=glob.glob(os.path.join(OUT,"**",f"confidence_{name}_model_0.json"),recursive=True)
    return hits[0] if hits else None

def binder_iptm(pid,name):
    p=conf_path(name)
    if not p: return None
    c=json.load(open(p))
    ntarget=len(cs[pid]["targets"])
    bidx=str(ntarget)
    pc=c.get("pair_chains_iptm",{})
    vals=[]
    for t in range(ntarget):
        try: vals.append(pc[bidx][str(t)])
        except: pass
    return float(np.mean(vals)) if vals else c.get("iptm")

res={}
for pid,model,k,name in manifest:
    v=binder_iptm(pid,name)
    res.setdefault(pid,{}).setdefault(model,[]).append(v)

ORDER=["7jzl","5vli","8upa","4oyd","9ju1","9kku","8t5e","9cc5","9nzh","9cce","9nds"]
print(f"{'pid':8}{'native':>8}{'base(24)':>12}{'cape(24)':>12}{'d':>8}  base>0.5  cape>0.5")
out={}
for pid in ORDER:
    nat=res[pid]["native"][0]
    base=res[pid]["base"]; cape=res[pid]["cape"]
    bm,bs=np.mean(base),np.std(base); cm,csd=np.mean(cape),np.std(cape)
    b_pass=sum(1 for v in base if v>0.5); c_pass=sum(1 for v in cape if v>0.5)
    out[pid]=dict(native=nat, base_mean=float(bm), base_sd=float(bs), cape_mean=float(cm), cape_sd=float(csd),
                  base_vals=base, cape_vals=cape, base_pass=b_pass, cape_pass=c_pass, n=len(base))
    print(f"{pid:8}{nat:8.3f}{bm:8.3f}±{bs:.3f}{cm:8.3f}±{csd:.3f}{cm-bm:+8.3f}  {b_pass}/{len(base)}      {c_pass}/{len(cape)}")

json.dump(out, open(f"{WORK}/iptm_full24.json","w"), indent=2)

allb=np.concatenate([out[p]["base_vals"] for p in ORDER])
allc=np.concatenate([out[p]["cape_vals"] for p in ORDER])
print(f"\nOVERALL (n={len(allb)} each): base {allb.mean():.3f}±{allb.std():.3f}  cape {allc.mean():.3f}±{allc.std():.3f}  (delta {allc.mean()-allb.mean():+.3f})")
print(f"Confident (ipTM>0.5): base {int((allb>0.5).sum())}/{len(allb)} ({100*(allb>0.5).mean():.1f}%)  cape {int((allc>0.5).sum())}/{len(allc)} ({100*(allc>0.5).mean():.1f}%)")
