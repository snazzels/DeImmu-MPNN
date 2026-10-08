import os,json,re
DEST=os.environ["CAPE_ROOT"] + "/CAPE_MPNN/data/output/binder_redesign_v1"
WORK=os.path.dirname(os.path.abspath(__file__))
cs=json.load(open(f"{DEST}/chain_seqs.json"))
YD=os.path.join(WORK,"yaml_msa"); os.makedirs(YD,exist_ok=True)
def recs(fa):
    out=[];cur=None
    for l in open(fa):
        l=l.rstrip("\n")
        if l.startswith(">"):
            if cur: out.append(cur)
            cur=[l,""]
        elif cur: cur[1]+=l.strip()
    if cur: out.append(cur)
    return out
def block(target_pairs, binder_id, binder_seq):
    s="version: 1\nsequences:\n"
    for ch,seq in target_pairs:
        s+=f"  - protein:\n      id: {ch}\n      sequence: {seq}\n"
    s+=f"  - protein:\n      id: {binder_id}\n      sequence: {binder_seq}\n      msa: empty\n"
    return s
manifest=[]
for pid,d in cs.items():
    binder=d["binder"]; targets=d["targets"]; seqs=d["seqs"]
    tp=[(ch,seqs[ch]) for ch in targets]
    open(os.path.join(YD,f"{pid}_native.yaml"),"w").write(block(tp,binder,seqs[binder]))
    manifest.append((pid,"native",0,f"{pid}_native"))
    for model in ("base","cape"):
        fa=os.path.join(DEST,"redesign_fastas",f"{pid}_{model}.fa")
        items=[]
        for hdr,seq in recs(fa)[1:]:
            m=re.search(r"global_score=([0-9.]+)",hdr); items.append((float(m.group(1)) if m else 999, seq.replace("X","")))
        items.sort(key=lambda x:x[0])
        for k,(gs,seq) in enumerate(items):  # ALL designs, not top-3
            name=f"{pid}_{model}_{k+1}"
            open(os.path.join(YD,f"{name}.yaml"),"w").write(block(tp,binder,seq))
            manifest.append((pid,model,k+1,name))
json.dump(manifest,open(os.path.join(WORK,"manifest_full24.json"),"w"),indent=2)
print("generated", len(manifest),"yamls")
