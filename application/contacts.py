import os,glob,math
AA3=set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL MSE".split())
PANEL={"9kku":("C",["A","B"]),"9ju1":("C",["A"]),"8upa":("A",["B"]),
       "5vli":("C",["A","B"]),"4oyd":("B",["A"]),"7jzl":("E",["A"])}
def atoms(pdb):
    d={}
    with open(pdb) as f:
        for l in f:
            if l[:6].strip() in ("ATOM","HETATM") and l[17:20].strip() in AA3:
                ch=l[21]; x=float(l[30:38]);y=float(l[38:46]);z=float(l[46:54])
                d.setdefault(ch,[]).append((x,y,z,int(l[22:26])))
    return d
for pid,(b,tg) in PANEL.items():
    d=atoms(os.path.join(WORK if False else os.path.join(os.path.dirname(os.path.abspath(__file__)),"complex"),pid+"_cplx.pdb"))
    ba=d.get(b,[])
    contacts=set(); ta=[]
    for t in tg: ta+=d.get(t,[])
    # count binder residues within 4.5A of any target atom
    binder_res=set()
    for x,y,z,ri in ba:
        for X,Y,Z,_ in ta:
            if (x-X)**2+(y-Y)**2+(z-Z)**2 < 4.5**2:
                binder_res.add(ri); break
    print(f"{pid}: binder {b} interface residues (<4.5A of target): {len(binder_res)} of {len({r for *_,r in ba})}")
