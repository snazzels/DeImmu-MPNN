#!/usr/bin/env python
"""Fig 3 (attribution + strategy), Panel B re-anchored on the DEPLOYED model f16b51e6.
Panel A intentionally unchanged: the single-objective ablations and the dual model they are
compared against were all trained at lr=3e-7, so that panel stays a matched comparison.
Same layout as figures/make_fig_attribution_strategy.py; data sources swapped to
the netMHC-direct summaries/CSV; Panel B rescaled (netMHC counts ~half the PWM)."""
import os, re, sys, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

OUT=os.environ["CAPE_ROOT"] + "/CAPE_MPNN/data/output"
FIG=os.environ["CAPE_ROOT"] + "/figures"
BLUE="#2c7fb8"; ORANGE="#e6550d"; GRAY="#7f7f7f"
C_RANDOM="#2e8bc0"; C_FILT="#d591b7"; C_DPO="#eab02e"; C_COMBO="#2ca25f"; FG="#222222"
plt.rcParams.update({"font.family":"sans-serif","font.size":15,"axes.edgecolor":FG,"axes.linewidth":0.9,"svg.fonttype":"none"})

def _parse(path,pat,what):
    if not os.path.exists(path): sys.exit(f"missing {what}: {path}")
    m=re.search(pat,open(path).read())
    if not m: sys.exit(f"cannot parse {what} from {path}")
    return float(m.group(1)),float(m.group(2))
def mhc2(tag): return _parse(f"{OUT}/validate_netmhciipan_fullset_{tag}_summary.txt",
                             r"Reduction:\s*mean\s*(-?[\d.]+)%\s*±\s*([\d.]+)%",f"II {tag}")
def mhc1(tag): return _parse(f"{OUT}/validate_netmhcpan_mhc1_fullset_{tag}_summary.txt",
                             r"Per-protein reduction:\s*(-?[\d.]+)%\s*\+/-\s*([\d.]+)%",f"I {tag}")
MODELS=[("MHC-II-only","mhc2only_k10","mhc2only_k5",BLUE),
        ("Dual-objective (λ=0, lr 3e-7)","lambda0_k10","dualL0_k5",GRAY),
        ("MHC-I-only (precursor)","mhcionly_k10","mhcionly_k5",ORANGE)]
DATA={lab:{"II":mhc2(t2),"I":mhc1(t1),"c":c} for lab,t2,t1,c in MODELS}

rows=list(csv.DictReader(open(f"{OUT}/validate_netmhciipan_sweep_f16b51e6.csv")))
prots=sorted(set(r["protein_name"] for r in rows))
# drop proteins with zero base-model presented windows: their percentage reduction is undefined
def _basemean(p):
    v=[int(r["netmhc_total"]) for r in rows if r["protein_name"]==p and r["model"]=="base"]
    return sum(v)/len(v) if v else 0.0
prots=[p for p in prots if _basemean(p)>0]
# (pwm_total, netmhc_total) per design; the filter SELECTS by PWM (the cheap, practical
# filter) and we MEASURE the kept designs by netMHCIIpan -> matches the text's primary bar.
def _pairs(p,m): return [(int(r["pwm_total"]),int(r["netmhc_total"])) for r in rows if r["protein_name"]==p and r["model"]==m]
FD=3
base_pp=np.array([np.mean([n for _,n in _pairs(p,"base")]) for p in prots])
dpo_pp =np.array([np.mean([n for _,n in _pairs(p,"finetuned")]) for p in prots])
filt_pp=np.array([np.mean([n for _,n in sorted(_pairs(p,"base"),key=lambda t:t[0])[:FD]]) for p in prots])
fdpo_pp=np.array([np.mean([n for _,n in sorted(_pairs(p,"finetuned"),key=lambda t:t[0])[:FD]]) for p in prots])
means=[np.nanmean(a) for a in (base_pp,filt_pp,dpo_pp,fdpo_pp)]
sems =[stats.sem(a,nan_policy="omit") for a in (base_pp,filt_pp,dpo_pp,fdpo_pp)]
red_filt=float(np.mean((base_pp-filt_pp)/base_pp)*100)
red_dpo =float(np.mean((base_pp-dpo_pp)/base_pp)*100)
red_fdpo=float(np.mean((base_pp-fdpo_pp)/base_pp)*100)
_,p_rf=stats.ttest_rel(base_pp,filt_pp); _,p_fd=stats.ttest_rel(filt_pp,dpo_pp)
_,p_rd=stats.ttest_rel(base_pp,dpo_pp); _,p_df=stats.ttest_rel(dpo_pp,fdpo_pp)

fig,(axA,axB)=plt.subplots(1,2,figsize=(13.5,5.8),gridspec_kw=dict(width_ratios=[1.0,1.2]))
CLASSES=["MHC class II","MHC class I"]; CKEY={"MHC class II":"II","MHC class I":"I"}
xA=np.arange(2); width=0.26
for j,lab in enumerate(DATA):
    mv=[DATA[lab][CKEY[c]][0] for c in CLASSES]
    sem=[DATA[lab][CKEY[c]][1]/np.sqrt(100) for c in CLASSES]   # s.e.m. over 100 proteins (matches Panel B)
    bars=axA.bar(xA+(j-1)*width,mv,width,yerr=sem,capsize=3.0,color=DATA[lab]["c"],edgecolor=FG,linewidth=0.8,error_kw=dict(ecolor=FG,lw=0.9))
    for b,mn in zip(bars,mv):
        axA.annotate(f"{mn:.1f}%",(b.get_x()+b.get_width()/2,mn),xytext=(0,4),textcoords="offset points",ha="center",va="bottom",fontsize=13,fontweight="bold",color=FG)
axA.set_xticks(xA); axA.set_xticklabels(CLASSES,fontsize=15)
axA.set_ylabel("Per-protein reduction in\nnetMHC-predicted presented windows (%)",fontsize=14)
axA.set_ylim(0,60); axA.axhline(0,color=FG,lw=0.8); axA.spines[["top","right"]].set_visible(False)
axA.yaxis.set_major_locator(plt.MultipleLocator(10)); axA.tick_params(length=4,labelsize=13.5)
axA.legend(handles=[Patch(facecolor=DATA[l]["c"],edgecolor=FG,label=l) for l in DATA],frameon=False,fontsize=13,loc="upper center",bbox_to_anchor=(0.5,-0.13),ncol=3,title="Fine-tuned model",title_fontsize=13.5,columnspacing=1.2,handlelength=1.3)
axA.set_title("A",loc="left",fontweight="bold",fontsize=21)

xB=np.arange(4)
axB.bar(xB,means,yerr=sems,width=0.66,color=[C_RANDOM,C_FILT,C_DPO,C_COMBO],capsize=5,error_kw=dict(ecolor=FG,lw=1.1))
for xi,m,lab in zip(xB,means,["ref.",f"−{red_filt:.0f}%",f"−{red_dpo:.0f}%",f"−{red_fdpo:.0f}%"]):
    axB.text(xi,m*0.5,lab,ha="center",va="center",color="white",fontweight="bold",fontsize=16)
base=means[0]; H=base*0.05
def _sig(x1,x2,y,txt):
    axB.plot([x1,x1,x2,x2],[y,y+H,y+H,y],lw=1.1,color=FG); axB.text((x1+x2)/2,y+H,txt,ha="center",va="bottom",fontsize=13)
_sig(0,1,base*1.10,f"p={p_rf:.0e}"); _sig(1,2,base*1.26,f"p={p_fd:.0e}")
_sig(0,2,base*1.42,f"p={p_rd:.0e}"); _sig(2,3,base*0.95,f"p={p_df:.0e}")
axB.set_xticks(xB); axB.set_xticklabels(["Random base (K=10)","Filtered base (3 of 10)","DPO fine-tuned (deployed)","DPO (deployed) + filter"],fontsize=12.5,rotation=20,ha="right",rotation_mode="anchor")
axB.set_xlim(-0.65,3.65)
axB.set_ylabel("Mean MHC-II presented windows\nper designed sequence (netMHCIIpan)",fontsize=14)
axB.set_ylim(0,base*1.62); axB.yaxis.set_major_locator(plt.MultipleLocator(10)); axB.spines[["top","right"]].set_visible(False); axB.tick_params(labelsize=13.5)
axB.set_title("B",loc="left",fontweight="bold",fontsize=21)
fig.tight_layout(w_pad=3.0)
op=f"{FIG}/fig_attribution_strategy_v7b_f16b51e6.png"
assert not os.path.exists(op),f"refuse overwrite {op}"
fig.savefig(op,dpi=300,bbox_inches="tight",facecolor="white")
print("Wrote",op)
print("Panel A:",{l:(round(DATA[l]["II"][0],1),round(DATA[l]["I"][0],1)) for l in DATA})
print(f"Panel B means={[round(m,1) for m in means]}  red_filt={red_filt:.1f} red_dpo={red_dpo:.1f} red_fdpo={red_fdpo:.1f}")
