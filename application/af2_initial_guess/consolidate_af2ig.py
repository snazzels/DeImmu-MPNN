import os
import json, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
WORK=os.path.dirname(os.path.abspath(__file__))
DEST=os.environ["CAPE_ROOT"] + "/CAPE_MPNN/data/output/binder_redesign_v1"
r=json.load(open(f"{WORK}/af2ig_native_results.json"))
ipt=json.load(open(f"{DEST}/iptm_results_msa.json"))
META={"7jzl":"SARS-CoV-2 RBD","5vli":"influenza HA","8upa":"gp130","4oyd":"EBV","9ju1":"VEGF-A","9kku":"VEGF-A"}
ORDER=["7jzl","5vli","8upa","4oyd","9ju1","9kku"]
THRESH=10.0

rows=[]
for pid in ORDER:
    d=r[pid]
    base_vals=[v["pae_interaction"] for k,v in d.items() if k.startswith("base_")]
    cape_vals=[v["pae_interaction"] for k,v in d.items() if k.startswith("cape_")]
    nat=d["native"]["pae_interaction"]
    rows.append(dict(pdb=pid, target=META[pid], n_base=len(base_vals), n_cape=len(cape_vals),
        native_pae_interaction=nat,
        base_pass_pct=round(100*np.mean([v<THRESH for v in base_vals]),1),
        cape_pass_pct=round(100*np.mean([v<THRESH for v in cape_vals]),1),
        base_median_pae=round(float(np.median(base_vals)),2),
        cape_median_pae=round(float(np.median(cape_vals)),2),
        boltz_iptm_base=ipt[pid]["base_mean"], boltz_iptm_cape=ipt[pid]["cape_mean"]))

with open(f"{WORK}/af2ig_results.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(x) for x in rows]

allb=[r[p][k]["pae_interaction"] for p in ORDER for k in r[p] if k.startswith("base_")]
allc=[r[p][k]["pae_interaction"] for p in ORDER for k in r[p] if k.startswith("cape_")]
overall_base=100*np.mean([v<THRESH for v in allb]); overall_cape=100*np.mean([v<THRESH for v in allc])

S=[]
S.append("AF2 INITIAL-GUESS VALIDATION (Bennett et al. 2023 protocol) — second binding validator")
S.append("="*80)
S.append("Reimplemented PyRosetta-free from github.com/nrbennet/dl_binder_design (af2_initial_guess),")
S.append("using the vendored patched AlphaFold2 (model_1_ptm, initial_guess=True, num_recycle=3,")
S.append("single-sequence/no MSA, fixed-backbone initial guess = crystal complex coordinates).")
S.append("Metric: pae_interaction (mean bidirectional inter-chain PAE, Angstrom); pass = <10A (Bennett's own threshold).")
S.append("All 24 designs/model/complex scored (not just top-3).")
S.append("")
S.append(f"{'pdb':6}{'target':16}{'native_pae':>11}{'base pass%':>12}{'cape pass%':>12}{'base_med':>10}{'cape_med':>10}   boltz ipTM base/cape")
for x in rows:
    S.append(f"{x['pdb']:6}{x['target']:16}{x['native_pae_interaction']:>11.2f}{x['base_pass_pct']:>11.1f}%{x['cape_pass_pct']:>11.1f}%{x['base_median_pae']:>10.2f}{x['cape_median_pae']:>10.2f}   {x['boltz_iptm_base']:.3f}/{x['boltz_iptm_cape']:.3f}")
S.append("")
S.append(f"Overall pass rate (pae_interaction<{THRESH}A, n=144 designs/model): base {overall_base:.1f}%  cape {overall_cape:.1f}%")
S.append("CAPE does not show a lower AF2-initial-guess pass rate than base -- no evidence of degraded predicted binding;")
S.append("directionally corroborates the Boltz-2 finding (binding retained on 5/6 complexes).")
S.append("")
S.append("Per-complex notes:")
S.append("- 8upa (gp130): strong agreement between validators -- both high pass rate, cape >= base.")
S.append("- 4oyd (EBV), 7jzl (RBD/LCB1): bimodal (all-or-nothing) AF2-ig behaviour, a known property of this filter;")
S.append("  cape pass rate >= base in both cases.")
S.append("- 5vli (influenza HA): AF2-ig fails to confidently dock ANY variant, including the native crystal sequence")
S.append("  (native pae_interaction 27.4A). This is a validator floor effect (single-sequence/no-MSA AF2-ptm lacks")
S.append("  power on this larger two-chain target), not evidence against binding -- Boltz-2 (run with a target MSA)")
S.append("  found this complex's binding strongly retained (ipTM 0.96/0.96).")
S.append("- 9ju1/9kku (VEGF-A): both validators agree these are the weakest interfaces in the panel (0% AF2-ig pass;")
S.append("  lowest Boltz ipTM). AF2-ig's binary pass/fail can't resolve the base-vs-cape drop Boltz captured for 9kku")
S.append("  since neither clears the strict 10A threshold.")
open(f"{WORK}/af2ig_summary_final.txt","w").write("\n".join(S))
print("\n".join(S))

# figure: pass rate per complex, base vs cape, both validators
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.6))
x=np.arange(len(ORDER)); w=0.35
labels=[f"{p}\n{META[p]}" for p in ORDER]
b_pass=[d["base_pass_pct"] for d in rows]; c_pass=[d["cape_pass_pct"] for d in rows]
ax1.bar(x-w/2,b_pass,w,color="#9e9e9e",label="base ProteinMPNN")
ax1.bar(x+w/2,c_pass,w,color="#2b7bba",label="CAPE-MPNN-II (λ=1.0)")
ax1.set_xticks(x); ax1.set_xticklabels(labels,fontsize=8)
ax1.set_ylabel("AF2 initial-guess pass rate (%)\n(pae_interaction < 10 Å, n=24/model)")
ax1.set_title("A  AF2 initial-guess (Bennett 2023)",fontsize=11,loc="left",fontweight="bold")
ax1.legend(fontsize=8,frameon=False); ax1.spines[['top','right']].set_visible(False)
ax1.set_ylim(0,100)

b_iptm=[d["boltz_iptm_base"] for d in rows]; c_iptm=[d["boltz_iptm_cape"] for d in rows]
ax2.bar(x-w/2,b_iptm,w,color="#9e9e9e",label="base ProteinMPNN")
ax2.bar(x+w/2,c_iptm,w,color="#2b7bba",label="CAPE-MPNN-II (λ=1.0)")
ax2.axhline(0.5,ls="--",lw=1,color="#c0392b")
ax2.set_xticks(x); ax2.set_xticklabels(labels,fontsize=8)
ax2.set_ylabel("Boltz-2 interface ipTM")
ax2.set_title("B  Boltz-2 (target MSA)",fontsize=11,loc="left",fontweight="bold")
ax2.legend(fontsize=8,frameon=False,loc="lower left"); ax2.spines[['top','right']].set_visible(False)
ax2.set_ylim(0,1.05)
plt.tight_layout()
plt.savefig(f"{WORK}/af2ig_vs_boltz_figure.png",dpi=160,bbox_inches="tight")
print("\nsaved figure")
