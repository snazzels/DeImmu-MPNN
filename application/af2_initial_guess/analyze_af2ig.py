import os
import json, os
import numpy as np
WORK=os.path.dirname(os.path.abspath(__file__))
DEST=os.environ["CAPE_ROOT"] + "/CAPE_MPNN/data/output/binder_redesign_v1"
r=json.load(open(f"{WORK}/af2ig_native_results.json"))
ipt=json.load(open(f"{DEST}/iptm_results_msa.json"))
THRESH=10.0
ORDER=["7jzl","5vli","8upa","4oyd","9ju1","9kku"]
print(f"{'pid':6}{'native_pae':>11}{'base pass%':>12}{'cape pass%':>12}{'base_med':>10}{'cape_med':>10}   boltz(base/cape ipTM top3)")
summary={}
for pid in ORDER:
    d=r[pid]
    base_vals=[v["pae_interaction"] for k,v in d.items() if k.startswith("base_")]
    cape_vals=[v["pae_interaction"] for k,v in d.items() if k.startswith("cape_")]
    nat=d["native"]["pae_interaction"]
    base_pass=100*np.mean([v<THRESH for v in base_vals])
    cape_pass=100*np.mean([v<THRESH for v in cape_vals])
    summary[pid]=dict(native_pae=nat, n_base=len(base_vals), n_cape=len(cape_vals),
                       base_pass_pct=base_pass, cape_pass_pct=cape_pass,
                       base_median=float(np.median(base_vals)), cape_median=float(np.median(cape_vals)),
                       boltz_base=ipt[pid]["base_mean"], boltz_cape=ipt[pid]["cape_mean"])
    print(f"{pid:6}{nat:11.2f}{base_pass:11.1f}%{cape_pass:11.1f}%{np.median(base_vals):10.2f}{np.median(cape_vals):10.2f}   {ipt[pid]['base_mean']:.3f}/{ipt[pid]['cape_mean']:.3f}")
json.dump(summary, open(f"{WORK}/af2ig_summary.json","w"), indent=2)
allb=[r[p][k]["pae_interaction"] for p in ORDER for k in r[p] if k.startswith("base_")]
allc=[r[p][k]["pae_interaction"] for p in ORDER for k in r[p] if k.startswith("cape_")]
print(f"\nOverall pass rate (pae_interaction<{THRESH}A): base {100*np.mean([v<THRESH for v in allb]):.1f}%  cape {100*np.mean([v<THRESH for v in allc]):.1f}%  (n={len(allb)} each)")
