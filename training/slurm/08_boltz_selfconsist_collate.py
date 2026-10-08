#!/usr/bin/env python
"""Collate the Boltz-2 full-length self-consistency results across sweep configs.
Run after 07_boltz_selfconsist.sbatch finishes. Parses each per-config
selfconsistency_summary.json (the 'paired' block) into one table.
Usage: PF=<project> python 08_boltz_selfconsist_collate.py"""
import os, json, glob, csv
PF = os.environ["PF"]; OD = os.path.join(PF, "CAPE_MPNN", "data", "output")

# hash -> (beta, lr, netMHCIIpan MHC-II reduction %) for context (from the sweep Pareto)
META = {
    "7e908919": ("0.05", "3e-7", 30.2), "45cc9026": ("0.02", "3e-7", 58.8),
    "9d8808b4": ("0.02", "1e-6", 76.9), "35842f8a": ("0.01", "3e-7", 81.0),
    "e842a798": ("0.01", "1e-6", 95.8), "f16b51e6": ("0.05", "1e-6", 37.9),
    "005c9a50": ("0.01", "1e-7", 23.3), "4de4315e": ("0.02", "1e-7", 20.6),
    "52f726b6": ("0.1", "1e-6", 28.3), "72757141": ("0.2", "1e-6", 24.1),
    "7379f298": ("0.05", "1e-7", 17.0), "b3e54f2f": ("0.1", "3e-7", 21.7),
    "b4d73ff2": ("0.2", "1e-7", 10.6), "ea3c3424": ("0.1", "1e-7", 12.5),
    "fdbace08": ("0.2", "3e-7", 18.0),
}

rows = [("model_id", "beta", "lr", "mhc2_reduction_pct", "n_prot",
         "base_tm", "ft_tm", "delta_ft_minus_base", "delta_std", "paired_t_p", "ft_better")]
for js in sorted(glob.glob(os.path.join(OD, "selfconsistency_boltz_sweep_*", "selfconsistency_summary.json"))):
    mid = os.path.basename(os.path.dirname(js)).replace("selfconsistency_boltz_sweep_", "")
    beta, lr, red = META.get(mid, ("?", "?", ""))
    p = json.load(open(js)).get("paired")
    if not p:
        print(f"  {mid}: no 'paired' block (n<3 proteins scored?) — skipping"); continue
    rows.append((mid, beta, lr, red, p["n"],
                 round(p["base_tm"], 3), round(p["ft_tm"], 3), round(p["delta"], 3),
                 round(p["delta_std"], 3), round(p["t_p"], 4), f"{p['ft_better']}/{p['n']}"))

outp = os.path.join(OD, "sweep_boltz_selfconsist_table.csv")
with open(outp, "w", newline="") as fh:
    csv.writer(fh).writerows(rows)
print(f"wrote {outp}")
for r in rows[1:]:
    print("  ", r)
if len(rows) == 1:
    print("  (no results yet — has 07_boltz_selfconsist finished?)")
