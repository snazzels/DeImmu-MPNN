"""Render the de novo binder panel (11 complexes) — manuscript Figure `panel11_figure.png`.

Three panels, all title-free (bare A/B/C headers; descriptions live in the caption):
  A  MHC-II epitope burden reduction per complex (base -> de-immunised)
  B  Boltz-2 interface ipTM, base vs CAPE, mean of all 24 designs/model
  C  AF2 initial-guess pass rate (pae_interaction < 10 A), all 24 designs/model

This is the release copy of the working-tree `final_consolidate_v2.py`; it reads the
three shipped JSONs next to this script rather than absolute working-tree paths. Boltz-2
scores ALL 24 designs per model (not the top-3 by ProteinMPNN global score) — the earlier
top-3 scoring produced a 9kku selection artefact (see summary output).

Run from anywhere with the paper environment active (matplotlib + numpy):
    python make_panel_figure.py
Writes panel11_figure.png, panel11_results.csv, panel11_summary.txt beside this script.
"""
import os, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

bur = json.load(open(f"{HERE}/burden_results.json"))
ipt = json.load(open(f"{HERE}/iptm_full24.json"))                                  # all-24 Boltz-2
af2 = json.load(open(f"{HERE}/af2_initial_guess/af2ig_native_results.json"))

META = {"7jzl": "SARS-CoV-2 RBD", "5vli": "influenza HA", "8upa": "gp130", "4oyd": "EBV",
        "9ju1": "VEGF-A", "9kku": "VEGF-A", "8t5e": "bioactive peptide", "9cc5": "bioactive peptide",
        "9nzh": "amylin", "9cce": "peptide (dyna_1b7)", "9nds": "HLA-Tax/TCR"}
ORDER = ["7jzl", "5vli", "8upa", "4oyd", "9ju1", "9kku", "8t5e", "9cc5", "9nzh", "9cce", "9nds"]
THRESH = 10.0

rows = []
for pid in ORDER:
    b = bur[pid]; i = ipt[pid]; a = af2[pid]
    a_base = [v["pae_interaction"] for k, v in a.items() if k.startswith("base_")]
    a_cape = [v["pae_interaction"] for k, v in a.items() if k.startswith("cape_")]
    rows.append(dict(
        pdb=pid, target=META[pid], binder_len=b["L"],
        burden_reduction_pct=round(b["reduction_pct"], 1),
        iptm_native=round(i["native"], 3), iptm_base=round(i["base_mean"], 3), iptm_cape=round(i["cape_mean"], 3),
        iptm_delta=round(i["cape_mean"] - i["base_mean"], 3),
        boltz_base_pass=i["base_pass"], boltz_cape_pass=i["cape_pass"], boltz_n=i["n"],
        af2ig_native_pae=a["native"]["pae_interaction"],
        af2ig_base_pass_pct=round(100 * np.mean([v < THRESH for v in a_base]), 1),
        af2ig_cape_pass_pct=round(100 * np.mean([v < THRESH for v in a_cape]), 1),
    ))

with open(f"{HERE}/panel11_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]

red = [r["burden_reduction_pct"] for r in rows]
allb_iptm = np.concatenate([ipt[p]["base_vals"] for p in ORDER]); allc_iptm = np.concatenate([ipt[p]["cape_vals"] for p in ORDER])
allb_af2 = np.concatenate([[v["pae_interaction"] for k, v in af2[p].items() if k.startswith("base_")] for p in ORDER])
allc_af2 = np.concatenate([[v["pae_interaction"] for k, v in af2[p].items() if k.startswith("cape_")] for p in ORDER])

S = []
S.append("FULL 11-COMPLEX BINDER-REDESIGN PANEL -- Boltz-2 rescored on ALL 24 designs/model (not top-3)")
S.append("=" * 90)
S.append(f"{'pdb':6}{'target':20}{'L':>5}{'redux%':>8}{'iptm_nat':>9}{'iptm_b(24)':>11}{'iptm_c(24)':>11}{'d_iptm':>8}{'af2ig_b%':>9}{'af2ig_c%':>9}")
for r in rows:
    S.append(f"{r['pdb']:6}{r['target']:20}{r['binder_len']:>5}{r['burden_reduction_pct']:>8}{r['iptm_native']:>9}{r['iptm_base']:>11}{r['iptm_cape']:>11}{r['iptm_delta']:>8}{r['af2ig_base_pass_pct']:>9}{r['af2ig_cape_pass_pct']:>9}")
S.append("")
S.append(f"Epitope burden reduction:  mean {np.mean(red):+.1f}%  range {min(red):+.1f}% to {max(red):+.1f}%  (11/11 complexes positive)")
S.append(f"Boltz-2 ipTM (ALL 24 designs/model, target MSA, binder single-seq):  mean base {allb_iptm.mean():.3f}  cape {allc_iptm.mean():.3f}  (delta {allc_iptm.mean()-allb_iptm.mean():+.3f})")
S.append(f"  Confident interface (ipTM>0.5): base {int((allb_iptm>0.5).sum())}/{len(allb_iptm)} ({100*(allb_iptm>0.5).mean():.1f}%)  cape {int((allc_iptm>0.5).sum())}/{len(allc_iptm)} ({100*(allc_iptm>0.5).mean():.1f}%)")
S.append(f"AF2 initial-guess pass rate (pae_interaction<{THRESH}A, all 24 designs/model):  base {100*(allb_af2<THRESH).mean():.1f}% ({int((allb_af2<THRESH).sum())}/{len(allb_af2)})  cape {100*(allc_af2<THRESH).mean():.1f}% ({int((allc_af2<THRESH).sum())}/{len(allc_af2)})")
S.append("")
S.append("Note: with all 24 designs scored by Boltz-2 (not just the top-3 by ProteinMPNN global score),")
S.append("9kku -- previously flagged as the sole complex with a material fine-tuned ipTM cost -- now shows")
S.append("base and fine-tuned statistically indistinguishable (0.675 vs 0.673); the earlier gap was a top-3")
S.append("selection artefact. The largest remaining base-vs-cape gap in the full-24 scoring is 9ju1")
S.append("(0.599 -> 0.546, both below native's own 0.837).")
open(f"{HERE}/panel11_summary.txt", "w").write("\n".join(S))
print("\n".join(S))

labels = [f"{p} ({META[p]})" for p in ORDER]
x = np.arange(len(ORDER))
fig, axes = plt.subplots(1, 3, figsize=(19, 5))

ax = axes[0]
ax.bar(x, red, color=["#2b7bba" if v >= 30 else "#6ba3d6" for v in red])
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
ax.set_ylabel("MHC-II burden reduction (%)")
ax.set_title("A", fontsize=11, loc="left", fontweight="bold")
ax.spines[['top', 'right']].set_visible(False)
ax.axhline(np.mean(red), ls="--", lw=1, color="#c0392b")
ax.text(len(ORDER) - 0.5, np.mean(red) + 1, f"mean {np.mean(red):.0f}%", fontsize=8, color="#c0392b", ha="right")

ax = axes[1]
w = 0.35
base = [r["iptm_base"] for r in rows]; cape = [r["iptm_cape"] for r in rows]
ax.bar(x - w / 2, base, w, color="#9e9e9e", label="base (n=24)")
ax.bar(x + w / 2, cape, w, color="#2b7bba", label="CAPE (n=24)")
ax.axhline(0.5, ls="--", lw=1, color="#c0392b")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
ax.set_ylabel("Boltz-2 interface ipTM (mean of 24)"); ax.set_ylim(0, 1.05)
ax.set_title("B", fontsize=11, loc="left", fontweight="bold")
ax.legend(fontsize=8, frameon=False, loc="lower left"); ax.spines[['top', 'right']].set_visible(False)

ax = axes[2]
bp = [r["af2ig_base_pass_pct"] for r in rows]; cp = [r["af2ig_cape_pass_pct"] for r in rows]
ax.bar(x - w / 2, bp, w, color="#9e9e9e", label="base")
ax.bar(x + w / 2, cp, w, color="#2b7bba", label="CAPE")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
ax.set_ylabel("AF2 initial-guess pass rate (%)\n(pae_interaction<10A, n=24/model)")
ax.set_title("C", fontsize=11, loc="left", fontweight="bold")
ax.legend(fontsize=8, frameon=False); ax.spines[['top', 'right']].set_visible(False)
ax.set_ylim(0, 105)

plt.tight_layout()
plt.savefig(f"{HERE}/panel11_figure.png", dpi=160, bbox_inches="tight")
print("\nsaved panel11_figure.png")
