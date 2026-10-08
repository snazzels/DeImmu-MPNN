#!/usr/bin/env python
"""Assemble the sweep results into a table + reduction-vs-recovery Pareto plot.
Run after 02_sweep_eval completes. Best-effort parsing of the per-config summaries;
reports whatever is present. Usage: PF=<project> python 03_pareto.py"""
import os, re, glob, csv, statistics as st
PF = os.environ["PF"]; OD = os.path.join(PF, "CAPE_MPNN", "data", "output")

def val(path, key):
    if not os.path.exists(path): return None
    for l in open(path):
        s = l.strip()
        if s.startswith(key + ":"): return s.split(":", 1)[1].strip().strip("'\"")
    return None

def grab_pct(path, pat):
    if not os.path.exists(path): return None
    m = re.search(pat, open(path).read())
    return float(m.group(1)) if m else None

def net_reduction(tag):
    """Mean per-protein netMHCIIpan reduction, recomputed from the raw per-sequence
    CSV rather than parsed from the summary line. The summary's mean is fragile: a
    protein with 0 base presented windows makes its per-protein % reduction divide by
    zero, producing a sentinel (e.g. fdbace08 = -199999982.1%) that destroys the mean
    (median stays sane). Recomputing here and dropping base==0 proteins reproduces every
    clean config's summary exactly (e.g. 7e908919 -> 30.2%) and repairs the degenerate one."""
    path = os.path.join(OD, f"validate_netmhciipan_{tag}.csv")
    if not os.path.exists(path): return None
    base, ft = {}, {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            d = base if r["model"] == "base" else ft
            d.setdefault(r["protein_name"], []).append(float(r["netmhc_total"]))
    reds = []
    for p in base:
        bm = st.mean(base[p])
        if bm == 0: continue          # per-protein reduction undefined; drop it
        reds.append((bm - st.mean(ft[p])) / bm * 100)
    return round(st.mean(reds), 1) if reds else None

rows = [("model_id","beta","lr","netmhc_reduction_pct","mixmhc2pred_reduction_pct","seq_recovery_ft_pct")]
data = []
for summ in sorted(glob.glob(os.path.join(OD, "validate_mixmhc2pred_sweep_*_summary.txt"))):
    tag = re.search(r"validate_mixmhc2pred_(sweep_\w+?)_summary", summ).group(1)
    mid = tag.replace("sweep_", "")
    hp = os.path.join(PF, "CAPE_MPNN", "artefacts", "CAPE-MPNN", "models", mid, "dpo_hparams.yaml")
    beta, lr = val(hp, "beta"), val(hp, "lr")
    mix = grab_pct(summ, r"per-protein reduction:\s*([\-0-9.]+)%")
    net = net_reduction(tag)
    rec = grab_pct(os.path.join(OD, f"seq_recovery_{tag}", "seq_recovery_summary.txt"), r"Fine-tuned:\s*([0-9.]+)%")
    rows.append((mid, beta, lr, net, mix, rec)); data.append((beta, lr, net, mix, rec, mid))

with open(os.path.join(OD, "sweep_pareto_table.csv"), "w", newline="") as fh:
    csv.writer(fh).writerows(rows)
print("wrote sweep_pareto_table.csv"); [print("  ", r) for r in rows[1:]]

# plot netMHCIIpan reduction vs sequence recovery (the Pareto trade-off)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pts = [(float(net), float(rec), beta) for beta, lr, net, mix, rec, mid in data if net and rec]
    if pts:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        xs = [p[1] for p in pts]; ys = [p[0] for p in pts]
        ax.scatter(xs, ys, c=[float(p[2]) for p in pts], cmap="viridis", s=45, edgecolor="k", linewidth=0.4)
        # NB: pts tuples are (net, rec, beta) -- annotate at (rec, net) to match the scatter.
        # A previous version unpacked `for x, y, b in pts`, which placed every label at the
        # TRANSPOSED coordinate (reduction, recovery); all but one fell outside the axes and
        # the survivor looked like a stray floating label.
        for net, rec, b in pts:
            ax.annotate(f"β{b}", (rec, net), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("sequence recovery (%)"); ax.set_ylabel("netMHCIIpan MHC-II reduction (%)")
        ax.set_title("DPO sweep: de-immunization vs fidelity Pareto"); ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(OD, "sweep_pareto.png"), dpi=300)
        print("wrote sweep_pareto.png")
    else:
        print("no (reduction, recovery) pairs yet — plot skipped")
except Exception as e:
    print("plot skipped:", e)
