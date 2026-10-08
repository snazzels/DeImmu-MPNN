#!/usr/bin/env python
"""Held-out allele generalisation test (roadmap step P2.3 / §21-'more alleles?').

Question: the model was DPO-trained to avoid presentation on SIX DRB1 alleles.
Does the reduction it learned TRANSFER to alleles it never saw — other DRB1, the
DRB3/4/5 paralogs, and the structurally distinct DQ and DP loci? If yes, the model
learned general MHC-II groove chemistry (broad coverage for free, and a stronger
claim than the six trained motifs); if the reduction collapses on held-out alleles,
it overfit to six motifs and the training panel must be widened.

Method: re-score the SAME base vs fine-tuned designs from Track 7
(eval_deimmunisation.csv) with netMHCIIpan on every allele, using ONE measuring
stick (netMHCIIpan) for both trained and held-out groups so the comparison is
apples-to-apples. Report per-group base-vs-finetuned % reduction; the trained group
reproduces Track 8 (~38%) and is the reference the held-out groups are judged against.

NOTE ON CIRCULARITY: netMHCIIpan is the PWM's teacher, so this is NOT an
immunogenicity-validation axis. It is the correct tool for THIS
question — does predicted presentation drop on unseen alleles — which is about
transfer across alleles, not about immunogenicity.

Outputs (data/output/):
  allele_generalisation.csv          per-sequence per-allele netMHCIIpan counts
  allele_generalisation_summary.txt  per-group reduction table + interpretation
  allele_generalisation_summary.json machine-readable
  allele_generalisation_figure.png   reduction by allele group
"""
import os, sys, csv, json, argparse, subprocess, tempfile, time, hashlib
import numpy as np
from scipy import stats

PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_CSV    = os.path.join(PF, "data", "output", "eval_deimmunisation.csv")
OUT_CSV     = os.path.join(PF, "data", "output", "allele_generalisation.csv")
OUT_SUMMARY = os.path.join(PF, "data", "output", "allele_generalisation_summary.txt")
OUT_JSON    = os.path.join(PF, "data", "output", "allele_generalisation_summary.json")
OUT_FIG     = os.path.join(PF, "data", "output", "allele_generalisation_figure.png")

# Allele groups. 'trained' = the six DRB1 the DPO objective actually used.
GROUPS = {
    "trained DRB1":   ["DRB1_0101","DRB1_0301","DRB1_0401","DRB1_0701","DRB1_1101","DRB1_1501"],
    "held-out DRB1":  ["DRB1_0404","DRB1_0801","DRB1_0901","DRB1_1201","DRB1_1301","DRB1_1401"],
    "DRB3/4/5":       ["DRB3_0101","DRB4_0101","DRB5_0101"],
    "HLA-DQ":         ["HLA-DQA10501-DQB10201","HLA-DQA10301-DQB10302","HLA-DQA10102-DQB10602"],
    "HLA-DP":         ["HLA-DPA10103-DPB10401","HLA-DPA10201-DPB10101","HLA-DPA10103-DPB10201"],
}
ALL_ALLELES = [a for g in GROUPS.values() for a in g]
THRESHOLD   = 2.0   # %Rank_EL <= 2% = presented (same as training/Track 8)
PEPTIDE_LEN = 15
BATCH_SIZE  = 5000


def run_netmhciipan(peptides, allele):
    ranks = {}
    for i in range(0, len(peptides), BATCH_SIZE):
        batch = peptides[i:i+BATCH_SIZE]
        with tempfile.TemporaryDirectory() as td:
            pf = os.path.join(td, "in.pep")
            open(pf, "w").write("\n".join(batch))
            out = subprocess.run(["netMHCIIpan","-inptype","1","-f",pf,"-a",allele],
                                 capture_output=True, check=True)
            for line in out.stdout.decode().splitlines():
                p = line.split()
                if len(p) < 10:
                    continue
                try: int(p[0])
                except ValueError: continue
                pep, rank = p[2], float(p[9])
                if pep not in ranks or rank < ranks[pep]:
                    ranks[pep] = rank
    return ranks


def kmers(seq):
    seq = seq.replace("*","").replace("-","").replace("/","")
    return [seq[i:i+PEPTIDE_LEN] for i in range(len(seq)-PEPTIDE_LEN+1)]


def select_proteins(rows, n):
    proteins = sorted(set(r["protein_name"] for r in rows))
    plen = {r["protein_name"]: int(r["protein_length"]) for r in rows}
    lengths = np.array([plen[p] for p in proteins])
    q = np.percentile(lengths, [25,50,75]); edges = [0,q[0],q[1],q[2],9999]
    rng = np.random.default_rng(42); per = max(1, n//4); sel=[]
    for lo,hi in zip(edges[:-1], edges[1:]):
        bucket = [p for p,l in zip(proteins,lengths) if lo<=l<hi]
        sel.extend(rng.choice(bucket, size=min(per,len(bucket)), replace=False).tolist())
    return sel[:n]


def pick_seqs(rows, prot, model, k):
    r = [x for x in rows if x["protein_name"]==prot and x["model"]==model]
    r.sort(key=lambda x: int(x["n_presented_total"]))
    if len(r) <= k: return [x["sequence"] for x in r]
    idx = np.round(np.linspace(0, len(r)-1, k)).astype(int)
    return [r[i]["sequence"] for i in idx]


def main(a):
    t0 = time.time()

    # Optional parametrisation (matches evaluate_mhc1_arm.py / validate_with_netmhciipan.py):
    # score a specific model's designs and version outputs so a new run never
    # overwrites a frozen prior result (project never-overwrite rule).
    global EVAL_CSV, OUT_CSV, OUT_SUMMARY, OUT_JSON, OUT_FIG
    if getattr(a, "in_csv", None):
        EVAL_CSV = a.in_csv
    if getattr(a, "out_tag", None):
        _od = os.path.join(PF, "data", "output")
        OUT_CSV     = os.path.join(_od, f"allele_generalisation_{a.out_tag}.csv")
        OUT_SUMMARY = os.path.join(_od, f"allele_generalisation_{a.out_tag}_summary.txt")
        OUT_JSON    = os.path.join(_od, f"allele_generalisation_{a.out_tag}_summary.json")
        OUT_FIG     = os.path.join(_od, f"allele_generalisation_{a.out_tag}_figure.png")
    if not a.from_csv:
        assert not os.path.exists(OUT_CSV), (
            f"Refusing to overwrite existing {OUT_CSV}; pass a distinct --out_tag.")

    if a.from_csv:
        # Recompute summary/figure from an existing scored CSV (no netMHCIIpan re-run).
        recs = list(csv.DictReader(open(OUT_CSV)))
        for r in recs:
            for k in list(r):
                if k.startswith("grp_") or k.startswith("n_"):
                    r[k] = int(r[k])
        proteins = sorted(set(r["protein_name"] for r in recs))
        print(f"Loaded {len(recs)} scored sequences from {OUT_CSV} ({len(proteins)} proteins)")
    else:
        rows = list(csv.DictReader(open(EVAL_CSV)))
        if getattr(a, "protein_list", None):
            # Pin the sample explicitly. REQUIRED for any run whose result will be
            # compared against another model's run -- see the note in --protein_list help.
            want = [ln.strip() for ln in open(a.protein_list) if ln.strip()
                    and not ln.startswith("#")]
            have = set(r["protein_name"] for r in rows)
            missing = [p for p in want if p not in have]
            assert not missing, (
                f"--protein_list asks for {len(missing)} protein(s) absent from {EVAL_CSV}: "
                f"{missing[:8]}{'…' if len(missing) > 8 else ''}. "
                "A pinned comparison cannot silently drop proteins.")
            proteins = want
            print(f"{len(proteins)} proteins PINNED from {a.protein_list}")
        else:
            proteins = select_proteins(rows, a.n_proteins)
            print(f"{len(proteins)} proteins selected (stratified by length)")
            print("  ⚠️  Sample was auto-selected, NOT pinned. This run's reductions are valid\n"
                  "      internally (base vs fine-tuned are paired on this set) but are NOT\n"
                  "      comparable to another model's run. Pass --protein_list for that.",
                  flush=True)

        recs = []
        for prot in proteins:
            for model in ["base","finetuned"]:
                for i, seq in enumerate(pick_seqs(rows, prot, model, a.k_seqs)):
                    recs.append({"protein_name":prot,"model":model,"seq_idx":i,"sequence":seq})
        print(f"{len(recs)} sequences ({a.k_seqs}/model/protein)")

        all_km = list({k for r in recs for k in kmers(r["sequence"])})
        print(f"{len(all_km)} unique 15-mers; scoring {len(ALL_ALLELES)} alleles with netMHCIIpan…")

        ranks = {}
        for al in ALL_ALLELES:
            ta = time.time()
            ranks[al] = run_netmhciipan(all_km, al)
            npres = sum(1 for v in ranks[al].values() if v <= THRESHOLD)
            print(f"  {al:26s} {time.time()-ta:5.1f}s  {npres}/{len(all_km)} presented", flush=True)

        for r in recs:
            km = kmers(r["sequence"])
            for al in ALL_ALLELES:
                ar = ranks[al]
                r[f"n_{al}"] = sum(1 for k in km if ar.get(k, 999) <= THRESHOLD)
        for r in recs:
            for g, als in GROUPS.items():
                r[f"grp_{g}"] = sum(r[f"n_{al}"] for al in als)

        with open(OUT_CSV, "w", newline="") as f:
            fn = ["protein_name","model","seq_idx"] + [f"grp_{g}" for g in GROUPS] + \
                 [f"n_{al}" for al in ALL_ALLELES] + ["sequence"]
            w = csv.DictWriter(f, fieldnames=fn); w.writeheader(); w.writerows(recs)

    # ── protein-set fingerprint ───────────────────────────────────────────────
    # Recorded in every output so a cross-run comparison can be checked rather than
    # assumed: two runs on different protein sets differ by model AND by sample,
    # and the two causes cannot be separated after the fact.
    _pset = sorted(set(proteins))
    _pset_md5 = hashlib.md5("\n".join(_pset).encode()).hexdigest()[:12]

    # ── per-group paired reduction (per protein: mean over seqs, base vs ft) ──
    lines, J = [], {"groups":{},
                    "protein_set": _pset,
                    "protein_set_md5": _pset_md5,
                    "n_proteins": len(_pset)}
    def emit(s=""): print(s); lines.append(s)
    emit("="*78); emit("HELD-OUT ALLELE GENERALISATION  (roadmap P2.3)"); emit("="*78)
    emit(f"Designs: Track 7 base vs fine-tuned, {len(proteins)} proteins x {a.k_seqs} seqs/model")
    emit(f"Scorer: netMHCIIpan-4.3, %Rank_EL <= {THRESHOLD}. One scorer for all groups.")
    emit("Reduction = per-protein mean presented windows, base vs fine-tuned (paired).")
    emit("The 'trained DRB1' row is the reference (reproduces Track 8, ~38%).")
    emit("")
    emit(f"{'allele group':<16}{'#al':>4}{'base':>9}{'finetuned':>11}{'reduction':>11}{'p (Wilcoxon)':>15}{'improved':>10}")
    emit("-"*78)
    for g, als in GROUPS.items():
        bpp, fpp = [], []
        for p in proteins:
            b = [r[f"grp_{g}"] for r in recs if r["protein_name"]==p and r["model"]=="base"]
            f = [r[f"grp_{g}"] for r in recs if r["protein_name"]==p and r["model"]=="finetuned"]
            if b and f: bpp.append(np.mean(b)); fpp.append(np.mean(f))
        bpp, fpp = np.array(bpp), np.array(fpp)
        # normalise per allele so groups of different size are comparable in the mean cols
        bpa, fpa = bpp/len(als), fpp/len(als)
        red = 100*(bpp.sum()-fpp.sum())/bpp.sum() if bpp.sum()>0 else float("nan")
        try: w_s, w_p = stats.wilcoxon(bpp, fpp)
        except ValueError: w_p = float("nan")
        imp = int((fpp < bpp).sum())
        emit(f"{g:<16}{len(als):>4}{bpa.mean():>9.1f}{fpa.mean():>11.1f}{red:>10.0f}%{w_p:>15.2g}{imp:>7}/{len(bpp)}")
        J["groups"][g] = dict(n_alleles=len(als), base_per_allele=float(bpa.mean()),
                              ft_per_allele=float(fpa.mean()), reduction_pct=float(red),
                              wilcoxon_p=float(w_p), improved=imp, n_proteins=int(len(bpp)))
    emit("-"*78); emit("")

    ref = J["groups"]["trained DRB1"]["reduction_pct"]
    def verdict(g):
        r = J["groups"][g]["reduction_pct"]; p = J["groups"][g]["wilcoxon_p"]
        ratio = r/ref if ref > 0 else float("nan")
        sig = "significant" if p < 0.05 else "n.s."
        if p >= 0.05 or r < 5:      label = "does NOT transfer"
        elif ratio >= 0.8:          label = "transfers fully"
        elif ratio >= 0.5:          label = "transfers substantially (partial attenuation)"
        else:                       label = "transfers weakly"
        return r, ratio, sig, label
    emit("INTERPRETATION  (retention = reduction / trained-DRB1 reference)")
    emit(f"  trained DRB1 (reference)   : {ref:4.0f}%")
    for g in ["held-out DRB1","DRB3/4/5","HLA-DQ","HLA-DP"]:
        r, ratio, sig, label = verdict(g)
        ret = f"{100*ratio:.0f}% of ref" if ref > 0 else "n/a"
        emit(f"  {g:<24}: {r:4.0f}%  ({ret:>11}, {sig})  -> {label}")
    emit("")
    # Bottom line is DERIVED, never hardcoded. It used to be a static paragraph asserting
    # "DP is the exception (no transfer)", which was written when that happened to be true
    # of the then-current model and silently contradicted the computed table for later ones
    # (e.g. abec4d2c: DP 36%, 84% of ref, p=3.7e-05 -> "transfers fully"). Fixed 2026-08-24.
    _held = ["held-out DRB1","DRB3/4/5","HLA-DQ","HLA-DP"]
    _v = {g: verdict(g) for g in _held}
    _transfers = [g for g in _held if _v[g][3] in ("transfers fully",
                                                   "transfers substantially (partial attenuation)")]
    _weak      = [g for g in _held if _v[g][3] == "transfers weakly"]
    _fails     = [g for g in _held if _v[g][3] == "does NOT transfer"]

    def _join(gs):
        gs = list(gs)
        if not gs:  return "none"
        if len(gs) == 1: return gs[0]
        return ", ".join(gs[:-1]) + " and " + gs[-1]

    emit("  Bottom line (derived from the table above, not a fixed narrative):")
    if _transfers:
        emit(f"    Transfers to {_join(_transfers)} " +
             "(" + "; ".join(f"{g} {_v[g][0]:.0f}% = {100*_v[g][1]:.0f}% of ref" for g in _transfers) + ").")
    if _weak:
        emit(f"    Transfers WEAKLY to {_join(_weak)} " +
             "(" + "; ".join(f"{g} {_v[g][0]:.0f}% = {100*_v[g][1]:.0f}% of ref" for g in _weak) + ").")
    if _fails:
        emit(f"    Does NOT transfer to {_join(_fails)} " +
             "(" + "; ".join(f"{g} {_v[g][0]:.0f}%, p={_v[g][2]}" for g in _fails) + ").")
    if len(_transfers) == len(_held):
        emit("    Every held-out group transfers: the model learned broadly-applicable groove")
        emit("    chemistry rather than six memorised motifs, and most of the MHC-II repertoire")
        emit("    is covered without retraining.")
    elif _weak or _fails:
        _gap = _join(_weak + _fails)
        emit(f"    {_gap} is the remaining gap; if that coverage matters it must be trained")
        emit("    explicitly (distinct heterodimer groove, weaker predictor).")
    emit("")
    emit("  NOTE: these are within-run paired comparisons (base vs fine-tuned on ONE protein")
    emit("  set) and are valid as such. Do NOT compare reductions across two runs of this")
    emit("  script unless their protein sets match -- check protein_set_md5 below, or run")
    emit("  tools/check_protein_match.py on the two CSVs.")
    emit("")
    emit(f"  protein_set_md5 : {_pset_md5}  (n={len(proteins)})")
    emit("")
    emit(f"Runtime: {(time.time()-t0)/60:.1f} min"); emit("="*78)

    open(OUT_SUMMARY,"w").write("\n".join(lines)+"\n")
    json.dump(J, open(OUT_JSON,"w"), indent=2)
    print(f"\nWrote {OUT_SUMMARY}, {OUT_JSON}, {OUT_CSV}")

    if not a.no_figure:
        make_fig(J)


def make_fig(J):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = list(J["groups"].keys())
    reds = [J["groups"][g]["reduction_pct"] for g in groups]
    colors = ["#4C72B0"] + ["#DD8452"]*(len(groups)-1)  # trained blue, held-out orange
    fig, ax = plt.subplots(figsize=(8,4.6))
    bars = ax.bar(range(len(groups)), reds, color=colors, edgecolor="k", linewidth=0.4)
    ref = J["groups"]["trained DRB1"]["reduction_pct"]
    ax.axhline(ref, ls="--", c="#4C72B0", lw=1, label=f"trained reference ({ref:.0f}%)")
    ax.axhline(0, c="k", lw=0.6)
    for i,(g,r) in enumerate(zip(groups,reds)):
        if r >= 0:
            ax.text(i, r+1, f"{r:.0f}%", ha="center", va="bottom", fontsize=9)
        else:  # label centred inside the negative bar, white, to avoid the tick labels
            ax.text(i, r/2, f"{r:.0f}%", ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g+("\n(trained)" if g=="trained DRB1" else "\n(held-out)") for g in groups], fontsize=8)
    ax.set_ylabel("MHC-II presentation reduction\n(base → fine-tuned, netMHCIIpan)")
    ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT_FIG, dpi=300)
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_proteins", type=int, default=25)
    p.add_argument("--k_seqs", type=int, default=4)
    p.add_argument("--from_csv", action="store_true",
                   help="recompute summary/figure from allele_generalisation.csv (no netMHCIIpan re-run)")
    p.add_argument("--no_figure", action="store_true")
    p.add_argument("--in_csv", type=str, default=None,
                   help="source designs CSV (default eval_deimmunisation.csv); e.g. the deployed lambda=1.0 designs")
    p.add_argument("--out_tag", type=str, default=None,
                   help="suffix for versioned outputs (e.g. v3_creg1.0_seed42)")
    p.add_argument("--protein_list", type=str, default=None,
                   help="file with one protein_name per line, pinning the exact sample. "
                        "REQUIRED whenever this run will be compared with another model's run. "
                        "Without it, select_proteins() derives length quartiles from THIS csv's "
                        "protein population, so two models with different eval sets get different "
                        "samples even though the RNG is seeded -- which is how the deployed model "
                        "and abec4d2c ended up sharing only 15 of 24 proteins.")
    main(p.parse_args())
