#!/usr/bin/env python
"""Binder-chain monomer self-consistency for the two complexes ESMFold2 could not do — P2-9 / JF-W13.

JF-W13: `selfconsistency_binder_chains_v2.py` folds the redesigned binder chains with ESMFold2
and skips 9cce (197 aa) and 9nds (248 aa) at its ~160 aa ceiling — i.e. it excludes the two
complexes most in need of the check, 9cce being the one flagged marginal on both binding
validators. Boltz-2 has no such ceiling. This script scores those two, and only those two.

⚠️ COMPARABILITY — READ BEFORE QUOTING THESE NUMBERS ALONGSIDE THE OTHER NINE.
The nine complexes in `selfconsistency_binder_v2_f16b51e6/results.txt` were folded with
ESMFold2; these two are folded with Boltz-2. A TM-to-native from one predictor is NOT
interchangeable with a TM-to-native from the other, so the two rows must not be pooled into
the ESMFold mean or ranked against those nine. What IS valid is the within-complex paired
base-vs-cape contrast, because both arms are folded by the same predictor on the same
backbone — that is the load-bearing statistic here, and it is the one JF-W13 asked for.

The TM statistic is imported verbatim from the ESMFold script (same gap-aware 1:1
correspondence, same d0, same top-3-by-global_score design selection), so the only variable
introduced is the structure predictor.

Prerequisite: run `selfconsistency_fold_boltz.py --outdir <OUT>` first to populate <OUT>/preds/.

Run (cape_mpnn env; CPU only):
    python CAPE_MPNN/tools/selfconsistency_binder_chains_boltz.py
"""
import os, sys, json
import numpy as np
from scipy import stats

PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PF, "tools"))

# Reuse the EXACT helpers of the ESMFold arm so the statistic cannot drift.
from selfconsistency_binder_chains_v2 import (          # noqa: E402
    binder_ca_by_resseq, read_pred_ca, kabsch_tm, top_designs, BINDER, TOPK, BR,
)

OUT = os.path.join(PF, "data", "output", "selfconsistency_binder_boltz_p2_9_f16b51e6")
PRED = os.path.join(OUT, "preds")
PIDS = ("9cce", "9nds")


def safe(design_id):
    return design_id.replace("|", "__").replace("/", "_")


def main():
    rows = []
    for pid in PIDS:
        chain = BINDER[pid]
        cplx = os.path.join(BR, "complexes", f"{pid}_cplx.pdb")
        ca, order = binder_ca_by_resseq(cplx, chain)
        if not ca:
            print(f"{pid}: no CA for chain {chain}, skip"); continue
        first = min(order)
        native_seq, _ = top_designs(pid, "base", TOPK)

        def tm_for(seq, design_id):
            p = os.path.join(PRED, safe(design_id) + ".pdb")
            if not os.path.exists(p):
                print(f"    missing prediction {p}"); return None
            L = len(seq)
            nat = np.full((L, 3), np.nan)
            for rs, xyz in ca.items():
                pos = rs - first
                if 0 <= pos < L:
                    nat[pos] = xyz
            valid = ~np.isnan(nat[:, 0])
            pred = read_pred_ca(p)
            if len(pred) != L:
                print(f"    {design_id}: pred {len(pred)} != seq {L}"); return None
            if valid.sum() < 15:
                return None
            _, tm = kabsch_tm(pred[valid], nat[valid], int(valid.sum()))
            return tm

        rec = {"pid": pid, "L": len(native_seq), "n_native_ca": len(order)}
        rec["native"] = tm_for(native_seq, f"{pid}|native")
        for mdl in ("base", "cape"):
            _, seqs = top_designs(pid, mdl, TOPK)
            tms = [tm_for(s, f"{pid}|{mdl}|{i+1}") for i, s in enumerate(seqs)]
            keep = [t for t in tms if t is not None]
            rec[mdl] = float(np.mean(keep)) if keep else None
            rec[mdl + "_all"] = keep
            rec[mdl + "_n"] = len(keep)
        rows.append(rec)
        print(f"  {pid}: native {rec['native']}, base {rec['base']}, cape {rec['cape']}", flush=True)

    L = ["=" * 78,
         "BINDER-CHAIN MONOMER SELF-CONSISTENCY — Boltz-2 arm (P2-9 / JF-W13)",
         "The two complexes the ESMFold2 arm skipped at its ~160 aa ceiling.",
         "TM to native binder backbone; fixed 1:1 correspondence, gap-aware; top-3 by global_score.",
         "=" * 78,
         "⚠️  DO NOT POOL WITH THE NINE ESMFold2 COMPLEXES — different structure predictor.",
         "    Valid comparison here is base vs cape WITHIN a complex (same predictor, same backbone).",
         "-" * 78,
         f"{'pid':6}{'L':>5}{'native':>9}{'base':>9}{'cape':>9}{'cape-base':>11}"]
    for r in rows:
        nat = f"{r['native']:.3f}" if r.get("native") is not None else "  -  "
        b, c = r.get("base"), r.get("cape")
        d = f"{c - b:+.3f}" if (b is not None and c is not None) else "   -   "
        L.append(f"{r['pid']:6}{r['L']:>5}{nat:>9}"
                 f"{(f'{b:.3f}' if b is not None else '  -  '):>9}"
                 f"{(f'{c:.3f}' if c is not None else '  -  '):>9}{d:>11}")
    L.append("-" * 78)
    pairs = [(r["base"], r["cape"]) for r in rows if r.get("base") is not None and r.get("cape") is not None]
    if pairs:
        bs = [p[0] for p in pairs]; cs = [p[1] for p in pairs]
        L.append(f"{'MEAN':6}{'':>5}{'':>9}{np.mean(bs):9.3f}{np.mean(cs):9.3f}{np.mean(cs)-np.mean(bs):+11.3f}")
        L.append(f"n = {len(pairs)} complexes — far too few for a meaningful paired test; "
                 f"per-complex values above are the result.")
    # per-design spread, which is what n=3 per arm actually supports
    L.append("")
    L.append("Per-design TM (top-3 each arm), and a within-complex unpaired t across those 3+3:")
    for r in rows:
        if r.get("base_all") and r.get("cape_all"):
            t, p = stats.ttest_ind(r["base_all"], r["cape_all"])
            L.append(f"  {r['pid']}: base {[round(x,3) for x in r['base_all']]}  "
                     f"cape {[round(x,3) for x in r['cape_all']]}  t={t:+.2f} p={p:.3f}")
    L.append("=" * 78)
    txt = "\n".join(L)
    print("\n" + txt)
    open(os.path.join(OUT, "results.txt"), "w").write(txt + "\n")
    json.dump(rows, open(os.path.join(OUT, "results.json"), "w"), indent=2)
    print(f"\nwrote {OUT}/results.txt and results.json")


if __name__ == "__main__":
    main()
