#!/usr/bin/env python
"""Stage 3/3 of the ESMFold self-consistency benchmark (roadmap step P1.3).

For every design, superimpose the ESMFold2 prediction onto the native backbone it
was designed for and report TM-score + Cα-RMSD. Correspondence is exact and fixed
(design residue i <-> native residue i); residues without native density (mask==0)
are dropped. Superposition is RMSD-optimal (Kabsch); the TM-score is normalised by
the number of aligned native residues. This fixed-correspondence TM is the standard
ProteinMPNN self-consistency metric and is a (mild) lower bound on the TM-optimal
value TMalign would report.

Question this answers: does the DPO de-immunisation (finetuned) degrade designed-
sequence foldability relative to the base ProteinMPNN, and how does foldability
trade off against MHC-II epitope burden?

Reads
-----
data/output/selfconsistency/designs.csv        (Stage 1: model, mhc2_total)
data/output/selfconsistency/refs/<name>.npz     (Stage 1: native backbone + mask)
data/output/selfconsistency/preds/<id>.pdb       (Stage 2: predictions)
data/output/selfconsistency/fold_metrics.csv     (Stage 2: plddt, ptm)

Writes
------
data/output/selfconsistency/selfconsistency_results.csv
data/output/selfconsistency/selfconsistency_summary.txt
data/output/selfconsistency/selfconsistency_summary.json
data/output/selfconsistency/selfconsistency_figure.png
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from scipy import stats

PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PF, "data", "output", "selfconsistency")
REF_DIR = os.path.join(OUT_DIR, "refs")
PRED_DIR = os.path.join(OUT_DIR, "preds")
TM_THRESH = 0.85   # paper_outline foldability bar


def safe(design_id):
    return design_id.replace("|", "__").replace("/", "_")


def read_ca_from_pdb(path):
    """Ordered Cα coordinates (one per residue) from a single-chain PDB."""
    ca = []
    with open(path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                ca.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(ca, dtype=np.float64)


def kabsch_rmsd_tm(P, Q, L_norm):
    """Superimpose P onto Q (RMSD-optimal), return (rmsd, tm) over the L pairs.
    TM normalised by L_norm (native aligned length)."""
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    V, _, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1, 1, d])
    R = V @ D @ Wt
    Pr = Pc @ R
    di = np.linalg.norm(Pr - Qc, axis=1)
    rmsd = float(np.sqrt((di ** 2).mean()))
    d0 = 1.24 * (max(L_norm, 19) - 15) ** (1.0 / 3.0) - 1.8
    tm = float((1.0 / L_norm) * np.sum(1.0 / (1.0 + (di / d0) ** 2)))
    return rmsd, tm


def score_one(design_id, protein_name, ref_cache):
    pdb = os.path.join(PRED_DIR, safe(design_id) + ".pdb")
    if not os.path.exists(pdb):
        return None
    if protein_name not in ref_cache:
        z = np.load(os.path.join(REF_DIR, f"{protein_name}.npz"), allow_pickle=True)
        ref_cache[protein_name] = (z["bb"][:, 1, :].astype(np.float64), z["mask"].astype(bool))
    ref_ca_full, valid = ref_cache[protein_name]
    pred_ca = read_ca_from_pdb(pdb)
    if len(pred_ca) != len(ref_ca_full):
        return {"error": f"len mismatch pred {len(pred_ca)} vs ref {len(ref_ca_full)}"}
    ref_ca = ref_ca_full[valid]
    pred_ca = pred_ca[valid]
    L = len(ref_ca)
    if L < 15:
        return {"error": f"too few valid residues ({L})"}
    rmsd, tm = kabsch_rmsd_tm(pred_ca, ref_ca, L)
    return {"tm": tm, "rmsd": rmsd, "n_aligned": L}


def main(a):
    designs = pd.read_csv(os.path.join(OUT_DIR, "designs.csv"))
    fold = pd.read_csv(os.path.join(OUT_DIR, "fold_metrics.csv"))
    df = designs.merge(fold[["design_id", "plddt", "ptm", "status"]], on="design_id", how="left")

    ref_cache = {}
    recs = []
    for row in df.itertuples(index=False):
        base = dict(design_id=row.design_id, protein_name=row.protein_name, model=row.model,
                    seq_idx=row.seq_idx, protein_length=row.protein_length,
                    mhc2_total=row.mhc2_total, plddt=row.plddt, ptm=row.ptm,
                    fold_status=row.status)
        res = score_one(row.design_id, row.protein_name, ref_cache)
        if res is None:
            base.update(tm=np.nan, rmsd=np.nan, n_aligned=np.nan, note="no_pdb")
        elif "error" in res:
            base.update(tm=np.nan, rmsd=np.nan, n_aligned=np.nan, note=res["error"])
        else:
            base.update(**res, note="")
        recs.append(base)
    out = pd.DataFrame(recs)
    out.to_csv(os.path.join(OUT_DIR, "selfconsistency_results.csv"), index=False)

    lines, J = [], {}
    def emit(s=""):
        print(s); lines.append(s)

    emit("=" * 78)
    emit("ESMFOLD2 SELF-CONSISTENCY  (roadmap step P1.3)")
    emit("=" * 78)
    n_fold = int((out.tm.notna()).sum())
    emit(f"Designs scored (TM computed) : {n_fold} / {len(out)}")
    emit(f"Fold failures (OOM/err/no pdb): {int(out.tm.isna().sum())}")
    emit(f"TM normalised by native aligned length; correspondence fixed 1:1; "
         f"masked (missing-density) residues excluded.")
    emit(f"Foldability bar (paper_outline): TM >= {TM_THRESH}")
    emit("")

    def block(sub, label):
        s = sub[sub.tm.notna()]
        if len(s) == 0:
            emit(f"{label:<12} n=0"); return None
        emit(f"{label:<12} n={len(s):<4} "
             f"TM {s.tm.mean():.3f}±{s.tm.std():.3f} (med {s.tm.median():.3f})  "
             f"RMSD {s.rmsd.mean():.2f}±{s.rmsd.std():.2f} Å  "
             f"TM>={TM_THRESH}: {100*(s.tm>=TM_THRESH).mean():.0f}%  "
             f"pLDDT {pd.to_numeric(s.plddt,errors='coerce').mean():.2f}")
        return dict(n=len(s), tm_mean=float(s.tm.mean()), tm_median=float(s.tm.median()),
                    tm_std=float(s.tm.std()), rmsd_mean=float(s.rmsd.mean()),
                    frac_pass=float((s.tm>=TM_THRESH).mean()),
                    plddt_mean=float(pd.to_numeric(s.plddt,errors='coerce').mean()))

    emit("-" * 78); emit("Per-design summary by model"); emit("-" * 78)
    for mdl in ["native", "base", "finetuned"]:
        J[mdl] = block(out[out.model == mdl], mdl)
    emit("")
    emit("The 'native' row is a positive control: folding the wild-type sequence should")
    emit("recover the native backbone at high TM, validating the whole pipeline.")
    emit("")

    # Paired per-protein: base vs finetuned (mean TM over K designs per protein)
    piv = (out[out.model.isin(["base", "finetuned"]) & out.tm.notna()]
           .groupby(["protein_name", "model"]).tm.mean().unstack("model").dropna())
    emit("-" * 78); emit(f"Paired base vs finetuned foldability (n={len(piv)} proteins)"); emit("-" * 78)
    if len(piv) >= 3:
        b, f = piv["base"].values, piv["finetuned"].values
        dtm = f - b
        t_stat, t_p = stats.ttest_rel(f, b)
        try: w_stat, w_p = stats.wilcoxon(f, b)
        except ValueError: w_stat, w_p = float("nan"), float("nan")
        emit(f"base      mean TM : {b.mean():.3f}")
        emit(f"finetuned mean TM : {f.mean():.3f}")
        emit(f"Delta (ft - base) : {dtm.mean():+.3f} ± {dtm.std():.3f}  "
             f"(finetuned better on {int((dtm>0).sum())}/{len(dtm)} proteins)")
        emit(f"Paired t-test     : t={t_stat:.2f}, p={t_p:.3g}")
        emit(f"Wilcoxon          : W={w_stat:.1f}, p={w_p:.3g}")
        J["paired"] = dict(n=len(piv), base_tm=float(b.mean()), ft_tm=float(f.mean()),
                           delta=float(dtm.mean()), delta_std=float(dtm.std()),
                           t_p=float(t_p), wilcoxon_p=float(w_p),
                           ft_better=int((dtm>0).sum()))
        emit("")
        emit("Interpretation: a Delta near zero (n.s.) means de-immunisation preserves")
        emit("foldability; a significantly negative Delta would mean the MHC-II objective")
        emit("costs structural quality — the trade-off the paper must quantify.")
    emit("")

    # Foldability vs immunogenicity coupling
    emit("-" * 78); emit("Foldability vs MHC-II burden (per design)"); emit("-" * 78)
    for mdl in ["base", "finetuned"]:
        s = out[(out.model == mdl) & out.tm.notna()]
        if len(s) > 3:
            r, p = stats.pearsonr(s.mhc2_total, s.tm)
            emit(f"{mdl:<10} Pearson(TM, MHC-II windows) r={r:+.3f} p={p:.3g}  "
                 f"(mean MHC-II {s.mhc2_total.mean():.1f})")
    emit("")

    with open(os.path.join(OUT_DIR, "selfconsistency_summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT_DIR, "selfconsistency_summary.json"), "w") as fh:
        json.dump(J, fh, indent=2)
    print(f"\nWrote results.csv, summary.txt, summary.json to {OUT_DIR}")

    if not a.no_figure:
        make_figure(out)


def make_figure(out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"native": "#555555", "base": "#4C72B0", "finetuned": "#DD8452"}
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.4))

    # (A) TM distribution by model
    a0 = ax[0]
    data, labels, colors = [], [], []
    for mdl in ["native", "base", "finetuned"]:
        s = out[(out.model == mdl) & out.tm.notna()]
        if len(s): data.append(s.tm.values); labels.append(mdl); colors.append(C[mdl])
    parts = a0.violinplot(data, showmedians=True)
    for pc, c in zip(parts["bodies"], colors): pc.set_facecolor(c); pc.set_alpha(0.7)
    a0.axhline(TM_THRESH, ls="--", c="k", lw=1, label=f"TM={TM_THRESH}")
    a0.set_xticks(range(1, len(labels) + 1)); a0.set_xticklabels(labels)
    a0.set_ylabel("TM-score to native backbone"); a0.set_title("(A) Self-consistency by model")
    a0.legend(fontsize=8)

    # (B) paired base vs finetuned per protein
    a1 = ax[1]
    piv = (out[out.model.isin(["base", "finetuned"]) & out.tm.notna()]
           .groupby(["protein_name", "model"]).tm.mean().unstack("model").dropna())
    if len(piv):
        a1.scatter(piv["base"], piv["finetuned"], s=26, alpha=0.75,
                   color=C["finetuned"], edgecolor="k", linewidth=0.3)
        lo = min(piv.min().min(), 0.2); a1.plot([lo, 1], [lo, 1], "k--", lw=1)
        a1.set_xlabel("base mean TM"); a1.set_ylabel("finetuned mean TM")
        a1.set_title(f"(B) Paired per protein (n={len(piv)})")
        a1.set_xlim(lo, 1); a1.set_ylim(lo, 1)

    # (C) foldability vs MHC-II burden
    a2 = ax[2]
    for mdl in ["base", "finetuned"]:
        s = out[(out.model == mdl) & out.tm.notna()]
        a2.scatter(s.mhc2_total, s.tm, s=18, alpha=0.5, color=C[mdl], label=mdl)
    a2.axhline(TM_THRESH, ls="--", c="k", lw=1)
    a2.set_xlabel("MHC-II presented windows"); a2.set_ylabel("TM-score")
    a2.set_title("(C) Foldability vs immunogenicity"); a2.legend(fontsize=8)

    for a in ax: a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "selfconsistency_figure.png"), dpi=300)
    print(f"Wrote selfconsistency_figure.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no_figure", action="store_true")
    p.add_argument("--outdir", default=OUT_DIR)
    args = p.parse_args()
    OUT_DIR = args.outdir
    REF_DIR = os.path.join(OUT_DIR, "refs")
    PRED_DIR = os.path.join(OUT_DIR, "preds")
    main(args)
