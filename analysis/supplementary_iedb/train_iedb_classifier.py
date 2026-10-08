#!/usr/bin/env python
"""Train an IEDB immunogenicity classifier -- the independent validation axis.

Prerequisite: mine_IEDB_negatives.py.

WHAT THIS IS FOR
----------------
Every immunogenicity number in this project comes from netMHCIIpan, which the
training PWM was derived from -- so the evaluation is circular. Worse, Track 14
showed netMHCIIpan presentation predicts experimental immunogenicity at AUC 0.492
in the deployment regime (chance).

This script trains a classifier on EXPERIMENTAL T-cell outcomes (IEDB), never on
netMHCIIpan output, to give an evaluation axis independent of the training
objective.

CRITICAL: THE CLASSIFIER MUST BE VALIDATED BEFORE IT IS USED
------------------------------------------------------------
It is trained on the binder-matched dataset, where every peptide is a predicted
MHC-II binder and only the experimental outcome differs. Two confounds could let
it post a good AUC while learning nothing about immunogenicity:

  1. Overlapping-scan leakage. 6.1% of peptides share an 11-mer with another
     peptide (IEDB is full of overlapping peptide walks across one antigen).
     Random splits would put near-duplicates in both train and test.
     -> Mitigated by GROUPED splits on source molecule.

  2. Organism shortcut. 80% of the data is two organisms (B. pertussis 7,193;
     M. tuberculosis 4,559) with very different class balance (13.9% vs 3.6%
     immunogenic). A model can score well by recognising "pertussis-like"
     composition.
     -> Detected by the HELD-OUT ORGANISM split. If AUC collapses there, the
        molecule-split AUC was substantially organism identity, not immunogenicity.

Note on the paper outline: it specifies "MMseqs2 clustering at 30% identity".
That is meaningless for 15-mers (30% of 15 = ~4 residues). Grouping by source
molecule is the correct leakage control here, and grouping by source organism is
the stronger generalisation test.

CONTROLS (all reported, all must behave)
----------------------------------------
  * shuffled labels        -> AUC must be ~0.5, else the split leaks
  * binding-only features  -> netMHCIIpan %Ranks alone; expect ~0.5 (Track 14)
  * sequence models        -> logistic regression and an MLP on one-hot 15x20

Outputs
-------
data/output/iedb_classifier_summary.txt
data/output/iedb_classifier_figure.png
data/output/iedb_classifier.pt          (best model, only if it validates)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import mannwhitneyu

PF = os.environ.get("PF", os.path.join(os.path.dirname(__file__), ".."))
IEDB = os.path.join(PF, "data", "input", "immuno", "mhc_2", "iedb")
OUT_TXT = os.path.join(PF, "data", "output", "iedb_classifier_summary.txt")
OUT_FIG = os.path.join(PF, "data", "output", "iedb_classifier_figure.png")
OUT_PT = os.path.join(PF, "data", "output", "iedb_classifier.pt")

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
L = 15
ALLELES = ["DRB1_0101", "DRB1_0301", "DRB1_0401", "DRB1_0701", "DRB1_1101", "DRB1_1501"]

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── metrics ───────────────────────────────────────────────────────────────────

def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    p, n = s[y == 1], s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    u, _ = mannwhitneyu(p, n, alternative="two-sided")
    return u / (len(p) * len(n))


def auprc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    rec = tp / max(y.sum(), 1)
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


# ── featurisation ─────────────────────────────────────────────────────────────

def onehot(peptides):
    X = np.zeros((len(peptides), L * 20), dtype=np.float32)
    for i, p in enumerate(peptides):
        for j, a in enumerate(p[:L]):
            if a in AA_IDX:
                X[i, j * 20 + AA_IDX[a]] = 1.0
    return X


def composition(peptides):
    """Register-invariant features: AA frequency + dipeptide counts.

    The MHC-II binding core sits at a variable register inside the 15-mer, so a
    positional one-hot has a poor inductive bias. These features are position
    independent, so a failure here cannot be blamed on the encoding.
    """
    X = np.zeros((len(peptides), 20 + 400), dtype=np.float32)
    for i, p in enumerate(peptides):
        for a in p:
            if a in AA_IDX:
                X[i, AA_IDX[a]] += 1.0
        for a, b in zip(p, p[1:]):
            if a in AA_IDX and b in AA_IDX:
                X[i, 20 + AA_IDX[a] * 20 + AA_IDX[b]] += 1.0
    X[:, :20] /= L
    X[:, 20:] /= (L - 1)
    return X


class CNN(nn.Module):
    """Small 1D CNN over the one-hot peptide; translation-equivariant, so it can
    find a binding-core motif at any register."""

    def __init__(self, d=None, ch=64, p=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(20, ch, 5, padding=2), nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU(),
        )
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(ch * 2, 1)

    def forward(self, x):
        x = x.view(-1, L, 20).transpose(1, 2)      # (B, 20, L)
        h = self.conv(x)
        h = torch.cat([h.max(dim=2).values, h.mean(dim=2)], dim=1)
        return self.fc(self.drop(h)).squeeze(-1)


def binding_features(df):
    """netMHCIIpan %Ranks only -- the 'can binding alone do it?' control."""
    X = df[ALLELES].to_numpy(np.float32)
    X = np.log1p(np.clip(X, 0, 100))
    return np.c_[X, df.n_alleles_presented.to_numpy(np.float32)]


# ── grouped splitting ─────────────────────────────────────────────────────────

def group_split(groups, y, frac=(0.7, 0.15, 0.15), seed=0):
    """Assign whole groups to train/val/test, greedily balancing positives."""
    rng = np.random.default_rng(seed)
    g = pd.DataFrame({"g": groups, "y": y})
    stats = g.groupby("g").y.agg(["size", "sum"]).sample(frac=1.0, random_state=seed)
    targets = np.array(frac) * len(g)
    cur = np.zeros(3)
    assign = {}
    # big groups first so a single huge antigen cannot swamp a split
    for gname, row in stats.sort_values("size", ascending=False).iterrows():
        k = int(np.argmax(targets - cur))
        assign[gname] = k
        cur[k] += row["size"]
    idx = np.array([assign[x] for x in groups])
    return idx


def organism_split(org, y, holdout=("Bordetella pertussis Tohama I",)):
    """Train on everything except the held-out organism(s); test on them."""
    te = np.isin(org, list(holdout))
    return (~te).astype(int) * 0 + np.where(te, 2, 0)  # 0 = train, 2 = test


# ── models ────────────────────────────────────────────────────────────────────

class LogReg(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Linear(d, 1)

    def forward(self, x):
        return self.f(x).squeeze(-1)


class MLP(nn.Module):
    def __init__(self, d, h=256, p=0.4):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h // 2, 1),
        )

    def forward(self, x):
        return self.f(x).squeeze(-1)


def train_model(model, Xtr, ytr, Xva, yva, epochs=200, lr=1e-3, wd=1e-4, patience=30,
                batch_size=256):
    """Minibatch AdamW with early stopping on validation AUROC.

    Full-batch gradient descent (one update per epoch) badly undertrains the MLP
    and would produce a false 'no signal' verdict. Minibatching gives each model
    a fair shot before we conclude anything about the data.
    """
    model = model.to(DEV)
    Xtr_t = torch.tensor(Xtr, device=DEV)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    Xva_t = torch.tensor(Xva, device=DEV)
    n = len(ytr)

    # class weighting: negatives outnumber positives ~4.5:1
    pos_w = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)],
                         dtype=torch.float32, device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            sva = model(Xva_t).cpu().numpy()
        a = auroc(yva, sva)
        if a > best + 1e-4:
            best, best_state, bad = a, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        s = model(torch.tensor(X, device=DEV)).cpu().numpy()
    return auroc(y, s), auprc(y, s), s


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    torch.manual_seed(0)
    np.random.seed(0)

    ds = pd.read_csv(os.path.join(IEDB, "binder_matched_dataset.csv"))
    ann = pd.read_csv(os.path.join(IEDB, "peptide_source_annotation.csv"))
    df = ds.merge(ann, on="peptide", how="left")
    df["source_molecule"] = df.source_molecule.fillna("UNKNOWN")
    df["source_organism"] = df.source_organism.fillna("UNKNOWN")
    df["y"] = (df.label == "immunogenic").astype(int)

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit("IEDB IMMUNOGENICITY CLASSIFIER  (roadmap step P1.4)")
    emit("=" * 78)
    emit(f"Peptides: {len(df):,}   immunogenic: {df.y.sum():,}   "
         f"non-immunogenic: {(1-df.y).sum():,}   ({(1-df.y).sum()/max(df.y.sum(),1):.2f}:1)")
    emit(f"Source molecules: {df.source_molecule.nunique():,}   "
         f"organisms: {df.source_organism.nunique():,}")
    emit(f"Device: {DEV}")
    emit("")
    emit("Every peptide is a predicted MHC-II binder (binder-matched set), so the")
    emit("classifier cannot succeed by learning binding. Trained on experimental")
    emit("T-cell outcomes only; netMHCIIpan output is never an input feature")
    emit("(except in the explicit binding-only control).")
    emit("")

    emit("-" * 78)
    emit("Composition by source organism (the confound to worry about)")
    emit("-" * 78)
    g = df.groupby("source_organism").y.agg(["size", "mean"]).sort_values("size", ascending=False)
    emit(f"{'organism':<48}{'n':>8}{'% immunogenic':>16}")
    for name, row in g.head(5).iterrows():
        emit(f"{str(name)[:46]:<48}{int(row['size']):>8,}{100*row['mean']:>15.1f}%")
    emit("")

    X_seq = onehot(df.peptide.tolist())
    X_comp = composition(df.peptide.tolist())
    X_bind = binding_features(df)
    y = df.y.to_numpy()

    results = {}

    # ── Split A: grouped by source molecule ──────────────────────────────────
    emit("=" * 78)
    emit("SPLIT A -- grouped by source molecule (prevents overlapping-scan leakage)")
    emit("=" * 78)
    sp = group_split(df.source_molecule.to_numpy(), y, seed=args.seed)
    tr, va, te = sp == 0, sp == 1, sp == 2
    emit(f"train {tr.sum():,} (pos {y[tr].sum():,})   "
         f"val {va.sum():,} (pos {y[va].sum():,})   test {te.sum():,} (pos {y[te].sum():,})")
    if y[te].sum() < 20:
        emit("WARNING: too few positives in test split; AUC will be unstable.")
    emit("")

    MODELS = [
        ("logistic regression (one-hot)", "X_seq", LogReg),
        ("MLP (one-hot)", "X_seq", MLP),
        ("CNN (one-hot)", "X_seq", CNN),
        ("MLP (composition+dipeptide)", "X_comp", MLP),
        ("CONTROL binding-only (%Rank)", "X_bind", LogReg),
    ]
    FEAT = {"X_seq": X_seq, "X_comp": X_comp, "X_bind": X_bind}

    emit(f"{'model':<34}{'val AUC':>10}{'test AUC':>11}{'test AUPRC':>13}")
    best_of_split, best_name = -1.0, None
    for name, fk, Model in MODELS:
        X = FEAT[fk]
        m, vauc = train_model(Model(X.shape[1]), X[tr], y[tr], X[va], y[va],
                              epochs=args.epochs)
        tauc, tprc, s = evaluate(m, X[te], y[te])
        emit(f"{name:<34}{vauc:>10.3f}{tauc:>11.3f}{tprc:>13.3f}")
        results[name] = dict(val=vauc, test=tauc, auprc=tprc)
        if not name.startswith("CONTROL") and vauc > best_of_split:
            best_of_split, best_name, best_model = vauc, name, m

    # shuffled-label control
    rng = np.random.default_rng(0)
    y_sh = y.copy()
    y_sh[tr] = rng.permutation(y_sh[tr])
    m, vauc = train_model(MLP(X_seq.shape[1]), X_seq[tr], y_sh[tr], X_seq[va], y[va],
                          epochs=min(args.epochs, 120))
    tauc, tprc, _ = evaluate(m, X_seq[te], y[te])
    emit(f"{'CONTROL shuffled labels':<34}{vauc:>10.3f}{tauc:>11.3f}{tprc:>13.3f}")
    results["shuffled"] = dict(val=vauc, test=tauc, auprc=tprc)
    emit("")
    emit("Baseline AUPRC (predict-all-positive) = "
         f"{y[te].mean():.3f}  -- compare AUPRC against this, not against 0.5")
    emit("")

    # ── Split B: held-out organism ───────────────────────────────────────────
    emit("=" * 78)
    emit("SPLIT B -- held-out organism (does it generalise, or learn the organism?)")
    emit("=" * 78)
    holdout = args.holdout
    te_o = (df.source_organism == holdout).to_numpy()
    trva = ~te_o
    # carve a val set out of train by molecule
    sp2 = group_split(df.source_molecule.to_numpy()[trva], y[trva], frac=(0.85, 0.15, 0.0),
                      seed=args.seed)
    tr_o = np.zeros(len(df), bool)
    va_o = np.zeros(len(df), bool)
    tr_o[np.where(trva)[0][sp2 == 0]] = True
    va_o[np.where(trva)[0][sp2 == 1]] = True
    emit(f"held out: {holdout}")
    emit(f"train {tr_o.sum():,} (pos {y[tr_o].sum():,})   "
         f"val {va_o.sum():,} (pos {y[va_o].sum():,})   test {te_o.sum():,} (pos {y[te_o].sum():,})")
    emit("")
    emit(f"{'model':<34}{'val AUC':>10}{'test AUC':>11}{'test AUPRC':>13}")
    for name, fk, Model in MODELS:
        X = FEAT[fk]
        m, vauc = train_model(Model(X.shape[1]), X[tr_o], y[tr_o], X[va_o], y[va_o],
                              epochs=args.epochs)
        tauc, tprc, _ = evaluate(m, X[te_o], y[te_o])
        emit(f"{name:<34}{vauc:>10.3f}{tauc:>11.3f}{tprc:>13.3f}")
        results[f"orgsplit::{name}"] = dict(val=vauc, test=tauc, auprc=tprc)
    emit(f"\nBaseline AUPRC on held-out organism = {y[te_o].mean():.3f}")
    emit("")

    # ── Verdict ──────────────────────────────────────────────────────────────
    seq_models = [n for n, _, _ in MODELS if not n.startswith("CONTROL")]
    mol_auc = max(results[n]["test"] for n in seq_models)
    org_auc = max(results[f"orgsplit::{n}"]["test"] for n in seq_models)
    sh_auc = results["shuffled"]["test"]
    bind_auc = results["CONTROL binding-only (%Rank)"]["test"]

    emit("=" * 78)
    emit("VERDICT -- is this classifier usable as an independent validation axis?")
    emit("=" * 78)
    emit(f"molecule-split test AUC (best of {len(seq_models)} seq models) : {mol_auc:.3f}")
    emit(f"organism-split test AUC (best of {len(seq_models)} seq models) : {org_auc:.3f}")
    emit(f"shuffled-label control          : {sh_auc:.3f}   (must be ~0.5)")
    emit(f"binding-only control            : {bind_auc:.3f}   (expected ~0.5, cf. Track 14)")
    emit("")

    ok_shuffle = abs(sh_auc - 0.5) < 0.06
    usable = (mol_auc >= 0.65) and ok_shuffle and (org_auc >= 0.60)

    if not ok_shuffle:
        emit("FAIL: shuffled-label control is not at chance. The split leaks. Do not")
        emit("      use this classifier or trust any of the AUCs above.")
    elif mol_auc < 0.65:
        emit("FAIL: the classifier does not learn immunogenicity from sequence alone")
        emit("      (molecule-split AUC below 0.65). This is itself a publishable")
        emit("      result: within predicted MHC-II binders, 15-mer sequence carries")
        emit("      little information about whether a T-cell response occurs.")
        emit("      It ALSO means we cannot use it as the independent validation axis,")
        emit("      and the honest conclusion is that no computational axis available")
        emit("      to us can validate the de-immunisation claim. Escalate: the")
        emit("      peptide-level ELISpot becomes necessary, not optional.")
    elif org_auc < 0.60:
        emit("PARTIAL: decent molecule-split AUC but it collapses on a held-out")
        emit("      organism. Much of the signal is organism identity (amino-acid")
        emit("      composition typical of one pathogen), not immunogenicity.")
        emit("      Do NOT use it to score designed sequences: ProteinMPNN designs")
        emit("      resemble no organism in the training set, so the model is being")
        emit("      applied far out of distribution.")
    else:
        emit("PASS: the classifier generalises across source molecules AND to a")
        emit("      held-out organism, with controls at chance. It is usable as an")
        emit("      independent, experimentally-grounded validation axis.")
        emit("      NEXT: score base vs DPO designs (P1.4b).")
    emit("")
    emit("Caveat that applies regardless: designed sequences are novel and resemble")
    emit("no IEDB source organism, so any classifier is extrapolating when applied")
    emit("to them. Report that explicitly.")
    emit("=" * 78)

    with open(OUT_TXT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_TXT}")

    if usable:
        torch.save({"state_dict": best_model.state_dict(), "arch": "MLP",
                    "d": X_seq.shape[1], "results": results}, OUT_PT)
        print(f"Wrote {OUT_PT}")
    else:
        print("Classifier did not validate; no model saved (by design).")

    with open(OUT_TXT.replace(".txt", ".json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # ── figure ───────────────────────────────────────────────────────────────
    if not args.no_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        names = ["logistic regression (one-hot)", "MLP (one-hot)", "CNN (one-hot)",
                 "MLP (composition+dipeptide)", "CONTROL binding-only (%Rank)", "shuffled"]
        labels = ["logreg", "MLP", "CNN", "MLP\ncomp+dipep",
                  "binding-only\n(control)", "shuffled\n(control)"]
        ax = axes[0]
        vals = [results[n]["test"] for n in names]
        cols = ["#4C72B0"]*4 + ["#999999", "#999999"]
        ax.bar(labels, vals, color=cols)
        ax.tick_params(axis="x", labelsize=7)
        ax.axhline(0.5, color="k", ls="--", lw=1, label="chance")
        ax.axhline(0.65, color="crimson", ls=":", lw=1, label="usability bar")
        ax.set_ylim(0.3, 1.0)
        ax.set_ylabel("test AUROC")
        ax.set_title("(A)", loc="left")
        ax.legend(fontsize=8)

        ax = axes[1]
        onames = [f"orgsplit::{n}" for n in names[:5]]
        ovals = [results[n]["test"] for n in onames]
        ax.bar(labels[:5], ovals, color=cols[:5])
        ax.tick_params(axis="x", labelsize=7)
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.axhline(0.60, color="crimson", ls=":", lw=1, label="generalisation bar")
        ax.set_ylim(0.3, 1.0)
        ax.set_ylabel("test AUROC")
        ax.set_title("(B)", loc="left")
        ax.legend(fontsize=8)

        for a in axes:
            a.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT_FIG, dpi=300)
        print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", default="Bordetella pertussis Tohama I",
                    help="source organism held out entirely for SPLIT B")
    ap.add_argument("--no_figure", action="store_true")
    main(ap.parse_args())
