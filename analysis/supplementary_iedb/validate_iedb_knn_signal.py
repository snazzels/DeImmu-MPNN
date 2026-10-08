#!/usr/bin/env python
"""Validate a non-classifier (k-NN/similarity) IEDB immunogenicity signal.

Phase 0 of the three-arm ablation plan.

WHY THIS SCRIPT EXISTS
-----------------------
train_iedb_classifier.py (Track 15) trained a classifier to distinguish
immunogenic vs non-immunogenic MHC-II binders directly from experimental IEDB
outcomes. It reached AUC 0.66 on an in-distribution (source-molecule-grouped)
split but collapsed to ~0.52 (chance) under a held-out-*organism* split --
meaning it had learned organism identity ("pertussis-like composition"), not
immunogenicity. All four architectures tried (including a register-invariant
composition+dipeptide model) showed the same collapse, so the confound lives
in the data, not the model class.

Before spending any GPU time wiring a new "IEDB-only" DPO reward arm, this
script checks whether a much simpler, non-parametric k-NN/similarity signal
in the SAME feature space survives the SAME held-out-organism test. If it
also collapses to chance, that is a second, independent confirmation of the
Track 15 finding (no model class rescues this), not a wasted effort -- and
the training/wiring/GPU work described in the ablation plan should not
proceed.

FEATURE SPACE
-------------
Reuses composition() from train_iedb_classifier.py verbatim: register-
invariant 420-dim (20 AA-frequency + 400 dipeptide-frequency) features. Do
not reimplement this -- it is imported directly from that script.

SIGNAL DEFINITION
------------------
For a query peptide, score = mean_dist(k nearest non-immunogenic refs)
                            - mean_dist(k nearest immunogenic refs)
in composition() space (cosine distance). Higher score = more IEDB-favorable
(farther from known-immunogenic, closer to known-non-immunogenic) --
consistent with the existing additive reward convention in
preference_pair.py (higher preference = better).

SHORTCUT MITIGATION (applied up front, not discovered after failure)
----------------------------------------------------------------------
The held-out organism is entirely excluded from both reference pools (a
stronger and simpler version of "leave-one-organism-out", since only one
organism is held out per split here). On top of that, each reference pool is
capped at --organism_cap peptides per remaining source_organism before
neighbor search, so the two dominant pathogens (B. pertussis, M.
tuberculosis) cannot swamp the neighbor vote for peptides from other,
smaller organisms. The script reports AUC both WITH and WITHOUT this cap so
the effect of the mitigation itself is visible, not assumed.

VALIDATION BAR
--------------
Same two organism holdouts as Track 15 (B. pertussis, M. tuberculosis), same
auroc() metric, so the numbers below are directly comparable to the Track 15
numbers:
    held-out organism AUC (Track 15 classifiers): 0.460-0.522 (chance)
Go: AUC materially above 0.6 on both holdouts -> proceed to Phase 1 (wire
    into cape-mpnn.py / preference_pair.py per the ablation plan).
No-go: AUC stays near 0.5 -> stop; document as a second confirmation that no
    usable non-circular IEDB signal exists at the peptide level.

Outputs
-------
data/output/iedb_knn_signal_summary.txt
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

PF = os.environ.get("PF", os.path.join(os.path.dirname(__file__), ".."))
IEDB = os.path.join(PF, "data", "input", "immuno", "mhc_2", "iedb")
OUT_TXT = os.path.join(PF, "data", "output", "iedb_knn_signal_summary.txt")

sys.path.insert(0, os.path.dirname(__file__))
from train_iedb_classifier import composition, auroc, group_split  # noqa: E402

HOLDOUTS = ["Bordetella pertussis Tohama I", "Mycobacterium tuberculosis"]


def cap_per_organism(idx, organism, cap, rng):
    """Return a subset of idx with at most `cap` entries per organism value."""
    if cap is None:
        return idx
    df = pd.DataFrame({"idx": idx, "org": organism[idx]})
    out = []
    for _, group in df.groupby("org"):
        if len(group) <= cap:
            out.append(group["idx"].to_numpy())
        else:
            out.append(rng.choice(group["idx"].to_numpy(), size=cap, replace=False))
    return np.concatenate(out)


def knn_score(X_query, X_pos_ref, X_neg_ref, k):
    """score = mean dist to k nearest neg refs - mean dist to k nearest pos refs.

    Higher = farther from immunogenic refs, closer to non-immunogenic refs
    (IEDB-favorable). Cosine distance; robust to raw count-magnitude
    differences in the composition() feature vectors.
    """
    d_pos = cdist(X_query, X_pos_ref, metric="cosine")
    d_neg = cdist(X_query, X_neg_ref, metric="cosine")
    k_pos = min(k, X_pos_ref.shape[0])
    k_neg = min(k, X_neg_ref.shape[0])
    mean_d_pos = np.sort(d_pos, axis=1)[:, :k_pos].mean(axis=1)
    mean_d_neg = np.sort(d_neg, axis=1)[:, :k_neg].mean(axis=1)
    return mean_d_neg - mean_d_pos


def evaluate_holdout(df, X, holdout, k, organism_cap, seed):
    rng = np.random.default_rng(seed)
    is_test_org = (df.source_organism == holdout).to_numpy()
    ref_pool = np.where(~is_test_org)[0]
    query_idx = np.where(is_test_org)[0]

    y = df.y.to_numpy()
    organism = df.source_organism.to_numpy()

    pos_ref_all = ref_pool[y[ref_pool] == 1]
    neg_ref_all = ref_pool[y[ref_pool] == 0]

    pos_ref_capped = cap_per_organism(pos_ref_all, organism, organism_cap, rng)
    neg_ref_capped = cap_per_organism(neg_ref_all, organism, organism_cap, rng)

    s_uncapped = knn_score(X[query_idx], X[pos_ref_all], X[neg_ref_all], k)
    s_capped = knn_score(X[query_idx], X[pos_ref_capped], X[neg_ref_capped], k)

    y_test = y[query_idx]
    return dict(
        n_test=len(query_idx),
        n_pos_test=int(y_test.sum()),
        n_pos_ref=len(pos_ref_all),
        n_neg_ref=len(neg_ref_all),
        n_pos_ref_capped=len(pos_ref_capped),
        n_neg_ref_capped=len(neg_ref_capped),
        auc_uncapped=auroc(y_test, s_uncapped),
        auc_capped=auroc(y_test, s_capped),
    )


def shuffled_control(df, X, holdout, k, organism_cap, seed):
    """Same protocol with shuffled labels -- must land at ~0.5 or the split leaks."""
    rng = np.random.default_rng(seed)
    df_sh = df.copy()
    is_test_org = (df.source_organism == holdout).to_numpy()
    ref_idx = np.where(~is_test_org)[0]
    y_sh = df.y.to_numpy().copy()
    y_sh[ref_idx] = rng.permutation(y_sh[ref_idx])
    df_sh["y"] = y_sh
    return evaluate_holdout(df_sh, X, holdout, k, organism_cap, seed)


def main(args):
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    ds = pd.read_csv(os.path.join(IEDB, "binder_matched_dataset.csv"))
    ann = pd.read_csv(os.path.join(IEDB, "peptide_source_annotation.csv"))
    df = ds.merge(ann, on="peptide", how="left")
    df["source_organism"] = df.source_organism.fillna("UNKNOWN")
    df["y"] = (df.label == "immunogenic").astype(int)

    emit("=" * 78)
    emit("IEDB k-NN/SIMILARITY SIGNAL VALIDATION (roadmap #19, ablation Phase 0)")
    emit("=" * 78)
    emit(f"Peptides: {len(df):,}   immunogenic: {df.y.sum():,}   "
         f"non-immunogenic: {(1 - df.y).sum():,}")
    emit(f"Feature space: composition() [20 AA-freq + 400 dipeptide-freq, "
         f"register-invariant] -- reused from train_iedb_classifier.py")
    emit(f"k = {args.k} neighbors, cosine distance, organism_cap = {args.organism_cap}")
    emit("")
    emit("Reference (Track 15, same auroc() metric, held-out-organism split):")
    emit("  best classifier test AUC -- B. pertussis holdout : 0.522")
    emit("  best classifier test AUC -- M. tuberculosis holdout: 0.501 (range 0.460-0.501)")
    emit("  (chance; this is the bar the k-NN signal below must clear)")
    emit("")

    X = composition(df.peptide.tolist())

    results = {}
    for holdout in HOLDOUTS:
        emit("-" * 78)
        emit(f"HELD-OUT ORGANISM: {holdout}")
        emit("-" * 78)
        r = evaluate_holdout(df, X, holdout, args.k, args.organism_cap, args.seed)
        emit(f"test n={r['n_test']:,} (pos {r['n_pos_test']:,})   "
             f"ref pool: {r['n_pos_ref']:,} immunogenic / {r['n_neg_ref']:,} non-immunogenic")
        emit(f"  after organism cap: {r['n_pos_ref_capped']:,} immunogenic / "
             f"{r['n_neg_ref_capped']:,} non-immunogenic refs")
        emit(f"  test AUC, uncapped refs : {r['auc_uncapped']:.3f}")
        emit(f"  test AUC, organism-capped refs : {r['auc_capped']:.3f}")

        rs = shuffled_control(df, X, holdout, args.k, args.organism_cap, args.seed)
        emit(f"  CONTROL shuffled labels, capped refs : {rs['auc_capped']:.3f}  (must be ~0.5)")
        emit("")
        results[holdout] = r
        results[f"{holdout}::shuffled"] = rs

    emit("=" * 78)
    emit("VERDICT")
    emit("=" * 78)
    best_auc = max(results[h]["auc_capped"] for h in HOLDOUTS)
    emit(f"Best held-out-organism AUC across both holdouts (capped refs): {best_auc:.3f}")
    if best_auc > 0.6:
        emit("GO: materially above chance on both organism holdouts. Proceed to")
        emit("Phase 1 (wire IEDB_SIM reward into cape-mpnn.py / preference_pair.py),")
        emit("but retain the same skepticism Track 15 taught us -- recheck for a")
        emit("subtler shortcut before trusting this at face value.")
    else:
        emit("NO-GO: stays near chance, same failure mode as the Track 15 classifier.")
        emit("This is a second, independent (non-parametric) confirmation that no")
        emit("usable non-circular IEDB signal exists at the peptide level with this")
        emit("data. Do not proceed to Phase 1 / GPU training without new direction.")
    emit("")

    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    emit(f"Wrote {OUT_TXT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=25, help="number of nearest neighbors")
    ap.add_argument("--organism_cap", type=int, default=200,
                     help="max reference peptides per source_organism (mitigation)")
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
