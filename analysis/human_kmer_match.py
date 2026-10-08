#!/usr/bin/env python
"""P1-9 / R2-W5 -- how self-like are the designs? Human-proteome 9-mer match rate.

THE CHALLENGE THIS ANSWERS. The paper's objective minimises predicted MHC class II
presentation and says nothing about tolerance. R2 pointed out that the tolerance
side is never measured despite being cited, and asked for the base-versus-fine-tuned
9-mer match rate against the human proteome.

WHY 9-MERS. Nine residues is the MHC class II binding core -- the stretch that sits
in the groove and that a T-cell receptor reads together with the MHC. A 9-mer that
occurs somewhere in the human proteome is one the immune system has had the
opportunity to become tolerant to; a 9-mer that does not occur is, in this narrow
sense, foreign. Window length is therefore not a free parameter here.

WHAT A RESULT WOULD MEAN, FIXED BEFORE RUNNING.
  * Fine-tuned match rate materially LOWER than base -> the objective is making
    designs less self-like while it lowers predicted presentation. That is a real
    tension: it would lower the number of presented windows while raising the
    chance that whatever remains is seen as foreign, and it must be reported as a
    limitation rather than folded into the headline.
  * Match rate unchanged -> the reduction is not bought at the cost of self-likeness,
    which is the more favourable outcome and the one the paper would want.
  * Fine-tuned HIGHER -> the objective is incidentally moving designs toward
    human-like sequence, which would be a (modest) point in its favour.

WHAT THIS IS NOT. Sequence identity to a human 9-mer is a crude proxy for tolerance.
Central tolerance depends on thymic expression, not on mere occurrence in the
proteome; a self 9-mer from a tissue-restricted, poorly-expressed protein confers
little tolerance. It also says nothing about whether the matched 9-mer is presented.
The comparison is nonetheless meaningful because it is PAIRED on identical backbones
and both arms are scored identically, so a difference between arms is attributable
to the objective even though the absolute level is not interpretable.

METHOD. Every 9-mer of the human SwissProt reference proteome is encoded as a
uint64 (5 bits per residue, 45 bits used) and sorted once; lookup is a binary search.
This avoids holding ~10^7 Python strings in memory. Windows containing a
non-standard residue (X, B, Z, U, O) are skipped in both the proteome and the
designs, and the count of skipped design windows is reported -- silently dropping
them would inflate the match rate.

Refuses to overwrite existing output.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

K = 9
AA = "ACDEFGHIKLMNPQRSTVWY"
CODE = {a: i + 1 for i, a in enumerate(AA)}      # 0 reserved for "invalid"


def encode_kmers(seq):
    """uint64 codes for every valid K-mer of seq. Invalid windows are dropped."""
    v = np.fromiter((CODE.get(c, 0) for c in seq), dtype=np.uint64,
                    count=len(seq))
    if len(v) < K:
        return np.empty(0, dtype=np.uint64), 0
    win = np.lib.stride_tricks.sliding_window_view(v, K)
    valid = (win != 0).all(axis=1)
    codes = np.zeros(win.shape[0], dtype=np.uint64)
    for j in range(K):
        codes = (codes << np.uint64(5)) | win[:, j]
    return codes[valid], int((~valid).sum())


def read_fasta(path):
    seq, name = [], None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    yield "".join(seq)
                name, seq = line, []
            else:
                seq.append(line.upper())
    if name is not None:
        yield "".join(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="per-design CSV with a 'sequence' column")
    ap.add_argument("--proteome", required=True, help="human proteome FASTA")
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--out_dir", default="CAPE_MPNN/data/output")
    args = ap.parse_args()

    out_csv = os.path.join(args.out_dir, f"{args.out_tag}.csv")
    out_txt = os.path.join(args.out_dir, f"{args.out_tag}_summary.txt")
    for p in (out_csv, out_txt):
        if os.path.exists(p):
            sys.exit(f"REFUSING TO OVERWRITE {p} -- use a new --out_tag.")

    print("[proteome] encoding …", flush=True)
    chunks, n_prot, n_res = [], 0, 0
    for s in read_fasta(args.proteome):
        n_prot += 1
        n_res += len(s)
        c, _ = encode_kmers(s)
        if c.size:
            chunks.append(c)
    if not chunks:
        sys.exit("FATAL: no k-mers parsed from the proteome. Wrong file?")
    human = np.unique(np.concatenate(chunks))
    print(f"[proteome] {n_prot} proteins, {n_res} residues, {human.size} unique {K}-mers",
          flush=True)
    # Sanity: a proteome this size must yield k-mers of the right order of magnitude.
    if human.size < 1e6:
        sys.exit(f"FATAL: only {human.size} unique {K}-mers -- the proteome looks truncated.")

    d = pd.read_csv(args.csv)
    for col in ("protein_name", "model", "sequence"):
        if col not in d.columns:
            sys.exit(f"FATAL: column '{col}' not in {args.csv}")

    rows, skipped_total, win_total = [], 0, 0
    for _, r in d.iterrows():
        codes, skipped = encode_kmers(str(r.sequence).upper())
        skipped_total += skipped
        win_total += codes.size + skipped
        if codes.size == 0:
            continue
        idx = np.searchsorted(human, codes)
        idx[idx >= human.size] = 0
        hit = human[idx] == codes
        rows.append({"protein_name": r.protein_name, "model": r.model,
                     "n_kmers": int(codes.size), "n_human": int(hit.sum()),
                     "pct_human": 100.0 * hit.sum() / codes.size})
    per_design = pd.DataFrame(rows)
    per_prot = per_design.groupby(["protein_name", "model"])["pct_human"].mean().unstack()
    if "base" not in per_prot or "finetuned" not in per_prot:
        sys.exit("FATAL: need both 'base' and 'finetuned' arms.")
    per_prot = per_prot.dropna()
    per_prot.to_csv(out_csv)

    b = per_prot["base"].to_numpy(float)
    f = per_prot["finetuned"].to_numpy(float)
    diff = f - b
    p_t = stats.ttest_rel(f, b).pvalue
    try:
        p_w = stats.wilcoxon(f, b).pvalue
    except ValueError:
        p_w = float("nan")
    ci = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(),
                          scale=stats.sem(diff)) if len(diff) > 1 else (float("nan"),) * 2

    L, A = [], None
    A = L.append
    A(f"HUMAN-PROTEOME {K}-MER MATCH RATE (P1-9 / R2-W5)")
    A("=" * 78)
    A(f"source CSV : {os.path.basename(args.csv)}")
    A(f"proteome   : {os.path.basename(args.proteome)} — {n_prot} proteins, "
      f"{human.size} unique {K}-mers")
    A(f"proteins   : {len(per_prot)} paired")
    A(f"windows    : {win_total} total, {skipped_total} skipped "
      f"({100.0*skipped_total/max(win_total,1):.3f}%)")
    A(f"             skipped windows span a chain separator '/' or an unknown residue 'X'.")
    A(f"             A {K}-mer crossing a chain junction is not a real peptide, so excluding")
    A(f"             it is correct here. NOTE that the 15-mer window builders used for the")
    A(f"             burden metric STRIP '/' instead, and therefore do count windows across")
    A(f"             junctions; that convention is preserved there because the published")
    A(f"             counts depend on it. The two tools differ deliberately, not by accident.")
    A("")
    A(f"  base       {b.mean():.3f}% of {K}-mers occur in the human proteome (sd {b.std(ddof=1):.3f})")
    A(f"  fine-tuned {f.mean():.3f}%                                   (sd {f.std(ddof=1):.3f})")
    A(f"  paired difference (fine-tuned - base) = {diff.mean():+.3f} pp")
    A(f"  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]   paired t p={p_t:.3g}   Wilcoxon p={p_w:.3g}")
    A(f"  designs more self-like after fine-tuning: {int((diff>0).sum())}/{len(diff)}")
    A("")
    A("READING (fixed before the numbers were computed):")
    A("  lower  -> the objective makes designs less self-like while lowering predicted")
    A("            presentation; a tension to report, not to fold into the headline.")
    A("  flat   -> the reduction is not bought at the cost of self-likeness.")
    A("  higher -> designs incidentally move toward human-like sequence.")
    A("")
    if p_t < 0.05 and diff.mean() < 0:
        A(f"  OBSERVED: LOWER by {abs(diff.mean()):.3f} pp (p={p_t:.3g}). Report as a limitation.")
    elif p_t < 0.05:
        A(f"  OBSERVED: HIGHER by {diff.mean():.3f} pp (p={p_t:.3g}).")
    else:
        A(f"  OBSERVED: no detectable change ({diff.mean():+.3f} pp, p={p_t:.3g}).")
    A("")
    A("  CAVEAT, which belongs with any use of this number: occurrence in the proteome")
    A("  is a crude tolerance proxy. Central tolerance tracks thymic expression, not")
    A("  mere presence, and this says nothing about whether a matched core is presented.")
    A("  The paired design is what makes the between-arm difference interpretable; the")
    A("  absolute level is not.")
    A("=" * 78)

    with open(out_txt, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[out] {out_csv}\n[out] {out_txt}")


if __name__ == "__main__":
    main()
