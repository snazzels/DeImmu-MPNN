#!/usr/bin/env python
"""Direct netMHCpan (MHC class I) burden validation on a design eval CSV.

Post-hoc evaluation uses the netMHC tools, never the PWMs: the PWMs are the
training signal, so a PWM-based reduction is partly self-referential. This is
the MHC-I counterpart of tools/validate_with_netmhciipan.py.

Protocol (matches the frozen validate_netmhcpan_mhc1_fullset_dualL0_k5 run):
  6 HLA-A/B/C alleles, 8/9/10-mers, presented := %Rank_EL <= 2.0,
  k sequences per model per protein (seq_idx 0..k-1), unique k-mers scored once
  per allele+length and cached, then counted back per sequence.

Usage:
  PF=<repo>/CAPE_MPNN PATH=<...>/external/programs/bin:$PATH \
    python validate_with_netmhcpan.py --in_csv <eval.csv> --out_tag <tag> [--k 5]

Writes data/output/validate_netmhcpan_mhc1_<tag>.csv and _summary.txt.
Never overwrites: refuses if either output already exists.
"""
import os, sys, csv, argparse, subprocess, tempfile, collections
import statistics as st

ALLELES = ["HLA-A02:01", "HLA-A24:02", "HLA-B07:02",
           "HLA-B39:01", "HLA-C07:01", "HLA-C16:01"]
LENGTHS = [8, 9, 10]
THRESH = 2.0
BATCH = 20000
RANK_FIELD = 12          # netMHCpan %Rank_EL is field 12 (netMHCIIpan's is 9)


def kmers(seq, L):
    seq = seq.replace("X", "").replace("/", "").replace("-", "")
    return [seq[i:i + L] for i in range(len(seq) - L + 1)]


def score(peps, allele):
    """Return {peptide: best %Rank_EL} for one allele over one length class."""
    ranks = {}
    for i in range(0, len(peps), BATCH):
        with tempfile.NamedTemporaryFile("w", suffix=".pep", delete=False) as f:
            f.write("\n".join(peps[i:i + BATCH]))
            pf = f.name
        r = subprocess.run(["netMHCpan", "-p", "-f", pf, "-a", allele],
                           capture_output=True, text=True)
        os.unlink(pf)
        if r.returncode != 0:
            sys.exit(f"netMHCpan failed for {allele}:\n{r.stderr[:2000]}")
        for line in r.stdout.splitlines():
            p = line.split()
            if len(p) <= RANK_FIELD:
                continue
            try:
                int(p[0])                      # data rows start with an index
            except ValueError:
                continue
            pep = p[2]
            try:
                rank = float(p[RANK_FIELD])
            except ValueError:
                continue
            if pep not in ranks or rank < ranks[pep]:
                ranks[pep] = rank
    return ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--k", type=int, default=5,
                    help="sequences per model per protein (seq_idx 0..k-1)")
    args = ap.parse_args()

    # PF is derived from __file__, NOT from os.environ — this is the project-wide tool
    # convention. It used to read os.environ["PF"], which cost four tasks
    # of job 517739 roughly six hours of netMHCpan scoring EACH on 2026-08-24: the sbatch
    # exported PF as the project ROOT while this tool's own docstring documents
    # PF=<repo>/CAPE_MPNN, so every score was computed and then thrown away at the final
    # write with FileNotFoundError on <root>/data/output/. Deriving it from __file__ makes
    # the variable impossible to get wrong and is identical to the documented value.
    PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    od = os.path.join(PF, "data", "output")
    out_csv = os.path.join(od, f"validate_netmhcpan_mhc1_{args.out_tag}.csv")
    out_sum = os.path.join(od, f"validate_netmhcpan_mhc1_{args.out_tag}_summary.txt")
    # FAIL FAST. Everything below this point is cheap; the netMHCpan loop that follows is
    # hours. Verify the destination is real and writable BEFORE spending any of it.
    if not os.path.isdir(od):
        sys.exit(f"Output directory does not exist: {od}\n"
                 "  Refusing to start hours of netMHCpan scoring that could not be saved.")
    if not os.access(od, os.W_OK):
        sys.exit(f"Output directory is not writable: {od}")
    for p in (out_csv, out_sum):
        if os.path.exists(p):
            sys.exit(f"Refusing to overwrite existing {p}; pass a distinct --out_tag.")

    rows = [r for r in csv.DictReader(open(args.in_csv))
            if int(r["seq_idx"]) < args.k]
    if not rows:
        sys.exit(f"No rows with seq_idx < {args.k} in {args.in_csv}")
    models = sorted({r["model"] for r in rows})
    if set(models) != {"base", "finetuned"}:
        sys.exit(f"Expected base/finetuned models, found {models}")

    uniq = {L: set() for L in LENGTHS}
    for r in rows:
        for L in LENGTHS:
            uniq[L].update(kmers(r["sequence"], L))
    n_uniq = sum(len(v) for v in uniq.values())
    print(f"{len(rows)} sequences ({args.k}/model/protein); "
          f"{n_uniq} unique k-mers; one netMHCpan pass per allele+length",
          flush=True)

    rankmap = {a: {} for a in ALLELES}
    for L in LENGTHS:
        peps = sorted(uniq[L])
        for a in ALLELES:
            rankmap[a].update(score(peps, a))
            print(f"  scored L={L} allele={a}: {len(peps)} peptides", flush=True)

    def presented(seq):
        tot = 0
        for a in ALLELES:
            rm = rankmap[a]
            for L in LENGTHS:
                tot += sum(1 for k in kmers(seq, L) if rm.get(k, 999) <= THRESH)
        return tot

    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["protein_name", "model", "seq_idx", "netmhc_total", "sequence"])
        per = collections.defaultdict(lambda: {"base": [], "finetuned": []})
        for r in rows:
            n = presented(r["sequence"])
            per[r["protein_name"]][r["model"]].append(n)
            w.writerow([r["protein_name"], r["model"], r["seq_idx"], n, r["sequence"]])

    reds, bases, fts = [], [], []
    for pid in sorted(per):
        b, c = st.mean(per[pid]["base"]), st.mean(per[pid]["finetuned"])
        bases.append(b)
        fts.append(c)
        if b:
            reds.append(100 * (b - c) / b)

    L = [f"NETMHCpan (MHC-I) DIRECT — {args.out_tag}",
         f"in_csv: {args.in_csv}",
         f"Proteins: {len(per)}  seqs/model/protein: {args.k}  unique kmers: {n_uniq}",
         f"Alleles: {'+'.join(ALLELES)}  lengths: {LENGTHS}  threshold: %Rank_EL <= {THRESH}",
         f"Base per-protein mean:      {st.mean(bases):.2f}",
         f"Fine-tuned per-protein mean:{st.mean(fts):.2f}",
         f"Per-protein reduction:      {st.mean(reds):.1f}% +/- {st.stdev(reds):.1f}%"
         f"  (median {st.median(reds):.1f}%)",
         f"Improved: {sum(x > 0 for x in reds)}/{len(reds)}"]
    try:
        from scipy import stats
        t, p = stats.ttest_rel(bases, fts)
        L.append(f"Paired t={t:.3f} p={p:.4g}")
        w_, pw = stats.wilcoxon(bases, fts)
        L.append(f"Wilcoxon W={w_:.1f} p={pw:.4g}")
    except Exception as e:                      # scipy optional
        L.append(f"(paired tests skipped: {e})")

    L += ["", f"{'protein':12}{'base_mean':>10}{'ft_mean':>10}{'reduction%':>12}"]
    for pid in sorted(per):
        b, c = st.mean(per[pid]["base"]), st.mean(per[pid]["finetuned"])
        L.append(f"{pid:12}{b:>10.2f}{c:>10.2f}{(100*(b-c)/b if b else 0):>+12.1f}")

    open(out_sum, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:9]))
    print(f"\nWrote {out_csv}\nWrote {out_sum}")


if __name__ == "__main__":
    main()
