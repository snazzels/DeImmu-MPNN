#!/usr/bin/env python
"""What predicted MHC-II burden does an ordinary human protein carry?

THE QUESTION THIS ANSWERS (raised by the collaborating immunologist, 2026-09-28).
The manuscript reports a 41.4% reduction in predicted presented windows relative to
base ProteinMPNN. A reduction is relative and says nothing about the level reached.
The natural reference the question proposes: take human proteins -- sequences the
immune system demonstrably tolerates -- score them on netMHCIIpan, and see where the
designs sit against them.

WHY THIS IS NOT THE SAME AS THE BIOLOGICS REFERENCE POINT already in this project.
`biologics_reference_point.py` scored 12 approved therapeutic proteins on ProPred via
Rosetta's mhc_epitope. That set is small, hand-picked, and sits on the conservative
secondary instrument. This tool scores a large random draw from the human proteome on
netMHCIIpan -- the manuscript's PRIMARY axis -- so the numbers can be read straight
against the paper's own base and fine-tuned arms without a scorer conversion.

PROTOCOL IS THE FROZEN ONE, COPIED FROM validate_with_netmhciipan.py AND NOT RECHOSEN
HERE: 15-mer sliding windows, the six-allele DRB1 panel, presented := %Rank_EL <= 2.0,
`netmhc_total` = presented windows summed over the six alleles. Any deviation would put
this on a different basis from every number in the paper, which is the failure mode the
project's cross-run rule exists to prevent.

TWO STRATA, BECAUSE RAW COUNTS SCALE WITH LENGTH.
  length_matched -- drawn to match the design set's own length distribution (its length
                    quartiles, equal numbers per bin, range 78-496 aa). Comparable to the
                    paper on BOTH the raw per-protein count and the density, and this is
                    the primary stratum.
  proteome_wide  -- drawn at random from the whole proteome with no length filter beyond
                    a compute cap. Answers the question as literally asked. Only the
                    length-normalised quantities are interpretable here, because the human
                    length distribution has a long tail the designs do not have.

METRICS. `netmhc_total` is the paper's quantity. `per_100_residues` (= 100 * total /
length) and `epitope_fraction` (= total / (6 * n_windows), i.e. the fraction of
allele-window pairs predicted presented) are the scale-free forms; the latter is the
same definition the biologics tool used, so the two reference points can be placed on
one axis despite using different scorers.

THE READING, FIXED BEFORE THE NUMBERS WERE COMPUTED.
  Human proteins BELOW the de-immunised designs -> the designs remain more
    presentation-rich than self, the reduction is a step and not an arrival, and the
    paper should say so.
  Human proteins OVERLAPPING the de-immunised designs -> the reduction reaches the level
    that tolerated self-proteins occupy on this predictor.
  Human proteins ABOVE the de-immunised designs -> predicted presentation burden does
    not separate tolerated from untolerated sequence, and the metric's meaning is bounded
    by that. This would NOT invalidate the reduction, which is a matched base-vs-fine-tuned
    contrast on identical backbones; it would bound what a low burden can be claimed to
    promise. The existing biologics table already points this way: its ten human-sequence
    products had a HIGHER median burden (0.168) than the de-immunised designs (0.124).

THE INTERPRETIVE LIMIT, STATED UP FRONT RATHER THAN AS A CAVEAT AFTERWARDS. Human
proteins are not tolerated because they present few peptides. They present plenty; they
are tolerated because thymic negative selection and peripheral tolerance removed the
T cells that would respond. No presentation predictor represents that mechanism. So this
is a reference SCALE -- a ruler, not an experiment -- and the number it yields is an
upper bound on what predicted burden can tell us, not a design target. The controlled
comparison in this work remains base-vs-fine-tuned on identical backbones.

A FURTHER LIMIT SPECIFIC TO THIS COMPARISON. The two populations differ in composition,
length, fold class and origin at once. Length is the one confounder handled directly,
by the length_matched stratum; the others are not, and no difference here is attributable
to any single cause.

Outputs (refuses to overwrite):
  human_proteome_reference_<tag>.csv          per-protein table, both strata
  human_proteome_reference_<tag>_summary.txt  comparison against the design arms
  human_proteome_reference_<tag>_proteins.txt the drawn accessions, for reproducibility
"""
import os, sys, csv, argparse, subprocess, tempfile, hashlib, random, collections
import statistics as st
from multiprocessing import Pool

PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OD = os.path.join(PF, "data", "output")
FASTA = os.path.join(PF, "data", "input", "immuno", "mhc_2", "iedb", "cache",
                     "human_swissprot.fasta")
DESIGN_CSV = os.path.join(OD, "validate_netmhciipan_sweep_f16b51e6.csv")

# ── frozen protocol — identical to validate_with_netmhciipan.py ──────────────
ALLELES     = ["DRB1_0101", "DRB1_0301", "DRB1_0401", "DRB1_0701",
               "DRB1_1101", "DRB1_1501"]
THRESHOLD   = 2.0     # %Rank_EL <= 2% == "presented"
PEPTIDE_LEN = 15
BATCH_SIZE  = 5000

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")
# The two proteins the guard excludes from the design arm, so the design-side
# reference numbers recomputed here match the published guarded figures exactly.
DESIGN_EXCLUDE = {"4b6d_C", "7m10_A"}


# ── netMHCIIpan ──────────────────────────────────────────────────────────────

def _parse(lines):
    """Parse netMHCIIpan stdout -> {peptide: best %Rank_EL}."""
    ranks = {}
    for line in lines:
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            int(parts[0])
        except ValueError:
            continue
        pep, rank_el = parts[2], float(parts[9])
        if pep not in ranks or rank_el < ranks[pep]:
            ranks[pep] = rank_el
    return ranks


def score_allele(job):
    """Score every peptide against one allele. Returns (allele, {pep: rank})."""
    allele, peptides = job
    ranks = {}
    for i in range(0, len(peptides), BATCH_SIZE):
        batch = peptides[i:i + BATCH_SIZE]
        with tempfile.TemporaryDirectory() as td:
            pf = os.path.join(td, "in.pep")
            with open(pf, "w") as fh:
                fh.write("\n".join(batch))
            res = subprocess.run(["netMHCIIpan", "-inptype", "1", "-f", pf,
                                  "-a", allele],
                                 capture_output=True, check=True)
        ranks.update(_parse(res.stdout.decode().splitlines()))
    return allele, ranks


def windows(seq):
    return [seq[i:i + PEPTIDE_LEN] for i in range(len(seq) - PEPTIDE_LEN + 1)]


# ── proteome sampling ────────────────────────────────────────────────────────

def read_fasta(path):
    """Yield (accession, gene, sequence) for standard-residue entries."""
    acc = seq = gene = None
    out = []

    def flush():
        if acc and seq and set(seq) <= STD_AA and len(seq) >= PEPTIDE_LEN:
            out.append((acc, gene, seq))

    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            flush()
            parts = line[1:].split("|")
            acc = parts[1] if len(parts) > 2 else line[1:].split()[0]
            gene = "?"
            for tok in line.split():
                if tok.startswith("GN="):
                    gene = tok[3:]
            seq = ""
        elif acc is not None:
            seq += line
    flush()
    return out


def design_lengths():
    """Length distribution of the pinned design set (guarded, n=98)."""
    L = {}
    for r in csv.DictReader(open(DESIGN_CSV)):
        if r["protein_name"] not in DESIGN_EXCLUDE:
            L[r["protein_name"]] = int(r["protein_length"])
    return sorted(L.values())


def draw_length_matched(pool, lens, n, rng):
    """Stratified draw: equal numbers from each design-length quartile bin."""
    q = st.quantiles(lens, n=4)
    edges = [min(lens), q[0], q[1], q[2], max(lens)]
    per_bin = n // 4
    picked = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        cand = [p for p in pool if lo <= len(p[2]) <= hi]
        rng.shuffle(cand)
        picked += cand[:per_bin]
    return picked


def draw_proteome_wide(pool, n, rng, cap):
    cand = [p for p in pool if len(p[2]) <= cap]
    rng.shuffle(cand)
    return cand[:n], len(pool) - len(cand)


# ── design-side reference numbers, recomputed from the published CSV ─────────

def design_reference():
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    L = {}
    for r in csv.DictReader(open(DESIGN_CSV)):
        if r["protein_name"] in DESIGN_EXCLUDE:
            continue
        acc[r["protein_name"]][r["model"]].append(float(r["netmhc_total"]))
        L[r["protein_name"]] = int(r["protein_length"])
    rows = {"base": [], "finetuned": []}
    for name, m in acc.items():
        if "base" in m and "finetuned" in m:
            for arm in ("base", "finetuned"):
                tot = st.mean(m[arm])
                ln = L[name]
                rows[arm].append((tot, 100.0 * tot / ln,
                                  tot / (6.0 * (ln - PEPTIDE_LEN + 1))))
    return rows


def describe(vals):
    if not vals:
        return dict(n=0)
    qs = st.quantiles(vals, n=4) if len(vals) > 3 else [float("nan")] * 3
    return dict(n=len(vals), mean=st.mean(vals), median=st.median(vals),
                q1=qs[0], q3=qs[2], lo=min(vals), hi=max(vals))


def fmt(d, w=8, p=2):
    if d["n"] == 0:
        return "n=0"
    return (f"n={d['n']:<4d} mean {d['mean']:{w}.{p}f}  median {d['median']:{w}.{p}f}"
            f"  IQR {d['q1']:.{p}f}-{d['q3']:.{p}f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--n_matched", type=int, default=400)
    ap.add_argument("--n_wide", type=int, default=200)
    ap.add_argument("--length_cap", type=int, default=3000,
                    help="proteome_wide compute cap; longer proteins are not drawn")
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()

    out_csv = os.path.join(OD, f"human_proteome_reference_{args.tag}.csv")
    out_txt = os.path.join(OD, f"human_proteome_reference_{args.tag}_summary.txt")
    out_set = os.path.join(OD, f"human_proteome_reference_{args.tag}_proteins.txt")
    for p in (out_csv, out_txt, out_set):
        if os.path.exists(p):
            sys.exit(f"REFUSING TO OVERWRITE existing output: {p}")

    rng = random.Random(args.seed)
    print(f"reading {FASTA}")
    pool = read_fasta(FASTA)
    print(f"  {len(pool)} standard-residue human entries")

    lens = design_lengths()
    print(f"  design set n={len(lens)}, length {min(lens)}-{max(lens)}, "
          f"median {st.median(lens):.0f}")

    matched = draw_length_matched(pool, lens, args.n_matched, rng)
    wide, skipped = draw_proteome_wide(pool, args.n_wide, rng, args.length_cap)
    # A protein drawn into both strata is scored once and reported in both.
    entries = {}
    for p in matched:
        entries[p[0]] = [p, {"length_matched"}]
    for p in wide:
        entries.setdefault(p[0], [p, set()])[1].add("proteome_wide")
    print(f"  length_matched {len(matched)}, proteome_wide {len(wide)} "
          f"(>{args.length_cap} aa not drawn: {skipped}), unique {len(entries)}")

    md5 = hashlib.md5(",".join(sorted(entries)).encode()).hexdigest()[:12]
    print(f"  protein_set_md5 {md5}")

    peptides = sorted({w for p, _ in entries.values() for w in windows(p[2])})
    print(f"  {len(peptides)} unique 15-mers x {len(ALLELES)} alleles "
          f"= {len(peptides)*len(ALLELES)} scores")

    with Pool(min(args.procs, len(ALLELES))) as pool_:
        results = pool_.map(score_allele, [(a, peptides) for a in ALLELES])
    ranks = dict(results)
    for a in ALLELES:
        got = len(ranks[a])
        nz = sum(1 for v in ranks[a].values() if v > 0)
        print(f"  {a}: {got} scored, {nz} non-zero rank")
        if nz == 0:
            sys.exit(f"ABORT: all-zero %Rank for {a} — TMPDIR overflow or tcsh missing.")

    rows = []
    for a_, (p, strata) in sorted(entries.items()):
        acc, gene, seq = p
        wl = windows(seq)
        per = {a: sum(1 for w in wl if ranks[a].get(w, 999) <= THRESHOLD)
               for a in ALLELES}
        tot = sum(per.values())
        rows.append(dict(accession=acc, gene=gene, length=len(seq),
                         n_windows=len(wl),
                         stratum="+".join(sorted(strata)),
                         netmhc_total=tot,
                         per_100_residues=100.0 * tot / len(seq),
                         epitope_fraction=tot / (len(ALLELES) * len(wl)),
                         **{f"netmhc_{a}": per[a] for a in ALLELES}))

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out_set, "w") as fh:
        fh.write(f"# protein_set_md5 {md5}\n# seed {args.seed}\n")
        for r in rows:
            fh.write(f"{r['accession']}\t{r['gene']}\t{r['length']}\t{r['stratum']}\n")

    # ── summary ──────────────────────────────────────────────────────────────
    dref = design_reference()
    lm = [r for r in rows if "length_matched" in r["stratum"]]
    pw = [r for r in rows if "proteome_wide" in r["stratum"]]

    L = []
    A = L.append
    A("=" * 88)
    A("HUMAN-PROTEOME REFERENCE POINT — netMHCIIpan, the manuscript's primary axis")
    A("=" * 88)
    A(f"scorer    : netMHCIIpan, {len(ALLELES)} DRB1 alleles, 15-mer windows, "
      f"presented := %Rank_EL <= {THRESHOLD}")
    A(f"source    : {os.path.basename(FASTA)} ({len(pool)} standard-residue entries)")
    A(f"designs   : {os.path.basename(DESIGN_CSV)} (guarded n=98, the published arm)")
    A(f"sample    : seed {args.seed}, protein_set_md5 {md5}")
    A(f"metric    : netmhc_total = presented windows summed over the {len(ALLELES)} alleles")
    A("")
    A("PER-PROTEIN PREDICTED BURDEN")
    A("")
    A(f"  {'population':<34}{'raw count':<46}")
    A("  " + "-" * 84)
    for lab, vals in (
            ("human, length-matched", [r["netmhc_total"] for r in lm]),
            ("human, proteome-wide", [r["netmhc_total"] for r in pw]),
            ("base ProteinMPNN", [v[0] for v in dref["base"]]),
            ("de-immunised (f16b51e6)", [v[0] for v in dref["finetuned"]])):
        A(f"  {lab:<34}{fmt(describe(vals))}")
    A("")
    A(f"  {'population':<34}{'presented windows per 100 residues':<46}")
    A("  " + "-" * 84)
    for lab, vals in (
            ("human, length-matched", [r["per_100_residues"] for r in lm]),
            ("human, proteome-wide", [r["per_100_residues"] for r in pw]),
            ("base ProteinMPNN", [v[1] for v in dref["base"]]),
            ("de-immunised (f16b51e6)", [v[1] for v in dref["finetuned"]])):
        A(f"  {lab:<34}{fmt(describe(vals))}")
    A("")
    A(f"  {'population':<34}{'epitope_fraction (allele-window pairs)':<46}")
    A("  " + "-" * 84)
    for lab, vals in (
            ("human, length-matched", [r["epitope_fraction"] for r in lm]),
            ("human, proteome-wide", [r["epitope_fraction"] for r in pw]),
            ("base ProteinMPNN", [v[2] for v in dref["base"]]),
            ("de-immunised (f16b51e6)", [v[2] for v in dref["finetuned"]])):
        A(f"  {lab:<34}{fmt(describe(vals), w=8, p=4)}")
    A("")
    A("LENGTH CHECK — the one confounder this design controls")
    A(f"  design set        : median {st.median(lens):.0f} aa, range {min(lens)}-{max(lens)}")
    A(f"  human length-matched: median {st.median([r['length'] for r in lm]):.0f} aa, "
      f"range {min(r['length'] for r in lm)}-{max(r['length'] for r in lm)}")
    A(f"  human proteome-wide : median {st.median([r['length'] for r in pw]):.0f} aa, "
      f"range {min(r['length'] for r in pw)}-{max(r['length'] for r in pw)}")
    A("")
    A("WHERE THE DESIGNS SIT (density basis, length-matched human stratum)")
    hd = sorted(r["per_100_residues"] for r in lm)
    for lab, v in (("base ProteinMPNN", st.median([x[1] for x in dref["base"]])),
                   ("de-immunised", st.median([x[1] for x in dref["finetuned"]]))):
        pct = 100.0 * sum(1 for x in hd if x < v) / len(hd)
        A(f"  {lab:<24} median {v:5.2f} per 100 aa — above {pct:.0f}% of human proteins")
    A("")
    A("READING (fixed in the module docstring BEFORE these numbers were computed)")
    dm = st.median([x[1] for x in dref["finetuned"]])
    hm = st.median(hd)
    if hm < dm:
        A("  OBSERVED: human proteins carry a LOWER median density than the de-immunised")
        A("  designs. The reduction is a step toward, not to, the level of self-protein.")
    else:
        A("  OBSERVED: human proteins carry a HIGHER median density than the de-immunised")
        A("  designs. Predicted class II presentation therefore does not by itself")
        A("  separate tolerated sequence from untolerated sequence — tolerated self")
        A("  proteins are presentation-rich. This bounds what a burden reduction can")
        A("  promise; it does not touch the matched base-vs-fine-tuned contrast, which")
        A("  is a paired comparison on identical backbones and is unaffected.")
    A("")
    A("LIMITS ON WHAT THIS CAN SUPPORT")
    A("  1. Human proteins are tolerated by thymic negative selection and peripheral")
    A("     tolerance, not by presenting few peptides. No predictor used here")
    A("     represents that mechanism, so this is a reference SCALE, not a target.")
    A("  2. Two unrelated populations: composition, fold class and origin differ at")
    A("     once. Length is controlled by the length_matched stratum; nothing else is.")
    A("  3. The designs are scored as k=5 sequences per backbone averaged to a")
    A("     per-protein mean; each human protein contributes one natural sequence.")
    A("     The human spread is therefore between-protein variance only, while the")
    A("     design spread has had within-backbone variance averaged out of it.")
    A("  4. SwissProt canonical sequences include signal and propeptides that are not")
    A("     present in the mature circulating protein, unlike the biologics reference")
    A("     set, which was sliced to mature chains.")
    A("=" * 88)

    with open(out_txt, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {out_csv}\nwrote {out_txt}\nwrote {out_set}")


if __name__ == "__main__":
    main()
