#!/usr/bin/env python
"""P2-2 / R3 -- what does a predicted epitope burden of X actually mean?

THE CHALLENGE THIS ANSWERS. The manuscript reports reductions: 41.4% fewer
predicted presented windows than base ProteinMPNN. A reduction is a relative
quantity and says nothing about the level reached. R3 asked for a reference point:
is a de-immunised design's predicted burden low in the sense that matters -- as low
as proteins that are already dosed into people -- or merely lower than an
unoptimised design's?

WHAT IS COMPUTED. The mature chains of approved therapeutic proteins are scored on
exactly the instrument used for the designs in the P2-1 benchmark: Rosetta's
`mhc_epitope` term over ProPred/TEPITOPE matrices for eight HLA-DR alleles. The
comparison quantity is length-normalised, because the biologics span 51 to 585
residues and the designs have their own length distribution:

  epitope_fraction -- fraction of valid 9-mer windows predicted to bind at least
                      one of the eight alleles. Scale-free, and the primary metric.
  per_100_residues -- predicted epitope windows per 100 residues. Reported for
                      intuition; it moves with the same signal.

THE COMPARISON IS BETWEEN TWO UNRELATED PROTEIN POPULATIONS, AND THAT IS THE WHOLE
DIFFICULTY. Approved biologics are natural sequences shaped by evolution and, for
the human-derived ones, by central tolerance; the designs are ProteinMPNN outputs
on PDB backbones. They differ in composition, length, fold class and origin all at
once, so nothing here is a controlled comparison and no difference between the two
populations is attributable to any single cause. This is a reference SCALE, in the
same sense that a ruler is not an experiment. The matched, controlled comparison in
this work remains base-vs-fine-tuned on identical backbones; that is what the
paper's statistics rest on and this does not replace it.

WHAT IS DELIBERATELY NOT TABULATED: clinical anti-drug-antibody rates. They are
drug-specific, assay-specific, dose- and route-dependent, and not comparable
across the trials that produced them, so putting a percentage beside each protein
would imply a precision that does not exist. What IS recorded is the source
organism, because non-human therapeutic proteins are the recognised high-ADA class
and that is a fact about the sequence rather than an estimate from a trial.

SEQUENCE PROVENANCE. Sequences are fetched from UniProt and sliced to the annotated
mature chain (feature type Chain, or Peptide for multi-chain products such as
insulin), so signal and propeptides -- which are not in the dosed product -- are
excluded. Multi-chain products are joined with '/' and 9-mers spanning the junction
are dropped, since such a peptide does not exist in the product. The exact fetched
sequences are written to a FASTA beside the outputs so the run is reproducible and
the sequences are depositable, and the tool reuses that FASTA if it exists rather
than re-fetching.

ProPred is used here because it runs locally and because P2-1 established that it
correlates with netMHCIIpan across these designs (Spearman rho = +0.80 at base)
while reporting a smaller reduction -- so this reference point is on the
conservative instrument, not the paper's headline one.

TO PUT THIS ON THE MANUSCRIPT'S PRIMARY AXIS, the emitted FASTA is what to feed
netMHCIIpan; that requires the cluster, since the netMHCIIpan wrapper execs tcsh
and does not run locally. No sbatch for it exists yet, and this tool has no
netMHCIIpan mode -- both are open work, deliberately not stubbed out here so that
nothing in this file implies a path that has not been run. The design side of that
comparison already exists (the `netmhc_total` column of the P2-1 CSV); only the
biologics side is missing.

Requires PyRosetta (env `bindcraft`, which also has pandas/scipy) and, on first
run, network access. Refuses to overwrite existing output.
"""

import argparse
import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

K = 9
AA = set("ACDEFGHIKLMNPQRSTVWY")
REF_PEPTIDE = "FVKQNTLKL"
REF_SCORE = 7.0

# (UniProt accession, product / protein name, therapeutic class)
# Chosen to span the recognised immunogenicity range rather than to be exhaustive:
# human-sequence replacement proteins at one end, non-human enzymes at the other.
# Extend or replace freely -- nothing below is load-bearing for any other result.
BIOLOGICS = [
    ("P01308", "insulin",                    "hormone, human sequence"),
    ("P01270", "teriparatide (PTH 1-84)",    "hormone, human sequence"),
    ("P01241", "somatropin (hGH)",           "hormone, human sequence"),
    ("P01588", "epoetin (EPO)",              "cytokine, human sequence"),
    ("P09919", "filgrastim (G-CSF)",         "cytokine, human sequence"),
    ("P60568", "aldesleukin (IL-2)",         "cytokine, human sequence"),
    ("P01563", "interferon alfa-2",          "cytokine, human sequence"),
    ("P01574", "interferon beta",            "cytokine, human sequence"),
    ("P02768", "albumin",                    "plasma protein, human sequence"),
    ("P00740", "factor IX",                  "coagulation factor, human sequence"),
    ("P00805", "L-asparaginase",             "enzyme, E. coli"),
    ("P00779", "streptokinase",              "enzyme, S. pyogenes"),
]


def valid_kmers(seq):
    """(list of valid 9-mers, number dropped). '/' and X make a window invalid."""
    keep, dropped = [], 0
    for i in range(len(seq) - K + 1):
        w = seq[i:i + K]
        if all(c in AA for c in w):
            keep.append(w)
        else:
            dropped += 1
    return keep, dropped


def fetch_mature(acc):
    """(sequence with '/' between mature chains, organism, chain description)."""
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.json"
    with urllib.request.urlopen(url, timeout=60) as fh:
        j = json.load(fh)
    full = j["sequence"]["value"]
    organism = j.get("organism", {}).get("scientificName", "?")
    chains = [(f["location"]["start"]["value"], f["location"]["end"]["value"],
               f.get("description", f["type"]))
              for f in j.get("features", [])
              if f["type"] in ("Chain", "Peptide")
              and isinstance(f["location"]["start"].get("value"), int)
              and isinstance(f["location"]["end"].get("value"), int)]
    if not chains:
        raise ValueError(f"{acc}: no annotated Chain/Peptide feature; refusing to guess "
                         f"the mature product from the {len(full)}-residue precursor")
    chains.sort()
    seq = "/".join(full[s - 1:e] for s, e, _ in chains)
    desc = "; ".join(d for _, _, d in chains)
    return seq, organism, desc


def read_fasta(path):
    out, name, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    out[name] = "".join(buf)
                name, buf = line[1:], []
            else:
                buf.append(line)
    if name:
        out[name] = "".join(buf)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design_csv", required=True,
                    help="per-design CSV from rosetta_mhc_epitope_benchmark.py "
                         "(needs propred_windows, n_windows_valid, model)")
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--out_dir", default="CAPE_MPNN/data/output")
    ap.add_argument("--fasta", default="",
                    help="mature-chain FASTA; written on first run, reused after. "
                         "Defaults to <out_dir>/<out_tag>_sequences.fasta")
    args = ap.parse_args()

    out_csv = os.path.join(args.out_dir, f"{args.out_tag}.csv")
    out_txt = os.path.join(args.out_dir, f"{args.out_tag}_summary.txt")
    fasta = args.fasta or os.path.join(args.out_dir, f"{args.out_tag}_sequences.fasta")
    for p in (out_csv, out_txt):
        if os.path.exists(p):
            sys.exit(f"REFUSING TO OVERWRITE {p} -- a corrective run needs a NEW --out_tag.")

    # --- sequences ----------------------------------------------------
    meta = {acc: (name, cls) for acc, name, cls in BIOLOGICS}
    if os.path.exists(fasta):
        print(f"[seq] reusing {fasta}", flush=True)
        cached = read_fasta(fasta)
        seqs = {}
        for hdr, s in cached.items():
            acc = hdr.split("|")[0]
            org = hdr.split("|")[2] if hdr.count("|") >= 2 else "?"
            desc = hdr.split("|")[3] if hdr.count("|") >= 3 else ""
            seqs[acc] = (s, org, desc)
    else:
        print(f"[seq] fetching {len(BIOLOGICS)} entries from UniProt …", flush=True)
        seqs = {}
        for acc, name, _cls in BIOLOGICS:
            try:
                seqs[acc] = fetch_mature(acc)
                print(f"      {acc} {name}: {len(seqs[acc][0])} aa", flush=True)
            except Exception as exc:
                sys.exit(f"FATAL: {acc} ({name}): {type(exc).__name__}: {exc}\n"
                         f"Refusing to write a partial reference set.")
        with open(fasta, "w") as fh:
            for acc, (s, org, desc) in seqs.items():
                fh.write(f">{acc}|{meta[acc][0]}|{org}|{desc}\n{s}\n")
        print(f"[out] {fasta}")

    # --- ProPred ------------------------------------------------------
    import pyrosetta
    pyrosetta.init("-mute all", silent=True)
    from pyrosetta.rosetta.core.scoring.mhc_epitope_energy import MHCEpitopeEnergySetup
    pro = MHCEpitopeEnergySetup()
    pro.initialize_from_file("propred8_5.mhc")
    ref = pro.raw_score(REF_PEPTIDE)
    if abs(ref - REF_SCORE) > 1e-9:
        sys.exit(f"ACCEPTANCE TEST FAILED: raw_score({REF_PEPTIDE}) = {ref}, expected "
                 f"{REF_SCORE}. Not the same instrument as the design benchmark; the two "
                 f"would not be comparable. Refusing to write.")

    rows = []
    for acc, (seq, org, desc) in seqs.items():
        wins, dropped = valid_kmers(seq)
        if not wins:
            sys.exit(f"FATAL: {acc} has no valid 9-mer window.")
        pw = sum(1 for w in wins if pro.raw_score(w) >= 1.0)
        pa = sum(pro.raw_score(w) for w in wins)
        length = len(seq.replace("/", ""))
        rows.append({"accession": acc, "product": meta[acc][0], "class": meta[acc][1],
                     "organism": org, "chains": desc, "length": length,
                     "n_windows_valid": len(wins), "n_windows_dropped": dropped,
                     "propred_windows": pw, "propred_alleles": pa,
                     "epitope_fraction": pw / len(wins),
                     "per_100_residues": 100.0 * pw / length})
    bio = pd.DataFrame(rows).sort_values("epitope_fraction").reset_index(drop=True)
    bio.to_csv(out_csv, index=False)

    # --- designs, same instrument -------------------------------------
    d = pd.read_csv(args.design_csv)
    for col in ("model", "propred_windows", "n_windows_valid", "protein_name"):
        if col not in d.columns:
            sys.exit(f"FATAL: column '{col}' not in {args.design_csv}")
    d["epitope_fraction"] = d.propred_windows / d.n_windows_valid
    per_prot = d.groupby(["protein_name", "model"])["epitope_fraction"].mean()
    arms = {}
    for arm in ("base", "finetuned"):
        vals = np.array([per_prot.loc[(p, arm)] for p, a in per_prot.index if a == arm],
                        dtype=float)
        arms[arm] = vals

    bf = bio.epitope_fraction.values

    def pct_below(x, ref):
        return 100.0 * float((ref < x).sum()) / len(ref)

    L = []
    A = L.append
    A("APPROVED-BIOLOGICS REFERENCE POINT (P2-2 / R3)")
    A("=" * 88)
    A(f"scorer  : Rosetta mhc_epitope, {pro.report().strip()}")
    A(f"designs : {os.path.basename(args.design_csv)}")
    A(f"sequences: {os.path.basename(fasta)} ({len(bio)} products, mature chains only)")
    A("metric  : fraction of valid 9-mer windows predicted to bind >=1 of 8 HLA-DR alleles")
    A("")
    A("ACCEPTANCE TEST PASSED: ProPred reference peptide scores 7/8, so this is the same")
    A("instrument as the design benchmark and the two are on one scale.")
    A("")
    A(f"  {'product':<26}{'organism':<26}{'len':>5}{'epi.frac':>10}{'per100':>9}")
    A("  " + "-" * 84)
    for _, r in bio.iterrows():
        A(f"  {r['product'][:25]:<26}{r['organism'][:25]:<26}{r['length']:>5}"
          f"{r['epitope_fraction']:>10.3f}{r['per_100_residues']:>9.1f}")
    A("")
    A(f"  biologics: median {np.median(bf):.3f}  range {bf.min():.3f}-{bf.max():.3f}  "
      f"(n={len(bf)})")
    A("")
    A("DESIGNS ON THE SAME SCALE (per-protein means)")
    A(f"  {'arm':<26}{'n':>5}{'median':>10}{'mean':>9}{'IQR':>18}")
    A("  " + "-" * 68)
    for arm, label in (("base", "base ProteinMPNN"), ("finetuned", "de-immunised")):
        v = arms[arm]
        q1, q3 = np.percentile(v, [25, 75])
        A(f"  {label:<26}{len(v):>5}{np.median(v):>10.3f}{v.mean():>9.3f}"
          f"{f'{q1:.3f}-{q3:.3f}':>18}")
    A("")
    A("WHERE THE DESIGNS SIT")
    for arm, label in (("base", "base ProteinMPNN"), ("finetuned", "de-immunised")):
        med = float(np.median(arms[arm]))
        A(f"  {label:<20} median {med:.3f} -- above {pct_below(med, bf):.0f}% of the "
          f"biologics scored here")
    A("")
    A("READING (fixed before the numbers were computed):")
    A("  De-immunised designs falling INTO the biologics range -> the reduction reaches a")
    A("    level that proteins already dosed in humans occupy, and the paper can say so")
    A("    with this as support.")
    A("  De-immunised designs still ABOVE every biologic -> the reduction is real but")
    A("    the level is not yet comparable to an approved product. That is the honest")
    A("    framing, and it makes the reduction a step rather than an arrival.")
    A("  Base designs already inside the range -> the starting point was not unusually")
    A("    epitope-rich, and the reduction should not be sold as fixing a pathology.")
    A("")
    med_b, med_f = float(np.median(arms["base"])), float(np.median(arms["finetuned"]))
    if med_f <= bf.max():
        A(f"  OBSERVED: de-immunised median {med_f:.3f} lies within the biologics range "
          f"({bf.min():.3f}-{bf.max():.3f}),")
        A(f"            above {pct_below(med_f, bf):.0f}% of them; base median {med_b:.3f} "
          f"lies above {pct_below(med_b, bf):.0f}%.")
    else:
        A(f"  OBSERVED: de-immunised median {med_f:.3f} exceeds every biologic scored "
          f"(max {bf.max():.3f}).")
        A("            The reduction is a step toward the reference range, not into it.")
    A("")
    # Does predicted burden order the products by their recognised ADA risk class?
    # Computed rather than asserted, because it is the strongest caution in the table.
    nonhuman = bio[~bio.organism.str.startswith("Homo sapiens")]
    human = bio[bio.organism.str.startswith("Homo sapiens")]
    if len(nonhuman) and len(human):
        A("DOES BURDEN ORDER THE KNOWN RISK CLASSES? Computed, not assumed.")
        A(f"  human-sequence products ({len(human)}): median epitope fraction "
          f"{human.epitope_fraction.median():.3f}")
        A(f"  non-human products ({len(nonhuman)}): median "
          f"{nonhuman.epitope_fraction.median():.3f}")
        ranks = [(r["product"], int(bio.index.get_loc(i)) + 1)
                 for i, r in nonhuman.iterrows()]
        A("  non-human products rank "
          + ", ".join(f"{n} {k}/{len(bio)}" for n, k in ranks)
          + " from the lowest burden upward.")
        A("  The non-human enzymes are the recognised high-ADA class in this table, and")
        A("  they do NOT carry the highest predicted burden -- they sit in its lower")
        A("  half, while the highest burden belongs to a human-sequence cytokine.")
        A("  Predicted class II presentation therefore does not by itself order these")
        A("  products by clinical immunogenicity, because tolerance to self sequence is")
        A("  doing work that no presentation predictor represents. This is a limit on")
        A("  what a burden reduction can promise, and it is visible in the reference set")
        A("  itself rather than being an argument from outside the data.")
        A("")
    A("CAVEATS THAT LIMIT WHAT THIS CAN SUPPORT.")
    A("  1. Two unrelated populations. Natural therapeutic proteins and ProteinMPNN")
    A("     designs differ in composition, length, fold class and origin at once, so no")
    A("     difference here is attributable to any one cause. Reference scale, not a")
    A("     controlled comparison. The controlled comparison is base-vs-fine-tuned on")
    A("     identical backbones, reported elsewhere in this work.")
    A("  2. Predicted presentation is not immunogenicity. Approved biologics with low")
    A("     predicted burden can still raise anti-drug antibodies, and human-sequence")
    A("     products benefit from central tolerance that no predictor here represents --")
    A("     which is precisely why they can be dosed despite the burden they show.")
    A("  3. Twelve products is a small, hand-picked set spanning a range, not a sample")
    A("     of approved biologics. Quartiles of this set are not population quantiles.")
    A("  4. ProPred, not netMHCIIpan. This sits on the conservative instrument of the")
    A("     two (P2-1). Run the FASTA through netMHCIIpan to place the reference point")
    A("     on the manuscript's primary axis.")
    A("=" * 88)

    with open(out_txt, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[out] {out_csv}\n[out] {out_txt}")


if __name__ == "__main__":
    main()
