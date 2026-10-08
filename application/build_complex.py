#!/usr/bin/env python
"""
build_complex.py -- assemble a clean minimal binder-target complex from a PDB.

Keeps only the binder chain and its true target chain(s); drops everything
ProteinMPNN should not see: non-standard residues, waters/ligands (HETATM that
are not amino acids), alternate locations other than A, and any model past the
first. The result is the fixed-context complex you feed to redesign.sh.

If you are unsure which chains actually form the interface, run contacts.py
first -- several PDB depositions pair non-adjacent chains.

Example:
    python build_complex.py --pdb 9nds.pdb --binder A --targets C D E \\
        --out complex/9nds_cplx.pdb

Reproduce one entry of the paper's panel:
    python build_complex.py --pdb 9cce.pdb --binder A --targets D \\
        --out complex/9cce_cplx.pdb
"""
import argparse
import os

# standard amino acids + selenomethionine (MSE), which crystallographers use in
# place of MET; everything else (waters, ligands, modified residues) is dropped.
AA3 = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER "
          "THR TRP TYR VAL MSE".split())


def build(src, binder, targets, out):
    keep = set([binder] + list(targets))
    n = 0
    per = {}
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(src) as f, open(out, "w") as o:
        for line in f:
            if line.startswith("ENDMDL"):      # first model only
                break
            if line.startswith("TER"):
                o.write(line)
                continue
            if line[:6].strip() in ("ATOM", "HETATM"):
                resn = line[17:20].strip()
                if resn not in AA3:              # drop waters/ligands/modified
                    continue
                alt = line[16]
                if alt not in (" ", "A"):        # single altloc
                    continue
                ch = line[21]
                if ch not in keep:               # binder + targets only
                    continue
                o.write(line)
                n += 1
                per[ch] = per.get(ch, 0) + 1
        o.write("END\n")
    return n, per


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb", required=True, help="source PDB deposition")
    ap.add_argument("--binder", required=True,
                    help="binder chain id (the chain to redesign)")
    ap.add_argument("--targets", nargs="*", default=[],
                    help="target chain id(s) kept as fixed context (omit for a monomer)")
    ap.add_argument("--out", required=True, help="output complex PDB path")
    args = ap.parse_args()

    n, per = build(args.pdb, args.binder, args.targets, args.out)
    print(f"{args.out}: binder={args.binder} targets={args.targets} "
          f"atoms_kept={n} per_chain={per}")
    if per.get(args.binder, 0) == 0:
        raise SystemExit(f"ERROR: no atoms kept for binder chain '{args.binder}' "
                         f"-- check the chain id against the PDB.")


if __name__ == "__main__":
    main()
