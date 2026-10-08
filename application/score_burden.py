#!/usr/bin/env python
"""
score_burden.py -- predicted MHC class II epitope burden per design, with an
interface / core / surface breakdown so you can check the reduction is not
concentrated at the binding interface.

For each labelled design run (a ProteinMPNN output FASTA), reports mean predicted
MHC-II presented-window count per sequence, sequence recovery vs the native, and
per-region (I/C/S) mutation rate and epitope coverage. If both a reference run
(default label "base") and other runs are given, prints the burden reduction of
each other run relative to the reference.

Region classes are assigned from the complex structure (biotite SASA):
    I  interface  (buries > 10 A^2 on binding)
    C  core       (isolated relative SASA < 0.20)
    S  surface    (everything else)
For a monomer complex (no target chains) there is no interface, so residues fall
into C/S only -- which is correct.

Example (compare de-immunised vs base on one complex):
    python score_burden.py \\
        --complex complex/9nds_cplx.pdb --binder A \\
        --pwm /path/to/mhc_2/pwm --libs /path/to/CAPE_MPNN/libs \\
        --designs base:out/9nds_base/seqs/9nds_cplx.fa \\
                  cape:out/9nds_cape/seqs/9nds_cplx.fa \\
        --out 9nds_burden.json

--pwm and --libs default to env vars MHC2_PWM and CAPE_LIBS if set.
"""
import argparse
import json
import os
import sys

import numpy as np

# Tien 2013 theoretical max ASA
MAXASA = {'A': 129, 'R': 274, 'N': 195, 'D': 193, 'C': 167, 'E': 223, 'Q': 225,
          'G': 104, 'H': 224, 'I': 197, 'L': 201, 'K': 236, 'M': 224, 'F': 240,
          'P': 159, 'S': 155, 'T': 172, 'W': 285, 'Y': 263, 'V': 174}
THREE2ONE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
             'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
             'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
             'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
             'MSE': 'M'}


def classify(complex_path, binder):
    """Return (per-residue class list 'I'/'C'/'S', binder sequence) from structure."""
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile
    arr = PDBFile.read(complex_path).get_structure(model=1)
    arr = arr[struc.filter_amino_acids(arr)]

    def res_sasa(a):
        atom = struc.sasa(a, vdw_radii="ProtOr")
        rs = struc.apply_residue_wise(a, atom, np.nansum)
        starts = struc.get_residue_starts(a)
        info = [(a.chain_id[s], a.res_id[s], a.res_name[s]) for s in starts]
        return rs, info

    sas_c, info_c = res_sasa(arr)                 # SASA in the complex
    barr = arr[arr.chain_id == binder]
    sas_i, info_i = res_sasa(barr)                # SASA of the isolated binder
    cmap = {rid: s for k, (ch, rid, _) in enumerate(info_c)
            for s in [sas_c[k]] if ch == binder}
    classes, seq = [], []
    for k, (ch, rid, rn) in enumerate(info_i):
        aa = THREE2ONE.get(rn, 'A')
        seq.append(aa)
        mx = MAXASA.get(aa, 200)
        rasa_iso = sas_i[k] / mx
        d_abs = sas_i[k] - cmap.get(rid, sas_i[k])   # SASA buried on binding
        if d_abs > 10:
            cls = 'I'
        elif rasa_iso < 0.20:
            cls = 'C'
        else:
            cls = 'S'
        classes.append(cls)
    return classes, ''.join(seq)


def read_fa(path):
    """Return (native_seq, [design_seqs]) from a ProteinMPNN output FASTA."""
    seqs, block = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                if block:
                    seqs.append(''.join(block))
                    block = []
            else:
                block.append(line.strip())
        if block:
            seqs.append(''.join(block))
    return seqs[0], seqs[1:]


def align_classes(fa_native, struct_classes):
    """Map structure-derived classes onto the ProteinMPNN design frame.
    ProteinMPNN may emit 'X' for residues biotite dropped; those get class 'U'."""
    out, j = [], 0
    for ch in fa_native:
        if ch == 'X':
            out.append('U')
        else:
            out.append(struct_classes[j] if j < len(struct_classes) else 'U')
            j += 1
    return out


def score_run(fa_path, classes, pred, alleles, lengths):
    """Burden / recovery / per-class mutation & coverage for one design FASTA."""
    L = len(classes)
    native, designs = read_fa(fa_path)
    tot, recov = [], []
    mut_by_cls = {c: np.zeros(L) for c in 'ICSU'}
    cov_by_pos = np.zeros(L)
    cls_counts = {c: 0 for c in 'ICSU'}
    for c in classes:
        cls_counts[c] += 1
    for d in designs:
        if len(d) != L:                            # positional-alignment guard
            continue
        pres = pred.seq_presented(d, alleles=alleles, lengths=lengths)
        tot.append(len(pres))
        recov.append(sum(1 for i in range(L) if d[i] == native[i]) / L)
        for i in range(L):
            if d[i] != native[i]:
                mut_by_cls[classes[i]][i] += 1
        for (_pep, _al, _r, end) in pres:
            for i in range(max(0, end - 14), end + 1):
                if i < L:
                    cov_by_pos[i] += 1
    nd = len(tot)
    if nd == 0:
        raise SystemExit(f"ERROR: no design in {fa_path} matched the reference "
                         f"length {L}. Check --binder and the complex.")
    cov_cls = {cl: float(sum(cov_by_pos[i] for i in range(L) if classes[i] == cl)) / nd
               for cl in 'ICS'}
    mut_rate = {}
    for cl in 'ICS':
        npos = cls_counts[cl]
        mut_rate[cl] = float(sum(mut_by_cls[cl])) / (npos * nd) if npos > 0 else float('nan')
    return dict(n=nd, burden_mean=float(np.mean(tot)), burden_sd=float(np.std(tot)),
                recov_mean=float(np.mean(recov)), cov_cls=cov_cls, mut_rate=mut_rate,
                cls_counts=cls_counts)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--complex", required=True, help="minimal complex PDB (from build_complex.py)")
    ap.add_argument("--binder", required=True, help="binder chain id")
    ap.add_argument("--designs", required=True, nargs="+", metavar="LABEL:FASTA",
                    help="labelled ProteinMPNN output FASTAs, e.g. base:b.fa cape:c.fa")
    ap.add_argument("--pwm", default=os.environ.get("MHC2_PWM"),
                    help="Mhc2PredictorPwm PWM directory (or set MHC2_PWM)")
    ap.add_argument("--libs", default=os.environ.get("CAPE_LIBS"),
                    help="path to CAPE_MPNN/libs for the kit package (or set CAPE_LIBS)")
    ap.add_argument("--alleles",
                    default="DRB1_0101+DRB1_0301+DRB1_0401+DRB1_0701+DRB1_1101+DRB1_1501",
                    help="'+'-joined HLA-DRB1 panel (default: the paper's six alleles)")
    ap.add_argument("--reference", default="base",
                    help="label to report reductions against (default: base)")
    ap.add_argument("--out", help="write per-run results as JSON here")
    args = ap.parse_args()

    if not args.pwm:
        ap.error("--pwm is required (or set MHC2_PWM)")
    if args.libs:
        sys.path.insert(0, args.libs)
    try:
        from kit.bioinf.immuno.mhc_2 import Mhc2PredictorPwm, MHC_2_PEPTIDE_LENGTHS
    except ImportError as e:
        ap.error(f"cannot import kit.bioinf.immuno.mhc_2 ({e}); pass --libs "
                 f".../CAPE_MPNN/libs or set CAPE_LIBS")

    runs = []
    for spec in args.designs:
        if ":" not in spec:
            ap.error(f"--designs entry '{spec}' must be LABEL:FASTA")
        label, path = spec.split(":", 1)
        runs.append((label, path))

    pred = Mhc2PredictorPwm(args.pwm)
    struct_classes, _ = classify(args.complex, args.binder)
    # class frame aligned once using the reference run's native (identical native
    # across runs on the same input)
    ref_path = dict(runs).get(args.reference, runs[0][1])
    ref_native, _ = read_fa(ref_path)
    classes = align_classes(ref_native, struct_classes)
    L = len(classes)

    results = {}
    for label, path in runs:
        results[label] = score_run(path, classes, pred, args.alleles, MHC_2_PEPTIDE_LENGTHS)

    icnt = classes.count('I')
    ccnt = classes.count('C')
    scnt = classes.count('S')
    print(f"\ncomplex {os.path.basename(args.complex)}  binder {args.binder}  "
          f"L={L}  (I/C/S = {icnt}/{ccnt}/{scnt})")
    for label, r in results.items():
        print(f"  [{label:>6}] burden {r['burden_mean']:6.2f}±{r['burden_sd']:5.2f}   "
              f"recovery {r['recov_mean']:.3f}   "
              f"mut I/C/S {r['mut_rate']['I']:.2f}/{r['mut_rate']['C']:.2f}/{r['mut_rate']['S']:.2f}   "
              f"(n={r['n']})")
    ref = results.get(args.reference)
    if ref and ref['burden_mean'] > 0:
        for label, r in results.items():
            if label == args.reference:
                continue
            red = 100 * (ref['burden_mean'] - r['burden_mean']) / ref['burden_mean']
            print(f"  reduction {label} vs {args.reference}: {red:+.1f}%")

    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
