#!/usr/bin/env python
"""Illustrative structure figure: where on a fold do the de-immunising changes land?

PURPOSE, AND ITS LIMITS. Requested by the collaborating immunologist (2026-09-28) as
an illustration -- "nur zur Illustration". Nothing quantitative in the manuscript rests
on it. It shows one representative backbone from the pinned evaluation set with
(a) the per-residue predicted class II presentation burden before and after
de-immunisation, and (b) the positions at which the de-immunised model proposes a
different residue from base ProteinMPNN.

WHAT "THE PROPOSED CHANGES" ACTUALLY ARE, STATED PRECISELY BECAUSE THE FIGURE INVITES
A WRONG READING. Base and de-immunised sequences are two INDEPENDENT design runs on the
same fixed backbone; they are not a mutational trajectory and no residue was "mutated"
from one to the other. The highlighted positions are those where the representative
de-immunised design differs from the representative base design. Any caption must say
this, or a reader will take the marks for point mutations applied to a starting protein.

REPRESENTATIVE-SEQUENCE SELECTION, FIXED BEFORE LOOKING AT STRUCTURE. Each arm supplies
k=5 designs per backbone. The one whose netmhc_total is closest to that arm's own mean
is chosen, so the picture shows a typical member of each arm rather than the best case.

PROTOCOL for the per-residue burden is the frozen one: 15-mer windows, the six-allele
DRB1 panel, presented := %Rank_EL <= 2.0. A residue's burden is the number of
(allele, window) pairs that are predicted presented AND cover that residue, so it is
bounded by 6 x 15 = 90 and reflects how deeply a position sits inside predicted epitopes.

BACKBONE SOURCE. The ProteinMPNN 2021aug02 chain file, which is the very backbone the
designs were generated on -- not a folded model of the design. Only N/CA/C/O are written,
so the cartoon is real and no side-chain geometry is invented for residues whose
conformation was never modelled. Substituted positions are therefore marked on CA.

Outputs (refuses to overwrite) under data/output/epitope_structure_<protein>_<tag>/.
"""
import os, sys, csv, glob, argparse, subprocess, tempfile, collections
import statistics as st

PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Input resolution (deposition copy, 2026-10-05).
#
# In the RELEASE repo this file sits in figures/ and its design CSV is a
# deposited artefact under data/<section>/; in the original working tree it sat
# in CAPE_MPNN/tools/ and read CAPE_MPNN/data/output/. The CSV is therefore
# located by basename across both layouts, with $CAPE_ROOT as a fallback.
#
# The BACKBONE source is not deposited and cannot be: PDB_ROOT is the
# ProteinMPNN pdb_2021aug02 training set (tens of GB, and redistribution is not
# ours to grant). Point $PDB_CACHE_DIR at a local copy -- see environment.md.
# Unlike the CSV this is a hard external dependency, so it fails loudly with
# that instruction rather than resolving to something plausible but wrong.
# ---------------------------------------------------------------------------
_ROOTS = [os.path.join(PF, "data"),
          os.path.join(PF, "CAPE_MPNN", "data", "output")]
if os.environ.get("CAPE_ROOT"):
    _ROOTS.append(os.path.join(os.environ["CAPE_ROOT"],
                               "CAPE_MPNN", "data", "output"))


def datafile(basename):
    """Absolute path to a deposited input, searched release tree first."""
    for root in _ROOTS:
        direct = os.path.join(root, basename)
        if os.path.isfile(direct):
            return direct
        hit = sorted(glob.glob(os.path.join(root, "*", basename)))
        if hit:
            return hit[0]
    raise SystemExit(
        "cannot locate input %r; searched %s. Run from the release repo, or "
        "set CAPE_ROOT (environment.md)." % (basename, _ROOTS))


OD = os.path.join(PF, "data", "output")
DESIGN_CSV = datafile("validate_netmhciipan_sweep_f16b51e6.csv")
PDB_ROOT = os.environ.get(
    "PDB_CACHE_DIR",
    os.path.join(PF, "data", "input", "CAPE-MPNN", "pdb_2021aug02", "pdb"))

ALLELES = ["DRB1_0101", "DRB1_0301", "DRB1_0401", "DRB1_0701",
           "DRB1_1101", "DRB1_1501"]
THRESHOLD, PEPTIDE_LEN, BATCH = 2.0, 15, 5000

AA3 = dict(A="ALA", C="CYS", D="ASP", E="GLU", F="PHE", G="GLY", H="HIS",
           I="ILE", K="LYS", L="LEU", M="MET", N="ASN", P="PRO", Q="GLN",
           R="ARG", S="SER", T="THR", V="VAL", W="TRP", Y="TYR")


def netmhciipan(peptides, allele):
    ranks = {}
    for i in range(0, len(peptides), BATCH):
        with tempfile.TemporaryDirectory() as td:
            pf = os.path.join(td, "in.pep")
            open(pf, "w").write("\n".join(peptides[i:i + BATCH]))
            out = subprocess.run(["netMHCIIpan", "-inptype", "1", "-f", pf,
                                  "-a", allele], capture_output=True,
                                 check=True).stdout.decode()
        for line in out.splitlines():
            p = line.split()
            if len(p) < 10:
                continue
            try:
                int(p[0])
            except ValueError:
                continue
            if p[2] not in ranks or float(p[9]) < ranks[p[2]]:
                ranks[p[2]] = float(p[9])
    return ranks


def per_residue_burden(seq):
    """Presented (allele, window) pairs covering each residue. Also the window list."""
    wins = [seq[i:i + PEPTIDE_LEN] for i in range(len(seq) - PEPTIDE_LEN + 1)]
    uniq = sorted(set(wins))
    ranks = {a: netmhciipan(uniq, a) for a in ALLELES}
    cov = [0] * len(seq)
    presented = []
    for a in ALLELES:
        for i, w in enumerate(wins):
            if ranks[a].get(w, 999) <= THRESHOLD:
                presented.append((a, i))
                for j in range(i, i + PEPTIDE_LEN):
                    cov[j] += 1
    return cov, presented, len(wins)


def load_backbone(name):
    import torch
    pdb = name.split("_")[0]
    f = os.path.join(PDB_ROOT, pdb[1:3], f"{name}.pt")
    if not os.path.exists(f):
        sys.exit(f"no backbone file: {f}\n"
                 "The ProteinMPNN pdb_2021aug02 set is an external dependency and "
                 "is not deposited. Point $PDB_CACHE_DIR at a local copy "
                 "(see environment.md).")
    d = torch.load(f, map_location="cpu")
    return d["xyz"][:, :4], d["mask"][:, :4], d["seq"]


def write_pdb(path, xyz, mask, seq, bfac):
    """N/CA/C/O only. B-factor carries the per-residue burden."""
    names = ["N", "CA", "C", "O"]
    n = 0
    with open(path, "w") as fh:
        for i in range(len(seq)):
            res = AA3.get(seq[i], "GLY")
            for k, an in enumerate(names):
                if not bool(mask[i, k]):
                    continue
                x, y, z = (float(v) for v in xyz[i, k])
                n += 1
                fh.write(f"ATOM  {n:5d}  {an:<3s}{res:>4s} A{i+1:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfac[i]:6.2f}"
                         f"          {an[0]:>2s}\n")
        fh.write("TER\nEND\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein", default="2wjr_A")
    ap.add_argument("--tag", default="v1")
    # $PYMOL_BIN is the documented hook (environment.md); the flag still wins.
    ap.add_argument("--pymol", default=os.environ.get("PYMOL_BIN", "pymol"))
    ap.add_argument("--outdir", default=OD,
                    help="parent directory for the run's output folder "
                         "(default: data/output under the repo root)")
    args = ap.parse_args()

    outdir = os.path.join(args.outdir,
                          f"epitope_structure_{args.protein}_{args.tag}")
    if os.path.exists(outdir):
        sys.exit(f"REFUSING TO OVERWRITE existing output dir: {outdir}")
    os.makedirs(outdir)

    rows = [r for r in csv.DictReader(open(DESIGN_CSV))
            if r["protein_name"] == args.protein]
    if not rows:
        sys.exit(f"{args.protein} not in {os.path.basename(DESIGN_CSV)}")

    chosen, stats_ = {}, {}
    for arm in ("base", "finetuned"):
        sub = [r for r in rows if r["model"] == arm]
        mean = st.mean(float(r["netmhc_total"]) for r in sub)
        pick = min(sub, key=lambda r: abs(float(r["netmhc_total"]) - mean))
        chosen[arm] = pick["sequence"].replace("/", "")
        stats_[arm] = (mean, float(pick["netmhc_total"]))
        print(f"{arm:10s} arm mean {mean:6.2f}  representative seq_idx "
              f"{pick['seq_idx']} total {float(pick['netmhc_total']):.0f}")

    if len(chosen["base"]) != len(chosen["finetuned"]):
        sys.exit("arm sequences differ in length — not a shared backbone")

    xyz, mask, native = load_backbone(args.protein)
    L = len(chosen["base"])
    if len(native) != L:
        sys.exit(f"backbone {len(native)} residues vs sequence {L} — "
                 "multi-chain entry, pick a single-chain protein")

    print("scoring both arms with netMHCIIpan ...")
    cov, pres, nwin = {}, {}, None
    for arm in ("base", "finetuned"):
        cov[arm], pres[arm], nwin = per_residue_burden(chosen[arm])
        print(f"  {arm:10s} presented windows {len(pres[arm]):4d}  "
              f"max residue burden {max(cov[arm])}")

    subs = [i for i in range(L) if chosen["base"][i] != chosen["finetuned"][i]]
    delta = [cov["base"][i] - cov["finetuned"][i] for i in range(L)]

    for arm in ("base", "finetuned"):
        write_pdb(os.path.join(outdir, f"{args.protein}_{arm}.pdb"),
                  xyz, mask, chosen[arm], cov[arm])
    write_pdb(os.path.join(outdir, f"{args.protein}_delta.pdb"),
              xyz, mask, chosen["base"], delta)

    with open(os.path.join(outdir, "per_residue.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["resi", "native", "base_aa", "deimmunised_aa", "substituted",
                    "base_burden", "deimmunised_burden", "delta_burden"])
        for i in range(L):
            w.writerow([i + 1, native[i], chosen["base"][i],
                        chosen["finetuned"][i], int(i in subs),
                        cov["base"][i], cov["finetuned"][i], delta[i]])

    vmax = max(max(cov["base"]), max(cov["finetuned"])) or 1
    # Substitutions that sit inside a window the base design presents are the ones
    # doing de-immunising work; the rest are ordinary design variation between two
    # independent runs and would only add noise to an illustration.
    subs_in = [i for i in subs if cov["base"][i] > 0]
    sub_sel = "+".join(str(i + 1) for i in subs_in) or "0"
    # The spheres live in their OWN objects. Colouring a CA atom in the cartoon object
    # recolours that residue's cartoon segment and overwrites the burden spectrum —
    # which is exactly what went wrong in v1.
    pml = f"""
load {args.protein}_base.pdb, base
load {args.protein}_finetuned.pdb, deim
create subs_b, base and name CA and resi {sub_sel}
create subs_d, deim and name CA and resi {sub_sel}
hide everything
show cartoon, base or deim
bg_color white
set ray_opaque_background, 1
set cartoon_highlight_color, grey70
spectrum b, white_yellow_orange_red, base, minimum=0, maximum={vmax}
spectrum b, white_yellow_orange_red, deim, minimum=0, maximum={vmax}
show spheres, subs_b or subs_d
set sphere_scale, 0.30
color skyblue, subs_b
color skyblue, subs_d
set ray_shadows, 0
set antialias, 2
set depth_cue, 0
orient base
turn x, -15
disable deim
disable subs_d
png panel_base.png, width=1500, height=1500, dpi=300, ray=1
disable base
disable subs_b
enable deim
enable subs_d
png panel_deimmunised.png, width=1500, height=1500, dpi=300, ray=1
"""
    open(os.path.join(outdir, "render.pml"), "w").write(pml)

    if os.path.exists(args.pymol):
        print("rendering with pymol ...")
        r = subprocess.run([args.pymol, "-cq", "render.pml"], cwd=outdir,
                           capture_output=True)
        if r.returncode != 0:
            print("pymol failed:\n", r.stderr.decode()[-2000:])
        else:
            print("rendered", [f for f in sorted(os.listdir(outdir))
                               if f.endswith(".png")])
    else:
        print(f"pymol not found at {args.pymol} — render.pml written, run it manually")

    bt, ft = sum(cov["base"]), sum(cov["finetuned"])
    summary = f"""ILLUSTRATIVE EPITOPE-STRUCTURE FIGURE — {args.protein}
{'='*76}
backbone      : ProteinMPNN 2021aug02 chain file, {L} residues (the design backbone
                itself, not a folded model of the design)
scorer        : netMHCIIpan, {len(ALLELES)} DRB1 alleles, 15-mer windows,
                presented := %Rank_EL <= {THRESHOLD}; {nwin} windows per sequence
sequences     : the design of each arm whose netmhc_total is closest to that arm's
                own k=5 mean (base {stats_['base'][0]:.2f} -> picked {stats_['base'][1]:.0f};
                de-immunised {stats_['finetuned'][0]:.2f} -> picked {stats_['finetuned'][1]:.0f})

presented windows   base {len(pres['base']):4d}   de-immunised {len(pres['finetuned']):4d}
residue-burden sum  base {bt:5d}   de-immunised {ft:5d}   ({100*(bt-ft)/bt:.1f}% lower)
positions differing between the two designs: {len(subs)} of {L} ({100*len(subs)/L:.0f}%)

READING THE PANELS
  colour  white -> red, per-residue count of presented (allele, window) pairs
          covering that residue, on one shared scale 0-{vmax} for both panels
  spheres sky-blue CA spheres at the {len(subs_in)} positions where the de-immunised
          design proposes a different residue AND the base design presents a window
          covering that position. The other {len(subs)-len(subs_in)} differing positions
          lie outside every predicted base epitope and are not marked, since they are
          ordinary design variation between two independent runs.

THE CAPTION MUST SAY that the two arms are independent designs on one fixed backbone,
not a mutational path, or the spheres will be read as point mutations.
{'='*76}
"""
    open(os.path.join(outdir, "summary.txt"), "w").write(summary)
    print("\n" + summary)
    print("outputs in", outdir)


if __name__ == "__main__":
    main()
