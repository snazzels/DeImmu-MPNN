#!/usr/bin/env python
"""Is de-immunization harder on designs that already start with a low predicted burden?

Motivation: if some base ProteinMPNN designs already carry few predicted MHC-I/II
epitopes, there may be little headroom left, and the reported mean reductions could
be driven by the heavily-loaded designs while the already-clean ones barely move
(a floor effect). This also emits the per-entity numbers behind the paper.

Covers both applications and both MHC classes, base vs the deployed model f16b51e6:

  monomers  MHC-II  validate_netmhciipan_sweep_f16b51e6.csv          (per-seq -> per-protein)
  monomers  MHC-I   validate_netmhcpan_monomer_f16b51e6_summary.txt  (per-protein table)
  binders   MHC-II  validate_netmhciipan_binderpanel_v2.csv          (per-seq -> per-complex)
  binders   MHC-I   binder_redesign_v2_f16b51e6/
                      validate_netmhcpan_binderpanel_v2_summary.txt  (per-complex table)

Burden is also expressed as a DENSITY (presented windows per 100 residues), because
raw window counts scale with chain length and the two sets differ a lot in length
(monomers median ~272 aa, binder chains 40-248 aa). "Already low immunogenicity"
is a statement about density, not about a raw count.

Statistics are computed without scipy: the local numpy (1.21.5) is below scipy's
requirement and it emits a version warning, so Spearman is done by ranking and a
permutation test gives the p-value. Reproducible via a fixed seed.

Outputs (refuses to overwrite):
  baseline_burden_headroom.csv          tidy per-entity table = the numbers behind the paper
  baseline_burden_headroom_summary.txt  correlations, quartile bins, the low-burden tail
"""
import os, csv, re, sys, random, collections
import statistics as st

PF = os.environ.get("PF", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OD = os.path.join(PF, "data", "output")
OUT_CSV = os.path.join(OD, "baseline_burden_headroom.csv")
OUT_TXT = os.path.join(OD, "baseline_burden_headroom_summary.txt")
NPERM, SEED = 20000, 20260820


def per_protein(path, value_col="netmhc_total"):
    """Collapse a per-sequence eval CSV to per-protein base/ft means."""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    length = {}
    for r in csv.DictReader(open(path)):
        acc[r["protein_name"]][r["model"]].append(float(r[value_col]))
        length[r["protein_name"]] = int(r["protein_length"])
    out = {}
    for name, m in acc.items():
        if "base" in m and "finetuned" in m:
            out[name] = (st.mean(m["base"]), st.mean(m["finetuned"]), length[name])
    return out


def parse_table(path):
    """Parse the 'name base_mean cape_mean reduction%' tables emitted by the
    netMHCpan scorers."""
    out = {}
    for ln in open(path):
        m = re.match(r"^(\S+)\s+([\d.]+)\s+([\d.]+)\s+[+-][\d.]+\s*$", ln.strip())
        if m:
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


def spearman(xs, ys):
    """Spearman rho + two-sided permutation p (no scipy)."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):                      # average ties
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0 + 1
            i = j + 1
        return r

    def pearson(a, b):
        ma, mb = st.mean(a), st.mean(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
        return num / den if den else float("nan")

    ra, rb = rank(xs), rank(ys)
    rho = pearson(ra, rb)
    rnd = random.Random(SEED)
    shuf, hits = list(rb), 0
    for _ in range(NPERM):
        rnd.shuffle(shuf)
        if abs(pearson(ra, shuf)) >= abs(rho) - 1e-12:
            hits += 1
    return rho, (hits + 1) / (NPERM + 1)


# ------------------------------------------------------------------ assemble
mono2 = per_protein(os.path.join(OD, "validate_netmhciipan_sweep_f16b51e6.csv"))
bind2 = per_protein(os.path.join(OD, "validate_netmhciipan_binderpanel_v2.csv"))
mono1 = parse_table(os.path.join(OD, "validate_netmhcpan_monomer_f16b51e6_summary.txt"))
bind1 = parse_table(os.path.join(OD, "binder_redesign_v2_f16b51e6",
                                 "validate_netmhcpan_binderpanel_v2_summary.txt"))

rows = []
for setname, mhc2, mhc1 in (("monomer", mono2, mono1), ("binder", bind2, bind1)):
    for name, (b2, f2, L) in sorted(mhc2.items()):
        b1, f1 = mhc1.get(name, (None, None))
        rows.append(dict(
            entity=name, set=setname, length=L,
            mhc2_base=b2, mhc2_ft=f2,
            mhc2_base_density=100.0 * b2 / L,
            mhc2_abs_drop=b2 - f2,
            mhc2_pct_drop=(100.0 * (b2 - f2) / b2) if b2 else None,
            mhc1_base=b1, mhc1_ft=f1,
            mhc1_base_density=(100.0 * b1 / L) if b1 is not None else None,
            mhc1_abs_drop=(b1 - f1) if b1 is not None else None,
            mhc1_pct_drop=(100.0 * (b1 - f1) / b1) if b1 else None))

for p in (OUT_CSV, OUT_TXT):
    if os.path.exists(p):
        sys.exit(f"Refusing to overwrite {p}")

cols = list(rows[0])
with open(OUT_CSV, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r[k] is None else
                        (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]))
                    for k in cols})

# ------------------------------------------------------------------- analysis
L = ["BASELINE BURDEN vs ACHIEVED REDUCTION — is there a floor effect?",
     "=" * 78,
     "Deployed model f16b51e6 vs base ProteinMPNN. Burden = presented windows",
     "(netMHCIIpan 15-mers / netMHCpan 8-10-mers, %Rank_EL<=2). Density = per 100 aa.",
     f"Spearman rho with a {NPERM}-permutation two-sided p (seed {SEED}); no scipy.", ""]

for setname in ("monomer", "binder"):
    sub = [r for r in rows if r["set"] == setname]
    L.append(f"--- {setname}s (n={len(sub)}) " + "-" * 46)
    for cls in ("mhc2", "mhc1"):
        ok = [r for r in sub if r[f"{cls}_pct_drop"] is not None
              and r[f"{cls}_base"] not in (None, 0)]
        if not ok:
            L.append(f"  {cls.upper()}: no data"); continue
        dens = [r[f"{cls}_base_density"] for r in ok]
        pct = [r[f"{cls}_pct_drop"] for r in ok]
        absd = [r[f"{cls}_abs_drop"] for r in ok]
        rho_p, p_p = spearman(dens, pct)
        rho_a, p_a = spearman(dens, absd)
        lab = "MHC-II" if cls == "mhc2" else "MHC-I "
        L.append(f"  {lab} n={len(ok):3d}  base density {st.mean(dens):5.1f}/100aa "
                 f"(range {min(dens):.1f}-{max(dens):.1f})  mean drop {st.mean(pct):5.1f}%")
        L.append(f"         base density vs %  drop : rho={rho_p:+.3f} p={p_p:.4f}")
        L.append(f"         base density vs abs drop: rho={rho_a:+.3f} p={p_a:.4f}")
        q = sorted(ok, key=lambda r: r[f"{cls}_base_density"])
        n4 = max(1, len(q) // 4)
        for qi, (lo, hi) in enumerate(((0, n4), (n4, 2 * n4), (2 * n4, 3 * n4),
                                       (3 * n4, len(q)))):
            grp = q[lo:hi]
            if grp:
                L.append(f"         Q{qi+1} density "
                         f"{st.mean(r[f'{cls}_base_density'] for r in grp):5.1f}/100aa"
                         f"  ->  drop {st.mean(r[f'{cls}_pct_drop'] for r in grp):5.1f}%"
                         f"  (abs {st.mean(r[f'{cls}_abs_drop'] for r in grp):5.1f} windows)")
        L.append(f"         lowest-burden 5: " + ", ".join(
            f"{r['entity']} {r[f'{cls}_base_density']:.1f}->{r[f'{cls}_pct_drop']:+.0f}%"
            for r in q[:5]))
    L.append("")

# designs with no headroom at all
zero = [r for r in rows if r["mhc2_base"] == 0]
L.append(f"Designs with ZERO base MHC-II windows (no headroom, % undefined): "
         f"{len(zero)}" + (" — " + ", ".join(r['entity'] for r in zero) if zero else ""))
low = sorted(rows, key=lambda r: r["mhc2_base_density"])[:10]
L.append("Ten lowest MHC-II base densities (windows/100 aa), with outcome:")
for r in low:
    pc = "n/a" if r["mhc2_pct_drop"] is None else f"{r['mhc2_pct_drop']:+.1f}%"
    L.append(f"  {r['entity']:12s} {r['set']:8s} L={r['length']:4d}  "
             f"base {r['mhc2_base']:6.2f} ({r['mhc2_base_density']:5.1f}/100aa)  -> {pc}")

open(OUT_TXT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
print(f"\nWrote {OUT_CSV}\nWrote {OUT_TXT}")
