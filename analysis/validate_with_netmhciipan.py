#!/usr/bin/env python
"""
Validate de-immunisation using netMHCIIpan as the ground-truth predictor.

Takes a subset of proteins and sequences from the PWM-based eval CSV,
re-scores every 15-mer window with netMHCIIpan, and compares:
  (1) Does the fine-tuned model still show reduced immunogenicity?
  (2) How well does the PWM fast-predictor correlate with netMHCIIpan?

Usage:
    python tools/validate_with_netmhciipan.py [--n_proteins 20] [--k_seqs 5]

Output:
    data/output/validate_netmhciipan.csv         per-sequence results
    data/output/validate_netmhciipan_summary.txt summary stats
    data/output/validate_netmhciipan_figure.png  validation figure
"""

import os, sys, csv, argparse, subprocess, tempfile, time
import numpy as np
from scipy import stats
from collections import defaultdict

PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PF, "libs"))

EVAL_CSV     = os.path.join(PF, "data", "output", "eval_deimmunisation.csv")
OUT_CSV      = os.path.join(PF, "data", "output", "validate_netmhciipan.csv")
OUT_SUMMARY  = os.path.join(PF, "data", "output", "validate_netmhciipan_summary.txt")
OUT_FIG      = os.path.join(PF, "data", "output", "validate_netmhciipan_figure.png")

ALLELES      = ["DRB1_0101","DRB1_0301","DRB1_0401","DRB1_0701","DRB1_1101","DRB1_1501"]
THRESHOLD    = 2.0    # %Rank_EL ≤ 2% = "presented"
PEPTIDE_LEN  = 15
BATCH_SIZE   = 5000


# ── netMHCIIpan wrappers (adapted from MHC-II_rank_peptides.py) ──────────────

def run_netmhciipan(peptides: list, allele: str) -> dict:
    """Run netMHCIIpan on a list of 15-mers. Returns {peptide: best_%Rank_EL}."""
    ranks = {}
    batches = [peptides[i:i+BATCH_SIZE] for i in range(0, len(peptides), BATCH_SIZE)]
    for batch in batches:
        with tempfile.TemporaryDirectory() as tmpdir:
            pep_file = os.path.join(tmpdir, "input.pep")
            with open(pep_file, "w") as fh:
                fh.write("\n".join(batch))
            result = subprocess.run(
                ["netMHCIIpan", "-inptype", "1", "-f", pep_file, "-a", allele],
                capture_output=True, check=True
            )
            ranks.update(_parse_netmhciipan(result.stdout.decode().splitlines()))
    return ranks


def _parse_netmhciipan(lines: list) -> dict:
    """Parse netMHCIIpan stdout. Returns {peptide: %Rank_EL} keeping best score."""
    ranks = {}
    for line in lines:
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            int(parts[0])
        except ValueError:
            continue
        peptide  = parts[2]
        rank_el  = float(parts[9])
        if peptide not in ranks or rank_el < ranks[peptide]:
            ranks[peptide] = rank_el
    return ranks


def get_15mers(seq: str) -> list:
    """Slide a 15-mer window along a sequence. Returns list of peptides."""
    seq = seq.replace("*","").replace("-","").replace("/","")
    return [seq[i:i+PEPTIDE_LEN] for i in range(len(seq) - PEPTIDE_LEN + 1)]


def count_presented(seq: str, ranks_by_allele: dict) -> dict:
    """Count presented windows per allele and in total."""
    kmers   = get_15mers(seq)
    counts  = {}
    for allele in ALLELES:
        ar = ranks_by_allele.get(allele, {})
        counts[allele] = sum(1 for k in kmers if ar.get(k, 999) <= THRESHOLD)
    counts["total"] = sum(counts[allele] for allele in ALLELES)
    return counts


# ── Protein / sequence selection ──────────────────────────────────────────────

def select_proteins(rows: list, n_proteins: int) -> list:
    """Pick n_proteins stratified by protein length quartiles."""
    proteins = sorted(set(r["protein_name"] for r in rows))
    # Map protein → median PWM total (from base model)
    base_rows = [r for r in rows if r["model"] == "base"]
    prot_len  = {r["protein_name"]: int(r["protein_length"]) for r in rows}

    lengths = np.array([prot_len[p] for p in proteins])
    quartiles = np.percentile(lengths, [25, 50, 75])
    per_q = max(1, n_proteins // 4)

    rng = np.random.default_rng(42)
    selected = []
    edges = [0, quartiles[0], quartiles[1], quartiles[2], 9999]
    for lo, hi in zip(edges[:-1], edges[1:]):
        bucket = [p for p, l in zip(proteins, lengths) if lo <= l < hi]
        k = min(per_q, len(bucket))
        selected.extend(rng.choice(bucket, size=k, replace=False).tolist())

    return selected[:n_proteins]


def pick_sequences(rows: list, protein: str, model: str, k: int) -> list:
    """Pick k representative sequences for a protein+model (spread across PWM score range)."""
    r = [row for row in rows if row["protein_name"] == protein and row["model"] == model]
    r.sort(key=lambda x: int(float(x["n_presented_total"])))
    if len(r) <= k:
        return [row["sequence"] for row in r]
    # Evenly spaced from min to max
    indices = np.round(np.linspace(0, len(r)-1, k)).astype(int)
    return [r[i]["sequence"] for i in indices]


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    t0 = time.time()

    # Optional parametrisation: validate a specific model's designs and version outputs
    # so a new run never overwrites a frozen prior result.
    global EVAL_CSV, OUT_CSV, OUT_SUMMARY, OUT_FIG
    if getattr(args, "in_csv", None):
        EVAL_CSV = args.in_csv
    if getattr(args, "out_tag", None):
        _od = os.path.join(PF, "data", "output")
        OUT_CSV     = os.path.join(_od, f"validate_netmhciipan_{args.out_tag}.csv")
        OUT_SUMMARY = os.path.join(_od, f"validate_netmhciipan_{args.out_tag}_summary.txt")
        OUT_FIG     = os.path.join(_od, f"validate_netmhciipan_{args.out_tag}_figure.png")
    assert not os.path.exists(OUT_CSV), (
        f"Refusing to overwrite existing {OUT_CSV}; pass a distinct --out_tag."
    )

    print("Loading PWM eval results…")
    rows = []
    with open(EVAL_CSV) as f:
        rows = list(csv.DictReader(f))
    print(f"  {len(rows)} rows, {len(set(r['protein_name'] for r in rows))} proteins")

    # Select proteins and sequences.
    # --protein_list pins the EXACT sample. Without it, select_proteins() derives its
    # length quartiles from THIS csv's protein population, so two models with different
    # eval sets get different samples even though the RNG is seeded. Cross-model numbers
    # from unpinned runs are NOT comparable -- pass --protein_list to pin the sample.
    if getattr(args, "protein_list", None):
        want = [ln.strip() for ln in open(args.protein_list) if ln.strip()
                and not ln.startswith("#")]
        have = set(r["protein_name"] for r in rows)
        missing = [p for p in want if p not in have]
        assert not missing, (
            f"--protein_list asks for {len(missing)} protein(s) absent from the input CSV: "
            f"{missing[:8]}{'…' if len(missing) > 8 else ''}. "
            "A pinned comparison cannot silently drop proteins.")
        proteins = want
        print(f"\nPINNED {len(proteins)} proteins from {args.protein_list}")
    else:
        proteins = select_proteins(rows, args.n_proteins)
        print(f"\nSelected {len(proteins)} proteins (stratified by length)")
        print("  ⚠️  Sample auto-selected, NOT pinned: valid within this run (base vs\n"
              "      fine-tuned are paired), but NOT comparable to another model's run.\n"
              "      Pass --protein_list for cross-model work.", flush=True)
    import hashlib as _hl
    _pset_md5 = _hl.md5("\n".join(sorted(set(proteins))).encode()).hexdigest()[:12]
    print(f"  protein_set_md5 = {_pset_md5}  (n={len(set(proteins))})")

    seq_records = []   # list of dicts: protein, model, seq_idx, sequence, pwm_total
    for prot in proteins:
        for model in ["base", "finetuned"]:
            seqs = pick_sequences(rows, prot, model, args.k_seqs)
            # Get corresponding PWM scores
            prot_model_rows = [r for r in rows
                               if r["protein_name"]==prot and r["model"]==model]
            prot_model_rows.sort(key=lambda x: int(float(x["n_presented_total"])))
            if len(prot_model_rows) <= args.k_seqs:
                pwm_rows = prot_model_rows
            else:
                indices = np.round(np.linspace(0, len(prot_model_rows)-1,
                                               args.k_seqs)).astype(int)
                pwm_rows = [prot_model_rows[i] for i in indices]

            for i, (seq, pwm_row) in enumerate(zip(seqs, pwm_rows)):
                seq_records.append({
                    "protein_name":   prot,
                    "protein_length": int(float(pwm_row["protein_length"])),
                    "model":          model,
                    "seq_idx":        i,
                    "sequence":       seq,
                    "pwm_total":      int(float(pwm_row["n_presented_total"])),
                    **{f"pwm_{a}": int(float(pwm_row[a])) for a in ALLELES},
                })

    n_seqs = len(seq_records)
    print(f"Total sequences to evaluate: {n_seqs}  "
          f"({len(proteins)} proteins × {args.k_seqs} seqs × 2 models)")

    # Collect all unique 15-mers across all sequences
    all_kmers = set()
    for rec in seq_records:
        all_kmers.update(get_15mers(rec["sequence"]))
    all_kmers = list(all_kmers)
    print(f"\nUnique 15-mers to score: {len(all_kmers)}")

    # Run netMHCIIpan per allele
    ranks_by_allele = {}
    for allele in ALLELES:
        print(f"  Running netMHCIIpan for {allele}…", end=" ", flush=True)
        t_a = time.time()
        ranks_by_allele[allele] = run_netmhciipan(all_kmers, allele)
        print(f"{time.time()-t_a:.1f}s  "
              f"({sum(1 for v in ranks_by_allele[allele].values() if v <= THRESHOLD)} "
              f"presented out of {len(all_kmers)})")

    # Score each sequence
    print("\nScoring sequences…")
    for rec in seq_records:
        counts = count_presented(rec["sequence"], ranks_by_allele)
        rec["netmhc_total"] = counts["total"]
        for allele in ALLELES:
            rec[f"netmhc_{allele}"] = counts[allele]

    # Write CSV
    fieldnames = (
        ["protein_name", "protein_length", "model", "seq_idx",
         "pwm_total", "netmhc_total"]
        + [f"pwm_{a}" for a in ALLELES]
        + [f"netmhc_{a}" for a in ALLELES]
        + ["sequence"]
    )
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(seq_records)
    print(f"Results saved to {OUT_CSV}")

    # ── Statistics ────────────────────────────────────────────────────────────
    base_recs = [r for r in seq_records if r["model"] == "base"]
    ft_recs   = [r for r in seq_records if r["model"] == "finetuned"]

    base_netmhc = np.array([r["netmhc_total"] for r in base_recs])
    ft_netmhc   = np.array([r["netmhc_total"] for r in ft_recs])

    # Per-protein means
    base_pp = np.array([
        np.mean([r["netmhc_total"] for r in base_recs if r["protein_name"]==p])
        for p in proteins])
    ft_pp = np.array([
        np.mean([r["netmhc_total"] for r in ft_recs   if r["protein_name"]==p])
        for p in proteins])

    # ── Minimum-base guard (added 2026-08-25) ─────────────────────────────────
    # A per-protein PERCENTAGE reduction is undefined at base = 0 and dominated by
    # sampling noise just above it. The previous form, (base-ft)/(base+1e-9)*100, had
    # no guard at all: the 1e-9 only prevented a ZeroDivisionError. On the deployed
    # model's headline run this let 7m10_A through with a mean base burden of 0.1
    # windows (one presented window across ten base designs) and a reduction of
    # -300%, which on its own pulled the reported headline from 41.4% down to 37.9%
    # and inflated the sd to 39.1. Two independent draws of the SAME model on the
    # SAME 98 proteins disagreed by 3.4 pp because of that one protein; under this
    # guard they agree to 0.7 pp, inside the documented ~1.06 pp draw SE.
    # Note rank_sensitivity.py already excluded base == 0; this makes the convention
    # consistent across the codebase and tightens it from "> 0" to "> MIN_BASE".
    keep = base_pp > args.min_base
    dropped = [p for p, k in zip(proteins, keep) if not k]
    if dropped:
        print(f"  ⚠️  min-base guard: excluding {len(dropped)} protein(s) with mean base "
              f"burden <= {args.min_base} from the per-protein statistics:")
        for p, b in zip(proteins, base_pp):
            if b <= args.min_base:
                print(f"        {p}  base={b:.2f} windows")
    if keep.sum() < 2:
        raise SystemExit("min-base guard left fewer than 2 proteins; check the input.")
    proteins_kept = [p for p, k in zip(proteins, keep) if k]
    base_pp_g, ft_pp_g = base_pp[keep], ft_pp[keep]

    t_stat, p_val = stats.ttest_rel(base_pp_g, ft_pp_g)
    w_stat, w_p   = stats.wilcoxon(base_pp_g, ft_pp_g)
    reduction = (base_pp_g - ft_pp_g) / base_pp_g * 100
    # Denominator-free robustness check: total windows removed over total base windows.
    # Immune to the low-count pathology above, so a large gap between this and the
    # per-protein mean is a signal that small-denominator proteins are driving the mean.
    pooled_reduction = 100.0 * (base_pp_g.sum() - ft_pp_g.sum()) / base_pp_g.sum()

    # PWM vs netMHCIIpan correlation
    all_pwm    = np.array([r["pwm_total"]    for r in seq_records])
    all_netmhc = np.array([r["netmhc_total"] for r in seq_records])
    r_val, r_p = stats.pearsonr(all_pwm, all_netmhc)
    sp_val, sp_p = stats.spearmanr(all_pwm, all_netmhc)

    summary_lines = [
        "=" * 65,
        f"NETMHCIIPAN VALIDATION — CAPE-MPNN {args.out_tag or 'fine-tuned'} vs BASE v_48_020",
        "=" * 65,
        f"Proteins evaluated: {len(proteins)}",
        f"Sequences per model per protein: {args.k_seqs}",
        f"Total sequences: {n_seqs}",
        f"Unique 15-mers scored by netMHCIIpan: {len(all_kmers)}",
        f"Presentation threshold: %Rank_EL ≤ {THRESHOLD}%",
        "",
        "── netMHCIIpan-predicted presented windows ──────────────────",
        f"  Base model:   mean {base_netmhc.mean():.2f} ± {base_netmhc.std():.2f}"
        f"  (median {np.median(base_netmhc):.1f})",
        f"  Fine-tuned:   mean {ft_netmhc.mean():.2f} ± {ft_netmhc.std():.2f}"
        f"  (median {np.median(ft_netmhc):.1f})",
        "",
        "── Per-protein means ─────────────────────────────────────────",
        f"  Base:         {base_pp_g.mean():.2f} ± {base_pp_g.std():.2f}",
        f"  Fine-tuned:   {ft_pp_g.mean():.2f} ± {ft_pp_g.std():.2f}",
        f"  Reduction:    mean {reduction.mean():.1f}% ± {reduction.std():.1f}%"
        f"  (median {np.median(reduction):.1f}%)",
        f"  Pooled reduction (denominator-free): {pooled_reduction:.1f}%",
        f"  Improved:     {sum(ft_pp_g < base_pp_g)}/{len(proteins_kept)} proteins",
        f"  Min-base guard: base > {args.min_base} windows; "
        f"{len(dropped)} protein(s) excluded"
        + (f" ({', '.join(dropped)})" if dropped else ""),
        "",
        "── Statistical tests ─────────────────────────────────────────",
        f"  Paired t-test:    t={t_stat:.3f},  p={p_val:.4g}",
        f"  Wilcoxon:         W={w_stat:.1f}, p={w_p:.4g}",
        "",
        "── PWM vs netMHCIIpan correlation (all sequences) ───────────",
        f"  Pearson r:        {r_val:.3f}  (p={r_p:.3g})",
        f"  Spearman rho:     {sp_val:.3f} (p={sp_p:.3g})",
        "",
        f"Total runtime: {(time.time()-t0)/60:.1f} min",
        "=" * 65,
    ]
    summary = "\n".join(summary_lines)
    print("\n" + summary)
    with open(OUT_SUMMARY, "w") as f:
        f.write(summary + "\n")

    # ── Figure ────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.rcParams.update({
        "font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":8,"axes.labelsize":9,"axes.titlesize":9,
        "xtick.labelsize":7.5,"ytick.labelsize":7.5,
        "axes.linewidth":0.7,"legend.fontsize":7.5,"legend.frameon":False,
    })
    BLUE="#0072B2"; ORANGE="#E69F00"; GREEN="#009E73"

    fig = plt.figure(figsize=(11, 4.5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # A — violin comparison (netMHCIIpan)
    parts = ax1.violinplot([base_netmhc, ft_netmhc], positions=[1,2],
                            widths=0.55, showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], [BLUE, ORANGE]):
        pc.set_facecolor(col); pc.set_alpha(0.65)
    parts["cmedians"].set_colors(["white","white"]); parts["cmedians"].set_linewidth(1.5)
    rng = np.random.default_rng(0)
    for vals, x, col in [(base_netmhc,1,BLUE),(ft_netmhc,2,ORANGE)]:
        ax1.scatter(x+rng.uniform(-0.12,0.12,len(vals)), vals,
                    s=3, alpha=0.25, color=col, zorder=3)
    ax1.plot([1],[base_netmhc.mean()],"D",color="white",ms=5,zorder=5,
             markeredgecolor=BLUE,markeredgewidth=1.2)
    ax1.plot([2],[ft_netmhc.mean()],"D",color="white",ms=5,zorder=5,
             markeredgecolor=ORANGE,markeredgewidth=1.2)
    ax1.set_xticks([1,2])
    ax1.set_xticklabels(["Base\nv_48_020",f"Fine-tuned\n{args.out_tag or 'fine-tuned'}"],fontsize=7.5)
    ax1.set_ylabel("netMHCIIpan presented windows\nper designed sequence")
    ax1.set_title(f"A — netMHCIIpan distribution\n"
                  f"Reduction: {reduction.mean():.1f}% · p={p_val:.2g}",
                  fontweight="bold", loc="left")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # B — per-protein paired scatter (netMHCIIpan)
    mx = max(base_pp.max(), ft_pp.max())*1.05
    ax2.plot([0,mx],[0,mx],"--",color="grey",lw=0.8,alpha=0.6)
    ax2.scatter(base_pp, ft_pp, s=30, alpha=0.7, color=ORANGE, edgecolors="none")
    n_imp = sum(ft_pp < base_pp)
    ax2.text(0.97,0.05,f"{n_imp}/{len(proteins)} proteins improved",
             transform=ax2.transAxes,ha="right",fontsize=7,color=GREEN)
    ax2.set_xlabel("Base model (mean netMHCIIpan windows)")
    ax2.set_ylabel("Fine-tuned model")
    ax2.set_title(f"B — Per-protein comparison (n={len(proteins)})",
                  fontweight="bold",loc="left")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    # C — PWM vs netMHCIIpan scatter (all sequences, coloured by model)
    for recs, col, label in [(base_recs,BLUE,"Base"),(ft_recs,ORANGE,"Fine-tuned")]:
        pwm_v  = [r["pwm_total"]    for r in recs]
        netm_v = [r["netmhc_total"] for r in recs]
        ax3.scatter(pwm_v, netm_v, s=5, alpha=0.3, color=col, label=label, edgecolors="none")
    # overall trend
    m, b_int = np.polyfit(all_pwm, all_netmhc, 1)
    xf = np.linspace(all_pwm.min(), all_pwm.max(), 200)
    ax3.plot(xf, m*xf+b_int, color="black", lw=1.2, alpha=0.7,
             label=f"r={r_val:.2f}  ρ={sp_val:.2f}")
    ax3.set_xlabel("PWM presented windows (fast predictor)")
    ax3.set_ylabel("netMHCIIpan presented windows")
    ax3.set_title("C — PWM vs netMHCIIpan correlation", fontweight="bold", loc="left")
    ax3.legend(loc="upper left", markerscale=2)
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    fig.suptitle(
        f"netMHCIIpan Validation  —  {len(proteins)} proteins · "
        f"{args.k_seqs} seqs/model · {len(all_kmers):,} unique 15-mers scored",
        fontsize=9, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {OUT_FIG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_proteins", type=int, default=20,
                        help="Test proteins to validate (default: 20)")
    parser.add_argument("--min_base", type=float, default=1.0,
                        help="Exclude proteins whose MEAN base burden is <= this many "
                             "predicted windows from the per-protein statistics. A "
                             "percentage reduction is undefined at 0 and noise-dominated "
                             "just above it; the historical code had no guard at all and "
                             "a protein with base=0.1 contributed -300%% to the mean. "
                             "Set 0 to reproduce the old (unguarded) behaviour.")
    parser.add_argument("--k_seqs", type=int, default=5,
                        help="Sequences per model per protein (default: 5)")
    parser.add_argument("--in_csv", type=str, default=None,
                        help="deimmunisation eval CSV to validate (default: eval_deimmunisation.csv). "
                             "Use a seed-matched CSV, e.g. eval_deimmunisation_register_seed42.csv.")
    parser.add_argument("--out_tag", type=str, default=None,
                        help="suffix for output files (default: none). Versions outputs so a "
                             "new run never overwrites a prior one.")
    parser.add_argument("--protein_list", type=str, default=None,
                        help="file with one protein_name per line, pinning the exact sample. "
                             "REQUIRED whenever this run will be compared with another model's "
                             "run -- a seeded RNG is not sufficient, because the stratified draw "
                             "depends on the input CSV's own protein population.")
    args = parser.parse_args()
    main(args)
