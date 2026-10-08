#!/usr/bin/env python
"""
Evaluate de-immunisation: compare MHC-II immunogenicity of sequences
designed by the base ProteinMPNN (v_48_020) vs the DPO fine-tuned
CAPE-MPNN (model 057975d5, best.pt).

For N test proteins, generates K sequences per model, scores each
sequence for the number of presented 15-mer windows per HLA-DR allele,
and saves per-sequence results to a CSV.

Usage:
    python tools/evaluate_deimmunisation.py [--n_proteins 100] [--k_seqs 10]

Output:
    data/output/eval_deimmunisation.csv
    data/output/eval_deimmunisation_summary.txt
"""

import os, sys, argparse, csv, time
import numpy as np
import torch
from tqdm import tqdm

# ── Environment setup ─────────────────────────────────────────────────────────
PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # CAPE_MPNN/
sys.path.insert(0, os.path.join(PF, "libs"))

import kit
import kit.globals as G
from kit.data import DD, Split

G.ENV = DD()
G.ENV.PROJECT   = PF
G.ENV.INPUT     = os.path.join(PF, "data", "input")
G.ENV.ARTEFACTS = os.path.join(PF, "artefacts")
G.ENV.CONFIG    = os.path.join(PF, "configs")
G.JOB           = DD()
G.JOB.ID        = None
G.PROJECT_ENV   = G.ENV

kit.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {kit.DEVICE}")

from CAPE.MPNN.model import CapeMPNN
from CAPE.MPNN.data.utils import get_data_loaders_pdb, get_pdbs, sample_dict_and_probs
from CAPE.MPNN.overwrite import tied_featurize
from CAPE.MPNN.data.aux import S_to_seqs
from kit.bioinf.immuno.mhc_2 import Mhc2PredictorPwm, MHC_2_PEPTIDE_LENGTHS

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL_HASH = "057975d5"
DEFAULT_CKPT       = "best"
BASE_MODEL   = "v_48_020"
# PDB dataset location. Mirror cape-mpnn.py's resolution chain: honour a node-local
# stage via PDB_CACHE_DIR if present, else the network-mount copy under data/input,
# else the legacy local path. (Hardcoding only the local path crashed the sweep eval
# on the cluster, 2026-08-18 — that path exists only on the local machine.)
DATA_PATH    = next(
    (p for p in (os.environ.get("PDB_CACHE_DIR"),
                 os.path.join(PF, "data", "input", "CAPE-MPNN", "pdb_2021aug02"),
                 os.environ.get("PDB_CACHE_DIR", "")) if p and os.path.isdir(p)),
    os.path.join(PF, "data", "input", "CAPE-MPNN", "pdb_2021aug02"))
MHC2_ALLELES = "DRB1_0101+DRB1_0301+DRB1_0401+DRB1_0701+DRB1_1101+DRB1_1501"
PWM_DIR      = os.path.join(PF, "data", "input", "immuno", "mhc_2", "pwm")
OUT_DIR      = os.path.join(PF, "data", "output")
TEMPERATURE  = 0.2
RESCUT       = 3.5
MAX_LEN      = 500   # skip proteins longer than this to keep eval manageable


def seq_recovery(designed: str, native: str) -> float:
    """Fraction of positions where the designed residue matches the native.

    This is the quality axis that pairs with the epitope count: reduction rises with
    how far the model drifts from base ProteinMPNN, so a reduction reported without a
    fidelity number alongside it cannot distinguish epitope-specific editing from
    damage. Computed here rather than by seq_recovery_rerun.py because that tool draws
    its own protein sample, which would reintroduce the cross-checkpoint sampling
    mismatch that --protein_list exists to remove.

    Chain separators are stripped from both. The two are the same length by
    construction (the design is generated for this exact backbone), so a mismatch
    means something upstream changed and is worth failing loudly on.
    """
    d = designed.replace("/", "")
    n = native.replace("/", "")
    if len(d) != len(n):
        raise ValueError(
            f"design/native length mismatch: {len(d)} vs {len(n)}"
        )
    if not n:
        return float("nan")
    return sum(a == b for a, b in zip(d, n)) / len(n)


def score_sequence(seq: str, predictor: Mhc2PredictorPwm,
                   alleles: str, lengths: list) -> dict:
    """Return presented window counts: total and per-allele."""
    presented = predictor.seq_presented(seq, alleles=alleles, lengths=lengths)
    # presented is a list of (allele, position, score) tuples for windows above threshold
    allele_list = alleles.split("+")
    per_allele = {a: 0 for a in allele_list}
    for _peptide, allele, _rank, _pos in presented:
        per_allele[allele] = per_allele.get(allele, 0) + 1
    total = sum(per_allele.values())
    return {"total": total, **per_allele}


def main(args):
    t0 = time.time()

    MODEL_HASH = args.model_hash
    CKPT       = args.ckpt
    tag        = args.out_tag if args.out_tag else MODEL_HASH
    OUT_CSV     = os.path.join(OUT_DIR, f"eval_deimmunisation_{tag}.csv")
    OUT_SUMMARY = os.path.join(OUT_DIR, f"eval_deimmunisation_{tag}_summary.txt")
    assert not os.path.exists(OUT_CSV), (
        f"Refusing to overwrite existing {OUT_CSV}; pass a distinct --out_tag."
    )
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    print(f"Model hash: {MODEL_HASH}/{CKPT} | out tag: {tag} | seed: {args.seed}")

    # ── Load models ───────────────────────────────────────────────────────────
    CapeMPNN.base_model_pt_dir_path   = os.path.join(G.ENV.INPUT, "CAPE-MPNN", "vanilla_model_weights")
    CapeMPNN.base_model_yaml_dir_path = os.path.join(G.ENV.INPUT, "CAPE-MPNN", "base_hparams")

    if args.reuse_base_csv:
        print(f"Base arm: REUSING {args.reuse_base_csv} (not re-generating)")
        model_base = None
    else:
        print("Loading base model (v_48_020)…")
        model_base = CapeMPNN.from_file(BASE_MODEL)

    print(f"Loading fine-tuned model ({MODEL_HASH}/{CKPT}.pt)…")
    ft_path = os.path.join(PF, "artefacts", "CAPE-MPNN", "models",
                           MODEL_HASH, "ckpts", f"{CKPT}.pt")
    model_ft = CapeMPNN.from_file(ft_path)

    # ── MHC-II predictor ──────────────────────────────────────────────────────
    print("Loading MHC-II PWM predictor…")
    predictor = Mhc2PredictorPwm(PWM_DIR)

    # ── Test data ─────────────────────────────────────────────────────────────
    if args.protein_list:
        # Pin the evaluation to an EXPLICIT protein set, bypassing the sampler.
        #
        # Why. The test Dataset holds one entry per sequence CLUSTER and, per its
        # own docstring, "take this position's cluster and sample a random chain
        # from it"; its DataLoader adds shuffle=True and num_workers=4. So the
        # evaluated protein set is not reproducible, and --seed does not fix it
        # (seeding torch, numpy AND python random all fail to pin it). Empirically
        # all 15 configs of 02_sweep_eval.sbatch ran with --seed 42 and still drew
        # 15 different samples: pairwise overlap 33-45/100 by chain, 40-79/100 even
        # by cluster. The lambda=0 and lambda=1.0 arms of Table tab:sweep share
        # only 19 of 100 chains, so their reductions are not comparable arm-to-arm.
        #
        # How. loader_pdb() takes a "pdbid_chain" string directly and every chain
        # has its own file (pdb/<id[1:3]>/<id>_<chain>.pt), so the requested chains
        # are fed to the real get_pdbs() through a one-item-per-chain iterable.
        # get_pdbs is reused rather than reimplemented so the resulting dicts are
        # byte-for-byte what the sampled path produces. Filtering a sampled pool
        # cannot work instead: one pass yields only ~205 chains out of ~1553 test
        # clusters, and which ones varies per run.
        from CAPE.MPNN.ProteinMPNN.training.utils import loader_pdb

        want = [ln.strip() for ln in open(args.protein_list)
                if ln.strip() and not ln.lstrip().startswith("#")]
        print(f"Protein list: {len(want)} names from {args.protein_list}")
        params = {"LIST": f"{DATA_PATH}/list.csv",
                  "VAL": f"{DATA_PATH}/valid_clusters.txt",
                  "TEST": f"{DATA_PATH}/test_clusters.txt",
                  "DIR": f"{DATA_PATH}",
                  "DATCUT": "2030-Jan-01", "RESCUT": RESCUT, "HOMO": 0.70}

        class _PinnedLoader:
            """Yields one batch-of-1 per requested chain, in list order."""
            def __iter__(self):
                for name in want:
                    out = loader_pdb([name, ""], params)
                    yield {k: [v] for k, v in out.items()}

        pdb_dicts = get_pdbs(_PinnedLoader(), 1, MAX_LEN, len(want))
        by_name = {d["name"]: d for d in pdb_dicts}
        missing = [n for n in want if n not in by_name]
        print(f"  loaded {len(pdb_dicts)}/{len(want)} requested chains")
        if missing:
            # Hard failure, not a warning: evaluating 97 of 100 requested proteins
            # and reporting it as "the matched set" is the exact silent mismatch
            # this option exists to eliminate.
            sys.exit(f"FATAL: {len(missing)} requested chains could not be loaded "
                     f"(missing file, or longer than MAX_LEN={MAX_LEN}): "
                     f"{', '.join(missing[:10])}" + (" ..." if len(missing) > 10 else ""))
        pdb_dicts = [by_name[n] for n in want]     # deterministic order
    else:
        print(f"Building test data loader (PDB: {DATA_PATH})…")
        dl = get_data_loaders_pdb(DATA_PATH, [Split.TEST], rescut=RESCUT, debug=False)
        print(f"  {len(dl[Split.TEST].dataset)} test clusters available")
        print(f"Loading {args.n_proteins} test structures (max length {MAX_LEN})…")
        pdb_dicts = get_pdbs(dl[Split.TEST], 1, MAX_LEN, args.n_proteins)
        print(f"  Got {len(pdb_dicts)} structures after length filtering")

    # ── CSV header ────────────────────────────────────────────────────────────
    allele_list = MHC2_ALLELES.split("+")
    fieldnames = (
        ["protein_name", "protein_length", "model", "seq_idx", "sequence",
         "n_presented_total", "recovery"] + allele_list
    )
    rows = []

    # ── Cached base arm ───────────────────────────────────────────────────────
    # The base (v_48_020) designs do not depend on which fine-tuned checkpoint is
    # being evaluated, so a sweep over many checkpoints regenerates identical work
    # every time. Reusing one base pass halves the runtime and, more importantly,
    # pins ONE base reference: because generation is stochastic even under --seed
    # (see the --protein_list note above), independently-generated base arms differ
    # by ~0.25 recovery points run to run, which is the same order as the
    # differences between adjacent checkpoints on the reward plateau.
    cached_base = {}
    if args.reuse_base_csv:
        if not os.path.exists(args.reuse_base_csv):
            sys.exit(f"FATAL: --reuse_base_csv not found: {args.reuse_base_csv}")
        with open(args.reuse_base_csv) as f:
            for r in csv.DictReader(f):
                if r.get("model") != "base":
                    continue
                if "recovery" not in r or r["recovery"] in ("", None):
                    sys.exit("FATAL: --reuse_base_csv has no 'recovery' column; it predates "
                             "the recovery patch and cannot supply the fidelity axis.")
                # Validate that every numeric field parses, but WRITE BACK THE ORIGINAL
                # STRINGS. Casting here would round-trip integer window counts through
                # float and emit "119.0" where the source (and the freshly generated
                # finetuned rows) have "119" — numerically identical but a different
                # string, which breaks any consumer doing int() on the field.
                # validate_with_netmhciipan.py did exactly that in 5 places.
                for _col in ["protein_length", "seq_idx", "n_presented_total",
                             "recovery"] + allele_list:
                    if _col not in r:
                        sys.exit(f"FATAL: --reuse_base_csv lacks column {_col}; it was "
                                 f"written by an older schema or a different allele panel.")
                    try:
                        float(r[_col])
                    except (TypeError, ValueError):
                        sys.exit(f"FATAL: --reuse_base_csv has non-numeric {_col}="
                                 f"{r[_col]!r} for {r.get('protein_name')}")
                # Keep only the declared fieldnames: an extra column in the cached CSV
                # would make DictWriter raise at the very end, after all the GPU work.
                cached_base.setdefault(r["protein_name"], []).append(
                    {k: r[k] for k in fieldnames})
        # Hard validation, same philosophy as --protein_list: a partial match must
        # never be silently reported as a matched comparison.
        want_names = [d["name"] for d in pdb_dicts]
        missing = [n for n in want_names if n not in cached_base]
        extra   = [n for n in cached_base if n not in want_names]
        if missing or extra:
            sys.exit(f"FATAL: --reuse_base_csv protein set does not match this run "
                     f"({len(missing)} missing, {len(extra)} extra). "
                     f"missing: {', '.join(missing[:5])}  extra: {', '.join(extra[:5])}")
        bad_k = {n: len(v) for n, v in cached_base.items() if len(v) != args.k_seqs}
        if bad_k:
            sys.exit(f"FATAL: --reuse_base_csv has a different sequence count than "
                     f"--k_seqs={args.k_seqs} for {len(bad_k)} proteins, e.g. "
                     f"{list(bad_k.items())[:5]}")
        print(f"  cached base arm OK: {len(cached_base)} proteins × {args.k_seqs} seqs")

    # ── Main evaluation loop ──────────────────────────────────────────────────
    n_arms = 1 if args.reuse_base_csv else 2
    print(f"\nGenerating {args.k_seqs} sequences × {n_arms} model(s) × {len(pdb_dicts)} proteins…")
    for prot_idx, pdb_dict in enumerate(tqdm(pdb_dicts, desc="Proteins")):
        prot_name   = pdb_dict["name"]
        prot_seq    = pdb_dict["seq"]
        prot_length = len(prot_seq)

        # tied_featurize expects a list, even for a single protein
        batch = [pdb_dict]
        batch_structures = list(tied_featurize(batch, kit.DEVICE, None))

        if args.reuse_base_csv:
            rows.extend(cached_base[prot_name])
            arms = [("finetuned", model_ft)]
        else:
            arms = [("base", model_base), ("finetuned", model_ft)]

        for model_name, model in arms:
            for seq_idx in range(args.k_seqs):
                sample, _ = sample_dict_and_probs(
                    model, model, batch_structures, temperature=TEMPERATURE
                )
                # S_to_seqs returns a list of strings, one per protein in batch
                seqs = S_to_seqs(sample["S"], batch_structures[5])
                seq = seqs[0]   # batch size = 1

                scores = score_sequence(
                    seq, predictor, MHC2_ALLELES, MHC_2_PEPTIDE_LENGTHS
                )
                row = {
                    "protein_name":     prot_name,
                    "protein_length":   prot_length,
                    "model":            model_name,
                    "seq_idx":          seq_idx,
                    "sequence":         seq,
                    "n_presented_total": scores["total"],
                    "recovery":         round(seq_recovery(seq, prot_seq), 6),
                }
                for allele in allele_list:
                    row[allele] = scores[allele]
                rows.append(row)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {OUT_CSV}")

    # ── Summary ───────────────────────────────────────────────────────────────
    # NOTE: rows coming from --reuse_base_csv hold the ORIGINAL STRINGS (so the
    # written CSV stays byte-identical to its source), while freshly generated rows
    # hold numbers. Every numeric read below must therefore coerce explicitly --
    # np.mean() on a string array raises _UFuncNoLoopError and, under `set -e` in
    # the sbatch, killed 8 tasks of job 514056 AFTER they had written valid CSVs.
    base_rows = [r for r in rows if r["model"] == "base"]
    ft_rows   = [r for r in rows if r["model"] == "finetuned"]

    base_total = np.array([float(r["n_presented_total"]) for r in base_rows])
    ft_total   = np.array([float(r["n_presented_total"]) for r in ft_rows])

    # Per-protein means (average over K sequences per protein)
    proteins = sorted(set(r["protein_name"] for r in rows))
    base_per_prot = np.array([
        np.mean([float(r["n_presented_total"]) for r in base_rows if r["protein_name"] == p])
        for p in proteins
    ])
    ft_per_prot = np.array([
        np.mean([float(r["n_presented_total"]) for r in ft_rows if r["protein_name"] == p])
        for p in proteins
    ])

    # Paired t-test
    from scipy import stats
    t_stat, p_val = stats.ttest_rel(base_per_prot, ft_per_prot)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(base_per_prot, ft_per_prot)

    reduction_pct = (base_per_prot - ft_per_prot) / (base_per_prot + 1e-9) * 100

    summary_lines = [
        "=" * 60,
        "DE-IMMUNISATION EVALUATION — CAPE-MPNN vs BASE ProteinMPNN",
        "=" * 60,
        f"Fine-tuned model: {MODEL_HASH}/{CKPT}.pt",
        f"Base model: {BASE_MODEL}",
        f"Test proteins: {len(proteins)}",
        f"Sequences per model per protein: {args.k_seqs}",
        f"MHC-II alleles: {MHC2_ALLELES}",
        "",
        "── MHC-II presented windows per designed sequence ──────────",
        f"  Base model:      mean {base_total.mean():.2f} ± {base_total.std():.2f}  (median {np.median(base_total):.1f})",
        f"  Fine-tuned:      mean {ft_total.mean():.2f} ± {ft_total.std():.2f}  (median {np.median(ft_total):.1f})",
        "",
        "── Per-protein means (averaged over K sequences) ───────────",
        f"  Base model:      mean {base_per_prot.mean():.2f} ± {base_per_prot.std():.2f}",
        f"  Fine-tuned:      mean {ft_per_prot.mean():.2f} ± {ft_per_prot.std():.2f}",
        f"  Reduction:       mean {reduction_pct.mean():.1f}% ± {reduction_pct.std():.1f}%",
        f"                   median {np.median(reduction_pct):.1f}%",
        "",
        "",
        "── Sequence recovery vs native (quality axis) ──────────────",
        f"  Base model:      mean {np.mean([float(r['recovery']) for r in base_rows])*100:.2f}%",
        f"  Fine-tuned:      mean {np.mean([float(r['recovery']) for r in ft_rows])*100:.2f}%",
        f"  Cost:            {(np.mean([float(r['recovery']) for r in base_rows]) - np.mean([float(r['recovery']) for r in ft_rows]))*100:+.2f} points",
        "",
        "── Statistical tests ────────────────────────────────────────",
        f"  Paired t-test:   t={t_stat:.3f}, p={p_val:.4g}",
        f"  Wilcoxon:        W={wilcoxon_stat:.1f}, p={wilcoxon_p:.4g}",
        "",
        f"Total runtime: {(time.time()-t0)/60:.1f} min",
        "=" * 60,
    ]

    summary = "\n".join(summary_lines)
    print("\n" + summary)

    with open(OUT_SUMMARY, "w") as f:
        f.write(summary + "\n")
    print(f"Summary saved to {OUT_SUMMARY}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate de-immunisation of CAPE-MPNN")
    parser.add_argument("--n_proteins", type=int, default=100,
                        help="Number of test proteins to evaluate (default: 100)")
    parser.add_argument("--k_seqs",     type=int, default=10,
                        help="Sequences per model per protein (default: 10)")
    parser.add_argument("--model_hash", type=str, default=DEFAULT_MODEL_HASH,
                        help="Fine-tuned model hash dir under artefacts/CAPE-MPNN/models")
    parser.add_argument("--ckpt",       type=str, default=DEFAULT_CKPT,
                        help="Checkpoint name in that dir's ckpts/ (default: best)")
    parser.add_argument("--out_tag",    type=str, default=None,
                        help="Suffix for output files (default: the model hash). "
                             "Keeps versions from overwriting each other.")
    parser.add_argument("--seed",       type=int, default=None,
                        help="Torch/NumPy seed for reproducible sampling (default: unseeded)")
    parser.add_argument("--protein_list", type=str, default=None,
                        help="File with one protein name per line. Pins the evaluation "
                             "to exactly this set so different models are compared on "
                             "the same proteins; aborts if any are missing from the pool.")
    parser.add_argument("--pool",       type=int, default=600,
                        help="Structures to load before selecting --protein_list "
                             "(default 600; the test pool holds >=426)")
    parser.add_argument("--reuse_base_csv", type=str, default=None,
                        help="Take the base-model (v_48_020) rows from this existing eval "
                             "CSV instead of re-generating them. The base arm is half the "
                             "work of every run and is identical across checkpoints, so a "
                             "multi-checkpoint sweep runs ~2x faster AND every point shares "
                             "ONE base reference — removing the base-resampling noise "
                             "(observed spread 0.25 recovery points across 8 runs) that "
                             "otherwise limits comparisons between nearby checkpoints. "
                             "Aborts unless the cached protein set and per-protein sequence "
                             "count match this run exactly.")
    args = parser.parse_args()
    main(args)
