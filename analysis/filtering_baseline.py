#!/usr/bin/env python
"""
Filtering baseline: generate N sequences per protein from the BASE model,
keep the top K with lowest MHC-II PWM score, and compare against the DPO
fine-tuned model's K sequences.

This directly addresses the proposal's "two strategies":
  (1) Re-training (DPO) — already evaluated in eval_deimmunisation.py
  (2) Filtering large ensembles — this script

Usage:
    python tools/filtering_baseline.py [--n_generate 100] [--k_keep 10]

Output:
    data/output/filtering_baseline.csv
    data/output/filtering_baseline_summary.txt
"""

import os, sys, argparse, csv, time
import numpy as np

PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PF, "libs"))

import kit
import kit.globals as G
from kit.data import DD, Split
import torch

G.ENV = DD()
G.ENV.PROJECT   = PF
G.ENV.INPUT     = os.path.join(PF, "data", "input")
G.ENV.ARTEFACTS = os.path.join(PF, "artefacts")
G.ENV.CONFIG    = os.path.join(PF, "configs")
G.JOB = DD(); G.JOB.ID = None
G.PROJECT_ENV = G.ENV
kit.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {kit.DEVICE}")

from CAPE.MPNN.model import CapeMPNN
from CAPE.MPNN.data.utils import get_data_loaders_pdb, get_pdbs, sample_dict_and_probs
from CAPE.MPNN.overwrite import tied_featurize
from CAPE.MPNN.data.aux import S_to_seqs
from kit.bioinf.immuno.mhc_2 import Mhc2PredictorPwm, MHC_2_PEPTIDE_LENGTHS

EVAL_CSV     = os.path.join(PF, "data", "output", "eval_deimmunisation.csv")
OUT_CSV      = os.path.join(PF, "data", "output", "filtering_baseline.csv")
OUT_SUMMARY  = os.path.join(PF, "data", "output", "filtering_baseline_summary.txt")
DATA_PATH    = os.environ.get("PDB_CACHE_DIR", "")
MHC2_ALLELES = "DRB1_0101+DRB1_0301+DRB1_0401+DRB1_0701+DRB1_1101+DRB1_1501"
PWM_DIR      = os.path.join(PF, "data", "input", "immuno", "mhc_2", "pwm")
TEMPERATURE  = 0.2
RESCUT       = 3.5
MAX_LEN      = 500


def score_sequence_pwm(seq, predictor, alleles):
    """Count presented 15-mer windows across all alleles (total)."""
    presented = predictor.seq_presented(seq, alleles=alleles,
                                        lengths=MHC_2_PEPTIDE_LENGTHS)
    return sum(1 for _ in presented)


def main(args):
    t0 = time.time()

    # ── Load DPO eval results (already have fine-tuned + base random-10) ──────
    print("Loading existing eval results…")
    eval_rows = []
    with open(EVAL_CSV) as f:
        eval_rows = list(csv.DictReader(f))
    eval_proteins = sorted(set(r["protein_name"] for r in eval_rows))
    print(f"  {len(eval_proteins)} proteins in eval set")

    # ── Load models ───────────────────────────────────────────────────────────
    CapeMPNN.base_model_pt_dir_path   = os.path.join(G.ENV.INPUT, "CAPE-MPNN", "vanilla_model_weights")
    CapeMPNN.base_model_yaml_dir_path = os.path.join(G.ENV.INPUT, "CAPE-MPNN", "base_hparams")

    print("Loading base model (v_48_020)…")
    model_base = CapeMPNN.from_file("v_48_020")

    print("Loading MHC-II PWM predictor…")
    predictor = Mhc2PredictorPwm(PWM_DIR)

    # ── Test data loader ──────────────────────────────────────────────────────
    print(f"Building test data loader…")
    dl = get_data_loaders_pdb(DATA_PATH, [Split.TEST], rescut=RESCUT, debug=False)

    # We need the same 100 proteins as the eval run.
    # Load until we've seen all of them (get_pdbs shuffles, so load a large batch).
    print(f"Loading test structures to match eval proteins…")
    pdb_dicts_all = get_pdbs(dl[Split.TEST], 1, MAX_LEN, 500)
    pdb_by_name = {d["name"]: d for d in pdb_dicts_all}
    missing = [p for p in eval_proteins if p not in pdb_by_name]
    if missing:
        print(f"  Warning: {len(missing)} eval proteins not found in 500-protein sample: {missing[:3]}…")
    pdb_dicts = [pdb_by_name[p] for p in eval_proteins if p in pdb_by_name]
    print(f"  Matched {len(pdb_dicts)}/{len(eval_proteins)} eval proteins")

    # ── Generate N sequences per protein, keep best K by PWM score ───────────
    allele_list = MHC2_ALLELES.split("+")
    rows_out = []

    print(f"\nGenerating {args.n_generate} base sequences × {len(pdb_dicts)} proteins…")
    print(f"Will keep top {args.k_keep} lowest-immunogenicity sequences per protein.")

    from tqdm import tqdm
    for pdb_dict in tqdm(pdb_dicts, desc="Proteins"):
        prot_name   = pdb_dict["name"]
        prot_length = len(pdb_dict["seq"])
        batch_structures = list(tied_featurize([pdb_dict], kit.DEVICE, None))

        # Generate N sequences
        candidates = []
        for _ in range(args.n_generate):
            sample, _ = sample_dict_and_probs(
                model_base, model_base, batch_structures, temperature=TEMPERATURE
            )
            seq = S_to_seqs(sample["S"], batch_structures[5])[0]
            pwm_score = score_sequence_pwm(seq, predictor, MHC2_ALLELES)
            candidates.append((pwm_score, seq))

        # Sort by PWM score (ascending = less immunogenic = better)
        candidates.sort(key=lambda x: x[0])
        pwm_scores = [c[0] for c in candidates]

        # Record all candidates with their rank
        for rank, (pwm_score, seq) in enumerate(candidates):
            rows_out.append({
                "protein_name":   prot_name,
                "protein_length": prot_length,
                "rank":           rank,          # 0 = lowest immunogenicity
                "n_presented_total": pwm_score,
                "sequence":       seq,
            })

    # Write full candidate CSV
    fieldnames = ["protein_name", "protein_length", "rank",
                  "n_presented_total", "sequence"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nFull candidate set saved to {OUT_CSV}")

    # ── Compute summary statistics ────────────────────────────────────────────
    # Strategy comparisons per protein:
    #   (A) DPO fine-tuned: mean of 10 fine-tuned sequences (from eval CSV)
    #   (B) Filtered base: mean of top k_keep candidates
    #   (C) Random base (k_keep seqs): mean of first k_keep candidates (worst = random)
    #   (D) Random base (all N): mean of all N candidates (baseline)

    from scipy import stats

    def get_eval_mean(protein, model):
        r = [row for row in eval_rows
             if row["protein_name"] == protein and row["model"] == model]
        return np.mean([int(x["n_presented_total"]) for x in r]) if r else np.nan

    proteins_matched = [d["name"] for d in pdb_dicts]

    dpo_pp    = np.array([get_eval_mean(p, "finetuned") for p in proteins_matched])
    base_rand_pp = np.array([get_eval_mean(p, "base")   for p in proteins_matched])

    rows_by_prot = {p: sorted([r for r in rows_out if r["protein_name"] == p],
                               key=lambda x: x["rank"])
                    for p in proteins_matched}

    filt_pp = np.array([
        np.mean([r["n_presented_total"] for r in rows_by_prot[p][:args.k_keep]])
        for p in proteins_matched])

    # t-tests (paired, per protein)
    t_dpo_filt, p_dpo_filt   = stats.ttest_rel(filt_pp, dpo_pp)
    t_filt_rand, p_filt_rand = stats.ttest_rel(base_rand_pp, filt_pp)

    def fmt_arr(a):
        return f"mean {np.nanmean(a):.2f} ± {np.nanstd(a):.2f}  (median {np.nanmedian(a):.1f})"

    summary_lines = [
        "=" * 65,
        "FILTERING BASELINE — DPO vs POST-HOC FILTERING",
        "=" * 65,
        f"Base model:         v_48_020",
        f"DPO fine-tuned:     057975d5/best.pt",
        f"Proteins:           {len(proteins_matched)}",
        f"N generated (base): {args.n_generate}",
        f"K kept (filtered):  {args.k_keep}",
        "",
        "── MHC-II presented windows (per-protein means) ─────────────",
        f"  (A) DPO fine-tuned (K={args.k_keep} random seqs):  {fmt_arr(dpo_pp)}",
        f"  (B) Filtered base  (top {args.k_keep} of {args.n_generate}):    {fmt_arr(filt_pp)}",
        f"  (C) Random base    (K={args.k_keep} random seqs):  {fmt_arr(base_rand_pp)}",
        "",
        "── Pairwise comparisons ──────────────────────────────────────",
        f"  DPO vs Filtered: t={t_dpo_filt:.3f}, p={p_dpo_filt:.4g}  "
        + ("(DPO better)" if np.nanmean(dpo_pp) < np.nanmean(filt_pp)
           else "(Filtering better)" if p_dpo_filt < 0.05 else "(no sig. difference)"),
        f"  Filtered vs Random: t={t_filt_rand:.3f}, p={p_filt_rand:.4g}  (filtering helps)",
        "",
        f"  DPO reduction vs random base:    "
        f"{(np.nanmean(base_rand_pp)-np.nanmean(dpo_pp))/np.nanmean(base_rand_pp)*100:.1f}%",
        f"  Filtered reduction vs random:    "
        f"{(np.nanmean(base_rand_pp)-np.nanmean(filt_pp))/np.nanmean(base_rand_pp)*100:.1f}%",
        "",
        f"Total runtime: {(time.time()-t0)/60:.1f} min",
        "=" * 65,
    ]
    summary = "\n".join(summary_lines)
    print("\n" + summary)
    with open(OUT_SUMMARY, "w") as f:
        f.write(summary + "\n")
    print(f"Summary saved to {OUT_SUMMARY}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_generate", type=int, default=100,
                        help="Sequences to generate per protein from base model (default: 100)")
    parser.add_argument("--k_keep", type=int, default=10,
                        help="Top sequences to keep after filtering (default: 10)")
    args = parser.parse_args()
    main(args)
