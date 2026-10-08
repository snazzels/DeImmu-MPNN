#!/usr/bin/env python
"""Stage 2 (Boltz-2 variant) of the ESMFold self-consistency benchmark (P1.3).

Folds designs with Boltz-2 instead of ESMFold2. Boltz-2 uses memory-efficient
attention and handles proteins well past ESMFold2's ~160-residue OOM wall on a
24 GB card, so this variant covers the LARGER single-chain proteins that the
ESMFold2 pass cannot. Designed sequences have no natural homologs, so every chain
is folded single-sequence (`msa: empty`) — MSA mode would be wrong here.

RUN IN THE new_boltz ENV (Boltz 2.2.1):
    micromamba run -n new_boltz python tools/selfconsistency_fold_boltz.py --outdir <dir>

All sequences are batched into one `boltz predict` call so the model loads once.
Predictions are harvested into <outdir>/preds/<design_id>.pdb — the SAME layout
selfconsistency_score.py reads — and confidence into <outdir>/fold_metrics.csv,
so Stage 3 scoring is identical regardless of which folder produced the PDBs.

Outputs
-------
<outdir>/preds/<design_id>.pdb        predicted structures (Boltz-2)
<outdir>/fold_metrics.csv             design_id, plddt, ptm, n_res, status
"""
import os, sys, csv, json, glob, shutil, subprocess, argparse, time
import pandas as pd

PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def safe(design_id):
    return design_id.replace("|", "__").replace("/", "_")


def copy_retry(src, dst, tries=5):
    """Copy to the network mount, retrying transient EPERM under write load."""
    for i in range(tries):
        try:
            shutil.copy(src, dst); return True
        except PermissionError:
            if i == tries - 1: return False
            time.sleep(2 * (i + 1))
    return False


def write_yaml(path, seq, use_msa=False):
    # Boltz-2 input. Default: msa: empty -> single sequence (correct for de novo
    # designs, which have no homologs). use_msa=True omits that line so Boltz fetches
    # an MSA via --use_msa_server (only legitimate for the native-sequence control).
    msa_line = "" if use_msa else "      msa: empty\n"
    with open(path, "w") as f:
        f.write("version: 1\nsequences:\n  - protein:\n      id: A\n"
                f"      sequence: {seq}\n" + msa_line)


def find_pred_pdb(boltz_out, stem):
    pats = [
        os.path.join(boltz_out, "predictions", stem, f"{stem}_model_0.pdb"),
        os.path.join(boltz_out, "predictions", stem, f"{stem}_model_0.cif"),
    ]
    for p in pats:
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(boltz_out, "predictions", stem, f"{stem}_model_0.*"))
    return hits[0] if hits else None


def find_conf(boltz_out, stem):
    p = os.path.join(boltz_out, "predictions", stem, f"confidence_{stem}_model_0.json")
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            return {}
    return {}


def main(a):
    outdir = a.outdir
    designs = pd.read_csv(os.path.join(outdir, "designs.csv"))
    if a.limit:
        designs = designs.head(a.limit)
    # Strided shard: parallelise folding of one config across many GPUs/jobs. Each shard
    # folds a disjoint slice of designs.csv into the SAME shared preds/ dir (disjoint stems,
    # resume-skip prevents collisions), and writes its OWN fold_metrics file so concurrent
    # shards never clobber each other. Stage-3 score merges the shard metrics.
    if a.nshards > 1:
        designs = designs.iloc[a.shard::a.nshards].reset_index(drop=True)
        print(f"shard {a.shard}/{a.nshards}: folding {len(designs)} of the config's designs", flush=True)
    pred_dir = os.path.join(outdir, "preds"); os.makedirs(pred_dir, exist_ok=True)
    # Boltz writes many intermediate files; keep that heavy I/O on LOCAL disk. The
    # network mount this was developed against intermittently returned EPERM under
    # heavy write load, which crashed an earlier run mid-fold. Only the final PDBs
    # land on the mount. Set --workroot to a local scratch directory.
    workroot = a.workroot
    in_dir = os.path.join(workroot, "boltz_inputs"); os.makedirs(in_dir, exist_ok=True)
    work = os.path.join(workroot, "boltz_work"); os.makedirs(work, exist_ok=True)

    stem_to_id = {}
    n_written = 0
    for row in designs.itertuples(index=False):
        stem = safe(row.design_id)
        stem_to_id[stem] = (row.design_id, row.protein_name, row.model, row.seq_idx, len(row.sequence))
        if os.path.exists(os.path.join(pred_dir, stem + ".pdb")) and not a.override:
            continue  # resume
        write_yaml(os.path.join(in_dir, stem + ".yaml"), row.sequence, use_msa=a.use_msa)
        n_written += 1
    print(f"{len(stem_to_id)} designs; {n_written} to fold (rest already have PDBs)", flush=True)

    if n_written:
        cmd = ["boltz", "predict", in_dir, "--out_dir", work,
               "--output_format", "pdb", "--override",
               "--recycling_steps", str(a.recycling), "--diffusion_samples", "1",
               "--sampling_steps", str(a.sampling_steps), "--devices", "1"]
        if a.use_msa:
            cmd += ["--use_msa_server"]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    boltz_out = os.path.join(work, f"boltz_results_{os.path.basename(in_dir)}")
    fields = ["design_id", "protein_name", "model", "seq_idx", "plddt", "ptm", "n_res", "status"]
    metrics_name = "fold_metrics.csv" if a.nshards == 1 else f"fold_metrics_shard{a.shard}of{a.nshards}.csv"
    with open(os.path.join(outdir, metrics_name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for stem, (did, prot, mdl, sidx, n) in stem_to_id.items():
            dst = os.path.join(pred_dir, stem + ".pdb")
            rec = dict(design_id=did, protein_name=prot, model=mdl, seq_idx=sidx,
                       plddt="", ptm="", n_res=n, status="ok")
            if not os.path.exists(dst):
                src = find_pred_pdb(boltz_out, stem)
                if src and src.endswith(".pdb"):
                    if not copy_retry(src, dst):
                        rec["status"] = "copy_eperm"
                elif src:
                    rec["status"] = "cif_only"  # would need conversion
                else:
                    rec["status"] = "no_pred"
            conf = find_conf(boltz_out, stem)
            if conf:
                rec["plddt"] = round(float(conf.get("complex_plddt", conf.get("plddt", 0)) or 0), 4)
                rec["ptm"]   = round(float(conf.get("ptm", 0) or 0), 4)
            if not os.path.exists(dst) and rec["status"] == "ok":
                rec["status"] = "no_pred"
            w.writerow(rec)
    print(f"Harvested predictions into {pred_dir} and fold_metrics.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=os.path.join(PF, "data", "output", "selfconsistency_boltz"))
    p.add_argument("--workroot", default="/tmp/boltz_selfconsistency_work",
                   help="LOCAL disk for Boltz intermediates (mount EPERMs under write load)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--recycling", type=int, default=3)
    p.add_argument("--sampling_steps", type=int, default=200)
    p.add_argument("--override", action="store_true")
    p.add_argument("--use_msa", action="store_true",
                   help="fetch an MSA via --use_msa_server (native control only; NOT for designs)")
    p.add_argument("--nshards", type=int, default=1,
                   help="split designs.csv into N strided shards to parallelise folding across jobs")
    p.add_argument("--shard", type=int, default=0, help="0-based shard index in [0, nshards)")
    main(p.parse_args())
