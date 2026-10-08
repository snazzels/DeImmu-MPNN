#!/usr/bin/env python
"""Preflight: assert a training sbatch passes EVERY hashed hyperparameter explicitly.

Why this exists
---------------
Twice on 2026-08-19/20, `09_reanchor_train.sbatch` silently trained the wrong
model because it relied on an argparse default for a hashed hyperparameter:

  * `--mhc_2_predictor` was never passed -> defaulted to None -> the run
    optimized MHC class I only, the opposite of this project's subject.
  * `--mhc_1_alleles` was never passed -> defaulted to the single allele
    'HLA-A*02:01' instead of the six-allele HLA-A/B/C panel.

Neither failed loudly. A hparam that enters the model hash but is left to a
default produces a *valid-looking* model with a *plausible-looking* id, which is
the worst possible failure mode: ~18 h of GPU time, and it was only caught
because one array task happened to end up with no objective at all and tripped
an assertion.

The rule this enforces: **if a hyperparameter enters the model hash, the sbatch
must pass it explicitly.** Defaults are fine for things that do not change what
model you get; they are not fine for anything hashed.

Usage
-----
  python slurm/check_train_flags.py slurm/09_reanchor_train.sbatch [more.sbatch ...]
  python slurm/check_train_flags.py --reference f16b51e6 slurm/*.sbatch

Exit status 0 = all clear, 1 = at least one script would train an unintended
model. Run it before every `sbatch`.
"""
import os, re, sys, argparse, glob

PF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPE = os.path.join(PF, "CAPE_MPNN", "CAPE-MPNN", "cape-mpnn.py")
MODELS = os.path.join(PF, "CAPE_MPNN", "artefacts", "CAPE-MPNN", "models")

# Hashed hparams that legitimately need no explicit flag, with the reason.
# Keep this list SHORT and justified -- every entry is a place this check is blind.
EXEMPT = {
    "seed": "opt-in: absent means unseeded, which is the historical default and "
            "is dropped from the hash, so existing ids stay valid",
    "proteome_file_name": "unused in this project (no proteome-conditioned runs)",
    "preference_sampling_method": "unused; None is the only value ever trained",
}


def hashed_hparams():
    """Parse ModelManager.dpo_hparams_names out of cape-mpnn.py."""
    src = open(CAPE).read()
    m = re.search(r"ModelManager\.dpo_hparams_names\s*=\s*\[(.*?)\]", src, re.S)
    if not m:
        sys.exit(f"FATAL: could not find dpo_hparams_names in {CAPE}")
    return re.findall(r"'([^']+)'", m.group(1))


def invocation(path):
    """Return the cape-mpnn.py command as one string (following \\ continuations)."""
    lines = open(path).read().splitlines()
    # Must be an actual `python ... cape-mpnn.py` call. Matching a bare
    # "cape-mpnn.py" also hits things like the task-4 `grep -q "'seed'" ...
    # cape-mpnn.py` guard, which yields zero flags and a false alarm.
    call = re.compile(r"python\s+\S*cape-mpnn\.py")
    for i, ln in enumerate(lines):
        if call.search(ln) and not ln.strip().startswith("#"):
            buf = [ln]
            while buf[-1].rstrip().endswith("\\") and i + len(buf) < len(lines):
                buf.append(lines[i + len(buf)])
            return "\n".join(buf)
    return None


def resume_style(cmd):
    return "--cont_ckpt_id" in cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scripts", nargs="+")
    ap.add_argument("--reference", default="f16b51e6",
                    help="model id whose stored hparams are the comparison baseline")
    args = ap.parse_args()

    names = hashed_hparams()
    ref_yaml = os.path.join(MODELS, args.reference, "dpo_hparams.yaml")
    ref = {}
    if os.path.exists(ref_yaml):
        for ln in open(ref_yaml):
            if ":" in ln:
                k, v = ln.split(":", 1)
                ref[k.strip()] = v.strip().strip("'").strip('"')
    else:
        print(f"note: reference {args.reference} has no dpo_hparams.yaml; "
              f"skipping value comparison\n")

    bad = 0
    for path in args.scripts:
        cmd = invocation(path)
        name = os.path.basename(path)
        if cmd is None:
            print(f"—  {name}: no cape-mpnn.py invocation, skipped")
            continue
        if resume_style(cmd):
            print(f"—  {name}: --cont_ckpt_id resume, hparams restored from the "
                  f"checkpoint, skipped")
            continue

        passed = set(re.findall(r"--([A-Za-z0-9_]+)", cmd))
        missing = [n for n in names if n not in passed and n not in EXEMPT]
        exempt_missing = [n for n in names if n not in passed and n in EXEMPT]

        if missing:
            bad += 1
            print(f"✗  {name}: {len(missing)} HASHED hparam(s) left to an argparse "
                  f"default — this will silently train an unintended model:")
            for n in missing:
                r = f"   (deployed {args.reference} used: {ref[n]})" if n in ref else ""
                print(f"     --{n}{r}")
        else:
            print(f"✓  {name}: all {len(names) - len(exempt_missing)} hashed "
                  f"hparams passed explicitly")
        for n in exempt_missing:
            print(f"     · --{n} defaulted, exempt: {EXEMPT[n]}")

    print()
    if bad:
        print(f"FAIL: {bad} script(s) would train an unintended model. Do not submit.")
        return 1
    print("PASS: every hashed hyperparameter is set explicitly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
