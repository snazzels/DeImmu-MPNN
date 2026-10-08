#!/usr/bin/env python
"""Map a (beta, lr) sweep config to its trained model id by scanning the deposited
dpo_hparams.yaml files (the id is a hash of the hyperparameters, not known a priori).
Usage: PF=<project> python find_model_id.py <beta> <lr>   [CREG env overrides charge_reg_weight, default 1.0]
Prints the 8-char model id, or nothing if not found (train it first)."""
import sys, os, glob

beta, lr = float(sys.argv[1]), float(sys.argv[2])
creg = float(os.environ.get("CREG", "1.0"))
root = os.path.join(os.environ["PF"], "CAPE_MPNN", "artefacts", "CAPE-MPNN", "models")

def val(path, key):
    for l in open(path):
        s = l.strip()
        if s.startswith(key + ":"):
            return s.split(":", 1)[1].strip().strip("'\"")
    return None

for d in sorted(glob.glob(os.path.join(root, "*", "dpo_hparams.yaml"))):
    try:
        b = float(val(d, "beta")); r = float(val(d, "lr")); c = float(val(d, "charge_reg_weight") or 0)
    except (TypeError, ValueError):
        continue
    if abs(b - beta) < 1e-12 and abs(r - lr) < 1e-9 and abs(c - creg) < 1e-9:
        print(os.path.basename(os.path.dirname(d)))
        break
