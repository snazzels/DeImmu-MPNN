#!/bin/bash
# Chain the 3-phase Boltz-2 full-length self-consistency for ALL 15 sweep configs:
# generate(15) -> fold(15x8=120 shards) -> score(15). Resume-safe: configs already
# done (designs/preds/summary present) are skipped at every stage, so this is the
# right command whether starting fresh or topping up after the first 5.
# Run from $PF/slurm on the cluster login node. Prints the three job ids.
#   bash submit_boltz_sc.sh
# Scale knobs pass straight through, e.g. K_SEQS=2 N_PROTEINS=24 bash submit_boltz_sc.sh
# To fold ONLY the 10 not-yet-done configs (skip the trivial no-op tasks for the first 5),
# submit by hand instead: 07a --array=6-15 ; 07b --array=41-120 (afterok) ; 07c --array=6-15 (afterany).
set -euo pipefail
cd "$(dirname "$0")"

JA=$(sbatch --parsable 07a_boltz_sc_generate.sbatch)
echo "generate (07a) = $JA"
JB=$(sbatch --parsable --dependency=afterok:$JA 07b_boltz_sc_fold.sbatch)
echo "fold     (07b) = $JB   (afterok:$JA)"
# afterANY (not afterok): 07c self-heals any fold shards that were preempted, so a lost
# shard can't strand scoring. 07c folds only the missing designs (resume-safe) then scores.
JC=$(sbatch --parsable --dependency=afterany:$JB 07c_boltz_sc_score.sbatch)
echo "score    (07c) = $JC   (afterany:$JB, self-healing)"
echo
echo "when 07c finishes:  PF=\$PF python 08_boltz_selfconsist_collate.py  -> sweep_boltz_selfconsist_table.csv"
