# Analysis — main-text evaluation and Supplementary S1

Each script reproduces one result. They are released as run for the paper and
carry absolute paths / allele lists near the top; adapt those to your checkout.
All burden scoring uses the six-allele HLA-DRB1 panel
(`DRB1_0101/0301/0401/0701/1101/1501`).

| Script | Result |
|---|---|
| `evaluate_deimmunisation.py` | Headline MHC-II burden reduction on 100 held-out proteins (39.0%). |
| `evaluate_mhc1_arm.py` | The MHC-I arm of the joint objective. |
| `validate_with_netmhciipan.py` | Re-score a subset with netMHCIIpan itself (not the training PWM) to confirm the PWM surrogate. |
| `filtering_baseline.py` | Strategy 2: post-hoc filtering of base-model ensembles (the 24% baseline the model beats). |
| `selfconsistency_generate.py` / `selfconsistency_fold.py` / `selfconsistency_fold_boltz.py` / `selfconsistency_score.py` | Structure fidelity: ESMFold/Boltz-2 self-consistency TM-score, base vs fine-tuned. |
| `test_allele_generalisation.py` / `analyse_transfer_mechanism.py` | Transfer to held-out DRB1 alleles, DRB3/4/5, HLA-DQ (and the DP exception); DR-positional vs DQ-compositional mechanism. |
| `quantify_charge_shift.py` / `test_charge_regulariser.py` | The electrostatic-shortcut control and the λ sweep. |
| `build_register_pwm.py` | Register-anchored PWM (a sharper netMHCIIpan approximation; validated, not retrained into production). |

## `supplementary_iedb/` — the negative result (Supplementary S1)

These establish the paper's central caveat: predicted presentation is not yet a
validated proxy for measured immunogenicity in the deployment regime.

| Script | Result |
|---|---|
| `analyse_iedb_binding_signal.py` | netMHCIIpan presentation vs measured IEDB T-cell outcomes: AUC 0.492 (chance) in the deployment regime. |
| `train_iedb_classifier.py` | Classifier on IEDB T-cell assay labels: strong in-distribution, collapses to AUC ≈ 0.52 under held-out-organism split (a shortcut, not immunogenicity). |
| `validate_iedb_knn_signal.py` | A non-classifier k-NN signal, gated by the same organism-holdout test — collapses identically. The Phase-0 gate that stopped the three-arm IEDB ablation. |
