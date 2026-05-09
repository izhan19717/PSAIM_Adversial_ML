# Results Map

This file maps the final manuscript claims to the repository artifacts that support them.

Note: directory prefixes such as `c4`, `c5`, and `c6` are internal artifact-lineage labels from successive claim-audit passes. They do not denote official reviewer rounds or externally requested reviewer responses.

## Section VII-B: Stress Fragility

Primary data:

- `experiments/data/results/paper_final_claim_aligned_v3/stress_robustness.csv`
- `experiments/output/figures_paper_final_claim_aligned_v3/fig_2_robustness_degradation.pdf`
- `experiments/output/tables_paper_final_claim_aligned_v3/table_experiment_1.md`

Claim scope: learned baselines degrade under four orchestration-like stressors in the single-node proxy; the result is not a production-systems estimate.

## Section VII-C: PSAIM Signal Behaviour

Primary data:

- `experiments/data/results/paper_final_claim_aligned_v3/psaim_signals.csv`
- `experiments/data/results/claim_validation_simplified_psaim_training_behavior_v9/training_epistemic_probes.csv`
- `experiments/data/results/claim_validation_simplified_psaim_training_behavior_v9/exploration_behavior.csv`
- `experiments/output/claim_audit/claim_audit_paper_final_claim_aligned_v3.md`
- `experiments/data/results/c6_signal_switch_audit_v2_strict/c6_seed_level_values.csv`
- `experiments/output/c6_signal_switch_audit_v2_strict/C6_SIGNAL_SWITCH_AUDIT.md`
- `experiments/output/figures_paper_final_claim_aligned_v3/fig_3_psaim_signals.pdf`

Critical interpretation: training-checkpoint probes support within-regime epistemic decrease, and behavior probes support higher rejection in high-entropy blocks plus higher deferral in low-entropy blocks. The directional separation of `V_epi`, `V_ale`, gate values, and `r_int` is supported. A robust positive-low/negative-high intrinsic-reward sign switch is not supported across independently seeded runs; the final paper should retain the softened wording.

## Section VII-D/E: Downstream Performance And Ablations

Primary data:

- `experiments/data/results/paper_final_claim_aligned_v3/downstream_performance.csv`
- `experiments/data/results/paper_final_claim_aligned_v3/adaptation_lag.csv`
- `experiments/output/tables_paper_final_claim_aligned_v3/table_experiment_2.md`
- `experiments/output/tables_paper_final_claim_aligned_v3/table_appendix_ablations.md`

Critical interpretation: simplified PSAIM improves over plain DQN under stress in the proxy, but the SJF heuristic remains the strongest method on the original clean/stressed evaluations.

## Section VII-F: Heuristic-Breaking Duration Misreport

Primary data:

- `experiments/data/results/c4_positive_demo_v1/c4_slowdown_table.csv`
- `experiments/data/results/c4_action_distribution_v2_clean_baseline/duration_misreport_action_distribution_summary.csv`
- `experiments/output/c4_integrated_final/C4_INTEGRATED_FINAL_REPORT.md`
- `experiments/output/c4_integrated_final/figures/fig_c4_2_duration_misreport_positive_demo.pdf`

Claim scope: simplified PSAIM outperforms SJF on average slowdown and failure rate when the heuristic's duration-ordering input assumption is violated. SJF remains better on p95 completion time.

## Section VII-F Controls

Primary data:

- `experiments/output/c5_mechanism_controls_v1/C5_MECHANISM_CONTROL_RESULTS.md`
- `experiments/data/results/c5_mechanism_controls_v1/c5_experiment_a_psaim_vs_sjf_r_0p17_pairs.csv`
- `experiments/data/results/c5_mechanism_controls_v1/c5_experiment_b_rawvar_pairs.csv`
- `experiments/data/results/c5_mechanism_controls_v1/c5_experiment_b_signal_switch.csv`

Critical interpretation: a uniform random SJF rejection prior at the same marginal rate does not reproduce PSAIM's duration-misreport advantage, supporting state-conditioned rejection. PSAIM-RawVar matches simplified PSAIM at the proxy scale, so the explicit epistemic/aleatoric decomposition is not shown to be load-bearing for this specific advantage.
