# PSAIM Adversarial-ML Proxy Evaluation

This repository contains the code, final result artifacts, and paper-ready figures for the empirical proxy study in **Principled Learning for Service Orchestration: A Research Agenda**.

The study is deliberately scoped: it is a single-node, two-resource proxy for service orchestration, not a production-orchestrator benchmark. Its purpose is to test mechanism-level claims about RL fragility under orchestration-like perturbations and about a simplified PSAIM realisation of uncertainty-aware intrinsic motivation.

## What Is Included

- `experiments/src/`: single-node allocation environment and agents.
- `experiments/run_proxy_experiments.py`: main Section VII stress, signal, downstream, and ablation experiments.
- `experiments/run_c4_*.py`: reward-corruption, heuristic-breaking, action-distribution, and long-horizon drift extensions.
- `experiments/run_c5_mechanism_controls.py`: reviewer-requested mechanism controls.
- `experiments/run_c6_signal_switch_audit.py`: signal-switch reproducibility audit.
- `experiments/data/results/`: final CSV artifacts and run manifests used for the paper.
- `experiments/output/`: final paper tables, figures, and audit reports.
- `docs/`: reproducibility guide, results map, and archive policy.
- `paper/`: manuscript snapshot used to align the repository artifacts.

## Most Important Artifacts

- Main paper run: `experiments/data/results/paper_final_claim_aligned_v3/`
- Claim-validation probes: `experiments/data/results/claim_validation_simplified_psaim_training_behavior_v9/`
- Main paper figures: `experiments/output/figures_paper_final_claim_aligned_v3/`
- Main paper tables: `experiments/output/tables_paper_final_claim_aligned_v3/`
- Heuristic-breaking report: `experiments/output/c4_integrated_final/C4_INTEGRATED_FINAL_REPORT.md`
- Mechanism-control report: `experiments/output/c5_mechanism_controls_v1/C5_MECHANISM_CONTROL_RESULTS.md`
- Signal-switch audit: `experiments/output/c6_signal_switch_audit_v2_strict/C6_SIGNAL_SWITCH_AUDIT.md`

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Regenerate the main figures and tables from the included final CSVs:

```bash
python3 experiments/figures/generate_figures.py \
  --data-dir experiments/data/results/paper_final_claim_aligned_v3 \
  --output-dir experiments/output/figures_reproduced

python3 experiments/tables/generate_tables.py \
  --data-dir experiments/data/results/paper_final_claim_aligned_v3 \
  --output-dir experiments/output/tables_reproduced
```

Rerun the main proxy experiments:

```bash
python3 experiments/run_proxy_experiments.py \
  --experiment all \
  --seeds 10 \
  --train-episodes 100 \
  --eval-episodes 8 \
  --horizon 96 \
  --queue-capacity 12 \
  --visible-queue 5 \
  --alternating-block 24 \
  --results-dir experiments/data/results/reproduction_main
```

See `docs/REPRODUCIBILITY.md` for the full rerun protocol, including optional isolated MLflow tracking on `127.0.0.1:5001`.

## Scientific Scope

The final paper-facing claims are intentionally narrow:

- Learned RL baselines degrade under four orchestration-like perturbation classes in the proxy environment.
- Simplified PSAIM exhibits directional uncertainty-signal separation across hidden entropy regimes.
- Simplified PSAIM improves over plain DQN under the stressed downstream proxy evaluation, but does not beat SJF on the original clean/stressed evaluations.
- Under duration misreporting, where SJF's ordering assumption is intentionally violated, simplified PSAIM beats SJF on average slowdown and failure rate.
- C5 controls show the duration-misreport advantage is not reproduced by a uniform random SJF rejection prior at the same marginal rejection rate.
- C5/C6 also limit the mechanism claim: the explicit epistemic/aleatoric decomposition is not empirically load-bearing for the duration-misreport advantage in this proxy, and the strict intrinsic-reward sign switch is seed-sensitive.

For a claim-by-claim artifact map, see `docs/RESULTS_MAP.md`.
