# Reproducibility Guide

This guide describes how to inspect the paper-facing artifacts and rerun the proxy experiments.

## Environment

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

MLflow is optional for numerical reruns but recommended for auditable experiment tracking.

## Isolated MLflow

The repository includes scripts for a project-local MLflow server on `127.0.0.1:5001`. The scripts do not use or stop any existing server on `127.0.0.1:5000`.

```bash
bash experiments/mlflow/start_local_tracking.sh
```

The server writes local state under `experiments/output/mlflow/`, which is ignored by Git. Stop only this project-local server with:

```bash
bash experiments/mlflow/stop_local_tracking.sh
```

## Included Final Artifacts

The main paper-facing run is:

```text
experiments/data/results/paper_final_claim_aligned_v3/
```

The internal follow-up and mechanism-control artifacts are:

```text
experiments/data/results/c4_positive_demo_v1/
experiments/data/results/c4_reward_drift_v2/
experiments/data/results/c4_action_distribution_v2_clean_baseline/
experiments/data/results/claim_validation_simplified_psaim_training_behavior_v9/
experiments/data/results/c5_mechanism_controls_v1/
experiments/data/results/c6_signal_switch_exact_exp2_v1/
experiments/data/results/c6_signal_switch_audit_v2_strict/
```

Paper-ready figures and tables are under:

```text
experiments/output/figures_paper_final_claim_aligned_v3/
experiments/output/tables_paper_final_claim_aligned_v3/
experiments/output/c4_integrated_final/
experiments/output/claim_audit/
experiments/output/c5_mechanism_controls_v1/
experiments/output/c6_signal_switch_audit_v2_strict/
```

## Regenerate Tables And Figures From Included CSVs

```bash
python3 experiments/figures/generate_figures.py \
  --data-dir experiments/data/results/paper_final_claim_aligned_v3 \
  --output-dir experiments/output/figures_reproduced

python3 experiments/tables/generate_tables.py \
  --data-dir experiments/data/results/paper_final_claim_aligned_v3 \
  --output-dir experiments/output/tables_reproduced

python3 experiments/figures/generate_c4_integrated_report.py
```

The figure pipeline exports PDF, SVG, and 300 dpi PNG.

## Rerun Main Experiments

The final paper protocol uses 10 seeds, 100 training episodes, 8 evaluation episodes, horizon 96, queue capacity 12, visible queue 5, and alternating hidden-regime block length 24.

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

Add `--use-mlflow` after starting the isolated server if you want run tracking.

## Rerun Reviewer-Control Experiments

```bash
python3 experiments/run_c4_experiments.py \
  --results-dir experiments/data/results/reproduction_c4_positive_demo

python3 experiments/run_c4_reward_and_drift.py \
  --results-dir experiments/data/results/reproduction_c4_reward_drift

python3 experiments/run_c4_action_distribution.py \
  --results-dir experiments/data/results/reproduction_c4_action_distribution

python3 experiments/run_c5_mechanism_controls.py \
  --results-dir experiments/data/results/reproduction_c5_mechanism_controls \
  --paper-output-dir experiments/output/reproduction_c5_mechanism_controls

python3 experiments/run_c6_signal_switch_audit.py \
  --results-dir experiments/data/results/reproduction_c6_signal_switch_audit \
  --paper-output-dir experiments/output/reproduction_c6_signal_switch_audit \
  --c4-reference-dir experiments/data/results/paper_final_claim_aligned_v3 \
  --c5-reference-dir experiments/data/results/c5_mechanism_controls_v1
```

Some reruns are compute-heavy because each reported result uses 10 seeds and paired bootstrap summaries.
