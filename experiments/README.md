# Experiments

This directory contains the reproducible proxy-study implementation used by the paper.

## Layout

- `src/`: environment, stressors, workload generator, heuristics, DQN/RND/PG baselines, and simplified PSAIM.
- `config/`: machine-readable protocol and plotting manifests.
- `data/results/`: final paper-facing CSV artifacts and run manifests.
- `figures/`: deterministic figure-generation code.
- `tables/`: deterministic table-generation code.
- `mlflow/`: isolated local MLflow helper scripts.
- `output/`: final paper-ready figures, tables, and internal claim-audit reports.

## Main Commands

Run the main Section VII proxy evaluation:

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

Generate figures and tables from an existing result directory:

```bash
python3 experiments/figures/generate_figures.py \
  --data-dir experiments/data/results/paper_final_claim_aligned_v3 \
  --output-dir experiments/output/figures_reproduced

python3 experiments/tables/generate_tables.py \
  --data-dir experiments/data/results/paper_final_claim_aligned_v3 \
  --output-dir experiments/output/tables_reproduced
```

Run the internal mechanism-control and signal-audit studies:

```bash
python3 experiments/run_c5_mechanism_controls.py \
  --results-dir experiments/data/results/reproduction_c5_mechanism_controls \
  --paper-output-dir experiments/output/reproduction_c5_mechanism_controls

python3 experiments/run_c6_signal_switch_audit.py \
  --results-dir experiments/data/results/reproduction_c6_signal_switch_audit \
  --paper-output-dir experiments/output/reproduction_c6_signal_switch_audit \
  --c4-reference-dir experiments/data/results/paper_final_claim_aligned_v3 \
  --c5-reference-dir experiments/data/results/c5_mechanism_controls_v1
```

## MLflow

MLflow is optional. If enabled, use the isolated project-local server:

```bash
bash experiments/mlflow/start_local_tracking.sh
```

The scripts bind to `127.0.0.1:5001` and write local state under `experiments/output/mlflow/`, which is ignored by Git. They do not touch any service on `127.0.0.1:5000`.
