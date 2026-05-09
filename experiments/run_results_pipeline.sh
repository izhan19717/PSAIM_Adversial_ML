#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

RUN_PYTHON="${RUN_PYTHON:-}"
if [[ -z "${RUN_PYTHON}" ]]; then
  RUN_PYTHON="/usr/bin/python3"
fi

RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/data/results/latest}"
FIGURES_DIR="${FIGURES_DIR:-${PROJECT_ROOT}/output/figures_real}"
TABLES_DIR="${TABLES_DIR:-${PROJECT_ROOT}/output/tables_real}"

"${RUN_PYTHON}" "${PROJECT_ROOT}/run_proxy_experiments.py" \
  --experiment all \
  --results-dir "${RESULTS_DIR}" \
  "$@"

"${RUN_PYTHON}" "${PROJECT_ROOT}/figures/generate_figures.py" \
  --data-dir "${RESULTS_DIR}" \
  --output-dir "${FIGURES_DIR}"

"${RUN_PYTHON}" "${PROJECT_ROOT}/tables/generate_tables.py" \
  --data-dir "${RESULTS_DIR}" \
  --output-dir "${TABLES_DIR}"
