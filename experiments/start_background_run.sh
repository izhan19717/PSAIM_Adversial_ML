#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${SCRIPT_DIR}/output/jobs/${STAMP}"
LOG_DIR="${RUN_DIR}/logs"
RESULTS_DIR="${RUN_DIR}/results"
FIGURES_DIR="${RUN_DIR}/figures"
TABLES_DIR="${RUN_DIR}/tables"
PID_FILE="${RUN_DIR}/run.pid"
META_FILE="${RUN_DIR}/run_meta.env"
LOG_FILE="${LOG_DIR}/run.log"

mkdir -p "${LOG_DIR}"

RUN_PYTHON="${RUN_PYTHON:-/usr/bin/python3}"

SEEDS="${SEEDS:-10}"
TRAIN_EPISODES="${TRAIN_EPISODES:-100}"
EVAL_EPISODES="${EVAL_EPISODES:-8}"
HORIZON="${HORIZON:-96}"

cat > "${META_FILE}" <<EOF
RUN_DIR='${RUN_DIR}'
LOG_FILE='${LOG_FILE}'
RESULTS_DIR='${RESULTS_DIR}'
FIGURES_DIR='${FIGURES_DIR}'
TABLES_DIR='${TABLES_DIR}'
RUN_PYTHON='${RUN_PYTHON}'
SEEDS='${SEEDS}'
TRAIN_EPISODES='${TRAIN_EPISODES}'
EVAL_EPISODES='${EVAL_EPISODES}'
HORIZON='${HORIZON}'
EOF

nohup env \
  RUN_PYTHON="${RUN_PYTHON}" \
  RESULTS_DIR="${RESULTS_DIR}" \
  FIGURES_DIR="${FIGURES_DIR}" \
  TABLES_DIR="${TABLES_DIR}" \
  bash "${SCRIPT_DIR}/run_results_pipeline.sh" \
    --seeds "${SEEDS}" \
    --train-episodes "${TRAIN_EPISODES}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --horizon "${HORIZON}" \
    > "${LOG_FILE}" 2>&1 &

RUN_PID=$!
echo "${RUN_PID}" > "${PID_FILE}"

echo "Started background experiment run."
echo "PID: ${RUN_PID}"
echo "Run dir: ${RUN_DIR}"
echo "Log: ${LOG_FILE}"
