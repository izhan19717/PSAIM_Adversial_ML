#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS_DIR="${SCRIPT_DIR}/output/jobs"

LATEST_META="$(find "${JOBS_DIR}" -mindepth 2 -maxdepth 2 -name run_meta.env | sort | tail -n 1 || true)"
if [[ -z "${LATEST_META}" ]]; then
  echo "No background experiment run metadata found."
  exit 0
fi

# shellcheck disable=SC1090
source "${LATEST_META}"
PID_FILE="${RUN_DIR}/run.pid"

echo "Run dir: ${RUN_DIR}"
echo "Results: ${RESULTS_DIR}"
echo "Log: ${LOG_FILE}"

if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"
  if kill -0 "${PID}" >/dev/null 2>&1; then
    echo "Status: running (PID ${PID})"
    ps -p "${PID}" -o pid,%cpu,etime,cmd
  else
    echo "Status: not running"
  fi
else
  echo "Status: PID file missing"
fi

echo
echo "Recent log lines:"
tail -n 20 "${LOG_FILE}" || true
