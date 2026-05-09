#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PID_FILE="${PROJECT_ROOT}/experiments/output/mlflow/mlflow.pid"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
PORT="${MLFLOW_PORT:-5001}"

if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"

  if kill -0 "${PID}" >/dev/null 2>&1; then
    kill "${PID}"
    for _ in $(seq 1 10); do
      if ! kill -0 "${PID}" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
    rm -f "${PID_FILE}"
    echo "Stopped project-local MLflow server with PID ${PID}."
    exit 0
  fi

  rm -f "${PID_FILE}"
fi

if "${PYTHON_BIN}" - 127.0.0.1 "${PORT}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.2)
    raise SystemExit(0 if sock.connect_ex((host, port)) == 0 else 1)
PY
then
  echo "Port ${PORT} is still accepting connections, but no project-local PID file was available."
  echo "Leaving the listener untouched to avoid disturbing any unrelated service."
  exit 1
fi

echo "No project-local MLflow PID file found."
