#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/experiments/output/mlflow"
BACKEND_DIR="${OUTPUT_DIR}/backend"
ARTIFACT_DIR="${OUTPUT_DIR}/artifacts"
LOG_DIR="${OUTPUT_DIR}/logs"
PID_FILE="${OUTPUT_DIR}/mlflow.pid"

HOST="${MLFLOW_HOST:-127.0.0.1}"
PORT="${MLFLOW_PORT:-5001}"
WORKERS="${MLFLOW_WORKERS:-1}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

mkdir -p "${BACKEND_DIR}" "${ARTIFACT_DIR}" "${LOG_DIR}"

if "${PYTHON_BIN}" - "${HOST}" "${PORT}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.2)
    raise SystemExit(0 if sock.connect_ex((host, port)) == 0 else 1)
PY
then
  echo "Refusing to start: ${HOST}:${PORT} is already in use."
  exit 1
fi

if "${PYTHON_BIN}" - 127.0.0.1 5000 <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.2)
    raise SystemExit(0 if sock.connect_ex((host, port)) == 0 else 1)
PY
then
  echo "Detected an existing local service on 127.0.0.1:5000. It will not be touched."
fi

if [[ -n "${MLFLOW_BIN:-}" ]]; then
  MLFLOW_CMD=("${MLFLOW_BIN}")
elif command -v mlflow >/dev/null 2>&1; then
  MLFLOW_CMD=("$(command -v mlflow)")
elif "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("mlflow") else 1)
PY
then
  MLFLOW_CMD=("${PYTHON_BIN}" "-m" "mlflow")
else
  echo "MLflow is not available in the current environment."
  echo "Set MLFLOW_BIN=/path/to/mlflow or run this script from an environment where mlflow is installed."
  exit 1
fi

DB_PATH="${BACKEND_DIR}/mlflow.db"
mapfile -t URI_LINES < <("${PYTHON_BIN}" - "${DB_PATH}" "${ARTIFACT_DIR}" <<'PY'
from pathlib import Path
from urllib.parse import quote
import sys

db_path = Path(sys.argv[1]).resolve()
artifact_path = Path(sys.argv[2]).resolve()
db_uri = "sqlite:////" + quote(str(db_path).lstrip("/"), safe="/")
print(db_uri)
print(artifact_path.as_uri())
PY
)

DB_URI="${URI_LINES[0]}"
ARTIFACT_URI="${URI_LINES[1]}"
SERVER_LOG="${LOG_DIR}/mlflow_server.log"

echo "Using backend store: ${DB_PATH}"
echo "Using artifact root: ${ARTIFACT_DIR}"
echo "Starting isolated MLflow on http://${HOST}:${PORT}"

if [[ -f "${DB_PATH}" && -s "${DB_PATH}" ]]; then
  echo "Existing SQLite backend detected; running schema upgrade check."
  "${MLFLOW_CMD[@]}" db upgrade "${DB_URI}" >/dev/null
else
  echo "Fresh SQLite backend detected; letting mlflow server initialize the schema."
fi

SERVER_ARGS=(
  server
  --host "${HOST}"
  --port "${PORT}"
  --workers "${WORKERS}"
  --backend-store-uri "${DB_URI}"
  --default-artifact-root "${ARTIFACT_URI}"
  --no-serve-artifacts
)

if command -v setsid >/dev/null 2>&1; then
  nohup setsid "${MLFLOW_CMD[@]}" "${SERVER_ARGS[@]}" > "${SERVER_LOG}" 2>&1 </dev/null &
else
  nohup "${MLFLOW_CMD[@]}" "${SERVER_ARGS[@]}" > "${SERVER_LOG}" 2>&1 </dev/null &
fi

SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"

READY=0
for _ in $(seq 1 15); do
  if "${PYTHON_BIN}" - "${HOST}" "${PORT}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.5)
    raise SystemExit(0 if sock.connect_ex((host, port)) == 0 else 1)
PY
  then
    READY=1
    break
  fi
  sleep 1
done

if [[ "${READY}" -eq 1 ]]; then
  echo "MLflow started successfully on http://${HOST}:${PORT}"
  echo "PID: ${SERVER_PID}"
  echo "Log: ${SERVER_LOG}"
  exit 0
fi

echo "MLflow did not start successfully. Check ${SERVER_LOG}"
exit 1
