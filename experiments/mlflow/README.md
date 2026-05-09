# Local MLflow Tracking

This project uses an isolated MLflow tracking server on `127.0.0.1:5001`.

## Why it is isolated

Another MLflow service is already running on `127.0.0.1:5000` for a different project. The scripts in this directory are designed not to disturb that service.

## What the scripts do

- [start_local_tracking.sh](/home/izhan/CISOSE%20RL%20Experimentations/experiments/mlflow/start_local_tracking.sh)
  - starts a new tracking server bound to `127.0.0.1:5001`
  - uses a project-local SQLite backend
  - uses a project-local artifact directory
- [stop_local_tracking.sh](/home/izhan/CISOSE%20RL%20Experimentations/experiments/mlflow/stop_local_tracking.sh)
  - stops only the project-local server started by the companion script
- [tracking_helper.py](/home/izhan/CISOSE%20RL%20Experimentations/experiments/mlflow/tracking_helper.py)
  - provides helper utilities for parent and child runs, tags, config artifacts, and metrics logging

## Usage

```bash
bash experiments/mlflow/start_local_tracking.sh
```

Point only this project's process to the tracking server:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5001 python3 your_training_script.py
```

Do not export `MLFLOW_TRACKING_URI` persistently at shell startup unless you want all local projects to use the proxy-study tracking server.
