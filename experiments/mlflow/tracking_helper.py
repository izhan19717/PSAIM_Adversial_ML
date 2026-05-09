from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

try:
    import mlflow
except ImportError:  # pragma: no cover - environment-dependent
    mlflow = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_URI = "http://127.0.0.1:5001"
DEFAULT_EXPERIMENT_NAME = "cisose-rl-proxy-evaluation"


def ensure_mlflow_available() -> None:
    if mlflow is None:
        raise RuntimeError(
            "mlflow is not installed in the current Python environment. "
            "Install it or run from an environment where mlflow is available."
        )


def configure_tracking(
    tracking_uri: Optional[str] = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> None:
    ensure_mlflow_available()
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)


def default_tags() -> Dict[str, str]:
    return {
        "paper_revision": "proxy-study-v1",
        "tracking_stack": "isolated-mlflow",
        "workspace_root": str(PROJECT_ROOT),
    }


def merge_tags(*tag_groups: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for group in tag_groups:
        if not group:
            continue
        for key, value in group.items():
            merged[str(key)] = str(value)
    return merged


def log_json_artifact(payload: Mapping[str, Any], artifact_subdir: str, filename: str) -> None:
    ensure_mlflow_available()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / filename
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        mlflow.log_artifact(str(path), artifact_path=artifact_subdir)


def log_metrics(metrics: Mapping[str, float], step: Optional[int] = None) -> None:
    ensure_mlflow_available()
    for key, value in metrics.items():
        mlflow.log_metric(key, float(value), step=step)


def log_params(params: Mapping[str, Any]) -> None:
    ensure_mlflow_available()
    serializable = {str(key): str(value) for key, value in params.items()}
    mlflow.log_params(serializable)


def log_common_artifacts(config: Mapping[str, Any], tags: Mapping[str, Any]) -> None:
    log_json_artifact(config, artifact_subdir="config", filename="run_config.json")
    log_json_artifact(tags, artifact_subdir="meta", filename="run_tags.json")


@contextmanager
def parent_run(
    run_name: str,
    tracking_uri: Optional[str] = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tags: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
):
    configure_tracking(tracking_uri=tracking_uri, experiment_name=experiment_name)
    combined_tags = merge_tags(default_tags(), {"run_level": "parent"}, tags)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(combined_tags)
        if config:
            log_common_artifacts(config=config, tags=combined_tags)
        yield run


@contextmanager
def child_run(
    run_name: str,
    parent_run_id: str,
    tags: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
):
    ensure_mlflow_available()
    combined_tags = merge_tags(default_tags(), {"run_level": "child"}, tags)
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        mlflow.set_tag("mlflow.parentRunId", parent_run_id)
        mlflow.set_tags(combined_tags)
        if config:
            log_common_artifacts(config=config, tags=combined_tags)
        yield run


def make_child_tags(
    experiment_type: str,
    method: str,
    seed: int,
    workload_split: str,
    stress_scenario: Optional[str] = None,
    stress_level: Optional[str] = None,
    regime: Optional[str] = None,
) -> Dict[str, str]:
    tags = {
        "experiment_type": experiment_type,
        "method": method,
        "seed": str(seed),
        "workload_split": workload_split,
    }
    if stress_scenario is not None:
        tags["stress_scenario"] = stress_scenario
    if stress_level is not None:
        tags["stress_level"] = stress_level
    if regime is not None:
        tags["regime"] = regime
    return tags


def example_run_plan() -> Dict[str, Any]:
    return {
        "parent_runs": [
            {
                "name": "experiment_1_proxy_stress_test",
                "children": "method x seed x stress_scenario x stress_level x workload_split",
            },
            {
                "name": "experiment_2_psaim_lite",
                "children": "method x seed x regime_condition x workload_split",
            },
        ]
    }
