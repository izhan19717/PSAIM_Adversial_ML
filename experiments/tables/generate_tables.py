from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "experiments" / "data" / "results" / "paper_final_claim_aligned_v3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "output" / "tables"
METHOD_LABELS = {
    "heuristic_sjf_bestfit": "Heuristic",
    "deeprm_inspired_pg": "PG proxy",
    "plain_dqn": "DQN",
    "dqn_rnd": "DQN+RND",
    "psaim_lite": "Simplified PSAIM",
    "psaim_no_aleatoric": "Simplified PSAIM, no aleatoric penalty",
    "psaim_no_gate": "Simplified PSAIM, no gate",
    "psaim_no_freezing": "Simplified PSAIM, no freezing",
}


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(321)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_metric(group: pd.Series) -> str:
    values = group.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return "NA"
    mean = values.mean()
    lo, hi = bootstrap_ci(values)
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def write_outputs(df: pd.DataFrame, stem: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"{stem}.csv", index=False)
    (output_dir / f"{stem}.md").write_text(to_markdown_table(df), encoding="utf-8")


def to_markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    rows = [[str(value) for value in row] for row in df.to_numpy().tolist()]
    widths = [
        max(len(str(column)), *(len(row[idx]) for row in rows)) if rows else len(str(column))
        for idx, column in enumerate(columns)
    ]

    def format_row(values: list[str]) -> str:
        cells = [value.ljust(widths[idx]) for idx, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    header = format_row([str(column) for column in columns])
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [format_row(row) for row in rows]
    return "\n".join([header, divider, *body]) + "\n"


def build_experiment_1_table(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "stress_robustness.csv")
    df = df[df["scenario"].ne("clean")]
    rows = []
    for (scenario, severity, method), group in df.groupby(["scenario", "severity", "method"], dropna=False):
        rows.append(
            {
                "scenario": scenario,
                "severity": severity,
                "method": METHOD_LABELS.get(method, method),
                "average_slowdown": summarize_metric(group["average_slowdown"]),
                "p95_completion_time": summarize_metric(group["p95_completion_time"]),
                "utilization": summarize_metric(group["utilization"]),
                "task_failure_rate": summarize_metric(group["task_failure_rate"]),
                "degradation_pct": summarize_metric(group["degradation_pct"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["scenario", "severity", "method"]).reset_index(drop=True)


def build_experiment_2_table(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "downstream_performance.csv")
    methods = [
        "heuristic_sjf_bestfit",
        "deeprm_inspired_pg",
        "plain_dqn",
        "dqn_rnd",
        "psaim_lite",
    ]
    df = df[df["method"].isin(methods)]
    rows = []
    for (condition, method), group in df.groupby(["condition", "method"], dropna=False):
        rows.append(
            {
                "condition": condition,
                "method": METHOD_LABELS.get(method, method),
                "average_slowdown": summarize_metric(group["average_slowdown"]),
                "mean_completion_time": summarize_metric(group["mean_completion_time"]),
                "p95_completion_time": summarize_metric(group["p95_completion_time"]),
                "task_failure_rate": summarize_metric(group["task_failure_rate"]),
                "utilization": summarize_metric(group["utilization"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["condition", "method"]).reset_index(drop=True)


def build_appendix_ablation_table(data_dir: Path) -> pd.DataFrame:
    perf = pd.read_csv(data_dir / "downstream_performance.csv")
    perf = perf[perf["method"].isin(["psaim_lite", "psaim_no_aleatoric", "psaim_no_gate", "psaim_no_freezing"])]
    perf = perf[perf["condition"].eq("stressed")]
    lag = pd.read_csv(data_dir / "adaptation_lag.csv")
    lag = lag[lag["method"].isin(["psaim_lite", "psaim_no_aleatoric", "psaim_no_gate", "psaim_no_freezing"])]

    rows = []
    for method in ["psaim_lite", "psaim_no_aleatoric", "psaim_no_gate", "psaim_no_freezing"]:
        perf_group = perf[perf["method"].eq(method)]
        lag_group = lag[lag["method"].eq(method)]
        rows.append(
            {
                "method": METHOD_LABELS.get(method, method),
                "stressed_average_slowdown": summarize_metric(perf_group["average_slowdown"]),
                "stressed_task_failure_rate": summarize_metric(perf_group["task_failure_rate"]),
                "adaptation_lag_steps": summarize_metric(lag_group["adaptation_lag_steps"]),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate summary tables from experiment CSVs.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5001")
    parser.add_argument("--experiment-name", default="cisose-rl-proxy-evaluation")
    parser.add_argument("--run-name", default="paper_table_generation")
    parser.add_argument("--use-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    parent_ctx = None
    if args.use_mlflow:
        if mlflow is None:
            raise RuntimeError("MLflow was requested but is not importable in this Python environment.")
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment_name)
        parent_ctx = mlflow.start_run(run_name=args.run_name)
        mlflow.set_tags({"experiment_type": "table_generation", "paper_revision": "proxy-study-v1"})
        mlflow.log_params({"data_dir": str(data_dir), "output_dir": str(output_dir)})
    try:
        write_outputs(build_experiment_1_table(data_dir), "table_experiment_1", output_dir)
        write_outputs(build_experiment_2_table(data_dir), "table_experiment_2", output_dir)
        write_outputs(build_appendix_ablation_table(data_dir), "table_appendix_ablations", output_dir)
        if args.use_mlflow and mlflow is not None:
            mlflow.log_artifacts(str(output_dir), artifact_path="tables")
    finally:
        if parent_ctx is not None:
            mlflow.end_run()
    print(f"Wrote tables to {output_dir}")


if __name__ == "__main__":
    main()
