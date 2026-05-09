from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.agents import HeuristicAgent, set_global_seeds
from src.experiment_runner import (
    RuntimeConfig,
    build_experiment_2_agents,
    make_experiment_2_config,
    run_episode,
)
from src.proxy_env import ProxyAllocationEnv, RewardCorruptor, StressConfig, make_stress_config

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_LABELS = {
    "heuristic_sjf_bestfit": "SJF best-fit heuristic",
    "deeprm_inspired_pg": "PG proxy",
    "plain_dqn": "DQN",
    "dqn_rnd": "DQN+RND",
    "psaim_lite": "Simplified PSAIM",
    "psaim_no_gate": "Simplified PSAIM, no gate",
    "psaim_no_freezing": "Simplified PSAIM, no freezing",
}


def bootstrap_ci(values: Iterable[float], n_boot: int = 5000, seed: int = 371) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True)
    means = samples.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff_ci(left: pd.Series, right: pd.Series, n_boot: int = 5000, seed: int = 877) -> Tuple[float, float, float]:
    paired = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    if paired.empty:
        return float("nan"), float("nan"), float("nan")
    diffs = paired["left"].to_numpy(dtype=float) - paired["right"].to_numpy(dtype=float)
    lo, hi = bootstrap_ci(diffs, n_boot=n_boot, seed=seed)
    return float(diffs.mean()), lo, hi


def fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def fmt_ci(values: pd.Series) -> str:
    arr = values.to_numpy(dtype=float)
    mean = float(np.nanmean(arr))
    lo, hi = bootstrap_ci(arr)
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def to_markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    rows = [[str(value) for value in row] for row in df.to_numpy().tolist()]
    widths = [
        max(len(str(column)), *(len(row[idx]) for row in rows)) if rows else len(str(column))
        for idx, column in enumerate(columns)
    ]

    def format_row(values: List[str]) -> str:
        cells = [value.ljust(widths[idx]) for idx, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    header = format_row([str(column) for column in columns])
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [format_row(row) for row in rows]
    return "\n".join([header, divider, *body]) + "\n"


def template_dimensions(runtime: RuntimeConfig, seed: int, stress: Optional[StressConfig] = None) -> Tuple[int, int]:
    env = ProxyAllocationEnv(
        seed=seed,
        workload_mode="periodic",
        horizon=runtime.horizon,
        queue_capacity=runtime.queue_capacity,
        visible_queue=runtime.visible_queue,
        alternating_block=runtime.alternating_block,
        stress=stress or make_stress_config("clean", "clean"),
    )
    obs, _ = env.reset()
    return int(obs.shape[0]), int(env.action_space_n)


def train_agent(
    agent,
    seed: int,
    runtime: RuntimeConfig,
    workload_mode: str,
    stress: Optional[StressConfig] = None,
    reward_corruptor_config: Optional[StressConfig] = None,
) -> None:
    if isinstance(agent, HeuristicAgent):
        return
    env = ProxyAllocationEnv(
        seed=seed,
        workload_mode=workload_mode,
        horizon=runtime.horizon,
        queue_capacity=runtime.queue_capacity,
        visible_queue=runtime.visible_queue,
        alternating_block=runtime.alternating_block,
        stress=stress or make_stress_config("clean", "clean"),
    )
    for _episode in range(runtime.train_episodes):
        corruptor = (
            RewardCorruptor(reward_corruptor_config.reward_bias, reward_corruptor_config.reward_delay)
            if reward_corruptor_config is not None
            else None
        )
        run_episode(env, agent, training=True, reward_corruptor=corruptor)


def evaluate_agent(
    agent,
    seed: int,
    runtime: RuntimeConfig,
    workload_mode: str,
    stress: Optional[StressConfig] = None,
) -> Dict[str, float]:
    env = ProxyAllocationEnv(
        seed=seed,
        workload_mode=workload_mode,
        horizon=runtime.horizon,
        queue_capacity=runtime.queue_capacity,
        visible_queue=runtime.visible_queue,
        alternating_block=runtime.alternating_block,
        stress=stress or make_stress_config("clean", "clean"),
    )
    metrics = []
    for _episode in range(runtime.eval_episodes):
        episode_metrics, _ = run_episode(env, agent, training=False, adapt=False)
        metrics.append(episode_metrics)
    return pd.DataFrame(metrics).mean(numeric_only=True).to_dict()


def build_agents(state_dim: int, action_dim: int, seed: int, methods: List[str]) -> Dict[str, object]:
    return build_experiment_2_agents(
        state_dim,
        action_dim,
        seed,
        config=make_experiment_2_config(),
        methods=methods,
    )


def add_row(
    rows: List[Dict[str, object]],
    metrics: Dict[str, float],
    *,
    experiment: str,
    scenario: str,
    severity: str,
    severity_rank: int,
    method: str,
    seed: int,
    condition: str,
    clean_slowdown: Optional[float],
) -> None:
    payload: Dict[str, object] = dict(metrics)
    slowdown = float(payload.get("average_slowdown", np.nan))
    degradation = 0.0
    if clean_slowdown is not None:
        degradation = 100.0 * (slowdown - clean_slowdown) / max(clean_slowdown, 1e-6)
    payload.update(
        {
            "experiment": experiment,
            "scenario": scenario,
            "severity": severity,
            "severity_rank": severity_rank,
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "seed": seed,
            "condition": condition,
            "degradation_pct": degradation,
        }
    )
    rows.append(payload)


def run_reward_corruption(runtime: RuntimeConfig) -> List[Dict[str, object]]:
    methods = ["heuristic_sjf_bestfit", "deeprm_inspired_pg", "plain_dqn", "dqn_rnd", "psaim_lite"]
    stress = make_stress_config("reward_corruption", "high")
    rows: List[Dict[str, object]] = []

    for seed in range(runtime.seeds):
        set_global_seeds(11000 + seed)
        state_dim, action_dim = template_dimensions(runtime, 11000 + seed)
        clean_slowdowns: Dict[str, float] = {}

        clean_agents = build_agents(state_dim, action_dim, 12000 + seed, methods)
        for method, agent in clean_agents.items():
            train_agent(agent, 13000 + seed, runtime, workload_mode="periodic")
            metrics = evaluate_agent(agent, 14000 + seed, runtime, workload_mode="periodic")
            clean_slowdowns[method] = float(metrics["average_slowdown"])
            add_row(
                rows,
                metrics,
                experiment="reward_corruption_robustness",
                scenario="clean",
                severity="clean",
                severity_rank=0,
                method=method,
                seed=seed,
                condition="clean",
                clean_slowdown=None,
            )

        stressed_agents = build_agents(state_dim, action_dim, 15000 + seed, methods)
        for method, agent in stressed_agents.items():
            train_agent(
                agent,
                16000 + seed,
                runtime,
                workload_mode="periodic",
                reward_corruptor_config=stress,
            )
            metrics = evaluate_agent(
                agent,
                17000 + seed,
                runtime,
                workload_mode="periodic",
                stress=stress,
            )
            add_row(
                rows,
                metrics,
                experiment="reward_corruption_robustness",
                scenario="reward_corruption",
                severity="high",
                severity_rank=3,
                method=method,
                seed=seed,
                condition="stressed",
                clean_slowdown=clean_slowdowns[method],
            )
    return rows


def run_duration_misreport(runtime: RuntimeConfig) -> List[Dict[str, object]]:
    methods = ["heuristic_sjf_bestfit", "psaim_lite"]
    rows: List[Dict[str, object]] = []

    for seed in range(runtime.seeds):
        set_global_seeds(21000 + seed)
        state_dim, action_dim = template_dimensions(runtime, 21000 + seed)
        clean_slowdowns: Dict[str, float] = {}

        clean_agents = build_agents(state_dim, action_dim, 22000 + seed, methods)
        for method, agent in clean_agents.items():
            train_agent(agent, 23000 + seed, runtime, workload_mode="periodic")
            metrics = evaluate_agent(agent, 24000 + seed, runtime, workload_mode="periodic")
            clean_slowdowns[method] = float(metrics["average_slowdown"])
            add_row(
                rows,
                metrics,
                experiment="heuristic_breaking_shift",
                scenario="clean",
                severity="clean",
                severity_rank=0,
                method=method,
                seed=seed,
                condition="clean",
                clean_slowdown=None,
            )

        for severity_rank, severity in enumerate(["low", "medium", "high"], start=1):
            stress = make_stress_config("duration_misreport", severity)
            stressed_agents = build_agents(state_dim, action_dim, 25000 + (100 * severity_rank) + seed, methods)
            for method, agent in stressed_agents.items():
                train_agent(agent, 26000 + (100 * severity_rank) + seed, runtime, workload_mode="periodic", stress=stress)
                metrics = evaluate_agent(
                    agent,
                    27000 + (100 * severity_rank) + seed,
                    runtime,
                    workload_mode="periodic",
                    stress=stress,
                )
                add_row(
                    rows,
                    metrics,
                    experiment="heuristic_breaking_shift",
                    scenario="duration_misreport",
                    severity=severity,
                    severity_rank=severity_rank,
                    method=method,
                    seed=seed,
                    condition="stressed",
                    clean_slowdown=clean_slowdowns[method],
                )
    return rows


def build_slowdown_table(rows: pd.DataFrame) -> pd.DataFrame:
    table_rows: List[Dict[str, object]] = []
    grouped = rows.groupby(["experiment", "scenario", "severity_rank", "severity", "method", "method_label"], dropna=False)
    for (experiment, scenario, severity_rank, severity, method, method_label), group in grouped:
        table_rows.append(
            {
                "experiment": experiment,
                "scenario": scenario,
                "severity": severity,
                "method": method_label,
                "average_slowdown": fmt_ci(group["average_slowdown"]),
                "degradation_pct": fmt_ci(group["degradation_pct"]),
                "task_failure_rate": fmt_ci(group["task_failure_rate"]),
                "p95_completion_time": fmt_ci(group["p95_completion_time"]),
            }
        )
    table = pd.DataFrame(table_rows)
    return table.sort_values(["experiment", "scenario", "severity", "method"]).reset_index(drop=True)


def reward_pair_table(rows: pd.DataFrame) -> pd.DataFrame:
    stressed = rows[
        rows["experiment"].eq("reward_corruption_robustness")
        & rows["scenario"].eq("reward_corruption")
        & rows["severity"].eq("high")
    ]
    psaim = stressed[stressed["method"].eq("psaim_lite")].set_index("seed")
    dqn = stressed[stressed["method"].eq("plain_dqn")].set_index("seed")
    table_rows = []
    for metric in ["average_slowdown", "degradation_pct", "task_failure_rate", "p95_completion_time"]:
        mean, lo, hi = paired_diff_ci(psaim[metric], dqn[metric])
        table_rows.append(
            {
                "comparison": "Simplified PSAIM - DQN",
                "metric": metric,
                "paired_diff_mean_ci": f"{fmt(mean)} [{fmt(lo)}, {fmt(hi)}]",
                "status": "PSAIM lower" if np.isfinite(hi) and hi < 0.0 else "not decisive",
            }
        )
    return pd.DataFrame(table_rows)


def misreport_pair_table(rows: pd.DataFrame) -> pd.DataFrame:
    stressed = rows[
        rows["experiment"].eq("heuristic_breaking_shift")
        & rows["scenario"].eq("duration_misreport")
    ]
    table_rows = []
    for severity in ["low", "medium", "high"]:
        subset = stressed[stressed["severity"].eq(severity)]
        psaim = subset[subset["method"].eq("psaim_lite")].set_index("seed")
        sjf = subset[subset["method"].eq("heuristic_sjf_bestfit")].set_index("seed")
        for metric in ["average_slowdown", "degradation_pct"]:
            mean, lo, hi = paired_diff_ci(psaim[metric], sjf[metric])
            if np.isfinite(hi) and hi < 0.0:
                status = "PSAIM lower"
            elif np.isfinite(lo) and lo <= 0.0 <= hi:
                status = "statistical tie / inconclusive"
            else:
                status = "SJF lower"
            table_rows.append(
                {
                    "severity": severity,
                    "comparison": "Simplified PSAIM - SJF",
                    "metric": metric,
                    "paired_diff_mean_ci": f"{fmt(mean)} [{fmt(lo)}, {fmt(hi)}]",
                    "status": status,
                }
            )
    return pd.DataFrame(table_rows)


def experiment_statuses(rows: pd.DataFrame, reward_pairs: pd.DataFrame, misreport_pairs: pd.DataFrame) -> List[str]:
    lines: List[str] = []

    reward = rows[
        rows["experiment"].eq("reward_corruption_robustness")
        & rows["scenario"].eq("reward_corruption")
        & rows["severity"].eq("high")
    ]
    psaim_deg = float(reward[reward["method"].eq("psaim_lite")]["degradation_pct"].mean())
    dqn_deg = float(reward[reward["method"].eq("plain_dqn")]["degradation_pct"].mean())
    reward_diff = reward_pairs[reward_pairs["metric"].eq("degradation_pct")]["status"].iloc[0]
    reward_supported = reward_diff == "PSAIM lower" and 0.0 <= psaim_deg <= 50.0
    reward_partial = reward_diff == "PSAIM lower" or psaim_deg < dqn_deg
    if reward_supported:
        status = "supported"
    elif reward_partial:
        status = "partially supported"
    else:
        status = "refuted"
    lines.append(
        f"Reward-corruption robustness: {status}. Hypothesis: simplified PSAIM should degrade less than DQN under high reward corruption. "
        f"Interpretation: simplified PSAIM mean degradation was {psaim_deg:.1f}% versus DQN {dqn_deg:.1f}%."
    )

    severity_statuses = []
    for severity in ["low", "medium", "high"]:
        pair = misreport_pairs[
            misreport_pairs["severity"].eq(severity)
            & misreport_pairs["metric"].eq("average_slowdown")
        ].iloc[0]
        severity_statuses.append(f"{severity}: {pair['status']}")
    if any("PSAIM lower" in item for item in severity_statuses):
        status = "supported"
    elif any("statistical tie" in item for item in severity_statuses):
        status = "partially supported"
    else:
        status = "refuted"
    lines.append(
        "Heuristic-breaking duration misreporting: "
        f"{status}. Hypothesis: simplified PSAIM should beat or statistically tie SJF when reported durations invert the SJF ordering. "
        f"Interpretation: " + "; ".join(severity_statuses) + "."
    )

    lines.append(
        "Long-horizon regime drift: skipped in this positive-demonstration bundle because the 5x training and 5x evaluation protocol is substantially more expensive than the first two priority experiments. "
        "No result is claimed for the freezing/gate long-horizon hypothesis."
    )
    return lines


def write_outputs(rows: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_dir / "c4_raw_results.csv", index=False)
    slowdown = build_slowdown_table(rows)
    reward_pairs = reward_pair_table(rows)
    misreport_pairs = misreport_pair_table(rows)
    slowdown.to_csv(output_dir / "c4_slowdown_table.csv", index=False)
    reward_pairs.to_csv(output_dir / "c4_reward_corruption_pairs.csv", index=False)
    misreport_pairs.to_csv(output_dir / "c4_duration_misreport_pairs.csv", index=False)
    (output_dir / "c4_slowdown_table.md").write_text(to_markdown_table(slowdown), encoding="utf-8")
    (output_dir / "c4_reward_corruption_pairs.md").write_text(to_markdown_table(reward_pairs), encoding="utf-8")
    (output_dir / "c4_duration_misreport_pairs.md").write_text(to_markdown_table(misreport_pairs), encoding="utf-8")

    lines = [
        "# Positive-Demonstration Follow-Up Experiments",
        "",
        "All reported intervals are 95% nonparametric bootstrap confidence intervals over 10 seeds unless otherwise noted. "
        "Paired comparisons use same-seed paired bootstrap differences with 5000 resamples. Negative paired differences mean simplified PSAIM is lower/better.",
        "",
        "## Scenario Slowdown Table",
        "",
        to_markdown_table(slowdown),
        "## Reward-Corruption Paired Comparison",
        "",
        to_markdown_table(reward_pairs),
        "## Duration-Misreport Paired Comparison",
        "",
        to_markdown_table(misreport_pairs),
        "## Hypothesis Status",
        "",
    ]
    for statement in experiment_statuses(rows, reward_pairs, misreport_pairs):
        lines.append(f"- {statement}")
    lines.extend(
        [
            "",
            "## Protocol Notes",
            "",
            "- PSAIM hyperparameters were not changed from the main experiments.",
            "- The duration-misreport stressor changes only the reported duration feature in observations; true job durations, workload generation, execution dynamics, and completion-time metrics are unchanged.",
            "- The SJF best-fit heuristic and simplified PSAIM are evaluated on the same single-node, two-resource proxy environment.",
        ]
    )
    (output_dir / "C4_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "run_manifest.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run positive-demonstration follow-up experiments.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "experiments" / "data" / "results" / "c4_positive_demo_v1"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--queue-capacity", type=int, default=12)
    parser.add_argument("--visible-queue", type=int, default=5)
    parser.add_argument("--alternating-block", type=int, default=24)
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5001")
    parser.add_argument("--experiment-name", default="cisose-rl-proxy-evaluation")
    parser.add_argument("--run-name", default="c4_positive_demonstration")
    parser.add_argument("--use-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = RuntimeConfig(
        seeds=args.seeds,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        horizon=args.horizon,
        queue_capacity=args.queue_capacity,
        visible_queue=args.visible_queue,
        alternating_block=args.alternating_block,
        use_mlflow=args.use_mlflow,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        paper_revision="c4-positive-demo",
    )
    output_dir = Path(args.results_dir)

    parent_ctx = None
    if args.use_mlflow:
        if mlflow is None:
            raise RuntimeError("MLflow requested but not importable. Set PYTHONPATH to experiments/.deps/mlflow.")
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment_name)
        parent_ctx = mlflow.start_run(run_name=args.run_name)
        mlflow.set_tags({"experiment_type": "c4_positive_demonstration", "paper_revision": runtime.paper_revision})
        mlflow.log_params({**asdict(runtime), "script": "run_c4_experiments.py"})

    try:
        rows = pd.DataFrame([*run_reward_corruption(runtime), *run_duration_misreport(runtime)])
        write_outputs(rows, output_dir, args)
        if args.use_mlflow and mlflow is not None:
            mlflow.log_artifacts(str(output_dir), artifact_path="c4_results")
            reward = rows[
                rows["experiment"].eq("reward_corruption_robustness")
                & rows["scenario"].eq("reward_corruption")
                & rows["severity"].eq("high")
            ]
            for method in ["plain_dqn", "psaim_lite", "heuristic_sjf_bestfit"]:
                subset = reward[reward["method"].eq(method)]
                if not subset.empty:
                    mlflow.log_metric(f"reward_high_{method}_slowdown", float(subset["average_slowdown"].mean()))
                    mlflow.log_metric(f"reward_high_{method}_degradation_pct", float(subset["degradation_pct"].mean()))
        print(f"Wrote positive-demonstration outputs to {output_dir}")
    finally:
        if parent_ctx is not None and mlflow is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
