from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from run_c4_experiments import METHOD_LABELS, bootstrap_ci, paired_diff_ci, to_markdown_table
from src.agents import HeuristicAgent, set_global_seeds
from src.experiment_runner import RuntimeConfig, build_experiment_2_agents, make_experiment_2_config, run_episode
from src.proxy_env import ProxyAllocationEnv, RewardCorruptor, StressConfig, make_stress_config

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def fmt_ci(values: Iterable[float]) -> str:
    arr = np.asarray(list(values), dtype=float)
    mean = float(np.nanmean(arr))
    lo, hi = bootstrap_ci(arr)
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def template_dimensions(runtime: RuntimeConfig, seed: int, workload_mode: str = "periodic") -> Tuple[int, int]:
    env = ProxyAllocationEnv(
        seed=seed,
        workload_mode=workload_mode,
        horizon=runtime.horizon,
        queue_capacity=runtime.queue_capacity,
        visible_queue=runtime.visible_queue,
        alternating_block=runtime.alternating_block,
        stress=make_stress_config("clean", "clean"),
    )
    obs, _ = env.reset()
    return int(obs.shape[0]), int(env.action_space_n)


def build_agents(state_dim: int, action_dim: int, seed: int, methods: List[str]) -> Dict[str, object]:
    return build_experiment_2_agents(
        state_dim,
        action_dim,
        seed,
        config=make_experiment_2_config(),
        methods=methods,
    )


def train_agent(
    agent,
    seed: int,
    runtime: RuntimeConfig,
    workload_mode: str,
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
        stress=make_stress_config("clean", "clean"),
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
    adapt: bool = False,
    reward_corruptor_config: Optional[StressConfig] = None,
) -> Dict[str, float]:
    env = ProxyAllocationEnv(
        seed=seed,
        workload_mode=workload_mode,
        horizon=runtime.horizon,
        queue_capacity=runtime.queue_capacity,
        visible_queue=runtime.visible_queue,
        alternating_block=runtime.alternating_block,
        stress=make_stress_config("clean", "clean"),
    )
    metrics = []
    for _episode in range(runtime.eval_episodes):
        corruptor = (
            RewardCorruptor(reward_corruptor_config.reward_bias, reward_corruptor_config.reward_delay)
            if reward_corruptor_config is not None
            else None
        )
        episode_metrics, _ = run_episode(env, agent, training=False, adapt=adapt, reward_corruptor=corruptor)
        metrics.append(episode_metrics)
    return pd.DataFrame(metrics).mean(numeric_only=True).to_dict()


def append_row(
    rows: List[Dict[str, object]],
    metrics: Dict[str, float],
    *,
    experiment: str,
    scenario: str,
    severity: str,
    method: str,
    seed: int,
    clean_slowdown: Optional[float],
) -> None:
    payload: Dict[str, object] = dict(metrics)
    degradation = 0.0
    if clean_slowdown is not None:
        degradation = 100.0 * (float(payload["average_slowdown"]) - clean_slowdown) / max(clean_slowdown, 1e-6)
    payload.update(
        {
            "experiment": experiment,
            "scenario": scenario,
            "severity": severity,
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "seed": seed,
            "degradation_pct": degradation,
        }
    )
    rows.append(payload)


def run_reward_corruption_training_sweep(runtime: RuntimeConfig) -> pd.DataFrame:
    methods = ["heuristic_sjf_bestfit", "deeprm_inspired_pg", "plain_dqn", "dqn_rnd", "psaim_lite"]
    rows: List[Dict[str, object]] = []
    for seed in range(runtime.seeds):
        print(f"[reward-train] seed {seed}", flush=True)
        set_global_seeds(31000 + seed)
        state_dim, action_dim = template_dimensions(runtime, 31000 + seed)
        clean_slowdowns: Dict[str, float] = {}

        clean_agents = build_agents(state_dim, action_dim, 32000 + seed, methods)
        for method, agent in clean_agents.items():
            train_agent(agent, 33000 + seed, runtime, "periodic")
            metrics = evaluate_agent(agent, 34000 + seed, runtime, "periodic")
            clean_slowdowns[method] = float(metrics["average_slowdown"])
            append_row(
                rows,
                metrics,
                experiment="reward_corruption_training_sweep",
                scenario="clean",
                severity="clean",
                method=method,
                seed=seed,
                clean_slowdown=None,
            )

        for severity in ["low", "medium", "high"]:
            stress = make_stress_config("reward_corruption", severity)
            stressed_agents = build_agents(state_dim, action_dim, 35000 + (100 * len(rows)) + seed, methods)
            for method, agent in stressed_agents.items():
                train_agent(agent, 36000 + seed, runtime, "periodic", reward_corruptor_config=stress)
                metrics = evaluate_agent(agent, 37000 + seed, runtime, "periodic")
                append_row(
                    rows,
                    metrics,
                    experiment="reward_corruption_training_sweep",
                    scenario="reward_corruption_training",
                    severity=severity,
                    method=method,
                    seed=seed,
                    clean_slowdown=clean_slowdowns[method],
                )
    return pd.DataFrame(rows)


def run_reward_corruption_online_adaptation(runtime: RuntimeConfig) -> pd.DataFrame:
    methods = ["heuristic_sjf_bestfit", "deeprm_inspired_pg", "plain_dqn", "dqn_rnd", "psaim_lite"]
    rows: List[Dict[str, object]] = []
    stress = make_stress_config("reward_corruption", "high")
    for seed in range(runtime.seeds):
        print(f"[reward-online] seed {seed}", flush=True)
        set_global_seeds(41000 + seed)
        state_dim, action_dim = template_dimensions(runtime, 41000 + seed)

        clean_agents = build_agents(state_dim, action_dim, 42000 + seed, methods)
        for method, agent in clean_agents.items():
            train_agent(agent, 43000 + seed, runtime, "periodic")
            clean_metrics = evaluate_agent(agent, 44000 + seed, runtime, "periodic")
            append_row(
                rows,
                clean_metrics,
                experiment="reward_corruption_online_adaptation",
                scenario="clean",
                severity="clean",
                method=method,
                seed=seed,
                clean_slowdown=None,
            )

            stressed_metrics = evaluate_agent(
                agent,
                45000 + seed,
                runtime,
                "periodic",
                adapt=not isinstance(agent, HeuristicAgent),
                reward_corruptor_config=stress,
            )
            append_row(
                rows,
                stressed_metrics,
                experiment="reward_corruption_online_adaptation",
                scenario="reward_corruption_online",
                severity="high",
                method=method,
                seed=seed,
                clean_slowdown=float(clean_metrics["average_slowdown"]),
            )
    return pd.DataFrame(rows)


def run_long_horizon_drift(train_runtime: RuntimeConfig, eval_runtime: RuntimeConfig) -> pd.DataFrame:
    methods = ["psaim_lite", "psaim_no_gate", "psaim_no_freezing"]
    rows: List[Dict[str, object]] = []
    for seed in range(train_runtime.seeds):
        print(f"[drift] seed {seed}", flush=True)
        set_global_seeds(51000 + seed)
        state_dim, action_dim = template_dimensions(train_runtime, 51000 + seed, workload_mode="monotonic_drift")
        agents = build_agents(state_dim, action_dim, 52000 + seed, methods)
        for method, agent in agents.items():
            train_agent(agent, 53000 + seed, train_runtime, "monotonic_drift")
            metrics = evaluate_agent(agent, 54000 + seed, eval_runtime, "monotonic_drift")
            append_row(
                rows,
                metrics,
                experiment="long_horizon_monotonic_drift",
                scenario="monotonic_drift",
                severity="long_horizon",
                method=method,
                seed=seed,
                clean_slowdown=None,
            )
    return pd.DataFrame(rows)


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (experiment, scenario, severity, method, method_label), group in df.groupby(
        ["experiment", "scenario", "severity", "method", "method_label"], dropna=False
    ):
        rows.append(
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
    return pd.DataFrame(rows).sort_values(["experiment", "scenario", "severity", "method"]).reset_index(drop=True)


def pair_table(df: pd.DataFrame, experiment: str, scenario: str, comparisons: List[Tuple[str, str]]) -> pd.DataFrame:
    subset = df[df["experiment"].eq(experiment) & df["scenario"].eq(scenario)]
    rows: List[Dict[str, object]] = []
    for severity in sorted(subset["severity"].dropna().unique()):
        severity_subset = subset[subset["severity"].eq(severity)]
        for left_method, right_method in comparisons:
            left = severity_subset[severity_subset["method"].eq(left_method)].set_index("seed")
            right = severity_subset[severity_subset["method"].eq(right_method)].set_index("seed")
            for metric in ["average_slowdown", "degradation_pct", "task_failure_rate", "p95_completion_time"]:
                mean, lo, hi = paired_diff_ci(left[metric], right[metric])
                if np.isfinite(hi) and hi < 0.0:
                    status = f"{METHOD_LABELS.get(left_method, left_method)} lower"
                elif np.isfinite(lo) and lo <= 0.0 <= hi:
                    status = "not decisive"
                else:
                    status = f"{METHOD_LABELS.get(right_method, right_method)} lower"
                rows.append(
                    {
                        "experiment": experiment,
                        "scenario": scenario,
                        "severity": severity,
                        "comparison": f"{METHOD_LABELS.get(left_method, left_method)} - {METHOD_LABELS.get(right_method, right_method)}",
                        "metric": metric,
                        "paired_diff_mean_ci": f"{fmt(mean)} [{fmt(lo)}, {fmt(hi)}]",
                        "status": status,
                    }
                )
    return pd.DataFrame(rows)


def reward_status(df: pd.DataFrame, experiment: str, scenario: str) -> str:
    subset = df[df["experiment"].eq(experiment) & df["scenario"].eq(scenario)]
    parts: List[str] = []
    for severity in sorted(subset["severity"].dropna().unique()):
        severity_subset = subset[subset["severity"].eq(severity)]
        psaim = float(severity_subset[severity_subset["method"].eq("psaim_lite")]["degradation_pct"].mean())
        dqn = float(severity_subset[severity_subset["method"].eq("plain_dqn")]["degradation_pct"].mean())
        mean, lo, hi = paired_diff_ci(
            severity_subset[severity_subset["method"].eq("psaim_lite")].set_index("seed")["degradation_pct"],
            severity_subset[severity_subset["method"].eq("plain_dqn")].set_index("seed")["degradation_pct"],
        )
        if np.isfinite(hi) and hi < 0.0:
            status = "supported"
        elif psaim < dqn:
            status = "partially supported"
        else:
            status = "refuted"
        parts.append(
            f"{severity}: {status}, PSAIM degradation={psaim:.1f}%, DQN degradation={dqn:.1f}%, "
            f"paired PSAIM-DQN={mean:.1f}% [{lo:.1f}, {hi:.1f}]"
        )
    return f"{experiment}: " + "; ".join(parts) + "."


def drift_status(df: pd.DataFrame) -> str:
    subset = df[df["experiment"].eq("long_horizon_monotonic_drift")]
    full = subset[subset["method"].eq("psaim_lite")].set_index("seed")["average_slowdown"]
    statuses = []
    for ablation in ["psaim_no_gate", "psaim_no_freezing"]:
        other = subset[subset["method"].eq(ablation)].set_index("seed")["average_slowdown"]
        mean, lo, hi = paired_diff_ci(other, full)
        if np.isfinite(lo) and lo > 0.0:
            status = "supported"
        elif np.isfinite(mean) and mean > 0.0:
            status = "partially supported"
        else:
            status = "refuted"
        statuses.append(
            f"{METHOD_LABELS.get(ablation, ablation)} minus full={mean:.3f} [{lo:.3f}, {hi:.3f}] ({status})"
        )
    return "Long-horizon drift: " + "; ".join(statuses) + "."


def write_report(df: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary_table(df)
    reward_training_pairs = pair_table(
        df,
        "reward_corruption_training_sweep",
        "reward_corruption_training",
        [("psaim_lite", "plain_dqn")],
    )
    reward_online_pairs = pair_table(
        df,
        "reward_corruption_online_adaptation",
        "reward_corruption_online",
        [("psaim_lite", "plain_dqn")],
    )
    drift_pairs = pair_table(
        df,
        "long_horizon_monotonic_drift",
        "monotonic_drift",
        [("psaim_no_gate", "psaim_lite"), ("psaim_no_freezing", "psaim_lite")],
    )

    df.to_csv(output_dir / "raw_results.csv", index=False)
    summary.to_csv(output_dir / "summary_table.csv", index=False)
    reward_training_pairs.to_csv(output_dir / "reward_training_pairs.csv", index=False)
    reward_online_pairs.to_csv(output_dir / "reward_online_pairs.csv", index=False)
    drift_pairs.to_csv(output_dir / "long_horizon_drift_pairs.csv", index=False)

    (output_dir / "summary_table.md").write_text(to_markdown_table(summary), encoding="utf-8")
    (output_dir / "reward_training_pairs.md").write_text(to_markdown_table(reward_training_pairs), encoding="utf-8")
    (output_dir / "reward_online_pairs.md").write_text(to_markdown_table(reward_online_pairs), encoding="utf-8")
    (output_dir / "long_horizon_drift_pairs.md").write_text(to_markdown_table(drift_pairs), encoding="utf-8")

    lines = [
        "# Reward-Corruption and Long-Horizon Drift Follow-Up",
        "",
        "All intervals are 95% bootstrap intervals over 10 seeds. Paired comparisons use same-seed paired bootstrap with 5000 resamples. PSAIM hyperparameters are unchanged from the main experiments.",
        "",
        "## Summary Table",
        "",
        to_markdown_table(summary),
        "## Reward-Corruption Training-Time Pairing",
        "",
        to_markdown_table(reward_training_pairs),
        "## Reward-Corruption Online-Adaptation Pairing",
        "",
        to_markdown_table(reward_online_pairs),
        "## Long-Horizon Drift Pairing",
        "",
        to_markdown_table(drift_pairs),
        "## Hypothesis Status",
        "",
        f"- {reward_status(df, 'reward_corruption_training_sweep', 'reward_corruption_training')}",
        f"- {reward_status(df, 'reward_corruption_online_adaptation', 'reward_corruption_online')}",
        f"- {drift_status(df)}",
        "",
        "## Protocol Notes",
        "",
        "- Training-time reward corruption trains learned agents with delayed biased rewards, then evaluates task metrics without using corrupted task metrics as reported outcomes.",
        "- Online-adaptation reward corruption clean-trains learned agents and then allows online updates during evaluation using corrupted reward feedback.",
        "- Monotonic drift uses the same proxy allocator and workload generator with an added drift schedule: the probability of high-entropy arrivals increases linearly over the episode.",
    ]
    (output_dir / "C4_REWARD_DRIFT_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "run_manifest.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reward-corruption extension and long-horizon drift experiments.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "experiments" / "data" / "results" / "c4_reward_drift_v1"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--reward-train-episodes", type=int, default=100)
    parser.add_argument("--reward-eval-episodes", type=int, default=8)
    parser.add_argument("--reward-horizon", type=int, default=96)
    parser.add_argument("--drift-train-episodes", type=int, default=500)
    parser.add_argument("--drift-eval-episodes", type=int, default=8)
    parser.add_argument("--drift-horizon", type=int, default=480)
    parser.add_argument("--queue-capacity", type=int, default=12)
    parser.add_argument("--visible-queue", type=int, default=5)
    parser.add_argument("--alternating-block", type=int, default=24)
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5001")
    parser.add_argument("--experiment-name", default="cisose-rl-proxy-evaluation")
    parser.add_argument("--run-name", default="c4_reward_drift_extension")
    parser.add_argument("--use-mlflow", action="store_true")
    parser.add_argument("--skip-reward", action="store_true")
    parser.add_argument("--skip-drift", action="store_true")
    return parser.parse_args()


def make_runtime(args: argparse.Namespace, train_episodes: int, eval_episodes: int, horizon: int) -> RuntimeConfig:
    return RuntimeConfig(
        seeds=args.seeds,
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
        horizon=horizon,
        queue_capacity=args.queue_capacity,
        visible_queue=args.visible_queue,
        alternating_block=args.alternating_block,
        use_mlflow=args.use_mlflow,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        paper_revision="c4-reward-drift-extension",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.results_dir)
    parent_ctx = None
    if args.use_mlflow:
        if mlflow is None:
            raise RuntimeError("MLflow requested but not importable. Set PYTHONPATH to experiments/.deps/mlflow.")
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment_name)
        parent_ctx = mlflow.start_run(run_name=args.run_name)
        mlflow.set_tags({"experiment_type": "c4_reward_drift_extension", "paper_revision": "c4-reward-drift-extension"})
        mlflow.log_params(vars(args))

    try:
        frames: List[pd.DataFrame] = []
        if not args.skip_reward:
            reward_runtime = make_runtime(args, args.reward_train_episodes, args.reward_eval_episodes, args.reward_horizon)
            frames.append(run_reward_corruption_training_sweep(reward_runtime))
            Path(args.results_dir).mkdir(parents=True, exist_ok=True)
            pd.concat(frames, ignore_index=True).to_csv(Path(args.results_dir) / "partial_after_reward_training.csv", index=False)
            frames.append(run_reward_corruption_online_adaptation(reward_runtime))
            pd.concat(frames, ignore_index=True).to_csv(Path(args.results_dir) / "partial_after_reward_online.csv", index=False)
        if not args.skip_drift:
            drift_train_runtime = make_runtime(args, args.drift_train_episodes, args.drift_eval_episodes, args.reward_horizon)
            drift_eval_runtime = make_runtime(args, 1, args.drift_eval_episodes, args.drift_horizon)
            frames.append(run_long_horizon_drift(drift_train_runtime, drift_eval_runtime))
            pd.concat(frames, ignore_index=True).to_csv(Path(args.results_dir) / "partial_after_drift.csv", index=False)
        df = pd.concat(frames, ignore_index=True)
        write_report(df, output_dir, args)
        if args.use_mlflow and mlflow is not None:
            mlflow.log_artifacts(str(output_dir), artifact_path="c4_reward_drift")
        print(f"Wrote reward/drift follow-up outputs to {output_dir}")
    finally:
        if parent_ctx is not None and mlflow is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
