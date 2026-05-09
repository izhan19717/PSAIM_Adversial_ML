from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from run_c4_experiments import METHOD_LABELS, bootstrap_ci, paired_diff_ci, to_markdown_table
from src.agents import HeuristicAgent, QControlAgent, set_global_seeds
from src.experiment_runner import RuntimeConfig, build_experiment_2_agents, make_experiment_2_config, visible_queue_mask
from src.proxy_env import ProxyAllocationEnv, StressConfig, make_stress_config

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


def action_type(action: int, visible_queue: int) -> str:
    if action < visible_queue:
        return "allocate"
    if action == visible_queue:
        return "defer"
    return "reject"


def template_dimensions(runtime: RuntimeConfig, seed: int) -> Tuple[int, int]:
    env = ProxyAllocationEnv(
        seed=seed,
        workload_mode="periodic",
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


def train_agent(agent, seed: int, runtime: RuntimeConfig, stress: Optional[StressConfig] = None) -> None:
    if isinstance(agent, HeuristicAgent):
        return
    env = ProxyAllocationEnv(
        seed=seed,
        workload_mode="periodic",
        horizon=runtime.horizon,
        queue_capacity=runtime.queue_capacity,
        visible_queue=runtime.visible_queue,
        alternating_block=runtime.alternating_block,
        stress=stress or make_stress_config("clean", "clean"),
    )
    for _episode in range(runtime.train_episodes):
        run_episode_action_counts(env, agent, runtime, training=True)


def run_episode_action_counts(
    env: ProxyAllocationEnv,
    agent,
    runtime: RuntimeConfig,
    training: bool,
) -> Dict[str, float]:
    obs, _ = env.reset()
    agent.begin_episode(training=training)
    counts = {"allocate": 0, "defer": 0, "reject": 0}
    done = False
    episode_step = 0
    while not done:
        action_mask = visible_queue_mask(obs, env.visible_queue)
        if isinstance(agent, QControlAgent):
            action = agent.select_action(obs, action_mask, training=training, adapt=False)
        else:
            action = agent.select_action(obs, action_mask, training=training)
        counts[action_type(int(action), runtime.visible_queue)] += 1
        next_obs, reward, done, _info = env.step(int(action))
        agent.observe_transition(
            obs,
            int(action),
            float(reward),
            next_obs,
            done,
            training=training,
            adapt=False,
            episode_step=episode_step,
            regime=env.last_regime,
        )
        obs = next_obs
        episode_step += 1
    agent.end_episode(training=training, adapt=False)
    total = max(1, sum(counts.values()))
    return {
        "allocate_rate": counts["allocate"] / total,
        "defer_rate": counts["defer"] / total,
        "reject_rate": counts["reject"] / total,
        "allocate_count": float(counts["allocate"]),
        "defer_count": float(counts["defer"]),
        "reject_count": float(counts["reject"]),
        "total_actions": float(total),
    }


def evaluate_action_distribution(agent, seed: int, runtime: RuntimeConfig, stress: Optional[StressConfig]) -> Dict[str, float]:
    env = ProxyAllocationEnv(
        seed=seed,
        workload_mode="periodic",
        horizon=runtime.horizon,
        queue_capacity=runtime.queue_capacity,
        visible_queue=runtime.visible_queue,
        alternating_block=runtime.alternating_block,
        stress=stress or make_stress_config("clean", "clean"),
    )
    totals = {"allocate_count": 0.0, "defer_count": 0.0, "reject_count": 0.0, "total_actions": 0.0}
    for _episode in range(runtime.eval_episodes):
        episode = run_episode_action_counts(env, agent, runtime, training=False)
        for key in totals:
            totals[key] += float(episode[key])
    total_actions = max(1.0, totals["total_actions"])
    return {
        "allocate_rate": totals["allocate_count"] / total_actions,
        "defer_rate": totals["defer_count"] / total_actions,
        "reject_rate": totals["reject_count"] / total_actions,
        **totals,
    }


def run_duration_misreport_action_audit(runtime: RuntimeConfig) -> pd.DataFrame:
    methods = ["heuristic_sjf_bestfit", "psaim_lite"]
    rows: List[Dict[str, object]] = []
    for seed in range(runtime.seeds):
        print(f"[action-audit] seed {seed}", flush=True)
        set_global_seeds(61000 + seed)
        state_dim, action_dim = template_dimensions(runtime, 61000 + seed)

        clean_agents = build_agents(state_dim, action_dim, 61500 + seed, methods)
        for method, agent in clean_agents.items():
            train_agent(agent, 61600 + seed, runtime, stress=make_stress_config("clean", "clean"))
            rates = evaluate_action_distribution(agent, 61700 + seed, runtime, stress=make_stress_config("clean", "clean"))
            rows.append(
                {
                    "scenario": "clean",
                    "severity": "clean",
                    "severity_rank": 0,
                    "seed": seed,
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    **rates,
                }
            )

        for severity_rank, severity in enumerate(["low", "medium", "high"], start=1):
            stress = make_stress_config("duration_misreport", severity)
            agents = build_agents(state_dim, action_dim, 62000 + (100 * severity_rank) + seed, methods)
            for method, agent in agents.items():
                train_agent(agent, 63000 + (100 * severity_rank) + seed, runtime, stress=stress)
                rates = evaluate_action_distribution(agent, 64000 + (100 * severity_rank) + seed, runtime, stress=stress)
                rows.append(
                    {
                        "scenario": "duration_misreport",
                        "severity": severity,
                        "severity_rank": severity_rank,
                        "seed": seed,
                        "method": method,
                        "method_label": METHOD_LABELS.get(method, method),
                        **rates,
                    }
                )
    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (severity, severity_rank, method, method_label), group in df.groupby(
        ["severity", "severity_rank", "method", "method_label"], dropna=False
    ):
        rows.append(
            {
                "severity": severity,
                "method": method_label,
                "allocate_rate": fmt_ci(group["allocate_rate"]),
                "defer_rate": fmt_ci(group["defer_rate"]),
                "reject_rate": fmt_ci(group["reject_rate"]),
            }
        )
    order = {"clean": 0, "low": 1, "medium": 2, "high": 3}
    table = pd.DataFrame(rows)
    table["_severity_order"] = table["severity"].map(order).fillna(99)
    return table.sort_values(["_severity_order", "method"]).drop(columns=["_severity_order"]).reset_index(drop=True)


def build_pair_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for severity in ["clean", "low", "medium", "high"]:
        subset = df[df["severity"].eq(severity)]
        psaim = subset[subset["method"].eq("psaim_lite")].set_index("seed")
        sjf = subset[subset["method"].eq("heuristic_sjf_bestfit")].set_index("seed")
        for metric in ["allocate_rate", "defer_rate", "reject_rate"]:
            mean, lo, hi = paired_diff_ci(psaim[metric], sjf[metric])
            rows.append(
                {
                    "severity": severity,
                    "comparison": "Simplified PSAIM - SJF heuristic",
                    "metric": metric,
                    "paired_diff_mean_ci": f"{fmt(mean)} [{fmt(lo)}, {fmt(hi)}]",
                }
            )
    return pd.DataFrame(rows)


def write_outputs(df: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(df)
    pairs = build_pair_table(df)
    df.to_csv(output_dir / "duration_misreport_action_distribution_raw.csv", index=False)
    summary.to_csv(output_dir / "duration_misreport_action_distribution_summary.csv", index=False)
    pairs.to_csv(output_dir / "duration_misreport_action_distribution_pairs.csv", index=False)
    (output_dir / "duration_misreport_action_distribution_summary.md").write_text(to_markdown_table(summary), encoding="utf-8")
    (output_dir / "duration_misreport_action_distribution_pairs.md").write_text(to_markdown_table(pairs), encoding="utf-8")
    lines = [
        "# Duration-Misreport Action Distribution Audit",
        "",
        "This focused audit reruns the duration-misreport setup with action-type logging. Values are 10-seed means with 95% bootstrap CIs over seed-level action rates. Paired differences use same-seed paired bootstrap. The PSAIM hyperparameters and proxy environment are unchanged.",
        "",
        "## Action-Type Rates",
        "",
        to_markdown_table(summary),
        "## Paired Differences",
        "",
        to_markdown_table(pairs),
    ]
    (output_dir / "DURATION_MISREPORT_ACTION_DISTRIBUTION.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "run_manifest.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit action distributions under duration misreporting.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "experiments" / "data" / "results" / "c4_action_distribution_v1"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--queue-capacity", type=int, default=12)
    parser.add_argument("--visible-queue", type=int, default=5)
    parser.add_argument("--alternating-block", type=int, default=24)
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5001")
    parser.add_argument("--experiment-name", default="cisose-rl-proxy-evaluation")
    parser.add_argument("--run-name", default="c4_duration_misreport_action_distribution")
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
        paper_revision="c4-action-distribution",
    )
    parent_ctx = None
    if args.use_mlflow:
        if mlflow is None:
            raise RuntimeError("MLflow requested but not importable. Set PYTHONPATH to experiments/.deps/mlflow.")
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment_name)
        parent_ctx = mlflow.start_run(run_name=args.run_name)
        mlflow.set_tags({"experiment_type": "c4_action_distribution", "paper_revision": runtime.paper_revision})
        mlflow.log_params({**asdict(runtime), "script": "run_c4_action_distribution.py"})
    try:
        df = run_duration_misreport_action_audit(runtime)
        output_dir = Path(args.results_dir)
        write_outputs(df, output_dir, args)
        if args.use_mlflow and mlflow is not None:
            mlflow.log_artifacts(str(output_dir), artifact_path="c4_action_distribution")
        print(f"Wrote action-distribution outputs to {output_dir}")
    finally:
        if parent_ctx is not None and mlflow is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
