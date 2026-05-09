from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.agents import HeuristicAgent, QControlAgent, set_global_seeds
from src.experiment_runner import (
    RuntimeConfig,
    action_type,
    build_experiment_2_agents,
    make_experiment_2_config,
    run_episode,
    visible_queue_mask,
)
from src.proxy_env import ProxyAllocationEnv, make_stress_config

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run claim-specific validation probes for the CISOSE proxy study.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "experiments" / "data" / "results" / "claim_validation"))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--train-episodes", type=int, default=120)
    parser.add_argument("--eval-episodes", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--queue-capacity", type=int, default=12)
    parser.add_argument("--visible-queue", type=int, default=5)
    parser.add_argument("--alternating-block", type=int, default=24)
    parser.add_argument("--probe-states-per-regime", type=int, default=24)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--methods", default="psaim_lite")
    parser.add_argument("--intrinsic-scale", type=float, default=0.01)
    parser.add_argument("--lambda-aleatoric", type=float, default=0.80)
    parser.add_argument("--alpha-gate", type=float, default=0.16)
    parser.add_argument("--sigma0-sq", type=float, default=0.05)
    parser.add_argument("--epistemic-prior-floor", type=float, default=0.35)
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5001")
    parser.add_argument("--experiment-name", default="cisose-rl-proxy-evaluation")
    parser.add_argument("--run-name", default="claim_validation")
    parser.add_argument("--use-mlflow", action="store_true")
    return parser.parse_args()


def make_runtime(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
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
        paper_revision="proxy-study-claim-validation",
    )


def collect_probe_states(
    runtime: RuntimeConfig,
    seed: int,
    action_dim: int,
    probe_states_per_regime: int,
) -> List[Dict[str, object]]:
    env = ProxyAllocationEnv(
        seed=9000 + seed,
        workload_mode="alternating",
        horizon=runtime.horizon,
        queue_capacity=runtime.queue_capacity,
        visible_queue=runtime.visible_queue,
        alternating_block=runtime.alternating_block,
        stress=make_stress_config("clean", "clean"),
    )
    heuristic = HeuristicAgent(action_defer_index=action_dim - 2, action_reject_index=action_dim - 1)
    obs, _ = env.reset()
    rows: List[Dict[str, object]] = []
    counts = {"low_entropy": 0, "high_entropy": 0}
    target_per_regime = max(1, probe_states_per_regime)
    done = False
    step = 0
    while not done and min(counts.values()) < target_per_regime:
        regime = env.last_regime
        mask = visible_queue_mask(obs, runtime.visible_queue)
        action = heuristic.select_action(obs, mask, training=False)
        if counts[regime] < target_per_regime:
            rows.append(
                {
                    "probe_index": len(rows),
                    "regime": regime,
                    "step": step,
                    "action": int(action),
                    "action_type": action_type(action, runtime.visible_queue),
                    "obs": obs.copy(),
                }
            )
            counts[regime] += 1
        obs, _reward, done, _info = env.step(action)
        step += 1
    return rows


def probe_agent(agent: QControlAgent, probe_states: List[Dict[str, object]], seed: int, checkpoint_episode: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in probe_states:
        signals = agent.diagnostic_signal(item["obs"], action=int(item["action"]))
        for metric in ["V_epi", "V_ale", "V_ale_excess", "gate_h3", "r_int"]:
            rows.append(
                {
                    "seed": seed,
                    "checkpoint_episode": checkpoint_episode,
                    "probe_index": item["probe_index"],
                    "regime": item["regime"],
                    "action": item["action"],
                    "action_type": item["action_type"],
                    "metric": metric,
                    "value": signals.get(metric, np.nan),
                }
            )
    return rows


def summarize_behavior(signal_df: pd.DataFrame) -> pd.DataFrame:
    if signal_df.empty:
        return pd.DataFrame()
    pivot = signal_df.pivot_table(
        index=["seed", "method", "regime", "episode", "action", "action_type"],
        columns="metric",
        values="value",
        aggfunc="mean",
    ).reset_index()
    rows: List[Dict[str, object]] = []
    for (seed, method, regime), group in pivot.groupby(["seed", "method", "regime"]):
        total = max(1, len(group))
        payload: Dict[str, object] = {
            "seed": seed,
            "method": method,
            "regime": regime,
            "mean_V_epi": float(group.get("V_epi", pd.Series(dtype=float)).mean()),
            "mean_V_ale": float(group.get("V_ale", pd.Series(dtype=float)).mean()),
            "mean_V_ale_excess": float(group.get("V_ale_excess", pd.Series(dtype=float)).mean()),
            "mean_gate_h3": float(group.get("gate_h3", pd.Series(dtype=float)).mean()),
            "mean_r_int": float(group.get("r_int", pd.Series(dtype=float)).mean()),
            "positive_intrinsic_rate": float((group.get("r_int", pd.Series(dtype=float)) > 0.0).mean()),
        }
        for name in ["allocate", "defer", "reject"]:
            payload[f"{name}_rate"] = float((group["action_type"] == name).sum() / total)
        rows.append(payload)
    return pd.DataFrame(rows)


def run_claim_validation(
    runtime: RuntimeConfig,
    output_dir: Path,
    methods: List[str],
    probe_states_per_regime: int,
    checkpoint_every: int,
    config_overrides: Dict[str, float],
) -> None:
    training_signal_rows: List[Dict[str, object]] = []
    eval_signal_rows: List[Dict[str, object]] = []
    behavior_summary_rows: List[pd.DataFrame] = []
    downstream_rows: List[Dict[str, object]] = []
    config = make_experiment_2_config(**config_overrides)
    checkpoints = set(range(0, runtime.train_episodes + 1, max(1, runtime.train_episodes // 6)))
    checkpoints.update(range(0, runtime.train_episodes + 1, max(1, checkpoint_every)))
    checkpoints.add(runtime.train_episodes)

    for seed in range(runtime.seeds):
        set_global_seeds(seed + 7300)
        template_env = ProxyAllocationEnv(
            seed=seed + 7300,
            workload_mode="alternating",
            horizon=runtime.horizon,
            queue_capacity=runtime.queue_capacity,
            visible_queue=runtime.visible_queue,
            alternating_block=runtime.alternating_block,
            stress=make_stress_config("clean", "clean"),
        )
        obs, _ = template_env.reset()
        agents = build_experiment_2_agents(
            int(obs.shape[0]),
            int(template_env.action_space_n),
            seed + 7300,
            config=config,
            methods=methods,
        )
        probe_states = collect_probe_states(
            runtime,
            seed,
            int(template_env.action_space_n),
            probe_states_per_regime,
        )

        for method_name, agent in agents.items():
            train_env = ProxyAllocationEnv(
                seed=seed + 7300,
                workload_mode="alternating",
                horizon=runtime.horizon,
                queue_capacity=runtime.queue_capacity,
                visible_queue=runtime.visible_queue,
                alternating_block=runtime.alternating_block,
                stress=make_stress_config("clean", "clean"),
            )
            if method_name.startswith("psaim"):
                training_signal_rows.extend(probe_agent(agent, probe_states, seed, 0))
            for episode in range(1, runtime.train_episodes + 1):
                run_episode(train_env, agent, training=True)
                if method_name.startswith("psaim") and episode in checkpoints:
                    training_signal_rows.extend(probe_agent(agent, probe_states, seed, episode))

            eval_env = ProxyAllocationEnv(
                seed=17000 + seed,
                workload_mode="alternating",
                horizon=runtime.horizon,
                queue_capacity=runtime.queue_capacity,
                visible_queue=runtime.visible_queue,
                alternating_block=runtime.alternating_block,
                stress=make_stress_config("clean", "clean"),
            )
            metrics_per_episode: List[Dict[str, float]] = []
            method_signal_rows: List[Dict[str, object]] = []
            for eval_episode in range(runtime.eval_episodes):
                metrics, signals = run_episode(
                    eval_env,
                    agent,
                    training=False,
                    adapt=False,
                    capture_signals=method_name.startswith("psaim"),
                )
                metrics_per_episode.append(metrics)
                for row in signals:
                    row["seed"] = seed
                    row["method"] = method_name
                    row["eval_episode"] = eval_episode
                    row["action_type"] = action_type(row.get("action", np.nan), runtime.visible_queue)
                    method_signal_rows.append(row)
            if method_signal_rows:
                eval_signal_rows.extend(method_signal_rows)
                behavior_summary_rows.append(summarize_behavior(pd.DataFrame(method_signal_rows)))
            summary = pd.DataFrame(metrics_per_episode).mean(numeric_only=True).to_dict()
            summary.update({"seed": seed, "method": method_name, "condition": "alternating_hidden_regime"})
            downstream_rows.append(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    training_path = output_dir / "training_epistemic_probes.csv"
    eval_signal_path = output_dir / "behavior_signals.csv"
    behavior_path = output_dir / "exploration_behavior.csv"
    downstream_path = output_dir / "downstream_performance.csv"
    pd.DataFrame(training_signal_rows).to_csv(training_path, index=False)
    pd.DataFrame(eval_signal_rows).to_csv(eval_signal_path, index=False)
    if behavior_summary_rows:
        pd.concat(behavior_summary_rows, ignore_index=True).to_csv(behavior_path, index=False)
    else:
        pd.DataFrame().to_csv(behavior_path, index=False)
    pd.DataFrame(downstream_rows).to_csv(downstream_path, index=False)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "runtime": asdict(runtime),
                "methods": methods,
                "config_overrides": config_overrides,
                "probe_states_per_regime": probe_states_per_regime,
                "checkpoint_every": checkpoint_every,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    runtime = make_runtime(args)
    output_dir = Path(args.results_dir)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    parent_ctx = None
    if args.use_mlflow:
        if mlflow is None:
            raise RuntimeError("MLflow requested but not importable.")
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment_name)
        parent_ctx = mlflow.start_run(run_name=args.run_name)
        mlflow.set_tags({"experiment_type": "claim_validation", "paper_revision": runtime.paper_revision})
        mlflow.log_params({key: str(value) for key, value in vars(args).items()})
    try:
        run_claim_validation(
            runtime,
            output_dir,
            methods=methods,
            probe_states_per_regime=args.probe_states_per_regime,
            checkpoint_every=args.checkpoint_every,
            config_overrides={
                "intrinsic_scale": args.intrinsic_scale,
                "lambda_aleatoric": args.lambda_aleatoric,
                "alpha_gate": args.alpha_gate,
                "sigma0_sq": args.sigma0_sq,
                "epistemic_prior_floor": args.epistemic_prior_floor,
            },
        )
        if args.use_mlflow and mlflow is not None:
            mlflow.log_artifacts(str(output_dir), artifact_path="claim_validation")
    finally:
        if parent_ctx is not None:
            mlflow.end_run()
    print(f"Wrote claim-validation outputs to {output_dir}")


if __name__ == "__main__":
    main()
