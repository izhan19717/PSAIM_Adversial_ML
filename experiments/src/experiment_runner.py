from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .agents import AgentConfig, HeuristicAgent, PolicyGradientAgent, QControlAgent, set_global_seeds
from .proxy_env import ProxyAllocationEnv, RewardCorruptor, StressConfig, make_stress_config

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RuntimeConfig:
    seeds: int = 10
    train_episodes: int = 100
    eval_episodes: int = 8
    horizon: int = 96
    queue_capacity: int = 12
    visible_queue: int = 5
    alternating_block: int = 24
    use_mlflow: bool = False
    tracking_uri: str = "http://127.0.0.1:5001"
    experiment_name: str = "cisose-rl-proxy-evaluation"
    eval_adapt: bool = False
    paper_revision: str = "paper-final-claim-aligned-v3"


def visible_queue_mask(obs: np.ndarray, visible_queue: int, cpu_capacity: int = 8, memory_capacity: int = 8) -> List[Tuple[float, float, float, float]]:
    slots: List[Tuple[float, float, float, float]] = []
    offset = 7
    for _idx in range(visible_queue):
        cpu = float(obs[offset + 0]) * cpu_capacity
        mem = float(obs[offset + 1]) * memory_capacity
        duration = float(obs[offset + 2]) * 9.0
        wait = float(obs[offset + 3]) * 96.0
        slots.append((cpu, mem, duration, wait))
        offset += 4
    return slots


def maybe_start_mlflow(runtime: RuntimeConfig) -> None:
    if runtime.use_mlflow and mlflow is not None:
        mlflow.set_tracking_uri(runtime.tracking_uri)
        mlflow.set_experiment(runtime.experiment_name)


def run_episode(
    env: ProxyAllocationEnv,
    agent,
    training: bool,
    adapt: bool = False,
    reward_corruptor: Optional[RewardCorruptor] = None,
    capture_signals: bool = False,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    obs, info = env.reset()
    agent.begin_episode(training=training)
    signal_rows: List[Dict[str, float]] = []
    done = False
    episode_step = 0
    current_regime = info["regime"]
    rewards_trace: List[float] = []
    regime_trace: List[str] = []

    while not done:
        action_mask = visible_queue_mask(obs, env.visible_queue)
        if isinstance(agent, (QControlAgent,)):
            action = agent.select_action(obs, action_mask, training=training, adapt=adapt)
        else:
            action = agent.select_action(obs, action_mask, training=training)
        next_obs, reward, done, step_info = env.step(action)
        update_reward = reward_corruptor.push(reward) if reward_corruptor is not None else reward
        if isinstance(agent, PolicyGradientAgent):
            signals = agent.observe_transition(
                obs,
                action,
                update_reward,
                next_obs,
                done,
                training=training,
                adapt=adapt,
                episode_step=episode_step,
                regime=env.last_regime,
            )
        else:
            signals = agent.observe_transition(
                obs,
                action,
                update_reward,
                next_obs,
                done,
                training=training,
                adapt=adapt,
                episode_step=episode_step,
                regime=env.last_regime,
            )
        rewards_trace.append(float(reward))
        regime_trace.append(env.last_regime)
        if capture_signals and hasattr(agent, "consume_signal_rows"):
            for row in agent.consume_signal_rows():
                for metric in ["V_epi", "V_ale", "V_ale_excess", "gate_h3", "r_int"]:
                    signal_rows.append(
                        {
                            "episode": row["episode"],
                            "regime": row["regime"],
                            "action": row.get("action", np.nan),
                            "metric": metric,
                            "value": row.get(metric, np.nan),
                        }
                    )

        obs = next_obs
        episode_step += 1

    end_info = agent.end_episode(training=training, adapt=adapt)
    if capture_signals and hasattr(agent, "consume_signal_rows"):
        for row in agent.consume_signal_rows():
            for metric in ["V_epi", "V_ale", "V_ale_excess", "gate_h3", "r_int"]:
                signal_rows.append(
                    {
                        "episode": row["episode"],
                        "regime": row["regime"],
                        "action": row.get("action", np.nan),
                        "metric": metric,
                        "value": row.get(metric, np.nan),
                    }
                )
    metrics = env.episode_metrics()
    adaptation_lag = compute_adaptation_lag(rewards_trace, regime_trace)
    if adaptation_lag is not None:
        metrics["adaptation_lag_steps"] = adaptation_lag
    metrics.update({f"agent_{key}": value for key, value in end_info.items() if isinstance(value, (float, int))})
    return metrics, signal_rows


def compute_adaptation_lag(rewards_trace: List[float], regime_trace: List[str]) -> Optional[float]:
    if len(rewards_trace) < 8 or len(set(regime_trace)) < 2:
        return None
    lags: List[int] = []
    start = 0
    while start < len(regime_trace):
        regime = regime_trace[start]
        end = start
        while end < len(regime_trace) and regime_trace[end] == regime:
            end += 1
        block_rewards = np.asarray(rewards_trace[start:end], dtype=np.float32)
        if len(block_rewards) >= 6:
            stable_target = float(np.median(block_rewards[-min(5, len(block_rewards)) :]))
            rolling = np.convolve(block_rewards, np.ones(3) / 3.0, mode="valid")
            tolerance = 0.1 * max(abs(stable_target), 0.25)
            found = None
            for idx, value in enumerate(rolling, start=1):
                if abs(float(value) - stable_target) <= tolerance:
                    found = idx
                    break
            if found is not None:
                lags.append(found)
        start = end
    if not lags:
        return None
    return float(np.mean(lags))


def action_type(action: float, visible_queue: int) -> str:
    action_int = int(action)
    if action_int < visible_queue:
        return "allocate"
    if action_int == visible_queue:
        return "defer"
    return "reject"


def enrich_signal_row(row: Dict[str, float], visible_queue: int) -> Dict[str, object]:
    action = row.get("action", np.nan)
    enriched: Dict[str, object] = dict(row)
    if not np.isnan(action):
        enriched["action_type"] = action_type(action, visible_queue)
    else:
        enriched["action_type"] = "unknown"
    return enriched


def build_experiment_1_agents(state_dim: int, action_dim: int, seed: int) -> Dict[str, object]:
    cfg = AgentConfig()
    return {
        "heuristic_sjf_bestfit": HeuristicAgent(action_defer_index=action_dim - 2, action_reject_index=action_dim - 1),
        "deeprm_inspired_pg": PolicyGradientAgent(state_dim, action_dim, seed=seed + 11, config=cfg),
        "plain_dqn": QControlAgent("plain_dqn", state_dim, action_dim, seed=seed + 23, config=cfg),
    }


def make_experiment_2_config(**overrides: object) -> AgentConfig:
    """Return the simplified PSAIM/DQN hyperparameters used in the paper.

    These values are deliberately centralized because the C4/C5/C6 control
    studies must use the same simplified PSAIM configuration as the main
    Section VII experiments. Override only for diagnostics, not for paper
    claims unless the override is explicitly reported.
    """

    payload: Dict[str, object] = dict(
        warmup_steps=96,
        gradient_update_interval=4,
        target_update_interval=24,
        epsilon_decay_steps=2600,
        intrinsic_scale=0.001,
        rnd_scale=0.02,
        lambda_aleatoric=3.00,
        alpha_gate=0.16,
        sigma0_sq=0.05,
        probe_horizon=3,
    )
    payload.update(overrides)
    return AgentConfig(**payload)


def build_experiment_2_agents(
    state_dim: int,
    action_dim: int,
    seed: int,
    config: Optional[AgentConfig] = None,
    methods: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    cfg = config or make_experiment_2_config()
    all_agents = {
        "heuristic_sjf_bestfit": HeuristicAgent(action_defer_index=action_dim - 2, action_reject_index=action_dim - 1),
        "deeprm_inspired_pg": PolicyGradientAgent(state_dim, action_dim, seed=seed + 11, config=cfg),
        "plain_dqn": QControlAgent("plain_dqn", state_dim, action_dim, seed=seed + 21, config=cfg),
        "dqn_rnd": QControlAgent("dqn_rnd", state_dim, action_dim, seed=seed + 31, config=cfg, rnd=True),
        "psaim_lite": QControlAgent("psaim_lite", state_dim, action_dim, seed=seed + 41, config=cfg, psaim=True),
        "psaim_no_aleatoric": QControlAgent("psaim_no_aleatoric", state_dim, action_dim, seed=seed + 51, config=cfg, psaim=True, no_aleatoric=True),
        "psaim_no_gate": QControlAgent("psaim_no_gate", state_dim, action_dim, seed=seed + 61, config=cfg, psaim=True, no_gate=True),
        "psaim_no_freezing": QControlAgent("psaim_no_freezing", state_dim, action_dim, seed=seed + 71, config=cfg, psaim=True, no_freezing=True),
    }
    if methods is None:
        return all_agents
    selected = list(methods)
    return {name: all_agents[name] for name in selected}


def train_agent_on_clean_proxy(
    agent,
    seed: int,
    runtime: RuntimeConfig,
    workload_mode: str = "periodic",
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


def log_mlflow_run(name: str, tags: Dict[str, str], params: Dict[str, object], metrics: Dict[str, float]) -> None:
    if mlflow is None:
        return
    with mlflow.start_run(run_name=name, nested=True):
        mlflow.set_tags(tags)
        mlflow.log_params({key: str(value) for key, value in params.items()})
        for key, value in metrics.items():
            if isinstance(value, (float, int)):
                mlflow.log_metric(key, float(value))


def experiment_1(runtime: RuntimeConfig, output_dir: Path) -> None:
    seed_rows: List[Dict[str, object]] = []
    maybe_start_mlflow(runtime)
    parent_ctx = mlflow.start_run(run_name="experiment_1_proxy_stress_test") if runtime.use_mlflow and mlflow is not None else None
    if parent_ctx is not None:
        mlflow.set_tags({"experiment_type": "stress_test", "paper_revision": runtime.paper_revision})
        mlflow.log_params(asdict(runtime))

    try:
        for seed in range(runtime.seeds):
            set_global_seeds(seed)
            template_env = ProxyAllocationEnv(
                seed=seed,
                workload_mode="periodic",
                horizon=runtime.horizon,
                queue_capacity=runtime.queue_capacity,
                visible_queue=runtime.visible_queue,
                alternating_block=runtime.alternating_block,
                stress=make_stress_config("clean", "clean"),
            )
            obs, _ = template_env.reset()
            state_dim = int(obs.shape[0])
            action_dim = int(template_env.action_space_n)
            agents = build_experiment_1_agents(state_dim, action_dim, seed)

            for method_name, agent in agents.items():
                train_agent_on_clean_proxy(agent, seed=seed, runtime=runtime, workload_mode="periodic")
                clean_env = ProxyAllocationEnv(
                    seed=1000 + seed,
                    workload_mode="periodic",
                    horizon=runtime.horizon,
                    queue_capacity=runtime.queue_capacity,
                    visible_queue=runtime.visible_queue,
                    alternating_block=runtime.alternating_block,
                    stress=make_stress_config("clean", "clean"),
                )
                clean_metrics = []
                for _ in range(runtime.eval_episodes):
                    metrics, _ = run_episode(clean_env, agent, training=False, adapt=False)
                    clean_metrics.append(metrics)
                clean_df = pd.DataFrame(clean_metrics)
                clean_summary = clean_df.mean(numeric_only=True).to_dict()
                clean_summary.update({"scenario": "clean", "severity": "clean", "severity_rank": 0, "method": method_name, "seed": seed, "degradation_pct": 0.0})
                seed_rows.append(clean_summary)
                if runtime.use_mlflow and mlflow is not None:
                    log_mlflow_run(
                        name=f"exp1_{method_name}_seed{seed}_clean",
                        tags={"experiment_type": "stress_test", "method": method_name, "seed": str(seed), "stress_scenario": "clean", "stress_level": "clean"},
                        params={"seed": seed, "method": method_name},
                        metrics=clean_summary,
                    )

                clean_slowdown = clean_summary["average_slowdown"]
                for scenario in ["observation_noise", "reward_corruption", "distribution_shift", "co_tenant_interference", "co_tenant_matched_load_control"]:
                    for severity_rank, severity in enumerate(["low", "medium", "high"], start=1):
                        stress = make_stress_config(scenario, severity)
                        workload = "periodic" if scenario != "distribution_shift" else "high_entropy"
                        eval_agent = agent
                        reward_corrupted_training = scenario == "reward_corruption" and not isinstance(agent, HeuristicAgent)
                        if reward_corrupted_training:
                            eval_agent = build_experiment_1_agents(state_dim, action_dim, seed)[method_name]
                            train_agent_on_clean_proxy(
                                eval_agent,
                                seed=seed,
                                runtime=runtime,
                                workload_mode="periodic",
                                reward_corruptor_config=stress,
                            )
                        eval_env = ProxyAllocationEnv(
                            seed=2000 + seed + severity_rank,
                            workload_mode=workload,
                            horizon=runtime.horizon,
                            queue_capacity=runtime.queue_capacity,
                            visible_queue=runtime.visible_queue,
                            alternating_block=runtime.alternating_block,
                            stress=stress,
                        )
                        episode_metrics = []
                        adapt = scenario == "reward_corruption" and not reward_corrupted_training and not isinstance(agent, HeuristicAgent)
                        for _ in range(runtime.eval_episodes):
                            corruptor = RewardCorruptor(stress.reward_bias, stress.reward_delay) if adapt else None
                            metrics, _ = run_episode(eval_env, eval_agent, training=False, adapt=adapt, reward_corruptor=corruptor)
                            episode_metrics.append(metrics)
                        summary = pd.DataFrame(episode_metrics).mean(numeric_only=True).to_dict()
                        summary.update(
                            {
                                "scenario": scenario,
                                "severity": severity,
                                "severity_rank": severity_rank,
                                "method": method_name,
                                "seed": seed,
                                "degradation_pct": 100.0 * (summary["average_slowdown"] - clean_slowdown) / max(clean_slowdown, 1e-6),
                            }
                        )
                        seed_rows.append(summary)
                        if runtime.use_mlflow and mlflow is not None:
                            log_mlflow_run(
                                name=f"exp1_{method_name}_seed{seed}_{scenario}_{severity}",
                                tags={"experiment_type": "stress_test", "method": method_name, "seed": str(seed), "stress_scenario": scenario, "stress_level": severity},
                                params={"seed": seed, "method": method_name, "scenario": scenario, "severity": severity},
                                metrics=summary,
                        )

        df = pd.DataFrame(seed_rows)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "stress_robustness.csv"
        df.to_csv(output_path, index=False)
        if runtime.use_mlflow and mlflow is not None:
            mlflow.log_artifact(str(output_path), artifact_path="results")
            config_dir = PROJECT_ROOT / "experiments" / "config"
            if config_dir.exists():
                mlflow.log_artifacts(str(config_dir), artifact_path="config")
    finally:
        if parent_ctx is not None:
            mlflow.end_run()


def experiment_2(runtime: RuntimeConfig, output_dir: Path) -> None:
    signal_rows: List[Dict[str, object]] = []
    downstream_rows: List[Dict[str, object]] = []
    adaptation_rows: List[Dict[str, object]] = []
    maybe_start_mlflow(runtime)
    parent_ctx = mlflow.start_run(run_name="experiment_2_psaim_lite") if runtime.use_mlflow and mlflow is not None else None
    if parent_ctx is not None:
        mlflow.set_tags({"experiment_type": "psaim_lite", "paper_revision": runtime.paper_revision})
        mlflow.log_params(asdict(runtime))

    try:
        for seed in range(runtime.seeds):
            set_global_seeds(seed + 300)
            template_env = ProxyAllocationEnv(
                seed=seed + 300,
                workload_mode="alternating",
                horizon=runtime.horizon,
                queue_capacity=runtime.queue_capacity,
                visible_queue=runtime.visible_queue,
                alternating_block=runtime.alternating_block,
                stress=make_stress_config("clean", "clean"),
            )
            obs, _ = template_env.reset()
            state_dim = int(obs.shape[0])
            action_dim = int(template_env.action_space_n)
            agents = build_experiment_2_agents(state_dim, action_dim, seed + 300)
            for method_name, agent in agents.items():
                train_mode = "alternating" if method_name not in {"heuristic_sjf_bestfit"} else "periodic"
                train_agent_on_clean_proxy(agent, seed=seed + 300, runtime=runtime, workload_mode=train_mode)

                conditions = {
                    "clean": ("low_entropy", False),
                    "stressed": ("alternating", True),
                }
                for condition, (workload_mode, capture_signals) in conditions.items():
                    env = ProxyAllocationEnv(
                        seed=4000 + seed,
                        workload_mode=workload_mode,
                        horizon=runtime.horizon,
                        queue_capacity=runtime.queue_capacity,
                        visible_queue=runtime.visible_queue,
                        alternating_block=runtime.alternating_block,
                        stress=make_stress_config("clean", "clean"),
                    )
                    metrics_per_episode = []
                    for episode_idx in range(runtime.eval_episodes):
                        metrics, signals = run_episode(
                            env,
                            agent,
                            training=False,
                            adapt=runtime.eval_adapt and not isinstance(agent, HeuristicAgent),
                            capture_signals=capture_signals and method_name.startswith("psaim"),
                        )
                        metrics_per_episode.append(metrics)
                        if capture_signals and signals:
                            for row in signals:
                                row = enrich_signal_row(row, runtime.visible_queue)
                                row.update({"seed": seed, "method": method_name, "block": row["episode"] // runtime.alternating_block})
                                signal_rows.append(row)
                        if "adaptation_lag_steps" in metrics:
                            adaptation_rows.append({"method": method_name, "seed": seed, "switch_index": episode_idx + 1, "adaptation_lag_steps": metrics["adaptation_lag_steps"]})
                    summary = pd.DataFrame(metrics_per_episode).mean(numeric_only=True).to_dict()
                    summary.update({"method": method_name, "seed": seed, "condition": condition})
                    downstream_rows.append(summary)
                    if runtime.use_mlflow and mlflow is not None:
                        log_mlflow_run(
                            name=f"exp2_{method_name}_seed{seed}_{condition}",
                            tags={"experiment_type": "psaim_lite", "method": method_name, "seed": str(seed), "regime": condition},
                            params={"seed": seed, "method": method_name, "condition": condition},
                            metrics=summary,
                        )

        output_dir.mkdir(parents=True, exist_ok=True)
        signal_path = output_dir / "psaim_signals.csv"
        downstream_path = output_dir / "downstream_performance.csv"
        adaptation_path = output_dir / "adaptation_lag.csv"
        pd.DataFrame(signal_rows).to_csv(signal_path, index=False)
        pd.DataFrame(downstream_rows).to_csv(downstream_path, index=False)
        pd.DataFrame(adaptation_rows).to_csv(adaptation_path, index=False)
        if runtime.use_mlflow and mlflow is not None:
            for path in [signal_path, downstream_path, adaptation_path]:
                mlflow.log_artifact(str(path), artifact_path="results")
            config_dir = PROJECT_ROOT / "experiments" / "config"
            if config_dir.exists():
                mlflow.log_artifacts(str(config_dir), artifact_path="config")
    finally:
        if parent_ctx is not None:
            mlflow.end_run()


def write_run_manifest(runtime: RuntimeConfig, output_dir: Path) -> None:
    payload = {
        "runtime": asdict(runtime),
        "output_dir": str(output_dir),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
