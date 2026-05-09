from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.agents import (
    DEVICE,
    HeuristicAgent,
    QControlAgent,
    valid_action_indices_from_obs,
    set_global_seeds,
)
from src.experiment_runner import RuntimeConfig, make_experiment_2_config, visible_queue_mask
from src.proxy_env import ProxyAllocationEnv, StressConfig, make_stress_config

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3}
METHOD_LABELS = {
    "heuristic_sjf_bestfit": "SJF best-fit heuristic",
    "psaim_lite": "Simplified PSAIM",
    "psaim_rawvar": "PSAIM-RawVar",
    "sjf_r_0p10": "SJF-R(0.10)",
    "sjf_r_0p17": "SJF-R(0.17)",
    "sjf_r_0p18": "SJF-R(0.18)",
    "sjf_r_0p25": "SJF-R(0.25)",
}


def bootstrap_ci(values: Iterable[float], n_boot: int = 5000, seed: int = 371) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, arr.size), replace=True)
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


def fmt_ci(values: Iterable[float], digits: int = 3) -> str:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "NA"
    lo, hi = bootstrap_ci(arr)
    return f"{float(arr.mean()):.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


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


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def action_type(action: int, visible_queue: int) -> str:
    if action < visible_queue:
        return "allocate"
    if action == visible_queue:
        return "defer"
    return "reject"


class SJFStaticRejectAgent:
    """Uniform-random rejection control wrapped around the existing SJF heuristic."""

    def __init__(self, rejection_probability: float, action_dim: int, seed: int) -> None:
        self.rejection_probability = float(rejection_probability)
        self.name = f"sjf_r_{self.rejection_probability:.2f}".replace(".", "p")
        self.reject_index = action_dim - 1
        self.base = HeuristicAgent(action_defer_index=action_dim - 2, action_reject_index=action_dim - 1)
        self.rng = np.random.default_rng(seed)

    def begin_episode(self, training: bool) -> None:
        self.base.begin_episode(training=training)

    def select_action(self, obs: np.ndarray, action_mask: List[Tuple[float, float, float, float]], training: bool) -> int:
        queue_nonempty = float(obs[4]) > 1e-6
        if queue_nonempty and self.rng.random() < self.rejection_probability:
            return int(self.reject_index)
        return int(self.base.select_action(obs, action_mask, training=training))

    def observe_transition(self, *args, **kwargs) -> Dict[str, float]:
        return {}

    def end_episode(self, *args, **kwargs) -> Dict[str, float]:
        return {}


class RawVariancePSAIMAgent(QControlAgent):
    """PSAIM ablation that uses total ensemble return variance as a penalty."""

    def __init__(self, state_dim: int, action_dim: int, seed: int) -> None:
        super().__init__(
            "psaim_rawvar",
            state_dim,
            action_dim,
            seed=seed,
            config=make_experiment_2_config(),
            psaim=True,
        )

    def _raw_total_variance(self, next_obs: np.ndarray, done: bool) -> float:
        if done:
            return 0.0
        obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q_heads = self._source_net()(obs_tensor).squeeze(0)
            head_returns = self.config.gamma * self._masked_head_max(q_heads, next_obs)
        return max(0.0, float(torch.var(head_returns, unbiased=False).item()))

    def diagnostic_signal(
        self,
        obs: np.ndarray,
        action: Optional[int] = None,
        future_obs: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, float]:
        next_obs = (future_obs or [obs.copy()])[0]
        raw_var = self._raw_total_variance(next_obs, done=False)
        intrinsic = -self.config.lambda_aleatoric * np.log1p(raw_var / max(self.config.sigma0_sq, 1e-6))
        selected_action = 0 if action is None else int(action)
        return {
            "action": float(selected_action),
            "V_epi": float("nan"),
            "V_ale": float("nan"),
            "V_ale_excess": float("nan"),
            "gate_h3": float("nan"),
            "raw_var": float(raw_var),
            "r_int": float(intrinsic),
        }

    def _prepare_psaim_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        episode_step: Optional[int],
        regime: Optional[str],
    ) -> Dict[str, object]:
        return {
            "obs": obs.copy(),
            "action": int(action),
            "reward": float(reward),
            "next_obs": next_obs.copy(),
            "done": bool(done),
            "bootstrap_mask": self._bootstrap_mask(),
            "key": self._state_action_key(obs, action),
            "transition_features": next_obs.copy(),
            "episode_step": int(episode_step or 0),
            "regime": regime or "unknown",
            "raw_var": self._raw_total_variance(next_obs, done),
        }

    def _finalize_psaim_transition(self, item: Dict[str, object], training: bool, adapt: bool) -> Dict[str, float]:
        raw_var = float(item["raw_var"])
        intrinsic = float(-self.config.lambda_aleatoric * np.log1p(raw_var / max(self.config.sigma0_sq, 1e-6)))
        signal = {
            "V_epi": float("nan"),
            "V_ale": float("nan"),
            "V_ale_excess": float("nan"),
            "gate_h3": float("nan"),
            "raw_var": raw_var,
            "r_int": intrinsic,
        }
        if training or adapt:
            self._update_transition_stats(item["key"], item["transition_features"])
            shaped_reward = float(item["reward"] + self.config.intrinsic_scale * intrinsic)
            self._record_replay_transition(
                item["obs"],
                int(item["action"]),
                shaped_reward,
                item["next_obs"],
                bool(item["done"]),
                item["bootstrap_mask"],
            )
        self.last_signal = signal
        self.completed_signal_rows.append(
            {
                "episode": float(item["episode_step"]),
                "regime": str(item["regime"]),
                "action": float(item["action"]),
                **signal,
            }
        )
        return signal

    def observe_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        training: bool,
        adapt: bool,
        episode_step: Optional[int] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, float]:
        item = self._prepare_psaim_transition(obs, action, reward, next_obs, done, episode_step, regime)
        return self._finalize_psaim_transition(item, training=training, adapt=adapt)


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


def build_agent(method: str, state_dim: int, action_dim: int, seed: int):
    config = make_experiment_2_config()
    if method == "heuristic_sjf_bestfit":
        return HeuristicAgent(action_defer_index=action_dim - 2, action_reject_index=action_dim - 1)
    if method == "psaim_lite":
        return QControlAgent("psaim_lite", state_dim, action_dim, seed=seed + 41, config=config, psaim=True)
    if method == "psaim_rawvar":
        return RawVariancePSAIMAgent(state_dim, action_dim, seed=seed + 141)
    if method.startswith("sjf_r_"):
        probability = float(method.removeprefix("sjf_r_").replace("p", "."))
        return SJFStaticRejectAgent(probability, action_dim=action_dim, seed=seed + int(round(1000 * probability)))
    raise ValueError(f"Unknown mechanism-control method: {method}")


def is_learned_agent(agent) -> bool:
    return isinstance(agent, QControlAgent)


def run_episode_c5(
    env: ProxyAllocationEnv,
    agent,
    runtime: RuntimeConfig,
    training: bool,
    capture_signals: bool = False,
) -> Tuple[Dict[str, float], List[Dict[str, object]], Dict[str, float]]:
    obs, _info = env.reset()
    agent.begin_episode(training=training)
    done = False
    episode_step = 0
    action_counts = {"allocate": 0.0, "defer": 0.0, "reject": 0.0}
    signal_rows: List[Dict[str, object]] = []

    while not done:
        action_mask = visible_queue_mask(obs, env.visible_queue)
        if isinstance(agent, QControlAgent):
            action = agent.select_action(obs, action_mask, training=training, adapt=False)
        else:
            action = agent.select_action(obs, action_mask, training=training)
        action_counts[action_type(int(action), runtime.visible_queue)] += 1.0
        next_obs, reward, done, _step_info = env.step(int(action))
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
        if capture_signals and hasattr(agent, "consume_signal_rows"):
            for row in agent.consume_signal_rows():
                signal_rows.append(enrich_signal_row(row, runtime.visible_queue))
        obs = next_obs
        episode_step += 1

    agent.end_episode(training=training, adapt=False)
    if capture_signals and hasattr(agent, "consume_signal_rows"):
        for row in agent.consume_signal_rows():
            signal_rows.append(enrich_signal_row(row, runtime.visible_queue))

    metrics = env.episode_metrics()
    total_actions = max(1.0, sum(action_counts.values()))
    action_rates = {
        "allocate_rate": action_counts["allocate"] / total_actions,
        "defer_rate": action_counts["defer"] / total_actions,
        "reject_rate": action_counts["reject"] / total_actions,
        "allocate_count": action_counts["allocate"],
        "defer_count": action_counts["defer"],
        "reject_count": action_counts["reject"],
        "total_actions": total_actions,
    }
    return metrics, signal_rows, action_rates


def enrich_signal_row(row: Dict[str, object], visible_queue: int) -> Dict[str, object]:
    action = row.get("action", np.nan)
    action_value = float(action) if isinstance(action, (float, int, np.floating, np.integer)) else float("nan")
    enriched: Dict[str, object] = dict(row)
    enriched["action_type"] = action_type(int(action_value), visible_queue) if np.isfinite(action_value) else "unknown"
    return enriched


def train_agent(agent, seed: int, runtime: RuntimeConfig, workload_mode: str, stress: Optional[StressConfig] = None) -> None:
    if not is_learned_agent(agent):
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
        run_episode_c5(env, agent, runtime, training=True, capture_signals=False)


def evaluate_agent(
    agent,
    seed: int,
    runtime: RuntimeConfig,
    workload_mode: str,
    stress: Optional[StressConfig] = None,
    capture_signals: bool = False,
) -> Tuple[Dict[str, float], pd.DataFrame, Dict[str, float]]:
    env = ProxyAllocationEnv(
        seed=seed,
        workload_mode=workload_mode,
        horizon=runtime.horizon,
        queue_capacity=runtime.queue_capacity,
        visible_queue=runtime.visible_queue,
        alternating_block=runtime.alternating_block,
        stress=stress or make_stress_config("clean", "clean"),
    )
    metrics_rows: List[Dict[str, float]] = []
    signals: List[Dict[str, object]] = []
    count_totals = {"allocate_count": 0.0, "defer_count": 0.0, "reject_count": 0.0, "total_actions": 0.0}
    for episode_idx in range(runtime.eval_episodes):
        metrics, episode_signals, action_rates = run_episode_c5(
            env,
            agent,
            runtime,
            training=False,
            capture_signals=capture_signals,
        )
        metrics_rows.append(metrics)
        for row in episode_signals:
            row.update({"eval_episode": episode_idx})
            signals.append(row)
        for key in count_totals:
            count_totals[key] += float(action_rates[key])
    metrics_mean = pd.DataFrame(metrics_rows).mean(numeric_only=True).to_dict()
    total_actions = max(1.0, count_totals["total_actions"])
    action_summary = {
        "allocate_rate": count_totals["allocate_count"] / total_actions,
        "defer_rate": count_totals["defer_count"] / total_actions,
        "reject_rate": count_totals["reject_count"] / total_actions,
        **count_totals,
    }
    return metrics_mean, pd.DataFrame(signals), action_summary


def log_child_run(
    runtime: RuntimeConfig,
    run_name: str,
    tags: Dict[str, str],
    params: Dict[str, object],
    metrics: Dict[str, float],
) -> None:
    if not runtime.use_mlflow or mlflow is None:
        return
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.set_tags({**tags, "paper_revision": runtime.paper_revision})
        mlflow.log_params({key: str(value) for key, value in params.items()})
        for key, value in metrics.items():
            if isinstance(value, (float, int, np.floating, np.integer)) and np.isfinite(float(value)):
                mlflow.log_metric(key, float(value))


def run_duration_misreport_controls(runtime: RuntimeConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    methods = [
        "heuristic_sjf_bestfit",
        "sjf_r_0p10",
        "sjf_r_0p17",
        "sjf_r_0p18",
        "sjf_r_0p25",
        "psaim_lite",
        "psaim_rawvar",
    ]
    result_rows: List[Dict[str, object]] = []
    action_rows: List[Dict[str, object]] = []

    for seed in range(runtime.seeds):
        print(f"[c5-duration] seed {seed}", flush=True)
        state_dim, action_dim = template_dimensions(runtime, 71000 + seed)
        for severity, severity_rank in SEVERITY_ORDER.items():
            stress = make_stress_config("duration_misreport", severity)
            for method in methods:
                set_global_seeds(72000 + (1000 * severity_rank) + (100 * seed))
                agent = build_agent(method, state_dim, action_dim, 73000 + (1000 * severity_rank) + seed)
                train_agent(
                    agent,
                    seed=74000 + (1000 * severity_rank) + seed,
                    runtime=runtime,
                    workload_mode="periodic",
                    stress=stress,
                )
                metrics, _signals, actions = evaluate_agent(
                    agent,
                    seed=75000 + (1000 * severity_rank) + seed,
                    runtime=runtime,
                    workload_mode="periodic",
                    stress=stress,
                    capture_signals=False,
                )
                result_row = {
                    **metrics,
                    "experiment": "c5_duration_misreport_mechanism",
                    "scenario": "duration_misreport",
                    "severity": severity,
                    "severity_rank": severity_rank,
                    "method": method,
                    "method_label": method_label(method),
                    "seed": seed,
                }
                action_row = {
                    **actions,
                    "experiment": "c5_duration_misreport_mechanism",
                    "scenario": "duration_misreport",
                    "severity": severity,
                    "severity_rank": severity_rank,
                    "method": method,
                    "method_label": method_label(method),
                    "seed": seed,
                }
                result_rows.append(result_row)
                action_rows.append(action_row)
                log_child_run(
                    runtime,
                    run_name=f"c5_duration_{method}_seed{seed}_{severity}",
                    tags={
                        "experiment_type": "c5_duration_misreport",
                        "method": method,
                        "seed": str(seed),
                        "stress_scenario": "duration_misreport",
                        "stress_level": severity,
                    },
                    params={"seed": seed, "method": method, "severity": severity, **asdict(stress)},
                    metrics={**metrics, **actions},
                )
    return pd.DataFrame(result_rows), pd.DataFrame(action_rows)


def run_signal_separation(runtime: RuntimeConfig) -> pd.DataFrame:
    methods = ["psaim_lite", "psaim_rawvar"]
    signal_rows: List[Dict[str, object]] = []
    for seed in range(runtime.seeds):
        print(f"[c5-signal] seed {seed}", flush=True)
        state_dim, action_dim = template_dimensions(runtime, 81000 + seed)
        for method in methods:
            set_global_seeds(82000 + (100 * seed))
            agent = build_agent(method, state_dim, action_dim, 83000 + seed)
            train_agent(
                agent,
                seed=84000 + seed,
                runtime=runtime,
                workload_mode="alternating",
                stress=make_stress_config("clean", "clean"),
            )
            metrics, signals, actions = evaluate_agent(
                agent,
                seed=85000 + seed,
                runtime=runtime,
                workload_mode="alternating",
                stress=make_stress_config("clean", "clean"),
                capture_signals=True,
            )
            if not signals.empty:
                signals["experiment"] = "c5_signal_separation"
                signals["scenario"] = "alternating_hidden_regime"
                signals["severity"] = "clean"
                signals["method"] = method
                signals["method_label"] = method_label(method)
                signals["seed"] = seed
                signal_rows.extend(signals.to_dict("records"))
            signal_metrics = {
                **metrics,
                **actions,
                "mean_r_int": float(signals[signals["r_int"].notna()]["r_int"].mean()) if not signals.empty and "r_int" in signals else float("nan"),
            }
            log_child_run(
                runtime,
                run_name=f"c5_signal_{method}_seed{seed}",
                tags={
                    "experiment_type": "c5_signal_separation",
                    "method": method,
                    "seed": str(seed),
                    "stress_scenario": "clean",
                    "stress_level": "clean",
                    "regime": "alternating_hidden",
                },
                params={"seed": seed, "method": method, "workload_mode": "alternating"},
                metrics=signal_metrics,
            )
    return pd.DataFrame(signal_rows)


def summarize_performance(results: pd.DataFrame, methods: Optional[List[str]] = None) -> pd.DataFrame:
    subset = results.copy()
    if methods is not None:
        subset = subset[subset["method"].isin(methods)]
    rows: List[Dict[str, object]] = []
    for (severity_rank, severity, method, label), group in subset.groupby(
        ["severity_rank", "severity", "method", "method_label"], dropna=False
    ):
        rows.append(
            {
                "_severity_rank": int(severity_rank),
                "severity": severity,
                "method": label,
                "average_slowdown": fmt_ci(group["average_slowdown"]),
                "task_failure_rate": fmt_ci(group["task_failure_rate"]),
                "p95_completion_time": fmt_ci(group["p95_completion_time"]),
            }
        )
    table = pd.DataFrame(rows).sort_values(["_severity_rank", "method"]).reset_index(drop=True)
    return table.drop(columns=["_severity_rank"])


def summarize_actions(actions: pd.DataFrame, methods: Optional[List[str]] = None) -> pd.DataFrame:
    subset = actions.copy()
    if methods is not None:
        subset = subset[subset["method"].isin(methods)]
    rows: List[Dict[str, object]] = []
    for (severity_rank, severity, method, label), group in subset.groupby(
        ["severity_rank", "severity", "method", "method_label"], dropna=False
    ):
        rows.append(
            {
                "_severity_rank": int(severity_rank),
                "severity": severity,
                "method": label,
                "allocate_rate": fmt_ci(group["allocate_rate"]),
                "defer_rate": fmt_ci(group["defer_rate"]),
                "reject_rate": fmt_ci(group["reject_rate"]),
            }
        )
    table = pd.DataFrame(rows).sort_values(["_severity_rank", "method"]).reset_index(drop=True)
    return table.drop(columns=["_severity_rank"])


def sjf_reject_pair_table(results: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for severity in ["low", "medium", "high"]:
        subset = results[results["severity"].eq(severity)]
        psaim = subset[subset["method"].eq("psaim_lite")].set_index("seed")
        control = subset[subset["method"].eq("sjf_r_0p17")].set_index("seed")
        for metric in ["average_slowdown", "task_failure_rate"]:
            mean, lo, hi = paired_diff_ci(psaim[metric], control[metric])
            if np.isfinite(hi) and hi < 0.0:
                status = "PSAIM lower"
            elif np.isfinite(lo) and lo > 0.0:
                status = "SJF-R(0.17) lower"
            else:
                status = "draw / CI crosses zero"
            rows.append(
                {
                    "severity": severity,
                    "comparison": "Simplified PSAIM - SJF-R(0.17)",
                    "metric": metric,
                    "paired_diff_mean_ci": f"{fmt(mean)} [{fmt(lo)}, {fmt(hi)}]",
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def rawvar_pair_table(results: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for source, metric in [("performance", "average_slowdown"), ("action_distribution", "reject_rate")]:
        df = results if source == "performance" else actions
        for severity in ["low", "medium", "high"]:
            subset = df[df["severity"].eq(severity)]
            rawvar = subset[subset["method"].eq("psaim_rawvar")].set_index("seed")
            psaim = subset[subset["method"].eq("psaim_lite")].set_index("seed")
            mean, lo, hi = paired_diff_ci(rawvar[metric], psaim[metric])
            if np.isfinite(hi) and hi < 0.0:
                status = "RawVar lower"
            elif np.isfinite(lo) and lo > 0.0:
                status = "RawVar higher"
            else:
                status = "draw / CI crosses zero"
            rows.append(
                {
                    "severity": severity,
                    "comparison": "PSAIM-RawVar - Simplified PSAIM",
                    "metric": metric,
                    "paired_diff_mean_ci": f"{fmt(mean)} [{fmt(lo)}, {fmt(hi)}]",
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def signal_summary(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame(columns=["method", "regime", "metric", "value"])
    rows: List[Dict[str, object]] = []
    metrics = [metric for metric in ["r_int", "raw_var", "V_epi", "V_ale", "gate_h3"] if metric in signals.columns]
    for (method, label, regime), group in signals.groupby(["method", "method_label", "regime"], dropna=False):
        for metric in metrics:
            seed_means = (
                group.assign(_metric=pd.to_numeric(group[metric], errors="coerce"))
                .groupby("seed")["_metric"]
                .mean()
                .dropna()
            )
            rows.append({"method": label, "regime": regime, "metric": metric, "value": fmt_ci(seed_means)})
    return pd.DataFrame(rows).sort_values(["method", "metric", "regime"]).reset_index(drop=True)


def signal_switch_table(signals: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    columns = [
        "method",
        "low_entropy_r_int_mean",
        "high_entropy_r_int_mean",
        "paired_high_minus_low_ci",
        "surprise_agnostic_switch",
    ]
    if signals.empty:
        return pd.DataFrame(rows, columns=columns)
    wide_rows: List[Dict[str, object]] = []
    for (method, seed, regime), group in signals.groupby(["method", "seed", "regime"], dropna=False):
        r_values = pd.to_numeric(group["r_int"], errors="coerce") if "r_int" in group else pd.Series(dtype=float)
        r_values = r_values[np.isfinite(r_values)]
        if not r_values.empty:
            wide_rows.append({"method": method, "seed": seed, "regime": regime, "r_int": float(r_values.mean())})
    wide = pd.DataFrame(wide_rows)
    for method in ["psaim_lite", "psaim_rawvar"]:
        subset = wide[wide["method"].eq(method)]
        low = subset[subset["regime"].eq("low_entropy")].set_index("seed")
        high = subset[subset["regime"].eq("high_entropy")].set_index("seed")
        if low.empty or high.empty:
            continue
        mean, lo, hi = paired_diff_ci(high["r_int"], low["r_int"])
        low_mean = float(low["r_int"].mean())
        high_mean = float(high["r_int"].mean())
        sign_switch = low_mean > 0.0 and high_mean < 0.0
        rows.append(
            {
                "method": method_label(method),
                "low_entropy_r_int_mean": fmt(low_mean),
                "high_entropy_r_int_mean": fmt(high_mean),
                "paired_high_minus_low_ci": f"{fmt(mean)} [{fmt(lo)}, {fmt(hi)}]",
                "surprise_agnostic_switch": "yes" if sign_switch else "no",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def experiment_a_status(pair_table: pd.DataFrame) -> str:
    critical = pair_table[pair_table["metric"].isin(["average_slowdown", "task_failure_rate"])]
    strong = bool((critical["status"] == "PSAIM lower").all()) and len(critical) == 6
    if strong:
        return (
            "supported. Simplified PSAIM is significantly lower than SJF-R(0.17) on both slowdown "
            "and failure rate at all three duration-misreport severities."
        )
    if (critical["status"] == "SJF-R(0.17) lower").any():
        return (
            "refuted. At least one paired comparison favors the static-rejection heuristic, so the "
            "simpler hand-crafted control cannot be dismissed."
        )
    return (
        "partially supported / unresolved. PSAIM is not significantly lower than SJF-R(0.17) on every "
        "required metric and severity, so the strict mechanism claim should be weakened."
    )


def experiment_b_status(raw_pairs: pd.DataFrame, switch: pd.DataFrame) -> str:
    if switch.empty or "method" not in switch.columns:
        return (
            "unresolved. The signal-separation table has no paired low/high-regime rows, so the "
            "RawVar intrinsic-reward switch cannot be assessed from this run."
        )
    rawvar_switch = switch[switch["method"].eq(method_label("psaim_rawvar"))]
    psaim_switch = switch[switch["method"].eq(method_label("psaim_lite"))]
    rawvar_has_switch = False
    if not rawvar_switch.empty:
        rawvar_has_switch = rawvar_switch["surprise_agnostic_switch"].iloc[0] == "yes"
    psaim_has_switch = False
    if not psaim_switch.empty:
        psaim_has_switch = psaim_switch["surprise_agnostic_switch"].iloc[0] == "yes"
    slowdown = raw_pairs[raw_pairs["metric"].eq("average_slowdown")]
    reject = raw_pairs[raw_pairs["metric"].eq("reject_rate")]
    matches_slowdown = bool((slowdown["status"] == "draw / CI crosses zero").all()) and len(slowdown) == 3
    matches_reject = bool((reject["status"] == "draw / CI crosses zero").all()) and len(reject) == 3
    rawvar_not_worse_slowdown = bool((slowdown["status"].isin(["RawVar lower", "draw / CI crosses zero"])).all()) and len(slowdown) == 3
    rawvar_similar_or_lower_reject = bool((reject["status"].isin(["RawVar lower", "draw / CI crosses zero"])).all()) and len(reject) == 3
    if not psaim_has_switch:
        operational = (
            "PSAIM-RawVar matches or improves the downstream slowdown/rejection behavior"
            if rawvar_not_worse_slowdown and rawvar_similar_or_lower_reject
            else "PSAIM-RawVar differs on downstream behavior"
        )
        return (
            "mixed and claim-limiting. Simplified PSAIM itself does not show the strict positive-low / "
            "negative-high intrinsic-reward sign switch in this mechanism-control signal audit, so this run cannot "
            "support the paper's strict sign-switch wording. "
            f"{operational}, which means the explicit decomposition is not shown to be operationally "
            "load-bearing for the duration-misreport advantage under this proxy."
        )
    if matches_slowdown and matches_reject and rawvar_has_switch:
        return (
            "refuted for empirical necessity. PSAIM-RawVar matches simplified PSAIM on slowdown, rejection, "
            "and the signal switch, so the decomposition is not load-bearing in this proxy."
        )
    if not rawvar_has_switch:
        return (
            "supported for the signal mechanism. PSAIM-RawVar does not show the required positive-low / "
            "negative-high intrinsic-reward switch, so the explicit decomposition remains mechanistically "
            "relevant for the signal claim, even if downstream metrics must be interpreted separately."
        )
    return (
        "partially supported. PSAIM-RawVar shows the signal switch but differs on downstream behavior, "
        "indicating that the gate or aleatoric penalty, rather than decomposition alone, is likely load-bearing."
    )


def write_outputs(
    results: pd.DataFrame,
    actions: pd.DataFrame,
    signals: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "c5_duration_misreport_raw_results.csv", index=False)
    actions.to_csv(output_dir / "c5_duration_misreport_action_distribution_raw.csv", index=False)
    signals.to_csv(output_dir / "c5_signal_separation_raw.csv", index=False)

    experiment_a_methods = ["psaim_lite", "sjf_r_0p10", "sjf_r_0p17", "sjf_r_0p18", "sjf_r_0p25"]
    experiment_b_methods = ["heuristic_sjf_bestfit", "psaim_lite", "psaim_rawvar"]
    exp_a_perf = summarize_performance(results, methods=experiment_a_methods)
    exp_a_pairs = sjf_reject_pair_table(results)
    exp_b_perf = summarize_performance(results, methods=experiment_b_methods)
    exp_b_actions = summarize_actions(actions, methods=experiment_b_methods)
    exp_b_pairs = rawvar_pair_table(results, actions)
    exp_b_signals = signal_summary(signals)
    exp_b_switch = signal_switch_table(signals)

    tables = {
        "c5_experiment_a_sjf_static_rejection_performance": exp_a_perf,
        "c5_experiment_a_psaim_vs_sjf_r_0p17_pairs": exp_a_pairs,
        "c5_experiment_b_rawvar_performance": exp_b_perf,
        "c5_experiment_b_rawvar_action_distribution": exp_b_actions,
        "c5_experiment_b_rawvar_pairs": exp_b_pairs,
        "c5_experiment_b_signal_summary": exp_b_signals,
        "c5_experiment_b_signal_switch": exp_b_switch,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
        (output_dir / f"{name}.md").write_text(to_markdown_table(table), encoding="utf-8")

    lines = [
        "# PSAIM Mechanism-Control Follow-Up Experiments",
        "",
        "All experiments use the existing single-node, two-resource proxy environment and existing simplified PSAIM hyperparameters. "
        "The simulator and duration-misreport workload generator were not changed. Values are 10-seed means with 95% nonparametric bootstrap CIs unless this was a smoke run with fewer seeds. "
        "Paired comparisons use same-seed paired bootstrap with 5000 resamples; negative paired differences mean the left method is lower/better.",
        "",
        "## Experiment A: SJF + Static-Rejection Control",
        "",
        "Hypothesis: PSAIM's duration-misreport advantage is mechanism-driven, not reproduced by an SJF heuristic with a uniform random rejection prior.",
        "",
        "### Performance Table",
        "",
        to_markdown_table(exp_a_perf),
        "### Paired Comparison: Simplified PSAIM vs SJF-R(0.17)",
        "",
        to_markdown_table(exp_a_pairs),
        "### Experiment A Interpretation",
        "",
        f"Status: {experiment_a_status(exp_a_pairs)}",
        "",
        "## Experiment B: Raw Total-Variance Ablation",
        "",
        "Hypothesis: PSAIM's epistemic-aleatoric decomposition is responsible for its behavior beyond a simpler total-variance penalty.",
        "",
        "### Downstream Duration-Misreport Performance",
        "",
        to_markdown_table(exp_b_perf),
        "### Action-Distribution Audit",
        "",
        to_markdown_table(exp_b_actions),
        "### Paired Comparison: PSAIM-RawVar vs Simplified PSAIM",
        "",
        to_markdown_table(exp_b_pairs),
        "### Held-Out Signal-Separation Summary",
        "",
        to_markdown_table(exp_b_signals),
        "### Intrinsic-Reward Switch Check",
        "",
        to_markdown_table(exp_b_switch),
        "### Experiment B Interpretation",
        "",
        f"Status: {experiment_b_status(exp_b_pairs, exp_b_switch)}",
        "",
        "## Protocol Notes",
        "",
        "- SJF-R(p) rejects the head-of-queue task with probability p whenever the queue is non-empty; otherwise it delegates to the existing SJF best-fit heuristic. The rejection coin flip is uniform random and does not inspect state features.",
        "- PSAIM-RawVar uses the same K=5 scalar Q-head ensemble and main PSAIM lambda/sigma0 values, but replaces the intrinsic reward with `-lambda * log(1 + Var(Y_k)/sigma0^2)` and uses no gate or epistemic/aleatoric decomposition.",
        "- The RawVar implementation computes `Y_k = gamma * max_a Q_k(S', a)` over valid actions for the observed next state. No extra next-state sampler or simulator branch was introduced.",
        "- Draws are reported as draws; no directional claim should be made when the paired CI crosses zero.",
    ]
    (output_dir / "C5_MECHANISM_CONTROL_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "run_manifest.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mechanism-control follow-up experiments for PSAIM.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "experiments" / "data" / "results" / "c5_mechanism_controls_v1"))
    parser.add_argument("--paper-output-dir", default=str(PROJECT_ROOT / "experiments" / "output" / "c5_mechanism_controls_v1"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--queue-capacity", type=int, default=12)
    parser.add_argument("--visible-queue", type=int, default=5)
    parser.add_argument("--alternating-block", type=int, default=24)
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5001")
    parser.add_argument("--experiment-name", default="cisose-rl-proxy-evaluation")
    parser.add_argument("--run-name", default="c5_psaim_mechanism_controls")
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
        paper_revision="c5-mechanism-controls",
    )
    results_dir = Path(args.results_dir)
    paper_output_dir = Path(args.paper_output_dir)

    parent_ctx = None
    if args.use_mlflow:
        if mlflow is None:
            raise RuntimeError("MLflow requested but not importable. Set PYTHONPATH to experiments/.deps/mlflow.")
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment_name)
        parent_ctx = mlflow.start_run(run_name=args.run_name)
        mlflow.set_tags({"experiment_type": "c5_mechanism_controls", "paper_revision": runtime.paper_revision})
        mlflow.log_params({**asdict(runtime), "script": "run_c5_mechanism_controls.py"})

    try:
        results, actions = run_duration_misreport_controls(runtime)
        signals = run_signal_separation(runtime)
        write_outputs(results, actions, signals, results_dir, args)
        if paper_output_dir.resolve() != results_dir.resolve():
            if paper_output_dir.exists():
                shutil.rmtree(paper_output_dir)
            shutil.copytree(results_dir, paper_output_dir)
        if args.use_mlflow and mlflow is not None:
            mlflow.log_artifacts(str(results_dir), artifact_path="c5_mechanism_controls")
            mlflow.log_metric("c5_duration_rows", float(len(results)))
            mlflow.log_metric("c5_signal_rows", float(len(signals)))
        print(f"Wrote mechanism-control outputs to {results_dir}")
        print(f"Wrote paper-facing mechanism-control outputs to {paper_output_dir}")
    finally:
        if parent_ctx is not None and mlflow is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
