from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "experiments" / "data" / "demo"


def make_stress_robustness() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    scenarios = [
        "observation_noise",
        "reward_corruption",
        "distribution_shift",
        "co_tenant_interference",
        "co_tenant_matched_load_control",
    ]
    severity_rank = {"low": 1, "medium": 2, "high": 3}
    severity_names = list(severity_rank)
    methods = ["heuristic_sjf_bestfit", "deeprm_inspired_pg", "plain_dqn"]
    method_offset = {
        "heuristic_sjf_bestfit": 0.0,
        "deeprm_inspired_pg": 2.5,
        "plain_dqn": 1.2,
    }
    scenario_offset = {
        "observation_noise": 1.0,
        "reward_corruption": 1.7,
        "distribution_shift": 2.8,
        "co_tenant_interference": 3.5,
        "co_tenant_matched_load_control": 1.5,
    }
    rows = []

    for method in methods:
        for seed in range(10):
            clean_slowdown = 1.20 + method_offset[method] * 0.1 + rng.normal(0, 0.03)
            clean_completion = 17.0 + method_offset[method] + rng.normal(0, 0.4)
            clean_util = 0.69 + method_offset[method] * 0.01 + rng.normal(0, 0.01)
            clean_fail = max(0.0, 0.012 + method_offset[method] * 0.003 + rng.normal(0, 0.002))
            clean_p95 = 26.0 + method_offset[method] * 1.3 + rng.normal(0, 0.7)
            rows.append(
                {
                    "scenario": "clean",
                    "severity": "clean",
                    "severity_rank": 0,
                    "method": method,
                    "seed": seed,
                    "average_slowdown": clean_slowdown,
                    "mean_completion_time": clean_completion,
                    "p95_completion_time": clean_p95,
                    "utilization": clean_util,
                    "task_failure_rate": clean_fail,
                    "degradation_pct": 0.0,
                }
            )

            for scenario in scenarios:
                for severity in severity_names:
                    rank = severity_rank[severity]
                    base = scenario_offset[scenario] * rank
                    if method == "heuristic_sjf_bestfit":
                        degrade = 3.0 + 1.1 * base + rng.normal(0, 0.9)
                    elif method == "plain_dqn":
                        degrade = 5.5 + 1.9 * base + rng.normal(0, 1.2)
                    else:
                        degrade = 6.8 + 2.3 * base + rng.normal(0, 1.4)
                    slowdown = clean_slowdown * (1 + degrade / 100.0)
                    completion = clean_completion * (1 + (degrade * 0.75) / 100.0)
                    p95 = clean_p95 * (1 + (degrade * 1.05) / 100.0)
                    util_drop = 0.01 * rank if scenario != "co_tenant_matched_load_control" else 0.003 * rank
                    fail_bump = 0.006 * rank * (1.0 if method == "heuristic_sjf_bestfit" else 1.4)
                    rows.append(
                        {
                            "scenario": scenario,
                            "severity": severity,
                            "severity_rank": rank,
                            "method": method,
                            "seed": seed,
                            "average_slowdown": slowdown,
                            "mean_completion_time": completion,
                            "p95_completion_time": p95,
                            "utilization": max(0.4, clean_util - util_drop + rng.normal(0, 0.008)),
                            "task_failure_rate": min(1.0, clean_fail + fail_bump + rng.normal(0, 0.003)),
                            "degradation_pct": max(0.0, degrade),
                        }
                    )
    return pd.DataFrame(rows)


def make_psaim_signals() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    rows = []
    block_size = 25
    total_episodes = 150

    for seed in range(10):
        for episode in range(total_episodes):
            block = episode // block_size
            regime = "low_entropy" if block % 2 == 0 else "high_entropy"
            within = episode % block_size

            if regime == "low_entropy":
                v_epi = 1.35 - 0.028 * within + rng.normal(0, 0.03)
                v_ale = 0.42 + 0.01 * np.sin(within / 4.0) + rng.normal(0, 0.02)
                r_int = 0.50 - 0.012 * within + rng.normal(0, 0.03)
            else:
                v_epi = 1.10 - 0.014 * within + rng.normal(0, 0.03)
                v_ale = 0.95 + 0.015 * np.cos(within / 4.0) + rng.normal(0, 0.03)
                r_int = -0.10 - 0.008 * within + rng.normal(0, 0.03)

            rows.extend(
                [
                    {
                        "episode": episode,
                        "seed": seed,
                        "regime": regime,
                        "block": block,
                        "metric": "V_epi",
                        "value": v_epi,
                    },
                    {
                        "episode": episode,
                        "seed": seed,
                        "regime": regime,
                        "block": block,
                        "metric": "V_ale",
                        "value": v_ale,
                    },
                    {
                        "episode": episode,
                        "seed": seed,
                        "regime": regime,
                        "block": block,
                        "metric": "r_int",
                        "value": r_int,
                    },
                ]
            )
    return pd.DataFrame(rows)


def make_downstream_performance() -> pd.DataFrame:
    rng = np.random.default_rng(29)
    methods = [
        "heuristic_sjf_bestfit",
        "deeprm_inspired_pg",
        "plain_dqn",
        "dqn_rnd",
        "psaim_lite",
        "psaim_no_aleatoric",
        "psaim_no_gate",
        "psaim_no_freezing",
    ]
    condition_offset = {"clean": 0.0, "stressed": 0.28}
    method_quality = {
        "heuristic_sjf_bestfit": 0.18,
        "deeprm_inspired_pg": 0.24,
        "plain_dqn": 0.16,
        "dqn_rnd": 0.10,
        "psaim_lite": 0.0,
        "psaim_no_aleatoric": 0.08,
        "psaim_no_gate": 0.06,
        "psaim_no_freezing": 0.11,
    }
    rows = []

    for method in methods:
        for seed in range(10):
            for condition in ("clean", "stressed"):
                quality = method_quality[method] + condition_offset[condition]
                slowdown = 1.22 + quality + rng.normal(0, 0.04)
                completion = 16.8 + 4.0 * quality + rng.normal(0, 0.5)
                p95 = 25.0 + 5.8 * quality + rng.normal(0, 0.7)
                failure = max(0.0, 0.010 + 0.028 * quality + rng.normal(0, 0.002))
                utilization = 0.72 - 0.05 * quality + rng.normal(0, 0.01)
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "condition": condition,
                        "average_slowdown": slowdown,
                        "mean_completion_time": completion,
                        "p95_completion_time": p95,
                        "task_failure_rate": failure,
                        "utilization": utilization,
                    }
                )
    return pd.DataFrame(rows)


def make_adaptation_lag() -> pd.DataFrame:
    rng = np.random.default_rng(41)
    methods = [
        "plain_dqn",
        "dqn_rnd",
        "psaim_lite",
        "psaim_no_aleatoric",
        "psaim_no_gate",
        "psaim_no_freezing",
    ]
    lag_base = {
        "plain_dqn": 35,
        "dqn_rnd": 27,
        "psaim_lite": 17,
        "psaim_no_aleatoric": 23,
        "psaim_no_gate": 24,
        "psaim_no_freezing": 29,
    }
    rows = []
    for method in methods:
        for seed in range(10):
            for switch_index in range(1, 6):
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "switch_index": switch_index,
                        "adaptation_lag_steps": lag_base[method] + rng.normal(0, 2.4),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    make_stress_robustness().to_csv(DATA_DIR / "stress_robustness.csv", index=False)
    make_psaim_signals().to_csv(DATA_DIR / "psaim_signals.csv", index=False)
    make_downstream_performance().to_csv(DATA_DIR / "downstream_performance.csv", index=False)
    make_adaptation_lag().to_csv(DATA_DIR / "adaptation_lag.csv", index=False)
    print(f"Wrote demo datasets to {DATA_DIR}")


if __name__ == "__main__":
    main()
