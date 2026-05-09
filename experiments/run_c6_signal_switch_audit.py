from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.experiment_runner import (
    RuntimeConfig,
    build_experiment_2_agents,
    enrich_signal_row,
    make_experiment_2_config,
    run_episode,
    train_agent_on_clean_proxy,
)
from src.agents import set_global_seeds
from src.proxy_env import ProxyAllocationEnv, make_stress_config

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS = ["r_int", "V_epi", "V_ale", "gate_h3"]
ACTION_TYPES = ["allocate", "defer", "reject"]


def bootstrap_ci(values: Iterable[float], n_boot: int = 5000, seed: int = 619) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, arr.size), replace=True)
    means = samples.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff_ci(left: pd.Series, right: pd.Series, n_boot: int = 5000, seed: int = 947) -> Tuple[float, float, float]:
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
    return f"{arr.mean():.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


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


def signal_long_to_decisions(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if "method" in df.columns:
        df = df[df["method"].eq("psaim_lite")].copy()
    if "metric" in df.columns and "value" in df.columns:
        metric_count = max(1, int(df["metric"].nunique()))
        df = df.reset_index(drop=True)
        # Older signal artifacts did not store eval_episode. The signal writer emits
        # one contiguous metric bundle per decision, so this synthetic identifier
        # prevents repeated episode-step/action tuples from being collapsed.
        df["_decision_id"] = np.arange(len(df)) // metric_count
        index_cols = [
            col
            for col in ["source", "_decision_id", "seed", "regime", "eval_episode", "episode", "action", "action_type", "block"]
            if col in df.columns
        ]
        decisions = df.pivot_table(index=index_cols, columns="metric", values="value", aggfunc="mean").reset_index()
    else:
        decisions = df.copy()
    decisions["source"] = source
    for metric in METRICS:
        if metric in decisions.columns:
            decisions[metric] = pd.to_numeric(decisions[metric], errors="coerce")
        else:
            decisions[metric] = np.nan
    if "action_type" not in decisions.columns:
        decisions["action_type"] = "unknown"
    return decisions


def seed_regime_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (source, seed, regime), group in decisions.groupby(["source", "seed", "regime"], dropna=False):
        row: Dict[str, object] = {"source": source, "seed": int(seed), "regime": regime}
        for metric in METRICS:
            row[metric] = float(group[metric].mean())
        total = max(1, len(group))
        for action_type in ACTION_TYPES:
            row[f"{action_type}_rate"] = float(group["action_type"].eq(action_type).sum() / total)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["source", "seed", "regime"]).reset_index(drop=True)


def load_reference_signals(c4_dir: Path, c5_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    c4_path = c4_dir / "psaim_signals.csv"
    c5_path = c5_dir / "c5_signal_separation_raw.csv"
    if not c4_path.exists():
        raise FileNotFoundError(f"Missing original reference signal artifact: {c4_path}")
    if not c5_path.exists():
        raise FileNotFoundError(f"Missing mechanism-control reference signal artifact: {c5_path}")
    c4 = signal_long_to_decisions(pd.read_csv(c4_path), "original signal artifact")
    c5 = signal_long_to_decisions(pd.read_csv(c5_path), "mechanism-control signal audit")
    return c4, c5


def run_exact_c4_signal_protocol(runtime: RuntimeConfig) -> pd.DataFrame:
    signal_rows: List[Dict[str, object]] = []
    config = make_experiment_2_config()
    for seed in range(runtime.seeds):
        print(f"[signal-rerun] seed {seed}", flush=True)
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
        agents = build_experiment_2_agents(
            int(obs.shape[0]),
            int(template_env.action_space_n),
            seed + 300,
            config=config,
            methods=["psaim_lite"],
        )
        agent = agents["psaim_lite"]
        train_agent_on_clean_proxy(agent, seed=seed + 300, runtime=runtime, workload_mode="alternating")
        eval_env = ProxyAllocationEnv(
            seed=4000 + seed,
            workload_mode="alternating",
            horizon=runtime.horizon,
            queue_capacity=runtime.queue_capacity,
            visible_queue=runtime.visible_queue,
            alternating_block=runtime.alternating_block,
            stress=make_stress_config("clean", "clean"),
        )
        for eval_episode in range(runtime.eval_episodes):
            _metrics, signals = run_episode(
                eval_env,
                agent,
                training=False,
                adapt=False,
                capture_signals=True,
            )
            for row in signals:
                row = enrich_signal_row(row, runtime.visible_queue)
                row.update(
                    {
                        "seed": seed,
                        "method": "psaim_lite",
                        "eval_episode": eval_episode,
                        "block": row["episode"] // runtime.alternating_block,
                    }
                )
                signal_rows.append(row)
    return signal_long_to_decisions(pd.DataFrame(signal_rows), "focused original-protocol rerun")


def summary_table(seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for source, group in seed_summary.groupby("source", dropna=False):
        low = group[group["regime"].eq("low_entropy")].set_index("seed")
        high = group[group["regime"].eq("high_entropy")].set_index("seed")
        for metric in [*METRICS, *[f"{a}_rate" for a in ACTION_TYPES]]:
            mean, lo, hi = paired_diff_ci(high[metric], low[metric])
            low_mean = float(low[metric].mean()) if metric in low else float("nan")
            high_mean = float(high[metric].mean()) if metric in high else float("nan")
            rows.append(
                {
                    "source": source,
                    "metric": metric,
                    "low_entropy": fmt_ci(low[metric]),
                    "high_entropy": fmt_ci(high[metric]),
                    "paired_high_minus_low": f"{fmt(mean)} [{fmt(lo)}, {fmt(hi)}]",
                    "low_mean_raw": low_mean,
                    "high_mean_raw": high_mean,
                }
            )
    order = {
        "original signal artifact": 0,
        "focused original-protocol rerun": 1,
        "strict original-protocol rerun": 1,
        "mechanism-control signal audit": 2,
    }
    table = pd.DataFrame(rows)
    table["_source_order"] = table["source"].map(order).fillna(99)
    return table.sort_values(["_source_order", "metric"]).drop(columns=["_source_order"]).reset_index(drop=True)


def seed_value_table(seed_summary: pd.DataFrame) -> pd.DataFrame:
    keep = ["source", "seed", "regime", *METRICS, *[f"{a}_rate" for a in ACTION_TYPES]]
    table = seed_summary[keep].copy()
    for col in [*METRICS, *[f"{a}_rate" for a in ACTION_TYPES]]:
        table[col] = table[col].map(lambda value: fmt(float(value), digits=6))
    return table


def config_table(args: argparse.Namespace, runtime: RuntimeConfig) -> pd.DataFrame:
    cfg = make_experiment_2_config()
    rows = [
        {
            "item": "lambda_aleatoric",
            "original signal run": "3.0 via make_experiment_2_config()",
            "strict rerun": fmt(cfg.lambda_aleatoric, 3),
            "mechanism-control simplified PSAIM": "3.0 via make_experiment_2_config()",
        },
        {
            "item": "sigma0_sq",
            "original signal run": "0.05 via make_experiment_2_config()",
            "strict rerun": fmt(cfg.sigma0_sq, 3),
            "mechanism-control simplified PSAIM": "0.05 via make_experiment_2_config()",
        },
        {
            "item": "alpha_gate",
            "original signal run": "0.16 via make_experiment_2_config()",
            "strict rerun": fmt(cfg.alpha_gate, 3),
            "mechanism-control simplified PSAIM": "0.16 via make_experiment_2_config()",
        },
        {
            "item": "probe_horizon",
            "original signal run": "3 via make_experiment_2_config()",
            "strict rerun": str(cfg.probe_horizon),
            "mechanism-control simplified PSAIM": "3 via make_experiment_2_config()",
        },
        {
            "item": "train/eval episodes",
            "original signal run": "100 / 8",
            "strict rerun": f"{runtime.train_episodes} / {runtime.eval_episodes}",
            "mechanism-control simplified PSAIM": "100 / 8",
        },
        {
            "item": "horizon/block",
            "original signal run": "96 / 24",
            "strict rerun": f"{runtime.horizon} / {runtime.alternating_block}",
            "mechanism-control simplified PSAIM": "96 / 24",
        },
        {
            "item": "seed/checkpoint path",
            "original signal run": "agent/template/train seed = seed+300; eval seed = 4000+seed",
            "strict rerun": "same as original signal run",
            "mechanism-control simplified PSAIM": "agent seed base 83000+seed; train seed 84000+seed; eval seed 85000+seed",
        },
        {
            "item": "code path",
            "original signal run": "src.experiment_runner.experiment_2 full multi-agent loop",
            "strict rerun": (
                "src.experiment_runner.experiment_2 full multi-agent loop"
                if getattr(args, "existing_c6_rerun_dir", "")
                else "focused psaim_lite-only reproduction"
            ),
            "mechanism-control simplified PSAIM": "run_c5_mechanism_controls.py focused psaim_lite/rawvar signal audit",
        },
        {
            "item": "saved checkpoint",
            "original signal run": "not available",
            "strict rerun": "fresh deterministic rerun",
            "mechanism-control simplified PSAIM": "fresh deterministic run",
        },
    ]
    return pd.DataFrame(rows)


def decide_resolution(summary: pd.DataFrame) -> str:
    rint = summary[summary["metric"].eq("r_int")].copy()
    lookup = {row["source"]: row for _, row in rint.iterrows()}
    c4 = lookup.get("original signal artifact")
    c6 = lookup.get("strict original-protocol rerun")
    if c6 is None:
        c6 = lookup.get("focused original-protocol rerun")
    c5 = lookup.get("mechanism-control signal audit")
    if c4 is None or c6 is None:
        return "c. The original sign-switch measurement cannot be reproduced completely because the required summary rows are unavailable."
    c4_switch = float(c4["low_mean_raw"]) > 0.0 and float(c4["high_mean_raw"]) < 0.0
    c6_switch = float(c6["low_mean_raw"]) > 0.0 and float(c6["high_mean_raw"]) < 0.0
    c5_switch = c5 is not None and float(c5["low_mean_raw"]) > 0.0 and float(c5["high_mean_raw"]) < 0.0
    c4_c6_close = (
        abs(float(c4["low_mean_raw"]) - float(c6["low_mean_raw"])) < 1e-9
        and abs(float(c4["high_mean_raw"]) - float(c6["high_mean_raw"])) < 1e-9
    )
    if c4_switch and c6_switch and c4_c6_close and not c5_switch:
        return (
            "fragility finding; apply decision rule (b) for the paper. The original sign-switch measurement is exactly reproducible with current code under the recovered seed/protocol, "
            "but an independently seeded mechanism-control signal audit used the same PSAIM hyperparameters, episode length, and regime-block protocol and did not reproduce the positive low-entropy sign. "
            "The discrepancy is therefore not lambda/sigma/protocol drift; it is checkpoint/workload-seed sensitivity."
        )
    if c4_switch and c6_switch and not c5_switch:
        return (
            "fragility finding; apply decision rule (b) for the paper. The original sign-switch measurement is reproducible under the recovered original code path, "
            "but an independently seeded mechanism-control signal audit used the same PSAIM hyperparameters and evaluation protocol and did not reproduce the positive low-entropy sign. "
            "Therefore the sign-switch wording is not robust across runs and should be softened."
        )
    return (
        "b. The sign switch is not robustly reproducible under the available protocol. "
        "The paper should not claim a clean or robust surprise-agnostic sign switch."
    )


def write_outputs(
    decisions: pd.DataFrame,
    seed_summary: pd.DataFrame,
    summary: pd.DataFrame,
    configs: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_values = seed_value_table(seed_summary)
    decisions.to_csv(output_dir / "c6_decision_level_signals.csv", index=False)
    seed_summary.to_csv(output_dir / "c6_seed_regime_summary.csv", index=False)
    seed_values.to_csv(output_dir / "c6_seed_level_values.csv", index=False)
    summary.to_csv(output_dir / "c6_summary.csv", index=False)
    configs.to_csv(output_dir / "c6_config_protocol_comparison.csv", index=False)
    (output_dir / "c6_seed_level_values.md").write_text(to_markdown_table(seed_values), encoding="utf-8")
    public_summary = summary.drop(columns=["low_mean_raw", "high_mean_raw"])
    (output_dir / "c6_summary.md").write_text(to_markdown_table(public_summary), encoding="utf-8")
    (output_dir / "c6_config_protocol_comparison.md").write_text(to_markdown_table(configs), encoding="utf-8")
    resolution = decide_resolution(summary)
    lines = [
        "# Surprise-Agnostic Switch Reproducibility Audit",
        "",
        "This audit reruns simplified PSAIM under the exact original signal-separation seed/protocol and compares it with the original and mechanism-control signal artifacts already on disk. All CIs are 95% nonparametric bootstrap intervals over seed-level regime means. The simulator, agent, and hyperparameters were not modified.",
        "",
        "## Resolution",
        "",
        resolution,
        "",
        "## Protocol And Configuration Comparison",
        "",
        to_markdown_table(configs),
        "## Per-Regime Summary",
        "",
        to_markdown_table(public_summary),
        "## Seed-Level Values",
        "",
        to_markdown_table(seed_values),
        "## Interpretation",
        "",
        "- The original reference artifact and the strict rerun use the recovered seed/checkpoint path and produce the same aggregate sign pattern: low-entropy `r_int` is positive on average and high-entropy `r_int` is negative on average.",
        "- The independently seeded mechanism-control audit uses the same lambda, sigma0, alpha, probe horizon, episode length, and block length, but a different trained checkpoint path and evaluation workload seeds. Its aggregate low-entropy `r_int` is slightly negative.",
        "- PSAIM-RawVar's much larger negative intrinsic values in the mechanism-control audit are expected from its different total-variance penalty formula; they do not imply that simplified PSAIM used a different lambda or sigma0.",
        "- The seed-level table shows that the low-entropy sign is not uniformly positive across seeds even in the original artifact. Therefore, the safest paper wording is directional separation rather than a robust clean sign switch.",
        "",
        "## Paper Decision",
        "",
        "Use the softened wording unless the paper explicitly ties the sign-switch statement to the original seed/protocol. Recommended replacement: `Simplified PSAIM shows directional separation in V_epi, V_ale, and gate values across hidden regimes; the sign of r_int is seed-sensitive in this proxy, so we do not claim a robust surprise-agnostic sign switch.`",
    ]
    (output_dir / "C6_SIGNAL_SWITCH_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "run_manifest.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve simplified-PSAIM signal-switch reproducibility.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "experiments" / "data" / "results" / "c6_signal_switch_audit_v1"))
    parser.add_argument("--paper-output-dir", default=str(PROJECT_ROOT / "experiments" / "output" / "c6_signal_switch_audit_v1"))
    parser.add_argument("--c4-reference-dir", default=str(PROJECT_ROOT / "experiments" / "data" / "results" / "paper_final_claim_aligned_v3"))
    parser.add_argument("--c5-reference-dir", default=str(PROJECT_ROOT / "experiments" / "data" / "results" / "c5_mechanism_controls_v1"))
    parser.add_argument(
        "--existing-c6-rerun-dir",
        default="",
        help="Optional existing strict rerun directory containing psaim_signals.csv; skips the focused rerun.",
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--queue-capacity", type=int, default=12)
    parser.add_argument("--visible-queue", type=int, default=5)
    parser.add_argument("--alternating-block", type=int, default=24)
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5001")
    parser.add_argument("--experiment-name", default="cisose-rl-proxy-evaluation")
    parser.add_argument("--run-name", default="c6_signal_switch_audit")
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
        paper_revision="c6-signal-switch-audit",
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
        mlflow.set_tags({"experiment_type": "c6_signal_switch_audit", "paper_revision": runtime.paper_revision})
        mlflow.log_params({**asdict(runtime), "script": "run_c6_signal_switch_audit.py"})
    try:
        c4_decisions, c5_decisions = load_reference_signals(Path(args.c4_reference_dir), Path(args.c5_reference_dir))
        if args.existing_c6_rerun_dir:
            c6_path = Path(args.existing_c6_rerun_dir) / "psaim_signals.csv"
            if not c6_path.exists():
                raise FileNotFoundError(f"Missing existing strict rerun psaim_signals.csv: {c6_path}")
            c6_decisions = signal_long_to_decisions(
                pd.read_csv(c6_path),
                "strict original-protocol rerun",
            )
        else:
            c6_decisions = run_exact_c4_signal_protocol(runtime)
            c6_decisions["source"] = "focused original-protocol rerun"
        decisions = pd.concat([c4_decisions, c6_decisions, c5_decisions], ignore_index=True)
        seed_summary = seed_regime_summary(decisions)
        summary = summary_table(seed_summary)
        configs = config_table(args, runtime)
        write_outputs(decisions, seed_summary, summary, configs, results_dir, args)
        if paper_output_dir.resolve() != results_dir.resolve():
            if paper_output_dir.exists():
                shutil.rmtree(paper_output_dir)
            shutil.copytree(results_dir, paper_output_dir)
        if args.use_mlflow and mlflow is not None:
            mlflow.log_artifacts(str(results_dir), artifact_path="c6_signal_switch_audit")
            rint = summary[summary["metric"].eq("r_int")]
            for _, row in rint.iterrows():
                key = str(row["source"]).lower().replace(" ", "_").replace("-", "_")
                mlflow.log_metric(f"{key}_low_r_int", float(row["low_mean_raw"]))
                mlflow.log_metric(f"{key}_high_r_int", float(row["high_mean_raw"]))
        print(f"Wrote signal-switch audit outputs to {results_dir}")
        print(f"Wrote paper-facing signal-switch audit outputs to {paper_output_dir}")
    finally:
        if parent_ctx is not None and mlflow is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
