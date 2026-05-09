from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from paper_style import COLORBLIND_PALETTE, label_panels, save_figure, setup_style

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "experiments" / "data" / "results" / "paper_final_claim_aligned_v3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "output" / "figures"


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(123)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate_with_ci(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, key_tuple))
        values = group[value_col].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(values)
        row.update({"mean": values.mean(), "ci_low": lo, "ci_high": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_pipeline_schematic(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_axis_off()

    boxes = [
        ((0.03, 0.55), 0.18, 0.22, "Workload Generator\nLow/high entropy blocks\nStress scenarios"),
        ((0.28, 0.55), 0.18, 0.22, "Proxy Environment\nSingle node\nCPU + memory"),
        ((0.53, 0.55), 0.18, 0.22, "Learners\nHeuristic, PG, DQN,\nDQN+RND, Simplified PSAIM"),
        ((0.78, 0.55), 0.18, 0.22, "Isolated MLflow\n127.0.0.1:5001\nSQLite + local artifacts"),
        ((0.28, 0.15), 0.18, 0.22, "Metrics Store\nSlowdown, p95,\nutilization, failures"),
        ((0.53, 0.15), 0.18, 0.22, "Figure Pipeline\nDeterministic styles\nPDF/SVG/PNG"),
        ((0.78, 0.15), 0.18, 0.22, "Paper Assets\nMain figures\nMain + appendix tables"),
    ]

    for idx, ((x, y), w, h, text) in enumerate(boxes):
        patch = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.2,
            facecolor=COLORBLIND_PALETTE[idx % len(COLORBLIND_PALETTE)],
            alpha=0.15,
            edgecolor=COLORBLIND_PALETTE[idx % len(COLORBLIND_PALETTE)],
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    arrow_kw = dict(arrowstyle="->", linewidth=1.4, color="#444444")
    ax.annotate("", xy=(0.28, 0.66), xytext=(0.21, 0.66), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.53, 0.66), xytext=(0.46, 0.66), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.78, 0.66), xytext=(0.71, 0.66), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.37, 0.37), xytext=(0.37, 0.55), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.62, 0.37), xytext=(0.62, 0.55), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.87, 0.37), xytext=(0.87, 0.55), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.53, 0.26), xytext=(0.46, 0.26), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.78, 0.26), xytext=(0.71, 0.26), arrowprops=arrow_kw)
    ax.set_title("Proxy Evaluation and Observability Pipeline", pad=14)
    save_figure(fig, "fig_1_proxy_pipeline", output_dir=output_dir)
    plt.close(fig)


def plot_robustness_degradation(data_dir: Path, output_dir: Path) -> None:
    df = pd.read_csv(data_dir / "stress_robustness.csv")
    df = df[df["scenario"].ne("clean") & df["scenario"].ne("co_tenant_matched_load_control")]
    agg = aggregate_with_ci(df, ["scenario", "severity_rank", "method"], "degradation_pct")

    scenarios = [
        "observation_noise",
        "reward_corruption",
        "distribution_shift",
        "co_tenant_interference",
    ]
    scenario_titles = {
        "observation_noise": "Observation Noise",
        "reward_corruption": "Reward Corruption",
        "distribution_shift": "Distribution Shift",
        "co_tenant_interference": "Co-tenant Interference",
    }
    method_colors = {
        "heuristic_sjf_bestfit": COLORBLIND_PALETTE[0],
        "deeprm_inspired_pg": COLORBLIND_PALETTE[3],
        "plain_dqn": COLORBLIND_PALETTE[2],
    }
    label_map = {
        "heuristic_sjf_bestfit": "Heuristic",
        "deeprm_inspired_pg": "PG proxy",
        "plain_dqn": "DQN",
    }

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharex=True, sharey=True)
    for ax, scenario in zip(axes.flat, scenarios):
        subset = agg[agg["scenario"].eq(scenario)].sort_values("severity_rank")
        for method, group in subset.groupby("method"):
            ax.plot(group["severity_rank"], group["mean"], marker="o", color=method_colors[method], label=label_map[method])
            ax.fill_between(group["severity_rank"], group["ci_low"], group["ci_high"], color=method_colors[method], alpha=0.16)
        ax.set_title(scenario_titles[scenario])
        ax.set_xticks([1, 2, 3], ["Low", "Med", "High"])
        ax.set_ylabel("Degradation vs clean (%)")
        ax.set_xlabel("Stress severity")

    axes[0, 0].legend(loc="upper left", ncol=1)
    label_panels(axes.flat)
    fig.suptitle("Robustness Degradation Across Orchestration-Like Stressors", y=1.02, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig_2_robustness_degradation", output_dir=output_dir)
    plt.close(fig)


def add_regime_shading(ax: plt.Axes, total_episodes: int, block_size: int) -> None:
    for start in range(0, total_episodes, block_size):
        end = min(total_episodes, start + block_size)
        regime = "low" if (start // block_size) % 2 == 0 else "high"
        color = "#d8ecff" if regime == "low" else "#ffe7d1"
        ax.axvspan(start, end, color=color, alpha=0.35, linewidth=0)


def plot_psaim_signals(data_dir: Path, output_dir: Path, alternating_block: int) -> None:
    df = pd.read_csv(data_dir / "psaim_signals.csv")
    metrics = ["V_epi", "V_ale", "r_int"]
    titles = {
        "V_epi": "Epistemic signal $V_{epi}$",
        "V_ale": "Aleatoric signal $V_{ale}$",
        "r_int": "Intrinsic reward $r_{int}$",
    }
    ylabels = {
        "V_epi": "Signal value",
        "V_ale": "Signal value",
        "r_int": "Reward value",
    }
    colors = {"V_epi": COLORBLIND_PALETTE[0], "V_ale": COLORBLIND_PALETTE[1], "r_int": COLORBLIND_PALETTE[2]}

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), sharex=True)
    for ax, metric in zip(axes, metrics):
        subset = df[df["metric"].eq(metric)]
        agg = aggregate_with_ci(subset, ["episode"], "value").sort_values("episode")
        add_regime_shading(ax, total_episodes=int(subset["episode"].max() + 1), block_size=alternating_block)
        ax.plot(agg["episode"], agg["mean"], color=colors[metric])
        ax.fill_between(agg["episode"], agg["ci_low"], agg["ci_high"], color=colors[metric], alpha=0.18)
        ax.set_ylabel(ylabels[metric])
        ax.set_title(titles[metric], loc="left")
    axes[-1].set_xlabel("Decision step within held-out evaluation episode")
    label_panels(axes)
    fig.suptitle("Simplified PSAIM Signals Across Hidden Entropy Regime Blocks", y=1.01, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig_3_psaim_signals", output_dir=output_dir)
    plt.close(fig)


def plot_downstream_comparison(data_dir: Path, output_dir: Path) -> None:
    df = pd.read_csv(data_dir / "downstream_performance.csv")
    methods = [
        "heuristic_sjf_bestfit",
        "deeprm_inspired_pg",
        "plain_dqn",
        "dqn_rnd",
        "psaim_lite",
    ]
    label_map = {
        "heuristic_sjf_bestfit": "Heuristic",
        "deeprm_inspired_pg": "PG proxy",
        "plain_dqn": "DQN",
        "dqn_rnd": "DQN+RND",
        "psaim_lite": "Simplified PSAIM",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    for ax, metric, title in zip(
        axes,
        ["average_slowdown", "task_failure_rate"],
        ["Average slowdown", "Task failure rate"],
    ):
        subset = df[df["method"].isin(methods)]
        agg = aggregate_with_ci(subset, ["method", "condition"], metric)
        x = np.arange(len(methods))
        width = 0.34
        for idx, condition in enumerate(["clean", "stressed"]):
            cond = agg[agg["condition"].eq(condition)].set_index("method").loc[methods].reset_index()
            centers = x + (idx - 0.5) * width
            ax.bar(
                centers,
                cond["mean"],
                width=width,
                color=COLORBLIND_PALETTE[idx + 4],
                alpha=0.85,
                label=condition.capitalize(),
            )
            err_low = cond["mean"] - cond["ci_low"]
            err_high = cond["ci_high"] - cond["mean"]
            ax.errorbar(centers, cond["mean"], yerr=[err_low, err_high], fmt="none", color="black", capsize=3, linewidth=1)
        ax.set_xticks(x, [label_map[m] for m in methods], rotation=18)
        ax.set_title(title)
        ax.set_ylabel(title)

    axes[0].legend(loc="upper left")
    label_panels(axes)
    fig.suptitle("Downstream Clean and Stressed Performance", y=1.02, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig_4_downstream_comparison", output_dir=output_dir)
    plt.close(fig)


def plot_adaptation_lag(data_dir: Path, output_dir: Path) -> None:
    df = pd.read_csv(data_dir / "adaptation_lag.csv")
    methods = ["plain_dqn", "dqn_rnd", "psaim_lite", "psaim_no_aleatoric", "psaim_no_gate", "psaim_no_freezing"]
    label_map = {
        "plain_dqn": "DQN",
        "dqn_rnd": "DQN+RND",
        "psaim_lite": "Simplified PSAIM",
        "psaim_no_aleatoric": "No aleatoric",
        "psaim_no_gate": "No gate",
        "psaim_no_freezing": "No freezing",
    }
    agg = aggregate_with_ci(df, ["method"], "adaptation_lag_steps").set_index("method").loc[methods].reset_index()
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.bar(x, agg["mean"], color=[COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)] for i in range(len(methods))], alpha=0.85)
    err_low = agg["mean"] - agg["ci_low"]
    err_high = agg["ci_high"] - agg["mean"]
    ax.errorbar(x, agg["mean"], yerr=[err_low, err_high], fmt="none", color="black", capsize=3, linewidth=1)
    ax.set_xticks(x, [label_map[m] for m in methods], rotation=16)
    ax.set_ylabel("Adaptation lag (steps)")
    ax.set_title("Adaptation Lag After Hidden Regime Transitions")
    label_panels([ax], x=-0.05, y=1.03)
    fig.tight_layout()
    save_figure(fig, "fig_5_adaptation_lag", output_dir=output_dir)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figures from experiment CSVs.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--alternating-block", type=int, default=24)
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5001")
    parser.add_argument("--experiment-name", default="cisose-rl-proxy-evaluation")
    parser.add_argument("--run-name", default="paper_figure_generation")
    parser.add_argument("--use-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    parent_ctx = None
    if args.use_mlflow:
        if mlflow is None:
            raise RuntimeError("MLflow was requested but is not importable in this Python environment.")
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment_name)
        parent_ctx = mlflow.start_run(run_name=args.run_name)
        mlflow.set_tags({"experiment_type": "figure_generation", "paper_revision": "proxy-study-v1"})
        mlflow.log_params(
            {
                "data_dir": str(data_dir),
                "output_dir": str(output_dir),
                "alternating_block": args.alternating_block,
            }
        )
    try:
        plot_pipeline_schematic(output_dir=output_dir)
        plot_robustness_degradation(data_dir=data_dir, output_dir=output_dir)
        plot_psaim_signals(data_dir=data_dir, output_dir=output_dir, alternating_block=args.alternating_block)
        plot_downstream_comparison(data_dir=data_dir, output_dir=output_dir)
        plot_adaptation_lag(data_dir=data_dir, output_dir=output_dir)
        if args.use_mlflow and mlflow is not None:
            mlflow.log_artifacts(str(output_dir), artifact_path="figures")
    finally:
        if parent_ctx is not None:
            mlflow.end_run()
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
