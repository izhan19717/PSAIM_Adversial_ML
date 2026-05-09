from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INITIAL_DIR = PROJECT_ROOT / "experiments" / "data" / "results" / "c4_positive_demo_v1"
EXTENDED_DIR = PROJECT_ROOT / "experiments" / "data" / "results" / "c4_reward_drift_v2"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "output" / "c4_integrated_final"
FIGURE_DIR = OUTPUT_DIR / "figures"

METHOD_LABELS = {
    "heuristic_sjf_bestfit": "SJF heuristic",
    "deeprm_inspired_pg": "PG proxy",
    "plain_dqn": "DQN",
    "dqn_rnd": "DQN+RND",
    "psaim_lite": "Simplified PSAIM",
    "psaim_no_gate": "No gate",
    "psaim_no_freezing": "No freezing",
}
PALETTE = {
    "heuristic_sjf_bestfit": "#4c78a8",
    "deeprm_inspired_pg": "#f58518",
    "plain_dqn": "#54a24b",
    "dqn_rnd": "#b279a2",
    "psaim_lite": "#e45756",
    "psaim_no_gate": "#72b7b2",
    "psaim_no_freezing": "#ff9da6",
}
SEVERITY_ORDER = {"clean": 0, "low": 1, "medium": 2, "high": 3, "long_horizon": 4}
METHOD_ORDER = {
    "SJF heuristic": 0,
    "PG proxy": 1,
    "DQN": 2,
    "DQN+RND": 3,
    "Simplified PSAIM": 4,
    "No gate": 5,
    "No freezing": 6,
}


def bootstrap_ci(values: Iterable[float], n_boot: int = 5000, seed: int = 971) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True)
    means = samples.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff_ci(left: pd.Series, right: pd.Series, n_boot: int = 5000, seed: int = 319) -> Tuple[float, float, float]:
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


def fmt_ci(values: pd.Series, digits: int = 3) -> str:
    arr = values.to_numpy(dtype=float)
    mean = float(np.nanmean(arr))
    lo, hi = bootstrap_ci(arr)
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


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


def aggregate_table(
    df: pd.DataFrame,
    group_cols: List[str],
    metrics: List[str],
    method_labels: bool = True,
) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, key_tuple))
        if method_labels and "method" in row:
            row["method"] = METHOD_LABELS.get(row["method"], row["method"])
        for metric in metrics:
            row[metric] = fmt_ci(group[metric])
        rows.append(row)
    table = pd.DataFrame(rows)
    sort_cols = []
    if "severity" in table.columns:
        table["_severity_order"] = table["severity"].map(SEVERITY_ORDER).fillna(99)
        sort_cols.append("_severity_order")
    if "method" in table.columns:
        table["_method_order"] = table["method"].map(METHOD_ORDER).fillna(99)
        sort_cols.append("_method_order")
    sort_cols.extend([col for col in group_cols if col in table.columns and col not in {"severity", "method"}])
    table = table.sort_values(sort_cols).drop(columns=[col for col in ["_severity_order", "_method_order"] if col in table.columns])
    return table.reset_index(drop=True)


def pair_rows(
    df: pd.DataFrame,
    left_method: str,
    right_method: str,
    group_cols: List[str],
    metrics: List[str],
) -> pd.DataFrame:
    rows = []
    grouped = [((), df)] if not group_cols else df.groupby(group_cols, dropna=False)
    for keys, group in grouped:
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_cols, key_tuple)) if group_cols else {}
        left = group[group["method"].eq(left_method)].set_index("seed")
        right = group[group["method"].eq(right_method)].set_index("seed")
        for metric in metrics:
            mean, lo, hi = paired_diff_ci(left[metric], right[metric])
            if np.isfinite(hi) and hi < 0.0:
                status = f"{METHOD_LABELS[left_method]} lower"
            elif np.isfinite(lo) and lo <= 0.0 <= hi:
                status = "not decisive"
            else:
                status = f"{METHOD_LABELS[right_method]} lower"
            rows.append(
                {
                    **base,
                    "comparison": f"{METHOD_LABELS[left_method]} - {METHOD_LABELS[right_method]}",
                    "metric": metric,
                    "paired_diff_mean_ci": f"{fmt(mean)} [{fmt(lo)}, {fmt(hi)}]",
                    "status": status,
                }
            )
    table = pd.DataFrame(rows)
    if "severity" in table.columns:
        table["_severity_order"] = table["severity"].map(SEVERITY_ORDER).fillna(99)
        table = table.sort_values(["_severity_order", "comparison", "metric"]).drop(columns=["_severity_order"])
    return table.reset_index(drop=True)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in {
        "pdf": {},
        "svg": {},
        "png": {"dpi": 300},
    }.items():
        fig.savefig(FIGURE_DIR / f"{stem}.{ext}", bbox_inches="tight", **kwargs)


def plot_bar_with_ci(ax: plt.Axes, df: pd.DataFrame, methods: List[str], metric: str, title: str) -> None:
    x = np.arange(len(methods))
    means, lows, highs = [], [], []
    for method in methods:
        values = df[df["method"].eq(method)][metric]
        mean = float(values.mean())
        lo, hi = bootstrap_ci(values)
        means.append(mean)
        lows.append(mean - lo)
        highs.append(hi - mean)
    ax.bar(x, means, color=[PALETTE[m] for m in methods], alpha=0.86)
    ax.errorbar(x, means, yerr=[lows, highs], fmt="none", color="black", capsize=3, linewidth=1)
    ax.set_xticks(x, [METHOD_LABELS[m] for m in methods], rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(metric.replace("_", " "))


def plot_initial_reward_high(initial_df: pd.DataFrame) -> None:
    df = initial_df[
        initial_df["experiment"].eq("reward_corruption_robustness")
        & initial_df["scenario"].eq("reward_corruption")
        & initial_df["severity"].eq("high")
    ]
    methods = ["heuristic_sjf_bestfit", "deeprm_inspired_pg", "plain_dqn", "dqn_rnd", "psaim_lite"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    plot_bar_with_ci(axes[0], df, methods, "average_slowdown", "High Reward Corruption: Stressed Slowdown")
    plot_bar_with_ci(axes[1], df, methods, "degradation_pct", "High Reward Corruption: Degradation")
    axes[1].axhspan(0, 50, color="#54a24b", alpha=0.10, label="target 0-50%")
    axes[1].legend(loc="upper left")
    fig.suptitle("Initial C4 Reward-Corruption Result Refutes the High-Severity Hypothesis")
    fig.tight_layout()
    save_figure(fig, "fig_c4_1_initial_reward_corruption")
    plt.close(fig)


def plot_duration_misreport(initial_df: pd.DataFrame) -> None:
    df = initial_df[
        initial_df["experiment"].eq("heuristic_breaking_shift")
        & initial_df["scenario"].eq("duration_misreport")
    ]
    severities = ["low", "medium", "high"]
    methods = ["heuristic_sjf_bestfit", "psaim_lite"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharex=True)
    width = 0.36
    x = np.arange(len(severities))
    for idx, method in enumerate(methods):
        means, lows, highs = [], [], []
        for severity in severities:
            values = df[df["severity"].eq(severity) & df["method"].eq(method)]["average_slowdown"]
            mean = float(values.mean())
            lo, hi = bootstrap_ci(values)
            means.append(mean)
            lows.append(mean - lo)
            highs.append(hi - mean)
        centers = x + (idx - 0.5) * width
        axes[0].bar(centers, means, width=width, color=PALETTE[method], alpha=0.86, label=METHOD_LABELS[method])
        axes[0].errorbar(centers, means, yerr=[lows, highs], fmt="none", color="black", capsize=3, linewidth=1)
    axes[0].set_title("Average Slowdown Under Duration Misreporting")
    axes[0].set_ylabel("average slowdown")
    axes[0].set_xticks(x, [s.capitalize() for s in severities])
    axes[0].legend()

    pair = pair_rows(
        df,
        "psaim_lite",
        "heuristic_sjf_bestfit",
        ["severity"],
        ["average_slowdown"],
    )
    diffs = []
    lows = []
    highs = []
    for severity in severities:
        left = df[df["severity"].eq(severity) & df["method"].eq("psaim_lite")].set_index("seed")["average_slowdown"]
        right = df[df["severity"].eq(severity) & df["method"].eq("heuristic_sjf_bestfit")].set_index("seed")["average_slowdown"]
        mean, lo, hi = paired_diff_ci(left, right)
        diffs.append(mean)
        lows.append(mean - lo)
        highs.append(hi - mean)
    axes[1].bar(x, diffs, color=PALETTE["psaim_lite"], alpha=0.86)
    axes[1].errorbar(x, diffs, yerr=[lows, highs], fmt="none", color="black", capsize=3, linewidth=1)
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].set_xticks(x, [s.capitalize() for s in severities])
    axes[1].set_title("Paired Difference: PSAIM - SJF")
    axes[1].set_ylabel("slowdown difference")
    fig.suptitle("Positive Demonstration: Simplified PSAIM Beats SJF When Duration Ordering Is Misreported")
    fig.tight_layout()
    save_figure(fig, "fig_c4_2_duration_misreport_positive_demo")
    plt.close(fig)


def plot_reward_training_sweep(extended_df: pd.DataFrame) -> None:
    df = extended_df[
        extended_df["experiment"].eq("reward_corruption_training_sweep")
        & extended_df["scenario"].eq("reward_corruption_training")
    ]
    severities = ["low", "medium", "high"]
    methods = ["plain_dqn", "dqn_rnd", "psaim_lite"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharex=True)
    width = 0.25
    x = np.arange(len(severities))
    for idx, method in enumerate(methods):
        for ax, metric in zip(axes, ["average_slowdown", "degradation_pct"]):
            means, lows, highs = [], [], []
            for severity in severities:
                values = df[df["severity"].eq(severity) & df["method"].eq(method)][metric]
                mean = float(values.mean())
                lo, hi = bootstrap_ci(values)
                means.append(mean)
                lows.append(mean - lo)
                highs.append(hi - mean)
            centers = x + (idx - 1) * width
            ax.bar(centers, means, width=width, color=PALETTE[method], alpha=0.86, label=METHOD_LABELS[method])
            ax.errorbar(centers, means, yerr=[lows, highs], fmt="none", color="black", capsize=3, linewidth=1)
    axes[0].set_title("Training-Time Reward Corruption: Slowdown")
    axes[1].set_title("Training-Time Reward Corruption: Degradation")
    axes[0].set_ylabel("average slowdown")
    axes[1].set_ylabel("degradation vs clean (%)")
    for ax in axes:
        ax.set_xticks(x, [s.capitalize() for s in severities])
    axes[1].axhspan(0, 50, color="#54a24b", alpha=0.10)
    axes[0].legend(loc="upper left")
    fig.suptitle("Reward-Corruption Extension: PSAIM Helps at Low Severity but Not at High Severity")
    fig.tight_layout()
    save_figure(fig, "fig_c4_3_reward_training_sweep")
    plt.close(fig)


def plot_online_and_drift(extended_df: pd.DataFrame) -> None:
    online = extended_df[
        extended_df["experiment"].eq("reward_corruption_online_adaptation")
        & extended_df["scenario"].eq("reward_corruption_online")
        & extended_df["severity"].eq("high")
    ]
    drift = extended_df[extended_df["experiment"].eq("long_horizon_monotonic_drift")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    plot_bar_with_ci(axes[0], online, ["plain_dqn", "dqn_rnd", "psaim_lite"], "average_slowdown", "Online High Reward Corruption")
    plot_bar_with_ci(axes[1], drift, ["psaim_lite", "psaim_no_gate", "psaim_no_freezing"], "average_slowdown", "Long-Horizon Monotonic Drift")
    fig.suptitle("Secondary C4 Results: Online Reward Adaptation and Drift Ablations")
    fig.tight_layout()
    save_figure(fig, "fig_c4_4_online_reward_and_long_drift")
    plt.close(fig)


def build_integrated_report(initial_df: pd.DataFrame, extended_df: pd.DataFrame) -> str:
    metrics = ["average_slowdown", "degradation_pct", "task_failure_rate", "p95_completion_time"]
    initial_reward = initial_df[
        initial_df["experiment"].eq("reward_corruption_robustness")
        & initial_df["scenario"].eq("reward_corruption")
        & initial_df["severity"].eq("high")
    ]
    initial_reward_table = aggregate_table(initial_reward, ["method"], metrics)
    initial_reward_pair = pair_rows(initial_reward, "psaim_lite", "plain_dqn", [], metrics)

    duration = initial_df[
        initial_df["experiment"].eq("heuristic_breaking_shift")
        & initial_df["scenario"].eq("duration_misreport")
    ]
    duration_table = aggregate_table(duration, ["severity", "method"], metrics)
    duration_pair = pair_rows(duration, "psaim_lite", "heuristic_sjf_bestfit", ["severity"], ["average_slowdown", "degradation_pct", "task_failure_rate", "p95_completion_time"])

    reward_train = extended_df[
        extended_df["experiment"].eq("reward_corruption_training_sweep")
        & extended_df["scenario"].eq("reward_corruption_training")
    ]
    reward_train_table = aggregate_table(reward_train, ["severity", "method"], metrics)
    reward_train_pair = pair_rows(reward_train, "psaim_lite", "plain_dqn", ["severity"], metrics)

    reward_online = extended_df[
        extended_df["experiment"].eq("reward_corruption_online_adaptation")
        & extended_df["scenario"].eq("reward_corruption_online")
    ]
    reward_online_table = aggregate_table(reward_online, ["severity", "method"], metrics)
    reward_online_pair = pair_rows(reward_online, "psaim_lite", "plain_dqn", ["severity"], metrics)

    drift = extended_df[extended_df["experiment"].eq("long_horizon_monotonic_drift")]
    drift_table = aggregate_table(drift, ["method"], metrics)
    drift_pair = pair_rows(drift, "psaim_no_gate", "psaim_lite", [], metrics)
    drift_pair = pd.concat(
        [drift_pair, pair_rows(drift, "psaim_no_freezing", "psaim_lite", [], metrics)],
        ignore_index=True,
    )

    lines: List[str] = [
        "# Integrated C4 Empirical Report",
        "",
        "This report integrates the initial C4 positive-demonstration run and the later reward-corruption/long-horizon-drift extension. All values are 10-seed means with 95% nonparametric bootstrap confidence intervals. Paired comparisons use same-seed paired bootstrap differences with 5000 resamples. Negative paired differences mean the first method listed is lower/better.",
        "",
        "## Executive Verdict",
        "",
        "| Claim target | Status | Principal-scientist reading |",
        "| - | - | - |",
        "| PSAIM beats or statistically ties SJF in at least one operational regime | Strongly supported | Duration-misreporting breaks SJF's shortest-job ordering assumption. Simplified PSAIM has significantly lower average slowdown at low, medium, and high misreport severities. |",
        "| High-severity reward-corruption robustness from intrinsic reward | Refuted / not supported | Initial C4 and the later high-severity training-time sweep do not show PSAIM degradation in the desired 0-50% range. At high severity, PSAIM and DQN both collapse or are statistically indistinguishable. |",
        "| Low-severity reward-corruption robustness | Supported | In the training-time reward sweep, PSAIM has much lower slowdown, degradation, task-failure rate, and p95 completion than DQN at low reward-corruption severity. |",
        "| Online high reward-corruption adaptation | Partially supported | PSAIM has a statistically lower stressed slowdown than DQN, but degradation, failure-rate, and p95 differences are not decisive. |",
        "| Long-horizon drift gate/freezing mechanisms | Weak / partial | No-gate is not worse than full PSAIM. No-freezing is directionally worse, but paired CIs cross zero. This is not strong evidence that gate/freezing are load-bearing. |",
        "",
        "## Paper-Safe Claims",
        "",
        "- Strong: `Under a duration-misreport distribution shift that violates SJF's job-ordering assumption, simplified PSAIM significantly outperforms SJF best-fit on average slowdown across all tested severities.`",
        "- Strong with caveat: `The same duration-misreport regime also reduces PSAIM task-failure rate relative to SJF, although SJF retains lower p95 completion time in this proxy.`",
        "- Moderate: `Simplified PSAIM is robust to low-severity reward corruption relative to DQN.`",
        "- Weak/partial: `Under online high reward corruption, simplified PSAIM slightly improves slowdown relative to DQN, but the evidence is not broad across other metrics.`",
        "- Do not claim: `Simplified PSAIM is robust to severe training-time reward corruption.`",
        "- Do not claim: `The gate is load-bearing under long-horizon drift.`",
        "",
        "## Initial C4: High-Severity Reward-Corruption Robustness",
        "",
        "Hypothesis tested: PSAIM's intrinsic-reward signal is less coupled to corrupted extrinsic reward than bare DQN, so PSAIM should degrade less than DQN under severe reward corruption. The target positive demonstration was PSAIM degradation in the 0-50% range while DQN remains strongly degraded.",
        "",
        to_markdown_table(initial_reward_table),
        "",
        "Paired comparison against DQN:",
        "",
        to_markdown_table(initial_reward_pair),
        "",
        "Reading: refuted. Initial C4 shows PSAIM slowdown `14.850 [11.956, 17.237]` and degradation `352.924% [256.663, 434.635]`, while DQN slowdown is `12.801 [9.728, 15.655]` and degradation `274.814% [190.368, 351.466]`. The paired PSAIM-DQN differences are not decisive, and the PSAIM degradation is far outside the desired 0-50% band.",
        "",
        "## Initial C4: Heuristic-Breaking Duration Misreporting",
        "",
        "Hypothesis tested: when observed job durations are biased so SJF's shortest-job assumption is violated, PSAIM should remain robust while SJF degrades.",
        "",
        to_markdown_table(duration_table),
        "",
        "Paired comparison against SJF:",
        "",
        to_markdown_table(duration_pair),
        "",
        "Reading: strongly supported for average slowdown and degradation. PSAIM beats SJF at all three severities with paired CIs entirely below zero. The strongest paper claim should be framed around average slowdown and failure rate. Caveat: SJF still has lower p95 completion time, so we should not claim PSAIM dominates every metric.",
        "",
        "## Extended C4: Reward-Corruption Training-Time Sweep",
        "",
        "This run repeats reward corruption at low, medium, and high severities with PSAIM hyperparameters unchanged.",
        "",
        to_markdown_table(reward_train_table),
        "",
        "Paired comparison against DQN:",
        "",
        to_markdown_table(reward_train_pair),
        "",
        "Reading: low severity is supported; medium and high are not. At low severity, PSAIM-DQN slowdown difference is `-2.066 [-4.137, -0.653]` and degradation difference is `-59.725 [-117.389, -18.968]`. At high severity, PSAIM-DQN slowdown difference is `-0.117 [-4.754, 4.386]`, which is not decisive, and degradation remains around `290.702%` for PSAIM. This means the intrinsic signal is not enough to protect the agent from severe reward-channel poisoning.",
        "",
        "## Extended C4: Online High Reward-Corruption Adaptation",
        "",
        "This variant clean-trains agents and then allows online adaptation during high reward-corruption evaluation.",
        "",
        to_markdown_table(reward_online_table),
        "",
        "Paired comparison against DQN:",
        "",
        to_markdown_table(reward_online_pair),
        "",
        "Reading: partially supported. PSAIM has lower slowdown than DQN with paired difference `-0.092 [-0.167, -0.012]`, but degradation, failure-rate, and p95 completion-time differences are not decisive. This is a narrow result, not a broad reward-corruption robustness result.",
        "",
        "## Extended C4: Long-Horizon Monotonic Drift",
        "",
        "This run uses 5x longer training exposure and a 5x longer monotonic-drift evaluation episode. It compares full simplified PSAIM against no-gate and no-freezing ablations.",
        "",
        to_markdown_table(drift_table),
        "",
        "Paired ablation comparisons:",
        "",
        to_markdown_table(drift_pair),
        "",
        "Reading: weak/partial. No-gate is not worse than full PSAIM. No-freezing is directionally worse on slowdown (`0.354 [-0.019, 0.901]`) and p95 (`5.648 [-6.946, 21.652]`), but the CIs cross zero. We can describe this as suggestive, not conclusive.",
        "",
        "## Why The Reward-Corruption Hypothesis Fails At High Severity",
        "",
        "The current simplified PSAIM is not reward-corruption-aware. It adds a small intrinsic reward to the same corrupted extrinsic reward used by DQN. In the main configuration, `intrinsic_scale=0.001`, so under severe reward corruption the Bellman target is still dominated by corrupted extrinsic reward. The method estimates transition uncertainty and entropy-regime structure; it does not explicitly estimate reward-channel trustworthiness.",
        "",
        "Therefore, the high-severity failure is not merely a bad dataset draw. It is a theory-to-implementation mismatch: severe reward corruption attacks the reward labels directly, while simplified PSAIM's implemented defenses operate mainly on state-transition uncertainty. A stronger reward-corruption claim would require an explicit reward-trust mechanism, such as reward prediction ensembles, reward-disagreement gates, SLO consistency checks, robust target clipping, or downweighting extrinsic reward when reward corruption is detected.",
        "",
        "## Figure Package",
        "",
        "- `fig_c4_1_initial_reward_corruption`: high-severity reward-corruption refutation.",
        "- `fig_c4_2_duration_misreport_positive_demo`: positive demonstration against SJF.",
        "- `fig_c4_3_reward_training_sweep`: low/medium/high reward-corruption sweep.",
        "- `fig_c4_4_online_reward_and_long_drift`: online corruption and drift ablations.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    initial_df = pd.read_csv(INITIAL_DIR / "c4_raw_results.csv")
    extended_df = pd.read_csv(EXTENDED_DIR / "raw_results.csv")

    plot_initial_reward_high(initial_df)
    plot_duration_misreport(initial_df)
    plot_reward_training_sweep(extended_df)
    plot_online_and_drift(extended_df)

    report = build_integrated_report(initial_df, extended_df)
    (OUTPUT_DIR / "C4_INTEGRATED_FINAL_REPORT.md").write_text(report, encoding="utf-8")

    initial_df.to_csv(OUTPUT_DIR / "initial_c4_raw_results.csv", index=False)
    extended_df.to_csv(OUTPUT_DIR / "extended_c4_raw_results.csv", index=False)
    print(f"Wrote integrated C4 report to {OUTPUT_DIR}")
    print(f"Wrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
