from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def bootstrap_ci(values: Iterable[float], n_boot: int = 5000, seed: int = 173) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True)
    means = samples.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff_ci(left: pd.Series, right: pd.Series, n_boot: int = 5000, seed: int = 491) -> Tuple[float, float, float]:
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


def verdict(ok: bool, weak: bool = False) -> str:
    if ok:
        return "SUPPORTED"
    if weak:
        return "WEAK / PARTIAL"
    return "UNSUPPORTED"


def display_method(method: str) -> str:
    labels = {
        "heuristic_sjf_bestfit": "heuristic",
        "deeprm_inspired_pg": "PG proxy",
        "plain_dqn": "DQN",
        "dqn_rnd": "DQN+RND",
        "psaim_lite": "simplified PSAIM",
        "psaim_no_aleatoric": "simplified PSAIM, no aleatoric penalty",
        "psaim_no_gate": "simplified PSAIM, no gate",
        "psaim_no_freezing": "simplified PSAIM, no freezing",
    }
    return labels.get(method, method)


def analyze_exp1(data_dir: Path) -> List[str]:
    path = data_dir / "stress_robustness.csv"
    if not path.exists():
        return ["## Experiment 1 Stress Evidence", "", f"Missing `{path}`.", ""]
    df = pd.read_csv(path)
    scenarios = ["observation_noise", "reward_corruption", "distribution_shift", "co_tenant_interference"]
    methods = ["deeprm_inspired_pg", "plain_dqn"]
    lines = ["## Experiment 1 Stress Evidence", ""]
    lines.append("| method | scenario | high-severity degradation mean [95% CI] | claim status |")
    lines.append("| - | - | - | - |")
    for method in methods:
        for scenario in scenarios:
            subset = df[
                df["method"].eq(method)
                & df["scenario"].eq(scenario)
                & df["severity"].eq("high")
            ]
            values = subset["degradation_pct"].to_numpy(dtype=float)
            mean = float(np.mean(values)) if len(values) else float("nan")
            lo, hi = bootstrap_ci(values)
            supported = np.isfinite(lo) and lo > 0.0
            weak = np.isfinite(mean) and mean > 0.0
            lines.append(
                f"| {display_method(method)} | {scenario} | {fmt(mean)} [{fmt(lo)}, {fmt(hi)}] | {verdict(supported, weak)} |"
            )
    lines.append("")
    lines.append(
        "Reviewer rule used here: a perturbation is strongly supported only when the 95% bootstrap CI for high-severity degradation is entirely above zero. Positive mean with a crossing CI is marked partial."
    )
    lines.append("")
    return lines


def analyze_exp2_performance(data_dir: Path) -> List[str]:
    path = data_dir / "downstream_performance.csv"
    if not path.exists():
        return ["## Experiment 2 Downstream Evidence", "", f"Missing `{path}`.", ""]
    df = pd.read_csv(path)
    stressed = df[df["condition"].eq("stressed")]
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
    lines = ["## Experiment 2 Downstream Evidence", ""]
    lines.append("| method | stressed slowdown mean [95% CI] |")
    lines.append("| - | - |")
    for method in methods:
        values = stressed[stressed["method"].eq(method)]["average_slowdown"]
        if values.empty:
            continue
        mean = float(values.mean())
        lo, hi = bootstrap_ci(values)
        lines.append(f"| {display_method(method)} | {fmt(mean)} [{fmt(lo)}, {fmt(hi)}] |")
    lines.append("")
    baseline_methods = ["heuristic_sjf_bestfit", "deeprm_inspired_pg", "plain_dqn", "dqn_rnd"]
    psaim = stressed[stressed["method"].eq("psaim_lite")].set_index("seed")["average_slowdown"]
    lines.append("| comparison | PSAIM slowdown minus comparator [95% CI] | superiority status |")
    lines.append("| - | - | - |")
    for method in baseline_methods:
        other = stressed[stressed["method"].eq(method)].set_index("seed")["average_slowdown"]
        mean, lo, hi = paired_diff_ci(psaim, other)
        ok = np.isfinite(hi) and hi < 0.0
        weak = np.isfinite(mean) and mean < 0.0
        lines.append(
            f"| simplified PSAIM vs {display_method(method)} | {fmt(mean)} [{fmt(lo)}, {fmt(hi)}] | {verdict(ok, weak)} |"
        )
    lines.append("")
    lines.append(
        "Negative paired differences mean simplified PSAIM is better. The stronger statistical-superiority claim requires the entire paired bootstrap CI to be below zero."
    )
    lines.append("")
    return lines


def analyze_eval_signals(data_dir: Path) -> List[str]:
    path = data_dir / "psaim_signals.csv"
    if not path.exists():
        return ["## Held-Out PSAIM Signal Evidence", "", f"Missing `{path}`.", ""]
    df = pd.read_csv(path)
    subset = df[df["method"].eq("psaim_lite")] if "method" in df.columns else df
    lines = ["## Held-Out PSAIM Signal Evidence", ""]
    lines.append("| metric | low entropy mean [95% CI] | high entropy mean [95% CI] | high-low paired/pooled direction |")
    lines.append("| - | - | - | - |")
    for metric in ["V_epi", "V_ale", "V_ale_excess", "gate_h3", "r_int"]:
        metric_df = subset[subset["metric"].eq(metric)]
        low = metric_df[metric_df["regime"].eq("low_entropy")]["value"]
        high = metric_df[metric_df["regime"].eq("high_entropy")]["value"]
        low_mean, high_mean = low.mean(), high.mean()
        low_lo, low_hi = bootstrap_ci(low)
        high_lo, high_hi = bootstrap_ci(high)
        direction = "high > low" if high_mean > low_mean else "high <= low"
        lines.append(
            f"| {metric} | {fmt(low_mean)} [{fmt(low_lo)}, {fmt(low_hi)}] | {fmt(high_mean)} [{fmt(high_lo)}, {fmt(high_hi)}] | {direction} |"
        )
    lines.append("")
    return lines


def analyze_training_probes(claim_dir: Optional[Path]) -> List[str]:
    if claim_dir is None:
        return ["## Training-Episode Epistemic Evidence", "", "No claim-validation directory supplied.", ""]
    path = claim_dir / "training_epistemic_probes.csv"
    if not path.exists():
        return ["## Training-Episode Epistemic Evidence", "", f"Missing `{path}`.", ""]
    df = pd.read_csv(path)
    metric_df = df[df["metric"].eq("V_epi")]
    checkpoints = sorted(metric_df["checkpoint_episode"].dropna().unique())
    if not checkpoints:
        return ["## Training-Episode Epistemic Evidence", "", "No checkpoints found.", ""]
    first, last = checkpoints[0], checkpoints[-1]
    lines = ["## Training-Episode Epistemic Evidence", ""]
    lines.append("| regime | initial V_epi | final V_epi | final-initial paired diff [95% CI] | trend status |")
    lines.append("| - | - | - | - | - |")
    for regime in ["low_entropy", "high_entropy"]:
        reg = metric_df[metric_df["regime"].eq(regime)]
        by_seed = reg.groupby(["seed", "checkpoint_episode"])["value"].mean().unstack()
        initial = by_seed[first]
        final = by_seed[last]
        mean, lo, hi = paired_diff_ci(final, initial)
        ok = np.isfinite(hi) and hi < 0.0
        weak = np.isfinite(mean) and mean < 0.0
        lines.append(
            f"| {regime} | {fmt(initial.mean())} | {fmt(final.mean())} | {fmt(mean)} [{fmt(lo)}, {fmt(hi)}] | {verdict(ok, weak)} |"
        )
    lines.append("")
    lines.append(
        f"Training evidence uses fixed probe states evaluated at checkpoints {int(first)} to {int(last)}. This is the correct evidence type for the draft phrase `V_epi decreases within each regime as experience accumulates`."
    )
    lines.append("")
    return lines


def analyze_behavior(claim_dir: Optional[Path]) -> List[str]:
    if claim_dir is None:
        return ["## Exploration-Behavior Evidence", "", "No claim-validation directory supplied.", ""]
    path = claim_dir / "exploration_behavior.csv"
    if not path.exists():
        return ["## Exploration-Behavior Evidence", "", f"Missing `{path}`.", ""]
    df = pd.read_csv(path)
    df = df[df["method"].eq("psaim_lite")] if "method" in df.columns else df
    lines = ["## Exploration-Behavior Evidence", ""]
    lines.append("| behavior metric | low mean | high mean | high-low paired diff [95% CI] | status |")
    lines.append("| - | - | - | - | - |")
    metrics = ["positive_intrinsic_rate", "allocate_rate", "defer_rate", "reject_rate", "mean_r_int", "mean_V_epi", "mean_V_ale"]
    for metric in metrics:
        pivot = df.pivot_table(index="seed", columns="regime", values=metric, aggfunc="mean")
        if not {"low_entropy", "high_entropy"}.issubset(pivot.columns):
            continue
        mean, lo, hi = paired_diff_ci(pivot["high_entropy"], pivot["low_entropy"])
        separated = np.isfinite(lo) and (lo > 0.0 or hi < 0.0)
        weak = np.isfinite(mean) and abs(mean) > 1e-9
        lines.append(
            f"| {metric} | {fmt(pivot['low_entropy'].mean())} | {fmt(pivot['high_entropy'].mean())} | {fmt(mean)} [{fmt(lo)}, {fmt(hi)}] | {verdict(separated, weak)} |"
        )
    lines.append("")
    lines.append(
        "This section treats action distribution as the direct behavioral evidence. Intrinsic reward sign alone is not counted as behavior."
    )
    lines.append("")
    return lines


def write_report(data_dir: Path, claim_dir: Optional[Path], output_path: Path) -> None:
    lines: List[str] = ["# Claim Evidence Audit", ""]
    lines.extend(analyze_exp1(data_dir))
    lines.extend(analyze_exp2_performance(data_dir))
    lines.extend(analyze_eval_signals(data_dir))
    lines.extend(analyze_training_probes(claim_dir))
    lines.extend(analyze_behavior(claim_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote claim audit to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether experiment outputs support the exact paper claims.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--claim-dir")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_report(
        data_dir=Path(args.data_dir),
        claim_dir=Path(args.claim_dir) if args.claim_dir else None,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
