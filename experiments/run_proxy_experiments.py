from __future__ import annotations

import argparse
from pathlib import Path

from src.experiment_runner import RuntimeConfig, experiment_1, experiment_2, write_run_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run proxy-study RL experiments.")
    parser.add_argument("--experiment", choices=["exp1", "exp2", "all"], default="all")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--queue-capacity", type=int, default=12)
    parser.add_argument("--visible-queue", type=int, default=5)
    parser.add_argument("--alternating-block", type=int, default=24)
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5001")
    parser.add_argument("--use-mlflow", action="store_true")
    parser.add_argument(
        "--eval-adapt",
        action="store_true",
        help="Allow learned agents to continue updating during evaluation. Default is frozen evaluation.",
    )
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "experiments" / "data" / "results" / "latest"))
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
        eval_adapt=args.eval_adapt,
    )
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(runtime, results_dir)

    if args.experiment in {"exp1", "all"}:
        experiment_1(runtime, results_dir)
    if args.experiment in {"exp2", "all"}:
        experiment_2(runtime, results_dir)

    print(f"Wrote experiment outputs to {results_dir}")


if __name__ == "__main__":
    main()
