#!/usr/bin/env python3
"""Run a finite state-machine benchmark."""

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _validate_fresh_results(output_dir: Path, previous_summary: bytes | None, expected_episodes: int) -> None:

    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError("Summary missing")
    current_summary = summary_path.read_bytes()
    if previous_summary is not None and current_summary == previous_summary:
        raise RuntimeError("Summary unchanged")
    summary = json.loads(current_summary)
    actual_episodes = int(summary.get("total_episodes", -1))
    if actual_episodes != expected_episodes:
        raise RuntimeError(f"Episode count mismatch: expected={expected_episodes}, actual={actual_episodes}.")


def _build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description="Franka manipulation benchmark")
    parser.add_argument("--gate", type=int, choices=(1, 2), default=2, help="1: lift; 2: pick-and-place")
    parser.add_argument("--num_envs", type=int, default=16, help="Environment count")
    parser.add_argument("--num_episodes", type=int, default=100, help="Total episodes")
    parser.add_argument("--seed", type=int, default=2026, help="Seed")
    parser.add_argument("--output_dir", default="results/gate2_benchmark", help="Result directory")
    parser.add_argument("--object_xy_range", type=float, default=0.03, help="Object XY range (m)")
    parser.add_argument("--target_xy_range", type=float, default=0.05, help="Target XY range (m)")
    parser.add_argument("--mass_scale_range", type=float, default=0.30, help="Mass scale range")
    parser.add_argument("--target_z", type=float, default=None, help="Target height (m)")
    parser.add_argument("--device", default="cuda:0", help="Device")
    return parser


def _generate_figures(output_dir: Path) -> None:

    import matplotlib.pyplot as plt

    with (output_dir / "episodes.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    rate_names = [
        "grasp_success_rate", "lift_success_rate", "transport_success_rate", "place_success_rate", "overall_success_rate"
    ]
    rate_labels = ["Grasp", "Lift", "Transport", "Place", "Overall"]
    rate_values = [float(summary[name]) for name in rate_names]
    figure, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(
        rate_labels,
        rate_values,
        color=["#4c78a8", "#59a14f", "#76b7b2", "#f28e2b", "#e15759"],
    )
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Success rate")
    axis.set_title(f"Pick-and-place success rates ({len(rows)} episodes)")
    axis.bar_label(bars, fmt="%.3f")
    figure.tight_layout()
    figure.savefig(figure_dir / "success_rates.png", dpi=160)
    plt.close(figure)

    failures = Counter(row["failure_stage"] for row in rows if row["failure_stage"] != "none")
    labels = list(failures) or ["none"]
    values = [failures[key] for key in labels] if failures else [0]
    figure, axis = plt.subplots(figsize=(8, 4))
    bars = axis.bar(labels, values, color="#e15759")
    axis.set_ylabel("Episode count")
    axis.set_title("Pick-and-place failure breakdown")
    axis.bar_label(bars)
    figure.tight_layout()
    figure.savefig(figure_dir / "failure_breakdown.png", dpi=160)
    plt.close(figure)


def main() -> int:

    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    summary_path = output_dir / "summary.json"
    previous_summary = summary_path.read_bytes() if summary_path.is_file() else None
    simulator_python = Path("/isaac-sim/python.sh")
    python_command = str(simulator_python) if simulator_python.exists() else sys.executable
    command = [
        python_command,
        str(PROJECT_ROOT / "scripts" / "run_state_machine.py"),
        "--gate", str(args.gate),
        "--num_envs", str(args.num_envs),
        "--num_episodes", str(args.num_episodes),
        "--seed", str(args.seed),
        "--output_dir", str(output_dir),
        "--object_xy_range", str(args.object_xy_range),
        "--target_xy_range", str(args.target_xy_range),
        "--mass_scale_range", str(args.mass_scale_range),
        "--device", args.device,
        "--viz", "none",
    ]
    if args.target_z is not None:
        command.extend(["--target_z", str(args.target_z)])
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        return completed.returncode
    try:
        _validate_fresh_results(output_dir, previous_summary, args.num_episodes)
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as error:
        print(f"[ERROR] Result validation failed: {error}", file=sys.stderr, flush=True)
        return 1
    _generate_figures(output_dir)
    print(f"[FIGURES] directory={output_dir / 'figures'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
