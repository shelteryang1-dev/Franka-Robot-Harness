#!/usr/bin/env python3
"""Export ppo training metrics."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


parser = argparse.ArgumentParser(description="Export PPO training metrics")
parser.add_argument("--log_dir", required=True, help="Training log directory")
parser.add_argument("--output_dir", required=True, help="Output directory")
parser.add_argument("--num_envs", type=int, default=None, help="Environment count")
parser.add_argument("--seed", type=int, default=None, help="Training seed")
parser.add_argument("--rollout_steps", type=int, default=24, help="Rollout steps per environment")
args = parser.parse_args()


def _choose_tag(tags: list[str], keyword: str) -> str | None:

    candidates = [tag for tag in tags if keyword.lower() in tag.lower()]
    return min(candidates, key=len) if candidates else None


def main() -> int:

    log_dir = Path(args.log_dir).expanduser().resolve()
    event_files = sorted(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"TensorBoard events missing: {log_dir}")

    accumulator = EventAccumulator(str(event_files[-1]), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = accumulator.Tags().get("scalars", [])
    print("[TENSORBOARD] Scalar tags:")
    for tag in scalar_tags:
        print(f"  - {tag}")

    selected = {
        "mean_reward": _choose_tag(scalar_tags, "mean_reward"),
        "success_rate": _choose_tag(scalar_tags, "success_rate"),
        "steps_per_second": _choose_tag(scalar_tags, "fps"),
        "mean_episode_length": _choose_tag(scalar_tags, "mean_episode_length"),
    }
    selected = {name: tag for name, tag in selected.items() if tag is not None}
    if "mean_reward" not in selected:
        raise RuntimeError("Mean reward missing")


    values_by_step: dict[int, dict[str, float]] = {}
    for column, tag in selected.items():
        for event in accumulator.Scalars(tag):
            values_by_step.setdefault(event.step, {})[column] = float(event.value)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "training_metrics.csv"
    json_path = output_dir / "training_summary.json"
    figure_path = output_dir / "training_curve.png"
    columns = ["iteration", *selected.keys()]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for step in sorted(values_by_step):
            writer.writerow({"iteration": step, **values_by_step[step]})

    summary: dict[str, object] = {
        "log_dir": str(log_dir),
        "event_file": str(event_files[-1]),
        "num_iterations_recorded": len(values_by_step),
        "selected_tensorboard_tags": selected,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "rollout_steps_per_env": args.rollout_steps,
    }
    for column in selected:
        series = [(step, row[column]) for step, row in sorted(values_by_step.items()) if column in row]
        if series:
            best_step, best_value = max(series, key=lambda item: item[1])
            summary[column] = {
                "initial": series[0][1],
                "final": series[-1][1],
                "best": best_value,
                "best_iteration": best_step,
            }
    reward_events = accumulator.Scalars(selected["mean_reward"])
    if reward_events:
        summary["training_wall_time_s"] = reward_events[-1].wall_time - reward_events[0].wall_time
        if args.num_envs is not None:
            summary["total_environment_steps"] = len(reward_events) * args.num_envs * args.rollout_steps
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    reward_series = [
        (step, row["mean_reward"]) for step, row in sorted(values_by_step.items()) if "mean_reward" in row
    ]
    figure, axes = plt.subplots(2 if "success_rate" in selected else 1, 1, figsize=(9, 7), sharex=True)
    if not isinstance(axes, (list, tuple)) and not hasattr(axes, "__len__"):
        axes = [axes]
    axes[0].plot([item[0] for item in reward_series], [item[1] for item in reward_series], color="#2563eb")

    axes[0].set_ylabel("Mean reward")
    axes[0].set_title("PPO training curves")
    axes[0].grid(alpha=0.25)
    if "success_rate" in selected:
        success_series = [
            (step, row["success_rate"])
            for step, row in sorted(values_by_step.items())
            if "success_rate" in row
        ]
        axes[1].plot(
            [item[0] for item in success_series],
            [item[1] for item in success_series],
            color="#16a34a",
        )
        axes[1].set_ylabel("Training goal success rate")
        axes[1].grid(alpha=0.25)
    axes[-1].set_xlabel("Training iteration")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    print(f"[RESULT] metrics_csv={csv_path}")
    print(f"[RESULT] summary_json={json_path}")
    print(f"[RESULT] curve_png={figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
