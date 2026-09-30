#!/usr/bin/env python3
"""Plot trajectory tracking."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
args = parser.parse_args()
with (args.directory / "tracking.csv").open() as stream:
    rows = list(csv.DictReader(stream))
if not rows:
    raise ValueError("Feedback data empty")
fig, axes = plt.subplots(4, 2, figsize=(11, 10), constrained_layout=True)
summary = {}
for index, name in enumerate(sorted({row["joint_name"] for row in rows})):
    selected = [row for row in rows if row["joint_name"] == name]
    t0 = float(selected[0]["timestamp"])
    times = [float(row["timestamp"])-t0 for row in selected]
    desired = [float(row["commanded_position"]) for row in selected]
    actual = [float(row["actual_position"]) for row in selected]
    errors = [abs(a-b) for a, b in zip(desired, actual)]
    summary[name] = {"samples": len(errors), "mean_absolute_error_rad": sum(errors)/len(errors),
                     "last_feedback_error_rad": errors[-1], "max_error_rad": max(errors)}
    axis = axes.flat[index]
    axis.plot(times, desired, label="Command")
    axis.plot(times, actual, "--", label="Actual")
    axis.set(title=name, xlabel="Simulation time (s)", ylabel="Position (rad)")
    axis.grid(alpha=0.3)
    axis.legend()
axes.flat[-1].axis("off")
fig.savefig(args.directory / "tracking.png", dpi=150)
(args.directory / "tracking_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
