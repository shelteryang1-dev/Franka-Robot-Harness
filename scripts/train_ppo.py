#!/usr/bin/env python3
"""Train the PPO lift baseline."""

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_TRAIN_SCRIPT = Path(
    "/workspace/Program/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py"
)


def _add_default(name: str, value: str | None = None) -> None:

    if name in sys.argv:
        return
    sys.argv.append(name)
    if value is not None:
        sys.argv.append(value)


_add_default("--task", "Franka-Manipulation-Lift-PPO-v0")
_add_default(
    "--external_callback",
    "franka_manipulation_sim.rl.registration.register_rl_tasks",
)
_add_default("--num_envs", "64")
_add_default("--max_iterations", "10")
_add_default("--seed", "42")
_add_default("--device", "cuda:0")
_add_default("--viz", "none")

environment = os.environ.copy()
source_root = str(PROJECT_ROOT / "source")
environment["PYTHONPATH"] = f"{source_root}:{environment.get('PYTHONPATH', '')}".rstrip(":")

environment["GIT_PYTHON_REFRESH"] = "quiet"
os.execve(
    "/isaac-sim/python.sh",
    ["/isaac-sim/python.sh", str(OFFICIAL_TRAIN_SCRIPT), *sys.argv[1:]],
    environment,
)
