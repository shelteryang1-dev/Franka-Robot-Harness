#!/usr/bin/env python3
"""Run the ROS 2 state-publishing demo."""

import runpy
import sys
from pathlib import Path


def _add_default(name: str, value: str | None = None) -> None:

    if name in sys.argv:
        return
    sys.argv.append(name)
    if value is not None:
        sys.argv.append(value)


_add_default("--gate", "2")
_add_default("--num_envs", "1")
_add_default("--num_episodes", "3")
_add_default("--output_dir", "results/ros2_demo")
_add_default("--ros2_publish")
_add_default("--ros2_realtime")

runpy.run_path(str(Path(__file__).with_name("run_state_machine.py")), run_name="__main__")
