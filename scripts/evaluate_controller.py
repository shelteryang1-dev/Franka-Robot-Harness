#!/usr/bin/env python3
"""Select and evaluate a baseline controller."""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:

    parser = argparse.ArgumentParser(
        description="Evaluate Franka controller",
        epilog=(
            "Arguments forwarded to controller entry point. "
            "Append controller arguments after --controller."
        ),
    )
    parser.add_argument(
        "--controller",
        choices=("state_machine", "ppo"),
        required=True,
        help="Controller backend",
    )
    args, controller_args = parser.parse_known_args()

    target_script = "run_benchmark.py" if args.controller == "state_machine" else "eval_ppo.py"
    simulator_python = Path("/isaac-sim/python.sh")
    python_command = str(simulator_python) if simulator_python.exists() else sys.executable
    command = [python_command, str(PROJECT_ROOT / "scripts" / target_script), *controller_args]
    print(f"[EVAL] controller={args.controller}, entry={target_script}", flush=True)
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
