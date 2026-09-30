#!/usr/bin/env python3
"""Resume benchmark workers safely."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_portfolio_suite as suite


def wait_log(process, path, marker, seconds):
    budget = max(seconds, 360 if "READY" in marker and "CONTROL" not in marker else 150)
    deadline = time.monotonic() + budget
    while True:
        content = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        if process.poll() is not None:
            raise RuntimeError(f"Service exited: {path.name}, exit={process.returncode}")
        if marker in content:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Startup timeout ({budget}s): {path.name}")
        time.sleep(.5)


def archive_infrastructure_failure(output):
    state_file = output / "runner_state.json"
    if not state_file.exists():
        return
    state = json.loads(state_file.read_text(encoding="utf-8"))
    if state.get("status") != "infrastructure_failed":
        return
    try:
        os.kill(state["pid"], 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError("Runner already active")
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    if protocol["source_sha256"] != suite.source_hashes():
        raise RuntimeError("Source hash mismatch")
    case = (output / state["case"]).resolve()
    if case.parent != output or state["case"] not in {c["id"] for c in protocol["cases"]}:
        raise RuntimeError("Resume path mismatch")
    report = json.loads((case / "case_report.json").read_text(encoding="utf-8"))
    if "client_exit_code" in report or report.get("cleanup_error") or report.get("simulation_exit_code") is None:
        raise RuntimeError("Previous execution unresolved")
    for container in ("isaac-env", "franka-ros2-control"):
        result = subprocess.run(["docker", "top", container, "-eo", "pid,args"], capture_output=True, text=True, check=True)
        if any(name in result.stdout for name in ("run_harness_sim.py", "control_stack.py", "ros2_control_node")):
            raise RuntimeError("Controller still active")
    archive = output / "infrastructure_attempts" / (state["case"] + "_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    archive.mkdir(parents=True, exist_ok=False)
    for name in ("runner_state.json", "summary.json", "summary.csv"):
        if (output / name).exists():
            shutil.copy2(output / name, archive / name)
    case.rename(archive / "case")
    suite.save(archive / "recovery.json", {"previous_state": state,
        "reason": "Startup retry; wait=360s; source unchanged",
        "source_sha256": protocol["source_sha256"], "recovered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


if __name__ == "__main__":
    if "--resume" in sys.argv:
        output = (ROOT / sys.argv[sys.argv.index("--output") + 1]).resolve()
        if not output.is_relative_to((ROOT / "results").resolve()):
            raise RuntimeError("Invalid result path")
        archive_infrastructure_failure(output)
    suite.wait_log = wait_log
    raise SystemExit(suite.main())
