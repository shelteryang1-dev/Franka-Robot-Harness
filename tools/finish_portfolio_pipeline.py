#!/usr/bin/env python3
"""Finish portfolio pipeline."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from run_portfolio_suite import save, exec_prefix, ros_command, wait_log, stop_control_owned
from run_portfolio_supervised import wait_log


def isolated_demo(directory, instruction=None, trajectory=False, key=""):
    directory.mkdir()
    directory.chmod(0o777)
    for part in ("simulation", "control", "client"):
        (directory/part).mkdir()
        (directory/part).chmod(0o777)
    relative = str(directory.relative_to(ROOT))
    if trajectory:
        simulation = exec_prefix("isaac-env")+["/isaac-sim/python.sh", "tools/record_trajectory_sim.py",
            "--output_dir", relative+"/simulation", "--duration", "300", "--viz", "kit", "--device", "cuda:0"]
        ready = "[TRAJECTORY_RECORDING_READY]"
    else:
        simulation = exec_prefix("isaac-env")+["/isaac-sim/python.sh", "scripts/run_harness_sim.py",
            "--multi_object", "--output_dir", relative+"/simulation", "--duration", "900", "--task_timeout", "600",
            "--viz", "kit", "--device", "cuda:0", "--record_video", relative+"/raw.mp4",
            "--video_frames", "10000", "--stop_file", relative+"/simulation/STOP"]
        ready = "[HARNESS_READY]"
    control_args = ["ros2", "run", "franka_sim_ros2_control", "control_stack.py",
        "--contract", relative+"/simulation/asset_joint_contract.json", "--config",
        "ros2_control_ws/src/franka_sim_ros2_control/config/"+("controllers.yaml" if trajectory else "harness_controllers.yaml"),
        "--output", relative+"/control", "--duration", "800"]
    if not trajectory:
        control_args.append("--streaming")
    sim = control = None
    report = {"success": False}
    sim_log = (directory/"simulation.log").open("w", encoding="utf-8")
    ctrl_log = (directory/"control.log").open("w", encoding="utf-8")
    try:
        sim = subprocess.Popen(simulation, stdout=sim_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        wait_log(sim, directory/"simulation.log", ready, 180)
        control = subprocess.Popen(ros_command(control_args), stdout=ctrl_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        wait_log(control, directory/"control.log", "[CONTROL_STACK_READY]", 100)
        if trajectory:
            client = ["ros2", "run", "franka_sim_ros2_control", "trajectory_client.py",
                "--contract", relative+"/simulation/asset_joint_contract.json", "--output", relative+"/client"]
        else:
            client = ["python3", "scripts/harness_client.py", "--planner", "deepseek", "--key_stdin",
                "--instruction", instruction, "--output", relative+"/client", "--timeout", "650"]
        save(directory/"commands.json", {"simulation": simulation, "control": ros_command(control_args), "client": ros_command(client)})
        with (directory/"client.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(ros_command(client), input="" if trajectory else key,
                text=True, stdout=log, stderr=subprocess.STDOUT, timeout=720)
        report["client_exit_code"] = result.returncode
        detail = json.loads((directory/"client"/("trajectory.json" if trajectory else "result.json")).read_text(encoding="utf-8"))
        report["success"] = bool(result.returncode == 0 and detail.get("success"))
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if control:
            stop_control_owned(relative+"/control")
            report["control_exit_code"] = control.wait(timeout=20)
        if sim:
            (directory/"simulation/STOP").touch()
            report["simulation_exit_code"] = sim.wait(timeout=40)
        sim_log.close()
        ctrl_log.close()
        save(directory/"demo_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key_stdin", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    key = sys.stdin.read().strip() if args.key_stdin else os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        parser.error("API key missing")
    formal, output = (ROOT/args.formal).resolve(), (ROOT/args.output).resolve()
    if not formal.is_relative_to(ROOT/"results") or not output.is_relative_to(ROOT/"results"):
        parser.error("Invalid pipeline path")
    output.mkdir(parents=True, exist_ok=False)
    if args.detach:
        pid = os.fork()
        if pid:
            print(json.dumps({"pipeline_pid": pid, "output": str(output)}), flush=True)
            return 0
        os.setsid()
        descriptor = os.open(output/"pipeline.log", os.O_WRONLY | os.O_CREAT, 0o600)
        os.dup2(descriptor, 1)
        os.dup2(descriptor, 2)
        os.close(descriptor)
        os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    save(output/"status.json", {"pid": os.getpid(), "stage": "waiting_for_formal_suite"})
    deadline = time.monotonic()+4*3600
    while time.monotonic() < deadline:
        state_path = formal/"runner_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["status"] == "infrastructure_failed":
                raise RuntimeError("Benchmark startup failed")
            if state["status"] == "finished":
                if state["finished"] != 30:
                    raise RuntimeError("Benchmark incomplete")
                break
        time.sleep(15)
    else:
        raise TimeoutError("Benchmark timeout (4h)")
    subprocess.run([sys.executable, "scripts/report_portfolio_suite.py", str(formal), "--compress_traces"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "tools/export_portfolio_evidence.py", str(formal),
        "--output", str(output/"formal_evidence.zip")], cwd=ROOT, check=True)
    reports = []
    for name, start in (("language_sequence", 22), ("six_color_sort", 2)):
        save(output/"status.json", {"pid": os.getpid(), "stage": name})
        relative = str((output/name).relative_to(ROOT))
        result = subprocess.run([sys.executable, "tools/run_portfolio_supervised.py", "--planner", "deepseek",
            "--key_stdin", "--start", str(start), "--count", "1", "--record", "--output", relative],
            input=key, text=True, cwd=ROOT)
        protocol = json.loads((output/name/"protocol.json").read_text(encoding="utf-8"))
        case = output/name/protocol["cases"][0]["id"]
        subprocess.run([sys.executable, "scripts/build_portfolio_video.py", "--case", str(case),
            "--output", f"media/portfolio/{name}.mp4"], cwd=ROOT, check=True)
        reports.append({"name": name, "exit_code": result.returncode, "result": str(case/"client/result.json")})
    save(output/"status.json", {"pid": os.getpid(), "stage": "ppo_mixed_demo"})
    mixed = output/"ppo_mixed_demo"
    reports.append({"name": "ppo_mixed_demo", **isolated_demo(mixed,
        instruction="用PPO抬起红色方块，然后把它放到左侧，最后回到初始姿态", key=key)})
    subprocess.run([sys.executable, "scripts/build_portfolio_video.py", "--case", str(mixed),
        "--output", "media/portfolio/ppo_mixed_demo.mp4"], cwd=ROOT, check=True)
    save(output/"status.json", {"pid": os.getpid(), "stage": "standard_trajectory_demo"})
    trajectory = output/"standard_trajectory_demo"
    reports.append({"name": "standard_trajectory_demo", **isolated_demo(trajectory, trajectory=True)})
    save(output/"demos.json", reports)
    save(output/"status.json", {"pid": os.getpid(), "stage": "independent_regressions"})
    regressions = []
    for name, arguments in (("state_machine", ["scripts/run_benchmark.py", "--gate", "2",
            "--num_envs", "4", "--num_episodes", "4", "--seed", "2026", "--device", "cuda:0"]),
            ("ppo", ["scripts/eval_ppo.py", "--checkpoint", "checkpoints/ppo_lift/model_1499.pt",
             "--num_envs", "4", "--num_episodes", "4", "--seed", "43", "--device", "cuda:0", "--viz", "none"])):
        directory = output/("regression_"+name)
        directory.mkdir()
        directory.chmod(0o777)
        command = exec_prefix("isaac-env")+["/isaac-sim/python.sh", *arguments,
            "--output_dir", str(directory.relative_to(ROOT))]
        with (directory/"execution.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        regressions.append({"name": name, "exit_code": result.returncode, "command": command})
    save(output/"regressions.json", regressions)
    save(output/"status.json", {"pid": os.getpid(), "stage": "recordings_finished_need_manual_quality_review",
        "note": "Trajectory recording and tracking CSV saved"})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if "--output" in sys.argv:
            destination = (ROOT/sys.argv[sys.argv.index("--output")+1]).resolve()
            if destination.is_relative_to(ROOT/"results") and destination.exists():
                save(destination/"status.json", {"pid": os.getpid(), "stage": "failed",
                    "error": f"{type(error).__name__}: {error}"})
        raise
