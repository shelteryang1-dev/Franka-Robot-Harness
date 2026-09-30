#!/usr/bin/env python3
"""Run the registered portfolio benchmark."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from franka_manipulation_sim.harness.portfolio_suite import cases


def save(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def source_hashes():
    files = [*ROOT.glob("source/**/*.py"), *ROOT.glob("scripts/*.py"),
             *ROOT.glob("ros2_control_ws/src/**/*.py"), *ROOT.glob("ros2_control_ws/src/**/*.yaml"),
             ROOT/"checkpoints/ppo_lift/model_1499.pt"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files) if p.is_file()}


def exec_prefix(container):
    return ["docker", "exec", "-i", "-e", "PYTHONUNBUFFERED=1",
            "-e", "HTTP_PROXY=", "-e", "HTTPS_PROXY=", "-e", "ALL_PROXY=",
            "-e", "http_proxy=", "-e", "https_proxy=", "-e", "all_proxy=",
            "-e", "CUDA_VISIBLE_DEVICES=0", "-e",
            "FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/project/ros2_control_ws/src/franka_sim_ros2_control/config/dds_udp.xml",
            "-w", "/workspace/project", container] if container != "isaac-env" else [
            "docker", "exec", "-i", "-e", "PYTHONUNBUFFERED=1", "-e", "CUDA_VISIBLE_DEVICES=0",
            "-e", "HTTP_PROXY=", "-e", "HTTPS_PROXY=", "-e", "ALL_PROXY=",
            "-e", "http_proxy=", "-e", "https_proxy=", "-e", "all_proxy=",
            "-w", "/workspace/Program/franka_manipulation_sim", container]


def ros_command(arguments):
    return exec_prefix("franka-ros2-control")+["bash", "-c",
        "source /opt/ros/jazzy/setup.bash && source ros2_control_ws/install/setup.bash && "+shlex.join(arguments)]


def wait_log(process, path, marker, seconds):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        if marker in path.read_text(encoding="utf-8", errors="replace"):
            return
        if process.poll() is not None:
            raise RuntimeError(f"Service exited before ready: {path.name}, exit={process.returncode}")
        time.sleep(.5)
    raise TimeoutError(f"Startup timeout: {path.name}")


def stop_control_owned(unique_output):
    result = subprocess.run(["docker", "exec", "franka-ros2-control", "ps", "-eo", "pid,args"],
                            capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        pieces = line.strip().split(None, 1)
        if len(pieces) != 2 or not pieces[0].isdigit():
            continue
        argv = shlex.split(pieces[1])
        if (argv and Path(argv[0]).name.startswith("python")
                and any(Path(a).name == "control_stack.py" for a in argv)
                and unique_output in argv):
            subprocess.run(["docker", "exec", "franka-ros2-control", "kill", "-INT", pieces[0]], check=True)


def summarize(output, protocol):
    rows = []
    for case in protocol["cases"]:
        path = output/case["id"]/"case_report.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    report = {"protocol": "10 classes x 3 seeds; independent scenes; failures included",
        "planner": protocol["planner"], "planned": len(protocol["cases"]), "finished": len(rows),
        "client_attempted": sum("client_exit_code" in r for r in rows),
        "infrastructure_failures": sum("client_exit_code" not in r for r in rows),
        "planning_correct": sum(r.get("planning_correct", False) for r in rows),
        "task_successes": sum(r.get("task_success", False) for r in rows),
        "task_success_rate": sum(r.get("task_success", False) for r in rows)/len(rows) if rows else None,
        "cases": rows}
    save(output/"summary.json", report)
    fields = ["id", "class", "scene_seed", "initial_layout", "planning_correct", "task_success",
              "completed_steps", "physical_placements", "retries", "execution_wall_seconds", "client_exit_code", "error"]
    with (output/"summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--planner", choices=("deepseek", "rules", "fixed"), default="deepseek")
    parser.add_argument("--key_stdin", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--detach", action="store_true", help="Run in background")
    parser.add_argument("--record", action="store_true", help="Record a separate run")
    parser.add_argument("--video_spp", type=int, default=1)
    args = parser.parse_args()
    key = sys.stdin.read().strip() if args.key_stdin else os.environ.get("DEEPSEEK_API_KEY", "")
    if args.planner == "deepseek" and not key and not args.dry_run:
        parser.error("API key missing")
    selected = cases()[args.start:args.start+args.count]
    if not selected:
        parser.error("No cases selected")
    output = (ROOT/args.output).resolve()
    if not output.is_relative_to((ROOT/"results").resolve()):
        parser.error("Invalid result path")
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"version": 1, "planner": args.planner, "record": args.record, "cases": selected,
                "source_sha256": source_hashes(), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if (output/"protocol.json").exists():
        old = json.loads((output/"protocol.json").read_text(encoding="utf-8"))
        if not args.resume or any(old[k] != protocol[k] for k in ("planner", "record", "cases", "source_sha256")):
            parser.error("Protocol already exists")
        protocol = old
    else:
        save(output/"protocol.json", protocol)
    if args.dry_run:
        print(json.dumps({"protocol": str(output/"protocol.json"), "cases": len(selected)}))
        return 0
    if args.detach:
        pid = os.fork()
        if pid:
            print(json.dumps({"runner_pid": pid, "output": str(output)}), flush=True)
            return 0
        os.setsid()
        log = os.open(output/"runner.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.dup2(log, 1)
        os.dup2(log, 2)
        os.close(log)
        os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    save(output/"runner_state.json", {"pid": os.getpid(), "status": "running"})
    if (output/"STOP").exists():
        raise RuntimeError("Stop marker present")
    for case in selected:
        directory = output/case["id"]
        if (directory/"case_report.json").exists():
            continue
        if directory.exists():
            raise RuntimeError("Incomplete case already exists")
        if (output/"STOP").exists():
            break
        directory.mkdir()


        directory.chmod(0o777)
        for name in ("simulation", "control", "client"):
            folder = directory/name
            folder.mkdir()
            folder.chmod(0o777)
        relative = str(directory.relative_to(ROOT))
        save(directory/"case.json", case)
        save(directory/"expected_plan.json", case["expected_plan"])
        save(directory/"runner_state.json", {"status": "starting", "wall_time": time.time()})
        report = {k: case[k] for k in ("id", "class", "scene_seed", "initial_layout")}
        report.update(task_success=False, planning_correct=False)
        sim = control = None
        sim_log = (directory/"simulation.log").open("w", encoding="utf-8")
        control_log = (directory/"control.log").open("w", encoding="utf-8")
        started = time.monotonic()
        try:
            sim_args = exec_prefix("isaac-env")+["/isaac-sim/python.sh", "scripts/run_harness_sim.py",
                "--six_objects", "--scene_seed", str(case["scene_seed"]), "--position_jitter", ".005",
                "--initial_layout", case["initial_layout"], "--duration", "1500", "--task_timeout", "1000",
                "--device", "cuda:0", "--viz", "kit" if args.record else "none",
                "--output_dir", relative+"/simulation", "--stop_file", relative+"/simulation/STOP"]
            if args.record:
                sim_args += ["--record_video", relative+"/raw.mp4", "--video_frames", "18000",
                             "--video_spp", str(args.video_spp)]
            save(directory/"commands.json", {"simulation": sim_args,
                "note": "Credentials via stdin"})
            sim = subprocess.Popen(sim_args, stdout=sim_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            wait_log(sim, directory/"simulation.log", "[HARNESS_READY]", 180)
            control_args = ros_command(["ros2", "run", "franka_sim_ros2_control", "control_stack.py",
                "--contract", relative+"/simulation/asset_joint_contract.json",
                "--config", "ros2_control_ws/src/franka_sim_ros2_control/config/harness_controllers.yaml",
                "--output", relative+"/control", "--streaming", "--duration", "1400"])
            control = subprocess.Popen(control_args, stdout=control_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            wait_log(control, directory/"control.log", "[CONTROL_STACK_READY]", 100)
            client = ["python3", "scripts/harness_client.py", "--output", relative+"/client", "--timeout", "1050",
                      "--expected_plan", relative+"/expected_plan.json"]
            if args.planner == "fixed":
                client += ["--plan", relative+"/expected_plan.json"]
            else:
                client += ["--planner", args.planner, "--instruction", case["instruction"]]
                if args.planner == "deepseek":
                    client += ["--key_stdin"]
            save(directory/"commands.json", {"simulation": sim_args, "control": control_args,
                "client": ros_command(client), "note": "Credentials via stdin"})
            with (directory/"client.log").open("w", encoding="utf-8") as client_log:
                process = subprocess.run(ros_command(client), input=key if args.planner == "deepseek" else "",
                    text=True, stdout=client_log, stderr=subprocess.STDOUT, timeout=1120)
            report["client_exit_code"] = process.returncode
            result = json.loads((directory/"client/result.json").read_text(encoding="utf-8"))
            report["planning_correct"] = result.get("plan") == case["expected_plan"]
            report["task_success"] = bool(result.get("success") and report["planning_correct"])
            events = result.get("events", [])
            completed = [e for e in events if e["status"] == "step_succeeded"]
            report["completed_steps"] = len(completed)
            report["physical_placements"] = sum(isinstance(e.get("detail"), dict) and e["detail"].get("tray") is not None
                and not e["detail"].get("skipped") for e in completed)
            report["retries"] = sum(e["status"] == "retrying" for e in events)
            if "terminal_wall_time" in result:
                report["execution_wall_seconds"] = result["terminal_wall_time"]-result["submitted_wall_time"]
            terminal = [e for e in events if e["status"] in ("failed", "rejected", "cancelled", "succeeded")]
            report["terminal_event"] = terminal[-1] if terminal else None
            report["error"] = result.get("error") or (str(terminal[-1].get("detail")) if terminal and not report["task_success"] else None)
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
        finally:

            if control is not None:
                stop_control_owned(relative+"/control")
                try:
                    report["control_exit_code"] = control.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    report["cleanup_error"] = "Control service shutdown timeout"
            if sim is not None:
                stop = directory/"simulation/STOP"
                stop.parent.mkdir(parents=True, exist_ok=True)
                stop.touch()
                try:
                    report["simulation_exit_code"] = sim.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    report["cleanup_error"] = "Simulation shutdown timeout"
            sim_log.close()
            control_log.close()
            report["total_wall_seconds"] = time.monotonic()-started
            save(directory/"case_report.json", report)
            summary = summarize(output, protocol)
            print(json.dumps({"finished": summary["finished"], "successes": summary["task_successes"],
                              "case": case["id"], "error": report.get("error")}, ensure_ascii=False), flush=True)
        if report.get("cleanup_error"):
            raise RuntimeError(report["cleanup_error"])
        if "client_exit_code" not in report:

            save(output/"runner_state.json", {"pid": os.getpid(), "status": "infrastructure_failed",
                "case": case["id"], "error": report.get("error")})
            return 2
    summary = summarize(output, protocol)
    save(output/"runner_state.json", {"pid": os.getpid(), "status": "finished", "finished": summary["finished"]})
    return 0 if summary["finished"] == len(selected) and summary["task_successes"] == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
