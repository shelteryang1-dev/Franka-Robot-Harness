#!/usr/bin/env python3
"""Verify a multi-waypoint joint trajectory."""

import argparse
import csv
import json
import math
from pathlib import Path
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--invalid", action="store_true", help="Test joint-limit rejection")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    contract = json.loads(Path(args.contract).read_text())
    names = [f"panda_joint{i}" for i in range(1, 8)]
    rclpy.init(args=[])
    node = rclpy.create_node("franka_trajectory_client", parameter_overrides=[Parameter("use_sim_time", value=True)])
    action = ActionClient(node, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory")
    latest, raw_samples, rows, commands = {}, [], [], []
    report = {"success": False, "invalid_test": args.invalid}

    def receive(message, key):
        values = dict(zip(message.name, message.position))
        if not all(name in values and math.isfinite(values[name]) for name in names):
            return
        latest[key] = (time.monotonic(), values)
        if key == "raw":
            raw_samples.append([values[name] for name in names])

    node.create_subscription(JointState, "/joint_states", lambda msg: receive(msg, "joint"), 10)
    node.create_subscription(JointState, "/isaac_joint_states_raw", lambda msg: receive(msg, "raw"), 10)
    node.create_subscription(JointState, "/isaac_joint_commands", lambda msg: commands.append(list(msg.position)), 10)

    def spin_until(predicate, timeout):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
            if predicate():
                return
        raise TimeoutError("Action or feedback timeout")

    def feedback(message):
        f = message.feedback
        stamp = f.header.stamp.sec + f.header.stamp.nanosec*1e-9
        for name, desired, actual in zip(f.joint_names, f.desired.positions, f.actual.positions):
            rows.append({"timestamp": stamp, "joint_name": name, "commanded_position": desired,
                         "actual_position": actual, "error": desired-actual})

    try:
        if not action.wait_for_server(timeout_sec=30):
            raise TimeoutError("Trajectory action unavailable")
        spin_until(lambda: "raw" in latest and "joint" in latest and len(raw_samples) >= 5, 10)
        publishers = node.get_publishers_info_by_topic("/joint_states")
        report["joint_state_publishers"] = [p.node_name for p in publishers]
        if report["joint_state_publishers"] != ["joint_state_broadcaster"]:
            raise AssertionError("Unexpected joint-state publisher")
        initial = [latest["joint"][1][name] for name in names]
        target1, target2 = initial.copy(), initial.copy()
        target1[0] += 0.10
        target1[2] += 0.08
        target2[0] -= 0.08
        target2[2] += 0.12
        points = [(initial, 0.5), (target1, 2.5), (target2, 4.5), (initial, 6.5)]
        if args.invalid:
            invalid = initial.copy()
            invalid[6] = contract["joints"][names[6]]["upper"] + 0.3
            points = [(invalid, 0.1)]
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = names
        for positions, seconds in points:
            point = JointTrajectoryPoint(positions=positions)
            point.time_from_start.sec = int(seconds)
            point.time_from_start.nanosec = round((seconds-int(seconds))*1e9)
            goal.trajectory.points.append(point)
        report["waypoints"] = [{"positions": p, "time_from_start": t} for p, t in points]
        sent = action.send_goal_async(goal, feedback_callback=feedback)
        spin_until(sent.done, 10)
        handle = sent.result()
        report["accepted"] = handle.accepted
        if not handle.accepted:
            if not args.invalid:
                raise AssertionError("Valid trajectory rejected")
            report["success"] = True
        else:
            result = handle.get_result_async()
            spin_until(result.done, 45)
            value = result.result()
            report.update(action_status=value.status, error_code=value.result.error_code, error_string=value.result.error_string)
            if args.invalid:
                if value.status == 4 and value.result.error_code == 0:
                    raise AssertionError("Out-of-range trajectory accepted")
                report["success"] = True
            else:
                if value.status != 4 or value.result.error_code != 0 or not rows:
                    raise AssertionError("Trajectory failed or feedback missing")
                if time.monotonic()-latest["raw"][0] > 1.0:
                    raise AssertionError("Joint feedback stale")
                final = [latest["joint"][1][name] for name in names]
                error = [abs(a-b) for a, b in zip(final, initial)]
                physical_change = max(abs(v-init) for sample in raw_samples for v, init in zip(sample, raw_samples[0]))
                if max(error) > 0.02 or physical_change < 0.05 or not commands:
                    raise AssertionError("Trajectory verification failed")
                report.update(success=True, final_error_rad=dict(zip(names, error)),
                              mean_absolute_tracking_error_rad=sum(abs(r["error"]) for r in rows)/len(rows),
                              physical_max_change_rad=physical_change)
        report["raw_samples"] = len(raw_samples)
        report["hardware_command_samples"] = len(commands)
        report["feedback_rows"] = len(rows)
        report["commands_within_limits"] = all(len(cmd) == 7 and all(
            contract["joints"][name]["lower"]-1e-6 <= v <= contract["joints"][name]["upper"]+1e-6
            for name, v in zip(names, cmd)) for cmd in commands)
    except Exception as error:
        report["error"] = str(error)
    finally:
        with (output / "tracking.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["timestamp", "joint_name", "commanded_position", "actual_position", "error"])
            writer.writeheader()
            writer.writerows(rows)
        (output / "trajectory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        node.destroy_node()
        rclpy.shutdown()
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
