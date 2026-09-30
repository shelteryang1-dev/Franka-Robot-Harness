#!/usr/bin/env python3
"""Check end-effector and gripper motion."""

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from franka_manipulation_sim.ros2.runtime import prepare_runtime

prepare_runtime()
import rclpy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from tf2_msgs.msg import TFMessage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=120, help="Startup timeout (s)")
    parser.add_argument("--output", default="results/ros2_control/client.json")
    args = parser.parse_args()
    rclpy.init(args=None)
    node = rclpy.create_node("franka_external_control_test")
    latest, counts = {}, {key: 0 for key in ("ee", "joint", "object", "tf")}
    report = {"success": False, "position_tolerance_m": 0.005, "tests": {}}

    def receive(key, message):
        latest[key] = message
        counts[key] += 1

    for key, topic, kind in (("ee", "/ee_pose", PoseStamped), ("joint", "/joint_states", JointState),
                             ("object", "/object_pose", PoseStamped), ("tf", "/tf", TFMessage)):
        node.create_subscription(kind, topic, lambda msg, key=key: receive(key, msg), 10)
    pose_pub = node.create_publisher(PoseStamped, "/ee_target_pose", 1)
    grip_pub = node.create_publisher(Bool, "/gripper_command", 1)

    def wait_for(predicate, timeout, publish=None):
        deadline, next_send = time.monotonic() + timeout, 0
        while time.monotonic() < deadline:
            if publish and time.monotonic() >= next_send:
                publish()
                next_send = time.monotonic() + 0.1
            rclpy.spin_once(node, timeout_sec=0.02)
            if predicate():
                return
        raise TimeoutError("State feedback timeout")

    def position():
        p = latest["ee"].pose.position
        return [p.x, p.y, p.z]

    def fingers():
        msg = latest["joint"]
        values = [p for name, p in zip(msg.name, msg.position) if "finger_joint" in name]
        if len(values) != 2 or not all(math.isfinite(v) for v in values):
            raise ValueError("Finger feedback invalid")
        return values

    def check_gripper(closed):
        initial = fingers()
        previous = counts.copy()
        message = Bool(data=closed)
        stable, last_count = 0, counts["joint"]

        def reached():
            nonlocal stable, last_count
            if counts["joint"] == last_count:
                return False
            last_count = counts["joint"]
            good = all(v < 0.01 for v in fingers()) if closed else all(v > 0.03 for v in fingers())
            stable = stable + 1 if good else 0
            return stable >= 5 and all(counts[k] > previous[k] for k in counts)

        wait_for(reached, 20, lambda: grip_pub.publish(message))
        final = fingers()
        if not all(abs(a-b) > 0.02 for a, b in zip(initial, final)):
            raise AssertionError("Gripper displacement insufficient")
        report["tests"]["close" if closed else "open"] = {
            "initial_fingers_m": initial, "final_fingers_m": final,
            "feedback_counts": {k: counts[k]-previous[k] for k in counts},
        }

    try:
        wait_for(lambda: all(v >= 10 for v in counts.values()) and
                 pose_pub.get_subscription_count() > 0 and grip_pub.get_subscription_count() > 0, args.timeout)
        if latest["ee"].header.frame_id != "world" or latest["object"].header.frame_id != "world":
            raise AssertionError("Feedback frame must be world")
        report["topics"] = dict(node.get_topic_names_and_types())
        hold_start = position()
        hold_until = time.monotonic() + 1.0
        wait_for(lambda: time.monotonic() >= hold_until, 3)
        hold_drift = math.dist(hold_start, position())
        if not math.isfinite(hold_drift) or hold_drift > 0.005:
            raise AssertionError("Idle EE drift exceeds 5 mm")
        report["tests"]["default_hold"] = {"drift_m": hold_drift, "duration_seconds": 1.0}
        initial = position()
        target = copy.deepcopy(latest["ee"])
        target.pose.position.z += 0.05
        desired = [target.pose.position.x, target.pose.position.y, target.pose.position.z]
        previous = counts.copy()
        stable, last_count = 0, counts["ee"]

        def reached_pose():
            nonlocal stable, last_count
            if counts["ee"] == last_count:
                return False
            last_count = counts["ee"]
            error = math.dist(position(), desired)
            stable = stable+1 if math.isfinite(error) and error < 0.005 else 0
            return stable >= 10 and all(counts[k] > previous[k] for k in counts)

        def send_pose():

            target.header.stamp = node.get_clock().now().to_msg()
            pose_pub.publish(target)

        wait_for(reached_pose, 20, send_pose)
        final = position()
        if math.dist(initial, final) < 0.03:
            raise AssertionError("EE displacement below 3 cm")
        actual_q, desired_q = latest["ee"].pose.orientation, target.pose.orientation
        actual_q = [actual_q.x, actual_q.y, actual_q.z, actual_q.w]
        desired_q = [desired_q.x, desired_q.y, desired_q.z, desired_q.w]
        dot = sum(a*b for a, b in zip(actual_q, desired_q)) / (math.hypot(*actual_q)*math.hypot(*desired_q))
        angle_error = 2*math.acos(min(1.0, abs(dot)))
        if not math.isfinite(dot) or angle_error > 0.05:
            raise AssertionError("EE orientation error exceeds 0.05 rad")
        report["tests"]["ee"] = {
            "initial_world_m": initial, "target_world_m": desired, "final_world_m": final,
            "error_m": math.dist(final, desired),
            "orientation_error_rad": angle_error,
            "final_quaternion_xyzw": actual_q,
            "feedback_counts": {k: counts[k]-previous[k] for k in counts},
        }
        check_gripper(True)
        check_gripper(False)
        if not latest["tf"].transforms:
            raise AssertionError("TF missing")
        report["success"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        report["feedback_counts"] = counts
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        node.destroy_node()
        rclpy.shutdown()
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
