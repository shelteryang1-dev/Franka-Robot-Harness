#!/usr/bin/env python3
"""Inspect simulation state topics."""

import argparse
import json
import os
import platform
import sys
import time


def _restart_with_ros2_runtime_if_needed() -> None:

    if os.environ.get("FRANKA_ROS2_INSPECT_ENV_READY") == "1":
        return
    ubuntu_major = platform.freedesktop_os_release().get("VERSION_ID", "24").split(".")[0]
    ros2_distribution = "humble" if ubuntu_major == "22" else "jazzy"
    ros2_root = f"/isaac-sim/exts/isaacsim.ros2.core/{ros2_distribution}"
    environment = os.environ.copy()
    environment["ROS_DISTRO"] = ros2_distribution
    environment["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    environment["PYTHONPATH"] = f"{ros2_root}/rclpy:{environment.get('PYTHONPATH', '')}".rstrip(":")
    environment["LD_LIBRARY_PATH"] = (
        f"{ros2_root}/lib:{environment.get('LD_LIBRARY_PATH', '')}"
    ).rstrip(":")
    environment["FRANKA_ROS2_INSPECT_ENV_READY"] = "1"
    os.execve("/isaac-sim/python.sh", ["/isaac-sim/python.sh", *sys.argv], environment)


_restart_with_ros2_runtime_if_needed()

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage


parser = argparse.ArgumentParser(description="Inspect ROS 2 state topics")
parser.add_argument("--timeout", type=float, default=120.0, help="Timeout (s)")
parser.add_argument("--min_samples", type=int, default=10, help="Minimum pose samples")
args = parser.parse_args()


def _maximum_change(samples: list[list[float]]) -> float:

    if len(samples) < 2:
        return 0.0
    first = samples[0]
    return max(abs(value - first[index]) for sample in samples[1:] for index, value in enumerate(sample))


def main() -> int:

    rclpy.init(args=None)
    node = rclpy.create_node("franka_sim_state_inspector")
    quality = QoSProfile(depth=10)
    joint_samples: list[list[float]] = []
    ee_samples: list[list[float]] = []
    object_samples: list[list[float]] = []
    tf_messages = 0

    def receive_joint(message: JointState) -> None:
        if message.position:
            joint_samples.append(list(message.position))

    def receive_ee(message: PoseStamped) -> None:
        point = message.pose.position
        ee_samples.append([point.x, point.y, point.z])

    def receive_object(message: PoseStamped) -> None:
        point = message.pose.position
        object_samples.append([point.x, point.y, point.z])

    def receive_tf(message: TFMessage) -> None:
        nonlocal tf_messages
        if message.transforms:
            tf_messages += 1

    subscriptions = [
        node.create_subscription(JointState, "/joint_states", receive_joint, quality),
        node.create_subscription(PoseStamped, "/ee_pose", receive_ee, quality),
        node.create_subscription(PoseStamped, "/object_pose", receive_object, quality),
        node.create_subscription(TFMessage, "/tf", receive_tf, quality),
    ]

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        enough_samples = (
            len(joint_samples) >= args.min_samples
            and len(ee_samples) >= args.min_samples
            and len(object_samples) >= args.min_samples
            and tf_messages > 0
        )
        if enough_samples and _maximum_change(joint_samples) > 1.0e-4 and _maximum_change(ee_samples) > 1.0e-4:
            break

    discovered = dict(node.get_topic_names_and_types())
    report = {
        "topics_discovered": {
            topic: discovered.get(topic, [])
            for topic in ("/joint_states", "/tf", "/ee_pose", "/object_pose")
        },
        "joint_samples": len(joint_samples),
        "ee_pose_samples": len(ee_samples),
        "object_pose_samples": len(object_samples),
        "tf_messages": tf_messages,
        "max_joint_position_change": _maximum_change(joint_samples),
        "max_ee_position_change": _maximum_change(ee_samples),
        "max_object_position_change": _maximum_change(object_samples),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    required_topics = all(report["topics_discovered"].values())
    enough_data = (
        len(joint_samples) >= args.min_samples
        and len(ee_samples) >= args.min_samples
        and len(object_samples) >= args.min_samples
        and tf_messages > 0
    )
    state_changed = report["max_joint_position_change"] > 1.0e-4 and report["max_ee_position_change"] > 1.0e-4

    for subscription in subscriptions:
        node.destroy_subscription(subscription)
    node.destroy_node()
    rclpy.shutdown()
    return 0 if required_topics and enough_data and state_changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
