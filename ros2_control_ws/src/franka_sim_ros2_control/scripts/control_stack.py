#!/usr/bin/env python3
"""Supervise the ros2_control stack."""

import argparse
import json
import math
from pathlib import Path
import signal
import subprocess
import threading
import time
import xml.etree.ElementTree as ET

import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

NAMES = [f"panda_joint{i}" for i in range(1, 8)]


def description(contract, positions):
    robot = ET.Element("robot", name="franka_sim_control")
    ET.SubElement(robot, "link", name="panda_link0")
    for i, name in enumerate(NAMES, 1):
        cfg = contract["joints"][name]
        ET.SubElement(robot, "link", name=f"panda_link{i}")
        joint = ET.SubElement(robot, "joint", name=name, type="revolute")
        ET.SubElement(joint, "parent", link=f"panda_link{i-1}")
        ET.SubElement(joint, "child", link=f"panda_link{i}")
        ET.SubElement(joint, "limit", **{key: str(cfg[key]) for key in ("lower", "upper", "effort", "velocity")})
    system = ET.SubElement(robot, "ros2_control", name="IsaacFranka", type="system")
    hardware = ET.SubElement(system, "hardware")
    ET.SubElement(hardware, "plugin").text = "joint_state_topic_hardware_interface/JointStateTopicSystem"
    for name, value in (("joint_commands_topic", "/isaac_joint_commands"),
                        ("joint_states_topic", "/isaac_joint_states_raw"),
                        ("trigger_joint_command_threshold", "-1"), ("sum_wrapped_joint_states", "false")):
        ET.SubElement(hardware, "param", name=name).text = value
    for name in NAMES:
        joint = ET.SubElement(system, "joint", name=name)
        command = ET.SubElement(joint, "command_interface", name="position")
        # Initialize commands from measured joints.
        ET.SubElement(command, "param", name="initial_value").text = str(positions[name])
        for key, limit in (("min", "lower"), ("max", "upper")):
            ET.SubElement(command, "param", name=key).text = str(contract["joints"][name][limit])
        state = ET.SubElement(joint, "state_interface", name="position")
        ET.SubElement(state, "param", name="initial_value").text = str(positions[name])
        velocity = ET.SubElement(joint, "state_interface", name="velocity")
        ET.SubElement(velocity, "param", name="initial_value").text = "0.0"
    ET.indent(robot)
    return ET.tostring(robot, encoding="unicode")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--startup_timeout", type=float, default=30)
    parser.add_argument("--state_timeout", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=240)
    parser.add_argument("--streaming", action="store_true", help="Activate streaming control")
    args, _ = parser.parse_known_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    contract = json.loads(Path(args.contract).read_text())
    rclpy.init(args=[])
    from rclpy.parameter import Parameter
    node = rclpy.create_node("franka_control_supervisor", parameter_overrides=(
        [Parameter("use_sim_time", value=True)] if args.streaming else []))
    lock = threading.Lock()
    received = {"at": None, "stamp": None, "positions": None, "fault": None}

    def raw_state(message):
        with lock:
            if len(message.name) != len(set(message.name)) or len(message.position) != len(message.name):
                received["fault"] = "Joint names or lengths invalid"
                return
            values = dict(zip(message.name, message.position))
            if any(name not in values or not math.isfinite(values[name]) for name in NAMES):
                received["fault"] = "Joint feedback invalid"
                return
            stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
            if stamp != received["stamp"]:
                received.update(at=time.monotonic(), stamp=stamp, positions=values)

    def fault(message):
        with lock:
            received["fault"] = message.data

    node.create_subscription(JointState, "/isaac_joint_states_raw", raw_state, 10)
    node.create_subscription(String, "/isaac_control_fault", fault, 10)
    pub = node.create_publisher(String, "/robot_description",
                                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    def spin_node():
        from rclpy.executors import ExternalShutdownException
        try:
            rclpy.spin(node)
        except ExternalShutdownException:
            pass

    spin = threading.Thread(target=spin_node, daemon=True)
    spin.start()
    manager = None
    report = {"success": False, "lifecycle": []}
    log = (output / "controller_manager.log").open("w")

    def healthy():
        with lock:
            if received["fault"]:
                raise RuntimeError(received["fault"])
            if received["at"] is not None and time.monotonic()-received["at"] > args.state_timeout:
                raise TimeoutError("Joint feedback timeout")

    def cli(*arguments):
        process = subprocess.Popen(["ros2", "control", *arguments], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.monotonic()+25
        try:
            while process.poll() is None:
                healthy()
                if time.monotonic() > deadline:
                    raise TimeoutError("Controller service timeout")
                time.sleep(0.05)
            text = process.stdout.read()
            print(text, flush=True)
            if process.returncode:
                raise RuntimeError(f"Control command failed: {arguments}: {text}")
            return text
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    try:
        deadline = time.monotonic()+args.startup_timeout
        while received["at"] is None:
            healthy()
            if time.monotonic() > deadline:
                raise TimeoutError("Joint feedback unavailable")
            time.sleep(0.05)
        urdf = description(contract, received["positions"])
        (output / "franka_sim.ros2_control.urdf").write_text(urdf)
        pub.publish(String(data=urdf))
        manager = subprocess.Popen(["ros2", "run", "controller_manager", "ros2_control_node", "--ros-args",
                                    "--params-file", args.config], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        controllers = ["joint_state_broadcaster", "arm_controller"]
        if args.streaming:
            controllers.append("skill_controller")
        for controller in controllers:
            cli("load_controller", controller)
            report["lifecycle"].append({controller: "unconfigured", "output": cli("list_controllers")})
            cli("set_controller_state", controller, "inactive")
            report["lifecycle"].append({controller: "inactive", "output": cli("list_controllers")})
            if not (args.streaming and controller == "arm_controller"):
                cli("set_controller_state", controller, "active")
                report["lifecycle"].append({controller: "active", "output": cli("list_controllers")})
        for verb in ("list_controllers", "list_hardware_components", "list_hardware_interfaces"):
            (output / f"{verb}.txt").write_text(cli(verb))
        print("[CONTROL_STACK_READY]", flush=True)
        gateway = None
        control_status = node.create_publisher(String, "/harness/control_status", 10)
        if args.streaming:
            from trajectory_gateway import TrajectoryGateway
            gateway = TrajectoryGateway(node, lambda: [received["positions"][n] for n in NAMES],
                                        [received["positions"][n] for n in NAMES])
        start = time.monotonic()
        while time.monotonic()-start < args.duration:
            healthy()
            if gateway:
                gateway.tick()
                control_status.publish(String(data="ready"))
            if manager.poll() is not None:
                raise RuntimeError("Controller manager exited")
            time.sleep(0.05)
        report["success"] = True
    except KeyboardInterrupt:
        report["stopped_by_user"] = True
    except Exception as error:
        report["error"] = str(error)
        print(f"[CONTROL_FAULT] {error}", flush=True)
    finally:
        if manager and manager.poll() is None:
            import os
            os.killpg(manager.pid, signal.SIGINT)
            try:
                manager.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(manager.pid, signal.SIGKILL)
                manager.wait()
        log.close()
        (output / "supervisor.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        if rclpy.ok():
            rclpy.shutdown()
        spin.join(timeout=2)
        node.destroy_node()
    return 0 if report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
