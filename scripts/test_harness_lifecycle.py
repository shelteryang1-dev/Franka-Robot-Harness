#!/usr/bin/env python3
"""Test harness lifecycle."""

import argparse
import json
from pathlib import Path
import time
import uuid
import math
import subprocess
import sys
import rclpy
from std_msgs.msg import String
from sensor_msgs.msg import JointState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/harness/lifecycle")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rclpy.init(args=[])
    node = rclpy.create_node("harness_lifecycle_test")
    publisher = node.create_publisher(String, "/harness/command", 10)
    events, states, joints = [], {}, []
    node.create_subscription(String, "/harness/events", lambda m: events.append(json.loads(m.data)), 20)
    node.create_subscription(String, "/harness/state", lambda m: states.update(json.loads(m.data)), 10)
    node.create_subscription(JointState, "/joint_states", lambda m: joints.append(list(m.position)), 10)
    report = {"success": False}
    def wait(predicate, timeout=20):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.02)
            if predicate():
                return
        raise TimeoutError("Lifecycle test timeout")
    def got(identity, status):
        return any(e.get("task_id") == identity and e["status"] == status for e in events)
    def send(value):
        publisher.publish(String(data=json.dumps(value)))
    def pump(seconds):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.02)
    identity = str(uuid.uuid4())
    try:
        wait(lambda: states.get("control_ready") and joints and publisher.get_subscription_count())
        send({"task_id": identity+"invalid", "plan": {"steps": [{"skill": "shell"}]}})
        wait(lambda: got(identity+"invalid", "rejected"))
        regions = {"center": [.5, 0, .021], "left": [.5, .16, .021], "right": [.5, -.16, .021]}
        target = next(name for name, pose in regions.items()
                      if all(math.dist(pose, position) > .08 for position in states["objects"].values()))
        plan = {"steps": [{"skill": "pick_place", "object": "cube", "target": target}]}
        send({"task_id": identity, "plan": plan})
        wait(lambda: got(identity, "running"))
        send({"task_id": identity+"busy", "plan": plan})
        wait(lambda: got(identity+"busy", "rejected"))
        pump(.3)
        cancel = subprocess.Popen([sys.executable, str(Path(__file__).with_name("harness_client.py")),
            "--instruction", "停止当前任务", "--output", str(output/"natural_cancel")])
        wait(lambda: got(identity, "cancelled"))
        if cancel.wait(timeout=10) != 0:
            raise RuntimeError("Cancel request failed")
        pump(.5)
        before = joints[-1]
        pump(.5)
        drift = max(abs(a-b) for a,b in zip(before, joints[-1]))
        send({"task_id": identity, "plan": plan})
        wait(lambda: got(identity, "rejected"))
        report.update(success=drift < .03, joint_drift_rad=drift, samples=len(joints))
    except Exception as error:
        report["error"] = str(error)
        send({"task_id": identity, "cancel": True})
        pump(.3)
    finally:
        report["events"] = events
        (output/"result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        node.destroy_node()
        rclpy.shutdown()
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
