#!/usr/bin/env python3
"""Submit a skill plan or natural-language task."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"source"))
from franka_manipulation_sim.harness.contracts import validate_plan
from franka_manipulation_sim.harness.planner import plan_instruction
import rclpy
from std_msgs.msg import String


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--instruction")
    group.add_argument("--plan", help="Fixed plan file")
    group.add_argument("--cancel", help="Task ID to cancel")
    parser.add_argument("--key_stdin", action="store_true", help="Read API key from stdin")
    parser.add_argument("--output", default="results/harness/client")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--expected_plan", type=Path, help="Expected plan file")
    parser.add_argument("--planner", choices=("deepseek", "rules"), default="deepseek",
                        help="Planner backend")
    args = parser.parse_args()
    key = sys.stdin.read().strip() if args.key_stdin else None
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rclpy.init(args=[])
    node = rclpy.create_node("harness_task_client")
    pub = node.create_publisher(String, "/harness/command", 10)
    world, events = {}, []
    task_id = args.cancel or str(uuid.uuid4())
    report = {"task_id": task_id, "success": False}
    submitted = False
    cancel_current = args.instruction and args.instruction.strip() in ("停止", "取消当前任务", "停止当前任务")

    def receive_world(msg):
        world.update(json.loads(msg.data))
        world["received_at"] = time.monotonic()

    def receive_event(msg):
        event = json.loads(msg.data)
        if event.get("task_id") == task_id:
            events.append(event)
            print(json.dumps(event, ensure_ascii=False), flush=True)

    node.create_subscription(String, "/harness/state", receive_world, 10)
    node.create_subscription(String, "/harness/events", receive_event, 30)

    def wait(predicate, seconds):
        start = time.monotonic()
        while time.monotonic()-start < seconds:
            rclpy.spin_once(node, timeout_sec=.05)
            if predicate():
                return
        raise TimeoutError("Task service or result timeout")

    try:
        wait(lambda: bool(world) and pub.get_subscription_count() > 0
             and (args.cancel or cancel_current or world.get("control_ready")), args.timeout)
        if cancel_current:
            task_id = world.get("active_task")
            if not task_id:
                raise ValueError("No active task")
            report["task_id"] = task_id
        if args.cancel or cancel_current:
            request = {"task_id": task_id, "cancel": True}
        else:
            report["initial_world"] = {k:v for k,v in world.items() if k != "received_at"}
            if args.instruction:
                scene = {"objects": world["objects"], "targets": world.get("targets", ["left", "right", "center"])}
                if args.planner == "rules":
                    from franka_manipulation_sim.harness.language import parse_instruction
                    plan = parse_instruction(args.instruction, scene)
                    metadata = {"type": "restricted_chinese", "instruction": args.instruction}
                else:
                    plan, metadata = plan_instruction(args.instruction, scene, key)
                report["planner"] = metadata
            else:
                plan = validate_plan(json.loads(Path(args.plan).read_text()), tuple(world["objects"]))
                report["planner"] = {"type": "fixed_plan"}
            report["plan"] = plan
            if args.expected_plan:
                expected = json.loads(args.expected_plan.read_text(encoding="utf-8"))
                report["planning_correct"] = plan == expected
                if plan != expected:
                    raise ValueError("Plan mismatch")
            request = {"task_id": task_id, "plan": plan}
        report["submitted_wall_time"] = time.time()
        pub.publish(String(data=json.dumps(request)))
        submitted = True
        wait(lambda: any(e["status"] in ("succeeded", "failed", "rejected", "cancelled") for e in events), args.timeout)
        report["success"] = events[-1]["status"] == ("cancelled" if args.cancel or cancel_current else "succeeded")
        report["terminal_wall_time"] = time.time()

        terminal_step = events[-1].get("sim_step", -1)
        wait(lambda: world.get("step", -1) >= terminal_step, 5.)
    except Exception as error:
        report["success"] = False
        report["error"] = str(error)
        report["error_type"] = type(error).__name__

        if submitted:
            pub.publish(String(data=json.dumps({"task_id": task_id, "cancel": True})))
            for _ in range(5):
                rclpy.spin_once(node, timeout_sec=.05)
    finally:
        report["events"] = events
        report["final_world"] = {k:v for k,v in world.items() if k != "received_at"}
        (output/"result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        node.destroy_node()
        rclpy.shutdown()
    print(json.dumps({"success": report["success"], "output": str(output)}, ensure_ascii=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
