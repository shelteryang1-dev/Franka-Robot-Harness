"""Check tray sort."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"source"))
from franka_manipulation_sim.harness.trays import TRAYS, inside_tray
import rclpy
from std_msgs.msg import String

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="results/harness/tray_final_check.json")
args = parser.parse_args()
rclpy.init(args=[])
node = rclpy.create_node("tray_sort_checker")
last, report = {}, {"success": False, "stable_sim_seconds": 0.0}
def receive(message):
    world = json.loads(message.data)
    if world.get("active_task") or not last:
        report["stable_sim_seconds"] = 0.
    else:
        dt = (world["step"]-last["step"])*.02
        if dt > 0:
            checks = {}
            for name, tray in TRAYS.items():
                key = tray["object"]
                position = world["objects"][key]
                velocity = [(a-b)/dt for a,b in zip(position, last["objects"][key])]
                checks[name] = inside_tray(position, velocity, (0,0,0), name)
            report["stable_sim_seconds"] = report["stable_sim_seconds"]+dt if all(checks.values()) else 0.
            report.update(checks=checks, positions=world["objects"])
            report["success"] = report["stable_sim_seconds"] >= .5
    last.clear()
    last.update(world)
node.create_subscription(String, "/harness/state", receive, 10)
start = time.monotonic()
while not report["success"] and time.monotonic()-start < 30:
    rclpy.spin_once(node, timeout_sec=.05)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
node.destroy_node()
rclpy.shutdown()
print(json.dumps(report, ensure_ascii=False))
raise SystemExit(0 if report["success"] else 1)
