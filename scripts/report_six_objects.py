"""Report six objects."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="+")
    parser.add_argument("--output", help="Merged report directory")
    args = parser.parse_args()
    if len(args.directory) > 1 and not args.output:
        parser.error("Merged reports require --output")
    output = Path(args.output or args.directory[0])
    output.mkdir(parents=True, exist_ok=True)
    cases, sources = [], []
    for name in args.directory:
        root = Path(name)
        summary = json.loads((root/"summary.json").read_text(encoding="utf-8"))
        cases.extend((root, case) for case in summary["cases"])
        sources.append({"directory": name, "protocol": summary["protocol"],
                        "cases": [case["case"] for case in summary["cases"]]})
    if len({case["case"] for _, case in cases}) != len(cases):
        raise ValueError("Duplicate case ID")
    rows = []
    for root, case in sorted(cases, key=lambda pair: pair[1]["case"]):
        result = json.loads((root/f'case_{case["case"]:02d}'/"result.json").read_text(encoding="utf-8"))
        events = result.get("events", [])
        terminal = next((e for e in reversed(events) if e["status"] in ("succeeded", "failed", "rejected", "cancelled")), {})
        correct_plan = result.get("plan") == case["expected_plan"]
        passed = bool(result["success"] and correct_plan and terminal.get("status") == "succeeded")
        if passed != case["task_success"] or correct_plan != case["planning_correct"]:
            raise ValueError("Result mismatch")
        steps = [e for e in events if e["status"] == "step_succeeded"]
        placements = [e for e in steps if e.get("detail", {}).get("tray")]
        rows.append({"case": case["case"], "source": str(root), "planning_correct": correct_plan,
            "task_success": passed, "planned_steps": len(case["expected_plan"]["steps"]),
            "retry_events": sum(e["status"] == "retrying" for e in events),
            "completed_steps": len(steps), "physical_placements": sum(not e["detail"].get("skipped", False) for e in placements),
            "skipped_placements": sum(bool(e["detail"].get("skipped")) for e in placements),
            "wall_seconds": terminal.get("wall_seconds"), "terminal": terminal.get("status", result.get("error_type", "missing")),
            "detail": json.dumps(terminal.get("detail", result.get("error")), ensure_ascii=False)})
    if not rows:
        raise ValueError("Execution records missing")
    with (output/"audit.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = {"attempted_tasks":len(rows), "successful_tasks":sum(r["task_success"] for r in rows),
        "correct_plans":sum(r["planning_correct"] for r in rows),
        "completed_steps":sum(r["completed_steps"] for r in rows),
        "planned_steps":sum(r["planned_steps"] for r in rows),
        "physical_placements":sum(r["physical_placements"] for r in rows),
        "skipped_placements":sum(r["skipped_placements"] for r in rows),
        "retry_events":sum(r["retry_events"] for r in rows),
        "successful_without_retry":sum(r["task_success"] and r["retry_events"] == 0 for r in rows)}
    metrics["task_success_rate"] = metrics["successful_tasks"]/len(rows)
    metrics["planning_accuracy"] = metrics["correct_plans"]/len(rows)
    (output/"metrics.json").write_text(json.dumps({**metrics, "scene_runs": sources},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
