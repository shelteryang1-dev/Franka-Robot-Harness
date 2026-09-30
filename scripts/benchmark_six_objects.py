"""Benchmark six objects."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import harness_client

NAMES = [f"{color}_{letter}" for color in ("red","blue") for letter in "ABC"]
SAME = [(n, "red_tray" if n.startswith("red") else "blue_tray") for n in NAMES]
SWAP = [(n, "blue_tray" if n.startswith("red") else "red_tray") for n in NAMES]
CASES = [SAME, SWAP,
    [("red_A","red_tray"),("red_A","blue_tray"),("red_A","red_tray"),
     ("blue_B","blue_tray"),("blue_B","red_tray")],
    [(n,"red_tray") for n in NAMES], [(n,"blue_tray") for n in NAMES],
    [("red_A","red_tray"),("blue_A","blue_tray"),("red_B","blue_tray"),
     ("blue_B","red_tray"),("red_C","red_tray"),("blue_C","blue_tray")],
    [("red_C","blue_tray"),("blue_C","red_tray"),("red_A","blue_tray"),("blue_A","red_tray")], SAME]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--planner", choices=("rules","deepseek"), default="rules")
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--start_case", type=int, default=0, help="Starting case index")
    parser.add_argument("--key_stdin", action="store_true")
    args = parser.parse_args()
    if args.key_stdin:
        os.environ["DEEPSEEK_API_KEY"] = sys.stdin.read().strip()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    selected = list(enumerate(CASES))[args.start_case:args.start_case+args.cases]
    for index, pairs in selected:
        clauses = []
        for name, tray in pairs:
            color = "红" if name.startswith("red") else "蓝"
            dest = "红" if tray.startswith("red") else "蓝"
            clauses.append(f"把{name[-1]}{color}色物块放在{dest}色托盘里")
        instruction = "先"+"，然后".join(clauses)+"，最后回位"
        expected = {"steps": [{"skill":"pick_place","object":n,"target":t} for n,t in pairs]+[{"skill":"home"}]}
        directory = root/f"case_{index:02d}"
        sys.argv = ["client", "--planner", args.planner, "--instruction", instruction,
                    "--timeout", "650", "--output", str(directory)]
        code = harness_client.main()
        result = json.loads((directory/"result.json").read_text())
        events = result.get("events", [])
        steps = [e for e in events if e["status"] == "step_succeeded"]
        movement = [e for e in steps if e.get("detail",{}).get("tray") and not e["detail"].get("skipped")]
        rows.append({"case": index, "instruction": instruction,
            "planning_correct": result.get("plan") == expected,
            "task_success": bool(result["success"] and result.get("plan") == expected),
            "completed_steps": len(steps), "physical_placements": len(movement),
            "exit_code": code, "expected_plan": expected})
        summary = {"protocol": "Sequential tasks in one persistent scene",
            "planner": args.planner, "cases": rows, "attempted_tasks":len(rows),
            "planning_accuracy":sum(r["planning_correct"] for r in rows)/len(rows),
            "task_success_rate":sum(r["task_success"] for r in rows)/len(rows),
            "physical_placements":sum(r["physical_placements"] for r in rows)}
        (root/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
        with (root/"summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            fields = ["case", "instruction", "planning_correct", "task_success", "completed_steps", "physical_placements", "exit_code"]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        if not rows[-1]["task_success"]:
            return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
