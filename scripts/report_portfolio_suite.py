#!/usr/bin/env python3
"""Report portfolio suite."""

import argparse
import collections
import gzip
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--compress_traces", action="store_true")
    args = parser.parse_args()
    root = args.directory
    protocol = json.loads((root/"protocol.json").read_text(encoding="utf-8"))
    grouped = collections.defaultdict(list)
    rows = []
    for case in protocol["cases"]:
        directory = root/case["id"]
        result_path = directory/"client/result.json"
        report_path = directory/"case_report.json"
        if not report_path.exists():
            continue
        row = json.loads(report_path.read_text(encoding="utf-8"))
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            correct = result.get("plan") == case["expected_plan"]
            success = bool(result.get("success") and correct)
            if row["planning_correct"] != correct or row["task_success"] != success:
                raise ValueError(f"Result mismatch: {case['id']}")
            if success and not any(e["status"] == "succeeded" for e in result.get("events", [])):
                raise ValueError("Final state missing")
        rows.append(row)
        grouped[case["class"]].append(row)
        if args.compress_traces:
            trace = directory/"simulation/physics_trace.jsonl"
            if trace.exists():
                with trace.open("rb") as source, gzip.open(trace.with_suffix(trace.suffix+".gz"), "wb") as dest:

                    for chunk in iter(lambda: source.read(1024*1024), b""):
                        dest.write(chunk)
    attempted = [r for r in rows if "client_exit_code" in r]
    lengths = [r["execution_wall_seconds"] for r in attempted if "execution_wall_seconds" in r]
    successful = [r["execution_wall_seconds"] for r in attempted
                  if r["task_success"] and "execution_wall_seconds" in r]
    metrics = {"planner": protocol["planner"], "recording": protocol["record"],
        "planned_tasks": len(protocol["cases"]), "finished_cases": len(rows),
        "client_attempted": len(attempted), "infrastructure_failures": len(rows)-len(attempted),
        "plans_correct": sum(r["planning_correct"] for r in attempted),
        "complete_task_successes": sum(r["task_success"] for r in attempted),
        "complete_task_success_rate": sum(r["task_success"] for r in attempted)/len(attempted) if attempted else None,
        "mean_execution_wall_seconds_all_terminal_tasks": statistics.mean(lengths) if lengths else None,
        "median_execution_wall_seconds_all_terminal_tasks": statistics.median(lengths) if lengths else None,
        "mean_execution_wall_seconds_success_only": statistics.mean(successful) if successful else None,
        "physical_placements": sum(r.get("physical_placements", 0) for r in attempted),
        "failure_breakdown": dict(collections.Counter(r.get("error") or "unknown" for r in rows if not r["task_success"]))}
    (root/"metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Task benchmark", "", f"Planner: `{protocol['planner']}`",
        "Independent scenes; three layout seeds per class.",
        "10 task classes, 3 layout seeds.", "",
        f"Registered={len(protocol['cases'])}; recorded={len(rows)}; attempted={len(attempted)} cases",
        f"Correct plans={metrics['plans_correct']}/{len(attempted)}; successful tasks={metrics['complete_task_successes']}/{len(attempted)}.",
        "Startup failures and recording runs reported separately.", "",
        "| Class | Recorded | Correct plans | Successful tasks |", "|---|---:|---:|---:|"]
    for kind, group in grouped.items():
        lines.append(f"| {kind} | {len(group)} | {sum(r['planning_correct'] for r in group)} | {sum(r['task_success'] for r in group)} |")
    lines += ["", "## Failures", ""]
    for row in rows:
        if not row["task_success"]:
            lines.append(f"- `{row['id']}`：{row.get('error')}; steps={row.get('completed_steps', 0)}.")
    lines += ["", "## Files", "", "- `protocol.json`: instructions, plans, layouts, hashes.",
        "- `summary.csv`, `metrics.json`: results.",
        "- `client/result.json`: plan, states, events.",
        "- `simulation/events.jsonl`, `physics_trace.jsonl`: physical state.",
        "- `control/`: controller logs.",
        "- `commands.json`: launch commands.", "",
        "Failure labels use recorded terminal states.", ""]
    (root/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
