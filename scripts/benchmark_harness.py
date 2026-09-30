#!/usr/bin/env python3
"""Benchmark harness."""

import argparse
import json
import os
from pathlib import Path
import sys
import harness_client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key_stdin", action="store_true")
    parser.add_argument("--output", default="results/harness/benchmark")
    args = parser.parse_args()
    if args.key_stdin:
        os.environ["DEEPSEEK_API_KEY"] = sys.stdin.read().strip()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    cases = [("红色方块放到左侧，蓝色方块放到右侧，然后回到初始姿态", True),
             ("把红色方块从左侧搬到中间，再搬回左侧", True),
             ("如果红色方块已经在左侧就不需要移动它，把蓝色方块搬到中间", True),
             ("把蓝色方块搬回右侧，再回到初始姿态", True),
             ("抓取桌上的绿色杯子", False),
             ("将红色方块放到一个未定义的月球区域", False)]
    rows = []
    try:
        for index, (instruction, expected) in enumerate(cases):
            directory = root / f"case_{index:02d}"
            sys.argv = ["harness_client", "--instruction", instruction, "--output", str(directory)]
            code = harness_client.main()
            result = json.loads((directory/"result.json").read_text())

            rejected = not result.get("plan") and result.get("error_type") == "PlannerRejection"
            rows.append({"instruction": instruction, "expected_execution": expected,
                         "passed": result["success"] if expected else rejected,
                         "executed_successfully": result["success"], "exit_code": code,
                         "result": str(directory/"result.json")})
            (root/"summary.json").write_text(json.dumps({"cases": rows,
                "passed": sum(row["passed"] for row in rows), "total": len(rows)},
                ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        os.environ.pop("DEEPSEEK_API_KEY", None)
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
