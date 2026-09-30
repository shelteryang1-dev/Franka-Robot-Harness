#!/usr/bin/env python3
"""Export portfolio evidence."""

import argparse
import gzip
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.directory
    protocol = json.loads((root/"protocol.json").read_text(encoding="utf-8"))
    completed = [case for case in protocol["cases"] if (root/case["id"]/"case_report.json").exists()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("Snapshot already exists")
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("protocol.json", "summary.json", "summary.csv", "metrics.json", "REPORT.md", "runner_state.json"):
            path = root/name
            if path.exists():
                archive.write(path, root.name+"/"+name)
        for case in completed:
            directory = root/case["id"]
            for path in directory.rglob("*"):
                if not path.is_file() or path.suffix in (".mp4", ".gz") or path.name == "STOP":
                    continue
                relative = str(path.relative_to(root)).replace("\\", "/")
                if path.name == "physics_trace.jsonl":
                    archive.writestr(root.name+"/"+relative+".gz", gzip.compress(path.read_bytes(), mtime=0))
                else:
                    archive.write(path, root.name+"/"+relative)
        archive.writestr(root.name+"/SNAPSHOT.json", json.dumps({"completed_case_ids": [c["id"] for c in completed],
            "note": "Completed cases only"}, ensure_ascii=False, indent=2))
    print(json.dumps({"snapshot": str(args.output), "finished": len(completed)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
