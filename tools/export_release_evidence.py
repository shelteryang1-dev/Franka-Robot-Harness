#!/usr/bin/env python3
"""Export release evidence."""
import argparse
import gzip
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pipeline, formal, output = [(ROOT / p).resolve() for p in (args.pipeline, args.formal, args.output)]
    if not all(p.is_relative_to(ROOT / "results") for p in (pipeline, formal, output)):
        parser.error("Invalid result path")
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for base in (pipeline, formal / "infrastructure_attempts"):
            for path in base.rglob("*"):
                if not path.is_file() or path.suffix in (".mp4", ".zip", ".gz") or path.name == "STOP":
                    continue
                relative = str(path.relative_to(ROOT / "results/portfolio"))
                if path.name == "physics_trace.jsonl":
                    archive.writestr(relative + ".gz", gzip.compress(path.read_bytes(), mtime=0))
                else:
                    archive.write(path, relative)
    print(output)


if __name__ == "__main__":
    main()
