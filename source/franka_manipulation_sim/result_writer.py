"""CSV and JSON result output."""

import csv
import json
from pathlib import Path

from .metrics import summarize_rows


def write_results(rows: list[dict], output_dir: str | Path, metadata: dict) -> tuple[Path, Path, dict]:

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / "episodes.csv"
    json_path = target_dir / "summary.json"

    if not rows:
        raise ValueError("No completed episodes")

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {**metadata, **summarize_rows(rows), "episodes_csv": str(csv_path)}
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return csv_path, json_path, summary
