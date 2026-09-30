#!/usr/bin/env python3
"""Finalize portfolio media."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_portfolio_video import probe, stamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, required=True)
    args = parser.parse_args()
    pipeline = (ROOT / args.pipeline).resolve()
    if not pipeline.is_relative_to(ROOT / "results"):
        parser.error("Invalid pipeline path")
    raw = pipeline / "standard_trajectory_demo/simulation/raw.mp4"
    output = ROOT / "media/portfolio/standard_trajectory_demo.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error("Output already exists")
    result = json.loads((pipeline / "standard_trajectory_demo/client/trajectory.json").read_text())
    if not result["success"] or not result["accepted"] or result["error_code"] != 0:
        raise RuntimeError("Trajectory verification failed")
    before = probe(raw)
    subtitle = output.with_suffix(".srt")
    subtitle.write_text("1\n00:00:00,000 --> " + stamp(float(before["format"]["duration"])) +
        "\nJoint trajectory | 4 waypoints | 7 joints\nSimulation time | action succeeded\n", encoding="utf-8")
    command = ["ffmpeg", "-nostdin", "-n", "-i", str(raw), "-vf",
        "hqdn3d=0.8:0.6:1.5:1.2,subtitles=filename='" + str(subtitle) +
        "':force_style='FontName=Noto Sans CJK SC,FontSize=17,Outline=1,MarginV=14'",
        "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-map_metadata", "-1",
        "-movflags", "+faststart", str(output)]
    run = subprocess.run(command, capture_output=True, text=True, check=True)
    output.with_suffix(".encode.log").write_text(run.stderr, encoding="utf-8")
    after = probe(output)
    if before["streams"][0]["nb_read_frames"] != after["streams"][0]["nb_read_frames"]:
        raise RuntimeError("Frame count mismatch")
    manifest = {"raw": str(raw), "output": str(output), "before": before, "after": after,
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "command": command,
        "filter": "hqdn3d=0.8:0.6:1.5:1.2", "physical_task_success": result["success"],
        "manual_quality_review": "Review pending", "timing_note": "All frames; simulation time"}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    videos = []
    for name in ("language_sequence", "six_color_sort", "ppo_mixed_demo", "standard_trajectory_demo", "failure_repeat_seed42"):
        video = output.parent / (name + ".mp4")
        review = output.parent / (name + ".review")
        review.mkdir(exist_ok=False)
        scan = subprocess.run(["ffmpeg", "-nostdin", "-i", str(video), "-vf", "blackdetect=d=0.02:pix_th=0.02",
            "-an", "-f", "null", "-"], capture_output=True, text=True, check=True)
        black = [line for line in scan.stderr.splitlines() if "black_start:" in line]
        (review / "decode.log").write_text(scan.stderr, encoding="utf-8")
        subprocess.run(["ffmpeg", "-nostdin", "-n", "-v", "error", "-i", str(video), "-vf",
            "fps=1,scale=320:180,drawtext=text='%{pts\\:hms}':fontsize=15:fontcolor=yellow:x=4:y=4,tile=4x3",
            str(review / "sheet_%02d.jpg")], check=True)
        videos.append({"name": name, "black_events": black, "video": probe(video),
            "contact_sheets": len(list(review.glob("sheet_*.jpg")))})
    (pipeline / "video_decode_checks.json").write_text(json.dumps(videos, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(videos, ensure_ascii=False))


if __name__ == "__main__":
    main()
