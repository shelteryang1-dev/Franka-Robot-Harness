#!/usr/bin/env python3
"""Build portfolio video."""

import argparse
import bisect
import hashlib
import json
from pathlib import Path
import subprocess


def probe(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames:format=duration",
        "-of", "json", str(path)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def stamp(seconds):
    ms = round(seconds*1000)
    return f"{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no_denoise", action="store_true")
    args = parser.parse_args()
    raw = (args.case/"raw.mp4").resolve()
    output = args.output.resolve()
    if output.exists() or output == raw:
        parser.error("Output exists or matches input")
    output.parent.mkdir(parents=True, exist_ok=True)
    before = probe(raw)
    stream = before["streams"][0]
    n, d = map(int, stream["avg_frame_rate"].split("/"))
    fps = n/d
    count = int(stream["nb_read_frames"])
    duration = count/fps
    index_path = args.case/"simulation/video_index.jsonl"
    indices = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    sim_steps = [r["sim_step"] for r in indices]
    client = json.loads((args.case/"client/result.json").read_text(encoding="utf-8"))
    kind = client.get("planner", {}).get("type")
    planner = "Model planner" if client.get("planner", {}).get("model") else (
        "Rule parser" if kind == "restricted_chinese" else "Fixed plan")
    labels = [(0., f"{planner} | simulation time\nTask started")]
    for event in client.get("events", []):
        position = bisect.bisect_left(sim_steps, event.get("sim_step", 0))
        moment = (indices[min(position, len(indices)-1)]["frame"] if indices else 0)/fps
        if event["status"] == "running":
            step_number = event["step"]
            step = client["plan"]["steps"][step_number]
            name = step.get("object", "")
            name = name.replace("red_", "red ").replace("blue_", "blue ")
            target = {"red_tray": "Red tray", "blue_tray": "Blue tray"}.get(step.get("target"), "")
            operation = "Home trajectory" if step["skill"] == "home" else f"{name} → {target}"
            labels.append((moment, f"{planner} | simulation time\nStep {step_number+1}: {operation}"))
        elif event["status"] == "retrying":
            labels.append((moment, f"{planner} | simulation time\nRetry: {event.get('detail')}"))
    labels.sort(key=lambda item: item[0])
    subtitles = output.with_suffix(".srt")
    chunks = []
    for i, (start, text) in enumerate(labels):
        end = labels[i+1][0] if i+1 < len(labels) else duration
        if end > start:
            chunks.append(f"{len(chunks)+1}\n{stamp(start)} --> {stamp(end)}\n{text}\n")
    subtitles.write_text("\n".join(chunks), encoding="utf-8")

    denoise = None if args.no_denoise else "hqdn3d=0.8:0.6:1.5:1.2"
    escaped = str(subtitles).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    filters = ([denoise] if denoise else [])+[
        f"subtitles=filename='{escaped}':force_style='FontName=Noto Sans CJK SC,FontSize=17,Outline=1,MarginV=14'"]
    command = ["ffmpeg", "-nostdin", "-n", "-i", str(raw), "-vf", ",".join(filters),
        "-an", "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-map_metadata", "-1", "-movflags", "+faststart", str(output)]
    completed = subprocess.run(command, capture_output=True, text=True)
    output.with_suffix(".encode.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError("Video encoding failed")
    after = probe(output)
    if int(after["streams"][0]["nb_read_frames"]) != count:
        raise RuntimeError("Frame count mismatch")
    qa = output.with_suffix(".qa")
    qa.mkdir()
    for label, video in (("raw", raw), ("processed", output)):
        for frame in sorted({0, count//4, count//2, count*3//4, count-1}):
            subprocess.run(["ffmpeg", "-nostdin", "-n", "-v", "error", "-i", str(video),
                "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1", str(qa/f"{label}_{frame}.png")], check=True)
    black = subprocess.run(["ffmpeg", "-nostdin", "-i", str(output), "-vf",
        "blackdetect=d=0.02:pix_th=0.02", "-an", "-f", "null", "-"], capture_output=True, text=True, check=True)
    black_events = [line for line in black.stderr.splitlines() if "black_start:" in line]
    manifest = {"raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "raw": str(raw),
        "output": str(output), "before": before, "after": after, "filter": denoise,
        "command": command, "black_events": black_events, "planner": planner,
        "physical_task_success": client.get("success"), "manual_quality_review": "Review pending",
        "timing_note": "All frames; simulation time; task-active frames only"}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"video": str(output), "frames": count, "black_events": black_events}, ensure_ascii=False))
    return 0 if not black_events else 1


if __name__ == "__main__":
    raise SystemExit(main())
