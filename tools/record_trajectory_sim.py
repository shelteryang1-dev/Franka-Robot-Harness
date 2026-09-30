#!/usr/bin/env python3
"""Record trajectory sim."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"source"))
from franka_manipulation_sim.ros2.runtime import prepare_runtime
prepare_runtime()
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--duration", type=float, default=240)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.kit_args = (args.kit_args or "")+(
    " --enable isaacsim.ros2.bridge --/exts/isaacsim.ros2.bridge/ros_distro=jazzy"
    " --/exts/isaacsim.ros2.bridge/publish_without_verification=true"
    " --/persistent/rtx/modes/pt/enabled=true --/rtx/rendermode=PathTracing")
app = AppLauncher(args).app

import carb
import gymnasium as gym
import imageio.v2 as imageio
import torch
import isaaclab_tasks
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab_visualizers.kit import KitVisualizerCfg
from franka_manipulation_sim.tasks.env_cfg import FrankaManipulationEnvCfg, configure_randomization
from franka_manipulation_sim.ros2.state_publisher import Ros2StatePublisher
from franka_manipulation_sim.ros2.ros2_control_adapter import Ros2ControlAdapter


def main():
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = FrankaManipulationEnvCfg()
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = 1, args.device, 42
    for name in cfg.terminations.to_dict():
        setattr(cfg.terminations, name, None)
    cfg.commands.object_pose.debug_vis = cfg.scene.ee_frame.debug_vis = False
    names = [f"panda_joint{i}" for i in range(1, 8)]
    cfg.actions.arm_action = JointPositionActionCfg(asset_name="robot", joint_names=names,
        scale=1., use_default_offset=False, preserve_order=True)
    cfg.sim.visualizer_cfgs = [KitVisualizerCfg(eye=(1.55, 1.25, .95), lookat=(.48, 0., .18),
        headless=True, max_visible_envs=1, window_width=1280, window_height=720,
        randomly_sample_visible_envs=False, enable_live_plots=False)]
    cfg.viewer.resolution = (1280, 720)
    cfg.video_recorder.window_width, cfg.video_recorder.window_height = 1280, 720
    configure_randomization(cfg, 0, 0, .2, 0)
    env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=cfg, render_mode="rgb_array")
    env.reset()
    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "PathTracing")
    settings.set("/rtx/pathtracing/spp", 1)
    settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    settings.set("/rtx/pathtracing/optixDenoiser/blendFactor", 0.)
    raw, robot = env.unwrapped, env.unwrapped.scene["robot"]
    ids = [robot.joint_names.index(name) for name in names]
    limits = robot.data.joint_pos_limits.torch[0].tolist()
    initial = robot.data.joint_pos.torch[0].tolist()
    contract = {"source": cfg.scene.robot.spawn.usd_path, "joints": {name: {
        "lower": limits[i][0], "upper": limits[i][1], "initial": initial[i],
        "velocity": float(robot.data.joint_vel_limits.torch[0, i]),
        "effort": float(robot.data.joint_effort_limits.torch[0, i])}
        for i, name in enumerate(robot.joint_names)}}
    (output/"asset_joint_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    adapter = Ros2ControlAdapter(names, [limits[i] for i in ids], [initial[i] for i in ids])
    publisher = Ros2StatePublisher(raw, joint_topic="/isaac_joint_states_raw", publish_clock=True)
    for _ in range(16):
        env.render()
    writer = imageio.get_writer(str(output/"raw.mp4"), fps=50, codec="libx264", macro_block_size=1,
        ffmpeg_params=["-crf", "18", "-movflags", "+faststart"])
    writer.append_data(env.render())
    trace = (output/"physics_trace.jsonl").open("w", encoding="utf-8")
    action = torch.zeros((1, 8), device=raw.device)
    started, frames = time.monotonic(), 1
    print("[TRAJECTORY_RECORDING_READY]", flush=True)
    try:
        with torch.inference_mode():
            while app.is_running() and time.monotonic()-started < args.duration and not (output/"STOP").exists():
                tick = time.monotonic()
                target, grip = adapter.step(robot.data.joint_pos.torch[0, ids].tolist())
                action[0, :7], action[0, 7] = torch.tensor(target, device=raw.device), grip
                _, _, terminated, truncated, _ = env.step(action)
                if bool(terminated.any() or truncated.any()):
                    raise RuntimeError("Recording environment terminated")
                publisher.publish()
                if adapter.last_command is not None and frames < 3000:
                    writer.append_data(env.render())
                    frames += 1
                    trace.write(json.dumps({"frame": frames-1, "sim_step": int(raw.common_step_counter),
                        "joint_names": names, "actual": robot.data.joint_pos.torch[0, ids].tolist(),
                        "hardware_command": target, "wall_time": time.time()})+"\n")
                    trace.flush()
                time.sleep(max(0., raw.step_dt-(time.monotonic()-tick)))
    finally:
        writer.close()
        trace.close()
        (output/"recording.json").write_text(json.dumps({"frames": frames, "fps": 50,
            "wall_seconds": time.monotonic()-started, "fault": adapter.fault}), encoding="utf-8")
        adapter.close()
        publisher.close()
        env.close()


if __name__ == "__main__":
    code = 1
    try:
        main()
        code = 0
    finally:
        app.close(exit_code=code)
    raise SystemExit(code)
