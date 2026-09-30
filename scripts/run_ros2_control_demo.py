#!/usr/bin/env python3
"""Run the end-effector control demo."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from franka_manipulation_sim.ros2.runtime import prepare_runtime

prepare_runtime()
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--duration", type=float, default=120, help="Run duration (s)")
parser.add_argument("--output_dir", default="results/ros2_control")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs != 1:
    parser.error("ROS 2 control requires one environment")
if args.duration <= 0:
    parser.error("Duration must be > 0")
args.kit_args = (args.kit_args or "") + (
    " --enable isaacsim.ros2.bridge"
    f" --/exts/isaacsim.ros2.bridge/ros_distro={os.environ['ROS_DISTRO']}"
    " --/exts/isaacsim.ros2.bridge/publish_without_verification=true"
)
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import torch
import isaaclab_tasks
from isaaclab.utils.math import subtract_frame_transforms
from franka_manipulation_sim.tasks.env_cfg import FrankaManipulationEnvCfg, configure_randomization
from franka_manipulation_sim.ros2.state_publisher import Ros2StatePublisher
from franka_manipulation_sim.ros2.command_subscriber import Ros2CommandSubscriber


def main():
    env = publisher = subscriber = None
    report = {"success": False, "command_frame": "world", "action_frame": "robot_root"}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        cfg = FrankaManipulationEnvCfg()
        cfg.scene.num_envs = 1
        cfg.sim.device = args.device
        cfg.seed = 42


        cfg.scene.ee_frame.target_frames[0].offset.pos = tuple(cfg.actions.arm_action.body_offset.pos)
        cfg.scene.ee_frame.target_frames[0].offset.rot = tuple(cfg.actions.arm_action.body_offset.rot)
        configure_randomization(cfg, 0, 0, 0.2, 0)

        cfg.terminations.time_out = None
        cfg.terminations.state_machine_finished = None
        env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=cfg)
        env.reset()
        raw = env.unwrapped
        ee = raw.scene["ee_frame"]
        robot = raw.scene["robot"]
        initial_position = ee.data.target_pos_w.torch[0, 0].tolist()
        initial_quaternion = ee.data.target_quat_w.torch[0, 0].tolist()
        fingers = [i for i, name in enumerate(robot.joint_names) if "finger_joint" in name]
        closed = robot.data.joint_pos.torch[0, fingers].mean().item() < 0.02
        subscriber = Ros2CommandSubscriber(initial_position, initial_quaternion, closed)
        publisher = Ros2StatePublisher(raw)
        action = torch.zeros((1, 8), device=raw.device)

        root_p = torch.tensor([[1., 2., 3.]], device=raw.device)
        root_q = torch.tensor([[0., 0., 2**-0.5, 2**-0.5]], device=raw.device)
        test_p, test_q = subtract_frame_transforms(
            root_p, root_q, torch.tensor([[1., 3., 3.]], device=raw.device), root_q,
        )
        torch.testing.assert_close(test_p, torch.tensor([[1., 0., 0.]], device=raw.device), atol=1e-6, rtol=0)
        torch.testing.assert_close(test_q, torch.tensor([[0., 0., 0., 1.]], device=raw.device), atol=1e-6, rtol=0)
        report["nonidentity_frame_test"] = True
        started = time.monotonic()
        report["initial_ee_world"] = initial_position
        print("[ROS2_CONTROL_READY]", flush=True)
        with torch.inference_mode():
            while app.is_running() and time.monotonic() - started < args.duration:
                tick = time.monotonic()
                position, quaternion, gripper = subscriber.get_latest_command()


                pos_b, quat_b = subtract_frame_transforms(
                    robot.data.root_pos_w.torch, robot.data.root_quat_w.torch,
                    torch.tensor([position], device=raw.device),
                    torch.tensor([quaternion], device=raw.device),
                )
                action[:, :3], action[:, 3:7], action[:, 7] = pos_b, quat_b, gripper
                _, _, terminated, truncated, _ = env.step(action)
                publisher.publish()
                if bool(terminated.any() or truncated.any()):
                    raise RuntimeError("Environment terminated")
                time.sleep(max(0, raw.step_dt - (time.monotonic() - tick)))
        report.update(success=True, steps=int(raw.common_step_counter), wall_seconds=time.monotonic()-started)
    finally:
        if subscriber:
            report["commands"] = subscriber.counts
            subscriber.close()
        if publisher:
            publisher.close()
        if env:
            env.close()
        (output / "server.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        try:
            app.close(exit_code=exit_code)
        finally:
            raise SystemExit(exit_code)
