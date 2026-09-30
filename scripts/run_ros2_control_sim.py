#!/usr/bin/env python3
"""Run the ros2_control simulation backend."""

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
parser.add_argument("--duration", type=float, default=300)
parser.add_argument("--output_dir", default="results/ros2_control_stack")
parser.add_argument("--drop_feedback_after", type=float, default=0, help="Stop feedback after ready (s)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs != 1 or args.duration <= 0:
    parser.error("Requires one environment and duration > 0")
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
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from franka_manipulation_sim.tasks.env_cfg import FrankaManipulationEnvCfg, configure_randomization
from franka_manipulation_sim.ros2.state_publisher import Ros2StatePublisher
from franka_manipulation_sim.ros2.ros2_control_adapter import Ros2ControlAdapter


def main():
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    env = publisher = adapter = None
    report = {"success": False}
    try:
        cfg = FrankaManipulationEnvCfg()
        cfg.scene.num_envs, cfg.sim.device, cfg.seed = 1, args.device, 42
        cfg.terminations.time_out = cfg.terminations.state_machine_finished = None
        names = [f"panda_joint{i}" for i in range(1, 8)]
        cfg.actions.arm_action = JointPositionActionCfg(
            asset_name="robot", joint_names=names, scale=1.0, use_default_offset=False, preserve_order=True)
        configure_randomization(cfg, 0, 0, 0.2, 0)
        env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=cfg)
        env.reset()
        raw, robot = env.unwrapped, env.unwrapped.scene["robot"]
        indices = [robot.joint_names.index(name) for name in names]
        limits = robot.data.joint_pos_limits.torch[0].cpu().tolist()
        initial = robot.data.joint_pos.torch[0].cpu().tolist()
        velocity = robot.data.joint_vel_limits.torch[0].cpu().tolist()
        effort = robot.data.joint_effort_limits.torch[0].cpu().tolist()
        contract = {"source": cfg.scene.robot.spawn.usd_path, "units": "rad for arm, m for fingers",
                    "joints": {name: {"lower": limits[i][0], "upper": limits[i][1],
                                      "velocity": velocity[i], "effort": effort[i], "initial": initial[i]}
                               for i, name in enumerate(robot.joint_names)}}
        (output / "asset_joint_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
        adapter = Ros2ControlAdapter(names, [limits[i] for i in indices], [initial[i] for i in indices])
        publisher = Ros2StatePublisher(raw, joint_topic="/isaac_joint_states_raw", publish_clock=True)
        action = torch.zeros((1, 8), device=raw.device)
        start = time.monotonic()
        print("[ISAAC_CONTROL_READY]", flush=True)
        with torch.inference_mode():
            while app.is_running() and time.monotonic()-start < args.duration:
                tick = time.monotonic()
                targets, gripper = adapter.step(robot.data.joint_pos.torch[0, indices].tolist())
                action[0, :7] = torch.tensor(targets, device=raw.device)
                action[0, 7] = gripper
                _, _, terminated, truncated, _ = env.step(action)
                if bool(terminated.any() or truncated.any()):
                    raise RuntimeError("Environment terminated")
                if not args.drop_feedback_after or time.monotonic()-start < args.drop_feedback_after:
                    publisher.publish()
                time.sleep(max(0.0, raw.step_dt-(time.monotonic()-tick)))
        report.update(success=not bool(adapter.fault), steps=int(raw.common_step_counter),
                      commands=adapter.received, fault=adapter.fault)
    finally:
        if adapter:
            adapter.close()
        if publisher:
            publisher.close()
        if env:
            env.close()
        (output / "simulator.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(report, flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        traceback.print_exc()
    finally:
        app.close(exit_code=code)
    raise SystemExit(code)
