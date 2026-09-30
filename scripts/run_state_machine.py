#!/usr/bin/env python3
"""Run the state-machine baseline."""

import argparse
import importlib.metadata
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "source"
sys.path.insert(0, str(SOURCE_ROOT))


def _restart_with_ros2_runtime_if_needed() -> None:

    if "--ros2_publish" not in sys.argv or os.environ.get("FRANKA_ROS2_ENV_READY") == "1":
        return
    ubuntu_major = platform.freedesktop_os_release().get("VERSION_ID", "24").split(".")[0]
    ros2_distribution = "humble" if ubuntu_major == "22" else "jazzy"
    ros2_library_path = f"/isaac-sim/exts/isaacsim.ros2.core/{ros2_distribution}/lib"
    environment = os.environ.copy()
    environment["ROS_DISTRO"] = ros2_distribution
    environment["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    environment["LD_LIBRARY_PATH"] = (
        f"{ros2_library_path}:{environment.get('LD_LIBRARY_PATH', '')}"
    ).rstrip(":")
    environment["FRANKA_ROS2_ENV_READY"] = "1"
    os.execve("/isaac-sim/python.sh", ["/isaac-sim/python.sh", *sys.argv], environment)


_restart_with_ros2_runtime_if_needed()

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Franka state machine")
parser.add_argument("--gate", type=int, choices=(1, 2), default=2, help="1: lift; 2: pick-and-place")
parser.add_argument("--num_envs", type=int, default=1, help="Environment count")
parser.add_argument("--num_episodes", type=int, default=1, help="Total episodes")
parser.add_argument("--seed", type=int, default=42, help="Seed")
parser.add_argument("--output_dir", type=str, default="results/latest", help="Result directory")
parser.add_argument("--object_xy_range", type=float, default=0.0, help="Object XY range (m)")
parser.add_argument("--target_xy_range", type=float, default=0.0, help="Target XY range (m)")
parser.add_argument("--target_z", type=float, default=None, help="Target height (m)")
parser.add_argument(
    "--mass_scale_range",
    type=float,
    default=0.0,
    help="Mass scale range",
)
parser.add_argument(
    "--stage_timeout",
    type=float,
    default=3.0,
    help="Stage timeout (s)",
)
parser.add_argument("--ros2_publish", action="store_true", help="Publish ROS 2 state")
parser.add_argument(
    "--ros2_publish_every_n_steps",
    type=int,
    default=1,
    help="Pose publish interval (steps)",
)
parser.add_argument("--ros2_realtime", action="store_true", help="Run at simulation rate")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs < 1 or args_cli.num_episodes < 1:
    parser.error("Environment and episode counts must be >= 1")
if args_cli.stage_timeout <= 0.0:
    parser.error("Stage timeout must be > 0")
if args_cli.ros2_publish and args_cli.num_envs != 1:
    parser.error("ROS 2 requires one environment")
if args_cli.ros2_publish_every_n_steps < 1:
    parser.error("Publish interval must be >= 1")

if args_cli.ros2_publish:
    ros2_distribution = os.environ.setdefault("ROS_DISTRO", "jazzy")
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    ros2_library_path = f"/isaac-sim/exts/isaacsim.ros2.core/{ros2_distribution}/lib"
    current_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if ros2_library_path not in current_library_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{ros2_library_path}:{current_library_path}".rstrip(":")
    ros2_kit_args = (
        "--enable isaacsim.ros2.bridge "
        f"--/exts/isaacsim.ros2.bridge/ros_distro={ros2_distribution} "
        "--/exts/isaacsim.ros2.bridge/publish_without_verification=true"
    )
    args_cli.kit_args = f"{args_cli.kit_args or ''} {ros2_kit_args}".strip()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import warp as wp

import isaaclab
import isaaclab_tasks  # noqa: F401
from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData

from franka_manipulation_sim.episode_tracker import EpisodeThresholds, EpisodeTracker
from franka_manipulation_sim.pick_place_sm import PickPlaceStateMachine
from franka_manipulation_sim.result_writer import write_results
from franka_manipulation_sim.state_machine_termination import attach_state_machine_state
from franka_manipulation_sim.tasks import FrankaManipulationEnvCfg, configure_randomization


def _git_commit() -> str | None:

    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_scene_state(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    raw_env = env.unwrapped
    ee_frame = raw_env.scene["ee_frame"]
    ee_pos = ee_frame.data.target_pos_w.torch[..., 0, :].clone() - raw_env.scene.env_origins
    ee_quat = ee_frame.data.target_quat_w.torch[..., 0, :].clone()
    object_data: RigidObjectData = raw_env.scene["object"].data
    object_pos = object_data.root_pos_w.torch - raw_env.scene.env_origins
    target_pos = raw_env.command_manager.get_command("object_pose")[..., :3].clone()
    return torch.cat([ee_pos, ee_quat], dim=-1), object_pos.clone(), target_pos


def main() -> int:

    wp.init()
    run_started = datetime.now(timezone.utc)
    run_id = run_started.strftime(f"gate{args_cli.gate}_%Y%m%dT%H%M%SZ")

    target_z = args_cli.target_z if args_cli.target_z is not None else (0.35 if args_cli.gate == 1 else 0.021)
    env_cfg = FrankaManipulationEnvCfg()

    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.episode_length_s = 5.0 if args_cli.gate == 1 else 8.0
    configure_randomization(
        env_cfg,
        args_cli.object_xy_range,
        args_cli.target_xy_range,
        target_z,
        args_cli.mass_scale_range,
    )

    env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
    env.reset(seed=args_cli.seed)
    raw_env = env.unwrapped
    state_machine = PickPlaceStateMachine(
        dt=env_cfg.sim.dt * env_cfg.decimation,
        num_envs=raw_env.num_envs,
        device=raw_env.device,
        stage_timeout=args_cli.stage_timeout,
        gate=args_cli.gate,
    )

    attach_state_machine_state(raw_env, state_machine.sm_state)
    tracker = EpisodeTracker(raw_env.num_envs, raw_env.device, EpisodeThresholds())
    ros2_publisher = None
    if args_cli.ros2_publish:
        from franka_manipulation_sim.ros2 import Ros2StatePublisher

        ros2_publisher = Ros2StatePublisher(raw_env, args_cli.ros2_publish_every_n_steps)
        print("[ROS2] Publishing state", flush=True)

    ee_pose, object_pos, target_pos = _read_scene_state(env)
    object_mass = raw_env.scene["object"].data.body_mass.torch[:, 0].clone()
    all_env_ids = torch.arange(raw_env.num_envs, device=raw_env.device, dtype=torch.long)
    state_machine.reset_idx(all_env_ids, object_pos[:, 2])
    tracker.start(all_env_ids, object_pos, target_pos, object_mass)

    desired_orientation = torch.zeros((raw_env.num_envs, 4), device=raw_env.device)
    desired_orientation[:, 1] = 1.0
    rows: list[dict] = []

    while simulation_app.is_running() and len(rows) < args_cli.num_episodes:
        step_started = time.perf_counter()
        with torch.inference_mode():
            lift_pos = tracker.initial_object_pos.clone()
            lift_pos[:, 2] += 0.25
            lift_pose = torch.cat([lift_pos, desired_orientation], dim=-1)
            actions = state_machine.compute(
                ee_pose,
                torch.cat([object_pos, desired_orientation], dim=-1),
                lift_pose,
                torch.cat([target_pos, desired_orientation], dim=-1),
            )

            tracker.update(object_pos, state_machine.sm_state)
            _, _, terminated, truncated, _ = env.step(actions)
            if ros2_publisher is not None:
                ros2_publisher.publish()
            done = torch.logical_or(terminated, truncated)
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)

            ee_pose, object_pos, target_pos = _read_scene_state(env)
            if len(done_ids) > 0:
                completed_rows = tracker.finalize(
                    done_ids,
                    state_machine.sm_state,
                    state_machine.failure_code,
                    terminated,
                    truncated,
                    run_id,
                    args_cli.seed,
                    raw_env.step_dt,
                    args_cli.object_xy_range,
                    args_cli.target_xy_range,
                    args_cli.mass_scale_range,
                    args_cli.gate,
                )
                remaining = args_cli.num_episodes - len(rows)
                accepted_rows = completed_rows[:remaining]
                for offset, row in enumerate(accepted_rows):

                    row["episode_id"] = len(rows) + offset
                rows.extend(accepted_rows)
                for row in accepted_rows:
                    print(
                        "[EPISODE] "
                        f"episode={row['episode_id']} env={row['env_id']} success={row['overall_success']} "
                        f"lift_delta={row['lift_delta']:.4f} failure={row['failure_stage']}",
                        flush=True,
                    )
                state_machine.reset_idx(done_ids, object_pos[:, 2])
                object_mass = raw_env.scene["object"].data.body_mass.torch[:, 0].clone()
                tracker.start(done_ids, object_pos, target_pos, object_mass)
        if args_cli.ros2_realtime:
            remaining_step_time = raw_env.step_dt - (time.perf_counter() - step_started)
            if remaining_step_time > 0.0:
                time.sleep(remaining_step_time)

    metadata = {
        "gate": "gate1_pick_and_lift" if args_cli.gate == 1 else "gate2_pick_and_place",
        "run_id": run_id,
        "started_at_utc": run_started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "requested_episodes": args_cli.num_episodes,
        "device": str(raw_env.device),
        "isaac_lab_version": isaaclab.__version__,
        "isaac_sim_version": importlib.metadata.version("isaacsim"),
        "git_commit": _git_commit(),
        "randomization": {
            "object_xy_range": args_cli.object_xy_range,
            "target_xy_range": args_cli.target_xy_range,
            "target_z": target_z,
            "mass_scale_range": args_cli.mass_scale_range,
        },
    }
    csv_path, json_path, summary = write_results(rows, PROJECT_ROOT / args_cli.output_dir, metadata)
    print(f"[RESULT] episodes_csv={csv_path}", flush=True)
    print(f"[RESULT] summary_json={json_path}", flush=True)
    print(
        f"[SUMMARY] total={summary['total_episodes']} "
        f"grasp_rate={summary['grasp_success_rate']:.3f} "
        f"lift_rate={summary['lift_success_rate']:.3f} "
        f"transport_rate={summary['transport_success_rate']:.3f} "
        f"place_rate={summary['place_success_rate']:.3f} "
        f"overall_rate={summary['overall_success_rate']:.3f}",
        flush=True,
    )
    if ros2_publisher is not None:
        ros2_publisher.close()
    env.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
