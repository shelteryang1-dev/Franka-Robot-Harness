#!/usr/bin/env python3
"""Evaluate a frozen PPO lift checkpoint."""

import argparse
import importlib.metadata
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "source"
sys.path.insert(0, str(SOURCE_ROOT))


os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate PPO lift")
parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
parser.add_argument("--num_envs", type=int, default=64, help="Environment count")
parser.add_argument("--num_episodes", type=int, default=100, help="Total episodes")
parser.add_argument("--seed", type=int, default=43, help="Evaluation seed")
parser.add_argument("--output_dir", default="results/ppo/evaluation_latest", help="Result directory")
parser.add_argument("--grasp_delta", type=float, default=0.01, help="Grasp height threshold (m)")
parser.add_argument("--lift_delta", type=float, default=0.05, help="Lift height threshold (m)")
parser.add_argument("--goal_tolerance", type=float, default=0.05, help="Goal tolerance (m)")
parser.add_argument("--reach_tolerance", type=float, default=0.08, help="Reach tolerance (m)")
parser.add_argument("--show", action="store_true", help="Enable visualization")
parser.add_argument("--realtime", action="store_true", help="Run at simulation rate")
parser.add_argument("--video", action="store_true", help="Record policy video")
parser.add_argument("--video_length", type=int, default=250, help="Video length (steps)")
parser.add_argument("--video_dir", default="media/ppo_lift_demo", help="Video directory")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs < 1 or args_cli.num_episodes < 1:
    parser.error("Environment and episode counts must be >= 1")
if min(args_cli.grasp_delta, args_cli.lift_delta, args_cli.goal_tolerance, args_cli.reach_tolerance) <= 0.0:
    parser.error("Thresholds must be > 0")
if args_cli.grasp_delta >= args_cli.lift_delta:
    parser.error("Grasp threshold must be below lift threshold")
if args_cli.video_length < 1:
    parser.error("Video length must be >= 1")

if args_cli.video:
    args_cli.enable_cameras = True


if not args_cli.show:
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab
import isaaclab_tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_visualizers.kit import KitVisualizerCfg

from franka_manipulation_sim.result_writer import write_results
from franka_manipulation_sim.rl.agent_cfg import FrankaLiftPPORunnerCfg
from franka_manipulation_sim.rl.env_cfg import FrankaLiftRLEnvCfg_PLAY
from franka_manipulation_sim.rl.registration import PLAY_TASK_ID, register_rl_tasks


def _read_scene_state(raw_env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    origins = raw_env.scene.env_origins
    object_pos = raw_env.scene["object"].data.root_pos_w.torch - origins
    ee_pos = raw_env.scene["ee_frame"].data.target_pos_w.torch[..., 0, :] - origins

    target_pos = raw_env.command_manager.get_command("object_pose")[..., :3]
    object_mass = raw_env.scene["object"].data.body_mass.torch[:, 0]
    return object_pos.clone(), ee_pos.clone(), target_pos.clone(), object_mass.clone()


def _make_row(
    *,
    env_id: int,
    episode_id: int,
    run_id: str,
    checkpoint: Path,
    initial_object_pos: torch.Tensor,
    target_pos: torch.Tensor,
    last_object_pos: torch.Tensor,
    object_mass: torch.Tensor,
    max_object_z: torch.Tensor,
    min_ee_distance: torch.Tensor,
    min_goal_distance: torch.Tensor,
    steps: torch.Tensor,
    first_lift_step: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    step_dt: float,
) -> dict:

    initial = initial_object_pos[env_id].detach().cpu()
    target = target_pos[env_id].detach().cpu()
    final = last_object_pos[env_id].detach().cpu()
    lift_amount = float(max_object_z[env_id].item() - initial_object_pos[env_id, 2].item())
    reached = float(min_ee_distance[env_id].item()) <= args_cli.reach_tolerance
    grasped = lift_amount >= args_cli.grasp_delta
    lifted = lift_amount >= args_cli.lift_delta
    goal_reached = float(min_goal_distance[env_id].item()) <= args_cli.goal_tolerance
    completion_step = int(first_lift_step[env_id].item())

    if lifted:
        failure_stage = "none"
    elif bool(terminated[env_id].item()):
        failure_stage = "object_dropped"
    elif not reached:
        failure_stage = "reach_failed"
    elif not grasped:
        failure_stage = "grasp_failed"
    else:
        failure_stage = "lift_failed"

    return {
        "run_id": run_id,
        "controller": "ppo",
        "checkpoint": str(checkpoint),
        "episode_id": episode_id,
        "env_id": env_id,
        "seed": args_cli.seed,
        "object_initial_x": float(initial[0].item()),
        "object_initial_y": float(initial[1].item()),
        "object_initial_z": float(initial[2].item()),
        "target_x": float(target[0].item()),
        "target_y": float(target[1].item()),
        "target_z": float(target[2].item()),
        "object_xy_range": 0.25,
        "target_xy_range": 0.25,
        "mass_scale_range": 0.0,
        "object_mass_kg": float(object_mass[env_id].item()),
        "max_object_z": float(max_object_z[env_id].item()),
        "lift_delta": lift_amount,
        "min_ee_object_distance": float(min_ee_distance[env_id].item()),
        "min_goal_distance": float(min_goal_distance[env_id].item()),
        "goal_success": goal_reached,
        "min_transport_xy_error": 0.0,
        "final_object_x": float(final[0].item()),
        "final_object_y": float(final[1].item()),
        "final_object_z": float(final[2].item()),
        "final_position_error": float(torch.linalg.vector_norm(final - target).item()),
        "steps": int(steps[env_id].item()),
        "simulated_time_s": float(steps[env_id].item() * step_dt),
        "terminal_step": completion_step,
        "completed_before_timeout": completion_step > 0,
        "environment_terminated": bool(terminated[env_id].item()),
        "environment_truncated": bool(truncated[env_id].item()),
        "final_state": "RL_LIFT_SUCCESS" if lifted else "RL_FAILURE",
        "grasp_success": grasped,
        "lift_success": lifted,
        "transport_success": False,
        "place_success": False,
        "overall_success": lifted,
        "failure_stage": failure_stage,
    }


def main() -> int:

    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")

    register_rl_tasks()
    agent_cfg = FrankaLiftPPORunnerCfg()
    installed_rsl_rl = importlib.metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl)
    evaluation_device = args_cli.device or "cuda:0"
    agent_cfg.device = evaluation_device

    env_cfg = FrankaLiftRLEnvCfg_PLAY()
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = evaluation_device
    if args_cli.show or args_cli.video:

        env_cfg.viewer.eye = (1.55, 1.25, 0.95)
        env_cfg.viewer.lookat = (0.48, 0.0, 0.18)
    if args_cli.video:


        env_cfg.sim.render.antialiasing_mode = "FXAA"
        env_cfg.sim.render.rendering_mode = "performance"
        env_cfg.sim.render.enable_reflections = False
        env_cfg.sim.render.enable_global_illumination = False
        env_cfg.sim.render.enable_shadows = False
        env_cfg.sim.render.enable_ambient_occlusion = False

        env_cfg.sim.visualizer_cfgs = [
            KitVisualizerCfg(
                eye=(1.55, 1.25, 0.95),
                lookat=(0.48, 0.0, 0.18),
                headless=True,
                max_visible_envs=1,
                randomly_sample_visible_envs=False,
                enable_live_plots=False,
            )
        ]

    gym_env = gym.make(PLAY_TASK_ID, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_dir = (PROJECT_ROOT / args_cli.video_dir).resolve()
        video_dir.mkdir(parents=True, exist_ok=True)
        gym_env = gym.wrappers.RecordVideo(
            gym_env,
            video_folder=str(video_dir),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
        print(f"[VIDEO] Recording: {video_dir}", flush=True)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=raw_env.device)
    print(f"[PPO] Loaded: {checkpoint}", flush=True)

    obs = env.get_observations()
    object_pos, ee_pos, target_pos, object_mass = _read_scene_state(raw_env)
    initial_object_pos = object_pos.clone()
    initial_target_pos = target_pos.clone()
    last_object_pos = object_pos.clone()
    max_object_z = object_pos[:, 2].clone()
    min_ee_distance = torch.full((raw_env.num_envs,), float("inf"), device=raw_env.device)
    min_goal_distance = torch.full((raw_env.num_envs,), float("inf"), device=raw_env.device)
    steps = torch.zeros(raw_env.num_envs, dtype=torch.int64, device=raw_env.device)
    first_lift_step = torch.full((raw_env.num_envs,), -1, dtype=torch.int64, device=raw_env.device)

    run_started = datetime.now(timezone.utc)
    run_id = run_started.strftime("ppo_lift_eval_%Y%m%dT%H%M%SZ")
    rows: list[dict] = []

    while simulation_app.is_running() and len(rows) < args_cli.num_episodes:
        step_started = time.perf_counter()
        with torch.inference_mode():

            object_pos, ee_pos, target_pos, _ = _read_scene_state(raw_env)
            last_object_pos.copy_(object_pos)
            max_object_z = torch.maximum(max_object_z, object_pos[:, 2])
            min_ee_distance = torch.minimum(
                min_ee_distance,
                torch.linalg.vector_norm(ee_pos - object_pos, dim=-1),
            )
            min_goal_distance = torch.minimum(
                min_goal_distance,
                torch.linalg.vector_norm(object_pos - initial_target_pos, dim=-1),
            )
            steps += 1
            lifted_now = max_object_z >= initial_object_pos[:, 2] + args_cli.lift_delta
            newly_lifted = lifted_now & (first_lift_step < 0)
            first_lift_step = torch.where(newly_lifted, steps, first_lift_step)

            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            done_mask = dones.to(dtype=torch.bool)
            truncated = extras.get("time_outs", torch.zeros_like(done_mask)).to(dtype=torch.bool)
            terminated = done_mask & ~truncated
            done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)

            if hasattr(policy, "reset"):
                policy.reset(dones)

            if len(done_ids) > 0:
                for env_id in done_ids.detach().cpu().tolist():
                    if len(rows) >= args_cli.num_episodes:
                        break
                    row = _make_row(
                        env_id=env_id,
                        episode_id=len(rows),
                        run_id=run_id,
                        checkpoint=checkpoint,
                        initial_object_pos=initial_object_pos,
                        target_pos=initial_target_pos,
                        last_object_pos=last_object_pos,
                        object_mass=object_mass,
                        max_object_z=max_object_z,
                        min_ee_distance=min_ee_distance,
                        min_goal_distance=min_goal_distance,
                        steps=steps,
                        first_lift_step=first_lift_step,
                        terminated=terminated,
                        truncated=truncated,
                        step_dt=raw_env.step_dt,
                    )
                    rows.append(row)
                    print(
                        f"[EPISODE] episode={row['episode_id']} env={env_id} "
                        f"lift={row['lift_success']} goal={row['goal_success']} "
                        f"completion_step={row['terminal_step']} failure={row['failure_stage']}",
                        flush=True,
                    )

                # Step has already reset completed environments.
                reset_object_pos, reset_ee_pos, reset_target_pos, reset_mass = _read_scene_state(raw_env)
                initial_object_pos[done_ids] = reset_object_pos[done_ids]
                initial_target_pos[done_ids] = reset_target_pos[done_ids]
                last_object_pos[done_ids] = reset_object_pos[done_ids]
                object_mass[done_ids] = reset_mass[done_ids]
                max_object_z[done_ids] = reset_object_pos[done_ids, 2]
                min_ee_distance[done_ids] = torch.linalg.vector_norm(
                    reset_ee_pos[done_ids] - reset_object_pos[done_ids], dim=-1
                )
                min_goal_distance[done_ids] = torch.linalg.vector_norm(
                    reset_object_pos[done_ids] - reset_target_pos[done_ids], dim=-1
                )
                steps[done_ids] = 0
                first_lift_step[done_ids] = -1

        if args_cli.realtime:
            remaining = raw_env.step_dt - (time.perf_counter() - step_started)
            if remaining > 0.0:
                time.sleep(remaining)

    metadata = {
        "gate": "gate_rl1_ppo_lift",
        "controller": "ppo",
        "run_id": run_id,
        "started_at_utc": run_started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args_cli.seed,
        "training_seed": 42,
        "num_envs": args_cli.num_envs,
        "requested_episodes": args_cli.num_episodes,
        "device": str(raw_env.device),
        "checkpoint": str(checkpoint),
        "isaac_lab_version": isaaclab.__version__,
        "isaac_sim_version": importlib.metadata.version("isaacsim"),
        "rsl_rl_version": installed_rsl_rl,
        "observation_dimension": 36,
        "action_dimension": 8,
        "thresholds": {
            "reach_tolerance_m": args_cli.reach_tolerance,
            "grasp_delta_m": args_cli.grasp_delta,
            "lift_delta_m": args_cli.lift_delta,
            "goal_tolerance_m": args_cli.goal_tolerance,
        },
        "randomization": {
            "object_x_offset_m": [-0.1, 0.1],
            "object_y_offset_m": [-0.25, 0.25],
            "target_x_m": [0.4, 0.6],
            "target_y_m": [-0.25, 0.25],
            "target_z_m": [0.25, 0.5],
            "observation_corruption": False,
        },
    }
    csv_path, json_path, summary = write_results(rows, PROJECT_ROOT / args_cli.output_dir, metadata)
    print(f"[RESULT] episodes_csv={csv_path}", flush=True)
    print(f"[RESULT] summary_json={json_path}", flush=True)
    print(
        f"[SUMMARY] total={summary['total_episodes']} "
        f"grasp_rate={summary['grasp_success_rate']:.3f} "
        f"lift_rate={summary['lift_success_rate']:.3f} "
        f"goal_rate={summary['goal_success_rate']:.3f} "
        f"overall_rate={summary['overall_success_rate']:.3f} "
        f"mean_completion_step={summary['mean_terminal_step']:.1f}",
        flush=True,
    )
    env.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
