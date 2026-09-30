#!/usr/bin/env python3
"""Run the persistent manipulation scene."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from franka_manipulation_sim.ros2.runtime import prepare_runtime
prepare_runtime()
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--duration", type=float, default=600)
parser.add_argument("--output_dir", default="results/harness/live")
parser.add_argument("--multi_object", action="store_true", help="Add a blue cube")
parser.add_argument("--sort_trays", action="store_true", help="Add sorting trays")
parser.add_argument("--six_objects", action="store_true", help="Use six objects and two trays")
parser.add_argument("--scene_seed", type=int, default=42)
parser.add_argument("--initial_layout", choices=("table", "red_tray", "blue_tray"), default="table")
parser.add_argument("--trace_every", type=int, default=5, help="Trace interval (steps); 0 disables")
parser.add_argument("--stop_file", type=Path, help="Stop marker path")
parser.add_argument("--position_jitter", type=float, default=0., help="Position jitter (m), max 0.01")
parser.add_argument("--task_timeout", type=float, default=120., help="Task timeout (s)")
parser.add_argument("--record_video", help="Task video path")
parser.add_argument("--video_frames", type=int, default=2400)
parser.add_argument("--video_spp", type=int, default=1, help="Path tracing samples per frame")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.six_objects:
    args.sort_trays = True
if args.record_video:
    args.kit_args = (args.kit_args or "") + " --/persistent/rtx/modes/pt/enabled=true --/rtx/rendermode=PathTracing"
args.kit_args = (args.kit_args or "") + " --enable isaacsim.ros2.bridge --/exts/isaacsim.ros2.bridge/ros_distro=jazzy --/exts/isaacsim.ros2.bridge/publish_without_verification=true"
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks
import rclpy
from std_msgs.msg import String, Float64MultiArray, Bool
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from franka_manipulation_sim.tasks.env_cfg import FrankaManipulationEnvCfg, configure_randomization
from franka_manipulation_sim.ros2.state_publisher import Ros2StatePublisher
from franka_manipulation_sim.ros2.ros2_control_adapter import Ros2ControlAdapter
from franka_manipulation_sim.harness.runtime import TaskRuntime
from franka_manipulation_sim.harness.sim_skills import SimSkills


def main():
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = FrankaManipulationEnvCfg()
    cfg.seed = args.scene_seed
    cfg.scene.num_envs, cfg.sim.device = 1, args.device
    # Keep scene state between skills.
    for name in cfg.terminations.to_dict():
        setattr(cfg.terminations, name, None)
    if args.record_video:
        from isaaclab_visualizers.kit import KitVisualizerCfg
        cfg.sim.visualizer_cfgs = [KitVisualizerCfg(eye=(1.55, 1.25, .95), lookat=(.48, 0., .18),
            headless=True, max_visible_envs=1, window_width=1280, window_height=720,
            randomly_sample_visible_envs=False, enable_live_plots=False)]
        cfg.viewer.resolution = (1280, 720)
        cfg.video_recorder.window_width, cfg.video_recorder.window_height = 1280, 720
    ik_cfg = cfg.actions.arm_action.copy()
    names = [f"panda_joint{i}" for i in range(1, 8)]
    cfg.actions.arm_action = JointPositionActionCfg(asset_name="robot", joint_names=names,
                                                   scale=1., use_default_offset=False, preserve_order=True)
    configure_randomization(cfg, 0, 0, .2, 0)
    if args.multi_object or args.sort_trays:
        import isaaclab.sim as sim_utils
        cfg.scene.blue_object = cfg.scene.object.copy()
        cfg.scene.blue_object.prim_path = "{ENV_REGEX_NS}/BlueObject"
        cfg.scene.blue_object.init_state.pos = (.6, .10, .055)
        cfg.scene.blue_object.spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(.05, .15, .9))
        cfg.scene.object.spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(.9, .1, .05))
    if args.sort_trays:
        from franka_manipulation_sim.harness.trays import add_trays
        add_trays(cfg, large=args.six_objects)
        cfg.scene.object.init_state.pos = (.42, -.04, .055)
        cfg.scene.blue_object.init_state.pos = (.62, .04, .055)
        cfg.commands.object_pose.debug_vis = False
        cfg.scene.ee_frame.debug_vis = False
    if args.six_objects:
        from franka_manipulation_sim.harness.portfolio_suite import initial_positions
        cfg.scene.blue_object = None
        template = cfg.scene.object.copy()
        for name, position in initial_positions(args.initial_layout, args.scene_seed, args.position_jitter).items():
            obj = template.copy()
            # Keep the asset name expected by the reset event.
            obj.prim_path = "{ENV_REGEX_NS}/"+("Object" if name == "red_A" else name)
            obj.init_state.pos = position
            shade = {"A": 1., "B": .75, "C": .5}[name[-1]]
            color = (.9*shade,.06,.04) if name.startswith("red") else (.04,.12,.9*shade)
            obj.spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
            setattr(cfg.scene, "object" if name == "red_A" else name, obj)
    env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=cfg,
                   render_mode="rgb_array" if args.record_video else None)
    env.reset()
    if args.record_video:
        import carb
        settings = carb.settings.get_settings()
        settings.set("/rtx/rendermode", "PathTracing")

        settings.set("/rtx/pathtracing/spp", max(1, args.video_spp))
        settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
        settings.set("/rtx/pathtracing/optixDenoiser/blendFactor", 0.0)
    writer, frames = None, 0
    if args.record_video:
        import imageio.v2 as imageio
        path = ROOT/args.record_video
        path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(16):
            env.render()
        writer = imageio.get_writer(str(path), fps=50, codec="libx264", macro_block_size=1,
                                   ffmpeg_params=["-crf", "18", "-movflags", "+faststart"])

        writer.append_data(env.render())
    raw, robot = env.unwrapped, env.unwrapped.scene["robot"]
    skills = SimSkills(raw, ik_cfg, ROOT/"checkpoints/ppo_lift/model_1499.pt")
    skills.trays_enabled = args.sort_trays

    from franka_manipulation_sim.harness.ppo_skill import LiftPolicy
    skills.policy = LiftPolicy(skills.checkpoint, raw)
    with torch.inference_mode():
        skills.policy.reset(skills.object_position())
        for _ in range(5):
            skills.policy.compute(raw.scene["object"].data.root_pos_w.torch)
        skills.policy.reset(skills.object_position())
        torch.cuda.synchronize(raw.device)
    limits = robot.data.joint_pos_limits.torch[0].tolist()
    contract = {"source": cfg.scene.robot.spawn.usd_path, "joints": {name: {
        "lower": limits[i][0], "upper": limits[i][1],
        "velocity": float(robot.data.joint_vel_limits.torch[0, i]),
        "effort": float(robot.data.joint_effort_limits.torch[0, i])}
        for i, name in enumerate(robot.joint_names)}}
    (output/"asset_joint_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    adapter = Ros2ControlAdapter(names, [limits[i] for i in skills.ids], skills.target[0].tolist())
    publisher = Ros2StatePublisher(raw, joint_topic="/isaac_joint_states_raw", publish_clock=True)
    node = rclpy.create_node("franka_harness")
    events = node.create_publisher(String, "/harness/events", 20)
    world = node.create_publisher(String, "/harness/state", 10)
    commands = node.create_publisher(Float64MultiArray, "/skill_controller/commands", 1)
    gripper = node.create_publisher(Bool, "/gripper_command", 1)
    trajectories = node.create_publisher(String, "/harness/trajectory_request", 10)
    skills.send_trajectory = lambda value: trajectories.publish(String(data=json.dumps(value)))
    def trajectory_result(message):
        value = json.loads(message.data)
        skills.trajectory_results[value["id"]] = value
        if not runtime.active:
            skills.trajectory_id = None
    node.create_subscription(String, "/harness/trajectory_result", trajectory_result, 10)
    control_seen = [0.]
    node.create_subscription(String, "/harness/control_status",
                             lambda message: control_seen.__setitem__(0, time.monotonic()), 10)
    journal = (output/"events.jsonl").open("a", encoding="utf-8")
    trace = (output/"physics_trace.jsonl").open("a", encoding="utf-8")
    video_index = (output/"video_index.jsonl").open("a", encoding="utf-8") if writer else None
    (output/"run_config.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()}, ensure_ascii=False, indent=2), encoding="utf-8")

    def emit(event):
        event["sim_step"] = int(raw.common_step_counter)
        text = json.dumps(event, ensure_ascii=False)
        journal.write(text+"\n")
        journal.flush()
        events.publish(String(data=text))
        print(text, flush=True)

    runtime = TaskRuntime(skills, emit, timeout=args.task_timeout)

    def receive(message):
        task_id = None
        try:
            if len(message.data) > 32000:
                raise ValueError("Task request too long")
            request = json.loads(message.data)
            task_id = request.get("task_id")
            if request.get("cancel"):
                runtime.cancel(task_id)
            else:
                if (adapter.last_command is None or adapter.fault or time.monotonic()-control_seen[0] > 1
                        or skills.trajectory_id):
                    raise ValueError("Control stack unavailable")
                runtime.submit(task_id, request["plan"])
        except Exception as error:
            emit({"task_id": task_id, "status": "rejected", "detail": str(error)})

    node.create_subscription(String, "/harness/command", receive, 10)
    action = torch.zeros((1, 8), device=raw.device)
    start = time.monotonic()
    print("[HARNESS_READY]", flush=True)
    try:
        with torch.inference_mode():
            while (app.is_running() and time.monotonic()-start < args.duration
                   and not (args.stop_file and args.stop_file.exists())):
                tick = time.monotonic()
                rclpy.spin_once(node, timeout_sec=0.)
                if adapter.fault:
                    runtime.finish("failed", adapter.fault)
                if runtime.active and time.monotonic()-control_seen[0] > 1:
                    runtime.finish("failed", "Controller heartbeat timeout")
                q, grip = skills.compute()
                commands.publish(Float64MultiArray(data=q))
                gripper.publish(Bool(data=grip < 0))
                # Apply only targets returned by the hardware bridge.
                actual_target, actual_grip = adapter.step(skills.current_joints()[0].tolist())
                action[0, :7] = torch.tensor(actual_target, device=raw.device)
                action[0, 7] = actual_grip
                _, _, terminated, truncated, _ = env.step(action)
                if writer and runtime.active and frames < args.video_frames:
                    writer.append_data(env.render())
                    video_index.write(json.dumps({"frame": frames+1, "sim_step": int(raw.common_step_counter),
                        "task_id": runtime.active, "wall_time": time.time()})+"\n")
                    video_index.flush()
                    frames += 1
                if args.trace_every > 0 and int(raw.common_step_counter) % args.trace_every == 0:
                    frame = raw.scene["ee_frame"].data
                    sample = {"sim_step": int(raw.common_step_counter), "sim_time": raw.common_step_counter*raw.step_dt,
                        "wall_time": time.time(), "task_id": runtime.active,
                        "sm_state": int(skills.sm.sm_state[0]) if skills.sm is not None else None,
                        "selected_object": skills.object_key, "joint_names": robot.joint_names,
                        "joint_positions": robot.data.joint_pos.torch[0].tolist(),
                        "joint_velocities": robot.data.joint_vel.torch[0].tolist(),
                        "requested_arm_target": q, "hardware_arm_target": actual_target,
                        "ee_position_world": frame.target_pos_w.torch[0, 0].tolist(),
                        "ee_quaternion_wxyz": frame.target_quat_w.torch[0, 0].tolist(),
                        "objects": {name: {"position": (raw.scene[key].data.root_pos_w.torch[0]-raw.scene.env_origins[0]).tolist(),
                            "quaternion_wxyz": raw.scene[key].data.root_quat_w.torch[0].tolist(),
                            "linear_velocity": raw.scene[key].data.root_lin_vel_w.torch[0].tolist(),
                            "angular_velocity": raw.scene[key].data.root_ang_vel_w.torch[0].tolist()}
                            for name, key in skills.objects.items()}}
                    trace.write(json.dumps(sample)+"\n")
                    trace.flush()
                if any(float(raw.scene[key].data.root_pos_w.torch[0, 2]) < -.02 for key in skills.objects.values()):
                    adapter.fault = "Object fell from table"
                if bool(terminated.any() or truncated.any()):
                    raise RuntimeError("Environment terminated")
                publisher.publish()
                runtime.tick()
                elapsed = time.monotonic()-tick
                if elapsed > .5:
                    print(f"[SLOW_STEP] elapsed={elapsed:.3f}s", flush=True)
                world.publish(String(data=json.dumps({"objects": {name: (raw.scene[key].data.root_pos_w.torch[0]-raw.scene.env_origins[0]).tolist()
                    for name, key in skills.objects.items()},
                    "targets": (["red_tray", "blue_tray"] if args.sort_trays else ["left", "right", "center"]),
                    "active_task": runtime.active, "control_ready": adapter.last_command is not None and not adapter.fault
                    and time.monotonic()-control_seen[0] < 1 and not skills.trajectory_id,
                    "fault": adapter.fault, "step": int(raw.common_step_counter)})))
                time.sleep(max(0., raw.step_dt-(time.monotonic()-tick)))
    finally:
        runtime.finish("cancelled", "Simulation stopped")
        journal.close()
        trace.close()
        if video_index:
            video_index.close()
        if writer:
            writer.close()
        node.destroy_node()
        adapter.close()
        publisher.close()
        env.close()


if __name__ == "__main__":
    code = 1
    try:
        main()
        code = 0
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        app.close(exit_code=code)
