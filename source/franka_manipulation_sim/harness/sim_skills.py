"""Simulation skill execution."""

import time
import uuid
import torch
from ..pick_place_sm import PickPlaceStateMachine, FAILURE_NAMES
from .contracts import REGIONS, position_reached
from .trays import TRAYS, SIX_OBJECTS, MIN_OBJECT_CLEARANCE, inside_tray, free_slot


class SimSkills:
    """Execute skills against the current simulation state."""

    object_names = ("cube",)

    def __init__(self, raw, ik_cfg, checkpoint=None):
        self.raw, self.robot = raw, raw.scene["robot"]
        self.ids = [self.robot.joint_names.index(f"panda_joint{i}") for i in range(1, 8)]
        self.ik = ik_cfg.class_type(ik_cfg, raw)
        self.sm = None
        self.gripper = 1.0
        self.target = self.current_joints()
        self.orientation = torch.tensor([[0., 1., 0., 0.]], device=raw.device)
        self.started = 0.
        self.policy = None
        self.policy_active = False
        self.checkpoint = checkpoint
        self.object_key = "object"
        self.objects = {"cube": "object"}
        if "blue_object" in raw.scene.rigid_objects:
            self.objects["blue_cube"] = "blue_object"
        self.six_objects = "red_B" in raw.scene.rigid_objects
        if self.six_objects:
            self.objects = {name: ("object" if name == "red_A" else name) for name in SIX_OBJECTS}
        self.object_names = tuple(self.objects)
        self.trajectory_id = None
        self.pending_pick = None
        self.trajectory_results = {}
        self.trays_enabled = False

    def current_joints(self):
        return self.robot.data.joint_pos.torch[:, self.ids].clone()

    def object_position(self):
        return self.raw.scene[self.object_key].data.root_pos_w.torch.clone()-self.raw.scene.env_origins

    def start(self, step, _prepared=False):
        self.step, self.started = step, time.monotonic()
        self.skipped = False
        self.stable_steps, self.settle_steps = 0, 0
        target = step.get("target")
        if target in TRAYS:
            if not self.trays_enabled:
                raise ValueError("Trays unavailable")
        self.object_key = self.objects.get(step.get("object"), self.object_key)
        self.initial = self.object_position()
        self.policy_active = False
        if self.six_objects and step["skill"] == "pick_place" and not _prepared:
            # Return home before regrasping to limit posture drift.

            if any(float(self.raw.scene[key].data.root_pos_w.torch[0, 2]) > .065 for key in self.objects.values()):
                raise ValueError("Object airborne; grasp blocked")
            self.pending_pick, self.sm, self.gripper = step, None, 1.0
            self.trajectory_id = str(uuid.uuid4())
            self.send_trajectory({"id": self.trajectory_id, "name": "home"})
            return
        if step["skill"] == "home":
            if any(float(self.raw.scene[key].data.root_pos_w.torch[0, 2]) > .065 for key in self.objects.values()):
                raise ValueError("Object airborne; home blocked")
            self.sm = None
            self.trajectory_id = str(uuid.uuid4())
            self.send_trajectory({"id": self.trajectory_id, "name": "home"})
            return
        if step["skill"] == "ppo_lift":
            if self.initial[0, 2].item() > .06:
                raise ValueError("PPO lift requires object on table")
            if self.policy is None:
                from .ppo_skill import LiftPolicy
                self.policy = LiftPolicy(self.checkpoint, self.raw)
            self.policy.reset(self.initial)
            self.policy_active, self.sm = True, None
            return
        if step["skill"] == "open_gripper":
            self.sm, self.gripper = None, 1.0
            return
        if step["skill"] == "place":
            if self.initial[0, 2].item() < .06 or self.gripper >= 0:
                raise ValueError("Place requires held object")
            self.initial[:, 2] = .021
        elif self.initial[0, 2].item() > 0.06:
            raise ValueError("Grasp requires stable object")
        if target in TRAYS and step["skill"] == "pick_place":
            data = self.raw.scene[self.object_key].data
            if inside_tray(self.initial[0].tolist(), data.root_lin_vel_w.torch[0].tolist(),
                           data.root_ang_vel_w.torch[0].tolist(), target, self.six_objects):
                self.sm, self.skipped = None, True
                return
        self.sm = PickPlaceStateMachine(self.raw.step_dt, 1, self.raw.device,
                                       gate=2 if step["skill"] in ("pick_place", "place") else 1,
                                       stage_timeout=6.0)
        self.sm.reset_idx(None, self.initial[:, 2])
        if step["skill"] == "place":
            self.sm.sm_state[:] = 5
        self.destination = torch.tensor([REGIONS.get(step.get("target"), REGIONS["center"])],
                                        device=self.raw.device)
        if target in TRAYS:
            positions = {name: (self.raw.scene[key].data.root_pos_w.torch[0]-self.raw.scene.env_origins[0]).tolist()
                         for name, key in self.objects.items()}
            self.destination[:] = torch.tensor([free_slot(target, positions, step["object"], self.six_objects)],
                                               device=self.raw.device)
        if step["skill"] in ("pick_place", "place"):
            for name, key in self.objects.items():
                if key != self.object_key:
                    other = self.raw.scene[key].data.root_pos_w.torch[0]-self.raw.scene.env_origins[0]
                    if torch.linalg.vector_norm(other-self.destination[0]).item() < MIN_OBJECT_CLEARANCE:
                        raise ValueError(f"Target occupied by {name}.")
            if step["skill"] == "pick_place" and position_reached(self.initial[0].tolist(), self.destination[0].tolist(), .03):
                self.sm, self.skipped = None, True

    def compute(self):
        if self.trajectory_id:
            self.target = self.current_joints()
            return self.target[0].tolist(), self.gripper
        if self.policy_active:
            self.target, self.gripper = self.policy.compute(self.raw.scene[self.object_key].data.root_pos_w.torch)
            return self.target[0].tolist(), self.gripper
        if self.sm is None:
            return self.target[0].tolist(), self.gripper
        frame = self.raw.scene["ee_frame"].data
        ee_pos = frame.target_pos_w.torch[:, 0]-self.raw.scene.env_origins
        ee_quat = frame.target_quat_w.torch[:, 0]
        obj = self.object_position()
        lift = self.initial.clone()
        lift[:, 2] += 0.25
        action = self.sm.compute(torch.cat([ee_pos, ee_quat], -1),
                                 torch.cat([obj, self.orientation], -1),
                                 torch.cat([lift, self.orientation], -1),
                                 torch.cat([self.destination, self.orientation], -1))
        self.ik.process_actions(action[:, :7])
        # Compute targets here; PhysX advances in the main loop.
        self.ik.apply_actions()
        self.target = self.robot.data.joint_pos_target.torch[:, self.ids].clone()
        self.gripper = float(action[0, 7])
        return self.target[0].tolist(), self.gripper

    def poll(self):
        tray = self.step.get("target")
        if tray in TRAYS and (self.skipped or (self.sm is not None and int(self.sm.sm_state[0]) == 9)):
            data = self.raw.scene[self.object_key].data
            actual = self.object_position()[0].tolist()
            fingers = self.robot.data.joint_pos.torch[0, -2:]
            inside = inside_tray(actual, data.root_lin_vel_w.torch[0].tolist(),
                                 data.root_ang_vel_w.torch[0].tolist(), tray, self.six_objects)
            self.stable_steps = self.stable_steps+1 if inside and bool((fingers > .03).all()) else 0
            self.settle_steps += 1
            detail = {"tray": tray, "object_position": actual, "inside": inside,
                      "object": self.step.get("object"), "skipped": self.skipped,
                      "stable_seconds": self.stable_steps*self.raw.step_dt}
            if detail["stable_seconds"] >= .5:
                return "succeeded", detail
            if self.settle_steps*self.raw.step_dt > 3:
                return "failed", detail
            return "running", None
        if self.skipped:
            return "succeeded", {"skipped": True, "reason": "Object already at target"}
        if self.trajectory_id:
            result = self.trajectory_results.get(self.trajectory_id)
            if result:
                self.trajectory_id = None
                if self.pending_pick is not None:
                    step, self.pending_pick = self.pending_pick, None
                    if result["success"]:
                        self.start(step, _prepared=True)
                        return "running", None
                return ("succeeded" if result["success"] else "failed"), result
            if time.monotonic()-self.started > 95:
                return "failed", "Trajectory result timeout"
            return "running", None
        if self.policy_active:
            delta = float(self.object_position()[0, 2]-self.initial[0, 2])
            if delta >= .05:
                return "succeeded", {"lift_delta": delta, "controller": "ppo"}
            if time.monotonic()-self.started > 15:
                return "failed", "PPO lift timeout"
            return "running", None
        if self.step["skill"] == "open_gripper":
            fingers = self.robot.data.joint_pos.torch[0, -2:]
            if bool((fingers > 0.035).all()):
                return "succeeded", {"finger_positions": fingers.tolist()}
            if time.monotonic()-self.started > 5:
                return "failed", "Gripper open timeout"
            return "running", None
        state = int(self.sm.sm_state[0])
        if state == 10:
            print({"skill_failure": FAILURE_NAMES[int(self.sm.failure_code[0])],
                   "object": self.step.get("object"), "actual": self.object_position()[0].tolist(),
                   "target": self.destination[0].tolist(), "joints": self.current_joints()[0].tolist()}, flush=True)
            return "failed", FAILURE_NAMES[int(self.sm.failure_code[0])]
        if state == 9:
            actual = self.object_position()[0].tolist()
            success = (position_reached(actual, self.destination[0].tolist())
                       if self.step["skill"] in ("pick_place", "place") else actual[2]-float(self.initial[0, 2]) >= 0.05)
            return ("succeeded" if success else "failed"), {"object_position": actual,
                                                                          "target": self.destination[0].tolist()}
        return "running", None

    def hold(self):
        self.pending_pick = None
        if self.trajectory_id:
            self.send_trajectory({"id": self.trajectory_id, "cancel": True})

        self.sm = None
        self.policy_active = False
        self.target = self.current_joints()

    def verify_plan(self, steps):
        """Check each object's final requested target."""
        latest = {}
        for step in steps:
            if step.get("object"):
                latest[step["object"]] = step.get("target")
        checks = {}
        for name, target in latest.items():
            if target not in TRAYS:
                continue
            data = self.raw.scene[self.objects[name]].data
            pos = (data.root_pos_w.torch[0]-self.raw.scene.env_origins[0]).tolist()
            checks[name] = {"target": target, "position": pos,
                "passed": inside_tray(pos, data.root_lin_vel_w.torch[0].tolist(),
                    data.root_ang_vel_w.torch[0].tolist(), target, self.six_objects)}
        return all(c["passed"] for c in checks.values()), checks

    def can_retry(self, reason):
        """Retry grasp or lift failures only for low, stationary objects."""
        if not isinstance(reason, str) or reason not in ("grasp_failed", "lift_failed"):
            return False
        data = self.raw.scene[self.object_key].data
        return (float(self.object_position()[0, 2]) < .04
                and torch.linalg.vector_norm(data.root_lin_vel_w.torch[0]).item() < .02)
