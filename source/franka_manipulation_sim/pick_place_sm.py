# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# Copyright (c) 2026, Franka Manipulation Simulation contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Differential-IK pick-and-place state machine."""

from collections.abc import Sequence

import torch
import warp as wp


SM_SUCCESS_VALUE = 9
SM_FAILURE_VALUE = 10


class SmState:

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)
    MOVE_ABOVE_TARGET = wp.constant(5)
    LOWER_OBJECT = wp.constant(6)
    RELEASE_OBJECT = wp.constant(7)
    RETREAT = wp.constant(8)
    SUCCESS = wp.constant(SM_SUCCESS_VALUE)
    FAILURE = wp.constant(SM_FAILURE_VALUE)


class FailureCode:

    NONE = wp.constant(0)
    APPROACH_ABOVE_TIMEOUT = wp.constant(1)
    APPROACH_TIMEOUT = wp.constant(2)
    GRASP_FAILED = wp.constant(3)
    LIFT_FAILED = wp.constant(4)
    OBJECT_DROPPED = wp.constant(5)
    TRANSPORT_FAILED = wp.constant(6)
    GOAL_NOT_REACHED = wp.constant(7)


STATE_NAMES = {
    0: "REST",
    1: "APPROACH_ABOVE_OBJECT",
    2: "APPROACH_OBJECT",
    3: "GRASP_OBJECT",
    4: "LIFT_OBJECT",
    5: "MOVE_ABOVE_TARGET",
    6: "LOWER_OBJECT",
    7: "RELEASE_OBJECT",
    8: "RETREAT",
    9: "SUCCESS",
    10: "FAILURE",
}

FAILURE_NAMES = {
    0: "none",
    1: "approach_above_timeout",
    2: "approach_timeout",
    3: "grasp_failed",
    4: "lift_failed",
    5: "object_dropped",
    6: "transport_failed",
    7: "goal_not_reached",
}


@wp.func
def _position_close(current_pose: wp.transform, desired_pose: wp.transform, threshold: float) -> bool:

    current_pos = wp.transform_get_translation(current_pose)
    desired_pos = wp.transform_get_translation(desired_pose)
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def _infer_gate1_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    failure_code: wp.array(dtype=int),
    initial_object_z: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    lift_pose: wp.array(dtype=wp.transform),
    desired_object_pose: wp.array(dtype=wp.transform),
    desired_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    approach_offset: wp.array(dtype=wp.transform),
    move_offset: wp.array(dtype=wp.transform),
    release_offset: wp.array(dtype=wp.transform),
    gate: int,
    position_threshold: float,
    lift_height: float,
    goal_threshold: float,
    stage_timeout: float,
):

    env_id = wp.tid()
    state = sm_state[env_id]

    if state == SmState.REST:
        desired_ee_pose[env_id] = ee_pose[env_id]
        gripper_state[env_id] = 1.0
        if sm_wait_time[env_id] >= 0.2:
            sm_state[env_id] = SmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[env_id] = 0.0

    elif state == SmState.APPROACH_ABOVE_OBJECT:
        desired_ee_pose[env_id] = wp.transform_multiply(approach_offset[env_id], object_pose[env_id])
        gripper_state[env_id] = 1.0
        if _position_close(ee_pose[env_id], desired_ee_pose[env_id], position_threshold):
            sm_state[env_id] = SmState.APPROACH_OBJECT
            sm_wait_time[env_id] = 0.0
        elif sm_wait_time[env_id] >= stage_timeout:
            sm_state[env_id] = SmState.FAILURE
            failure_code[env_id] = FailureCode.APPROACH_ABOVE_TIMEOUT

    elif state == SmState.APPROACH_OBJECT:
        desired_ee_pose[env_id] = object_pose[env_id]
        gripper_state[env_id] = 1.0
        if _position_close(ee_pose[env_id], desired_ee_pose[env_id], position_threshold):
            sm_state[env_id] = SmState.GRASP_OBJECT
            sm_wait_time[env_id] = 0.0
        elif sm_wait_time[env_id] >= stage_timeout:
            sm_state[env_id] = SmState.FAILURE
            failure_code[env_id] = FailureCode.APPROACH_TIMEOUT

    elif state == SmState.GRASP_OBJECT:
        desired_ee_pose[env_id] = object_pose[env_id]
        gripper_state[env_id] = -1.0
        if sm_wait_time[env_id] >= 0.3:
            sm_state[env_id] = SmState.LIFT_OBJECT
            sm_wait_time[env_id] = 0.0

    elif state == SmState.LIFT_OBJECT:
        desired_ee_pose[env_id] = lift_pose[env_id]
        gripper_state[env_id] = -1.0
        object_z = wp.transform_get_translation(object_pose[env_id])[2]
        if object_z >= initial_object_z[env_id] + lift_height:
            if gate == 1:
                sm_state[env_id] = SmState.SUCCESS
            else:
                sm_state[env_id] = SmState.MOVE_ABOVE_TARGET
            sm_wait_time[env_id] = 0.0
        elif sm_wait_time[env_id] >= stage_timeout:
            sm_state[env_id] = SmState.FAILURE
            failure_code[env_id] = FailureCode.LIFT_FAILED

    elif state == SmState.MOVE_ABOVE_TARGET:
        desired_ee_pose[env_id] = wp.transform_multiply(move_offset[env_id], desired_object_pose[env_id])
        gripper_state[env_id] = -1.0
        object_pos = wp.transform_get_translation(object_pose[env_id])
        target_pos = wp.transform_get_translation(desired_object_pose[env_id])
        horizontal_error = wp.length(wp.vec3(object_pos[0] - target_pos[0], object_pos[1] - target_pos[1], 0.0))
        if object_pos[2] <= initial_object_z[env_id] + 0.015 and horizontal_error > goal_threshold:
            sm_state[env_id] = SmState.FAILURE
            failure_code[env_id] = FailureCode.OBJECT_DROPPED
        elif _position_close(ee_pose[env_id], desired_ee_pose[env_id], position_threshold):
            sm_state[env_id] = SmState.LOWER_OBJECT
            sm_wait_time[env_id] = 0.0
        elif sm_wait_time[env_id] >= stage_timeout:
            sm_state[env_id] = SmState.FAILURE
            failure_code[env_id] = FailureCode.TRANSPORT_FAILED

    elif state == SmState.LOWER_OBJECT:
        desired_ee_pose[env_id] = wp.transform_multiply(release_offset[env_id], desired_object_pose[env_id])
        gripper_state[env_id] = -1.0
        if _position_close(ee_pose[env_id], desired_ee_pose[env_id], position_threshold):
            sm_state[env_id] = SmState.RELEASE_OBJECT
            sm_wait_time[env_id] = 0.0
        elif sm_wait_time[env_id] >= stage_timeout:
            sm_state[env_id] = SmState.FAILURE
            failure_code[env_id] = FailureCode.GOAL_NOT_REACHED

    elif state == SmState.RELEASE_OBJECT:
        desired_ee_pose[env_id] = wp.transform_multiply(release_offset[env_id], desired_object_pose[env_id])
        gripper_state[env_id] = 1.0
        if sm_wait_time[env_id] >= 0.5:
            sm_state[env_id] = SmState.RETREAT
            sm_wait_time[env_id] = 0.0

    elif state == SmState.RETREAT:
        desired_ee_pose[env_id] = wp.transform_multiply(move_offset[env_id], desired_object_pose[env_id])
        gripper_state[env_id] = 1.0
        object_pos = wp.transform_get_translation(object_pose[env_id])
        target_pos = wp.transform_get_translation(desired_object_pose[env_id])
        if _position_close(object_pose[env_id], desired_object_pose[env_id], goal_threshold) and sm_wait_time[env_id] >= 0.5:
            sm_state[env_id] = SmState.SUCCESS
            sm_wait_time[env_id] = 0.0
        elif sm_wait_time[env_id] >= stage_timeout:
            sm_state[env_id] = SmState.FAILURE
            failure_code[env_id] = FailureCode.GOAL_NOT_REACHED

    elif state == SmState.SUCCESS:
        desired_ee_pose[env_id] = wp.transform_multiply(move_offset[env_id], desired_object_pose[env_id])
        gripper_state[env_id] = 1.0 if gate == 2 else -1.0

    else:
        desired_ee_pose[env_id] = ee_pose[env_id]
        gripper_state[env_id] = 1.0

    sm_wait_time[env_id] = sm_wait_time[env_id] + dt[env_id]


class PickPlaceStateMachine:

    def __init__(
        self,
        dt: float,
        num_envs: int,
        device: torch.device | str,
        position_threshold: float = 0.01,
        lift_height: float = 0.05,
        goal_threshold: float = 0.05,
        stage_timeout: float = 3.0,
        gate: int = 1,
    ):
        self.num_envs = int(num_envs)
        self.device = device
        self.position_threshold = float(position_threshold)
        self.lift_height = float(lift_height)
        self.goal_threshold = float(goal_threshold)
        self.stage_timeout = float(stage_timeout)
        self.gate = int(gate)

        self.sm_dt = torch.full((self.num_envs,), float(dt), device=device)
        self.sm_state = torch.zeros((self.num_envs,), dtype=torch.int32, device=device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=device)
        self.failure_code = torch.zeros((self.num_envs,), dtype=torch.int32, device=device)
        self.initial_object_z = torch.zeros((self.num_envs,), device=device)
        self.desired_ee_pose = torch.zeros((self.num_envs, 7), device=device)
        self.desired_gripper_state = torch.ones((self.num_envs,), device=device)

        self.approach_offset = torch.zeros((self.num_envs, 7), device=device)
        self.approach_offset[:, 2] = 0.1
        self.approach_offset[:, -1] = 1.0
        self.move_offset = self.approach_offset.clone()
        self.move_offset[:, 2] = 0.15
        self.release_offset = self.approach_offset.clone()
        self.release_offset[:, 2] = 0.045

        self._sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self._sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self._sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self._failure_code_wp = wp.from_torch(self.failure_code, wp.int32)
        self._initial_object_z_wp = wp.from_torch(self.initial_object_z, wp.float32)
        self._desired_ee_pose_wp = wp.from_torch(self.desired_ee_pose, wp.transform)
        self._desired_gripper_state_wp = wp.from_torch(self.desired_gripper_state, wp.float32)
        self._approach_offset_wp = wp.from_torch(self.approach_offset, wp.transform)
        self._move_offset_wp = wp.from_torch(self.move_offset, wp.transform)
        self._release_offset_wp = wp.from_torch(self.release_offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None, initial_object_z: torch.Tensor) -> None:

        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0
        self.failure_code[env_ids] = 0
        self.initial_object_z[env_ids] = initial_object_z[env_ids]

    def compute(
        self,
        ee_pose: torch.Tensor,
        object_pose: torch.Tensor,
        lift_pose: torch.Tensor,
        desired_object_pose: torch.Tensor,
    ) -> torch.Tensor:

        wp.launch(
            kernel=_infer_gate1_state_machine,
            dim=self.num_envs,
            inputs=[
                self._sm_dt_wp,
                self._sm_state_wp,
                self._sm_wait_time_wp,
                self._failure_code_wp,
                self._initial_object_z_wp,
                wp.from_torch(ee_pose.contiguous(), wp.transform),
                wp.from_torch(object_pose.contiguous(), wp.transform),
                wp.from_torch(lift_pose.contiguous(), wp.transform),
                wp.from_torch(desired_object_pose.contiguous(), wp.transform),
                self._desired_ee_pose_wp,
                self._desired_gripper_state_wp,
                self._approach_offset_wp,
                self._move_offset_wp,
                self._release_offset_wp,
                self.gate,
                self.position_threshold,
                self.lift_height,
                self.goal_threshold,
                self.stage_timeout,
            ],
            device=self.device,
        )
        return torch.cat([self.desired_ee_pose, self.desired_gripper_state.unsqueeze(-1)], dim=-1)
