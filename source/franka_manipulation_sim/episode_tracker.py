"""Per-environment episode tracking."""

from dataclasses import dataclass

import torch

from .pick_place_sm import FAILURE_NAMES, SM_FAILURE_VALUE, SM_SUCCESS_VALUE, STATE_NAMES


@dataclass(frozen=True)
class EpisodeThresholds:

    grasp_height: float = 0.01
    lift_height: float = 0.05
    transport_xy_error: float = 0.06
    place_position_error: float = 0.05


class EpisodeTracker:
    """Track finite episodes independently per environment."""

    def __init__(self, num_envs: int, device: torch.device | str, thresholds: EpisodeThresholds):
        self.num_envs = int(num_envs)
        self.device = device
        self.thresholds = thresholds
        self.episode_index = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        self.initial_object_pos = torch.zeros((self.num_envs, 3), device=device)
        self.target_pos = torch.zeros((self.num_envs, 3), device=device)
        self.object_mass = torch.zeros(self.num_envs, device=device)
        self.last_object_pos = torch.zeros((self.num_envs, 3), device=device)
        self.max_object_z = torch.full((self.num_envs,), float("-inf"), device=device)
        self.min_transport_xy_error = torch.full((self.num_envs,), float("inf"), device=device)
        self.steps = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        self.terminal_step = torch.full((self.num_envs,), -1, dtype=torch.int64, device=device)

    def start(
        self,
        env_ids: torch.Tensor,
        object_pos: torch.Tensor,
        target_pos: torch.Tensor,
        object_mass: torch.Tensor,
    ) -> None:

        self.initial_object_pos[env_ids] = object_pos[env_ids]
        self.target_pos[env_ids] = target_pos[env_ids]
        self.object_mass[env_ids] = object_mass[env_ids]
        self.last_object_pos[env_ids] = object_pos[env_ids]
        self.max_object_z[env_ids] = object_pos[env_ids, 2]
        self.min_transport_xy_error[env_ids] = float("inf")
        self.steps[env_ids] = 0
        self.terminal_step[env_ids] = -1

    def update(self, object_pos: torch.Tensor, sm_state: torch.Tensor) -> None:

        self.last_object_pos.copy_(object_pos)
        self.max_object_z = torch.maximum(self.max_object_z, object_pos[:, 2])
        horizontal_error = torch.linalg.vector_norm(object_pos[:, :2] - self.target_pos[:, :2], dim=-1)
        lifted = object_pos[:, 2] >= self.initial_object_pos[:, 2] + self.thresholds.lift_height
        self.min_transport_xy_error = torch.where(
            lifted,
            torch.minimum(self.min_transport_xy_error, horizontal_error),
            self.min_transport_xy_error,
        )
        self.steps += 1
        just_finished = (self.terminal_step < 0) & (
            (sm_state == SM_SUCCESS_VALUE) | (sm_state == SM_FAILURE_VALUE)
        )
        self.terminal_step = torch.where(just_finished, self.steps, self.terminal_step)

    def finalize(
        self,
        env_ids: torch.Tensor,
        sm_state: torch.Tensor,
        failure_code: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        run_id: str,
        seed: int,
        step_dt: float,
        object_xy_range: float,
        target_xy_range: float,
        mass_scale_range: float,
        gate: int,
    ) -> list[dict]:

        rows: list[dict] = []
        for env_id in env_ids.detach().cpu().tolist():
            initial = self.initial_object_pos[env_id].detach().cpu()
            target = self.target_pos[env_id].detach().cpu()
            final = self.last_object_pos[env_id].detach().cpu()
            max_z = float(self.max_object_z[env_id].item())
            lift_delta = max_z - float(initial[2].item())
            grasp_success = lift_delta >= self.thresholds.grasp_height
            lift_success = lift_delta >= self.thresholds.lift_height
            final_error = float(torch.linalg.vector_norm(final - target).item())
            transport_success = float(self.min_transport_xy_error[env_id].item()) <= self.thresholds.transport_xy_error
            state_value = int(sm_state[env_id].item())
            failure_value = int(failure_code[env_id].item())
            place_success = final_error <= self.thresholds.place_position_error and state_value == SM_SUCCESS_VALUE
            completed_before_timeout = int(self.terminal_step[env_id].item()) > 0
            if gate == 1 and lift_success and completed_before_timeout:
                failure_stage = "none"
            elif gate == 2 and grasp_success and lift_success and transport_success and place_success:
                failure_stage = "none"
            elif failure_value != 0:
                failure_stage = FAILURE_NAMES.get(failure_value, "simulation_error")
            elif state_value == SM_FAILURE_VALUE:
                failure_stage = "simulation_error"
            elif state_value <= 1:
                failure_stage = "approach_above_timeout"
            elif state_value == 2:
                failure_stage = "approach_timeout"
            elif state_value == 3:
                failure_stage = "grasp_failed"
            elif state_value == 4:
                failure_stage = "lift_failed"
            else:
                failure_stage = "episode_timeout"

            rows.append(
                {
                    "run_id": run_id,
                    "episode_id": int(self.episode_index[env_id].item()),
                    "env_id": int(env_id),
                    "seed": int(seed),
                    "object_initial_x": float(initial[0].item()),
                    "object_initial_y": float(initial[1].item()),
                    "object_initial_z": float(initial[2].item()),
                    "target_x": float(target[0].item()),
                    "target_y": float(target[1].item()),
                    "target_z": float(target[2].item()),
                    "object_xy_range": float(object_xy_range),
                    "target_xy_range": float(target_xy_range),
                    "mass_scale_range": float(mass_scale_range),
                    "object_mass_kg": float(self.object_mass[env_id].item()),
                    "max_object_z": max_z,
                    "lift_delta": lift_delta,
                    "min_transport_xy_error": float(self.min_transport_xy_error[env_id].item()),
                    "final_object_x": float(final[0].item()),
                    "final_object_y": float(final[1].item()),
                    "final_object_z": float(final[2].item()),
                    "final_position_error": final_error,
                    "steps": int(self.steps[env_id].item()),
                    "simulated_time_s": float(self.steps[env_id].item() * step_dt),
                    "terminal_step": int(self.terminal_step[env_id].item()),
                    "completed_before_timeout": bool(completed_before_timeout),
                    "environment_terminated": bool(terminated[env_id].item()),
                    "environment_truncated": bool(truncated[env_id].item()),
                    "final_state": STATE_NAMES.get(state_value, "UNKNOWN"),
                    "grasp_success": bool(grasp_success),
                    "lift_success": bool(lift_success),
                    "transport_success": bool(transport_success) if gate == 2 else False,
                    "place_success": bool(place_success) if gate == 2 else False,
                    "overall_success": bool(lift_success and completed_before_timeout) if gate == 1 else bool(
                        grasp_success and lift_success and transport_success and place_success and completed_before_timeout
                    ),
                    "failure_stage": failure_stage,
                }
            )
            self.episode_index[env_id] += 1
        return rows
