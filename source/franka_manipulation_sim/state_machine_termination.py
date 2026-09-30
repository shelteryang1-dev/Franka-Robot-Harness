"""State-machine termination hook."""

import torch

from .pick_place_sm import SM_FAILURE_VALUE, SM_SUCCESS_VALUE


_STATE_ATTRIBUTE = "_franka_manipulation_state_machine_state"


def attach_state_machine_state(env, sm_state: torch.Tensor) -> None:

    if sm_state.ndim != 1 or sm_state.shape[0] != env.num_envs:
        raise ValueError(
            "State shape mismatch: "
            f"expected={env.num_envs}, shape={tuple(sm_state.shape)}."
        )
    if sm_state.device != torch.device(env.device):
        raise ValueError(
            "State device mismatch: "
            f"env={env.device}, state={sm_state.device}."
        )
    setattr(env, _STATE_ATTRIBUTE, sm_state)


def state_machine_finished(env) -> torch.Tensor:
    """Terminate on state-machine success or failure."""

    sm_state = getattr(env, _STATE_ATTRIBUTE, None)
    if sm_state is None:

        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return torch.logical_or(sm_state == SM_SUCCESS_VALUE, sm_state == SM_FAILURE_VALUE)
