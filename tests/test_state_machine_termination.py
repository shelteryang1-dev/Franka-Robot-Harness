"""Test state machine termination."""

from types import SimpleNamespace

import pytest
import torch

from franka_manipulation_sim.state_machine_termination import attach_state_machine_state, state_machine_finished


def _fake_env(num_envs: int = 4):

    return SimpleNamespace(num_envs=num_envs, device="cpu")


def test_unbound_state_machine_never_terminates_environment():

    result = state_machine_finished(_fake_env())
    assert result.tolist() == [False, False, False, False]


def test_success_and_failure_both_terminate_environment():

    env = _fake_env()
    sm_state = torch.tensor([0, 9, 10, 4], dtype=torch.int32)
    attach_state_machine_state(env, sm_state)
    assert state_machine_finished(env).tolist() == [False, True, True, False]


def test_binding_rejects_wrong_number_of_environments():

    env = _fake_env(num_envs=4)
    with pytest.raises(ValueError, match="State shape mismatch"):
        attach_state_machine_state(env, torch.zeros(3, dtype=torch.int32))
