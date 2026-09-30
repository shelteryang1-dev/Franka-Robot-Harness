"""Test ros2 control adapter."""
import pytest
from franka_manipulation_sim.ros2.ros2_control_adapter import validate_joint_command


def test_reorder():
    assert validate_joint_command(["b", "a"], [0.2, 0.1], ["a", "b"], [(-1, 1)]*2) == [0.1, 0.2]


@pytest.mark.parametrize("names,positions", [(["a", "a"], [0, 0]), (["a"], [0]),
                                          (["a", "b"], [float("nan"), 0]),
                                          (["a", "b"], [2, 0]), (["a", "b"], [0])])
def test_reject_invalid(names, positions):
    with pytest.raises(ValueError):
        validate_joint_command(names, positions, ["a", "b"], [(-1, 1)]*2)
