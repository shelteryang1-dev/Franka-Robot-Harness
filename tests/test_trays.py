"""Test trays."""
import pytest
from franka_manipulation_sim.harness.trays import inside_tray

@pytest.mark.parametrize("position,velocity,expected", [
    ((.5, .2, .029), (0, 0, 0), True),
    ((.58, .2, .029), (0, 0, 0), False),
    ((.5, .2, .1), (0, 0, 0), False),
    ((.5, .2, .029), (.1, 0, 0), False),
    ((float('nan'), .2, .029), (0, 0, 0), False)])
def test_tray(position, velocity, expected):
    assert inside_tray(position, velocity, (0, 0, 0), "red_tray") == expected
