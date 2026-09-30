"""Test portfolio suite."""

from franka_manipulation_sim.harness.contracts import validate_plan
from franka_manipulation_sim.harness.language import parse_instruction
from franka_manipulation_sim.harness.portfolio_suite import cases, initial_positions, NAMES
from franka_manipulation_sim.harness.trays import inside_tray


def test_thirty_registered_cases():
    suite = cases()
    assert len(suite) == len({c["id"] for c in suite}) == 30
    for c in suite:
        assert validate_plan(c["expected_plan"], NAMES) == c["expected_plan"]
        scene = {"objects": initial_positions(c["initial_layout"], c["scene_seed"]),
                 "targets": ["red_tray", "blue_tray"]}
        assert parse_instruction(c["instruction"], scene) == c["expected_plan"]


def test_layout_deterministic_and_distinct():
    assert initial_positions(seed=42) == initial_positions(seed=42)
    assert initial_positions(seed=42) != initial_positions(seed=43)
    for tray in ("red_tray", "blue_tray"):
        for seed in (42, 43, 44):
            positions = initial_positions(tray, seed)
            assert all(inside_tray(p, (0, 0, 0), (0, 0, 0), tray, True) for p in positions.values())
            import math
            assert min(math.dist(a, b) for i, a in enumerate(positions.values())
                       for b in list(positions.values())[i+1:]) > .07
