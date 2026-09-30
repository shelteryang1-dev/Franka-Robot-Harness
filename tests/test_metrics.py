"""Test metrics."""

from franka_manipulation_sim.metrics import summarize_rows


def test_summary_rates_and_failures():
    rows = [
        {
            "grasp_success": True,
            "lift_success": True,
            "transport_success": False,
            "place_success": False,
            "overall_success": True,
            "goal_success": True,
            "steps": 200,
            "terminal_step": 150,
            "final_position_error": 0.02,
            "failure_stage": "none",
        },
        {
            "grasp_success": False,
            "lift_success": False,
            "transport_success": False,
            "place_success": False,
            "overall_success": False,
            "goal_success": False,
            "steps": 300,
            "terminal_step": 200,
            "final_position_error": 0.30,
            "failure_stage": "grasp_failed",
        },
    ]
    summary = summarize_rows(rows)
    assert summary["total_episodes"] == 2
    assert summary["grasp_success_rate"] == 0.5
    assert summary["lift_success_rate"] == 0.5
    assert summary["overall_success_rate"] == 0.5
    assert summary["goal_success_rate"] == 0.5
    assert summary["mean_episode_steps"] == 250
    assert summary["median_episode_steps"] == 250.0
    assert summary["median_terminal_step"] == 175.0
    assert summary["failure_breakdown"] == {"grasp_failed": 1}


def test_summary_uses_middle_value_for_odd_episode_count():

    base_row = {
        "grasp_success": True,
        "lift_success": True,
        "transport_success": True,
        "place_success": True,
        "overall_success": True,
        "terminal_step": 10,
        "final_position_error": 0.01,
        "failure_stage": "none",
    }
    rows = [{**base_row, "steps": steps} for steps in (30, 10, 20)]
    summary = summarize_rows(rows)
    assert summary["mean_episode_steps"] == 20
    assert summary["median_episode_steps"] == 20
    assert summary["median_terminal_step"] == 10


def test_empty_summary_has_zero_median_episode_steps():

    summary = summarize_rows([])
    assert summary["median_episode_steps"] == 0.0
    assert summary["median_terminal_step"] == 0.0
    assert summary["goal_success_rate"] == 0.0
