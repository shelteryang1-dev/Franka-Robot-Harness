"""Episode metrics."""

from collections import Counter
from statistics import mean, median


def summarize_rows(rows: list[dict]) -> dict:

    total = len(rows)
    if total == 0:
        return {
            "total_episodes": 0,
            "grasp_success_rate": 0.0,
            "lift_success_rate": 0.0,
            "transport_success_rate": 0.0,
            "place_success_rate": 0.0,
            "goal_success_rate": 0.0,
            "overall_success_rate": 0.0,
            "mean_episode_steps": 0.0,
            "median_episode_steps": 0.0,
            "mean_terminal_step": 0.0,
            "median_terminal_step": 0.0,
            "mean_final_position_error": 0.0,
            "failure_breakdown": {},
        }

    def rate(field: str) -> float:
        return sum(bool(row[field]) for row in rows) / total

    failures = Counter(row["failure_stage"] for row in rows if row["failure_stage"] != "none")
    terminal_steps = [row["terminal_step"] for row in rows if row["terminal_step"] >= 0]
    return {
        "total_episodes": total,
        "grasp_success_rate": rate("grasp_success"),
        "lift_success_rate": rate("lift_success"),
        "transport_success_rate": rate("transport_success"),
        "place_success_rate": rate("place_success"),
        "goal_success_rate": sum(bool(row.get("goal_success", False)) for row in rows) / total,
        "overall_success_rate": rate("overall_success"),
        "mean_episode_steps": mean(row["steps"] for row in rows),
        "median_episode_steps": median(row["steps"] for row in rows),
        "mean_terminal_step": mean(terminal_steps) if terminal_steps else 0.0,
        "median_terminal_step": median(terminal_steps) if terminal_steps else 0.0,
        "mean_final_position_error": mean(row["final_position_error"] for row in rows),
        "failure_breakdown": dict(sorted(failures.items())),
    }
