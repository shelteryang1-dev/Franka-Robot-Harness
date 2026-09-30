"""Skill-plan validation."""

import math
from .trays import TRAYS

REGIONS = {"left": (0.5, 0.16, 0.021), "right": (0.5, -0.16, 0.021),
           "center": (0.5, 0.0, 0.021)}
REGIONS.update({name: tray["center"] for name, tray in TRAYS.items()})
SKILLS = {"pick_place", "place", "lift", "ppo_lift", "open_gripper", "home"}


def validate_plan(value, objects=("cube",)):
    """Validate skills, objects, targets, and holding state."""
    if not isinstance(value, dict) or set(value) != {"steps"}:
        raise ValueError("Plan must contain only steps")
    steps = value["steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= 20:
        raise ValueError("Plan length must be 1-20")
    clean = []
    held_object = None
    for step in steps:
        if not isinstance(step, dict) or set(step) - {"skill", "object", "target"}:
            raise ValueError("Unknown step field")
        skill = step.get("skill")
        if skill not in SKILLS:
            raise ValueError("Unknown skill")
        if skill not in ("open_gripper", "home") and step.get("object") not in objects:
            raise ValueError("Unknown object")
        if skill in ("pick_place", "place") and step.get("target") not in REGIONS:
            raise ValueError("Unknown target")
        if skill not in ("pick_place", "place") and "target" in step:
            raise ValueError("Target not supported for skill")
        if skill == "place" and step.get("object") != held_object:
            raise ValueError("place requires preceding lift of same object")
        clean.append(dict(step))
        if skill in ("lift", "ppo_lift"):
            held_object = step.get("object")
        elif skill in ("place", "pick_place", "open_gripper", "home"):
            held_object = None
    return {"steps": clean}


def position_reached(actual, target, tolerance=0.05):
    return (len(actual) == len(target) == 3 and all(math.isfinite(x) for x in (*actual, *target))
            and math.dist(actual, target) <= tolerance)
