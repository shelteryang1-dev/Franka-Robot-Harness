"""Registered six-object benchmark cases."""

from .trays import SIX_OBJECTS, tray_slots

NAMES = tuple(SIX_OBJECTS)
SEEDS = (42, 43, 44)


def initial_positions(layout="table", seed=42, jitter=.005):
    import random
    if layout not in ("table", "red_tray", "blue_tray"):
        raise ValueError("Unknown initial layout")
    if not 0 <= jitter <= .01:
        raise ValueError("Jitter must be in [0, 0.01] m")
    nominal = SIX_OBJECTS if layout == "table" else dict(zip(NAMES, tray_slots(layout, large=True)))
    rng = random.Random(seed)
    return {name: (p[0]+rng.uniform(-jitter, jitter), p[1]+rng.uniform(-jitter, jitter), p[2])
            for name, p in nominal.items()}


def cases():
    same = [(n, "red_tray" if n.startswith("red") else "blue_tray") for n in NAMES]
    opposite = [(n, "blue_tray" if n.startswith("red") else "red_tray") for n in NAMES]
    red = [(n, "red_tray") for n in NAMES]
    blue = [(n, "blue_tray") for n in NAMES]
    definitions = [
        ("same_color", "table", same),
        ("opposite_color", "table", opposite),
        ("all_red", "table", red),
        ("all_blue", "table", blue),
        ("full_red_to_blue", "red_tray", blue),
        ("full_blue_to_red", "blue_tray", red),
        ("repeat_object", "table", [("red_A", "red_tray"), ("red_A", "blue_tray"),
            ("red_A", "red_tray"), ("blue_B", "blue_tray"), ("blue_B", "red_tray")]),
        ("specified_order", "table", [("red_C", "blue_tray"), ("blue_C", "red_tray"),
            ("red_A", "blue_tray"), ("blue_A", "red_tray")]),
        ("mixed_assignment", "table", [("red_A", "red_tray"), ("blue_A", "blue_tray"),
            ("red_B", "blue_tray"), ("blue_B", "red_tray"),
            ("red_C", "red_tray"), ("blue_C", "blue_tray")]),
        ("move_then_restore", "table", [("red_A", "blue_tray"), ("blue_B", "red_tray"),
            ("red_A", "red_tray"), ("blue_B", "blue_tray")]),
    ]
    result = []
    for index, (kind, layout, pairs) in enumerate(definitions):
        clauses = [f"把{n[-1]}{'红' if n.startswith('red') else '蓝'}色物块放在"
                   f"{'红' if tray.startswith('red') else '蓝'}色托盘里" for n, tray in pairs]
        instruction = "先"+"，然后".join(clauses)+"，最后回位"
        expected = {"steps": [{"skill": "pick_place", "object": n, "target": t}
                              for n, t in pairs]+[{"skill": "home"}]}
        for seed in SEEDS:
            result.append({"id": f"class_{index:02d}_seed_{seed}", "class": kind,
                "scene_seed": seed, "initial_layout": layout, "position_jitter": .005,
                "instruction": instruction, "expected_plan": expected})
    return result
