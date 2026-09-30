"""Restricted Chinese instruction parser."""
import re
from .contracts import validate_plan


def parse_instruction(text, world):
    text = re.sub(r"\s+", "", text)
    clauses = re.split(r"[，,；;。]|然后|接着", text)
    steps, previous = [], None
    for clause in clauses:
        clause = re.sub(r"^(?:请|先|再|最后|接着)+", "", clause)
        if not clause:
            continue
        if clause in ("回位", "回到初始姿态", "回到初始位置"):
            steps.append({"skill": "home"})
            continue
        match = re.fullmatch(r"(?:把|将)?(?P<object>(?:[ABC](?:红|蓝)色?(?:物块|方块))|(?:red|blue)_[ABC]|它)"
                             r"(?:放入|放到|放在|移动到)(?P<color>红|蓝)色?托盘(?:里|中|内)?", clause)
        if not match:
            raise ValueError(f"Unsupported clause: {clause}")
        token = match["object"]
        if token == "它":
            if previous is None:
                raise ValueError("Object reference unresolved")
            obj = previous
        elif "_" in token:
            obj = token
        else:
            obj = ("red" if "红" in token else "blue")+"_"+token[0]
        previous = obj
        steps.append({"skill": "pick_place", "object": obj,
                      "target": "red_tray" if match["color"] == "红" else "blue_tray"})
    return validate_plan({"steps": steps}, tuple(world["objects"]))
