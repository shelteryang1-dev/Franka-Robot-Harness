"""Test six objects."""
import pytest
from franka_manipulation_sim.harness.language import parse_instruction
from franka_manipulation_sim.harness.trays import SIX_OBJECTS, tray_slots, free_slot, inside_tray

WORLD = {"objects": SIX_OBJECTS}

def test_order_and_cross_color():
    p = parse_instruction("先把A红色物块放在蓝色托盘里，然后把B蓝色物块放入红色托盘，再把它放到蓝色托盘中，最后回位", WORLD)
    assert [s.get("object") for s in p["steps"]] == ["red_A","blue_B","blue_B",None]
    assert [s.get("target") for s in p["steps"]] == ["blue_tray","red_tray","blue_tray",None]

@pytest.mark.parametrize("text", ["把D红色物块放入蓝色托盘", "把它放入红色托盘", "把A红色物块放入红色托盘，然后跳舞", "把红色物块放入红色托盘"])
def test_reject_ambiguity(text):
    with pytest.raises(ValueError):
        parse_instruction(text, WORLD)

def test_six_slots():
    slots = tray_slots("red_tray", True)
    assert len(slots) == 6
    assert all(inside_tray(p,(0,0,0),(0,0,0),"red_tray",True) for p in slots)
    with pytest.raises(ValueError):
        free_slot("red_tray", dict(zip("abcdef",slots)), "new",True)
    assert free_slot("red_tray", dict(zip("abcdef",slots)), "a",True) == slots[0]
