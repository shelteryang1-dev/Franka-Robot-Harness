"""Test harness."""

import pytest
from franka_manipulation_sim.harness.contracts import validate_plan, position_reached
from franka_manipulation_sim.harness.runtime import TaskRuntime


@pytest.mark.parametrize("value", [{}, {"steps": []}, {"steps": [{"skill": "shell"}]},
    {"steps": [{"skill": "lift", "object": "missing"}]},
    {"steps": [{"skill": "pick_place", "object": "cube", "target": "unknown"}]}])
def test_invalid_plan(value):
    with pytest.raises(ValueError):
        validate_plan(value)


def test_finite_success():
    assert position_reached([0, 0, 0], [0, 0, .01])
    assert not position_reached([float("nan"), 0, 0], [0, 0, 0])


class Backend:
    object_names = ("cube",)
    def __init__(self):
        self.starts, self.holds = 0, 0
    def start(self, step):
        self.starts += 1
    def poll(self):
        return "succeeded", {}
    def hold(self):
        self.holds += 1


def test_sequence_and_exclusive_owner():
    backend, events = Backend(), []
    runtime = TaskRuntime(backend, events.append)
    plan = {"steps": [{"skill": "lift", "object": "cube"}]*2}
    runtime.submit("one", plan)
    with pytest.raises(ValueError):
        runtime.submit("two", plan)
    runtime.tick()
    assert backend.starts == 2 and runtime.active == "one"
    runtime.tick()
    assert runtime.active is None and events[-1]["status"] == "succeeded"
    with pytest.raises(ValueError):
        runtime.submit("one", plan)


def test_cancel_holds():
    backend = Backend()
    runtime = TaskRuntime(backend, lambda event: None)
    runtime.submit("one", {"steps": [{"skill": "lift", "object": "cube"}]})
    runtime.cancel("one")
    assert backend.holds == 1 and runtime.active is None


def test_retry_is_bounded():
    class FailingBackend(Backend):
        def poll(self):
            return "failed", "grasp_failed"
        def can_retry(self, detail):
            return True
    backend, events = FailingBackend(), []
    runtime = TaskRuntime(backend, events.append)
    runtime.submit("retry", {"steps": [{"skill": "lift", "object": "cube"}]})
    runtime.tick()
    assert backend.starts == 2
    runtime.tick()
    assert runtime.active is None and events[-1]["status"] == "failed"
    assert sum(e["status"] == "retrying" for e in events) == 1


def test_timeout_is_terminal():
    backend, events = Backend(), []
    runtime = TaskRuntime(backend, events.append, timeout=-1)
    runtime.submit("timeout", {"steps": [{"skill": "lift", "object": "cube"}]})
    runtime.tick()
    assert events[-1]["status"] == "failed" and backend.holds == 1


@pytest.mark.parametrize("passed", [True, False])
def test_final_physical_verification_overrides_step_success(passed):
    class VerifiedBackend(Backend):
        def verify_plan(self, steps):
            return passed, {"cube": {"passed": passed}}
    events = []
    runtime = TaskRuntime(VerifiedBackend(), events.append)
    runtime.submit("verify", {"steps": [{"skill": "pick_place", "object": "cube", "target": "red_tray"}]})
    runtime.tick()
    assert events[-1]["status"] == ("succeeded" if passed else "failed")
    assert events[-1]["detail"]["final_checks"]["cube"]["passed"] is passed


def test_place_requires_immediately_held_same_object():
    with pytest.raises(ValueError, match="place"):
        validate_plan({"steps": [{"skill":"pick_place", "object":"cube", "target":"left"},
                                  {"skill":"place", "object":"cube", "target":"right"}]})
    assert validate_plan({"steps": [{"skill":"lift", "object":"cube"},
                                     {"skill":"place", "object":"cube", "target":"right"}]})["steps"][1]["skill"] == "place"
