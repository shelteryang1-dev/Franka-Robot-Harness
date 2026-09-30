"""Schedule skills and check task completion."""

import time
from .contracts import validate_plan


class TaskRuntime:
    """Execute one ordered task at a time."""

    def __init__(self, backend, emit, timeout=120.0):
        self.backend, self.emit, self.timeout = backend, emit, timeout
        self.active = None
        self.seen = set()
        self.index = 0
        self.steps = []
        self.started = 0.0
        self.retries = 0

    def submit(self, task_id, plan):
        if not isinstance(task_id, str) or not 1 <= len(task_id) <= 80:
            raise ValueError("Task ID length must be 1-80")
        if self.active:
            raise ValueError("Task already active")
        if task_id in self.seen:
            raise ValueError("Duplicate task ID")
        checked = validate_plan(plan, self.backend.object_names)
        self.active, self.steps, self.index, self.retries = task_id, checked["steps"], 0, 0
        self.seen.add(task_id)
        self.started = time.monotonic()
        self.emit({"task_id": task_id, "status": "accepted", "plan": checked})
        self._start_step()

    def _start_step(self):
        try:
            self.backend.start(self.steps[self.index])
            self.emit({"task_id": self.active, "status": "running", "step": self.index})
        except Exception as error:
            self.finish("failed", str(error))

    def tick(self):
        if not self.active:
            return
        try:
            if time.monotonic()-self.started > self.timeout:
                raise TimeoutError("Task timeout")
            status, detail = self.backend.poll()
            if status == "failed":
                if (self.retries < 1 and hasattr(self.backend, "can_retry")
                        and self.backend.can_retry(detail)):
                    self.retries += 1
                    self.emit({"task_id": self.active, "status": "retrying", "step": self.index,
                               "attempt": self.retries, "detail": detail})
                    self.backend.hold()
                    self._start_step()
                else:
                    self.finish("failed", detail)
            elif status == "succeeded":
                self.emit({"task_id": self.active, "status": "step_succeeded",
                           "step": self.index, "detail": detail})
                self.index += 1
                self.retries = 0
                if self.index == len(self.steps):
                    if hasattr(self.backend, "verify_plan"):
                        passed, checks = self.backend.verify_plan(self.steps)
                        detail = {"last_step": detail, "final_checks": checks}
                        if not passed:
                            self.finish("failed", detail)
                            return
                    self.finish("succeeded", detail)
                else:
                    self._start_step()
        except Exception as error:
            self.finish("failed", str(error))

    def finish(self, status, detail):
        if self.active:
            task_id = self.active
            self.backend.hold()
            self.active = None
            self.emit({"task_id": task_id, "status": status, "completed_steps": self.index,
                       "detail": detail, "wall_seconds": time.monotonic()-self.started})

    def cancel(self, task_id):
        if task_id != self.active or not self.active:
            raise ValueError("Active task not found")
        self.finish("cancelled", "Task cancelled")
