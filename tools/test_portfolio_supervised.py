"""Test portfolio supervised."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_portfolio_supervised import wait_log


class Process:
    def __init__(self, code=None):
        self.returncode = code

    def poll(self):
        return self.returncode


class ReadinessTests(unittest.TestCase):
    def test_ready_alive(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "log"
            path.write_text("[HARNESS_READY]", encoding="utf-8")
            wait_log(Process(), path, "[HARNESS_READY]", 180)

    def test_dead_process_not_ready(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "log"
            path.write_text("[HARNESS_READY]", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                wait_log(Process(1), path, "[HARNESS_READY]", 180)

    def test_final_deadline_read(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "log"
            path.write_text("", encoding="utf-8")
            def mark_ready(_):
                path.write_text("[HARNESS_READY]", encoding="utf-8")
            with patch("run_portfolio_supervised.time.monotonic", side_effect=[0, 359]), \
                 patch("run_portfolio_supervised.time.sleep", side_effect=mark_ready):
                wait_log(Process(), path, "[HARNESS_READY]", 180)

    def test_bounded_wait(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing"
            with patch("run_portfolio_supervised.time.monotonic", side_effect=[0, 360]):
                with self.assertRaises(TimeoutError):
                    wait_log(Process(), path, "[HARNESS_READY]", 180)


if __name__ == "__main__":
    unittest.main()
