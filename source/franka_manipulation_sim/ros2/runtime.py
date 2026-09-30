"""Runtime setup."""

import os
import platform
import sys
from pathlib import Path


def prepare_runtime():
    if os.environ.get("FRANKA_ROS2_CONTROL_READY") == "1":
        return
    major = platform.freedesktop_os_release().get("VERSION_ID", "24").split(".")[0]
    distro = "humble" if major == "22" else "jazzy"
    root = Path(f"/isaac-sim/exts/isaacsim.ros2.core/{distro}")
    if not root.is_dir():
        raise RuntimeError(f"ROS runtime missing: {root}")
    environment = os.environ.copy()
    environment.update(ROS_DISTRO=distro, RMW_IMPLEMENTATION="rmw_fastrtps_cpp",
                       FRANKA_ROS2_CONTROL_READY="1")
    for key, suffix in (("PYTHONPATH", "rclpy"), ("LD_LIBRARY_PATH", "lib")):
        environment[key] = f"{root / suffix}:{environment.get(key, '')}".rstrip(":")
    os.execve("/isaac-sim/python.sh", ["/isaac-sim/python.sh", *sys.argv], environment)
