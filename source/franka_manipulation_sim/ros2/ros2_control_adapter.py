"""ROS 2 joint-command adapter."""

import math
import time


def validate_joint_command(names, positions, expected, limits):
    """Validate and reorder joint position targets."""
    if len(names) != len(expected) or len(set(names)) != len(names) or set(names) != set(expected):
        raise ValueError("Joint names mismatch")
    if len(positions) != len(names) or not all(math.isfinite(v) for v in positions):
        raise ValueError("Joint targets invalid")
    values = dict(zip(names, positions))
    result = [values[name] for name in expected]
    for name, value, (low, high) in zip(expected, result, limits):
        if not low <= value <= high:
            raise ValueError(f"{name} target={value} outside [{low}, {high}]")
    return result


class Ros2ControlAdapter:
    """Hold measured joints on invalid or stale commands."""

    def __init__(self, names, limits, initial, timeout=1.0):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool, String

        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("isaac_joint_adapter")
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.names, self.limits = names, limits
        self.target = list(initial)
        self.gripper = 1.0
        self.last_command = None
        self.timeout = timeout
        self.fault = None
        self.received = 0
        self.fault_pub = self.node.create_publisher(String, "/isaac_control_fault", 10)
        self._string = String
        self.node.create_subscription(JointState, "/isaac_joint_commands", self._command, 1)
        self.node.create_subscription(Bool, "/gripper_command", self._gripper, 1)

    def _command(self, message):
        if self.fault:
            return
        try:
            self.target = validate_joint_command(message.name, message.position, self.names, self.limits)
        except ValueError as error:
            self.fault = str(error)
            self.node.get_logger().error(self.fault)
            return
        self.last_command = time.monotonic()
        self.received += 1

    def _gripper(self, message):
        if not self.fault:
            self.gripper = -1.0 if message.data else 1.0

    def step(self, current):
        for _ in range(6):
            self.executor.spin_once(timeout_sec=0.0)
        if self.last_command is not None and time.monotonic() - self.last_command > self.timeout:
            if not self.fault:
                self.fault = "Joint command timeout; holding position"
                self.node.get_logger().error(self.fault)
        if self.fault:
            # Latch once after a fault.
            if not hasattr(self, "hold"):
                self.hold = list(current)
            self.target = self.hold
            self.fault_pub.publish(self._string(data=self.fault))
        return self.target, self.gripper

    def close(self):
        self.executor.remove_node(self.node)
        self.executor.shutdown()
        self.node.destroy_node()
