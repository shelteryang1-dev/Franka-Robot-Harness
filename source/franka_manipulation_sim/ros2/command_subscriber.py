"""ROS 2 end-effector command adapter."""

import math


def validate_pose(frame_id, position, quaternion_xyzw):
    if frame_id != "world":
        raise ValueError("Command frame must be world")
    if len(position) != 3 or len(quaternion_xyzw) != 4:
        raise ValueError("Pose dimensions invalid")
    if not all(math.isfinite(v) for v in (*position, *quaternion_xyzw)):
        raise ValueError("Pose non-finite")
    norm = math.hypot(*quaternion_xyzw)
    if not math.isfinite(norm) or norm < 1e-6:
        raise ValueError("Quaternion norm invalid")
    x, y, z, w = (v / norm for v in quaternion_xyzw)
    return tuple(position), (x, y, z, w)


class Ros2CommandSubscriber:

    def __init__(self, position_w, quaternion_xyzw, gripper_closed=False):
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import Bool
        from rclpy.executors import SingleThreadedExecutor

        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node("franka_command_subscriber")
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.position = tuple(position_w)
        self.quaternion = tuple(quaternion_xyzw)
        self.gripper_closed = bool(gripper_closed)
        self.counts = {"pose_received": 0, "pose_rejected": 0, "gripper_received": 0}
        self.node.create_subscription(PoseStamped, "/ee_target_pose", self._pose, 1)
        self.node.create_subscription(Bool, "/gripper_command", self._gripper, 1)

    def _pose(self, message):
        p, q = message.pose.position, message.pose.orientation
        try:
            position, quaternion = validate_pose(message.header.frame_id, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w))
        except ValueError as error:
            self.counts["pose_rejected"] += 1
            self.node.get_logger().warning(str(error))
            return
        self.position, self.quaternion = position, quaternion
        self.counts["pose_received"] += 1

    def _gripper(self, message):

        self.gripper_closed = message.data
        self.counts["gripper_received"] += 1

    def get_latest_command(self):
        for _ in range(4):
            self.executor.spin_once(timeout_sec=0.0)
        return self.position, self.quaternion, -1.0 if self.gripper_closed else 1.0

    def close(self):
        self.executor.remove_node(self.node)
        self.executor.shutdown()
        self.node.destroy_node()
