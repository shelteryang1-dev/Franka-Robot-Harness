"""Single-environment ROS 2 state publishing."""

from __future__ import annotations

class Ros2StatePublisher:
    """Publish env-0 state and bridge TF."""

    def __init__(self, raw_env, publish_every_n_steps: int = 1, joint_topic: str = "/joint_states", publish_clock: bool = False):
        import omni.graph.core as og
        import rclpy
        import usdrt.Sdf
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState

        self._raw_env = raw_env
        self._rclpy = rclpy
        self._pose_type = PoseStamped
        self._joint_state_type = JointState
        self._publish_every_n_steps = max(1, int(publish_every_n_steps))
        self._step_count = 0

        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node("franka_sim_state_publisher")
        self._joint_publisher = self._node.create_publisher(JointState, joint_topic, 10)
        self._clock_publisher = None
        if publish_clock:
            from rosgraph_msgs.msg import Clock
            self._clock_type = Clock
            self._clock_publisher = self._node.create_publisher(Clock, "/clock", 10)
        self._ee_publisher = self._node.create_publisher(PoseStamped, "/ee_pose", 10)
        self._object_publisher = self._node.create_publisher(PoseStamped, "/object_pose", 10)
        self._joint_names = list(raw_env.scene["robot"].joint_names)

        robot_path = "/World/envs/env_0/Robot"
        object_path = "/World/envs/env_0/Object"
        graph_path = "/FrankaRos2StateGraph"
        og.Controller.edit(
            {
                "graph_path": graph_path,
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPhysicsStep", "isaacsim.core.nodes.OnPhysicsStep"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("PublishTF.inputs:topicName", "/tf"),
                    (
                        "PublishTF.inputs:targetPrims",
                        [usdrt.Sdf.Path(robot_path), usdrt.Sdf.Path(object_path)],
                    ),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPhysicsStep.outputs:step", "PublishTF.inputs:execIn"),
                    ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
                ],
            },
        )

    @staticmethod
    def _fill_stamp(header, seconds: float) -> None:

        header.stamp.sec, header.stamp.nanosec = divmod(round(seconds * 1_000_000_000), 1_000_000_000)

    def _make_pose(self, frame_id: str, position, quaternion_xyzw, simulation_time: float):

        message = self._pose_type()
        message.header.frame_id = frame_id
        self._fill_stamp(message.header, simulation_time)
        message.pose.position.x = float(position[0])
        message.pose.position.y = float(position[1])
        message.pose.position.z = float(position[2])

        message.pose.orientation.x = float(quaternion_xyzw[0])
        message.pose.orientation.y = float(quaternion_xyzw[1])
        message.pose.orientation.z = float(quaternion_xyzw[2])
        message.pose.orientation.w = float(quaternion_xyzw[3])
        return message

    def publish(self) -> None:

        self._step_count += 1
        if self._step_count % self._publish_every_n_steps != 0:
            return

        ee_frame = self._raw_env.scene["ee_frame"]
        object_data = self._raw_env.scene["object"].data
        robot_data = self._raw_env.scene["robot"].data

        ee_position = ee_frame.data.target_pos_w.torch[0, 0]
        ee_quaternion = ee_frame.data.target_quat_w.torch[0, 0]
        object_position = object_data.root_pos_w.torch[0]
        object_quaternion = object_data.root_quat_w.torch[0]
        simulation_time = float(self._raw_env.common_step_counter * self._raw_env.step_dt)
        if self._clock_publisher is not None:
            clock = self._clock_type()
            clock.clock.sec, clock.clock.nanosec = divmod(round(simulation_time * 1e9), 1_000_000_000)
            self._clock_publisher.publish(clock)

        joint_message = self._joint_state_type()
        self._fill_stamp(joint_message.header, simulation_time)
        joint_message.name = self._joint_names
        joint_message.position = robot_data.joint_pos.torch[0].detach().cpu().tolist()
        joint_message.velocity = robot_data.joint_vel.torch[0].detach().cpu().tolist()
        self._joint_publisher.publish(joint_message)

        self._ee_publisher.publish(
            self._make_pose("world", ee_position, ee_quaternion, simulation_time)
        )
        self._object_publisher.publish(
            self._make_pose("world", object_position, object_quaternion, simulation_time)
        )
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def close(self) -> None:

        if self._node is not None:
            self._node.destroy_node()
            self._node = None
