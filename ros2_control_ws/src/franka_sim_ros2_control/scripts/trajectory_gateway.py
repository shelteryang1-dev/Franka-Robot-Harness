"""Switch controllers for joint trajectories."""

import json
import time
from rclpy.action import ActionClient
from controller_manager_msgs.srv import SwitchController
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import String


class TrajectoryGateway:

    def __init__(self, node, current, home):
        self.node, self.current, self.home = node, current, list(home)
        self.switch = node.create_client(SwitchController, "/controller_manager/switch_controller")
        self.action = ActionClient(node, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory")
        self.output = node.create_publisher(String, "/harness/trajectory_result", 10)
        node.create_subscription(String, "/harness/trajectory_request", self.receive, 10)
        self.pending = None
        self.handle = None

    def receive(self, message):
        request = json.loads(message.data)
        if request.get("cancel"):
            if self.pending == request.get("id"):
                self.cancelled = True
                if self.handle:
                    self.handle.cancel_goal_async()
            return
        if self.pending or request.get("name") != "home":
            self.output.publish(String(data=json.dumps({"id": request.get("id"), "success": False,
                                                       "error": "Trajectory busy or pose unknown"})))
            return
        self.pending, self.started = request["id"], time.monotonic()
        self.started_sim = self.node.get_clock().now().nanoseconds * 1e-9
        self.cancelled, self.handle = False, None
        self.phase, self.future = "activate", self.set_mode(True)

    def set_mode(self, trajectory):
        request = SwitchController.Request()
        request.activate_controllers = ["arm_controller" if trajectory else "skill_controller"]
        request.deactivate_controllers = ["skill_controller" if trajectory else "arm_controller"]
        request.strictness, request.timeout.sec = 2, 5
        return self.switch.call_async(request)

    def tick(self):
        if not self.pending:
            return
        simulated = self.node.get_clock().now().nanoseconds * 1e-9 - self.started_sim
        # Use simulation time for trajectory progress.
        # Bound wall time separately.
        if time.monotonic()-self.started > 90 or simulated > 10 or simulated < 0:
            raise TimeoutError("Trajectory timeout")
        if not self.future.done():
            return
        value = self.future.result()
        if self.phase == "activate":
            if not value.ok:
                raise RuntimeError("Trajectory controller activation failed")
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = [f"panda_joint{i}" for i in range(1, 8)]
            for positions, sec, nanosec in ((self.current(), 0, 300000000), (self.home, 3, 0)):
                point = JointTrajectoryPoint(positions=positions)
                point.time_from_start.sec, point.time_from_start.nanosec = sec, nanosec
                goal.trajectory.points.append(point)
            self.future, self.phase = self.action.send_goal_async(goal), "accepted"
        elif self.phase == "accepted":
            self.handle = value
            if not value.accepted:
                self.result = {"success": False, "error": "Trajectory goal rejected"}
                self.future, self.phase = self.set_mode(False), "restore"
            else:
                if self.cancelled:
                    value.cancel_goal_async()
                self.future, self.phase = value.get_result_async(), "executed"
        elif self.phase == "executed":
            error = max(abs(a-b) for a,b in zip(self.current(), self.home))
            self.result = {"success": value.status == 4 and value.result.error_code == 0 and error < .03,
                           "action_status": value.status, "error_code": value.result.error_code,
                           "joint_error": error, "cancelled": self.cancelled}
            self.future, self.phase = self.set_mode(False), "restore"
        elif self.phase == "restore":
            if not value.ok:
                raise RuntimeError("Streaming controller activation failed")
            self.result["id"] = self.pending
            self.output.publish(String(data=json.dumps(self.result)))
            self.pending = None
