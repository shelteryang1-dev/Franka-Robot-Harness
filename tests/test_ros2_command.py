"""Test ros2 command."""

import math
import unittest
from franka_manipulation_sim.ros2.command_subscriber import validate_pose


class CommandTests(unittest.TestCase):
    def test_order_and_normalization(self):
        p, q = validate_pose("world", (1, 2, 3), (0, 0, 2, 2))
        self.assertEqual(p, (1, 2, 3))
        self.assertAlmostEqual(q[2], math.sqrt(0.5))
        self.assertAlmostEqual(q[3], math.sqrt(0.5))
        self.assertEqual(q[:2], (0, 0))

    def test_feedback_field_mapping(self):
        from types import SimpleNamespace as Fields
        from franka_manipulation_sim.ros2.state_publisher import Ros2StatePublisher
        publisher = Ros2StatePublisher.__new__(Ros2StatePublisher)
        publisher._pose_type = lambda: Fields(header=Fields(stamp=Fields()),
                                             pose=Fields(position=Fields(), orientation=Fields()))
        message = publisher._make_pose("world", [1, 2, 3], [0.1, 0.2, 0.3, 0.4], 0.9999999999)
        q = message.pose.orientation
        self.assertEqual((q.x, q.y, q.z, q.w), (0.1, 0.2, 0.3, 0.4))
        self.assertEqual((message.header.stamp.sec, message.header.stamp.nanosec), (1, 0))

    def test_invalid(self):
        for frame, pos, quat in (("base", (0, 0, 0), (0, 0, 0, 1)),
                                 ("world", (float("nan"), 0, 0), (0, 0, 0, 1)),
                                 ("world", (0, 0, 0), (0, 0, 0, 0)),
                                 ("world", (0, 0, 0), (float("inf"), 0, 0, 1))):
            with self.subTest(frame=frame, pos=pos, quat=quat), self.assertRaises(ValueError):
                validate_pose(frame, pos, quat)


if __name__ == "__main__":
    unittest.main()
