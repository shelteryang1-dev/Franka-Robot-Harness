"""Franka ros2 control.launch."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    return LaunchDescription([
        SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE",
                               get_package_share_directory("franka_sim_ros2_control")+"/config/dds_udp.xml"),
        DeclareLaunchArgument("contract", default_value="/workspace/project/results/ros2_control_stack/asset_joint_contract.json"),
        DeclareLaunchArgument("output", default_value="/workspace/project/results/ros2_control_stack/control"),
        Node(package="franka_sim_ros2_control", executable="control_stack.py", output="screen",
             arguments=["--contract", LaunchConfiguration("contract"), "--output", LaunchConfiguration("output"),
                        "--config", get_package_share_directory("franka_sim_ros2_control")+"/config/controllers.yaml"]),
    ])
