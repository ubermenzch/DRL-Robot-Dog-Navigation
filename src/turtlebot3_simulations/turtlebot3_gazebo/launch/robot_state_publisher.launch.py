#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Darby Lim

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ==================== 机器人模型说明 ====================
    # 机器人模型固定为 waffle（TURTLEBOT3_MODEL 用于世界文件，不用于机器人模型）
    # - Gazebo中实际使用的机器人SDF模型固定为：
    #   /root/DRL-Robot-Dog-Navigation/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_waffle/model.sdf
    #   此模型在世界文件中通过 <uri>model://turtlebot3_waffle</uri> 引用
    # - robot_state_publisher 使用URDF文件来发布TF变换（用于ROS2节点通信）
    #   如果环境变量 TURTLEBOT3_ROBOT_MODEL 存在则使用它，否则默认使用 waffle
    TURTLEBOT3_ROBOT_MODEL = os.environ.get("TURTLEBOT3_ROBOT_MODEL", "waffle")
    TURTLEBOT3_MODEL = os.environ.get("TURTLEBOT3_MODEL", "未设置")

    use_sim_time = LaunchConfiguration("use_sim_time", default="false")
    urdf_file_name = "turtlebot3_" + TURTLEBOT3_ROBOT_MODEL + ".urdf"

    # 调试信息：打印环境变量值，确保使用正确的机器人模型
    print("=" * 60)
    print("robot_state_publisher 环境变量检查:")
    print("  TURTLEBOT3_MODEL (世界文件): {}".format(TURTLEBOT3_MODEL))
    print("  TURTLEBOT3_ROBOT_MODEL (机器人模型): {}".format(TURTLEBOT3_ROBOT_MODEL))
    print("  urdf_file_name: {}".format(urdf_file_name))
    print("=" * 60)
    print("机器人SDF模型固定为: /root/DRL-Robot-Dog-Navigation/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_waffle/model.sdf")

    urdf = os.path.join(
        get_package_share_directory("turtlebot3_description"), "urdf", urdf_file_name
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulation (Gazebo) clock if true",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{"use_sim_time": use_sim_time}],
                arguments=[urdf],
            ),
        ]
    )
