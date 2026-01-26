import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

# 从环境变量读取TURTLEBOT3_MODEL，用于指定世界模型文件
# 此参数应在启动脚本中从train.yaml读取并设置
# 如果未设置，使用默认值waffle（向后兼容）
TURTLEBOT3_MODEL = os.environ.get("TURTLEBOT3_MODEL", "waffle")


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    pause = LaunchConfiguration("pause", default="false")
    # 世界文件路径：turtlebot3_drl/{TURTLEBOT3_MODEL}.model
    # 例如：turtlebot3_model=base_world_100by100 -> turtlebot3_drl/base_world_100by100.model
    world_file_name = "turtlebot3_drl/" + TURTLEBOT3_MODEL + ".model"
    world = os.path.join(
        get_package_share_directory("turtlebot3_gazebo"), "worlds", world_file_name
    )
    launch_file_dir = os.path.join(
        get_package_share_directory("turtlebot3_gazebo"), "launch"
    )
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_gazebo_ros, "launch", "gzserver.launch.py")
                ),
                launch_arguments={"world": world, "pause": pause}.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_gazebo_ros, "launch", "gzclient.launch.py")
                ),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [launch_file_dir, "/robot_state_publisher.launch.py"]
                ),
                launch_arguments={"use_sim_time": use_sim_time}.items(),
            ),
        ]
    )
