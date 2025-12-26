import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Pose, Twist
from visualization_msgs.msg import Marker
from rclpy.logging import LoggingSeverity

SEVERITY = LoggingSeverity.ERROR


class SensorSubscriber(Node):
    def __init__(self, env_id=0):
        super().__init__(f"sensor_subscriber_env_{env_id}")
        self.get_logger().set_level(SEVERITY)
        self.subscriber_ = self.create_subscription(
            LaserScan, "scan", self.scan_listener_callback, 1
        )
        self.subscriber_ = self.create_subscription(
            Odometry, "odom", self.odom_listener_callback, 1
        )
        self.latest_position = None
        self.latest_heading = None
        self.latest_scan = None

    def scan_listener_callback(self, msg):
        self.latest_scan = msg.ranges[:]
        #print(len(self.latest_scan))

    def odom_listener_callback(self, msg):
        self.latest_position = msg.pose.pose.position
        self.latest_heading = msg.pose.pose.orientation

    def get_latest_sensor(self):
        # print(self.latest_scan, self.latest_position, self.latest_heading)
        return self.latest_scan, self.latest_position, self.latest_heading


class ScanSubscriber(Node):
    def __init__(self, env_id=0):
        super().__init__(f"scan_subscriber_env_{env_id}")
        self.get_logger().set_level(SEVERITY)
        self.subscriber_ = self.create_subscription(
            LaserScan, "scan", self.listener_callback, 1
        )
        self.latest_scan = None

    def listener_callback(self, msg):
        self.latest_scan = msg.ranges[:]

    def get_latest_scan(self):
        return self.latest_scan


class OdomSubscriber(Node):
    def __init__(self, env_id=0):
        super().__init__(f"odom_subscriber_env_{env_id}")
        self.get_logger().set_level(SEVERITY)
        self.subscriber_ = self.create_subscription(
            Odometry, "odom", self.listener_callback, 1
        )
        self.latest_position = None
        self.latest_heading = None

    def listener_callback(self, msg):
        self.latest_position = msg.pose.pose.position
        self.latest_heading = msg.pose.pose.orientation

    def get_latest_odom(self):
        return self.latest_position, self.latest_heading


class ResetWorldClient(Node):
    def __init__(self, env_id=0):
        super().__init__(f"reset_world_client_env_{env_id}")
        self.get_logger().set_level(SEVERITY)
        self.reset_client = self.create_client(Empty, "/reset_world")
        self.wait_for_service(self.reset_client, "reset_world")

    def wait_for_service(self, client, service_name, timeout=10.0):
        self.get_logger().info(f"Waiting for {service_name} service...")
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(
                f"Service {service_name} not available after waiting."
            )
            raise RuntimeError(f"Service {service_name} not available.")

    def reset_world(self):
        self.get_logger().info("Calling /gazebo/reset_world service...")
        request = Empty.Request()
        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info("World reset successfully.")
        else:
            self.get_logger().error(f"Failed to reset world: {future.exception()}")


class PhysicsClient(Node):
    def __init__(self, env_id=0):
        super().__init__(f"physics_client_env_{env_id}")
        self.get_logger().set_level(SEVERITY)
        self.unpause_client = self.create_client(Empty, "/unpause_physics")
        self.pause_client = self.create_client(Empty, "/pause_physics")

        self.wait_for_service(self.unpause_client, "unpause_physics")
        self.wait_for_service(self.pause_client, "pause_physics")

    def wait_for_service(self, client, service_name, timeout=10.0):
        self.get_logger().info(f"Waiting for {service_name} service...")
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(
                f"Service {service_name} not available after waiting."
            )
            raise RuntimeError(f"Service {service_name} not available.")

    def pause_physics(self):
        self.get_logger().info("Calling /gazebo/pause_physics service...")
        request = Empty.Request()
        future = self.pause_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info("Physics paused successfully.")
        else:
            self.get_logger().error(f"Failed to pause physics: {future.exception()}")

    def unpause_physics(self):
        self.get_logger().info("Calling /gazebo/unpause_physics service...")
        request = Empty.Request()
        future = self.unpause_client.call_async(request)

        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info("Physics unpaused successfully.")
        else:
            self.get_logger().error(f"Failed to unpause physics: {future.exception()}")


class SetModelStateClient(Node):
    def __init__(self, env_id=0):
        super().__init__(f"set_entity_state_client_env_{env_id}")
        self.get_logger().set_level(SEVERITY)
        self.client = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        #print("SetModelStateClient::create_client Done")
        while not self.client.wait_for_service(timeout_sec=1.0):
            print("SetModelStateClient::wait_for_service")
            self.get_logger().info("Service not available, waiting again...")
        self.request = SetEntityState.Request()

    def set_state(self, name, new_pose):
        self.request.state.name = name
        self.request.state.pose = new_pose
        self.future = self.client.call_async(self.request)
        # 等待服务调用完成并返回结果
        rclpy.spin_until_future_complete(self, self.future)
        if self.future.result() is not None:
            response = self.future.result()
            return response.success
        return False


class GoalModelClient(Node):
    """用于在Gazebo中设置绿色圆柱体目标模型的位置（模型已在世界文件中定义）"""
    def __init__(self, env_id=0, model_name="goal_cylinder"):
        super().__init__(f"goal_model_client_env_{env_id}")
        self.get_logger().set_level(SEVERITY)
        self.model_name = model_name
        self.services_ready = False
        
        # 创建服务客户端（延迟初始化，不立即等待服务）
        self.set_state_client = self.create_client(SetEntityState, "/gazebo/set_entity_state")

    def _ensure_services_ready(self, timeout=30.0):
        """确保服务可用（延迟初始化）"""
        if self.services_ready:
            return True
        
        # 等待服务可用
        self.get_logger().info("Waiting for Gazebo services to be available...")
        if not self.set_state_client.wait_for_service(timeout_sec=timeout):
            self.get_logger().warn(f"Service set_entity_state not available after {timeout}s, will retry later")
            return False
        
        self.services_ready = True
        self.get_logger().info("Gazebo services are ready")
        return True

    def set_goal_position(self, x, y, z=1.0):
        """设置目标圆柱体的位置（模型已在世界文件中定义）
        Args:
            x, y: 目标位置的 x, y 坐标（米）
            z: 目标位置的 z 坐标（米），默认 1.0 使圆柱体在地面上方1米，不阻挡激光扫描
        """
        # 确保服务可用
        if not self._ensure_services_ready():
            self.get_logger().warn("Gazebo services not available, skipping goal position update")
            return
        
        # 更新模型位置
        request = SetEntityState.Request()
        request.state.name = self.model_name
        request.state.pose.position.x = float(x)
        request.state.pose.position.y = float(y)
        request.state.pose.position.z = float(z)
        
        # 设置方向（保持水平，无旋转）
        request.state.pose.orientation.x = 0.0
        request.state.pose.orientation.y = 0.0
        request.state.pose.orientation.z = 0.0
        request.state.pose.orientation.w = 1.0
        
        # 设置速度和角速度为0（保持静止）
        request.state.twist.linear.x = 0.0
        request.state.twist.linear.y = 0.0
        request.state.twist.linear.z = 0.0
        request.state.twist.angular.x = 0.0
        request.state.twist.angular.y = 0.0
        request.state.twist.angular.z = 0.0
        
        # 设置参考坐标系（使用world坐标系）
        request.state.reference_frame = "world"
        
        # 调用服务并等待完成
        future = self.set_state_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        # 检查结果
        if future.result() is not None:
            response = future.result()
            if response.success:
                self.get_logger().info(f"Goal cylinder set to position ({x:.3f}, {y:.3f}, {z:.3f})")
            else:
                error_msg = response.status_message if hasattr(response, 'status_message') else "Unknown error"
                self.get_logger().warn(f"Failed to set goal cylinder position: {error_msg}")
        else:
            self.get_logger().warn(f"Failed to set goal cylinder position: Service returned None")


class CmdVelPublisher(Node):
    def __init__(self, env_id=0):
        super().__init__(f"cmd_vel_publisher_env_{env_id}")
        self.get_logger().set_level(SEVERITY)
        self.publisher_ = self.create_publisher(Twist, "cmd_vel", 1)
        self.timer = self.create_timer(0.1, self.publish_cmd_vel)

    def publish_cmd_vel(self, linear_velocity=0.0, angular_velocity=0.0):
        twist_msg = Twist()
        # Set linear and angular velocities
        twist_msg.linear.x = float(linear_velocity)  # Example linear velocity (m/s)
        twist_msg.angular.z = float(
            angular_velocity
        )  # Example angular velocity (rad/s)
        self.publisher_.publish(twist_msg)


class MarkerPublisher(Node):
    def __init__(self, env_id=0):
        super().__init__(f"marker_publisher_env_{env_id}")
        self.get_logger().set_level(SEVERITY)
        self.publisher = self.create_publisher(Marker, f"/env_{env_id}/visualization_marker", 1)

    def publish(self, x, y):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.1

        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        self.publisher.publish(marker)
        self.get_logger().info("Publishing Marker")


def run_scan(args=None):
    rclpy.init()
    reading_laser = ScanSubscriber()
    reading_laser.get_logger().info("Hello friend!")
    rclpy.spin(reading_laser)

    reading_laser.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    run_scan()
