import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import LaserScan, Imu
import numpy as np
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Pose, Twist
from visualization_msgs.msg import Marker
from rclpy.logging import LoggingSeverity
import time
import threading

SEVERITY = LoggingSeverity.ERROR


class SensorSubscriber(Node):
    def __init__(self, env_id=0, scan_range=None, localization_noise_stddev=0.0, nodes_logger=None,
                 scan_max_freq=0.0, odom_max_freq=0.0, imu_max_freq=0.0,
                 scan_enable_log=False, odom_enable_log=False, imu_enable_log=False):
        super().__init__(f"sensor_subscriber_env_{env_id}")
        self.get_logger().set_level(SEVERITY)
        self.env_id = env_id
        self.nodes_logger = nodes_logger
        # 各传感器独立的日志开关
        self.scan_enable_log = scan_enable_log  # 是否记录激光雷达日志
        self.odom_enable_log = odom_enable_log  # 是否记录里程计日志
        self.imu_enable_log = imu_enable_log  # 是否记录IMU日志
        self.subscriber_ = self.create_subscription(
            LaserScan, "scan", self.scan_listener_callback, 1
        )
        self.subscriber_ = self.create_subscription(
            Odometry, "odom", self.odom_listener_callback, 1
        )
        self.imu_subscriber_ = self.create_subscription(
            Imu, "imu", self.imu_listener_callback, 1
        )
        self.latest_position = None
        self.latest_position_raw = None  # 未添加噪声的原始position
        self.latest_heading = None
        self.latest_scan = None
        self.latest_linear_velocity = 0.0
        self.latest_angular_velocity = 0.0
        # IMU 相关最新数据
        self.latest_imu_orientation = None
        self.latest_imu_linear_acc = None
        self.latest_imu_angular_vel = None
        self.scan_range = float(scan_range) if scan_range is not None else None
        self.localization_noise_stddev = float(localization_noise_stddev) if localization_noise_stddev is not None else 0.0
        # 频率限制参数（Hz），0或负数表示不限制
        # 先转换为 float，然后检查是否 > 0
        try:
            scan_max_freq_float = float(scan_max_freq) if scan_max_freq is not None else 0.0
            self.scan_max_freq = scan_max_freq_float if scan_max_freq_float > 0 else 0.0
        except (ValueError, TypeError):
            self.scan_max_freq = 0.0
            
        try:
            odom_max_freq_float = float(odom_max_freq) if odom_max_freq is not None else 0.0
            self.odom_max_freq = odom_max_freq_float if odom_max_freq_float > 0 else 0.0
        except (ValueError, TypeError):
            self.odom_max_freq = 0.0
            
        try:
            imu_max_freq_float = float(imu_max_freq) if imu_max_freq is not None else 0.0
            self.imu_max_freq = imu_max_freq_float if imu_max_freq_float > 0 else 0.0
        except (ValueError, TypeError):
            self.imu_max_freq = 0.0
        
        # 调试：打印设置后的值
        if nodes_logger is not None:
            nodes_logger.log(env_id, f"[SensorSubscriber] 频率限制参数设置完成: scan_max_freq={self.scan_max_freq} odom_max_freq={self.odom_max_freq} imu_max_freq={self.imu_max_freq} (传入值: scan={scan_max_freq} odom={odom_max_freq} imu={imu_max_freq})")
            nodes_logger.log(env_id, f"[SensorSubscriber] 日志开关设置: scan={self.scan_enable_log} odom={self.odom_enable_log} imu={self.imu_enable_log}")
        # 用于频率限制的时间戳（记录上次处理的时间）
        self.scan_last_process_time = None
        self.odom_last_process_time = None
        self.imu_last_process_time = None
        # 用于频率统计的时间窗口（滑动窗口，保留最近1秒的时间戳）
        self.scan_timestamps = []  # 最近1秒内的消息时间戳列表（真实callback频率）
        self.odom_timestamps = []  # 最近1秒内的消息时间戳列表（真实callback频率）
        self.imu_timestamps = []   # 最近1秒内的消息时间戳列表（真实callback频率）
        self.scan_update_timestamps = []  # 最近1秒内实际更新数据的时间戳列表（实际更新频率）
        self.odom_update_timestamps = []  # 最近1秒内实际更新数据的时间戳列表（实际更新频率）
        self.imu_update_timestamps = []   # 最近1秒内实际更新数据的时间戳列表（实际更新频率）
        self.scan_count = 0
        self.odom_count = 0
        self.imu_count = 0
        # 线程安全锁
        self._data_lock = threading.Lock()
        # 启动后台线程自动接收数据
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = None
        self._shutdown_flag = threading.Event()
        self._start_spin_thread()

    def _start_spin_thread(self):
        """启动后台线程自动接收ROS消息
        
        使用executor.spin()在后台线程中持续运行，当有消息到达时自动调用callback。
        这样主线程不会被阻塞，可以随时读取最新的传感器数据。
        """
        def spin_loop():
            try:
                # executor.spin()会持续运行，直到executor被shutdown
                # 当有消息到达时，会自动调用对应的callback函数
                self._executor.spin()
            except Exception as e:
                if not self._shutdown_flag.is_set():
                    self.get_logger().error(f"Error in spin thread: {e}")
        
        self._spin_thread = threading.Thread(target=spin_loop, daemon=True, name=f"sensor_subscriber_spin_{self.env_id}")
        self._spin_thread.start()
    
    def shutdown(self):
        """停止后台线程并清理资源"""
        if self._shutdown_flag is not None:
            self._shutdown_flag.set()
        if self._executor is not None:
            self._executor.shutdown()
        if self._spin_thread is not None and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)
        self.destroy_node()

    def scan_listener_callback(self, msg):
        current_time = time.time()
        self.scan_count += 1
        
        # 更新滑动窗口：添加当前时间戳，移除超过1秒的旧时间戳（真实callback频率）
        self.scan_timestamps.append(current_time)
        cutoff_time = current_time - 1.0
        self.scan_timestamps = [ts for ts in self.scan_timestamps if ts > cutoff_time]
        
        # 计算真实callback频率：统计最近1秒内的消息数量
        callback_freq = len(self.scan_timestamps) if len(self.scan_timestamps) > 0 else 0.0
        
        # 频率限制：如果设置了最大频率，检查是否应该跳过本次处理
        should_update = True
        if self.scan_max_freq > 0.0:
            if self.scan_last_process_time is not None:
                min_interval = 1.0 / self.scan_max_freq
                if current_time - self.scan_last_process_time < min_interval:
                    # 跳过本次处理，不更新数据
                    should_update = False
            if should_update:
                self.scan_last_process_time = current_time
        
        # 只有实际更新数据时才记录到update_timestamps
        if should_update:
            self.scan_update_timestamps.append(current_time)
            cutoff_time = current_time - 1.0
            self.scan_update_timestamps = [ts for ts in self.scan_update_timestamps if ts > cutoff_time]
        
        # 计算实际更新频率：统计最近1秒内实际更新数据的次数
        update_freq = len(self.scan_update_timestamps) if len(self.scan_update_timestamps) > 0 else 0.0
        
        # 如果被跳过，直接返回
        if not should_update:
            return
        
        scan = np.asarray(msg.ranges[:], dtype=np.float32)
        if self.scan_range is not None:
            scan[~np.isfinite(scan)] = self.scan_range
            scan = np.minimum(scan, self.scan_range)
        
        # 线程安全更新数据
        with self._data_lock:
            self.latest_scan = scan.tolist()
        
        # 记录日志（只有实际更新数据时才记录，且需要启用该传感器的日志）
        if self.nodes_logger is not None and self.scan_enable_log:
            scan_min = float(np.min(scan)) if scan.size > 0 else float('nan')
            scan_max = float(np.max(scan)) if scan.size > 0 else float('nan')
            scan_mean = float(np.mean(scan)) if scan.size > 0 else float('nan')
            callback_freq_str = f"{callback_freq:.2f}Hz" if callback_freq > 0 else "0.00Hz"
            update_freq_str = f"{update_freq:.2f}Hz" if update_freq > 0 else "0.00Hz"
            limit_str = f"limit={self.scan_max_freq:.1f}Hz" if self.scan_max_freq > 0 else "no_limit"
            self.nodes_logger.log(
                self.env_id,
                f"[SUBSCRIBER] /scan received count={self.scan_count} callback_freq={callback_freq_str} update_freq={update_freq_str} {limit_str} "
                f"ranges_len={len(scan)} min={scan_min:.4f} max={scan_max:.4f} mean={scan_mean:.4f} "
                f"angle_min={msg.angle_min:.4f} angle_max={msg.angle_max:.4f} angle_increment={msg.angle_increment:.4f}"
            )
        #print(len(self.latest_scan))

    def odom_listener_callback(self, msg):
        current_time = time.time()
        self.odom_count += 1
        
        # 更新滑动窗口：添加当前时间戳，移除超过1秒的旧时间戳（真实callback频率）
        self.odom_timestamps.append(current_time)
        cutoff_time = current_time - 1.0
        self.odom_timestamps = [ts for ts in self.odom_timestamps if ts > cutoff_time]
        
        # 计算真实callback频率：统计最近1秒内的消息数量
        callback_freq = len(self.odom_timestamps) if len(self.odom_timestamps) > 0 else 0.0
        
        # 频率限制：如果设置了最大频率，检查是否应该跳过本次处理
        should_update = True
        if self.odom_max_freq > 0.0:
            if self.odom_last_process_time is not None:
                min_interval = 1.0 / self.odom_max_freq
                if current_time - self.odom_last_process_time < min_interval:
                    # 跳过本次处理，不更新数据
                    should_update = False
            if should_update:
                self.odom_last_process_time = current_time
        
        # 只有实际更新数据时才记录到update_timestamps
        if should_update:
            self.odom_update_timestamps.append(current_time)
            cutoff_time = current_time - 1.0
            self.odom_update_timestamps = [ts for ts in self.odom_update_timestamps if ts > cutoff_time]
        
        # 计算实际更新频率：统计最近1秒内实际更新数据的次数
        update_freq = len(self.odom_update_timestamps) if len(self.odom_update_timestamps) > 0 else 0.0
        
        # 如果被跳过，直接返回
        if not should_update:
            return
        
        # copy + optional localization noise (Gaussian) on x/y
        pos = msg.pose.pose.position
        # 始终保存未添加噪声的原始position
        # 线程安全更新数据
        with self._data_lock:
            self.latest_position_raw = pos
            if self.localization_noise_stddev > 0.0:
                noisy_x = float(pos.x) + float(np.random.normal(0.0, self.localization_noise_stddev))
                noisy_y = float(pos.y) + float(np.random.normal(0.0, self.localization_noise_stddev))
                self.latest_position = type(pos)()
                self.latest_position.x = noisy_x
                self.latest_position.y = noisy_y
                self.latest_position.z = float(pos.z)
                # print(f"noisy_position: {self.latest_position.x}, {self.latest_position.y}, {self.latest_position.z}")
            else:
                self.latest_position = pos

            self.latest_heading = msg.pose.pose.orientation
            self.latest_linear_velocity = float(msg.twist.twist.linear.x)
            self.latest_angular_velocity = float(msg.twist.twist.angular.z)
        
        # 记录日志（只有实际更新数据时才记录，且需要启用该传感器的日志）
        if self.nodes_logger is not None and self.odom_enable_log:
            callback_freq_str = f"{callback_freq:.2f}Hz" if callback_freq > 0 else "0.00Hz"
            update_freq_str = f"{update_freq:.2f}Hz" if update_freq > 0 else "0.00Hz"
            limit_str = f"limit={self.odom_max_freq:.1f}Hz" if self.odom_max_freq > 0 else "no_limit"
            self.nodes_logger.log(
                self.env_id,
                f"[SUBSCRIBER] /odom received count={self.odom_count} callback_freq={callback_freq_str} update_freq={update_freq_str} {limit_str} "
                f"position=({pos.x:.4f},{pos.y:.4f},{pos.z:.4f}) "
                f"orientation=({msg.pose.pose.orientation.x:.4f},{msg.pose.pose.orientation.y:.4f},"
                f"{msg.pose.pose.orientation.z:.4f},{msg.pose.pose.orientation.w:.4f}) "
                f"linear_vel=({msg.twist.twist.linear.x:.4f},{msg.twist.twist.linear.y:.4f},{msg.twist.twist.linear.z:.4f}) "
                f"angular_vel=({msg.twist.twist.angular.x:.4f},{msg.twist.twist.angular.y:.4f},{msg.twist.twist.angular.z:.4f})"
            )
        # print(f"latest_linear_velocity: {self.latest_linear_velocity}, latest_angular_velocity: {self.latest_angular_velocity}")

    def imu_listener_callback(self, msg: Imu):
        """IMU 回调：记录线加速度、角速度和姿态（用于 reset 阶段的静止检测）"""
        current_time = time.time()
        self.imu_count += 1
        
        # 更新滑动窗口：添加当前时间戳，移除超过1秒的旧时间戳（真实callback频率）
        self.imu_timestamps.append(current_time)
        cutoff_time = current_time - 1.0
        self.imu_timestamps = [ts for ts in self.imu_timestamps if ts > cutoff_time]
        
        # 计算真实callback频率：统计最近1秒内的消息数量
        callback_freq = len(self.imu_timestamps) if len(self.imu_timestamps) > 0 else 0.0
        
        # 频率限制：如果设置了最大频率，检查是否应该跳过本次处理
        should_update = True
        if self.imu_max_freq > 0.0:
            if self.imu_last_process_time is not None:
                min_interval = 1.0 / self.imu_max_freq
                if current_time - self.imu_last_process_time < min_interval:
                    # 跳过本次处理，不更新数据
                    should_update = False
            if should_update:
                self.imu_last_process_time = current_time
        
        # 只有实际更新数据时才记录到update_timestamps
        if should_update:
            self.imu_update_timestamps.append(current_time)
            cutoff_time = current_time - 1.0
            self.imu_update_timestamps = [ts for ts in self.imu_update_timestamps if ts > cutoff_time]
        
        # 计算实际更新频率：统计最近1秒内实际更新数据的次数
        update_freq = len(self.imu_update_timestamps) if len(self.imu_update_timestamps) > 0 else 0.0
        
        # 如果被跳过，直接返回
        if not should_update:
            return
        
        # 线程安全更新数据
        with self._data_lock:
            self.latest_imu_orientation = msg.orientation
            self.latest_imu_linear_acc = msg.linear_acceleration
            self.latest_imu_angular_vel = msg.angular_velocity
        
        # 记录日志（只有实际更新数据时才记录，且需要启用该传感器的日志）
        if self.nodes_logger is not None and self.imu_enable_log:
            callback_freq_str = f"{callback_freq:.2f}Hz" if callback_freq > 0 else "0.00Hz"
            update_freq_str = f"{update_freq:.2f}Hz" if update_freq > 0 else "0.00Hz"
            limit_str = f"limit={self.imu_max_freq:.1f}Hz" if self.imu_max_freq > 0 else "no_limit"
            self.nodes_logger.log(
                self.env_id,
                f"[SUBSCRIBER] /imu received count={self.imu_count} callback_freq={callback_freq_str} update_freq={update_freq_str} {limit_str} "
                f"orientation=({msg.orientation.x:.4f},{msg.orientation.y:.4f},"
                f"{msg.orientation.z:.4f},{msg.orientation.w:.4f}) "
                f"linear_acc=({msg.linear_acceleration.x:.4f},{msg.linear_acceleration.y:.4f},{msg.linear_acceleration.z:.4f}) "
                f"angular_vel=({msg.angular_velocity.x:.4f},{msg.angular_velocity.y:.4f},{msg.angular_velocity.z:.4f})"
            )

    def get_latest_sensor(self):
        """线程安全地获取最新的传感器数据（scan, position, heading, velocities）"""
        with self._data_lock:
            # print(self.latest_scan, self.latest_position, self.latest_heading)
            return (
                self.latest_scan,
                self.latest_position,
                self.latest_heading,
                self.latest_linear_velocity,
                self.latest_angular_velocity,
                self.latest_position_raw,  # 返回未添加噪声的原始position
            )
    
    def get_latest_imu(self):
        """线程安全地获取最新的IMU数据（orientation, linear_acceleration, angular_velocity）"""
        with self._data_lock:
            return (
                self.latest_imu_orientation,
                self.latest_imu_linear_acc,
                self.latest_imu_angular_vel,
            )


class ScanSubscriber(Node):
    def __init__(self, env_id=0, nodes_logger=None):
        super().__init__(f"scan_subscriber_env_{env_id}")
        self.get_logger().set_level(SEVERITY)
        self.env_id = env_id
        self.nodes_logger = nodes_logger
        self.subscriber_ = self.create_subscription(
            LaserScan, "scan", self.listener_callback, 1
        )
        self.latest_scan = None
        self.scan_timestamps = []  # 最近1秒内的消息时间戳列表（用于频率统计）
        self.scan_count = 0
        # 线程安全锁
        self._data_lock = threading.Lock()
        # 启动后台线程自动接收数据
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = None
        self._shutdown_flag = threading.Event()
        self._start_spin_thread()
    
    def _start_spin_thread(self):
        """启动后台线程自动接收ROS消息
        
        使用executor.spin()在后台线程中持续运行，当有消息到达时自动调用callback。
        这样主线程不会被阻塞，可以随时读取最新的传感器数据。
        """
        def spin_loop():
            try:
                # executor.spin()会持续运行，直到executor被shutdown
                # 当有消息到达时，会自动调用对应的callback函数
                self._executor.spin()
            except Exception as e:
                if not self._shutdown_flag.is_set():
                    self.get_logger().error(f"Error in spin thread: {e}")
        
        self._spin_thread = threading.Thread(target=spin_loop, daemon=True, name=f"scan_subscriber_spin_{self.env_id}")
        self._spin_thread.start()
    
    def shutdown(self):
        """停止后台线程并清理资源"""
        if self._shutdown_flag is not None:
            self._shutdown_flag.set()
        if self._executor is not None:
            self._executor.shutdown()
        if self._spin_thread is not None and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)
        self.destroy_node()

    def listener_callback(self, msg):
        current_time = time.time()
        self.scan_count += 1
        
        # 更新滑动窗口：添加当前时间戳，移除超过1秒的旧时间戳
        self.scan_timestamps.append(current_time)
        # 清理超过1秒的旧时间戳
        cutoff_time = current_time - 1.0
        self.scan_timestamps = [ts for ts in self.scan_timestamps if ts > cutoff_time]
        
        # 计算频率：统计最近1秒内的消息数量
        freq = len(self.scan_timestamps) if len(self.scan_timestamps) > 0 else 0.0
        
        # 线程安全更新数据
        with self._data_lock:
            self.latest_scan = msg.ranges[:]
        
        # 记录日志
        if self.nodes_logger is not None:
            scan_array = np.asarray(msg.ranges)
            scan_min = float(np.min(scan_array)) if scan_array.size > 0 else float('nan')
            scan_max = float(np.max(scan_array)) if scan_array.size > 0 else float('nan')
            freq_str = f"{freq:.2f}Hz" if freq > 0 else "0.00Hz"
            self.nodes_logger.log(
                self.env_id,
                f"[SUBSCRIBER] /scan received count={self.scan_count} freq={freq_str} "
                f"ranges_len={len(msg.ranges)} min={scan_min:.4f} max={scan_max:.4f}"
            )

    def get_latest_scan(self):
        # 线程安全读取数据
        with self._data_lock:
            return self.latest_scan


class OdomSubscriber(Node):
    def __init__(self, env_id=0, nodes_logger=None):
        super().__init__(f"odom_subscriber_env_{env_id}")
        self.get_logger().set_level(SEVERITY)
        self.env_id = env_id
        self.nodes_logger = nodes_logger
        self.subscriber_ = self.create_subscription(
            Odometry, "odom", self.listener_callback, 1
        )
        self.latest_position = None
        self.latest_heading = None
        self.odom_timestamps = []  # 最近1秒内的消息时间戳列表（用于频率统计）
        self.odom_count = 0
        # 线程安全锁
        self._data_lock = threading.Lock()
        # 启动后台线程自动接收数据
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = None
        self._shutdown_flag = threading.Event()
        self._start_spin_thread()
    
    def _start_spin_thread(self):
        """启动后台线程自动接收ROS消息
        
        使用executor.spin()在后台线程中持续运行，当有消息到达时自动调用callback。
        这样主线程不会被阻塞，可以随时读取最新的传感器数据。
        """
        def spin_loop():
            try:
                # executor.spin()会持续运行，直到executor被shutdown
                # 当有消息到达时，会自动调用对应的callback函数
                self._executor.spin()
            except Exception as e:
                if not self._shutdown_flag.is_set():
                    self.get_logger().error(f"Error in spin thread: {e}")
        
        self._spin_thread = threading.Thread(target=spin_loop, daemon=True, name=f"odom_subscriber_spin_{self.env_id}")
        self._spin_thread.start()
    
    def shutdown(self):
        """停止后台线程并清理资源"""
        if self._shutdown_flag is not None:
            self._shutdown_flag.set()
        if self._executor is not None:
            self._executor.shutdown()
        if self._spin_thread is not None and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)
        self.destroy_node()

    def listener_callback(self, msg):
        current_time = time.time()
        self.odom_count += 1
        
        # 更新滑动窗口：添加当前时间戳，移除超过1秒的旧时间戳
        self.odom_timestamps.append(current_time)
        # 清理超过1秒的旧时间戳
        cutoff_time = current_time - 1.0
        self.odom_timestamps = [ts for ts in self.odom_timestamps if ts > cutoff_time]
        
        # 计算频率：统计最近1秒内的消息数量
        freq = len(self.odom_timestamps) if len(self.odom_timestamps) > 0 else 0.0
        
        # 线程安全更新数据
        with self._data_lock:
            self.latest_position = msg.pose.pose.position
            self.latest_heading = msg.pose.pose.orientation
        
        # 记录日志
        if self.nodes_logger is not None:
            freq_str = f"{freq:.2f}Hz" if freq > 0 else "0.00Hz"
            pos = msg.pose.pose.position
            self.nodes_logger.log(
                self.env_id,
                f"[SUBSCRIBER] /odom received count={self.odom_count} freq={freq_str} "
                f"position=({pos.x:.4f},{pos.y:.4f},{pos.z:.4f})"
            )

    def get_latest_odom(self):
        # 线程安全读取数据
        with self._data_lock:
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
    def __init__(self, env_id=0, nodes_logger=None):
        super().__init__(f"set_entity_state_client_env_{env_id}")
        self.env_id = env_id  # 保存环境ID用于日志输出
        self.nodes_logger = nodes_logger
        self.get_logger().set_level(SEVERITY)
        self.client = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        while not self.client.wait_for_service(timeout_sec=1.0):
            if self.nodes_logger is not None:
                self.nodes_logger.log(self.env_id, "SetModelStateClient::wait_for_service")
            else:
                print(f"[ROS_env {self.env_id}] SetModelStateClient::wait_for_service")
            self.get_logger().info("Service not available, waiting again...")
        self.request = SetEntityState.Request()

    def set_state(self, name, new_pose):
        self.request.state.name = name
        self.request.state.pose = new_pose
        # 清零线速度、角速度，避免重置后残留速度导致落下或晃动
        self.request.state.twist.linear.x = 0.0
        self.request.state.twist.linear.y = 0.0
        self.request.state.twist.linear.z = 0.0
        self.request.state.twist.angular.x = 0.0
        self.request.state.twist.angular.y = 0.0
        self.request.state.twist.angular.z = 0.0
        self.request.state.reference_frame = "world"
        self.future = self.client.call_async(self.request)
        # 等待服务调用完成并返回结果
        rclpy.spin_until_future_complete(self, self.future)
        if self.future.result() is not None:
            response = self.future.result()
            return response.success
        return False


class GoalModelClient(Node):
    """用于在Gazebo中设置绿色圆柱体目标模型的位置（模型已在世界文件中定义）"""
    def __init__(self, env_id=0, model_name="goal_cylinder", nodes_logger=None):
        super().__init__(f"goal_model_client_env_{env_id}")
        self.env_id = env_id
        self.nodes_logger = nodes_logger
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
        if self.nodes_logger is not None:
            self.nodes_logger.log(self.env_id, "GoalModelClient::wait_for_service")
        self.get_logger().info("Waiting for Gazebo services to be available...")
        if not self.set_state_client.wait_for_service(timeout_sec=timeout):
            self.get_logger().warn(f"Service set_entity_state not available after {timeout}s, will retry later")
            return False
        
        self.services_ready = True
        if self.nodes_logger is not None:
            self.nodes_logger.log(self.env_id, "GoalModelClient::services_ready")
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
                if self.nodes_logger is not None:
                    self.nodes_logger.log(self.env_id, f"GoalModelClient::set_goal_position x={x:.3f} y={y:.3f} z={z:.3f} ok")
                self.get_logger().info(f"Goal cylinder set to position ({x:.3f}, {y:.3f}, {z:.3f})")
            else:
                error_msg = response.status_message if hasattr(response, 'status_message') else "Unknown error"
                if self.nodes_logger is not None:
                    self.nodes_logger.log(self.env_id, f"GoalModelClient::set_goal_position failed {error_msg}")
                self.get_logger().warn(f"Failed to set goal cylinder position: {error_msg}")
        else:
            if self.nodes_logger is not None:
                self.nodes_logger.log(self.env_id, "GoalModelClient::set_goal_position failed Service returned None")
            self.get_logger().warn(f"Failed to set goal cylinder position: Service returned None")


class CmdVelPublisher(Node):
    def __init__(self, env_id=0, nodes_logger=None):
        super().__init__(f"cmd_vel_publisher_env_{env_id}")
        self.get_logger().set_level(SEVERITY)
        self.env_id = env_id
        self.nodes_logger = nodes_logger
        self.publisher_ = self.create_publisher(Twist, "cmd_vel", 1)
        self.publish_last_time = None
        self.publish_count = 0

    def publish_cmd_vel(self, linear_velocity=0.0, angular_velocity=0.0):
        current_time = time.time()
        self.publish_count += 1
        
        # 计算频率
        freq = None
        if self.publish_last_time is not None:
            dt = current_time - self.publish_last_time
            if dt > 0:
                freq = 1.0 / dt
        self.publish_last_time = current_time
        
        twist_msg = Twist()
        # Set linear and angular velocities
        twist_msg.linear.x = float(linear_velocity)  # Example linear velocity (m/s)
        twist_msg.angular.z = float(
            angular_velocity
        )  # Example angular velocity (rad/s)
        self.publisher_.publish(twist_msg)
        
        # 记录日志
        if self.nodes_logger is not None:
            freq_str = f"{freq:.2f}Hz" if freq is not None else "N/A"
            self.nodes_logger.log(
                self.env_id,
                f"[PUBLISHER] /cmd_vel published count={self.publish_count} freq={freq_str} "
                f"linear_velocity={linear_velocity:.4f} angular_velocity={angular_velocity:.4f}"
            )


class MarkerPublisher(Node):
    def __init__(self, env_id=0, nodes_logger=None):
        super().__init__(f"marker_publisher_env_{env_id}")
        self.get_logger().set_level(SEVERITY)
        self.env_id = env_id
        self.nodes_logger = nodes_logger
        self.publisher = self.create_publisher(Marker, f"/env_{env_id}/visualization_marker", 1)
        self.publish_last_time = None
        self.publish_count = 0

    def publish(self, x, y):
        current_time = time.time()
        self.publish_count += 1
        
        # 计算频率
        freq = None
        if self.publish_last_time is not None:
            dt = current_time - self.publish_last_time
            if dt > 0:
                freq = 1.0 / dt
        self.publish_last_time = current_time
        
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
        
        # 记录日志
        if self.nodes_logger is not None:
            freq_str = f"{freq:.2f}Hz" if freq is not None else "N/A"
            self.nodes_logger.log(
                self.env_id,
                f"[PUBLISHER] /env_{self.env_id}/visualization_marker published count={self.publish_count} freq={freq_str} "
                f"position=({x:.4f},{y:.4f}) type=CYLINDER"
            )
        self.get_logger().info("Publishing Marker")


def run_scan(args=None):
    rclpy.init()
    reading_laser = ScanSubscriber()
    reading_laser.get_logger().info("Hello friend!")
    # subscriber在后台线程中自动接收数据，主线程只需等待
    try:
        while True:
            time.sleep(1.0)
            scan = reading_laser.get_latest_scan()
            if scan is not None:
                reading_laser.get_logger().info(f"Received scan with {len(scan)} points")
    except KeyboardInterrupt:
        reading_laser.get_logger().info("Shutting down...")
    finally:
        reading_laser.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    run_scan()
