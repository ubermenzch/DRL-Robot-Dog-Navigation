import time
from datetime import datetime
import rclpy
from ros_nodes import (
    ScanSubscriber,
    OdomSubscriber,
    ResetWorldClient,
    ResetSimulationClient,
    SetModelStateClient,
    CmdVelPublisher,
    MarkerPublisher,
    PhysicsClient,
    SensorSubscriber,
    GoalModelClient,
)
import numpy as np
from geometry_msgs.msg import Pose, Twist
from squaternion import Quaternion
import math
from scipy.spatial import KDTree
from scipy.spatial import distance


class ROS_env:
    def __init__(
        self,
        init_target_distance=2.0,
        target_dist_increase=0.01,
        max_target_dist=15.0,
        target_reached_delta=0.3,
        collision_delta=0.25,
        args=None,
        neglect_angle = 30, # 忽略的视野角度（单位度）
        scan_range = 4.5,
        world_size = 10, # 单位
        obs_min_dist = 0,  # 障碍物圆心最小距离（单位米），用于约束障碍物/机器人/目标之间的最小间距
        obs_num = 8, # 默认8
        env_id = 0,  # 环境ID，用于topic命名空间
        # 障碍物生成方式
        obs_distribution_mode="uniform",  # "uniform"=均匀分布（当前默认策略），"random"=完全随机分布
        # 地图 / costmap 参数
        costmap_resolution=0.3,  # costmap 分辨率（米/格）
        obstacle_size=0.3,       # 障碍物在 costmap 中的等效边长（米，默认为正方形）
        # 机器人参数
        max_velocity=1.0,  # 线速度最大值（用于惩罚归一化）
        localization_noise_stddev=0.0,  # 定位噪声标准差（米）
        # 奖励函数参数
        goal_reward=1000.0,  # 到达目标的奖励
        collision_penalty_base=-1000.0,  # 碰撞惩罚系数
        angle_penalty_base=0.0,  # 角度偏差基础惩罚
        linear_penalty_base=-1.0,  # 线速度基础惩罚
        yawrate_penalty_base=0.0,  # 角速度惩罚系数（负值惩罚，0为关闭效果）
        enable_progress_reward=False,  # 是否启用进度奖惩
        progress_reward_base=1.0,  # 进度奖惩基础系数
        # 奖励/惩罚开关
        enable_obs_penalty=True,  # 是否启用障碍物距离惩罚
        enable_yawrate_penalty=True,  # 是否启用角速度惩罚
        enable_angle_penalty=True,  # 是否启用角度偏移惩罚
        enable_linear_penalty=True,  # 是否启用线速度惩罚
        enable_step_penalty=False,  # 是否启用时间步常数惩罚
        enable_target_distance_penalty=False,  # 是否启用终点距离惩罚
        enable_linear_acceleration_oscillation_penalty=False,  # 是否启用线速度加速度震荡惩罚
        enable_yawrate_oscillation_penalty=False,  # 是否启用角速度震荡惩罚
        # 障碍物距离惩罚参数
        obs_penalty_threshold=1.0,  # 障碍物距离惩罚阈值（米），低于此值开始惩罚；设为-1时根据速度自动计算（|v| * sim_time）
        min_obs_penalty_threshold=0.5,  # 动态计算阈值时的最小阈值下限，避免阈值过小
        obs_penalty_base=-10.0,  # 障碍物距离惩罚基础系数
        obs_penalty_power=2.0,  # 障碍物距离惩罚指数，值越大惩罚增长越快
        obs_penalty_high_weight=1.0,  # 中间高权重区域惩罚权重
        obs_penalty_low_weight=0.5,  # 两侧低权重区域惩罚权重
        obs_penalty_middle_ratio=0.4,  # 中间高权重区域比例（0-1）
        # 终点距离惩罚参数
        target_distance_penalty_base=-1.0,  # 终点距离惩罚基础系数（负值表示惩罚）
        # 时间步惩罚参数
        step_penalty_base=0.0,  # 每个step的固定惩罚（负值为惩罚，0为关闭效果）
        # 震荡惩罚参数
        linear_acceleration_oscillation_penalty_base=-1.0,  # 线速度加速度震荡惩罚基础系数（负值表示惩罚）
        yawrate_oscillation_penalty_base=-1.0,  # 角速度震荡惩罚基础系数（负值表示惩罚）
        # 奖励归一化参数
        reward_scale=1.0,  # 奖励缩放因子，用于控制每步整体奖励大小；1.0表示不缩放，0.1表示缩小10倍，10.0表示放大10倍
        # 连通区域选择参数
        region_select_bias=1.0,  # 连通区域选择概率：1.0=总是选最大连通区域；0.8=80%概率选最大区域，否则从其余区域随机选
        # 时间控制参数
        sim_time=0.1,  # 仿真步长，用于基于速度的动态阈值等
        step_sleep_time=0.1,  # step方法中的sleep时间（秒）
        reset_step_count=3,  # reset方法中调用step的次数
        # 地图复用参数
        goals_per_map=1,  # 每张地图的目标点数量（即一张地图可以用来产生多少个episode）
        # 传感器频率限制参数（Hz），0或负数表示不限制
        scan_max_freq=0.0,  # 激光雷达最大处理频率
        odom_max_freq=0.0,  # 里程计最大处理频率
        imu_max_freq=0.0,  # IMU最大处理频率
        # 传感器日志开关
        scan_enable_log=False,  # 是否记录激光雷达日志
        odom_enable_log=False,  # 是否记录里程计日志
        imu_enable_log=False,  # 是否记录IMU日志
        # 调试参数
        env_logger=None,  # 环境调试日志；若设置，所有环境调试信息写入 env_log_<timestamp>.log（含时间戳、env_id）
        reward_logger=None,  # 奖励调试日志；若设置，所有奖励信息写入 reward_log_<timestamp>.log（含时间戳、env_id）
        nodes_logger=None,  # ROS节点调试日志；若设置，所有ros_nodes.py中的调试信息写入 nodes_log_<timestamp>.log（含时间戳、env_id）
    ):
        # 记录初始化阶段步骤，便于定位卡点
        def _log(msg):
            print(f"[ROS_env {env_id}] {msg}")

        # 下面这些初始化阶段的日志在稳定运行时意义不大，且会刷屏，
        # 因此关闭默认打印，如需调试可临时取消注释。
        # _log("初始化 rclpy ...")
        rclpy.init(args=args)
        # _log("rclpy 初始化完成，开始创建ROS接口...")
        # 预先保存 scan_range，以便下方 SensorSubscriber 使用
        self.scan_range = scan_range
        self.env_id = env_id
        self.env_logger = env_logger
        self.reward_logger = reward_logger
        self.cmd_vel_publisher = CmdVelPublisher(env_id, nodes_logger=nodes_logger)
        self.robot_state_publisher = SetModelStateClient(env_id, nodes_logger=nodes_logger)
        self.world_reset = ResetWorldClient(env_id)
        # 在极端异常情况下用于整体重置仿真（/reset_simulation）
        try:
            self.reset_simulation_client = ResetSimulationClient(env_id)
        except Exception as e:
            # 若 /reset_simulation 不可用，不影响正常运行，仅在需要时记录日志
            if nodes_logger is not None:
                nodes_logger.log(env_id, f"[ResetSimulationClient] 初始化失败: {e}")
            self.reset_simulation_client = None
        self.physics_client = PhysicsClient(env_id)
        self.publish_target = MarkerPublisher(env_id, nodes_logger=nodes_logger)
        self.goal_model_client = GoalModelClient(env_id, nodes_logger=nodes_logger)
        self.sensor_subscriber = SensorSubscriber(
            env_id,
            scan_range=self.scan_range,
            localization_noise_stddev=localization_noise_stddev,
            nodes_logger=nodes_logger,
            scan_max_freq=scan_max_freq,
            odom_max_freq=odom_max_freq,
            imu_max_freq=imu_max_freq,
            scan_enable_log=scan_enable_log,
            odom_enable_log=odom_enable_log,
            imu_enable_log=imu_enable_log,
        )
        self.target_dist = init_target_distance
        self.target_dist_increase = target_dist_increase
        self.max_target_dist = max_target_dist
        self.target_reached_delta = target_reached_delta
        self.collision_delta = collision_delta
        self.step_count = 0
        self.neglect_angle = neglect_angle
        self.scan_range = scan_range
        self.world_size = world_size  # 单位米
        self.obs_min_dist = obs_min_dist  # 障碍物圆心最小距离（单位米）
        self.obs_num  = obs_num
        # 障碍物生成方式配置
        self.obs_distribution_mode = obs_distribution_mode
        # costmap 相关
        self.costmap_resolution = costmap_resolution
        self.obstacle_size = obstacle_size
        self.map_half = self.world_size / 2.0
        # costmap 的原点（左下角）定义为 (-map_half, -map_half)
        self.map_origin_x = -self.map_half
        self.map_origin_y = -self.map_half
        self.grid_width = int(np.ceil(self.world_size / self.costmap_resolution))
        self.grid_height = int(np.ceil(self.world_size / self.costmap_resolution))
        self.costmap = None  # 后续按需初始化
        # 障碍物位姿缓存：每个元素为 [x, y, yaw]，用于 costmap 和写回 Gazebo
        self.obstacle_poses = []  # [[x, y, yaw], ...]
        self.max_velocity = max_velocity
        self.target = None
        self.episode_start_position = None  # 记录每个episode开始时的机器人位置
        self.initial_target_distance = None  # 记录每个episode开始时的终点距离
        # 奖励函数参数
        self.goal_reward = goal_reward
        self.collision_penalty_base = collision_penalty_base
        self.angle_penalty_base = angle_penalty_base
        self.linear_penalty_base = linear_penalty_base
        self.yawrate_penalty_base = yawrate_penalty_base
        self.enable_progress_reward = enable_progress_reward
        self.progress_reward_base = progress_reward_base
        # 奖励/惩罚开关
        self.enable_obs_penalty = enable_obs_penalty
        self.enable_yawrate_penalty = enable_yawrate_penalty
        self.enable_angle_penalty = enable_angle_penalty
        self.enable_linear_penalty = enable_linear_penalty
        self.enable_step_penalty = enable_step_penalty
        self.enable_target_distance_penalty = enable_target_distance_penalty
        self.enable_linear_acceleration_oscillation_penalty = enable_linear_acceleration_oscillation_penalty
        self.enable_yawrate_oscillation_penalty = enable_yawrate_oscillation_penalty
        # 障碍物距离惩罚参数
        self.obs_penalty_threshold = obs_penalty_threshold
        self.obs_penalty_base = obs_penalty_base
        self.obs_penalty_power = obs_penalty_power
        self.obs_penalty_high_weight = obs_penalty_high_weight
        self.obs_penalty_low_weight = obs_penalty_low_weight
        self.min_obs_penalty_threshold = min_obs_penalty_threshold
        # 确保比例在合理范围
        self.obs_penalty_middle_ratio = np.clip(obs_penalty_middle_ratio, 0.0, 1.0)
        # 终点距离惩罚参数
        self.target_distance_penalty_base = target_distance_penalty_base
        # 时间步惩罚参数
        self.step_penalty_base = step_penalty_base
        # 震荡惩罚参数
        self.linear_acceleration_oscillation_penalty_base = linear_acceleration_oscillation_penalty_base
        self.yawrate_oscillation_penalty_base = yawrate_oscillation_penalty_base
        # 奖励归一化参数
        self.reward_scale = reward_scale
        # 记录上一step的线速度、角速度和加速度（用于震荡惩罚计算）
        self.prev_linear_velocity = None
        self.prev_angular_velocity = None
        self.prev_linear_acceleration = None
        # 记录上一step与终点的距离（用于进度奖惩计算）
        self.prev_distance_to_goal = None
        # 连通区域选择概率（内部按 [0,1] 裁剪）
        self.region_select_bias = region_select_bias
        # 时间控制参数
        self.sim_time = sim_time
        self.step_sleep_time = step_sleep_time
        self.reset_step_count = reset_step_count
        # 地图复用参数
        self.goals_per_map = goals_per_map
        # 为了避免首次 reset 时 obstacle_poses 为空导致 apply_obstacle_poses n=0，
        # 初始化时将计数置为 goals_per_map，强制第一次 reset 进入“重新生成地图/障碍物”分支。
        self.goals_count_for_current_map = int(goals_per_map)  # 当前地图已使用的目标点数量
        self.generated_map_count = 0  # 已生成的地图数量（仅在真正重新生成地图时递增）
        # 奖励分解统计（按 episode 累积）
        # _log("ROS接口与参数初始化完成，准备reset...")
        self.reset_episode_reward_breakdown()
        # for i in range(60):
        #     self.step(empty_step=True)
        self._env_log("ROS环境初始化完成")
        # self.reset()
        # _log("环境初始化完成")

    def _env_log(self, msg):
        """环境调试输出：若有 env_logger 则始终写入 env_log。"""
        if self.env_logger is not None:
            self.env_logger.log(self.env_id, msg)

    def reset_episode_reward_breakdown(self):
        """重置当前 episode 的奖励分解统计"""
        # 记录“本step”的奖励分量（供训练侧直接读取，避免用episode累计值做差分）
        # 结构：
        #   self.last_step_reward_parts = {"raw": {...}, "scaled": {...}}
        # raw：缩放前；scaled：乘以 reward_scale 后
        self.last_step_reward_parts = {"raw": {}, "scaled": {}}
        # 重置上一step的记录（用于震荡惩罚计算）
        self.prev_linear_velocity = None
        self.prev_angular_velocity = None
        self.prev_linear_acceleration = None


    def step(self, lin_velocity=0.0, ang_velocity=0.0, empty_step=False, log_terminal=True):
        """执行一步仿真。

        注意：该方法不再返回 last_action（动作由调用方自行维护）。
        empty_step=True 时仅发送零速度指令并更新一次传感器缓存，不采样状态或计算奖励。
        log_terminal=True 时记录 step terminal 日志（用于正常 episode 收集），False 时不记录（用于 reset 阶段采样初始状态）。
        """
        if empty_step:
            self.cmd_vel_publisher.publish_cmd_vel(0.0, 0.0)
            time.sleep(self.step_sleep_time)
            # subscriber在后台线程中自动接收数据，无需手动spin_once
            # 返回结构与非 empty_step 一致，便于上层统一处理（调用方通常会忽略 empty_step 的返回值）
            return None, None, None, None, None, False, False, 0.0, 0.0, 0.0

        self.cmd_vel_publisher.publish_cmd_vel(lin_velocity, ang_velocity)
        # 物理模拟的unpause/pause现在在episode级别控制，不在每个step中控制
        self.step_count+=1
        time.sleep(self.step_sleep_time)
        # subscriber在后台线程中自动接收数据，无需手动spin_once

        (
            latest_scan,
            latest_position,
            latest_orientation,
            current_linear_velocity,
            current_angular_velocity,
            position_raw,  # 未添加噪声的 position，用于计算 distance_raw
        ) = self.sensor_subscriber.get_latest_sensor()
        # 等待直到收到有效的雷达数据（或超时）
        if latest_scan is None:
            max_wait_sec = 2.0
            wait_start = time.time()
            while latest_scan is None and (time.time() - wait_start) < max_wait_sec:
                # subscriber在后台线程中自动接收数据，只需短暂等待后重试读取
                time.sleep(0.1)
                (
                    latest_scan,
                    latest_position,
                    latest_orientation,
                    current_linear_velocity,
                    current_angular_velocity,
                    position_raw,
                ) = self.sensor_subscriber.get_latest_sensor()
            if latest_scan is None:
                raise RuntimeError(
                    f"[ROS_env {self.env_id}] No laser scan data received after {max_wait_sec}s. "
                    f"Please check /scan topic and Gazebo sensor plugins."
                )
        latest_scan = np.array(latest_scan) 
        # print("latest_scan_len:",len(latest_scan))
        # 裁剪掉忽略的视野
        neglect_scan = int(np.ceil((self.neglect_angle/360)*len(latest_scan)))
        latest_scan = latest_scan[neglect_scan:len(latest_scan)-neglect_scan]
        #print(f" Laser scan data: {latest_scan}")
        distance, cos, sin, _ = self.get_dist_sincos(
            latest_position, latest_orientation
        )
        distance_raw, _, _, _ = self.get_dist_sincos(
            position_raw, latest_orientation
        )
        # 若初始化prev_distance_to_goal为None，则用当前距离初始化
        if self.prev_distance_to_goal is None:
            self.prev_distance_to_goal = distance
        collision = self.check_collision(latest_scan)
        goal = self.check_target(distance, collision)
        action = [lin_velocity, ang_velocity]
        reward = self.get_reward(goal, collision, action, latest_scan, distance, cos, sin)

        # 在 step 中统一更新上一距离，用于下一步的进度奖惩计算
        if np.isfinite(distance):
            # 若之前未初始化，则直接用当前距离作为起点；否则也更新为当前距离
            self.prev_distance_to_goal = distance

        if goal or collision:
            if log_terminal and self.env_logger is not None:
                self.env_logger.log(self.env_id, f"step terminal step={self.step_count} goal={goal} collision={collision} distance={distance:.4f} reward={reward:.4f}")

        return (
            latest_scan,
            distance,
            float(distance_raw),
            cos,
            sin,
            collision,
            goal,
            reward,
            float(current_linear_velocity),
            float(current_angular_velocity),
        )

    def _find_best_robot_target_combination(self, region_mask):
        """在给定的连通区域内一次性尝试所有可能的机器人-目标位置组合，找到满足距离约束的组合
        
        优化策略：
        1. 使用采样策略避免在超大区域中遍历所有组合
        2. 直接计算世界坐标距离进行判断
        
        Args:
            region_mask: 连通区域的mask（bool数组）
            
        Returns:
            (robot_pos, target_pos) 如果找到满足条件的组合，否则返回 (None, None)
        """
        indices = np.argwhere(region_mask)
        if indices.size == 0:
            if self.env_logger is not None:
                self.env_logger.log(self.env_id, "_find_best_robot_target_combination: 区域内没有有效索引")
            return None, None
        
        min_dist = self.target_reached_delta + 0.1
        max_dist = self.target_dist
        
        if self.env_logger is not None:
            self.env_logger.log(self.env_id, f"_find_best_robot_target_combination: 开始搜索，区域内有效格点数={len(indices)}, "
                      f"距离约束=[{min_dist:.2f}, {max_dist:.2f}]m")
        
        # 收集所有满足距离约束的组合
        valid_combinations = []
        
        # 如果区域太大，使用采样策略而不是完全遍历
        max_samples = 20  # 最大采样组合数，避免在超大区域中搜索过久
        total_combinations = len(indices) * len(indices)
        use_sampling = len(indices) > max_samples
        
        if use_sampling:
            # 采样策略：随机选择机器人位置，对每个机器人位置尝试所有可能的目标位置
            # 采样数量随indices增大而增大，确保在大区域中也能充分采样
            num_robot_samples = min(len(indices), max_samples)
            robot_indices = np.random.choice(len(indices), size=num_robot_samples, replace=False)
            if self.env_logger is not None:
                self.env_logger.log(self.env_id, f"_find_best_robot_target_combination: "
                          f"区域过大（{total_combinations}个组合），使用采样策略，随机选择{num_robot_samples}个机器人位置")
        else:
            robot_indices = range(len(indices))
        
        # 遍历选定的机器人位置
        for ridx in robot_indices:
            ry, rx = indices[ridx]
            
            if not use_sampling:
                if self.env_logger is not None:
                    self.env_logger.log(self.env_id, f"_find_best_robot_target_combination: "
                          f"处理机器人位置 [{ridx+1}/{len(indices)}] - 网格=({rx}, {ry})")
            
            robot_x, robot_y = self._grid_to_world_center(rx, ry)
            
            # 对于当前机器人位置，遍历所有可能的目标位置
            for tidx in range(len(indices)):
                ty, tx = indices[tidx]
                
                # 计算世界坐标距离
                target_x_center, target_y_center = self._grid_to_world_center(tx, ty)
                dist = np.linalg.norm([target_x_center - robot_x, target_y_center - robot_y])
                
                # 距离检查
                if min_dist <= dist <= max_dist:
                    valid_combinations.append({
                        'robot_pos': (robot_x, robot_y),
                        'target_grid': (tx, ty),
                        'dist': dist
                    })
        
        if len(valid_combinations) == 0:
            if self.env_logger is not None:
                self.env_logger.log(self.env_id, "_find_best_robot_target_combination: 未找到满足距离约束的组合")
            return None, None
        
        # 找到最大距离
        max_distance = max(combo['dist'] for combo in valid_combinations)
        
        # 筛选出距离等于最大距离的所有组合（最远的组合）
        farthest_combinations = [combo for combo in valid_combinations if combo['dist'] == max_distance]
        
        if self.env_logger is not None:
            self.env_logger.log(self.env_id, f"_find_best_robot_target_combination: "
                  f"找到{len(valid_combinations)}个有效组合，最大距离={max_distance:.2f}m, "
                  f"最远组合数={len(farthest_combinations)}")
        
        # 从最远的组合中随机选择一个
        selected = farthest_combinations[np.random.randint(len(farthest_combinations))]
        
        if self.env_logger is not None:
            self.env_logger.log(self.env_id, f"_find_best_robot_target_combination: "
                  f"选中组合 - 机器人位置=({selected['robot_pos'][0]:.2f}, {selected['robot_pos'][1]:.2f}), "
                  f"目标网格=({selected['target_grid'][0]}, {selected['target_grid'][1]}), "
                  f"距离={selected['dist']:.2f}m")
        
        # 在选定的格子的范围内随机生成终点位置
        tx, ty = selected['target_grid']
        target_x, target_y = self._grid_to_world_random(tx, ty)
        
        if self.env_logger is not None:
            self.env_logger.log(self.env_id, f"_find_best_robot_target_combination: "
                  f"最终目标位置=({target_x:.2f}, {target_y:.2f})")
        
        return selected['robot_pos'], (target_x, target_y)

    def reset_with_current_map(self):
        """使用当前地图（障碍物保持不变）重新采样机器人和目标位置
        
        每次调用都会：
        1. 重新构建costmap
        2. 以region_select_bias概率选择最大连通区域，否则随机选择其他连通区域
        3. 在选定的连通区域内一次性尝试所有可能的机器人-目标位置组合，找到满足距离约束的最佳组合
        """
        
        # 重新选择连通区域（以region_select_bias概率选择最大连通区域）
        region_mask = self._select_free_region_by_probability(self.region_select_bias)
        
        if region_mask is None or not region_mask.any():
            return False
        
        # 一次性尝试所有可能的机器人-目标位置组合
        self._env_log("reset_with_current_map: 开始查找最佳机器人-目标位置组合")
        robot_pos, target_pos = self._find_best_robot_target_combination(region_mask)
        
        if robot_pos is None or target_pos is None:
            self._env_log("reset_with_current_map: 未找到有效的机器人-目标位置组合，返回False")
            return False
        
        self._env_log(f"reset_with_current_map: 成功找到机器人位置={robot_pos}, 目标位置={target_pos}")
        
        # 设置机器人位置
        rx, ry = robot_pos
        tx, ty = target_pos
        angle = np.random.uniform(-np.pi, np.pi)
        self.set_position("turtlebot3_waffle", rx, ry, angle)
        
        self.episode_start_position = [rx, ry]
        self.target = [tx, ty]
        
        return True

    def reset(self, force_regenerate_map=False, use_reset_simulation=False):
        """重置环境
        
        Args:
            force_regenerate_map: 如果为True，强制重新生成地图（将map_count+1，goal_count置为0）
            use_reset_simulation: 如果为True，则优先调用 /reset_simulation，而不是 /reset_world
        """
        # 根据参数选择 reset_simulation 或 reset_world
        if use_reset_simulation and getattr(self, "reset_simulation_client", None) is not None:
            self._env_log("[reset] 使用 /reset_simulation 重置仿真（不调用 /reset_world）")
            try:
                self.reset_simulation_client.reset_simulation()
            except Exception as e:
                self._env_log(f"[reset] 调用 /reset_simulation 失败: {e}，回退到 /reset_world")
                self.world_reset.reset_world()
        else:
            self.world_reset.reset_world()
        # time.sleep(self.step_sleep_time*25)
        # 使用 repeated empty_step + IMU / 里程计数据，确保机器人完全静止且姿态与地面水平
        settle_start_time = time.time()
        max_settle_time = 5.0  # 防止极端情况下死循环
        robot_settled = False
        # 缓存最近一次“静止检测”相关的数值，便于超时时打印原因（对齐多环境 env_log 诊断风格）
        last_lin_vel = None
        last_ang_vel = None
        last_lin_acc_xy = None
        last_roll = None
        last_pitch = None
        last_lin_vel_ok = None
        last_ang_vel_ok = None
        last_lin_acc_ok = None
        last_attitude_ok = None
        while not robot_settled:
            # 检查超时
            elapsed = time.time() - settle_start_time
            if elapsed > max_settle_time:
                # 组装超时原因（仅包含未满足项）
                reasons = []
                try:
                    # 线速度/角速度/加速度/姿态：与下方判定条件保持一致
                    if last_lin_vel is not None and last_lin_vel_ok is False:
                        reasons.append(f"vel超时({last_lin_vel:.4f})")
                    if last_ang_vel is not None and last_ang_vel_ok is False:
                        reasons.append(f"ang超时({last_ang_vel:.4f})")
                    if last_lin_acc_xy is not None and last_lin_acc_ok is False:
                        reasons.append(f"acc超时({last_lin_acc_xy:.4f})")
                    if last_roll is not None and last_pitch is not None and last_attitude_ok is False:
                        reasons.append(f"attitude超时(roll={last_roll:.4f}, pitch={last_pitch:.4f})")
                except Exception:
                    pass

                reason_str = "、".join(reasons) if reasons else "unknown(传感器数据不足或未更新)"
                self._env_log(
                    f"[reset 超时] 静止检测超时 (>{max_settle_time}s，实际 {elapsed:.2f}s)，跳出等待。 "
                    f"超时原因: {reason_str}。 "
                    f"settled={robot_settled} "
                    f"lin_vel_ok={last_lin_vel_ok} ang_vel_ok={last_ang_vel_ok} "
                    f"lin_acc_ok={last_lin_acc_ok} attitude_ok={last_attitude_ok}"
                )
                # 超时时尝试调用 /reset_simulation，对Gazebo进行整体重置
                try:
                    if getattr(self, "reset_simulation_client", None) is not None:
                        self._env_log("[reset 超时] 调用 /reset_simulation 以强制重置仿真")
                        self.reset_simulation_client.reset_simulation()
                    else:
                        self._env_log("[reset 超时] reset_simulation_client 未初始化，无法调用 /reset_simulation")
                except Exception as e:
                    self._env_log(f"[reset 超时] 调用 /reset_simulation 失败: {e}")
                return (False, None, None, None, None, None, None, None, None, None, None, None)
            
            # 发送零速度指令，让机器人逐渐停稳
            self.step(lin_velocity=0.0, ang_velocity=0.0, empty_step=True)

            (
                latest_scan,
                latest_position,
                latest_orientation,
                current_linear_velocity,
                current_angular_velocity,
                _position_raw,
            ) = self.sensor_subscriber.get_latest_sensor()

            # 从 IMU 读取线加速度、角速度和姿态（线程安全）
            imu_orientation, imu_lin_acc, imu_ang_vel = self.sensor_subscriber.get_latest_imu()

            # 传感器数据尚未准备好，继续等待
            if (
                latest_scan is None
                or latest_position is None
                or latest_orientation is None
                or imu_lin_acc is None
                or imu_ang_vel is None
                or imu_orientation is None
            ):
                continue

            # 由 IMU 姿态计算 roll / pitch，用于判断相机视角是否与地面近似水平
            imu_quat = Quaternion(
                imu_orientation.w,
                imu_orientation.x,
                imu_orientation.y,
                imu_orientation.z,
            )
            roll, pitch, _ = imu_quat.to_euler(degrees=False)

            # 检查各项静止条件
            lin_vel_ok = abs(current_linear_velocity) < 1e-2
            ang_vel_ok = abs(current_angular_velocity) < 1e-2
            lin_acc_xy = math.hypot(float(imu_lin_acc.x), float(imu_lin_acc.y))
            lin_acc_ok = abs(lin_acc_xy) < 1e-1
            attitude_ok = abs(roll) < 0.02 and abs(pitch) < 0.02

            # 更新缓存：用于超时诊断
            last_lin_vel = float(current_linear_velocity)
            last_ang_vel = float(current_angular_velocity)
            last_lin_acc_xy = float(lin_acc_xy)
            last_roll = float(roll)
            last_pitch = float(pitch)
            last_lin_vel_ok = bool(lin_vel_ok)
            last_ang_vel_ok = bool(ang_vel_ok)
            last_lin_acc_ok = bool(lin_acc_ok)
            last_attitude_ok = bool(attitude_ok)

            robot_settled = lin_vel_ok and ang_vel_ok and lin_acc_ok and attitude_ok

        if not robot_settled:
            self._env_log("[ERROR] Robot not settled during reset")
            return (False, None, None, None, None, None, None, None, None, None, None, None)
        
        # 缓存当前障碍物位姿，失败时可恢复
        old_obstacle_poses = [p[:] for p in self.obstacle_poses]

        # 先完成所有可能失败的操作，全部成功后再统一修改计数器
        # 如果强制重新生成地图，将goals_count_for_current_map置为goals_per_map，确保触发重新生成
        if force_regenerate_map:
            self.goals_count_for_current_map = self.goals_per_map
            self._env_log(f"[强制重新生成地图] force_regenerate_map=True，将goals_count_for_current_map设置为{self.goals_per_map}")
        should_regenerate_map = (self.goals_count_for_current_map >= self.goals_per_map)
        
        if should_regenerate_map:
            if not self.generate_and_set_obstacles():
                self._env_log("[ERROR] Failed to generate and set obstacles during reset")
                # 恢复旧障碍物位姿与 costmap
                self.obstacle_poses = [p[:] for p in old_obstacle_poses]
                self._build_costmap_from_obstacles()
                return (False, None, None, None, None, None, None, None, None, None, None, None)
        else:
            # 不重新生成地图：reset_world() 会把 Gazebo 中的障碍物恢复到初始位姿
            # 这里用缓存的 obstacle_poses 重新设置一次障碍物，
            # 等效于“地图不变”，只是把 Gazebo 世界恢复到当前地图状态
            if not self.apply_obstacle_poses_to_gazebo():
                self._env_log("[ERROR] Failed to restore obstacle poses during reset")
                return (False, None, None, None, None, None, None, None, None, None, None, None)
        
        if not self._spawn_robot_and_target_from_current_costmap():
            self._env_log("[WARNING] Failed to spawn robot and target during reset")
            return (False, None, None, None, None, None, None, None, None, None, None, None)
        
        # 采样初始状态并检查碰撞
        # 使用 log_terminal=False 避免 reset 失败时的 step 被误计入 episode 统计
        self.prev_distance_to_goal = None
        (
            latest_scan,
            distance,
            distance_raw,
            cos,
            sin,
            collision,
            goal,
            reward,
            current_linear_velocity,
            current_angular_velocity,
        ) = self.step(lin_velocity=0, ang_velocity=0, empty_step=False, log_terminal=False)
        
        if collision:
            self._env_log("[ERROR] Collision detected during reset")
            return (False, None, None, None, None, None, None, None, None, None, None, None)

        def _all_finite(x) -> bool:
            """递归检查对象中是否存在 NaN/Inf。"""
            if x is None:
                return True
            # numpy / 标量
            try:
                arr = np.asarray(x)
                if arr.dtype.kind in {"f", "c"}:  # float/complex
                    return bool(np.isfinite(arr).all())
                # 非浮点（如 bool/int）默认视为有限
                return True
            except Exception:
                pass

            # 容器递归
            if isinstance(x, dict):
                return all(_all_finite(v) for v in x.values())
            if isinstance(x, (list, tuple)):
                return all(_all_finite(v) for v in x)

            # 兜底：python 数值
            try:
                xf = float(x)
                return math.isfinite(xf)
            except Exception:
                # 无法判定的类型：保守认为通过（避免因日志对象等导致 reset 失败）
                return True

        # 对关键返回数据做 NaN/Inf 检查
        _reset_payload = (
            latest_scan,
            distance,
            distance_raw,
            cos,
            sin,
            collision,
            goal,
            reward,
            float(current_linear_velocity),
            float(current_angular_velocity),
        )
        if not _all_finite(_reset_payload):
            self._env_log("[ERROR] NaN/Inf detected during reset payload validation")
            return (False, None, None, None, None, None, None, None, None, None, None, None)

        # 所有操作成功，更新计数器
        if should_regenerate_map:
            self.generated_map_count += 1
            self.goals_count_for_current_map = 1
            if force_regenerate_map:
                self._env_log(f"[强制重新生成地图完成] generated_map_count={self.generated_map_count}, goals_count_for_current_map重置为1")
        else:
            self.goals_count_for_current_map += 1

        # 初始化episode相关状态
        self.initial_target_distance = distance_raw
        self.step_count = 0
        self.reset_episode_reward_breakdown()
        self._env_log(f"reset done distance={distance:.4f} cos={cos:.4f} sin={sin:.4f} collision={collision} goal={goal} reward={reward:.4f}")
        
        last_action = [0.0, 0.0]
        return (
            True,
            latest_scan,
            distance,
            distance_raw,
            cos,
            sin,
            collision,
            goal,
            last_action,
            reward,
            float(current_linear_velocity),
            float(current_angular_velocity),
        )


    def generate_obstacle_pose(self, name):
        """根据配置选择障碍物生成方式（仅生成障碍物位姿，不写入 Gazebo）：
        - 均匀分布（默认）
        - 随机分布（obs_distribution_mode=\"random\"）

        注意：会使用 obs_min_dist 约束障碍物之间的最小圆心距离。
        """
        # 根据障碍物实际边长和最小间距，确定安全偏移范围（避免障碍物贴边或相互过近）
        half_size = self.obstacle_size / 2.0
        bias = self.world_size/2 - max(self.obs_min_dist/2, half_size)
        angle = np.random.uniform(-np.pi, np.pi)

        # 如果配置为随机分布，则直接使用纯随机采样逻辑
        mode = getattr(self, "obs_distribution_mode", "uniform")
        if str(mode).lower() == "random":
            return self.fallback_random_position(name, bias, angle)

        # ==== 默认：改进的障碍物均匀分布策略 ====
        # 使用改进的 Farthest Point 采样实现均匀分布
        if not hasattr(self, 'candidate_points'):
            self.generate_uniform_candidates(bias)
        
        attempts = 0
        max_attempts = len(self.candidate_points)  # 最多尝试所有候选点
        
        while attempts < max_attempts:
            if not self.candidate_points:
                break
                
            # 选择离已有障碍物最远的点（只考虑障碍物，不包含机器人/终点）
            if self.obstacle_poses:
                existing_points = np.array([[p[0], p[1]] for p in self.obstacle_poses])
                candidate_array = np.array(self.candidate_points)
                
                # 计算所有候选点到最近已有障碍物的距离
                dists = distance.cdist(candidate_array, existing_points).min(axis=1)
                
                # 选择距离最大的点
                selected_idx = np.argmax(dists)
                x, y = self.candidate_points.pop(selected_idx)
            else:
                # 第一个点随机选择
                idx = np.random.randint(len(self.candidate_points))
                x, y = self.candidate_points.pop(idx)
            
            # 使用 obs_min_dist 约束：保证当前候选点与已有障碍物的最小圆心距离不小于 obs_min_dist
            if self.check_position(x, y, self.obs_min_dist):
                # 只在内存中记录障碍物信息，暂不写入 Gazebo
                self.obstacle_poses.append([x, y, angle])
                return True
            
            attempts += 1
        
        # 如果候选点用完或全部尝试失败：均匀分布模式下直接返回失败，
        # 由上层逻辑决定是否整体重试均匀分布，而不是退回到随机分布
        return False

    # ==================== costmap / 连通区域相关工具函数 ====================

    def _init_costmap(self):
        """初始化 costmap（False=空闲，True=占用）

        注意：不再强制将外圈一圈设置为障碍物，完全由障碍物和世界边界本身决定可行区域。
        """
        # 初始全部为空闲格子，由障碍物标记占用
        self.costmap = np.zeros((self.grid_height, self.grid_width), dtype=bool)

    def _world_to_grid(self, x, y):
        """世界坐标 -> 栅格索引 (ix, iy)，超出范围则返回 None"""
        ix = int(np.floor((x - self.map_origin_x) / self.costmap_resolution))
        iy = int(np.floor((y - self.map_origin_y) / self.costmap_resolution))
        if ix < 0 or ix >= self.grid_width or iy < 0 or iy >= self.grid_height:
            return None
        return ix, iy

    def _mark_obstacle_on_costmap(self, cx, cy):
        """根据障碍物中心和边长，在 costmap 上标记占用栅格

        规则（已按你提出的“只要遮住一点就算占用”实现）：
        - 只要障碍物的外接正方形与某个栅格有任意面积重叠，该栅格就视为占用。
        - 不再仅以“格子中心在障碍物范围内”为准。
        """
        if self.costmap is None:
            return

        half = self.obstacle_size / 2.0
        x_min, x_max = cx - half, cx + half
        y_min, y_max = cy - half, cy + half
        res = self.costmap_resolution

        # ===== 按“任意相交就算占用”的规则计算索引区间 =====
        # 每个栅格 (ix, iy) 覆盖的世界坐标区间为：
        #   x ∈ [map_origin_x + ix*res, map_origin_x + (ix+1)*res]
        #   y ∈ [map_origin_y + iy*res, map_origin_y + (iy+1)*res]
        # 与障碍物 [x_min, x_max] / [y_min, y_max] 存在交集的条件为：
        #   x_min < cell_x_max 且 x_max > cell_x_min
        #   y_min < cell_y_max 且 y_max > cell_y_min

        # 可能发生交集的最小 / 最大格子索引
        ix_min = int(np.floor((x_min - self.map_origin_x) / res))
        ix_max = int(np.floor((x_max - self.map_origin_x) / res))
        iy_min = int(np.floor((y_min - self.map_origin_y) / res))
        iy_max = int(np.floor((y_max - self.map_origin_y) / res))

        # 边界裁剪
        ix_min = max(0, ix_min)
        iy_min = max(0, iy_min)
        ix_max = min(self.grid_width - 1, ix_max)
        iy_max = min(self.grid_height - 1, iy_max)

        if ix_min > ix_max or iy_min > iy_max:
            return

        # 直接用切片批量标记
        self.costmap[iy_min:iy_max+1, ix_min:ix_max+1] = True

    def _build_costmap_from_obstacles(self):
        """由当前障碍物圆心列表构建 costmap"""
        self._init_costmap()
        for x, y, _yaw in self.obstacle_poses:
            self._mark_obstacle_on_costmap(x, y)

    def _log_costmap(self, region_mask=None, robot_pos=None, target_pos=None):
        """
        将当前 costmap 可视化打印到日志中：
        - '#' = 障碍物占用
        - '.' = 最大连通自由区域中的可行走单元
        - ' ' = 其它空闲单元
        - 'R' = 机器人起点
        - 'T' = 目标点
        """
        if self.costmap is None:
            self._env_log("[costmap] costmap is None, nothing to print")
            return
        try:
            h, w = self.costmap.shape
            # 布尔矩阵太占日志，这里不再逐行打印，只保留字符画

            # 默认全部空格
            canvas = np.full((h, w), ' ', dtype='<U1')

            # 障碍物
            canvas[self.costmap] = '#'

            # 最大连通区域（自由区域）
            if region_mask is not None:
                # 只在非障碍物上标记自由区域
                free_region = region_mask & (~self.costmap)
                canvas[free_region] = '.'

            # 机器人 & 目标覆盖在最上层
            if robot_pos is not None:
                gxgy = self._world_to_grid(robot_pos[0], robot_pos[1])
                if gxgy is not None:
                    gx, gy = gxgy
                    if 0 <= gy < h and 0 <= gx < w:
                        canvas[gy, gx] = 'R'
            if target_pos is not None:
                gxgy = self._world_to_grid(target_pos[0], target_pos[1])
                if gxgy is not None:
                    gx, gy = gxgy
                    if 0 <= gy < h and 0 <= gx < w:
                        canvas[gy, gx] = 'T'

            # 若需要调试 costmap，可临时取消下面的注释进行可视化打印
            # print(f"[costmap] resolution={self.costmap_resolution}, size=({h},{w})")
            # # 为了让 y 轴向上，对行进行反转打印
            # for row in canvas[::-1]:
            #     print(''.join(row))
        except Exception as e:
            self._env_log(f"[costmap] visualize failed: {e}")

    def _select_free_region_by_probability(self, region_select_bias):
        """在 costmap 中按概率选择一个“足够大”的连通自由区域，返回 bool mask（True=该区域内）

        规则：
        - 收集所有连通自由区域（由 False 区域构成），
        - 从中按概率选择一个"格子数量 >= 2"的区域作为结果；
        - 若不存在满足条件的区域，则返回 None。
        
        Args:
            region_select_bias: 选择最大连通区域的概率 p ∈ [0,1]
                - 以概率 p 在所有最大连通区域中随机选择一个
                - 否则（1-p）在剩余较小区域中等概率随机选择一个
                - 若不存在"剩余区域"，则仍然从最大区域中选
        
        Returns:
            bool mask（True=该区域内），如果找不到则返回 None
        """
        if self.costmap is None:
            return None
        free = ~self.costmap
        visited = np.zeros_like(free, dtype=bool)

        regions = []  # 存储 (mask, size)

        height, width = free.shape
        # 4-邻接
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for iy in range(height):
            for ix in range(width):
                if not free[iy, ix] or visited[iy, ix]:
                    continue
                # BFS
                queue = [(iy, ix)]
                visited[iy, ix] = True
                current_cells = []

                while queue:
                    cy, cx = queue.pop(0)
                    current_cells.append((cy, cx))
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width and free[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))

                # 为当前连通块生成 mask
                size = len(current_cells)
                if size > 0:
                    mask = np.zeros_like(free, dtype=bool)
                    for cy, cx in current_cells:
                        mask[cy, cx] = True
                    regions.append((mask, size))

        if not regions:
            return None

        # 只保留格子数量 >= 2 的连通区域（至少能放下机器人和目标的两个不同格子）
        valid_regions = [r for r in regions if r[1] >= 2]
        if not valid_regions:
            return None

        # ========= 概率优先选择“最大连通区域” =========
        # 规则：
        # - 将 region_select_bias 视为概率 p ∈ [0,1]
        # - 以概率 p 在所有最大连通区域中随机选择一个
        # - 否则（1-p）在剩余较小区域中等概率随机选择一个；
        # - 若不存在“剩余区域”，则仍然从最大区域中选。
        sizes = np.array([s for (_, s) in valid_regions], dtype=float)
        max_size = sizes.max()
        largest_indices = np.where(sizes == max_size)[0]

        # 其他非最大区域
        other_indices = np.array([i for i in range(len(valid_regions)) if i not in largest_indices])

        # 概率裁剪到 [0,1]
        p = float(region_select_bias)
        if not np.isfinite(p):
            p = 1.0
        p = max(0.0, min(1.0, p))

        if np.random.rand() < p or other_indices.size == 0:
            # 按概率或在没有其他区域时，从最大区域中选
            idx = int(np.random.choice(largest_indices))
        else:
            # 否则在剩余区域中等概率选择
            idx = int(np.random.choice(other_indices))

        return valid_regions[idx][0]

    def _grid_to_world_center(self, ix, iy):
        """栅格中心 -> 世界坐标"""
        x = self.map_origin_x + (ix + 0.5) * self.costmap_resolution
        y = self.map_origin_y + (iy + 0.5) * self.costmap_resolution
        return x, y

    def _grid_to_world_random(self, ix, iy):
        """栅格范围内随机位置 -> 世界坐标"""
        x_min = self.map_origin_x + ix * self.costmap_resolution
        x_max = self.map_origin_x + (ix + 1) * self.costmap_resolution
        y_min = self.map_origin_y + iy * self.costmap_resolution
        y_max = self.map_origin_y + (iy + 1) * self.costmap_resolution
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        return x, y

    def _sample_robot_and_target_on_region(self, region_mask):
        """在给定的（已选定）连通区域内采样机器人与目标位置（满足距离约束，且尽量相互远一点）

        策略：
        - 直接从连通区域的格子中随机选择一个格子作为机器人位置（格子中心）；
        - 遍历连通区域内的所有连通格，选择符合距离范围条件且距离最远的格子作为终点。
        """
        indices = np.argwhere(region_mask)
        if indices.size == 0:
            return None, None

        # 直接从连通区域的格子中随机选择一个格子作为机器人位置
        ridx = np.random.randint(len(indices))
        ry, rx = indices[ridx]
        robot_x, robot_y = self._grid_to_world_center(rx, ry)

        # 采样目标位置，满足距离约束且尽量远
        min_dist = self.target_reached_delta + 0.1
        max_dist = self.target_dist

        best_target_grid = None
        best_dist = -1.0

        # 遍历连通区域内的所有连通格
        for tidx in range(len(indices)):
            ty, tx = indices[tidx]
            # 使用格子中心计算距离（用于比较选择最佳格子）
            target_x_center, target_y_center = self._grid_to_world_center(tx, ty)
            dist = np.linalg.norm([target_x_center - robot_x, target_y_center - robot_y])
            # 若连通格距离符合距离范围条件，则与最大距离进行对比
            if min_dist <= dist <= max_dist and dist > best_dist:
                best_dist = dist
                best_target_grid = (tx, ty)  # 记录格子索引，而不是中心坐标

        if best_target_grid is not None:
            # 在最佳格子的范围内随机生成终点位置
            tx, ty = best_target_grid
            target_x, target_y = self._grid_to_world_random(tx, ty)
            return (robot_x, robot_y), (target_x, target_y)

        # 若未找到合适目标，则返回 None 触发后备方案
        return (robot_x, robot_y), None

    def _find_nearest_free_grid_center(self, current_x, current_y):
        """找到当前位置最近的空格子中心
        
        Args:
            current_x, current_y: 当前位置的世界坐标
            
        Returns:
            (x, y) 最近的空格子中心的世界坐标，如果找不到则返回None
        """
        if self.costmap is None:
            return None
        
        # 将当前位置转换为栅格坐标
        current_gxgy = self._world_to_grid(current_x, current_y)
        if current_gxgy is None:
            return None
        
        current_gx, current_gy = current_gxgy
        
        # 构建自由区域的mask（非障碍物区域）
        free_mask = ~self.costmap
        
        if not free_mask.any():
            return None
        
        # 找到所有空格子的索引
        free_indices = np.argwhere(free_mask)
        
        if free_indices.size == 0:
            return None
        
        # 计算当前格子到所有空格子的距离（使用栅格距离，更快）
        free_array = np.array(free_indices)
        distances = np.sqrt((free_array[:, 0] - current_gy)**2 + (free_array[:, 1] - current_gx)**2)
        
        # 找到最近的空格子索引
        nearest_idx = np.argmin(distances)
        nearest_gy, nearest_gx = free_indices[nearest_idx]
        
        # 转换为世界坐标（格子中心）
        x, y = self._grid_to_world_center(nearest_gx, nearest_gy)
        return x, y

    def generate_uniform_candidates(self, bias):
        """生成均匀的候选点集合"""
        # 创建均匀网格
        grid_size = int(np.ceil(np.sqrt(300)))  # 约300个候选点
        x = np.linspace(-bias, bias, grid_size)
        y = np.linspace(-bias, bias, grid_size)
        xx, yy = np.meshgrid(x, y)
        candidate_points = np.vstack([xx.ravel(), yy.ravel()]).T
        
        # 添加随机扰动避免网格对齐
        perturbation = np.random.uniform(-0.5, 0.5, candidate_points.shape) * (bias / grid_size)
        candidate_points += perturbation
        
        # 边界处理
        np.clip(candidate_points, -bias, bias, out=candidate_points)
        
        self.candidate_points = candidate_points.tolist()

    def fallback_random_position(self, name, bias, angle):
        """后备随机位置方法"""
        try_time = 0
        max_tries = 500
        while try_time < max_tries:
            try_time += 1
            x = np.random.uniform(-bias, bias)
            y = np.random.uniform(-bias, bias)
            # 使用 obs_min_dist 约束障碍物之间的最小圆心距离
            if self.check_position(x, y, self.obs_min_dist):
                # 只在内存中记录障碍物信息，暂不写入 Gazebo
                self.obstacle_poses.append([x, y, angle])
                return True
        return False

    def _wait_for_odom_update(self, expected_x, expected_y, max_wait_time=2.0, position_tolerance=0.5):
        """等待里程计更新到期望位置
        
        Args:
            expected_x, expected_y: 期望的机器人位置
            max_wait_time: 最大等待时间（秒）
            position_tolerance: 位置容差（米），里程计位置与期望位置的距离小于此值则认为已更新
        """
        wait_start = time.time()
        position_raw = None
        while time.time() - wait_start < max_wait_time:
            # subscriber在后台线程中自动接收数据，直接读取即可
            (
                _latest_scan,
                _latest_position,
                _latest_orientation,
                _current_linear_velocity,
                _current_angular_velocity,
                position_raw,
            ) = self.sensor_subscriber.get_latest_sensor()
            
            if position_raw is not None:
                odom_x = position_raw.x
                odom_y = position_raw.y
                distance = np.linalg.norm([odom_x - expected_x, odom_y - expected_y])
                if distance < position_tolerance:
                    if self.env_logger is not None:
                        self.env_logger.log(self.env_id, f"odom updated to ({odom_x:.2f}, {odom_y:.2f}), expected ({expected_x:.2f}, {expected_y:.2f}), distance={distance:.3f}")
                    return True
            
            time.sleep(0.05)  # 短暂等待后重试
        
        # 超时后记录错误（视为负面事件），但不直接使调用失败
        if position_raw is not None:
            odom_x = position_raw.x
            odom_y = position_raw.y
            distance = np.linalg.norm([odom_x - expected_x, odom_y - expected_y])
            self._env_log(f"[ERROR] odom update timeout: current=({odom_x:.2f}, {odom_y:.2f}), expected=({expected_x:.2f}, {expected_y:.2f}), distance={distance:.3f}")
        else:
            self._env_log(f"[ERROR] odom update timeout: no odom data received")
        return False

    def set_position(self, name, x, y, angle, wait_for_odom_update=False):
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        # 对障碍物模型，保证其中心高度为自身高度的一半（这里为 0.3 / 2 = 0.15）
        # 参见 obstacle_cylinder_small/model.sdf 中的 <pose>0 0 0.15 ...</pose>
        if name.startswith("obstacle"):
            pose.position.z = 0.15
        else:
            pose.position.z = 0.0
        pose.orientation.x = quaternion.x
        pose.orientation.y = quaternion.y
        pose.orientation.z = quaternion.z
        pose.orientation.w = quaternion.w

        success = self.robot_state_publisher.set_state(name, pose)
        if not success:
            self._env_log(f"[ERROR] set_position failed name={name} x={x:.2f} y={y:.2f} angle={angle:.2f}")
            return success
        
        # 如果是机器人位置设置，且需要等待里程计更新，则等待里程计更新到新位置
        if wait_for_odom_update and name == "turtlebot3_waffle":
            odom_updated = self._wait_for_odom_update(x, y, max_wait_time=2.0, position_tolerance=0.5)
            if not odom_updated:
                self._env_log(f"set_position failed: odom update timeout name={name} x={x:.2f} y={y:.2f} angle={angle:.2f}")
                return False
        
        return success

    def generate_and_set_obstacles(self):
        """
        生成障碍物位姿、更新 costmap，并将障碍物位姿写入 Gazebo。
        
        功能：
        1) 生成障碍物位姿（保存到 self.obstacle_poses）
        2) 基于障碍物更新 costmap，并校验存在可用自由连通区域
        3) 根据当前缓存的障碍物位姿，将障碍物设置到 Gazebo
        
        返回：
            bool: 成功返回 True，失败返回 False
        """
        # 1) 生成障碍物位姿（仅记录到内存）
        self.generate_obstacle_poses()

        # 2) 基于当前障碍物圆心构建 costmap
        self._build_costmap_from_obstacles()

        # 3) 轻量校验：必须存在可用连通自由区域
        region_mask = self._select_free_region_by_probability(self.region_select_bias)
        if region_mask is None or not region_mask.any():
            self._log_costmap(region_mask, None, None)
            return False

        # 4) 将障碍物位姿真正写入 Gazebo
        if not self.apply_obstacle_poses_to_gazebo():
            if self.env_logger is not None:
                self.env_logger.log(self.env_id, f"obstacles apply failed n={len(self.obstacle_poses)}")
            return False

        if self.env_logger is not None:
            self.env_logger.log(self.env_id, f"obstacles generated and applied n={len(self.obstacle_poses)}")
        # 当前仅维护障碍物位姿；距离约束统一基于 obstacle_poses 计算
        return True

    def generate_obstacle_poses(self):
        """生成障碍物位姿（仅更新内存中的缓存，不写入 Gazebo）。"""
        # 清空之前的障碍物数据
        self.obstacle_poses = []

        # 确保每次重新生成均匀候选点（均匀分布模式依赖）
        if hasattr(self, "candidate_points"):
            delattr(self, "candidate_points")

        # 生成障碍物位姿（generate_obstacle_pose 会写入 obstacle_poses）
        for i in range(0, self.obs_num):
            name = "obstacle" + str(i + 1)
            # 如果设置失败（例如实体不存在），就跳过该障碍物
            if not self.generate_obstacle_pose(name):
                continue

    def apply_obstacle_poses_to_gazebo(self):
        """根据当前缓存的障碍物位姿数组，将障碍物位置实际写入 Gazebo。
        
        Returns:
            bool: 所有障碍物设置成功返回 True，任意失败返回 False
        """
        for idx, (ox, oy, oang) in enumerate(self.obstacle_poses):
            name = f"obstacle{idx+1}"
            if not self.set_position(name, ox, oy, oang):
                self._env_log(f"[costmap][warn] final set_position failed for {name}")
                if self.env_logger is not None:
                    self.env_logger.log(self.env_id, f"apply_obstacle_poses failed at {name} (idx={idx+1}/{len(self.obstacle_poses)})")
                return False
        # 所有障碍物设置成功
        if self.env_logger is not None:
            self.env_logger.log(self.env_id, f"apply_obstacle_poses success n={len(self.obstacle_poses)}")
        return True
    
    def _spawn_robot_and_target_from_current_costmap(self):
        """基于当前 costmap 生成机器人和终点（不改变障碍物）。"""
        # costmap 可能在某些情况下丢失/未构建，这里兜底重建一次
        if getattr(self, "costmap", None) is None:
            self._build_costmap_from_obstacles()

        region_mask = self._select_free_region_by_probability(self.region_select_bias)
        if region_mask is None or not region_mask.any():
            self._log_costmap(region_mask, None, None)
            return False

        robot_pos, target_pos = self._find_best_robot_target_combination(region_mask)
        if robot_pos is None or target_pos is None:
            self._log_costmap(region_mask, None, None)
            return False

        rx, ry = robot_pos
        tx, ty = target_pos

        angle = np.random.uniform(-np.pi, np.pi)
        
        # 持续设置机器人位置，直到设置成功且里程计更新成功
        max_retries = 5
        retry_count = 0
        robot_set_success = False
        
        while retry_count < max_retries:
            retry_count += 1
            
            # 尝试设置机器人位置
            set_success = self.set_position("turtlebot3_waffle", rx, ry, angle, wait_for_odom_update=True)
            
            if not set_success:
                self._env_log(f"[WARNING] set_position failed (attempt {retry_count}/{max_retries}) robot=({rx:.2f},{ry:.2f}) angle={angle:.4f}")
                time.sleep(0.1)  # 短暂等待后重试
                continue
            else:
                robot_set_success = True
                break
        
        if not robot_set_success:
            self._env_log(f"[ERROR] _spawn_robot_and_target_from_current_costmap failed after {max_retries} attempts: "
                         f"robot position=({rx:.2f},{ry:.2f}) angle={angle:.4f}")
            return False

        self.episode_start_position = [rx, ry]
        self.target = [tx, ty]

        # 终点位置在这里统一“实际设置”（发布 + Gazebo 模型位置）
        self.publish_target.publish(self.target[0], self.target[1])
        # 在Gazebo中设置目标圆柱体位置（高度0.1，中心0.05，底面贴地，对应 waffle.model 中 goal_cylinder）
        self.goal_model_client.set_goal_position(self.target[0], self.target[1], z=0.05)

        self._log_costmap(region_mask, self.episode_start_position, self.target)
        if self.env_logger is not None:
            self.env_logger.log(self.env_id, f"spawn_robot_and_target ok robot=({rx:.2f},{ry:.2f}) target=({self.target[0]:.2f},{self.target[1]:.2f})")
        return True

    def check_position(self, x, y, min_dist):
        """检查给定位置与当前已生成障碍物之间的最小距离是否大于等于 min_dist。"""
        for ox, oy, _yaw in self.obstacle_poses:
            distance_vector = [ox - x, oy - y]
            dist = np.linalg.norm(distance_vector)
            if dist < min_dist:
                return False
        return True

    def check_collision(self, laser_scan):
        if min(laser_scan) < self.collision_delta:
            return True
        return False

    def check_target(self, distance, collision):
        if distance < self.target_reached_delta and not collision:
            self.target_dist += self.target_dist_increase
            if self.target_dist > self.max_target_dist:
                self.target_dist = self.max_target_dist
            return True
        return False

    def get_dist_sincos(self, odom_position, odom_orientation):
        # Calculate robot heading from odometry data
        # 确保里程计数据存在
        if odom_position is None:
            return self.target_reached_delta+0.1,1,0,0

        odom_x = odom_position.x
        odom_y = odom_position.y
        quaternion = Quaternion(
            odom_orientation.w,
            odom_orientation.x,
            odom_orientation.y,
            odom_orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)
        pose_vector = [np.cos(angle), np.sin(angle)]
        goal_vector = [self.target[0] - odom_x, self.target[1] - odom_y]

        distance = np.linalg.norm(goal_vector)
        cos, sin = self.cossin(pose_vector, goal_vector)

        return distance, cos, sin, angle

    def get_reward(self,goal, collision, action, laser_scan,distance, cos, sin):
        # Reward log helper：每步都写入 reward_log
        def _reward_log(msg: str):
            if self.reward_logger is not None:
                self.reward_logger.log(self.env_id, f"get_reward: {msg}")
        
        scale = self.reward_scale

        if goal:
            # 返回缩放后的奖励（不包含step_penalty）
            goal_scaled = self.goal_reward * scale
            total = goal_scaled
            # 记录本step分量（raw+scaled）
            self.last_step_reward_parts = {
                "raw": {
                    "goal": float(self.goal_reward),
                    "collision": 0.0,
                    "obs": 0.0,
                    "yawrate": 0.0,
                    "angle": 0.0,
                    "linear": 0.0,
                    "step_penalty": 0.0,
                    "target_distance": 0.0,
                    "progress": 0.0,
                    "linear_acc_osc": 0.0,
                    "yawrate_osc": 0.0,
                },
                "scaled": {
                    "goal": float(goal_scaled),
                    "collision": 0.0,
                    "obs": 0.0,
                    "yawrate": 0.0,
                    "angle": 0.0,
                    "linear": 0.0,
                    "step_penalty": 0.0,
                    "target_distance": 0.0,
                    "progress": 0.0,
                    "linear_acc_osc": 0.0,
                    "yawrate_osc": 0.0,
                },
            }
            goal_msg = (
                f"GOAL=True step={self.step_count} "
                f"distance={distance:.4f} cos={cos:.4f} sin={sin:.4f} "
                f"action=[{action[0]:.3f},{action[1]:.3f}] "
                f"goal_reward={self.goal_reward:.4f}/{goal_scaled:.4f} scale={scale:.3f} "
                f"step_penalty=0.0000/0.0000 total={total:.4f}"
            )
            # 只记录到 reward_log，不再写入 env_log
            _reward_log(goal_msg)
            return total
        elif collision:
            # 碰撞：只给碰撞惩罚（不叠加时间步惩罚）
            collision_scaled = self.collision_penalty_base * scale
            total = collision_scaled
            # 记录本step分量（raw+scaled）
            self.last_step_reward_parts = {
                "raw": {
                    "goal": 0.0,
                    "collision": float(self.collision_penalty_base),
                    "obs": 0.0,
                    "yawrate": 0.0,
                    "angle": 0.0,
                    "linear": 0.0,
                    "step_penalty": 0.0,
                    "target_distance": 0.0,
                    "progress": 0.0,
                    "linear_acc_osc": 0.0,
                    "yawrate_osc": 0.0,
                },
                "scaled": {
                    "goal": 0.0,
                    "collision": float(collision_scaled),
                    "obs": 0.0,
                    "yawrate": 0.0,
                    "angle": 0.0,
                    "linear": 0.0,
                    "step_penalty": 0.0,
                    "target_distance": 0.0,
                    "progress": 0.0,
                    "linear_acc_osc": 0.0,
                    "yawrate_osc": 0.0,
                },
            }
            # 碰撞时必打印（定位训练崩坏/异常episode很有用）
            laser_min = float(np.min(laser_scan)) if laser_scan is not None and len(laser_scan) > 0 else float("nan")
            collision_msg = (
                f"COLLISION=True step={self.step_count} "
                f"distance={distance:.4f} laser_min={laser_min:.4f} "
                f"action=[{action[0]:.3f},{action[1]:.3f}] "
                f"collision_penalty={self.collision_penalty_base:.4f}/{collision_scaled:.4f} scale={scale:.3f} "
                f"step_penalty=0.0000/0.0000 total={total:.4f}"
            )
            _reward_log(collision_msg)
            return total
        else:
            step_penalty = self.step_penalty_base if self.enable_step_penalty else 0.0
            # ================进度奖惩================
            progress_reward = 0.0
            delta_dist = None
            if (
                self.enable_progress_reward
                and self.prev_distance_to_goal is not None
                and np.isfinite(distance)
                and np.isfinite(self.prev_distance_to_goal)
            ):
                denom = max(self.step_sleep_time * self.max_velocity, 1e-6)
                delta_dist = self.prev_distance_to_goal - distance
                progress_reward = self.progress_reward_base * (delta_dist / denom)
            # ================时间步常数惩罚================
            # ================计算障碍物距离惩罚================
            # 当最近障碍物距离低于阈值时，距离越接近0惩罚越大
            # 当distance >= threshold时，惩罚为0
            # 支持动态阈值：配置为-1时，阈值按|v| * sim_time计算
            obs_penalty = 0.0
            threshold = None
            high_min = None
            side_left_min = None
            side_right_min = None
            if self.enable_obs_penalty:
                # 获取最近障碍物距离（分段加权）
                if self.obs_penalty_threshold < 0:
                    # 动态阈值：按 |线速度| * sim_time 计算，但不低于 min_obs_penalty_threshold
                    threshold = max(self.min_obs_penalty_threshold, abs(action[0]) * self.sim_time)
                else:
                    # 固定阈值：直接使用配置值
                    threshold = self.obs_penalty_threshold
                
                def calc_penalty(dist):
                    # 检查dist是否为有效数值，且小于阈值
                    if np.isfinite(dist) and dist < threshold:
                        penalty_value = self.obs_penalty_base * np.power(threshold - dist, self.obs_penalty_power)
                        # 确保返回值为有限数值
                        return penalty_value if np.isfinite(penalty_value) else 0.0
                    return 0.0

                # 拆分雷达：中间40%为高权重，两侧各20%为低权重
                n = len(laser_scan)
                middle_ratio = self.obs_penalty_middle_ratio
                middle_ratio = min(middle_ratio, 1.0)
                side_ratio = max(0.0, (1.0 - middle_ratio) / 2.0)

                left_end = int(side_ratio * n)
                middle_end = int((side_ratio + middle_ratio) * n)

                high_scan = laser_scan[left_end:middle_end] if middle_end > left_end else np.array([])
                side_left_scan = laser_scan[:left_end] if left_end > 0 else np.array([])
                side_right_scan = laser_scan[middle_end:] if middle_end < n else np.array([])

                # 最小距离（若分段为空则设为inf确保不产生惩罚）
                high_min = np.min(high_scan) if high_scan.size > 0 else np.inf
                side_left_min = np.min(side_left_scan) if side_left_scan.size > 0 else np.inf
                side_right_min = np.min(side_right_scan) if side_right_scan.size > 0 else np.inf

                # 分别计算惩罚并加权
                high_penalty = calc_penalty(high_min) * self.obs_penalty_high_weight
                side_penalty = max(calc_penalty(side_left_min), calc_penalty(side_right_min)) * self.obs_penalty_low_weight

                obs_penalty = high_penalty + side_penalty
            # ================计算最近障碍物距离惩罚================ 

            # ================计算角速度惩罚================
            yawrate_penalty = 0.0
            if self.enable_yawrate_penalty:
                yawrate_penalty = self.yawrate_penalty_base * abs(action[1])
            # ================计算角速度惩罚================

            # ================计算角度偏移惩罚================
            angle_penalty = 0
            if self.enable_angle_penalty:
                # atan2已经将角度规范化到[-π, π]范围，直接取绝对值即可
                current_angle = math.atan2(sin, cos)
                angle_diff = abs(current_angle)
                # 将 (1 - cos(angle_diff)) 从 [0, 2] 归一化到 [0, 1]
                normalized_angle_value = (1 - math.cos(angle_diff)) / 2.0
                angle_penalty = self.angle_penalty_base * normalized_angle_value
            # ================计算角度偏移惩罚==========================

            # ================线速度惩罚================
            linear_penalty = 0
            if self.enable_linear_penalty:
                # 线速度越接近最大速度，惩罚越小
                # 将 action[0] 裁剪到 [0, max_velocity] 范围，然后归一化到 [0, 1]
                if self.max_velocity > 0:
                    clipped_action = np.clip(action[0], 0, self.max_velocity)
                    normalized_value = (self.max_velocity - clipped_action) / self.max_velocity
                else:
                    # 如果 max_velocity 为 0，使用默认归一化值
                    normalized_value = 1.0 if action[0] <= 0 else 0.0
                linear_penalty = self.linear_penalty_base * normalized_value
            # ================线速度惩罚================
            
            # ================终点距离惩罚================
            # 根据当前终点距离/生成终点时的真实终点距离计算惩罚
            # 若当前终点距离为3，生成终点时的真实终点距离为6，则惩罚为3/6=0.5
            target_distance_penalty = 0.0
            max_distane_ratio = 2.0
            if self.enable_target_distance_penalty and self.initial_target_distance is not None and self.initial_target_distance > 0:
                # 检查distance是否为有效数值
                if np.isfinite(distance) and distance >= 0:
                    # 限制distance的最大值，防止异常大值
                    distance_clipped = min(distance, 1000.0)
                    # 计算当前距离与初始距离的比值
                    distance_ratio = distance_clipped / max(self.initial_target_distance, 1.0)
                    # 限制distance_ratio的最大值，防止数值溢出（最大比值设为100）
                    distance_ratio = min(distance_ratio, max_distane_ratio)
                    # 惩罚 = base * 距离比值（距离越远，比值越大，惩罚越大）,通过除以max_distane_ratio来归一化
                    target_distance_penalty = self.target_distance_penalty_base * distance_ratio / max_distane_ratio
                    # 最终检查：确保惩罚值是有限数值，防止NaN和inf
                    if not np.isfinite(target_distance_penalty):
                        target_distance_penalty = 0.0
            # ================终点距离惩罚================
            
            # ================线速度加速度震荡惩罚================
            # 根据当前action的线速度和上一action的线速度计算加速度
            # 如果当前加速度和上一加速度的符号不同，则给予惩罚
            linear_acceleration_oscillation_penalty = 0.0
            if self.enable_linear_acceleration_oscillation_penalty:
                current_linear_velocity = action[0]
                step_time = self.step_sleep_time  # 每个step的持续时间
                
                # 防止除零错误
                if step_time <= 0:
                    step_time = 0.1  # 使用默认值
                
                if self.prev_linear_velocity is not None:
                    # 计算当前加速度
                    current_acceleration = (current_linear_velocity - self.prev_linear_velocity) / step_time
                    
                    if self.prev_linear_acceleration is not None:
                        # 检查加速度符号是否改变
                        if (current_acceleration > 0 and self.prev_linear_acceleration < 0) or \
                           (current_acceleration < 0 and self.prev_linear_acceleration > 0):
                            # 符号改变，计算加速度差值
                            acceleration_diff = abs(current_acceleration - self.prev_linear_acceleration)
                            linear_acceleration_oscillation_penalty = self.linear_acceleration_oscillation_penalty_base * acceleration_diff
                    
                    # 更新上一加速度（用于下一次计算）
                    self.prev_linear_acceleration = current_acceleration
                
                # 更新上一线速度（用于下一次计算）
                self.prev_linear_velocity = current_linear_velocity
            # ================线速度加速度震荡惩罚================
            
            # ================角速度震荡惩罚================
            # 如果当前角速度和上一角速度符号不同，则给予惩罚
            yawrate_oscillation_penalty = 0.0
            if self.enable_yawrate_oscillation_penalty:
                current_angular_velocity = action[1]
                
                if self.prev_angular_velocity is not None:
                    # 检查角速度符号是否改变
                    if (current_angular_velocity > 0 and self.prev_angular_velocity < 0) or \
                       (current_angular_velocity < 0 and self.prev_angular_velocity > 0):
                        # 符号改变，计算角速度差值
                        yawrate_diff = abs(current_angular_velocity - self.prev_angular_velocity)
                        yawrate_oscillation_penalty = self.yawrate_oscillation_penalty_base * yawrate_diff
                
                # 更新上一角速度
                self.prev_angular_velocity = current_angular_velocity
            # ================角速度震荡惩罚================
            
            # 记录本 step 奖励分量到当前 episode 统计（先各自缩放，再求和）
            obs_penalty_scaled = obs_penalty * scale
            yawrate_penalty_scaled = yawrate_penalty * scale
            angle_penalty_scaled = angle_penalty * scale
            linear_penalty_scaled = linear_penalty * scale
            target_distance_penalty_scaled = target_distance_penalty * scale
            linear_accel_osc_penalty_scaled = linear_acceleration_oscillation_penalty * scale
            yawrate_osc_penalty_scaled = yawrate_oscillation_penalty * scale
            progress_reward_scaled = progress_reward * scale
            scaled_step_penalty = step_penalty * scale

            # 记录本step分量（raw+scaled）：训练侧可直接读取，避免差分episode累计值
            self.last_step_reward_parts = {
                "raw": {
                    "goal": 0.0,
                    "collision": 0.0,
                    "obs": float(obs_penalty),
                    "yawrate": float(yawrate_penalty),
                    "angle": float(angle_penalty),
                    "linear": float(linear_penalty),
                    "step_penalty": float(step_penalty),
                    "target_distance": float(target_distance_penalty),
                    "progress": float(progress_reward),
                    "linear_acc_osc": float(linear_acceleration_oscillation_penalty),
                    "yawrate_osc": float(yawrate_oscillation_penalty),
                },
                "scaled": {
                    "goal": 0.0,
                    "collision": 0.0,
                    "obs": float(obs_penalty_scaled),
                    "yawrate": float(yawrate_penalty_scaled),
                    "angle": float(angle_penalty_scaled),
                    "linear": float(linear_penalty_scaled),
                    "step_penalty": float(scaled_step_penalty),
                    "target_distance": float(target_distance_penalty_scaled),
                    "progress": float(progress_reward_scaled),
                    "linear_acc_osc": float(linear_accel_osc_penalty_scaled),
                    "yawrate_osc": float(yawrate_osc_penalty_scaled),
                },
            }

            # 计算所有惩罚项缩放后的总和（保证与各缩放分量之和完全一致）
            total = (
                yawrate_penalty_scaled
                + obs_penalty_scaled
                + angle_penalty_scaled
                + linear_penalty_scaled
                + scaled_step_penalty
                + target_distance_penalty_scaled
                + linear_accel_osc_penalty_scaled
                + yawrate_osc_penalty_scaled
                + progress_reward_scaled
            )

            # 调试输出：始终写入 env_log（按步频节流 / 异常时额外输出）；是否 print 由 _env_log 决定
            dbg_every = int(getattr(self, "reward_debug_every", 20))
            laser_min = float(np.min(laser_scan)) if laser_scan is not None and len(laser_scan) > 0 else float("nan")
            near_obs = (threshold is not None) and np.isfinite(laser_min) and (laser_min < threshold)
            bad_number = (not np.isfinite(total)) or (not np.isfinite(distance))
            should_log = (dbg_every > 0 and (self.step_count % dbg_every == 0)) or near_obs or bad_number
            
            # 构建奖励日志消息（每步都记录到 reward_log）
            dd = "None" if delta_dist is None else f"{delta_dist:.4f}"
            reward_msg = (
                f"step={self.step_count} distance={distance:.4f} prev_dist={self.prev_distance_to_goal} "
                f"delta_dist={dd} cos={cos:.4f} sin={sin:.4f} "
                f"action=[{action[0]:.3f},{action[1]:.3f}] "
                f"laser_min={laser_min:.4f} threshold={threshold} "
                f"high_min={high_min} side_left_min={side_left_min} side_right_min={side_right_min} "
                f"progress={progress_reward:.4f}/{progress_reward_scaled:.4f} "
                f"obs={obs_penalty:.4f}/{obs_penalty_scaled:.4f} "
                f"yawrate={yawrate_penalty:.4f}/{yawrate_penalty_scaled:.4f} "
                f"angle={angle_penalty:.4f}/{angle_penalty_scaled:.4f} "
                f"linear={linear_penalty:.4f}/{linear_penalty_scaled:.4f} "
                f"tgt_dist_pen={target_distance_penalty:.4f}/{target_distance_penalty_scaled:.4f} "
                f"lin_acc_osc={linear_acceleration_oscillation_penalty:.4f}/{linear_accel_osc_penalty_scaled:.4f} "
                f"yaw_osc={yawrate_oscillation_penalty:.4f}/{yawrate_osc_penalty_scaled:.4f} "
                f"step_pen={step_penalty:.4f}/{scaled_step_penalty:.4f} "
                f"scale={self.reward_scale:.3f} total={total:.4f}"
            )
            
            # 每步都写入 reward_log
            _reward_log(reward_msg)

            return total

    @staticmethod
    def cossin(vec1, vec2):
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        cos = np.dot(vec1, vec2)
        sin = np.cross(vec1, vec2).item()

        return cos, sin
