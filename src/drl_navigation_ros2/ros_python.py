import time
import rclpy
from ros_nodes import (
    ScanSubscriber,
    OdomSubscriber,
    ResetWorldClient,
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
        # 奖励函数参数
        goal_reward=1000.0,  # 到达目标的奖励
        collision_penalty_base=-1000.0,  # 碰撞惩罚系数
        angle_penalty_base=0.0,  # 角度偏差基础惩罚
        linear_penalty_base=-1.0,  # 线速度基础惩罚
        yawrate_penalty_base=0.0,  # 角速度惩罚系数（负值惩罚，0为关闭效果）
        # 奖励/惩罚开关
        enable_obs_penalty=True,  # 是否启用障碍物距离惩罚
        enable_yawrate_penalty=True,  # 是否启用角速度惩罚
        enable_angle_penalty=True,  # 是否启用角度偏移惩罚
        enable_linear_penalty=True,  # 是否启用线速度惩罚
        enable_target_distance_penalty=False,  # 是否启用终点距离惩罚
        enable_linear_acceleration_oscillation_penalty=False,  # 是否启用线速度加速度震荡惩罚
        enable_yawrate_oscillation_penalty=False,  # 是否启用角速度震荡惩罚
        reward_debug=False,  # 是否打印奖励/惩罚明细
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
        # 震荡惩罚参数
        linear_acceleration_oscillation_penalty_base=-1.0,  # 线速度加速度震荡惩罚基础系数（负值表示惩罚）
        yawrate_oscillation_penalty_base=-1.0,  # 角速度震荡惩罚基础系数（负值表示惩罚）
        # 连通区域选择参数
        region_select_bias=1.0,  # 连通区域选择概率：1.0=总是选最大连通区域；0.8=80%概率选最大区域，否则从其余区域随机选
        # 时间控制参数
        sim_time=0.1,  # 仿真步长，用于基于速度的动态阈值等
        step_sleep_time=0.1,  # step方法中的sleep时间（秒）
        eval_sleep_time=1.0,  # eval方法中的sleep时间（秒）
        reset_step_count=3,  # reset方法中调用step的次数
        # 地图复用参数
        goals_per_map=1  # 每张地图的目标点数量（即一张地图可以用来产生多少个episode）
    ):
        # 记录初始化阶段步骤，便于定位卡点
        def _log(msg):
            print(f"[ROS_env {env_id}] {msg}")

        # 下面这些初始化阶段的日志在稳定运行时意义不大，且会刷屏，
        # 因此关闭默认打印，如需调试可临时取消注释。
        # _log("初始化 rclpy ...")
        rclpy.init(args=args)
        # _log("rclpy 初始化完成，开始创建ROS接口...")
        self.env_id = env_id
        self.cmd_vel_publisher = CmdVelPublisher(env_id)
        self.scan_subscriber = ScanSubscriber(env_id)
        self.odom_subscriber = OdomSubscriber(env_id)
        self.robot_state_publisher = SetModelStateClient(env_id)
        self.world_reset = ResetWorldClient(env_id)
        self.physics_client = PhysicsClient(env_id)
        self.publish_target = MarkerPublisher(env_id)
        self.goal_model_client = GoalModelClient(env_id)
        self.element_positions = []
        self.sensor_subscriber = SensorSubscriber(env_id)
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
        # 障碍物仅先在内存中记录（位置+朝向），确认 costmap 成功后再一次性写入 Gazebo
        self.obstacle_positions = []  # 单独记录障碍物圆心，用于构建 costmap
        self.obstacle_angles = []     # 与 obstacle_positions 对应的朝向
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
        # 奖励/惩罚开关
        self.enable_obs_penalty = enable_obs_penalty
        self.enable_yawrate_penalty = enable_yawrate_penalty
        self.enable_angle_penalty = enable_angle_penalty
        self.enable_linear_penalty = enable_linear_penalty
        self.enable_target_distance_penalty = enable_target_distance_penalty
        self.enable_linear_acceleration_oscillation_penalty = enable_linear_acceleration_oscillation_penalty
        self.enable_yawrate_oscillation_penalty = enable_yawrate_oscillation_penalty
        self.reward_debug = reward_debug
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
        # 震荡惩罚参数
        self.linear_acceleration_oscillation_penalty_base = linear_acceleration_oscillation_penalty_base
        self.yawrate_oscillation_penalty_base = yawrate_oscillation_penalty_base
        # 记录上一step的线速度、角速度和加速度（用于震荡惩罚计算）
        self.prev_linear_velocity = None
        self.prev_angular_velocity = None
        self.prev_linear_acceleration = None
        # 连通区域选择概率（内部按 [0,1] 裁剪）
        self.region_select_bias = region_select_bias
        # 时间控制参数
        self.sim_time = sim_time
        self.step_sleep_time = step_sleep_time
        self.eval_sleep_time = eval_sleep_time
        self.reset_step_count = reset_step_count
        # 地图复用参数
        self.goals_per_map = goals_per_map
        self.goals_count_for_current_map = 0  # 当前地图已使用的目标点数量
        # 奖励分解统计（按 episode 累积）
        # _log("ROS接口与参数初始化完成，准备reset...")
        self.reset_episode_reward_breakdown()
        # for i in range(60):
        #     self.step(empty_step=True)
        print(f"环境 {self.env_id} ROS环境变量赋值完成，开始reset地图...")
        self.reset()
        # _log("环境初始化完成")

    def reset_episode_reward_breakdown(self):
        """重置当前 episode 的奖励分解统计"""
        self.episode_obs_penalty = 0.0
        self.episode_yawrate_penalty = 0.0
        self.episode_angle_penalty = 0.0
        self.episode_linear_penalty = 0.0
        self.episode_target_distance_penalty = 0.0
        self.episode_linear_acceleration_oscillation_penalty = 0.0
        self.episode_yawrate_oscillation_penalty = 0.0
        self.episode_goal_reward = 0.0
        self.episode_collision_penalty = 0.0
        # 标记本 episode 是否已经发生过碰撞（用于保证碰撞惩罚每个 episode 只记一次）
        self._has_collision_this_episode = False
        # 重置上一step的记录（用于震荡惩罚计算）
        self.prev_linear_velocity = None
        self.prev_angular_velocity = None
        self.prev_linear_acceleration = None

    def step(self, lin_velocity=0.0, ang_velocity=0.0, empty_step=False):
        """执行一步仿真；empty_step=True 时仅发送零速度指令，不采样传感器或计算奖励。"""
        if empty_step:
            self.cmd_vel_publisher.publish_cmd_vel(0.0, 0.0)
            time.sleep(self.step_sleep_time)
            return None, None, None, None, False, False, [0.0, 0.0], 0.0

        self.step_count+=1
        self.cmd_vel_publisher.publish_cmd_vel(lin_velocity, ang_velocity)
        # 物理模拟的unpause/pause现在在episode级别控制，不在每个step中控制
        time.sleep(self.step_sleep_time)
        rclpy.spin_once(self.sensor_subscriber)

        (
            latest_scan,
            latest_position,
            latest_orientation,
        ) = self.sensor_subscriber.get_latest_sensor()
        if latest_scan is None:
            # 创建默认激光数据（360个点，距离10米）
            print(f"[ROS_env {self.env_id}] No laser scan data received, using default values.")
            latest_scan = [self.collision_delta+0.5] * 180
        latest_scan = np.array(latest_scan) 
        # print("latest_scan_len:",len(latest_scan))
        # 裁剪掉忽略的视野
        neglect_scan = int(np.ceil((self.neglect_angle/180)*len(latest_scan)))
        latest_scan = latest_scan[neglect_scan:len(latest_scan)-neglect_scan]
        latest_scan[latest_scan > self.scan_range] = self.scan_range # 把所有距离超过scan_range的值修改为scan_range
        #print(f" Laser scan data: {latest_scan}")
        distance, cos, sin, _ = self.get_dist_sincos(
            latest_position, latest_orientation
        )
        collision = self.check_collision(latest_scan)
        goal = self.check_target(distance, collision)
        action = [lin_velocity, ang_velocity]
        reward = self.get_reward(goal, collision, action, latest_scan,distance,cos,sin)

        return latest_scan, distance, cos, sin, collision, goal, action, reward

    def _find_best_robot_target_combination(self, region_mask):
        """在给定的连通区域内一次性尝试所有可能的机器人-目标位置组合，找到满足距离约束的组合
        
        策略：收集所有满足距离约束的组合，找到距离最远的那些组合，然后从中随机选择一个
        
        Args:
            region_mask: 连通区域的mask（bool数组）
            
        Returns:
            (robot_pos, target_pos) 如果找到满足条件的组合，否则返回 (None, None)
        """
        indices = np.argwhere(region_mask)
        if indices.size == 0:
            return None, None
        
        min_dist = self.target_reached_delta + 0.1
        max_dist = self.target_dist
        
        # 收集所有满足距离约束的组合
        valid_combinations = []
        
        # 遍历所有可能的机器人位置
        for ridx in range(len(indices)):
            ry, rx = indices[ridx]
            robot_x, robot_y = self._grid_to_world_center(rx, ry)
            
            # 对于当前机器人位置，遍历所有可能的目标位置
            for tidx in range(len(indices)):
                ty, tx = indices[tidx]
                target_x_center, target_y_center = self._grid_to_world_center(tx, ty)
                dist = np.linalg.norm([target_x_center - robot_x, target_y_center - robot_y])
                
                # 检查距离约束
                if min_dist <= dist <= max_dist:
                    valid_combinations.append({
                        'robot_pos': (robot_x, robot_y),
                        'target_grid': (tx, ty),
                        'dist': dist
                    })
        
        if len(valid_combinations) == 0:
            return None, None
        
        # 找到最大距离
        max_distance = max(combo['dist'] for combo in valid_combinations)
        
        # 筛选出距离等于最大距离的所有组合（最远的组合）
        farthest_combinations = [combo for combo in valid_combinations if combo['dist'] == max_distance]
        
        # 从最远的组合中随机选择一个
        selected = farthest_combinations[np.random.randint(len(farthest_combinations))]
        
        # 在选定的格子的范围内随机生成终点位置
        tx, ty = selected['target_grid']
        target_x, target_y = self._grid_to_world_random(tx, ty)
        
        return selected['robot_pos'], (target_x, target_y)

    def reset_with_current_map(self):
        """使用当前地图（障碍物保持不变）重新采样机器人和目标位置
        
        每次调用都会：
        1. 重新构建costmap
        2. 以region_select_bias概率选择最大连通区域，否则随机选择其他连通区域
        3. 在选定的连通区域内一次性尝试所有可能的机器人-目标位置组合，找到满足距离约束的最佳组合
        """
        # 基于当前障碍物重新构建costmap
        self._build_costmap_from_obstacles()
        
        # 重新选择连通区域（以region_select_bias概率选择最大连通区域）
        region_mask = self._find_largest_free_region()
        
        if region_mask is None or not region_mask.any():
            return False
        
        # 一次性尝试所有可能的机器人-目标位置组合
        robot_pos, target_pos = self._find_best_robot_target_combination(region_mask)
        
        if robot_pos is None or target_pos is None:
            return False
        
        # 设置机器人位置
        rx, ry = robot_pos
        tx, ty = target_pos
        angle = np.random.uniform(-np.pi, np.pi)
        self.set_position("turtlebot3_waffle", rx, ry, angle)
        
        self.episode_start_position = [rx, ry]
        self.target = [tx, ty]
        
        # 更新元素位置：障碍物 + 机器人 + 目标（障碍物位置不变）
        self.element_positions = list(self.obstacle_positions) + [
            self.episode_start_position,
            self.target,
        ]
        
        return True

    def reset(self):
        # print(f"环境 {env_id} 开始reset地图")
        # 根据goals_per_map参数决定是否重新生成地图
        should_regenerate_map = False
        
        if self.goals_per_map <= 1:
            # 如果goals_per_map <= 1，每次都重新生成地图（保持向后兼容）
            should_regenerate_map = True
        else:
            # 如果计数器为0，说明是第一次或者刚刚初始化，需要重新生成地图
            if self.goals_count_for_current_map == 0:
                should_regenerate_map = True
            # 检查当前地图是否已经达到目标点数量
            elif self.goals_count_for_current_map >= self.goals_per_map:
                # 达到数量，需要重新生成地图
                should_regenerate_map = True
        
        if should_regenerate_map:
            # 重新生成地图（包括障碍物、机器人和目标）
            self.world_reset.reset_world()
            position_set = self.set_positions()
            while not position_set:
                position_set = self.set_positions()
            # 重新生成地图后，这是新地图的第1个目标点
            self.goals_count_for_current_map = 1
        else:
            # 当前地图还未达到goals_per_map个终点，必须继续使用当前地图
            # 重新选择连通区域（以region_select_bias概率选择最大连通区域），
            # 并在选定的连通区域内重新采样机器人和目标位置
            reset_success = self.reset_with_current_map()
            if not reset_success:
                # 如果重置失败，说明当前地图确实无法继续使用
                # 但为了满足goals_per_map的要求，我们强制重新生成地图并重置计数器
                # 这种情况应该很少发生，如果频繁发生，说明地图生成逻辑有问题
                print(f"[WARNING] reset_with_current_map failed for map with {self.goals_count_for_current_map}/{self.goals_per_map} goals. "
                      f"Forcing map regeneration (this violates goals_per_map requirement for current map).")
                self.world_reset.reset_world()
                position_set = self.set_positions()
                while not position_set:
                    position_set = self.set_positions()
                # 重新生成地图后，这是新地图的第1个目标点
                self.goals_count_for_current_map = 1
            else:
                # 成功重置机器人和目标位置，增加当前地图的目标点计数
                self.goals_count_for_current_map += 1

        self.publish_target.publish(self.target[0], self.target[1])
        # 在Gazebo中设置目标圆柱体位置（高度0.1，中心0.05，底面贴地，对应 waffle.model 中 goal_cylinder）
        self.goal_model_client.set_goal_position(self.target[0], self.target[1], z=0.05)
        # 确保服务调用完成
        rclpy.spin_once(self.goal_model_client, timeout_sec=0.1)
        time.sleep(1)  # 短暂延迟，确保Gazebo处理位置设置
        
        # 要调用多次step，否则会因为gazebo加载等相关问题引发的错误
        # 前几次仅发送0速度，不采样传感器，避免产生无意义奖励
        for _ in range(max(1, self.reset_step_count)):
            self.step(empty_step=True)

        # 再进行一次真实的零速采样，确保返回有效观测，避免上层len()报错
        latest_scan, distance, cos, sin, collision, goal, last_action, reward = self.step(
            lin_velocity=0.0, ang_velocity=0.0, empty_step=False
        )
        # 记录episode开始时的终点距离（用于终点距离惩罚计算）
        if self.episode_start_position is not None and self.target is not None:
            self.initial_target_distance = np.linalg.norm([
                self.target[0] - self.episode_start_position[0],
                self.target[1] - self.episode_start_position[1]
            ])
        else:
            self.initial_target_distance = distance  # 如果无法计算，使用当前距离作为初始距离
        # 采样后重置计步，保证新episode从0开始计数
        self.step_count = 0
        # 重置阶段可能因为初始位置接近目标/障碍导致奖励被计入，出于公平性将奖励分解清零
        self.reset_episode_reward_breakdown()
        # print(f"环境 {env_id} reset地图完成")
        return latest_scan, distance, cos, sin, collision, goal, last_action, reward

    def set_target_position(self, robot_position):
        pos = False
        while not pos:
            # 使用极坐标采样，确保实际距离在 [target_reached_delta, target_dist] 范围内
            # 采样角度（0到2π）
            theta = np.random.uniform(0, 2 * np.pi)
            # 采样距离：为了避免终点离机器人太近，这里改为在
            # [max(target_dist-1, target_reached_delta+0.1), target_dist] 范围内均匀采样
            r_min = max(self.target_reached_delta + 0.1, self.target_dist - 1.0)
            r = np.random.uniform(r_min, self.target_dist)
            # 转换为笛卡尔坐标
            dist_x = r * np.cos(theta)
            dist_y = r * np.sin(theta)
            x = robot_position[0] + dist_x
            y = robot_position[1] + dist_y
            pos = self.check_position(x, y, self.obs_min_dist)
        self.element_positions.append([x, y])
        return [x, y]

    def set_random_position(self, name):
        """根据配置选择障碍物生成方式：均匀分布或随机分布

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
                
            # 选择离已有障碍物最远的点
            if self.element_positions:
                existing_points = np.array(self.element_positions)
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
            
            # 使用 obs_min_dist 约束：保证当前候选点与已有元素的最小圆心距离不小于 obs_min_dist
            if self.check_position(x, y, self.obs_min_dist):
                # 只在内存中记录障碍物信息，暂不写入 Gazebo
                self.element_positions.append([x, y])
                self.obstacle_positions.append([x, y])
                self.obstacle_angles.append(angle)
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
        for pos in self.obstacle_positions:
            self._mark_obstacle_on_costmap(pos[0], pos[1])

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
            print("[costmap] costmap is None, nothing to print")
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
            print(f"[costmap] visualize failed: {e}")

    def _find_largest_free_region(self):
        """在 costmap 中随机选择一个“足够大”的连通自由区域，返回 bool mask（True=该区域内）

        规则调整为：
        - 不再总是选取“最大”的连通区域；
        - 收集所有连通自由区域（由 False 区域构成），
        - 从中随机选择一个“格子数量 >= 2”的区域作为结果；
        - 若不存在满足条件的区域，则返回 None。
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
        p = float(self.region_select_bias)
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
                self.element_positions.append([x, y])
                self.obstacle_positions.append([x, y])
                self.obstacle_angles.append(angle)
                return True
        return False

    def set_robot_position(self):
        bias = self.world_size/2 # 机器人生成位置偏移范围（-1是安全阈值，避免机器人生成在围墙上）
        angle = np.random.uniform(-np.pi, np.pi)
        pos = False
        while not pos:
            x = np.random.uniform(-bias, bias)
            y = np.random.uniform(-bias, bias)
            pos = self.check_position(x, y, self.obs_min_dist)
        self.set_position("turtlebot3_waffle", x, y, angle)
        return x, y

    def set_position(self, name, x, y, angle):
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
        return success

    def set_spawn_and_target_position(self):
        """
        使用 costmap 逻辑：
        1) 由障碍物构建 costmap
        2) 以region_select_bias概率选择最大连通区域，否则随机选择其他连通区域
        3) 在该区域内采样机器人和目标位置（满足距离约束）
        """
        # 基于当前障碍物圆心构建 costmap
        self._build_costmap_from_obstacles()
        region_mask = self._find_largest_free_region()

        if region_mask is None or not region_mask.any():
            # 无可用连通区域，直接失败
            self._log_costmap(region_mask, None, None)
            return False

        # 一次性尝试所有可能的机器人-目标位置组合
        robot_pos, target_pos = self._find_best_robot_target_combination(region_mask)
        
        if robot_pos is None or target_pos is None:
            # 未找到满足条件的组合，打印 costmap 并返回失败
            self._log_costmap(region_mask, None, None)
            return False

        # 使用找到的最佳组合
        rx, ry = robot_pos
        tx, ty = target_pos

        # ========== 到这里说明 costmap 方案已经确定成功 ==========
        # 1) 先在 Gazebo 中真正设置所有障碍物位置
        for idx, (ox, oy) in enumerate(self.obstacle_positions):
            name = f"obstacle{idx+1}"
            # 优先使用预先采样的角度，长度不够则随机一个
            if idx < len(self.obstacle_angles):
                oang = self.obstacle_angles[idx]
            else:
                oang = np.random.uniform(-np.pi, np.pi)
            if not self.set_position(name, ox, oy, oang):
                print(f"[costmap][warn] final set_position failed for {name}")

        # 2) 再设置机器人初始位姿
        angle = np.random.uniform(-np.pi, np.pi)
        self.set_position("turtlebot3_waffle", rx, ry, angle)

        self.episode_start_position = [rx, ry]
        self.target = [tx, ty]

        # 更新元素位置：障碍物 + 机器人 + 目标
        self.element_positions = list(self.obstacle_positions) + [
            self.episode_start_position,
            self.target,
        ]

        # 打印被采纳的 costmap 与机器人/目标位置
        self._log_costmap(region_mask, self.episode_start_position, self.target)
        return True

    def set_positions(self):
        """
        地图生成逻辑：
        1) 先随机/均匀生成障碍物，记录障碍物中心点 self.obstacle_positions
        2) 基于障碍物构建 costmap，并找到最大连通自由区域
        3) 在该连通区域内生成机器人和终点位置（满足距离约束）
        """
        self.element_positions = []
        self.obstacle_positions = []
        # 确保每次重新生成均匀候选点
        if hasattr(self, "candidate_points"):
            delattr(self, "candidate_points")

        max_attempts = 10  # 避免极端情况下无限循环
        for attempt in range(max_attempts):
            self.element_positions = []
            self.obstacle_positions = []
            if hasattr(self, "candidate_points"):
                delattr(self, "candidate_points")

            # 1) 先生成障碍物
            for i in range(0, self.obs_num):
                name = "obstacle" + str(i + 1)
                # 尝试设置障碍物位置，如果实体不存在则跳过
                if not self.set_random_position(name):
                    # 如果设置位置失败（可能是实体不存在），跳过该障碍物
                    continue

            # 2&3) 基于 costmap 的机器人和目标点生成
            if self.set_spawn_and_target_position():
                return True

            print(f"[costmap][warn] set_positions attempt {attempt+1}/{max_attempts} failed, retrying...")

        print("[costmap][error] set_positions failed after max attempts, keep last state")
        return False  # 仍返回False以示失败，但不会回退旧逻辑

    def check_position(self, x, y, min_dist):
        pos = True
        for element in self.element_positions:
            distance_vector = [element[0] - x, element[1] - y]
            distance = np.linalg.norm(distance_vector)
            if distance < min_dist:
                pos = False
        return pos

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
        if goal:
            # return base_goal_reward*np.exp(-0.01 * self.step_count)  # 指数型奖励，用时越少奖励越高
            #print(f"Reached goal in {self.step_count} steps.")
            # 记录本 episode 的目标奖励
            self.episode_goal_reward += self.goal_reward
            return self.goal_reward
        elif collision:
            # 碰撞惩罚：每个 episode 只计算一次
            if not getattr(self, "_has_collision_this_episode", False):
                # 第一次碰撞：记录惩罚
                self._has_collision_this_episode = True
                self.episode_collision_penalty += self.collision_penalty_base
                return self.collision_penalty_base
            else:
                # 同一 episode 后续再次检测到 collision，不再重复扣分
                return 0.0
        else:
            # ================计算最近障碍物距离惩罚================
            
            # 当最近障碍物距离低于阈值时，距离越接近0惩罚越大
            # 当distance >= threshold时，惩罚为0
            # 支持动态阈值：配置为-1时，阈值按|v| * sim_time计算
            if self.enable_obs_penalty:
                # 获取最近障碍物距离（分段加权）
                if self.obs_penalty_threshold < 0:
                    # 动态阈值：按 |线速度| * sim_time 计算，但不低于 min_obs_penalty_threshold
                    threshold = max(self.min_obs_penalty_threshold, abs(action[0]) * self.sim_time)
                else:
                    # 固定阈值：直接使用配置值
                    threshold = self.obs_penalty_threshold
                
                def calc_penalty(dist):
                    if dist < threshold:
                        return self.obs_penalty_base * np.power(threshold - dist, self.obs_penalty_power)
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
            else:
                obs_penalty = 0.0
            # ================计算最近障碍物距离惩罚================ 

            # ================计算角速度惩罚================
            if self.enable_yawrate_penalty:
                yawrate_penalty = self.yawrate_penalty_base * abs(action[1])
            else:
                yawrate_penalty = 0.0
            # ================计算角速度惩罚================

            # ================计算角度偏移惩罚================
            # 计算当前角度（弧度）
            current_angle = math.atan2(sin, cos)
            # 理想角度（正对目标点）
            target_angle = 0.0
            # 计算最小角度差（考虑圆周性）
            angle_diff = abs(math.atan2(math.sin(current_angle - target_angle), 
                                    math.cos(current_angle - target_angle)))
            # 角度惩罚（假设角度差在0到π范围内，超过π则取反）
            if self.enable_angle_penalty:
                angle_penalty = self.angle_penalty_base * (1 - math.cos(angle_diff)) / 2
            else:
                angle_penalty = 0
            # ================计算角度偏移惩罚==========================

            # ================线速度惩罚================
            if self.enable_linear_penalty:
                # 线速度越接近最大速度，惩罚越小
                linear_penalty = self.linear_penalty_base * (self.max_velocity - action[0])
            else:
                linear_penalty = 0
            # ================线速度惩罚================
            
            # ================终点距离惩罚================
            # 根据当前终点距离/生成终点时的真实终点距离计算惩罚
            # 若当前终点距离为3，生成终点时的真实终点距离为6，则惩罚为3/6=0.5
            if self.enable_target_distance_penalty and self.initial_target_distance is not None and self.initial_target_distance > 0:
                # 计算当前距离与初始距离的比值
                distance_ratio = distance / self.initial_target_distance
                # 惩罚 = base * 距离比值（距离越远，比值越大，惩罚越大）
                target_distance_penalty = self.target_distance_penalty_base * distance_ratio
            else:
                target_distance_penalty = 0.0
            # ================终点距离惩罚================
            
            # ================线速度加速度震荡惩罚================
            # 根据当前action的线速度和上一action的线速度计算加速度
            # 如果当前加速度和上一加速度的符号不同，则给予惩罚
            linear_acceleration_oscillation_penalty = 0.0
            if self.enable_linear_acceleration_oscillation_penalty:
                current_linear_velocity = action[0]
                step_time = self.step_sleep_time  # 每个step的持续时间
                
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
            
            # 记录本 step 奖励分量到当前 episode 统计
            self.episode_obs_penalty += obs_penalty
            self.episode_yawrate_penalty += yawrate_penalty
            self.episode_angle_penalty += angle_penalty
            self.episode_linear_penalty += linear_penalty
            self.episode_target_distance_penalty += target_distance_penalty
            self.episode_linear_acceleration_oscillation_penalty += linear_acceleration_oscillation_penalty
            self.episode_yawrate_oscillation_penalty += yawrate_oscillation_penalty
            if self.reward_debug:
                parts = []
                if self.enable_yawrate_penalty:
                    parts.append(f"yawrate_penalty:{yawrate_penalty:.4f}")
                if self.enable_obs_penalty:
                    parts.append(f"obs_penalty:{obs_penalty:.4f}")
                if self.enable_angle_penalty:
                    parts.append(f"angle_penalty:{angle_penalty:.4f}")
                if self.enable_linear_penalty:
                    parts.append(f"linear_penalty:{linear_penalty:.4f}")
                if self.enable_target_distance_penalty:
                    parts.append(f"target_distance_penalty:{target_distance_penalty:.4f}")
                if self.enable_linear_acceleration_oscillation_penalty:
                    parts.append(f"linear_acc_osc_penalty:{linear_acceleration_oscillation_penalty:.4f}")
                if self.enable_yawrate_oscillation_penalty:
                    parts.append(f"yawrate_osc_penalty:{yawrate_oscillation_penalty:.4f}")
                print(" | ".join(parts))
            
            return yawrate_penalty + obs_penalty + angle_penalty + linear_penalty + target_distance_penalty + linear_acceleration_oscillation_penalty + yawrate_oscillation_penalty

    @staticmethod
    def cossin(vec1, vec2):
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        cos = np.dot(vec1, vec2)
        sin = np.cross(vec1, vec2).item()

        return cos, sin
