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
        obs_min_dist = 0,  # 障碍物圆心最小距离（单位米）
        obs_num = 8, # 默认8
        env_id = 0,  # 环境ID，用于topic命名空间
        # 障碍物生成方式
        obs_distribution_mode="uniform",  # "uniform"=均匀分布（当前默认策略），"random"=完全随机分布
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
        reward_debug=False,  # 是否打印奖励/惩罚明细
        # 障碍物距离惩罚参数
        obs_penalty_threshold=1.0,  # 障碍物距离惩罚阈值（米），低于此值开始惩罚；设为-1时根据速度自动计算（|v| * sim_time）
        min_obs_penalty_threshold=0.5,  # 动态计算阈值时的最小阈值下限，避免阈值过小
        obs_penalty_base=-10.0,  # 障碍物距离惩罚基础系数
        obs_penalty_power=2.0,  # 障碍物距离惩罚指数，值越大惩罚增长越快
        obs_penalty_high_weight=1.0,  # 中间高权重区域惩罚权重
        obs_penalty_low_weight=0.5,  # 两侧低权重区域惩罚权重
        obs_penalty_middle_ratio=0.4,  # 中间高权重区域比例（0-1）
        # 时间控制参数
        sim_time=0.1,  # 仿真步长，用于基于速度的动态阈值等
        step_sleep_time=0.1,  # step方法中的sleep时间（秒）
        eval_sleep_time=1.0,  # eval方法中的sleep时间（秒）
        reset_step_count=3  # reset方法中调用step的次数
    ):
        rclpy.init(args=args)
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
        self.max_velocity = max_velocity
        self.target = None
        self.episode_start_position = None  # 记录每个episode开始时的机器人位置
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
        # 时间控制参数
        self.sim_time = sim_time
        self.step_sleep_time = step_sleep_time
        self.eval_sleep_time = eval_sleep_time
        self.reset_step_count = reset_step_count
        # 奖励分解统计（按 episode 累积）
        self.reset_episode_reward_breakdown()
        self.reset()

    def reset_episode_reward_breakdown(self):
        """重置当前 episode 的奖励分解统计"""
        self.episode_obs_penalty = 0.0
        self.episode_yawrate_penalty = 0.0
        self.episode_angle_penalty = 0.0
        self.episode_linear_penalty = 0.0
        self.episode_goal_reward = 0.0
        self.episode_collision_penalty = 0.0
        # 标记本 episode 是否已经发生过碰撞（用于保证碰撞惩罚每个 episode 只记一次）
        self._has_collision_this_episode = False

    def step(self, lin_velocity=0.0, ang_velocity=0.0):
        self.step_count+=1
        self.cmd_vel_publisher.publish_cmd_vel(lin_velocity, ang_velocity)
        self.physics_client.unpause_physics()
        time.sleep(self.step_sleep_time)
        rclpy.spin_once(self.sensor_subscriber)
        self.physics_client.pause_physics()

        (
            latest_scan,
            latest_position,
            latest_orientation,
        ) = self.sensor_subscriber.get_latest_sensor()
        if latest_scan is None:
            # 创建默认激光数据（360个点，距离10米）
            print("No laser scan data received, using default values.")
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

    def reset(self):
        # 重置奖励分解统计
        self.reset_episode_reward_breakdown()
        self.step_count = 0  # 重置计步
        self.world_reset.reset_world()
        action = [0, 0]
        self.cmd_vel_publisher.publish_cmd_vel(
            linear_velocity=action[0], angular_velocity=action[1]
        )
        position_set = self.set_positions()
        while not position_set:
            print("Failed to set positions, retrying...")
            position_set = self.set_positions()

        self.publish_target.publish(self.target[0], self.target[1])
        # 在Gazebo中设置目标圆柱体位置（z=1.0 使圆柱体在地面上方1米，不阻挡激光扫描）
        self.goal_model_client.set_goal_position(self.target[0], self.target[1], z=1.0)
        # 确保服务调用完成
        rclpy.spin_once(self.goal_model_client, timeout_sec=0.1)
        time.sleep(0.05)  # 短暂延迟，确保Gazebo处理位置设置
        # 要调用多次step，否则会因为gazebo加载等相关问题引发的错误
        for _ in range(self.reset_step_count):
            latest_scan, distance, cos, sin, collision, goal, action, reward = self.step(
                lin_velocity=action[0], ang_velocity=action[1]
            )
            if collision:
                print("Collision detected in reset duration")
            elif goal:
                print("Goal detected in reset duration")
        return latest_scan, distance, cos, sin, collision, goal, action, reward

    def set_target_position(self, robot_position):
        pos = False
        # world_size 为地图边长，地图中心为 (0, 0)，地图有效范围为 [-world_size/2, world_size/2]
        map_half = self.world_size / 2.0
        # 留出一定安全边界，避免目标生成在围墙上或刚好贴边
        margin = max(self.target_reached_delta, self.obs_min_dist, 0.5)
        while not pos:
            # 使用极坐标采样，确保实际距离在 [target_reached_delta, target_dist] 范围内
            # 采样角度（0到2π）
            theta = np.random.uniform(0, 2 * np.pi)
            # 采样距离（在 [target_reached_delta, target_dist] 范围内）
            r = np.random.uniform(self.target_reached_delta+0.1, self.target_dist)
            # 转换为笛卡尔坐标
            dist_x = r * np.cos(theta)
            dist_y = r * np.sin(theta)
            x = robot_position[0] + dist_x
            y = robot_position[1] + dist_y

            # 同时保证目标中心在地图内（考虑安全边界 margin）
            if not (-map_half + margin <= x <= map_half - margin and
                    -map_half + margin <= y <= map_half - margin):
                continue

            pos = self.check_position(x, y, self.obs_min_dist)
        self.element_positions.append([x, y])
        return [x, y]

    def set_random_position(self, name):
        """根据配置选择障碍物生成方式：均匀分布或随机分布"""
        bias = self.world_size/2 - self.obs_min_dist/2
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
            
            # 检查是否满足所有距离条件
            if self.check_position(x, y, self.obs_min_dist):
                self.element_positions.append([x, y])
                # 设置障碍物位置，如果实体不存在则返回False
                if not self.set_position(name, x, y, angle):
                    # 实体不存在，移除已添加的位置并返回False
                    self.element_positions.pop()
                    return False
                return True
            
            attempts += 1
        
        # 如果候选点用完或全部尝试失败：均匀分布模式下直接返回失败，
        # 由上层逻辑决定是否整体重试均匀分布，而不是退回到随机分布
        return False

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
            if self.check_position(x, y, self.obs_min_dist):
                self.element_positions.append([x, y])
                # 设置障碍物位置，如果实体不存在则返回False
                if not self.set_position(name, x, y, angle):
                    # 实体不存在，移除已添加的位置并返回False
                    self.element_positions.pop()
                    return False
                return True
        return False

    def set_robot_position(self):
        bias = self.world_size/2 - 1 # 机器人生成位置偏移范围（-1是安全阈值，避免机器人生成在围墙上）
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
        pose.position.z = 0.0
        pose.orientation.x = quaternion.x
        pose.orientation.y = quaternion.y
        pose.orientation.z = quaternion.z
        pose.orientation.w = quaternion.w

        success = self.robot_state_publisher.set_state(name, pose)
        return success

    def set_spawn_and_target_position(self):
        self.episode_start_position = self.set_robot_position()
        self.target = self.set_target_position(self.episode_start_position)
        self.element_positions.append(self.episode_start_position)
        # print(f"Robot position set to: {robot_position}")
        # print(f"Target position set to: {self.target}")

    def set_positions(self):
        self.element_positions = []
        self.set_spawn_and_target_position()

        for i in range(0, self.obs_num):
            name = "obstacle" + str(i + 1)
            # 尝试设置障碍物位置，如果实体不存在则跳过
            if not self.set_random_position(name):
                # 如果设置位置失败（可能是实体不存在），跳过该障碍物
                # 不返回False，继续设置其他障碍物
                continue
        return True  # 成功设置所有位置，返回True

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
            # 记录本 step 奖励分量到当前 episode 统计
            self.episode_obs_penalty += obs_penalty
            self.episode_yawrate_penalty += yawrate_penalty
            self.episode_angle_penalty += angle_penalty
            self.episode_linear_penalty += linear_penalty
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
                print(" | ".join(parts))
            
            return yawrate_penalty + obs_penalty + angle_penalty+linear_penalty

    @staticmethod
    def cossin(vec1, vec2):
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        cos = np.dot(vec1, vec2)
        sin = np.cross(vec1, vec2).item()

        return cos, sin
