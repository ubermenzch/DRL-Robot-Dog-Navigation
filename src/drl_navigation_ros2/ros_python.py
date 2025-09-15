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
        max_steps = 300,
        world_size = 15, # 单位
        obs_min_dist = 2,  # 障碍物圆心最小距离（单位米）
        obs_num = 20 # 默认30
    ):
        rclpy.init(args=args)
        self.cmd_vel_publisher = CmdVelPublisher()
        self.scan_subscriber = ScanSubscriber()
        self.odom_subscriber = OdomSubscriber()
        self.robot_state_publisher = SetModelStateClient()
        self.world_reset = ResetWorldClient()
        self.physics_client = PhysicsClient()
        self.publish_target = MarkerPublisher()
        self.element_positions = []
        self.sensor_subscriber = SensorSubscriber()
        self.target_dist = init_target_distance
        self.target_dist_increase = target_dist_increase
        self.max_target_dist = max_target_dist
        self.target_reached_delta = target_reached_delta
        self.collision_delta = collision_delta
        self.step_count = 0
        self.env_count = 0 # 初始值设置为-1抵消初始化的自增
        self.goal_count = 0
        self.collision_count = 0
        self.neglect_angle = neglect_angle
        self.scan_range = scan_range
        self.max_steps = max_steps
        self.world_size = world_size  # 单位米
        self.obs_min_dist = obs_min_dist  # 障碍物圆心最小距离（单位米）
        self.obs_num  = obs_num
        self.target = None
        self.episode_start_position = None  # 记录每个episode开始时的机器人位置
        self.reset()

    def step(self, lin_velocity=0.0, ang_velocity=0.0):
        self.step_count+=1
        self.cmd_vel_publisher.publish_cmd_vel(lin_velocity, ang_velocity)
        self.physics_client.unpause_physics()
        time.sleep(0.1)
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
        self.env_count+=1
        self.step_count = 0  # 重置计步
        self.world_reset.reset_world()
        action = [0.0, 0.0]
        self.cmd_vel_publisher.publish_cmd_vel(
            linear_velocity=action[0], angular_velocity=action[1]
        )
        position_set = self.set_positions()
        while not position_set:
            print("Failed to set positions, retrying...")
            position_set = self.set_positions()

        self.publish_target.publish(self.target[0], self.target[1])

        latest_scan, distance, cos, sin, _, _, action, reward = self.step(
            lin_velocity=action[0], ang_velocity=action[1]
        )
        return latest_scan, distance, cos, sin, False, False, action, reward

    def eval(self, scenario):
        self.cmd_vel_publisher.publish_cmd_vel(0.0, 0.0)

        self.target = [scenario[-1].x, scenario[-1].y]
        self.publish_target.publish(self.target[0], self.target[1])

        for element in scenario[:-1]:
            self.set_position(element.name, element.x, element.y, element.angle)

        self.physics_client.unpause_physics()
        time.sleep(1)
        latest_scan, distance, cos, sin, _, _, a, reward = self.step(
            lin_velocity=0.0, ang_velocity=0.0
        )
        return latest_scan, distance, cos, sin, False, False, a, reward

    def set_target_position(self, robot_position):
        pos = False
        bias = self.world_size/2 - 1 # 目标生成位置偏移范围（-1是安全阈值，避免目标生成在围墙上）
        while not pos:
            x = np.clip(
                robot_position[0]
                + np.random.uniform(-self.target_dist, self.target_dist),
                -bias,
                bias,
            )
            y = np.clip(
                robot_position[1]
                + np.random.uniform(-self.target_dist, self.target_dist),
                -bias,
                bias,
            )
            pos = self.check_position(x, y, self.obs_min_dist)
        self.element_positions.append([x, y])
        return [x, y]

    # def set_random_position(self, name):
    #     bias = self.world_size/2-self.obs_min_dist/2
    #     angle = np.random.uniform(-np.pi, np.pi)
    #     pos = False
    #     try_time = 0
    #     try_max = 1000
    #     while not pos and try_time < try_max:
    #         try_time += 1
    #         x = np.random.uniform(-bias, bias)
    #         y = np.random.uniform(-bias, bias)
    #         pos = self.check_position(x, y, self.obs_min_dist)
    #     # print(f"Set position for {name}: x={x}, y={y}, angle={angle}")
    #     # print("try time: ", try_time)
    #     if try_time >= try_max:
    #         return False  # 如果尝试次数超过最大值，返回False表示设置位置失败
    #     self.element_positions.append([x, y])
    #     self.set_position(name, x, y, angle)
    #     return True  # 成功设置位置，返回True

    def set_random_position(self, name):
        bias = self.world_size/2 - self.obs_min_dist/2
        angle = np.random.uniform(-np.pi, np.pi)
        
        # ==== 改进的障碍物分布策略 ====
        # 使用改进的Farthest Point采样实现均匀分布
        if not hasattr(self, 'candidate_points') or self.env_count % 5 == 0:
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
                self.set_position(name, x, y, angle)
                return True
            
            attempts += 1
        
        # 后备方案：纯随机采样
        return self.fallback_random_position(name, bias, angle)

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
                self.set_position(name, x, y, angle)
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

        self.robot_state_publisher.set_state(name, pose)
        rclpy.spin_once(self.robot_state_publisher)

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
            if not self.set_random_position(name):
                return False  # 如果设置位置失败，返回False
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
            self.goal_count += 1
            base_goal_reward = 100.0  # 基础目标奖励
            # return base_goal_reward*np.exp(-0.01 * self.step_count)  # 指数型奖励，用时越少奖励越高
            goal_reward = 1000
            #print(f"Reached goal in {self.step_count} steps. Total goals: {self.goal_count}")
            return  goal_reward # 达到目标基础奖励100+避障奖励180+角速度惩罚147，保证到达终点奖励非负
        elif collision:
            base_collision_penalty = -500.0  # 基础碰撞惩罚
            self.collision_count += 1
            collision_rate = self.collision_count / self.env_count if self.env_count > 0 else 0
            # 惩罚随碰撞率增加，碰撞惩罚最大为base_collision_penalty*e=-542
            collision_penalty = base_collision_penalty * np.exp(collision_rate)
            # print(f"Collision occurred! Total collisions: {self.collision_count}, Collision rate: {collision_rate:.3f}")
            return collision_penalty
        else:
            # 计算最近障碍物距离惩罚
            obs_penalty_base = 0
            obs_x = np.mean(laser_scan)
            # print(f"Laser scan mean: {obs_x}")
            obs_c = -2.0 # exp变量的系数
            obs_penalty = obs_penalty_base * np.exp(obs_c*obs_x) # 400步下，每步平均障碍物距离最小时，该惩罚总为-180

            #计算角速度惩罚
            base_yawrate_penalty = -4
            yawrate_x = abs(action[1])
            yawrate_c = 0.4 # exp变量的系数
            yawrate_penalty = base_yawrate_penalty*(np.exp(yawrate_c*yawrate_x)-1) # 1500步下，每步角速度值最大时，该惩罚总为-147
            
            # ================计算角度偏移惩罚==========================
            # 计算当前角度（弧度）
            current_angle = math.atan2(sin, cos)
            # 理想角度（正对目标点）
            target_angle = 0.0
            # 计算最小角度差（考虑圆周性）
            angle_diff = abs(math.atan2(math.sin(current_angle - target_angle), 
                                    math.cos(current_angle - target_angle)))
            # 角度惩罚（假设角度差在0到π范围内，超过π则取反）
            angle_base_penalty = -1.0 # 假设角度偏差基础惩罚
            angle_penalty = angle_base_penalty * (1 - math.cos(angle_diff)) / 2
            # ================计算角度偏移惩罚==========================

            # 线速度奖励（线速度越大奖励越大)-角速度绝对值惩罚（绝对值越大惩罚越大)-障碍物距离惩罚（障碍物距离越小惩罚越大）
            #print(f"Yawrate penalty: {yawrate_penalty} Obs penalty: {obs_penalty}")
            return yawrate_penalty + obs_penalty + angle_penalty

    @staticmethod
    def cossin(vec1, vec2):
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        cos = np.dot(vec1, vec2)
        sin = np.cross(vec1, vec2).item()

        return cos, sin
