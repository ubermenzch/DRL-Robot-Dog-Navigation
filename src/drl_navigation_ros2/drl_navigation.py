from ros_dog_nodes import (
    TargetPathSubscriber,
    GnssSubscriber,
    DogCmdVelPublisher,
    ReusedCostMapSubscriber,
    OdomSubscriber,
)
from apscheduler.schedulers.background import BackgroundScheduler
import math
from SAC.SAC_dog import SAC
import rclpy
import utils
import numpy as np
import threading
from colorama import Fore, Back, Style

class Agent:
    def __init__(self,
    robot_type,
    is_indoor_test=False,
    angle_resolution=1,  # 角度分辨率（单位：度）
    scan_range=4.5,
    output_rate=10.0,
    target_treshold=10.0,
    is_navigation_test = False,  # 是否是导航测试（原地不动判断避障倾向）
    target_path_topic='/navigation_control2',  # 目标路径话题名称
    costmap_topic='/planner/reused_costmap',  # 代价地图话题名称
    ):
        # 参数初始化
        self.robot_type = robot_type # robot_type = {ysc,unitree}
        self.is_indoor_test=is_indoor_test
        self.is_navigation_test = is_navigation_test  # 是否是导航测试（原地不动判断避障倾向）
        self.angle_resolution = angle_resolution
        self.scan_range = scan_range  # 传感器扫描范围
        self.output_rate = output_rate  # 输出频率
        self.target_treshold = target_treshold  # 到达目标点距离阈值
        self.action_dim = 2  # number of actions produced by the model
        self.max_action = 1  # maximum absolute value of output actions
        self.state_dim = 25  # number of input values in the neural network (vector length of state input)
        self.bin_num=self.state_dim-5
        self.target_gnss = None  # 当前目标点的GNSS坐标
        self.target_distance = None  # 当前目标点距离智能体的距离
        self.passed_target_count = 0  # 经过的目标点数量
        self.action = [0.0, 0.0]  # 当前动作，线速度和角速度
        self.last_action = [0.0, 0.0]  # 上一个动作，线速度和角速度
        self.max_velocity = 0.65  # 最大线速度
        self.max_yawrate = 0.5  # 最大角速度绝对值
        self.target_sin = 0.0  # 目标点相对于智能体的sin值
        self.target_cos = 1.0  # 目标点相对于智能体的cos值
        self.target_angle = 0.0
        print("Parameters Initialized")

        self.model = SAC(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            max_action=self.max_action,
            load_model=True,
        )
        print("SAC Model Initialized")

        rclpy.init()
        self.dog_cmd_vel_publisher=DogCmdVelPublisher(robot_type = self.robot_type)
        self.target_path_subscriber=TargetPathSubscriber(topic_name=target_path_topic)
        self.gnss_subscriber = GnssSubscriber(robot_type=self.robot_type)
        self.reused_costmap_subscriber = ReusedCostMapSubscriber(angle_resolution=(self.angle_resolution/180)*math.pi,
        scan_range=self.scan_range,
        output_rate=self.output_rate,
        bin_num=self.bin_num,
        topic_name=costmap_topic
        ) 
        self.odom_subscriber = OdomSubscriber(self.robot_type)
        scheduler = BackgroundScheduler()
        scheduler.add_job(self.visualize, 'interval', seconds=(1.0/self.output_rate)*2)
        scheduler.start()
        # self.vision_timer = self.create_timer(1.0/self.output_rate,self.vision_timer)
        # 多线程激活不同节点
        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.target_path_subscriber)
        self.executor.add_node(self.gnss_subscriber)
        self.executor.add_node(self.reused_costmap_subscriber)
        self.executor.add_node(self.odom_subscriber)
        exec_thread = threading.Thread(target=self.executor.spin, daemon=True)
        exec_thread.start()
        print("Publisher and Subscriber Initialized")

        print(f"{self.robot_type} Robot Dog Initialized")

    def calc_target_distance(self):
        if self.is_indoor_test:
            return self.target_treshold*2  # 如果是室内测试，直接返回一个较大的距离
        if self.robot_type == "ysc":
            distance = utils.distance(self.gnss_subscriber.gnss['latitude'],
            self.gnss_subscriber.gnss['longitude'],
            self.target_gnss['latitude'],
            self.target_gnss['longitude']
            )
        elif self.robot_type == "unitree":
            gcjLat, gcjLng = utils.wgs2gcj(self.gnss_subscriber.gnss['latitude'],self.gnss_subscriber.gnss['longitude'])
            distance = utils.distance(gcjLat,
            gcjLng,
            self.target_gnss['latitude'],
            self.target_gnss['longitude']
            )
        return distance
    
    def calc_target_angle(self):
        if self.is_indoor_test:
            return 1.0, 0.0, 0.0  # 如果是室内测试，直接返回正前方（机器狗正朝着目标点）
        agent_vector = np.array([np.cos(self.odom_subscriber.heading), np.sin(self.odom_subscriber.heading)])
        target_vector = np.array([
            self.target_gnss['longitude'] - self.gnss_subscriber.gnss['longitude'],
            self.target_gnss['latitude'] - self.gnss_subscriber.gnss['latitude']
            ])
        cos = 1.0
        sin = 0.0
        target_vector_norm = np.linalg.norm(target_vector) # 计算目标向量的模
        if target_vector_norm > 1e-6:
            #   点积 = |a|·|b|·cosθ → cosθ = (a·b) / (|a||b|)
            #   叉积 = |a|·|b|·sinθ → sinθ = (a×b) / (|a||b|)
            dot_product = np.dot(agent_vector, target_vector)
            cross_product = np.cross(agent_vector, target_vector)  # 2D叉积返回标量
            cos = dot_product /  target_vector_norm
            sin = cross_product / target_vector_norm

        # 使用atan2(sin, cos)计算角度（单位：弧度）
        angle_rad = math.atan2(sin, cos)
        # 将弧度转换为角度
        angle_deg = round(math.degrees(angle_rad),2)

        return cos, sin, angle_deg

    def update_target_gnss(self):
        if self.is_indoor_test:
            # 室内测试模式：设置一个虚拟目标点用于避障测试
            if not self.target_gnss:
                self.target_gnss = {'latitude': 0.0, 'longitude': 0.0}  # 虚拟目标点
            self.target_distance = self.calc_target_distance()
            return
        
        if self.target_gnss: # 如果有目标点
            self.target_distance = self.calc_target_distance()
            if self.target_distance <= self.target_treshold:  # 如果到达目标点
                self.passed_target_count += 1  # 经过目标点计数
                if self.target_path_subscriber.has_target_path():# 如果还有目标路径
                    self.target_gnss = self.target_path_subscriber.get_next_target_gnss()  # 获取下一个目标点
                    self.target_distance = self.calc_target_distance()
                else: # 如果没有目标路径
                    self.target_gnss = None
                    self.target_distance = None
                    self.passed_target_count = 0  # 重置经过目标点计数
        elif self.target_path_subscriber.has_target_path():
            self.target_gnss = self.target_path_subscriber.get_next_target_gnss()
            self.target_distance = self.calc_target_distance()
        else:
            self.target_gnss = None
            self.target_distance = None
            self.passed_target_count = 0  # 重置经过目标点计数

        
        
    def visualize(self,extra_info=" "):
        """
        实时可视化OccupancyGrid消息数据 - 增量更新完整地图版本
        
        符号表示:
            '#' - 值100 (障碍物)
            '?' - 值-1 (未知区域)
            'D' - 机器狗当前位置 (中心) - 蓝色
            '*' - 机器狗预测轨迹 (黄色)
        """
        # 记录当前输出行数，用于下次更新
        if not hasattr(self, '_last_viz_lines'):
            self._last_viz_lines = 0
            # 首次执行时清屏
            print("\033c", end="")
        else:
            # 非首次执行，移动光标到起始位置
            print(f"\033[{self._last_viz_lines}A", end="")
        
        # 信息显示部分
        lines = []
        info_dist = {
            '是否是室内测试': self.is_indoor_test,
            '目标点距离': f"{self.target_distance:.2f}m" if self.target_gnss else "无目标",
            '目标点方位': f"{self.target_angle:.2f}°" if self.target_angle else "无",
            '目标点进度': f"{self.passed_target_count}/{self.target_path_subscriber.total_point_count}",
            'GNSS状态': "有效" if self.gnss_subscriber.is_gnss_valid else "无效",
            '复用地图状态': "有效" if self.reused_costmap_subscriber.is_reused_costmap_valid else "无效"
        }
        
        for key, value in info_dist.items():
            lines.append(f"{key}：{value}")
        
        # 动作信息
        if self.action:
            v, w = self.action
            lines.append(f"当前动作: 线速度={v:.2f}m/s, 角速度={w:.2f}rad/s")
        
        # 如果没有有效的地图数据
        if not self.reused_costmap_subscriber.is_reused_costmap_valid:
            # 打印所有行并记录行数
            print("\n".join(lines))
            self._last_viz_lines = len(lines)
            return
        
        # 获取地图参数
        edge_length = self.reused_costmap_subscriber.reused_costmap.info.height
        resolution = self.reused_costmap_subscriber.reused_costmap.info.resolution
        
        # 只打印整个地图一次，后续使用光标控制
        if not hasattr(self, '_map_grid'):
            # 第一次初始化地图缓存
            self._map_grid = [[' ' for _ in range(edge_length)] for _ in range(edge_length)]
            self._map_changed = [[False for _ in range(edge_length)] for _ in range(edge_length)]
        else:
            # 重置变更标记
            self._map_changed = [[False for _ in range(edge_length)] for _ in range(edge_length)]
        
        # 获取机器狗当前位置（地图中心）
        map_center = (edge_length/2,edge_length/2)
        trajectory_points = utils.calculate_trajectory(map_center, self.action, resolution=resolution,edge_length=edge_length)

        # 更新地图缓存中需要变化的部分
        for y in range(edge_length):
            for x in range(edge_length):
                # 当前值
                value = self.reused_costmap_subscriber.reused_costmap.data[
                    self.reused_costmap_subscriber.x_y_to_index(x, y, edge_length)]
                pos = (x, y)
                
                if pos == map_center: # 检查是否为机器狗当前位置
                    new_char = "\033[1;34mD\033[0m"  # 蓝色粗体 D
                elif pos in trajectory_points: # 检查是否为预测轨迹点
                    new_char = "\033[1;33m*\033[0m"  # 黄色粗体 * (1;33m)
                elif value == 100:
                    new_char = '#'
                elif value == -1:
                    new_char = '?'
                else:
                    new_char = ' '
                
                # 检查是否需要更新
                if self._map_grid[y][x] != new_char:
                    self._map_grid[y][x] = new_char
                    self._map_changed[y][x] = True
        
        # 构建地图输出（只包含需要重绘的行）
        map_lines = []
        for y in range(edge_length-1, -1, -1):  # 从顶部行开始（地图坐标系）
            row_changed = any(self._map_changed[y])
            
            if row_changed or not hasattr(self, '_prev_map'):
                # 构建整行字符串
                row_str = ''.join(self._map_grid[y])
                map_lines.append(row_str)
        
        # 添加地图到输出
        lines.extend(map_lines)
        
        # 打印所有行并记录行数
        print("\n".join(lines))
        obs_min_distance = self.reused_costmap_subscriber.get_obs_min_distance()
        print(f"前方障碍物距离（分区）：{obs_min_distance} (len)={len(obs_min_distance)}")
        self._last_viz_lines = len(lines)

    def run(self):
        self.last_action = [0.0,0.0]
        while True:
            self.update_target_gnss()
            # 若数据未全部有效，则发送停止指令（线速度、角速度均为0的控制指令）
            if bool(self.target_gnss) and (self.is_indoor_test or self.gnss_subscriber.is_gnss_valid) and self.reused_costmap_subscriber.is_reused_costmap_valid:
                # 计算智能体前方180度范围内20个分区中距离自身最近的障碍物距离
                obs_min_distance = self.reused_costmap_subscriber.get_obs_min_distance()
                #print(obs_min_distance)

                # 计算目标相对于智能体的角度（以智能体坐标系为基准）
                self.traget_cos, self.traget_sin, self.target_angle = self.calc_target_angle()

                # 准备深度强化模型输入
                state = self.model.prepare_state(obs_min_distance, self.target_distance, self.traget_cos, self.traget_sin, self.last_action)
                # print(f"state:{state}")
                self.action = self.model.get_action(state, add_noise=False)
                self.action = [(self.action[0] + 1) / (2/self.max_velocity), self.action[1]*self.max_yawrate] # 把线速度从[-1,1]规范到[0,1]
            else:
                self.action = [0.0, 0.0]  # 停止动作
            if self.is_navigation_test:
                self.dog_cmd_vel_publisher.publish_cmd_vel(linear_velocity=0.0,angular_velocity=0.0)
                self.last_action = [0.0,0.0]
            else:
                self.dog_cmd_vel_publisher.publish_cmd_vel(linear_velocity=self.action[0],angular_velocity=self.action[1])
                self.last_action=self.action
            

def main(args=None):
    agent = Agent(robot_type="ysc",is_indoor_test=False,is_navigation_test=False)
    agent.run()

if __name__ == "__main__":
    main()