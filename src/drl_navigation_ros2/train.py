



from pathlib import Path
import yaml
import os
import shutil

from SAC.SAC import SAC
from ros_python import ROS_env
from replay_buffer import ReplayBuffer
import torch
import numpy as np
import utils
from pretrain_utils import Pretraining
from datetime import datetime
from collections import deque
import time

def concatenate_state_history(current_state, state_history, state_history_steps, base_state_dim):
    """将当前state与历史state拼接
    
    Args:
        current_state: 当前step的状态（list或numpy array）
        state_history: 历史state队列（deque）
        state_history_steps: 包含历史多少step（例如：2表示包含当前step和之前2个step，共3个step）
        base_state_dim: 基础状态维度（单个时间步的状态向量长度）
    
    Returns:
        拼接后的state（list），如果state_history_steps为0则只返回当前state
        如果历史不足，用零填充
    """
    if state_history_steps <= 0:
        return list(current_state)
    
    # 计算需要的历史步数（不包括当前step）
    needed_history_steps = state_history_steps
    history_list = list(state_history)
    available_history_steps = len(history_list)
    
    # 拼接：历史state（从旧到新）+ 当前state
    concatenated = []
    
    # 如果历史不足，用零填充
    missing_steps = needed_history_steps - available_history_steps
    if missing_steps > 0:
        zero_state = [0.0] * base_state_dim
        for _ in range(missing_steps):
            concatenated.extend(zero_state)
    
    # 添加实际的历史state
    for hist_state in history_list:
        concatenated.extend(hist_state)
    
    # 最后添加当前state
    concatenated.extend(current_state)
    
    return concatenated

class EpisodeStatistics:
    """Episode统计信息管理器"""
    
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.recent_episodes = deque(maxlen=window_size)
        self.total_episodes = 0
    
    def add_episode_result(self, goal, collision, timeout, reward):
        """添加一个episode的结果"""
        self.total_episodes += 1
        episode_data = {
            'goal': goal,
            'collision': collision,
            'timeout': timeout,
            'reward': reward
        }
        self.recent_episodes.append(episode_data)
    
    def get_statistics(self):
        """获取统计信息"""
        if len(self.recent_episodes) == 0:
            return {
                'total_episodes': self.total_episodes,
                'window_size': 0,
                'goal_rate': 0.0,
                'collision_rate': 0.0,
                'timeout_rate': 0.0,
                'avg_reward': 0.0
            }
        
        window_size = len(self.recent_episodes)
        goals = sum(1 for ep in self.recent_episodes if ep['goal'])
        collisions = sum(1 for ep in self.recent_episodes if ep['collision'])
        timeouts = sum(1 for ep in self.recent_episodes if ep['timeout'])
        total_reward = sum(ep['reward'] for ep in self.recent_episodes)
        
        return {
            'total_episodes': self.total_episodes,
            'window_size': window_size,
            'goal_rate': goals / window_size,
            'collision_rate': collisions / window_size,
            'timeout_rate': timeouts / window_size,
            'avg_reward': total_reward / window_size
        }

class TrainingLossTracker:
    """训练损失和样本抽样统计跟踪器"""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.recent_losses = deque(maxlen=window_size)
        self.total_trainings = 0
        self.total_samples_drawn = 0  # 总抽样样本数
        self.total_samples_added = 0  # 总添加样本数
    
    def add_loss(self, critic_loss, batch_size, training_iterations):
        """添加一次训练的损失和抽样数"""
        self.total_trainings += 1
        self.recent_losses.append(critic_loss)
        # 计算本次训练抽样的总数
        self.total_samples_drawn += batch_size * training_iterations
    
    def add_samples_to_buffer(self, num_samples):
        """记录添加到缓冲区的样本数"""
        self.total_samples_added += num_samples
    
    def get_average_loss(self):
        """获取平均损失"""
        if len(self.recent_losses) == 0:
            return 0.0
        return sum(self.recent_losses) / len(self.recent_losses)
    
    def get_loss_count(self):
        """获取记录的损失数量"""
        return len(self.recent_losses)
    
    def get_avg_sample_usage(self):
        """计算样本平均被抽样次数"""
        if self.total_samples_added == 0:
            return 0.0
        return self.total_samples_drawn / self.total_samples_added

def print_statistics_report(stats):
    """打印统计报告"""
    current_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"统计报告 - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Episode总数：{stats['total_episodes']}")
    print(f"统计窗口大小：{stats['window_size']}")
    print(f"平均奖励: {stats['avg_reward']:.2f}")
    print(f"成功率: {stats['goal_rate']:.3f} ({stats['goal_rate']*100:.1f}%)")
    print(f"碰撞率: {stats['collision_rate']:.3f} ({stats['collision_rate']*100:.1f}%)")
    print(f"超时率: {stats['timeout_rate']:.3f} ({stats['timeout_rate']*100:.1f}%)")
    print(f"{'='*60}")

def load_config(config_path=None):
    """加载配置文件"""
    if config_path is None:
        # 默认配置文件路径
        config_path = Path(__file__).parent.parent.parent / "config" / "train.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def main(args=None):
    """Main training function"""
    # ==================== 加载配置文件 ====================
    # 可以通过环境变量指定配置文件路径
    config_path_env = os.environ.get('TRAIN_CONFIG_PATH', None)
    config = load_config(config_path_env)
    
    # 获取实际使用的配置文件路径用于打印
    actual_config_path = config_path_env if config_path_env else (Path(__file__).parent.parent.parent / "config" / "train.yaml")
    
    # ==================== 设置ROS_DOMAIN_ID ====================
    # 从配置文件读取ROS_DOMAIN_ID，如果环境变量已设置则优先使用环境变量
    ros_domain_id = os.environ.get('ROS_DOMAIN_ID', None)
    if ros_domain_id is None:
        ros_domain_id = 77  # 单环境训练默认使用77
        os.environ['ROS_DOMAIN_ID'] = str(ros_domain_id)
    print(f"ROS_DOMAIN_ID: {ros_domain_id}")
    print(f"使用配置文件: {actual_config_path}")
    
    # ==================== 基础参数 ====================
    is_code_debug = config.get('is_code_debug', False)
    
    # ==================== 机器人和环境参数 ====================
    max_velocity = config.get('max_velocity', 1.0)
    max_yawrate = config.get('max_yawrate', 20.0)
    max_acceleration = config.get('max_acceleration', 1.0)
    max_deceleration = config.get('max_deceleration', 1.0)
    neglect_angle = config.get('neglect_angle', 0)
    scan_range = config.get('scan_range', 5)
    max_target_dist = config.get('max_target_dist', 15.0)
    init_target_distance = config.get('init_target_distance', 2.0)
    target_dist_increase = config.get('target_dist_increase', 0.01)
    target_reached_delta = config.get('target_reached_delta', 0.3)
    collision_delta = config.get('collision_delta', 0.25)
    world_size = config.get('world_size', 10)
    goals_per_map = config.get('goals_per_map', 1)
    obs_min_dist = config.get('obs_min_dist', 0.0)
    obs_num = config.get('obs_num', 8)
    costmap_resolution = config.get('costmap_resolution', 0.3)
    obstacle_size = config.get('obstacle_size', 0.3)
    obs_distribution_mode = config.get('obs_distribution_mode', 'uniform')
    
    # ==================== 模型参数 ====================
    action_dim = config.get('action_dim', 2)
    max_action = config.get('max_action', 1)
    base_state_dim = config.get('base_state_dim', 25)
    state_history_steps = config.get('state_history_steps', 0)
    
    # 动态计算state_dim
    if state_history_steps > 0:
        state_dim = base_state_dim * (1 + state_history_steps)
        print(f"启用历史state模式: base_state_dim={base_state_dim}, state_history_steps={state_history_steps}, 最终state_dim={state_dim}")
    else:
        state_dim = base_state_dim
        print(f"未启用历史state模式: state_dim={state_dim}")
    
    hidden_dim = config.get('hidden_dim', 1024)
    hidden_depth = config.get('hidden_depth', 2)
    actor_update_frequency = config.get('actor_update_frequency', 1)
    critic_target_update_frequency = config.get('critic_target_update_frequency', 2)
    
    # 设备选择
    # 注意：如果通过环境变量 CUDA_VISIBLE_DEVICES 设置了 GPU，PyTorch 视角下 GPU 索引从 0 开始
    device_str = config.get('device', 'auto')
    gpu_id = config.get('gpu_id', None)
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    
    if device_str == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    # 如果设置了 CUDA_VISIBLE_DEVICES，PyTorch 只能看到指定的 GPU，索引总是从 0 开始
    # 因此应该使用 cuda 或 cuda:0，忽略配置文件中可能指定的其他 GPU 索引
    if cuda_visible_devices and torch.cuda.is_available() and device.type == 'cuda':
        device = torch.device("cuda")
        print(f"GPU CUDA Ready (物理GPU: {cuda_visible_devices}, PyTorch使用: {device})")
    elif torch.cuda.is_available() and device.type == 'cuda':
        print(f"GPU CUDA Ready (使用设备: {device})")
    else:
        print("CPU Ready")
    
    # ==================== 训练参数 ====================
    max_training_count = config.get('max_training_count', 5000)
    episode = 0  # 当前回合数
    base_report_every = config.get('report_every', 20)
    stats_window_size = config.get('stats_window_size', base_report_every)
    train_every_n = config.get('train_every_n', 2)
    # report_every 由配置决定；若未配置，则与统计窗口保持一致
    report_every = base_report_every
    loss_window_size = config.get('loss_window_size', 10)
    training_iterations = config.get('training_iterations', 500)
    batch_size = config.get('batch_size', 40)
    buffer_size = config.get('buffer_size', 50000)
    max_steps_fixed = config.get('max_steps', 300)
    max_steps_ratio = config.get('max_steps_ratio', 0)
    max_steps_min = config.get('max_steps_min', 50)
    steps = 0  # 当前回合的步数
    pretrain = config.get('pretrain', False)
    pretraining_iterations = config.get('pretraining_iterations', 50)
    load_model = config.get('load_model', True)
    save_every = config.get('save_every', 10)
    
    # ==================== 路径参数 ====================
    load_path = Path(config.get('load_path', "/home/zc/DRL-Robot-Navigation-ROS2/src/drl_navigation_ros2/models/SAC"))
    # 保存目录：以时间戳创建子目录，三个模型文件名保持默认
    # 优先使用环境变量中的时间戳（来自start_training.sh），确保与日志文件时间戳一致
    base_save_path = Path(config.get('save_path', "/home/zc/DRL-Robot-Navigation-ROS2/models/single_env_SAC_15m_newreward"))
    timestamp = os.environ.get('TRAINING_TIMESTAMP', datetime.now().strftime("%Y%m%d_%H%M%S"))
    save_path = base_save_path / timestamp
    pretrain_data_path = config.get('pretrain_data_path', "src/drl_navigation_ros2/assets/data.yml")
    
    # 确保保存目录存在（不再覆盖旧目录）
    save_path.mkdir(parents=True, exist_ok=True)
    # 保存本次训练使用的配置快照
    if actual_config_path and Path(actual_config_path).exists():
        try:
            shutil.copy(actual_config_path, save_path / "config_used.yaml")
        except Exception as e:
            print(f"警告: 复制配置到保存目录失败: {e}")
    
    # ==================== 调试模式覆盖 ====================
    if is_code_debug:
        max_training_count = 50
        train_every_n = 1
        training_iterations = 10
        report_every = 10
        stats_window_size = 10
        loss_window_size = 10
        pretrain = False
    
    # ==================== 奖励函数参数 ====================
    goal_reward = config.get('goal_reward', 1000.0)
    base_collision_penalty = config.get('collision_penalty_base', config.get('base_collision_penalty', -1000.0))
    angle_base_penalty = config.get('angle_penalty_base', config.get('angle_base_penalty', 0.0))
    base_linear_penalty = config.get('linear_penalty_base', config.get('base_linear_penalty', -1.0))
    yawrate_penalty_base = config.get('yawrate_penalty_base', 0.0)
    enable_obs_penalty = config.get('enable_obs_penalty', True)
    enable_yawrate_penalty = config.get('enable_yawrate_penalty', True)
    enable_angle_penalty = config.get('enable_angle_penalty', True)
    enable_linear_penalty = config.get('enable_linear_penalty', True)
    enable_target_distance_penalty = config.get('enable_target_distance_penalty', False)
    enable_linear_acceleration_oscillation_penalty = config.get('enable_linear_acceleration_oscillation_penalty', False)
    enable_yawrate_oscillation_penalty = config.get('enable_yawrate_oscillation_penalty', False)
    reward_debug = config.get('reward_debug', False)
    
    # ==================== 障碍物距离惩罚参数 ====================
    obs_penalty_threshold = config.get('obs_penalty_threshold', 1.0)
    min_obs_penalty_threshold = config.get('min_obs_penalty_threshold', 0.5)
    obs_penalty_base = config.get('obs_penalty_base', -10.0)
    obs_penalty_power = config.get('obs_penalty_power', 2.0)
    obs_penalty_high_weight = config.get('obs_penalty_high_weight', 1.0)
    obs_penalty_low_weight = config.get('obs_penalty_low_weight', 0.5)
    obs_penalty_middle_ratio = config.get('obs_penalty_middle_ratio', 0.4)
    
    # ==================== 终点距离惩罚参数 ====================
    target_distance_penalty_base = config.get('target_distance_penalty_base', -1.0)
    
    # ==================== 震荡惩罚参数 ====================
    linear_acceleration_oscillation_penalty_base = config.get('linear_acceleration_oscillation_penalty_base', -1.0)
    yawrate_oscillation_penalty_base = config.get('yawrate_oscillation_penalty_base', -1.0)
    
    # ==================== 连通区域选择参数 ====================
    region_select_bias = config.get('region_select_bias', 1.0)
    
    # ==================== 时间控制参数 ====================
    sim_time = config.get('sim_time', 0.1)
    step_sleep_time = config.get('step_sleep_time', 0.1)
    eval_sleep_time = config.get('eval_sleep_time', 1.0)
    reset_step_count = config.get('reset_step_count', 3)
    
    # ==================== 动作噪声参数 ====================
    action_noise_std = config.get('action_noise_std', 0.2)
    
    model = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        save_every=save_every,
        load_model=load_model,
        save_directory=save_path,
        load_directory=load_path,
        action_noise_std=action_noise_std,
        hidden_dim=hidden_dim,
        hidden_depth=hidden_depth,
        actor_update_frequency=actor_update_frequency,
        critic_target_update_frequency=critic_target_update_frequency,
        base_state_dim=base_state_dim,  # 传递base_state_dim给SAC模型
    )  # instantiate a model
    print("Model Loaded")
    ros = ROS_env(
        max_velocity=max_velocity,
        neglect_angle=neglect_angle,
        scan_range=scan_range,
        max_target_dist=max_target_dist,
        init_target_distance=init_target_distance,
        target_dist_increase=target_dist_increase,
        target_reached_delta=target_reached_delta,
        collision_delta=collision_delta,
        world_size=world_size,
        obs_min_dist=obs_min_dist,
        obs_num=obs_num,
        costmap_resolution=costmap_resolution,
        obstacle_size=obstacle_size,
        obs_distribution_mode=obs_distribution_mode,
        goal_reward=goal_reward,
        collision_penalty_base=base_collision_penalty,
        angle_penalty_base=angle_base_penalty,
        linear_penalty_base=base_linear_penalty,
        yawrate_penalty_base=yawrate_penalty_base,
        enable_obs_penalty=enable_obs_penalty,
        enable_yawrate_penalty=enable_yawrate_penalty,
        enable_angle_penalty=enable_angle_penalty,
        enable_linear_penalty=enable_linear_penalty,
        enable_target_distance_penalty=enable_target_distance_penalty,
        enable_linear_acceleration_oscillation_penalty=enable_linear_acceleration_oscillation_penalty,
        enable_yawrate_oscillation_penalty=enable_yawrate_oscillation_penalty,
        reward_debug=reward_debug,
        obs_penalty_threshold=obs_penalty_threshold,
        min_obs_penalty_threshold=min_obs_penalty_threshold,
        obs_penalty_base=obs_penalty_base,
        obs_penalty_power=obs_penalty_power,
        obs_penalty_high_weight=obs_penalty_high_weight,
        obs_penalty_low_weight=obs_penalty_low_weight,
        obs_penalty_middle_ratio=obs_penalty_middle_ratio,
        target_distance_penalty_base=target_distance_penalty_base,
        linear_acceleration_oscillation_penalty_base=linear_acceleration_oscillation_penalty_base,
        yawrate_oscillation_penalty_base=yawrate_oscillation_penalty_base,
        region_select_bias=region_select_bias,
        sim_time=sim_time,
        step_sleep_time=step_sleep_time,
        eval_sleep_time=eval_sleep_time,
        reset_step_count=reset_step_count,
        goals_per_map=goals_per_map,
    )  # instantiate ROS environment
    print("ROS Environment Initialized")

    # 只有在预训练开启时，才加载预存经验并进行预训练
    if pretrain:
        pretraining = Pretraining(
            file_names=[pretrain_data_path],
            model=model,
            replay_buffer=ReplayBuffer(buffer_size=buffer_size, random_seed=42),
            reward_function=ros.get_reward,
        )  # instantiate pre-trainind
        print("Replay Buffer Loading")
        replay_buffer = (
            pretraining.load_buffer()
        )  # fill buffer with experiences from the data.yml file
        print("Replay Buffer Loaded")
        if pretrain:
            pretraining.train(
                pretraining_iterations=pretraining_iterations,
                replay_buffer=replay_buffer,
                iterations=training_iterations,
                batch_size=batch_size,
            )  # run pre-training
        print("Load Saved Buffer Done")
    else:
        replay_buffer = ReplayBuffer(
            buffer_size=buffer_size, random_seed=42
        )  # if not experiences are loaded, instantiate an empty buffer
    
    # 初始化统计管理器
    statistics = EpisodeStatistics(window_size=stats_window_size)
    # 初始化损失跟踪器
    loss_tracker = TrainingLossTracker(window_size=loss_window_size)
    
    latest_scan, distance, cos, sin, collision, goal, last_action, reward = ros.step(
        lin_velocity=0.0, ang_velocity=0.0
    )  # get the initial step state
    #print("latest_scan_len:",len(latest_scan))
    print("="*20+f"Training Start"+"="*20)
    total_reward = 0.0
    episode_reward = 0.0
    episode_steps_count = 0  # 记录当前episode的步数
    
    # 初始化历史state队列（仅在启用历史state时使用）
    state_history = deque(maxlen=state_history_steps) if state_history_steps > 0 else None
    
    # 计算初始episode的max_steps
    if max_steps_ratio == 0:
        max_steps = int(max_steps_fixed)
    else:
        calculated_max_steps = int(distance * max_steps_ratio)
        max_steps = max(calculated_max_steps, max_steps_min)
    
    while True:  # train until max_training_count is reached
        state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, last_action
        )  # get state a state representation from returned data from the environment
        
        # 如果启用历史state，拼接历史state
        if state_history is not None:
            state_with_history = concatenate_state_history(state, state_history, state_history_steps, base_state_dim)
            model_action = model.get_action(state_with_history, True)  # 使用拼接后的state
        else:
            model_action = model.get_action(state, True)  # get an action from the model
        ros_action = utils.transfor_action(
            model_action,
            max_velocity=max_velocity,
            max_yawrate=max_yawrate,
            prev_linear_velocity=last_action[0],
            max_acceleration=max_acceleration,
            max_deceleration=max_deceleration,
            step_time=step_sleep_time,
        )
        # ros_action = [1.0, 0.0]
        latest_scan, distance, cos, sin, collision, goal, last_action, reward = ros.step(
            lin_velocity=ros_action[0], ang_velocity=ros_action[1]
        )  # get data from the environment
        episode_reward += reward
        episode_steps_count += 1
        #print("cos:", cos, "sin:", sin, "distance:", distance)
        next_state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, last_action
        )  # get a next state representation
        
        # 如果启用历史state，拼接历史state（用于next_state）并更新历史队列
        if state_history is not None:
            # 先将当前state添加到历史队列（用于下次循环）
            state_history.append(state)
            # 拼接next_state的历史（此时历史队列已包含当前state）
            next_state_with_history = concatenate_state_history(next_state, state_history, state_history_steps, base_state_dim)
            # 经验池存储拼接后的state
            replay_buffer.add(
                state_with_history, model_action, reward, terminal, next_state_with_history
            )  # add experience to the replay buffer
        else:
            # 经验池需要存储模型原始输出（未限幅）的动作，以便训练时保持策略分布一致
            replay_buffer.add(
                state, model_action, reward, terminal, next_state
            )  # add experience to the replay buffer

        if (
            terminal or steps == max_steps
        ):  # reset environment of terminal stat ereached, or max_steps were taken
            # episode结束，发送3条[0,0]速度指令停止机器人（不暂停物理模拟）
            for _ in range(ros.reset_step_count):
                ros.cmd_vel_publisher.publish_cmd_vel(linear_velocity=0.0, angular_velocity=0.0)
                time.sleep(ros.step_sleep_time)
            
            # 过滤步数小于等于1的不合理episode
            if steps <= 1:
                # 打印过滤信息
                current_time = datetime.now()
                print(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} [已过滤] Steps: {steps} (步数过少，不计入统计)")
                # 重置环境并继续，不增加episode编号
                latest_scan, distance, cos, sin, collision, goal, last_action, reward = ros.reset()
                episode_reward = 0.0
                episode_steps_count = 0
                steps = 0
                # 重置历史state队列
                if state_history is not None:
                    state_history.clear()
                # 重新计算max_steps
                if max_steps_ratio == 0:
                    max_steps = int(max_steps_fixed)
                else:
                    calculated_max_steps = int(distance * max_steps_ratio)
                    max_steps = max(calculated_max_steps, max_steps_min)
            else:
                # 正常的episode处理
                total_reward += episode_reward
                # target_distance
                td = np.linalg.norm([ros.target[0] - ros.episode_start_position[0], ros.target[1] - ros.episode_start_position[1]])
                
                # 判断episode结束原因
                timeout = not goal and not collision
                if goal:
                    episode_ending = "Goal"
                elif collision:
                    episode_ending = "Collision"
                else:
                    episode_ending= "Timeout"
                
                current_time = datetime.now()
                # 从环境中读取本 episode 各奖励项累计值
                goal_sum = getattr(ros, "episode_goal_reward", 0.0)
                collision_sum = getattr(ros, "episode_collision_penalty", 0.0)
                obs_sum = getattr(ros, "episode_obs_penalty", 0.0)
                yaw_sum = getattr(ros, "episode_yawrate_penalty", 0.0)
                angle_sum = getattr(ros, "episode_angle_penalty", 0.0)
                linear_sum = getattr(ros, "episode_linear_penalty", 0.0)

                # 读取所有奖惩开关状态（确保所有开启的项都被打印）
                enable_obs = getattr(ros, "enable_obs_penalty", False)
                enable_yawrate = getattr(ros, "enable_yawrate_penalty", False)
                enable_angle = getattr(ros, "enable_angle_penalty", False)
                enable_linear = getattr(ros, "enable_linear_penalty", False)
                enable_target_distance = getattr(ros, "enable_target_distance_penalty", False)
                enable_linear_acc_osc = getattr(ros, "enable_linear_acceleration_oscillation_penalty", False)
                enable_yawrate_osc = getattr(ros, "enable_yawrate_oscillation_penalty", False)

                # 构建只包含启用奖励项的明细
                detail_parts = []
                # 目标与碰撞始终打印（核心结束条件）
                detail_parts.append(f"goal={goal_sum:.2f}")
                detail_parts.append(f"collision={collision_sum:.2f}")
                # 其余奖励项按开关控制（所有开启的项都要打印，即使值为0也要显示）
                if enable_obs:
                    detail_parts.append(f"obs={obs_sum:.2f}")
                if enable_yawrate:
                    detail_parts.append(f"yawrate={yaw_sum:.2f}")
                if enable_angle:
                    detail_parts.append(f"angle={angle_sum:.2f}")
                if enable_linear:
                    detail_parts.append(f"linear={linear_sum:.2f}")
                if enable_target_distance:
                    target_distance_sum = getattr(ros, "episode_target_distance_penalty", 0.0)
                    detail_parts.append(f"target_distance={target_distance_sum:.2f}")
                if enable_linear_acc_osc:
                    linear_acc_osc_sum = getattr(ros, "episode_linear_acceleration_oscillation_penalty", 0.0)
                    detail_parts.append(f"linear_acc_osc={linear_acc_osc_sum:.2f}")
                if enable_yawrate_osc:
                    yawrate_osc_sum = getattr(ros, "episode_yawrate_oscillation_penalty", 0.0)
                    detail_parts.append(f"yawrate_osc={yawrate_osc_sum:.2f}")

                # 将结束状态放在最前面，然后是总reward，最后是其他奖励分项
                detail_parts_with_total = [f"end={episode_ending}", f"total_reward={episode_reward:.2f}"] + detail_parts
                detail_str = ", ".join(detail_parts_with_total)

                # 打印包含奖励明细的日志（只显示开启的奖励项）
                # ros.target_dist 是配置的目标距离上限，td 是实际生成的终点距离
                print(
                    f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"Episode: {episode} "
                    f"Target Distance: {ros.target_dist:.2f} (actual: {td:.2f}) Steps: {steps}\n"
                    f"  Reward Detail: {detail_str}"
                )
                
                # 添加到统计
                statistics.add_episode_result(goal, collision, timeout, episode_reward)
                
                # 记录添加到缓冲区的样本数
                loss_tracker.add_samples_to_buffer(episode_steps_count)
                
                episode_reward = 0.0
                episode_steps_count = 0
                episode += 1
                
                # 每report_every个episode打印统计报告
                if episode % report_every == 0:
                    stats = statistics.get_statistics()
                    print_statistics_report(stats)
                
                latest_scan, distance, cos, sin, collision, goal, a, reward = ros.reset()
                # 重置历史state队列
                if state_history is not None:
                    state_history.clear()
                # 重新计算max_steps（基于新的target_distance）
                if max_steps_ratio == 0:
                    max_steps = int(max_steps_fixed)
                else:
                    calculated_max_steps = int(distance * max_steps_ratio)
                    max_steps = max(calculated_max_steps, max_steps_min)
                
                if episode % train_every_n == 0:
                    # 物理模拟已在episode结束时暂停，这里不需要再次暂停
                    # 记录训练开始时间
                    train_start_time = time.time()
                    
                    # 训练模型并获取损失值
                    avg_critic_loss, critic_losses, avg_actor_loss, actor_losses = model.train(
                        replay_buffer=replay_buffer,
                        iterations=training_iterations,
                        batch_size=batch_size,
                    )  # train the model and update its parameters
                    critic_loss = avg_critic_loss  # 保持向后兼容
                    
                    # 训练完成后，物理模拟会在下一次step时自动恢复（unpause）
                    
                    # 计算训练耗时
                    train_time = time.time() - train_start_time
                    
                    # 记录损失和抽样数
                    loss_tracker.add_loss(critic_loss, batch_size, training_iterations)
                    avg_loss = loss_tracker.get_average_loss()
                    loss_count = loss_tracker.get_loss_count()
                    
                    # 获取抽样统计
                    buffer_size = replay_buffer.size()
                    avg_sample_usage = loss_tracker.get_avg_sample_usage()
                    
                    # 打印训练信息
                    print(f"训练 #{loss_tracker.total_trainings} - Critic Loss: {critic_loss:.4f} | 平均损失(最近{loss_count}次): {avg_loss:.4f} | 耗时: {train_time:.2f}s")
                    print(f"  总抽样数: {loss_tracker.total_samples_drawn} | 总添加样本数: {loss_tracker.total_samples_added} | 缓冲区大小: {buffer_size} | 平均抽样次数: {avg_sample_usage:.2f}")
                    
                    # 检查是否达到最大训练次数
                    if loss_tracker.total_trainings >= max_training_count:
                        print(f"达到最大训练次数 {max_training_count}，训练完成！")
                        break
                steps = 0
        else:
            steps += 1
    
    # ==================== 训练完成 ====================
    print(f"\n{'='*60}")
    print(f"训练完成！")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
