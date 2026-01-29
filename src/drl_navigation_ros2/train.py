


from pathlib import Path
import yaml
import os
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import math
from SAC.SAC import SAC
from ros_python import ROS_env
from replay_buffer import ReplayBuffer, StratifiedReplayBuffer, NumpyReplayBuffer, NumpyStratifiedReplayBuffer
import torch
import numpy as np
import utils
from pretrain_utils import Pretraining
from datetime import datetime
from collections import deque
import time

from logging_utils import CollectLogger, EnvLogger, RewardLogger, NodesLogger

def concatenate_state_history(current_state, state_history, state_history_steps, base_state_dim):
    """将当前state与历史state拼接
    
    注意：此函数已被新的实现方式替代（与 multi_env_train.py 对齐）。
    新的实现方式直接遍历 state_history 中的所有状态进行拼接，不再使用此函数。
    保留此函数以防将来需要。
    
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


def config_to_image(config_data, output_path, title="Training Configuration"):
    """
    将配置数据转换为图片（使用matplotlib）

    Args:
        config_data: 配置字典
        output_path: 输出图片路径
        title: 图片标题
    """

    # 将配置格式化为可读文本
    def format_config_text(config, indent=0):
        """递归格式化配置为文本"""
        lines = []
        prefix = "  " * indent

        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.extend(format_config_text(value, indent + 1))
                else:
                    lines.append(f"{prefix}{key}: {value}")
        elif isinstance(config, list):
            for i, item in enumerate(config):
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}[{i}]:")
                    lines.extend(format_config_text(item, indent + 1))
                else:
                    lines.append(f"{prefix}[{i}]: {item}")
        else:
            lines.append(f"{prefix}{config}")

        return lines

    # 生成配置文本
    timestamp = os.environ.get('TRAINING_TIMESTAMP', datetime.now().strftime("%Y%m%d_%H%M%S"))
    header = [
        "=" * 60,
        f"  {title}",
        f"  Generated: {timestamp}",
        "=" * 60,
        ""
    ]

    config_lines = format_config_text(config_data)
    all_lines = header + config_lines

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, len(all_lines) * 0.15))  # 动态高度
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(all_lines))
    ax.axis('off')

    # 设置字体
    font_props = {'family': 'monospace', 'size': 10}

    # 绘制文本
    for i, line in enumerate(all_lines):
        ax.text(0.02, len(all_lines) - i - 0.5, line, fontdict=font_props,
                verticalalignment='center', color='black')

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')

    # 关闭图形以释放内存
    plt.close(fig)

    # 设置为只读权限
    os.chmod(output_path, 0o444)

    print(f"配置图片已保存到: {output_path}")
    print(f"包含 {len(config_lines)} 个配置项")


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
    """训练次数和样本抽样统计跟踪器"""
    
    def __init__(self):
        self.total_trainings = 0
        self.total_samples_drawn = 0  # 总抽样样本数
        self.total_samples_added = 0  # 总添加样本数
    
    def add_loss(self, critic_loss, batch_size, training_iterations):
        """添加一次训练的损失和抽样数"""
        self.total_trainings += 1
        # 计算本次训练抽样的总数
        self.total_samples_drawn += batch_size * training_iterations
    
    def add_samples_to_buffer(self, num_samples):
        """记录添加到缓冲区的样本数"""
        self.total_samples_added += num_samples
    
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
    clog = None
    env_logger = None
    reward_logger = None
    nodes_logger = None
    # ==================== 加载配置文件 ====================
    # 加载配置文件（不再依赖 TRAIN_CONFIG_PATH 环境变量，可通过函数参数传入）
    config = load_config(args.config_path) if args and hasattr(args, 'config_path') else load_config()

    # 保存实际使用的配置文件路径用于打印
    actual_config_path = Path(args.config_path) if args and hasattr(args, 'config_path') else (Path(__file__).parent.parent.parent / "config" / "train.yaml")
    
    # ==================== 单环境日志系统（对齐 multi_env_train.py） ====================
    # 单环境默认开启日志：基于 TRAINING_TIMESTAMP + single_env_log_dir（或启动脚本传入的 LOG_DIR）自动生成路径。
    # 若用户显式在 config 中提供 collect/env/reward/nodes_log_path，则优先使用显式配置。
    training_timestamp = os.environ.get("TRAINING_TIMESTAMP", datetime.now().strftime("%Y%m%d_%H%M%S"))
    env_log_dir = os.environ.get("LOG_DIR", "").strip()
    if env_log_dir:
        log_dir = Path(env_log_dir)
    else:
        base_log_dir = Path(config.get('single_env_log_dir', "log/single_env_training"))
        log_dir = base_log_dir / f"train_{training_timestamp}"

    # 自动生成日志文件路径（默认启用）
    default_collect_log_path = str(log_dir / f"collect_log_{training_timestamp}.log")
    default_env_log_path = str(log_dir / f"env_log_{training_timestamp}.log")
    default_reward_log_path = str(log_dir / f"reward_log_{training_timestamp}.log")
    default_nodes_log_path = str(log_dir / f"nodes_log_{training_timestamp}.log")

    collect_log_path = (config.get('collect_log_path') or default_collect_log_path).strip()
    env_log_path = (config.get('env_log_path') or default_env_log_path).strip()
    reward_log_path = (config.get('reward_log_path') or default_reward_log_path).strip()
    nodes_log_path = (config.get('nodes_log_path') or default_nodes_log_path).strip()

    clog = CollectLogger(collect_log_path)
    env_logger = EnvLogger(env_log_path)
    reward_logger = RewardLogger(reward_log_path)
    nodes_logger = NodesLogger(nodes_log_path)

    # ==================== 设置ROS_DOMAIN_ID ====================
    # 训练脚本以配置文件为准：单环境使用 single_env_ros_domain_id
    # 为向后兼容，若不存在则尝试旧键 ros_domain_id，最后默认 0
    ros_domain_id = int(config.get('single_env_ros_domain_id', config.get('ros_domain_id', 0)))
    # 将值写入环境变量，供后续 ROS 节点使用（rclpy/子进程读取）
    os.environ['ROS_DOMAIN_ID'] = str(ros_domain_id)
    print(f"ROS_DOMAIN_ID: {ros_domain_id}")
    print(f"使用配置文件: {actual_config_path}")
    
    # collect 日志：整体训练启动（对齐多环境风格）
    if clog:
        clog.log(
            0,
            "train_start",
            {"ros_domain_id": ros_domain_id, "log_dir": str(log_dir), "config_path": str(actual_config_path)},
            "started",
        )
        # 单环境也补充一条“collect_start”，含 env_id 和 ros_domain_id，方便统一分析
        clog.log(
            0,
            "collect_start",
            {"env_id": 0, "ros_domain_id": ros_domain_id},
            "started",
        )

    # ==================== 设置TURTLEBOT3_MODEL ====================
    # 从配置文件读取TURTLEBOT3_MODEL，如果环境变量已设置则优先使用环境变量
    turtlebot3_model = config.get('turtlebot3_model', 'waffle')
    os.environ['TURTLEBOT3_MODEL'] = str(turtlebot3_model)  # 同步到环境变量供 ROS 使用
    print(f"TURTLEBOT3_MODEL: {turtlebot3_model}")
    
    # ==================== 机器人和环境参数 ====================
    max_velocity = config.get('max_velocity', 1.0)
    max_yawrate = config.get('max_yawrate', 20.0)
    max_acceleration = config.get('max_acceleration', 1.0)
    max_deceleration = config.get('max_deceleration', 1.0)
    neglect_angle = config.get('neglect_angle', 0)
    scan_range = config.get('scan_range', 5)
    localization_noise_stddev = config.get('localization_noise_stddev', 0.0)
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
    # 单步 state 维度（base_state_dim）不再在 train.yaml 中手动配置，改为由 bin_num + 非激光特征数动态计算
    state_history_steps = config.get('state_history_steps', 0)
    bin_num = config.get('bin_num', 72)
    # base_state_dim/state_dim 已在上方根据 bin_num 动态计算，这里不再重复读取
    non_lidar_dim = 7  # distance,cos,sin(3) + last_action(2) + current_v,current_w(2)
    base_state_dim = int(bin_num) + non_lidar_dim

    # 动态计算state_dim
    if state_history_steps > 0:
        state_dim = base_state_dim * (1 + state_history_steps)
        print(f"启用历史state模式: base_state_dim={base_state_dim}, state_history_steps={state_history_steps}, 最终state_dim={state_dim}")
    else:
        state_dim = base_state_dim
        print(f"未启用历史state模式: state_dim={state_dim}")
    
    hidden_layers = config.get('hidden_layers', [1024, 512])
    
    # ==================== 训练参数 ====================
    episode = 0  # 当前回合数
    stats_window_size = config.get('stats_window_size', 20)
    # 新增：用于保存最佳模型的统计窗口长度（默认使用 stats_window_size）
    best_model_window_size = config.get('best_model_window_size', stats_window_size)
    # 缓冲区达到该大小后开始在线更新
    buffer_train_start_size = config.get('buffer_train_start_size', 1000)
    n_step_per_train = config.get('n_step_per_train', 1)  # 每收集n步执行一次梯度更新
    # 训练停止条件：episode 数量
    max_episodes = config.get('max_episodes', 1000)
    # 最终评估episode数量
    eval_episodes = config.get('eval_episodes', 50)
    # 取消 train_every_n 逻辑（单环境在线更新每步训练）
    train_every_n = None
    training_iterations = config.get('training_iterations', 500)
    batch_size = config.get('batch_size', 40)
    buffer_size = config.get('buffer_size', 50000)
    max_steps_fixed = config.get('max_steps', 300)
    max_steps_ratio = config.get('max_steps_ratio', 0)
    max_steps_min = config.get('max_steps_min', 50)
    # 不在训练脚本里手动统计步数，统一使用环境内部的 ros.step_count
    pretrain = config.get('pretrain', False)
    pretraining_iterations = config.get('pretraining_iterations', 50)
    load_model = config.get('load_model', True)
    # 单环境训练优先使用 save_every_single_env，如果未设置则使用 save_every
    save_every = config.get('save_every_single_env', config.get('save_every', 10))
    enable_action_noise = config.get('enable_action_noise', False)  # 是否开启动作噪声
    
    # 注意：无论是否加载模型，都只按照train.yaml中设置的参数来创建模型（和multi_env_train.py中的逻辑一样）
    discount_factor = config.get('discount_factor', 0.99)  # 折扣因子（gamma），从配置文件 train.yaml 读取，统一用于计算总回报和所有 Reward Detail 分项的折扣回报
    actor_update_frequency = config.get('actor_update_frequency', 1)
    critic_target_update_frequency = config.get('critic_target_update_frequency', 2)
    # 学习率相关参数（从配置文件读取，若未配置则使用默认值）
    actor_lr = config.get('actor_lr', 1e-4)
    critic_lr = config.get('critic_lr', 1e-4)
    alpha_lr = config.get('alpha_lr', 1e-4)
    
    # 设备选择
    # 注意：如果通过环境变量 CUDA_VISIBLE_DEVICES 设置了 GPU，PyTorch 视角下 GPU 索引从 0 开始
    gpu_id = config.get('gpu_id', None)
    # 设备选择：优先使用 CUDA，否则 CPU
    # 单环境训练不再依赖环境变量选择设备；若需限定 GPU，请在启动时自行设置 CUDA_VISIBLE_DEVICES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU CUDA Ready (使用设备: {device})")
    else:
        print("CPU Ready")
    
    # ==================== 路径参数 ====================
    # 模型加载路径
    load_path = Path(config.get('load_path', "/home/zc/DRL-Robot-Navigation-ROS2/src/drl_navigation_ros2/models/SAC"))

    # 使用上方日志系统生成的 log_dir 作为单环境训练的产物根目录
    # 说明：log_dir 在“单环境日志系统”段落里已按 LOG_DIR 或 single_env_log_dir + TRAINING_TIMESTAMP 计算。

    # 模型保存目录：实时模型保存到 model 子目录，最佳模型保存到 best_model 子目录
    model_save_dir = log_dir / "model"
    best_model_save_dir = log_dir / "best_model"
    pretrain_data_path = config.get('pretrain_data_path', "src/drl_navigation_ros2/assets/data.yml")

    # 确保保存目录存在（不再覆盖旧目录）
    model_save_dir.mkdir(parents=True, exist_ok=True)
    best_model_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"模型保存目录已准备: {model_save_dir}")
    print(f"最好模型保存目录已准备: {best_model_save_dir}")
    # 保存本次训练使用的配置快照到日志目录（图片格式，只读）
    if actual_config_path and Path(actual_config_path).exists():
        try:
            config_filename = f"config_{training_timestamp}.png"
            target_config_path = log_dir / config_filename
            # 将YAML配置转换为图片格式保存
            with open(actual_config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
            config_to_image(config_data, target_config_path, "DRL Robot Dog Navigation - Training Config")
        except Exception as e:
            print(f"警告: 保存配置快照到日志目录失败: {e}")
    
    # ==================== 奖励函数参数 ====================
    goal_reward = config.get('goal_reward', 1000.0)
    base_collision_penalty = config.get('collision_penalty_base', config.get('base_collision_penalty', -1000.0))
    angle_base_penalty = config.get('angle_penalty_base', config.get('angle_base_penalty', 0.0))
    base_linear_penalty = config.get('linear_penalty_base', config.get('base_linear_penalty', -1.0))
    yawrate_penalty_base = config.get('yawrate_penalty_base', 0.0)
    reward_scale = config.get('reward_scale', 1.0)  # 奖励缩放因子，用于控制每步整体奖励大小
    enable_obs_penalty = config.get('enable_obs_penalty', True)
    enable_yawrate_penalty = config.get('enable_yawrate_penalty', True)
    enable_angle_penalty = config.get('enable_angle_penalty', True)
    enable_linear_penalty = config.get('enable_linear_penalty', True)
    enable_step_penalty = config.get('enable_step_penalty', False)
    enable_target_distance_penalty = config.get('enable_target_distance_penalty', False)
    enable_progress_reward = config.get('enable_progress_reward', False)
    enable_linear_acceleration_oscillation_penalty = config.get('enable_linear_acceleration_oscillation_penalty', False)
    enable_yawrate_oscillation_penalty = config.get('enable_yawrate_oscillation_penalty', False)
    progress_reward_base = config.get('progress_reward_base', 1.0)
    
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
    # ==================== 时间步惩罚参数 ====================
    step_penalty_base = config.get('step_penalty_base', 0.0)
    
    # ==================== 震荡惩罚参数 ====================
    linear_acceleration_oscillation_penalty_base = config.get('linear_acceleration_oscillation_penalty_base', -1.0)
    yawrate_oscillation_penalty_base = config.get('yawrate_oscillation_penalty_base', -1.0)
    
    # ==================== 连通区域选择参数 ====================
    region_select_bias = config.get('region_select_bias', 1.0)
    
    # ==================== 时间控制参数 ====================
    sim_time = config.get('sim_time', 0.1)
    step_sleep_time = config.get('step_sleep_time', 0.1)
    reset_step_count = config.get('reset_step_count', 3)
    
    # ==================== 动作噪声参数 ====================
    action_noise_std = config.get('action_noise_std', 0.2)
    
    actor_grad_clip_value = config.get('actor_grad_clip_value', 0.0)
    critic_grad_clip_value = config.get('critic_grad_clip_value', 0.0)
    model = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        discount=discount_factor,  # 传递折扣因子
        actor_lr=actor_lr,  # 传递Actor学习率
        critic_lr=critic_lr,  # 传递Critic学习率
        alpha_lr=alpha_lr,  # 传递温度参数学习率
        save_every=save_every,
        load_model=load_model,
        save_directory=model_save_dir,
        load_directory=load_path,
        action_noise_std=action_noise_std,
        hidden_layers=hidden_layers,
        actor_update_frequency=actor_update_frequency,
        critic_target_update_frequency=critic_target_update_frequency,
        base_state_dim=base_state_dim,  # 传递base_state_dim给SAC模型
        bin_num=bin_num,  # 激光scan分桶数量
        actor_grad_clip_value=actor_grad_clip_value,  # 传递Actor梯度裁剪值
        critic_grad_clip_value=critic_grad_clip_value,  # 传递Critic梯度裁剪值
    )  # instantiate a model
    print("Model Loaded")
    
    # ==================== 传感器频率限制 & 日志开关（与 multi_env_train 对齐） ====================
    # 频率限制配置（Hz），0 或负数表示不限制
    sensor_freq_cfg = config.get('sensor_freq_limit', {}) or {}
    scan_max_freq = sensor_freq_cfg.get('scan_max_freq', 0.0)
    odom_max_freq = sensor_freq_cfg.get('odom_max_freq', 0.0)
    imu_max_freq = sensor_freq_cfg.get('imu_max_freq', 0.0)

    # 传感器日志开关配置（来自 train.yaml 的 sensor_log_enable 段）
    sensor_log_cfg = config.get('sensor_log_enable', {}) or {}

    def _to_bool(val, default=False):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ('true', '1', 'yes', 'on')
        return bool(val) if val is not None else default

    scan_enable_log = _to_bool(sensor_log_cfg.get('scan_enable_log', False)) if isinstance(sensor_log_cfg, dict) else False
    odom_enable_log = _to_bool(sensor_log_cfg.get('odom_enable_log', False)) if isinstance(sensor_log_cfg, dict) else False
    imu_enable_log = _to_bool(sensor_log_cfg.get('imu_enable_log', False)) if isinstance(sensor_log_cfg, dict) else False

    print(f"单环境传感器日志开关: scan={scan_enable_log}, odom={odom_enable_log}, imu={imu_enable_log}")

    ros = ROS_env(
        env_id=0,
        env_logger=env_logger,
        reward_logger=reward_logger,
        nodes_logger=nodes_logger,
        max_velocity=max_velocity,
        neglect_angle=neglect_angle,
        scan_range=scan_range,
        localization_noise_stddev=localization_noise_stddev,
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
        enable_step_penalty=enable_step_penalty,
        enable_target_distance_penalty=enable_target_distance_penalty,
        enable_progress_reward=enable_progress_reward,
        enable_linear_acceleration_oscillation_penalty=enable_linear_acceleration_oscillation_penalty,
        enable_yawrate_oscillation_penalty=enable_yawrate_oscillation_penalty,
        obs_penalty_threshold=obs_penalty_threshold,
        min_obs_penalty_threshold=min_obs_penalty_threshold,
        obs_penalty_base=obs_penalty_base,
        obs_penalty_power=obs_penalty_power,
        obs_penalty_high_weight=obs_penalty_high_weight,
        obs_penalty_low_weight=obs_penalty_low_weight,
        obs_penalty_middle_ratio=obs_penalty_middle_ratio,
        # 终点距离惩罚参数
        target_distance_penalty_base=target_distance_penalty_base,
        step_penalty_base=step_penalty_base,
        linear_acceleration_oscillation_penalty_base=linear_acceleration_oscillation_penalty_base,
        yawrate_oscillation_penalty_base=yawrate_oscillation_penalty_base,
        progress_reward_base=progress_reward_base,
        region_select_bias=region_select_bias,
        sim_time=sim_time,
        step_sleep_time=step_sleep_time,
        reset_step_count=reset_step_count,
        goals_per_map=goals_per_map,
        reward_scale=reward_scale,  # 奖励缩放因子
        # 传感器频率限制参数
        scan_max_freq=scan_max_freq,
        odom_max_freq=odom_max_freq,
        imu_max_freq=imu_max_freq,
        # 传感器日志开关（从 train.yaml 的 sensor_log_enable 配置读取）
        scan_enable_log=scan_enable_log,
        odom_enable_log=odom_enable_log,
        imu_enable_log=imu_enable_log,
    )  # instantiate ROS environment

    # 只有在预训练开启时，才加载预存经验并进行预训练
    if pretrain:
        pretraining = Pretraining(
            file_names=[pretrain_data_path],
            model=model,
            replay_buffer=ReplayBuffer(
                buffer_size=buffer_size, 
                random_seed=42,
                recent_buffer_ratio=config.get('recent_buffer_ratio', 0.1),
                recent_batch_ratio=config.get('recent_batch_ratio', 0.3)
            ),
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
        replay_buffer = NumpyReplayBuffer(
            buffer_size=buffer_size,
            dtype=np.float32,
            recent_buffer_ratio=config.get('recent_buffer_ratio', 0.1),
            recent_batch_ratio=config.get('recent_batch_ratio', 0.3),
        )  # 单环境训练统一使用单一经验池
    
    # 初始化统计管理器
    statistics = EpisodeStatistics(window_size=stats_window_size)
    # 初始化训练统计跟踪器
    loss_tracker = TrainingLossTracker()
    
    print("="*20+f"Training Start"+"="*20)
    
    # 直接读取 ros_python.py 中 get_reward() 记录的“本step分量”，避免用episode累计值做差分
    
    # 初始化历史state队列
    # 历史state设置（长度为 state_history_steps + 1，用于存储 [s_{t-k}, ..., s_t]）
    # 即使 state_history_steps 为 0，也保持长度为 1，此时等价于只使用当前 state
    state_history = deque(maxlen=state_history_steps + 1)
    
    # 初始化steps_since_last_train（用于在线训练计数）
    steps_since_last_train = 0
    
    # 外层循环：episode级循环
    while True:  # train until max_episodes is reached
        # 检查是否达到最大episode数
        if max_episodes > 0 and episode >= max_episodes:
            print(f"达到最大episode数 {max_episodes}，训练完成！")
            break
        
        # 初始化episode相关变量
        episode_reward = 0.0
        gamma_power = 1.0  # 重置折扣因子幂次
        # 当前episode步数：使用 ros.step_count（reset 内会清零）
        
        # 重置折扣后的奖励分项统计
        discounted_goal_sum = 0.0
        discounted_collision_sum = 0.0
        discounted_obs_sum = 0.0
        discounted_yawrate_sum = 0.0
        discounted_angle_sum = 0.0
        discounted_linear_sum = 0.0
        discounted_step_sum = 0.0
        discounted_progress_sum = 0.0
        discounted_target_distance_sum = 0.0
        discounted_linear_acc_osc_sum = 0.0
        discounted_yawrate_osc_sum = 0.0
        
        # 重置环境并获取初始状态，循环调用直到成功
        reset_success = False
        while not reset_success:
            reset_success, latest_scan, distance, distance_raw, cos, sin, collision, goal, last_action, reward, current_v, current_w = ros.reset(use_reset_simulation=True)
            if not reset_success:
                # 只写入 env_log，不再输出到综合训练日志/stdout；标记为 [ERROR]
                if env_logger is not None:
                    env_logger.log(0, "[ERROR] reset()失败，重试中...")
        
        # 计算当前episode的max_steps
        if max_steps_ratio == 0:
            max_steps = int(max_steps_fixed)
        else:
            calculated_max_steps = int(distance * max_steps_ratio)
            max_steps = max(calculated_max_steps, max_steps_min)
        
        # 重置历史state队列并用初始state填充
        reset_state, _ = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, last_action, current_v, current_w
        )
        # 清空并用 s0 填满历史队列（state_history_steps+1 个 s0）
        state_history.clear()
        for _ in range(state_history_steps + 1):
            state_history.append(list(reset_state))  # 避免同一对象重复引用
        # 预先展开得到当前输入 x0，后续循环中复用并逐步更新
        current_state_with_history = []
        for hist_state in state_history:
            current_state_with_history.extend(hist_state)
        
        # 内层循环：step级循环
        while True:  # 循环直到episode结束（terminal或达到max_steps）
            # 构造当前输入 x_t（统一从 current_state_with_history 中读取；
            #    当 state_history_steps 为 0 时，current_state_with_history 仅包含当前 state）
            model_action = model.get_action(current_state_with_history, enable_action_noise)  # 使用拼接后的state，根据配置决定是否添加噪声
            # model_action=[1.0,1.0]
            max_velocity = float(config.get('max_velocity', 1.0))
            lin_velocity = (float(model_action[0]) + 1.0) * (max_velocity / 2.0)
            lin_velocity = min(max(lin_velocity, 0.0), max_velocity)

            max_yawrate = float(config["max_yawrate"])
            ang_velocity = float(model_action[1]) * (max_yawrate / 180.0) * math.pi

            ros_action = [lin_velocity, ang_velocity]
            latest_scan, distance, distance_raw, cos, sin, collision, goal, reward, current_v, current_w = ros.step(
                lin_velocity=ros_action[0], ang_velocity=ros_action[1]
            )  # get data from the environment
            # 调用方维护 last_action（本次实际执行动作），用于下一时刻观测拼接/动作约束
            last_action = [float(ros_action[0]), float(ros_action[1])]
            # 计算折扣回报：G_0 = r_0 + γ*r_1 + γ²*r_2 + ...（使用统一的 discount_factor）
            episode_reward += gamma_power * reward
            
            # 计算各奖励分项的折扣累加值（用于 Reward Detail 显示，使用与总回报相同的 discount_factor）
            parts = getattr(ros, "last_step_reward_parts", None) or {}
            parts_scaled = parts.get("scaled", {}) if isinstance(parts, dict) else {}
            discounted_goal_sum += gamma_power * float(parts_scaled.get("goal", 0.0) or 0.0)
            discounted_collision_sum += gamma_power * float(parts_scaled.get("collision", 0.0) or 0.0)
            discounted_obs_sum += gamma_power * float(parts_scaled.get("obs", 0.0) or 0.0)
            discounted_yawrate_sum += gamma_power * float(parts_scaled.get("yawrate", 0.0) or 0.0)
            discounted_angle_sum += gamma_power * float(parts_scaled.get("angle", 0.0) or 0.0)
            discounted_linear_sum += gamma_power * float(parts_scaled.get("linear", 0.0) or 0.0)
            discounted_step_sum += gamma_power * float(parts_scaled.get("step_penalty", 0.0) or 0.0)
            discounted_progress_sum += gamma_power * float(parts_scaled.get("progress", 0.0) or 0.0)
            discounted_target_distance_sum += gamma_power * float(parts_scaled.get("target_distance", 0.0) or 0.0)
            discounted_linear_acc_osc_sum += gamma_power * float(parts_scaled.get("linear_acc_osc", 0.0) or 0.0)
            discounted_yawrate_osc_sum += gamma_power * float(parts_scaled.get("yawrate_osc", 0.0) or 0.0)
            
            gamma_power *= discount_factor  # 更新折扣因子幂次
            #print("cos:", cos, "sin:", sin, "distance:", distance)
            next_state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, last_action, current_v, current_w
            )  # get a next state representation
            
            # 单环境训练：强制使用“单一缓冲区”，逐 step 写入（不按 outcome 分层存储）
            # 更新历史并构造下一时刻输入 x_{t+1}
            # 将 next_state 压入历史，形成 [s_{t-k+1}, ..., s_t, s_{t+1}]
            state_history.append(list(next_state))  # 避免同一对象重复引用
            next_state_with_history = []
            for hist_state in state_history:
                next_state_with_history.extend(hist_state)
            # 存储 (x_t, action, reward, done, x_{t+1})
            replay_buffer.add(
                current_state_with_history, model_action, reward, terminal, next_state_with_history
            )
            # 下一个循环直接使用 x_{t+1}
            current_state_with_history = next_state_with_history
            # 记录添加到缓冲区的样本数（每次添加后立即记录）
            loss_tracker.add_samples_to_buffer(1)

            # 在线训练：缓冲区达到阈值后，每收集 n_step_per_train 步进行一次梯度更新
            steps_since_last_train += 1
            if n_step_per_train > 0 and replay_buffer.size() >= buffer_train_start_size and steps_since_last_train >= n_step_per_train:
                # 与多环境训练保持一致：SAC.train 现在返回8个值
                (
                    critic_loss,      # 平均 critic loss（标量）
                    _critic_losses,   # 本次训练所有 critic loss 列表（此处未使用）
                    actor_loss,       # 平均 actor loss（标量或 None）
                    _actor_losses,    # 本次训练所有 actor loss 列表（此处未使用）
                    critic_grad,      # critic 梯度范数统计 dict 或 None
                    actor_grad,       # actor 梯度范数统计 dict 或 None
                    _entropy,         # 平均熵值（未使用）
                    _alpha_grad,      # alpha 梯度范数（未使用）
                ) = model.train(
                    replay_buffer=replay_buffer,
                    iterations=1,
                    batch_size=batch_size,
                    stats=statistics.get_statistics(),
                )
                steps_since_last_train = 0
                if critic_loss is not None:
                    loss_tracker.add_loss(critic_loss, batch_size, 1)
                    # 打印每次梯度更新的训练信息（单步在线更新）
                    current_stats = statistics.get_statistics()
                    buffer_current_size = replay_buffer.size()
                    avg_sample_usage = loss_tracker.get_avg_sample_usage()
                    current_time = datetime.now()
                    # 处理可能为None的ActorLoss和梯度统计
                    actor_loss_str = f"{actor_loss:.4f}" if actor_loss is not None else "None"
                    critic_grad_before = critic_grad.get("before") if isinstance(critic_grad, dict) else None
                    critic_grad_after = critic_grad.get("after") if isinstance(critic_grad, dict) else None
                    actor_grad_before = actor_grad.get("before") if isinstance(actor_grad, dict) else None
                    actor_grad_after = actor_grad.get("after") if isinstance(actor_grad, dict) else None
                    critic_grad_str = (
                        f"before={critic_grad_before:.4f},after={critic_grad_after:.4f}"
                        if critic_grad_before is not None and critic_grad_after is not None
                        else "None"
                    )
                    actor_grad_str = (
                        f"before={actor_grad_before:.4f},after={actor_grad_after:.4f}"
                        if actor_grad_before is not None and actor_grad_after is not None
                        else "None"
                    )
                    print(
                        f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} "
                        f"[DEBUG] GradUpdate #{loss_tracker.total_trainings} "
                        f"CriticLoss={critic_loss:.4f} "
                        f"ActorLoss={actor_loss_str} "
                        f"CriticGrad({critic_grad_str}) "
                        f"ActorGrad({actor_grad_str}) "
                        f"BatchSize={batch_size} "
                        f"BufferSize={buffer_current_size} "
                        f"AvgSampleUsage={avg_sample_usage:.2f} "
                        f"Episodes={current_stats.get('total_episodes', 0)}"
                    )

            # 检查episode是否结束（terminal或达到max_steps）
            if terminal or ros.step_count >= max_steps:
                # print(f"Episode {episode} ended with {ros.step_count} steps")
                # print(f"Episode {episode} terminal: {terminal}")
                break  # 跳出内层step循环，进入episode结束处理
        # Episode结束处理 设置为0暂时停用
        if ros.step_count <= 0:
            # 打印过滤信息
            current_time = datetime.now()
            print(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} [已过滤] Steps: {ros.step_count} (步数过少，不计入统计)")
            # 不增加episode编号，直接进入下一个episode
            continue
        else:
            # 正常的episode处理；以 ros.initial_target_distance 为准
            td = getattr(ros, "initial_target_distance", None)
            if td is None and ros.episode_start_position is not None and ros.target is not None:
                td = np.linalg.norm([ros.target[0] - ros.episode_start_position[0], ros.target[1] - ros.episode_start_position[1]])
            if td is None:
                td = ros.target_dist
            
            # 判断episode结束原因
            timeout = not goal and not collision
            if goal:
                episode_ending = "Goal"
            elif collision:
                episode_ending = "Collision"
            else:
                episode_ending = "Timeout"
                
            current_time = datetime.now()
            # 使用折扣后的奖励分项值（已在每个 step 中计算）

            # 读取所有奖惩开关状态（确保所有开启的项都被打印）
            enable_obs = getattr(ros, "enable_obs_penalty", False)
            enable_yawrate = getattr(ros, "enable_yawrate_penalty", False)
            enable_angle = getattr(ros, "enable_angle_penalty", False)
            enable_linear = getattr(ros, "enable_linear_penalty", False)
            enable_step = getattr(ros, "enable_step_penalty", False)
            enable_target_distance = getattr(ros, "enable_target_distance_penalty", False)
            enable_linear_acc_osc = getattr(ros, "enable_linear_acceleration_oscillation_penalty", False)
            enable_yawrate_osc = getattr(ros, "enable_yawrate_oscillation_penalty", False)
            enable_progress = getattr(ros, "enable_progress_reward", False)

            # 构建只包含启用奖励项的明细
            detail_parts = []
            # 目标与碰撞始终打印（核心结束条件）
            detail_parts.append(f"goal={discounted_goal_sum:.2f}")
            detail_parts.append(f"collision={discounted_collision_sum:.2f}")
            # step：所有“非终止型”的开启分项汇总为 step_total
            step_total = 0.0
            if enable_obs:
                step_total += discounted_obs_sum
            if enable_yawrate:
                step_total += discounted_yawrate_sum
            if enable_angle:
                step_total += discounted_angle_sum
            if enable_linear:
                step_total += discounted_linear_sum
            if enable_step:
                step_total += discounted_step_sum
            if enable_target_distance:
                step_total += discounted_target_distance_sum
            if enable_linear_acc_osc:
                step_total += discounted_linear_acc_osc_sum
            if enable_yawrate_osc:
                step_total += discounted_yawrate_osc_sum
            if enable_progress:
                step_total += discounted_progress_sum
            detail_parts.append(f"step={step_total:.2f}")
            # 其余奖励项按开关控制（所有开启的项都要打印，即使值为0也要显示）
            if enable_obs:
                detail_parts.append(f"obs={discounted_obs_sum:.2f}")
            if enable_yawrate:
                detail_parts.append(f"yawrate={discounted_yawrate_sum:.2f}")
            if enable_angle:
                detail_parts.append(f"angle={discounted_angle_sum:.2f}")
            if enable_linear:
                detail_parts.append(f"linear={discounted_linear_sum:.2f}")
            if enable_step:
                detail_parts.append(f"step_penalty={discounted_step_sum:.2f}")
            if enable_target_distance:
                detail_parts.append(f"target_distance={discounted_target_distance_sum:.2f}")
            if enable_linear_acc_osc:
                detail_parts.append(f"linear_acc_osc={discounted_linear_acc_osc_sum:.2f}")
            if enable_yawrate_osc:
                detail_parts.append(f"yawrate_osc={discounted_yawrate_osc_sum:.2f}")
            if enable_progress:
                detail_parts.append(f"progress={discounted_progress_sum:.2f}")

            # 将结束状态放在最前面，然后是总reward，最后是其他奖励分项
            detail_parts_with_total = [f"end={episode_ending}", f"total_reward={episode_reward:.2f}"] + detail_parts
            detail_str = ", ".join(detail_parts_with_total)

            # 打印包含奖励明细的日志（只显示开启的奖励项）
            # ros.target_dist 是配置的目标距离上限，td 为 ros.initial_target_distance（首步 step 的 distance）
            print(
                f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"Episode: {episode} "
                f"Target Distance: {ros.target_dist:.2f} (actual: {td:.2f}) Steps: {ros.step_count}\n"
                f"  Reward Detail: {detail_str}"
            )
            
            # 添加到统计
            statistics.add_episode_result(goal, collision, timeout, episode_reward)
                
            # 增加episode编号（重置逻辑已在episode开始时完成）
            episode += 1
    
    # ==================== 训练完成 ====================
    if clog:
        clog.log(0, "train_end", {"episodes": episode}, "done")

    for _lg in (clog, env_logger, reward_logger, nodes_logger):
        try:
            if _lg:
                _lg.close()
        except Exception:
            pass

    print(f"\n{'='*60}")
    print(f"训练完成！")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
