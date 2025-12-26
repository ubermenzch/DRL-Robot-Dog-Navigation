#!/usr/bin/env python3
"""
多环境并行评估脚本
使用多个进程并行收集episode数据进行评估：
- X个环境并行收集数据，使用相同的加载模型
- 分段式评估：max_target_dist分别设置为3、6、9、12，每个阶段收集100个episode
- 每个阶段结束后统计并输出评估结果
"""
import argparse
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np
import torch
import sys
import os
import yaml
from datetime import datetime
from collections import deque
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from SAC.SAC import SAC
from ros_python import ROS_env
import utils


def write_episode_result(log_file, file_lock, goal, collision, timeout, reward, distance, steps):
    """将episode结果写入日志文件（互斥写入）"""
    episode_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'goal': goal,
        'collision': collision,
        'timeout': timeout,
        'reward': reward,
        'target_distance': distance,  # 目标距离
        'steps': steps
    }
    
    with file_lock:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(episode_data) + '\n')


def analyze_results_by_distance(data_log_file, distance_ranges):
    """按目标距离区间分析结果
    
    Args:
        data_log_file: 数据日志文件路径
        distance_ranges: 距离区间列表，如[(0, 3), (3, 6), (6, 9), (9, 12)]
    
    Returns:
        每个距离区间的统计结果字典
    """
    if not os.path.exists(data_log_file):
        return {}
    
    # 读取所有episode数据
    all_episodes = []
    with open(data_log_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                episode_data = json.loads(line)
                all_episodes.append(episode_data)
    
    # 按距离区间分组统计
    results = {}
    for min_dist, max_dist in distance_ranges:
        # 筛选出该区间的episodes
        range_episodes = [ep for ep in all_episodes 
                         if min_dist <= ep['target_distance'] < max_dist]
        
        if len(range_episodes) == 0:
            results[f'{min_dist}-{max_dist}m'] = None
            continue
        
        total_episodes = len(range_episodes)
        goals = sum(1 for ep in range_episodes if ep['goal'])
        collisions = sum(1 for ep in range_episodes if ep['collision'])
        timeouts = sum(1 for ep in range_episodes if ep['timeout'])
        total_reward = sum(ep['reward'] for ep in range_episodes)
        total_steps = sum(ep['steps'] for ep in range_episodes)
        
        results[f'{min_dist}-{max_dist}m'] = {
            'distance_range': f'{min_dist}-{max_dist}m',
            'total_episodes': total_episodes,
            'goal_count': goals,
            'collision_count': collisions,
            'timeout_count': timeouts,
            'goal_rate': goals / total_episodes,
            'collision_rate': collisions / total_episodes,
            'timeout_rate': timeouts / total_episodes,
            'avg_reward': total_reward / total_episodes,
            'avg_steps': total_steps / total_episodes
        }
    
    return results


def report_episode_complete(total_episodes, episode_counter, evaluation_complete, counter_lock):
    """报告一个episode完成，返回是否应继续"""
    with counter_lock:
        episode_counter.value += 1
        
        # 检查是否完成所有episode
        if episode_counter.value >= total_episodes:
            evaluation_complete.value = True
            return False
        
        return True


def is_evaluation_complete(evaluation_complete):
    """检查评估是否完成"""
    return evaluation_complete.value


def get_progress(total_episodes, episode_counter, counter_lock):
    """获取当前进度信息"""
    with counter_lock:
        current_count = episode_counter.value
        progress = current_count / total_episodes
        return f"已完成: {current_count}/{total_episodes}", progress


def collect_evaluation_episodes(env_id, model_path, config,
                                 total_episodes, episode_counter, evaluation_complete, counter_lock,
                                 data_log_file, file_lock, distance_intervals=None, episodes_per_interval=None, remaining_episodes=0):
    """单个环境的评估数据收集进程（持续运行，自动切换距离区间）
    
    Args:
        distance_intervals: 距离区间列表，如[3, 6, 9]
        episodes_per_interval: 每个区间的episode数量
        remaining_episodes: 最后一个区间额外的episode数量
    """
    try:
        print(f"环境 {env_id} 开始初始化...")
        
        # 设置正确的ROS域ID
        ros_domain_id = env_id + 1
        os.environ['ROS_DOMAIN_ID'] = str(ros_domain_id)
        print(f"环境 {env_id} 设置ROS_DOMAIN_ID={ros_domain_id}")
        
        # 设置设备
        gpu_id = config.get('gpu_id', 0)
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        print(f"环境 {env_id} 使用设备: {device}")
        
        # 加载模型（只加载一次）
        model = SAC(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            max_action=config['max_action'],
            device=device,
            hidden_dim=config.get('hidden_dim', 1024),
            hidden_depth=config.get('hidden_depth', 3),
            save_every=0,
            load_model=True,
            load_directory=model_path,
            action_noise_std=config.get('action_noise_std', 0.2)
        )
        print(f"环境 {env_id} 模型加载完成: {model_path}")
        
        # 初始化ROS环境（使用第一个距离区间的目标距离）
        if distance_intervals is None or len(distance_intervals) == 0:
            distance_intervals = [config.get('max_target_dist', 3.0)]
        if episodes_per_interval is None:
            episodes_per_interval = total_episodes // len(distance_intervals)
        
        # 初始使用第一个距离区间
        current_interval_idx = 0
        current_target_dist = distance_intervals[0]
        
        ros_env = ROS_env(
            env_id=env_id,  # 传递正确的环境ID
            max_velocity=config['max_velocity'],
            init_target_distance=current_target_dist,  # 使用当前阶段的目标距离
            target_dist_increase=0.0,  # 不自动增长
            max_target_dist=current_target_dist,
            target_reached_delta=config['target_reached_delta'],
            collision_delta=config['collision_delta'],
            neglect_angle=config['neglect_angle'],
            scan_range=config['scan_range'],
            world_size=config['world_size'],
            obs_min_dist=config['obs_min_dist'],
            obs_num=config['obs_num'],
            costmap_resolution=config.get('costmap_resolution', 0.3),
            obstacle_size=config.get('obstacle_size', 0.3),
            obs_distribution_mode=config.get('obs_distribution_mode', 'uniform'),
            # 奖励函数参数
            goal_reward=config.get('goal_reward', 1000.0),
            collision_penalty_base=config.get('collision_penalty_base', config.get('base_collision_penalty', -1000.0)),
            angle_penalty_base=config.get('angle_penalty_base', config.get('angle_base_penalty', 0.0)),
            linear_penalty_base=config.get('linear_penalty_base', config.get('base_linear_penalty', -1.0)),
            yawrate_penalty_base=config.get('yawrate_penalty_base', 0.0),
            enable_obs_penalty=config.get('enable_obs_penalty', True),
            enable_yawrate_penalty=config.get('enable_yawrate_penalty', True),
            enable_angle_penalty=config.get('enable_angle_penalty', True),
            enable_linear_penalty=config.get('enable_linear_penalty', True),
            enable_target_distance_penalty=config.get('enable_target_distance_penalty', False),
            enable_linear_acceleration_oscillation_penalty=config.get('enable_linear_acceleration_oscillation_penalty', False),
            enable_yawrate_oscillation_penalty=config.get('enable_yawrate_oscillation_penalty', False),
            # 障碍物距离惩罚参数
            obs_penalty_threshold=config.get('obs_penalty_threshold', 1.0),
            min_obs_penalty_threshold=config.get('min_obs_penalty_threshold', 0.5),
            obs_penalty_base=config.get('obs_penalty_base', -10.0),
            obs_penalty_power=config.get('obs_penalty_power', 2.0),
            obs_penalty_high_weight=config.get('obs_penalty_high_weight', 1.0),
            obs_penalty_low_weight=config.get('obs_penalty_low_weight', 0.5),
            obs_penalty_middle_ratio=config.get('obs_penalty_middle_ratio', 0.4),
            # 终点距离惩罚参数
            target_distance_penalty_base=config.get('target_distance_penalty_base', -1.0),
            # 震荡惩罚参数
            linear_acceleration_oscillation_penalty_base=config.get('linear_acceleration_oscillation_penalty_base', -1.0),
            yawrate_oscillation_penalty_base=config.get('yawrate_oscillation_penalty_base', -1.0),
            # 时间控制
            sim_time=config.get('sim_time', 0.1),
            step_sleep_time=config.get('step_sleep_time', 0.1),
            eval_sleep_time=config.get('eval_sleep_time', 1.0),
            reset_step_count=config.get('reset_step_count', 3),
            reward_debug=config.get('reward_debug', False),
            # 连通区域选择偏好
            region_select_bias=config.get('region_select_bias', 1.0),
        )
        
        print(f"环境 {env_id} 初始化完成，开始评估数据收集...")
        print(f"环境 {env_id} 距离区间列表: {distance_intervals}, 每个区间 {episodes_per_interval} 个episode")
        
        # 每个环境的episode计数器
        local_episode_counter = 0
        
        while not is_evaluation_complete(evaluation_complete):
            try:
                # 根据已完成的episode数计算当前应该处于哪个区间
                with counter_lock:
                    completed_episodes = episode_counter.value
                
                # 计算当前应该处于哪个区间
                # 需要考虑最后一个区间有额外的episode
                num_intervals = len(distance_intervals)
                calculated_interval_idx = 0
                cumulative_episodes = 0
                
                for idx in range(num_intervals):
                    # 当前区间的episode数量
                    interval_episodes = episodes_per_interval
                    if idx == num_intervals - 1:
                        # 最后一个区间包含额外的episode
                        interval_episodes += remaining_episodes
                    
                    cumulative_episodes += interval_episodes
                    if completed_episodes < cumulative_episodes:
                        calculated_interval_idx = idx
                        break
                else:
                    # 如果所有区间都完成了，使用最后一个区间
                    calculated_interval_idx = num_intervals - 1
                
                # 如果当前区间索引发生变化，更新目标距离
                if calculated_interval_idx != current_interval_idx:
                    # 确保不超过区间数量
                    if calculated_interval_idx < len(distance_intervals):
                        current_interval_idx = calculated_interval_idx
                        current_target_dist = distance_intervals[current_interval_idx]
                        # 更新ROS环境的目标距离设置
                        ros_env.max_target_dist = current_target_dist
                        ros_env.target_dist = current_target_dist
                        print(f"环境 {env_id} 切换到距离区间 {current_interval_idx + 1}/{len(distance_intervals)}: 目标距离 = {current_target_dist}m (已完成 {completed_episodes} 个episode)")
                
                # 重置环境（会使用更新后的max_target_dist）
                latest_scan, distance, cos, sin, collision, goal, last_action, reward = ros_env.reset()
                state, terminal = model.prepare_state(
                    latest_scan, distance, cos, sin, collision, goal, last_action
                )
                
                episode_reward = 0
                episode_steps = 0
                
                # 使用ros_python.py中已计算的初始终点距离
                target_distance = getattr(ros_env, "initial_target_distance", None)
                if target_distance is None:
                    # 如果initial_target_distance未设置，回退到计算方式
                    if ros_env.episode_start_position is not None and ros_env.target is not None:
                        target_distance = np.linalg.norm([
                            ros_env.target[0] - ros_env.episode_start_position[0],
                            ros_env.target[1] - ros_env.episode_start_position[1]
                        ])
                    else:
                        # 如果episode_start_position或target未设置，使用配置的最大目标距离
                        target_distance = config['max_target_dist']
                
                # 最大步数策略（与训练时保持一致）：
                # - 默认：max_steps = max(distance * max_steps_ratio, max_steps_min)
                # - 当 max_steps_ratio == 0 时，使用固定 max_steps_fixed
                if config.get('max_steps_ratio', 0) == 0:
                    max_steps = int(config.get('max_steps_fixed', config.get('max_steps_min', 50)))
                else:
                    calculated_max_steps = int(target_distance * config['max_steps_ratio'])
                    max_steps = max(calculated_max_steps, config['max_steps_min'])
                
                # 收集一个episode的数据（评估时不添加噪声）
                while not terminal and episode_steps < max_steps:
                    # 获取动作（评估时不添加噪声）
                    model_action = model.get_action(state, add_noise=False)
                    ros_action = utils.transfor_action(
                        model_action, 
                        max_velocity=config['max_velocity'], 
                        max_yawrate=config['max_yawrate'],
                        prev_linear_velocity=last_action[0],
                        max_acceleration=config.get('max_acceleration', 1.0),
                        max_deceleration=config.get('max_deceleration', 1.0),
                        step_time=config.get('step_sleep_time', 0.1),
                    )
                    
                    # 执行动作
                    latest_scan, distance, cos, sin, collision, goal, last_action, reward = ros_env.step(
                        lin_velocity=ros_action[0], ang_velocity=ros_action[1]
                    )
                    
                    episode_reward += reward
                    episode_steps += 1
                    
                    # 准备下一个状态
                    next_state, terminal = model.prepare_state(
                        latest_scan, distance, cos, sin, collision, goal, last_action
                    )
                    
                    state = next_state
                
                # 判断episode结束原因
                if goal:
                    episode_ending = "Goal"
                    timeout = False
                elif collision:
                    episode_ending = "Collision"
                    timeout = False
                else:
                    episode_ending = "Timeout"
                    timeout = True
                
                # 使用ros_python.py中已计算的初始终点距离
                target_distance = getattr(ros_env, "initial_target_distance", None)
                if target_distance is None:
                    # 如果initial_target_distance未设置，回退到计算方式
                    if ros_env.episode_start_position is not None and ros_env.target is not None:
                        target_distance = np.linalg.norm([
                            ros_env.target[0] - ros_env.episode_start_position[0],
                            ros_env.target[1] - ros_env.episode_start_position[1]
                        ])
                    else:
                        target_distance = ros_env.target_dist  # 如果无法计算，使用配置的目标距离
                
                # 写入日志文件
                write_episode_result(
                    data_log_file, file_lock,
                    goal, collision, timeout, episode_reward, target_distance, episode_steps
                )
                
                # 更新本地episode计数器
                local_episode_counter += 1
                
                # 获取当前时间
                current_time = datetime.now()
                
                # 读取本 episode 奖励分项（仅打印开启的项，保持与训练日志一致的明细格式）
                goal_sum = getattr(ros_env, "episode_goal_reward", 0.0)
                collision_sum = getattr(ros_env, "episode_collision_penalty", 0.0)
                obs_sum = getattr(ros_env, "episode_obs_penalty", 0.0)
                yaw_sum = getattr(ros_env, "episode_yawrate_penalty", 0.0)
                angle_sum = getattr(ros_env, "episode_angle_penalty", 0.0)
                linear_sum = getattr(ros_env, "episode_linear_penalty", 0.0)
                target_distance_sum = getattr(ros_env, "episode_target_distance_penalty", 0.0)
                linear_acc_osc_sum = getattr(ros_env, "episode_linear_acceleration_oscillation_penalty", 0.0)
                yawrate_osc_sum = getattr(ros_env, "episode_yawrate_oscillation_penalty", 0.0)

                # 读取所有奖惩开关状态（确保所有开启的项都被打印）
                enable_obs = getattr(ros_env, "enable_obs_penalty", False)
                enable_yawrate = getattr(ros_env, "enable_yawrate_penalty", False)
                enable_angle = getattr(ros_env, "enable_angle_penalty", False)
                enable_linear = getattr(ros_env, "enable_linear_penalty", False)
                enable_target_distance = getattr(ros_env, "enable_target_distance_penalty", False)
                enable_linear_acc_osc = getattr(ros_env, "enable_linear_acceleration_oscillation_penalty", False)
                enable_yawrate_osc = getattr(ros_env, "enable_yawrate_oscillation_penalty", False)

                detail_parts = [
                    f"goal={goal_sum:.2f}",
                    f"collision={collision_sum:.2f}",
                ]
                # 所有开启的奖惩项都要打印（即使值为0也要显示）
                if enable_obs:
                    detail_parts.append(f"obs={obs_sum:.2f}")
                if enable_yawrate:
                    detail_parts.append(f"yawrate={yaw_sum:.2f}")
                if enable_angle:
                    detail_parts.append(f"angle={angle_sum:.2f}")
                if enable_linear:
                    detail_parts.append(f"linear={linear_sum:.2f}")
                if enable_target_distance:
                    detail_parts.append(f"target_distance={target_distance_sum:.2f}")
                if enable_linear_acc_osc:
                    detail_parts.append(f"linear_acc_osc={linear_acc_osc_sum:.2f}")
                if enable_yawrate_osc:
                    detail_parts.append(f"yawrate_osc={yawrate_osc_sum:.2f}")

                # 将结束状态放在最前面，然后是总reward，最后是其他奖励分项
                detail_parts_with_total = [f"end={episode_ending}", f"total_reward={episode_reward:.2f}"] + detail_parts
                detail_str = ", ".join(detail_parts_with_total)
                
                # 输出详细的episode信息（含奖励明细），时间戳放在最前面，格式与训练日志一致
                # ros_env.target_dist 是配置的目标距离上限，target_distance 是实际生成的终点距离
                episode_info = (
                    f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} 环境 {env_id} "
                    f"Episode: {local_episode_counter} "
                    f"Target Distance: {ros_env.target_dist:.2f} (actual: {target_distance:.2f}) Steps: {episode_steps}\n"
                    f"  Reward Detail: {detail_str}"
                )
                print(episode_info)
                
                # 报告episode完成
                should_continue = report_episode_complete(
                    total_episodes, episode_counter, evaluation_complete, counter_lock
                )
                
                if not should_continue:
                    print(f"环境 {env_id} 评估完成（已完成所有 {len(distance_intervals)} 个距离区间），退出")
                    break
                
            except Exception as e:
                print(f"环境 {env_id} Episode出错: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)
    
    except Exception as e:
        print(f"环境 {env_id} 初始化失败: {e}")
        import traceback
        traceback.print_exc()


def print_distance_range_report(stats):
    """打印距离区间统计报告（输出到终端，由shell脚本重定向到日志文件）"""
    if stats is None:
        return
    
    lines = [
        f"\n{'='*80}",
        f"距离区间评估报告 - {stats['distance_range']}",
        f"{'='*80}",
        f"Episode总数: {stats['total_episodes']}",
        f"平均奖励: {stats['avg_reward']:.2f}",
        f"平均步数: {stats['avg_steps']:.1f}",
        f"成功次数: {stats['goal_count']} ({stats['goal_rate']*100:.1f}%)",
        f"碰撞次数: {stats['collision_count']} ({stats['collision_rate']*100:.1f}%)",
        f"超时次数: {stats['timeout_count']} ({stats['timeout_rate']*100:.1f}%)",
        f"{'='*80}\n"
    ]
    
    # 输出到终端（日志会由shell脚本重定向到日志文件）
    for line in lines:
        print(line)


class ParallelMultiEnvEvaluator:
    """并行多环境评估器"""
    
    def __init__(self, 
                 num_envs=4,
                 total_episodes=3000,
                 max_target_dist=12.0,
                 distance_ranges=[(0, 3), (3, 6), (6, 9), (9, 12)],
                 distance_intervals=None,  # 距离区间列表，如[3, 6, 9]，用于分段评估
                 state_dim=25,
                 action_dim=2,
                 max_action=1,
                 device=None,
                 model_load_dir=None,
                 max_velocity=1.0,
                 neglect_angle=0,
                 max_yawrate=20.0,
                 scan_range=5,
                 init_target_distance=2.0,
                 target_dist_increase=0.01,
                 target_reached_delta=0.3,
                 collision_delta=0.25,
                 world_size=15,
                 obs_min_dist=2,
                 obs_num=20,
                 gpu_id=0,
                 hidden_dim=1024,
                 hidden_depth=3,  # 默认值改为3，与训练脚本保持一致
                 eval_log_dir=None,
                 # max_steps相关参数
                 max_steps_ratio=0,
                 max_steps_fixed=100,
                 max_steps_min=50,
                 # 奖励函数参数
                 goal_reward=1000.0,
                 base_collision_penalty=-1000.0,
                 angle_base_penalty=0.0,
                 base_linear_penalty=-1.0,
                 yawrate_penalty_base=0.0,
                 enable_obs_penalty=True,
                 enable_yawrate_penalty=True,
                 enable_angle_penalty=True,
                 enable_linear_penalty=True,
                 # 障碍物距离惩罚参数
                 obs_penalty_threshold=1.0,
                 min_obs_penalty_threshold=0.5,
                 obs_penalty_base=-10.0,
                 obs_penalty_power=2.0,
                 obs_penalty_high_weight=1.0,
                 obs_penalty_low_weight=0.5,
                 obs_penalty_middle_ratio=0.4,
                 # 时间控制参数
                 sim_time=0.1,
                 step_sleep_time=0.1,
                 eval_sleep_time=1.0,
                 reset_step_count=3,
                 # 动作噪声参数
                 action_noise_std=0.2,
                 # 调试参数
                 reward_debug=False):
        
        self.num_envs = num_envs
        self.total_episodes = total_episodes
        self.max_target_dist = max_target_dist  # 已弃用，保留用于向后兼容
        # 如果未提供distance_intervals，使用默认值
        if distance_intervals is None:
            self.distance_intervals = [3.0]
        else:
            self.distance_intervals = distance_intervals
        # 从distance_intervals自动生成distance_ranges（用于统计）
        # 例如：[3, 6, 9] -> [(0, 3), (3, 6), (6, 9)]
        self.distance_ranges = []
        prev_dist = 0.0
        for dist in self.distance_intervals:
            self.distance_ranges.append((prev_dist, dist))
            prev_dist = dist
        self.max_velocity = max_velocity
        self.neglect_angle = neglect_angle
        self.max_yawrate = max_yawrate
        self.scan_range = scan_range
        self.init_target_distance = init_target_distance
        self.target_dist_increase = target_dist_increase
        self.target_reached_delta = target_reached_delta
        self.collision_delta = collision_delta
        self.world_size = world_size
        self.obs_min_dist = obs_min_dist
        self.obs_num = obs_num
        
        # 设备配置
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
            print(f"使用GPU: cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
            print("CUDA不可用，使用CPU")
        
        # 模型配置
        self.model_load_dir = Path(model_load_dir) if model_load_dir else Path("src/drl_navigation_ros2/models/SAC")
        
        # 创建评估日志目录（使用与shell脚本相同的日志目录）
        # 优先使用环境变量LOG_DIR（由shell脚本传递），确保使用同一个日志目录
        log_dir_env = os.environ.get('LOG_DIR')
        if log_dir_env:
            # 使用shell脚本传递的日志目录
            eval_log_dir_path = Path(log_dir_env)
            print(f"使用shell脚本传递的日志目录: {eval_log_dir_path}")
        elif eval_log_dir:
            # 如果指定了eval_log_dir（向后兼容）
            eval_log_dir_path = Path(eval_log_dir)
        else:
            # 默认使用log/evaluation目录（不应该到达这里，因为shell脚本会传递LOG_DIR）
            script_dir = Path(__file__).parent.parent.parent
            eval_log_base_dir = script_dir / "log" / "evaluation"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            eval_log_dir_path = eval_log_base_dir / f"eval_{timestamp}"
            print(f"警告: 未找到LOG_DIR环境变量，创建新的日志目录: {eval_log_dir_path}")
        
        eval_log_dir_path.mkdir(parents=True, exist_ok=True)
        
        # 数据日志文件路径（在子目录下）
        # evaluation_data.jsonl 保存在与shell脚本日志相同的目录中
        # 不再创建evaluation.log，所有日志都输出到shell脚本的日志文件中
        self.data_log_file = eval_log_dir_path / "evaluation_data.jsonl"
        
        # 初始化共享的计数器变量
        self.episode_counter = mp.Value('i', 0)  # 已完成的episode数
        self.evaluation_complete = mp.Value('b', False)  # 评估是否完成
        self.counter_lock = mp.Lock()  # 计数器锁
        
        # 文件锁，用于保护数据日志文件的互斥写入
        self.file_lock = mp.Lock()
        
        # 配置字典
        self.config = {
            'num_envs': num_envs,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'max_action': max_action,
            'max_velocity': max_velocity,
            'neglect_angle': neglect_angle,
            'max_yawrate': max_yawrate,
            'scan_range': scan_range,
            'max_target_dist': max_target_dist,
            'init_target_distance': init_target_distance,
            'target_dist_increase': target_dist_increase,
            'target_reached_delta': target_reached_delta,
            'collision_delta': collision_delta,
            'world_size': world_size,
            'obs_min_dist': obs_min_dist,
            'obs_num': obs_num,
            'gpu_id': gpu_id,
            'hidden_dim': hidden_dim,
            'hidden_depth': hidden_depth,
            # max_steps相关参数
            'max_steps_ratio': max_steps_ratio,
            'max_steps_fixed': max_steps_fixed,
            'max_steps_min': max_steps_min,
            # 奖励函数参数（同时保存两种参数名以兼容不同代码路径）
            'goal_reward': goal_reward,
            'base_collision_penalty': base_collision_penalty,  # 旧参数名（向后兼容）
            'collision_penalty_base': base_collision_penalty,  # 训练配置中的参数名
            'angle_base_penalty': angle_base_penalty,  # 旧参数名（向后兼容）
            'angle_penalty_base': angle_base_penalty,  # 训练配置中的参数名
            'base_linear_penalty': base_linear_penalty,  # 旧参数名（向后兼容）
            'linear_penalty_base': base_linear_penalty,  # 训练配置中的参数名
            'yawrate_penalty_base': yawrate_penalty_base,
            'enable_obs_penalty': enable_obs_penalty,
            'enable_yawrate_penalty': enable_yawrate_penalty,
            'enable_angle_penalty': enable_angle_penalty,
            'enable_linear_penalty': enable_linear_penalty,
            # 障碍物距离惩罚参数
            'obs_penalty_threshold': obs_penalty_threshold,
            'min_obs_penalty_threshold': min_obs_penalty_threshold,
            'obs_penalty_base': obs_penalty_base,
            'obs_penalty_power': obs_penalty_power,
            'obs_penalty_high_weight': obs_penalty_high_weight,
            'obs_penalty_low_weight': obs_penalty_low_weight,
            'obs_penalty_middle_ratio': obs_penalty_middle_ratio,
            # 时间控制参数
            'sim_time': sim_time,
            'step_sleep_time': step_sleep_time,
            'eval_sleep_time': eval_sleep_time,
            'reset_step_count': reset_step_count,
            # 动作噪声参数
            'action_noise_std': action_noise_std,
            # 调试参数
            'reward_debug': reward_debug
        }
        
        # 初始化信息不再打印，所有信息已在shell脚本的日志中记录
    
    def run_evaluation(self):
        """运行并行多环境评估（按距离区间分段评估）"""
        
        # 将 Path 对象转换为字符串
        model_load_dir_str = str(self.model_load_dir)
        data_log_file_str = str(self.data_log_file)
        
        # 计算每个距离区间的episode数量
        num_intervals = len(self.distance_intervals)
        episodes_per_interval = self.total_episodes // num_intervals
        remaining_episodes = self.total_episodes % num_intervals
        
        print(f"\n开始分段评估，距离区间: {self.distance_intervals}")
        print(f"每个距离区间评估 {episodes_per_interval} 个episode")
        if remaining_episodes > 0:
            print(f"最后一个距离区间额外评估 {remaining_episodes} 个episode")
        print("进程将持续运行，自动切换距离区间\n")
        
        # 重置计数器
        self.episode_counter.value = 0
        self.evaluation_complete.value = False
        
        # 启动数据收集进程（只启动一次，进程会持续运行并自动切换区间）
        collect_processes = []
        for env_id in range(self.num_envs):
            p = mp.Process(
                target=collect_evaluation_episodes,
                args=(env_id, model_load_dir_str, self.config,
                      self.total_episodes, self.episode_counter, 
                      self.evaluation_complete, self.counter_lock,
                      data_log_file_str, self.file_lock, 
                      self.distance_intervals, episodes_per_interval, remaining_episodes)
            )
            p.start()
            collect_processes.append(p)
        
        try:
            # 等待所有进程完成（它们会自动切换距离区间）
            while True:
                # 检查是否所有episode都完成
                if is_evaluation_complete(self.evaluation_complete):
                    print(f"\n所有距离区间的评估完成，等待进程退出...")
                    break
                
                # 显示进度
                progress_info, progress = get_progress(
                    self.total_episodes, self.episode_counter, self.counter_lock
                )
                # print(f"进度: {progress_info} ({progress*100:.1f}%)", end='\r')
                
                time.sleep(1)
            
            # 等待所有进程结束
            for p in collect_processes:
                p.join(timeout=10)
            
            # 输出所有距离区间的统计结果（在所有分段评估完成后）
            summary_lines = [
                "\n\n" + "="*80,
                f"评估完成 - 按距离区间统计 (总episode数: {self.total_episodes})",
                "="*80
            ]
            
            for line in summary_lines:
                print(line)
            
            # 按距离区间分析结果（从数据日志文件读取）
            all_stats = analyze_results_by_distance(self.data_log_file, self.distance_ranges)
            for distance_range, stats in all_stats.items():
                if stats:
                    print_distance_range_report(stats)
            
            # 输出结束信息
            end_lines = [
                "="*80,
                f"评估结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"评估数据已保存至: {self.data_log_file}",
                "="*80
            ]
            
            for line in end_lines:
                print(line)
                
        except KeyboardInterrupt:
            print("\n收到中断信号，正在停止评估...")
        
        finally:
            # 清理所有进程
            for p in collect_processes:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        p.kill()
            
            print("并行评估已停止")


def load_config(config_path=None):
    """加载配置文件"""
    if config_path is None:
        # 默认配置文件路径
        config_path = Path(__file__).parent.parent.parent / "config" / "evaluation.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='并行多环境评估脚本')
    
    # 配置文件参数
    parser.add_argument('--config', type=str, default=None, help='配置文件路径（默认：config/evaluation.yaml）')
    
    # 评估参数（命令行参数可以覆盖配置文件）
    parser.add_argument('--num_envs', type=int, default=None, help='并行环境数量（覆盖配置文件）')
    parser.add_argument('--total_episodes', type=int, default=None, help='总评估episode数量（覆盖配置文件）')
    parser.add_argument('--max_target_dist', type=float, default=None, help='固定的最大目标距离（覆盖配置文件）')
    # distance_ranges已移除，现在从distance_intervals自动生成
    parser.add_argument('--gpu_id', type=int, default=None, help='使用的GPU编号（覆盖配置文件）')
    
    # 环境参数（命令行参数可以覆盖配置文件）
    parser.add_argument('--max_velocity', type=float, default=None, help='最大速度（覆盖配置文件）')
    parser.add_argument('--neglect_angle', type=int, default=None, help='忽略角度（覆盖配置文件）')
    parser.add_argument('--max_yawrate', type=float, default=None, help='最大偏航率（覆盖配置文件）')
    parser.add_argument('--scan_range', type=int, default=None, help='扫描范围（覆盖配置文件）')
    parser.add_argument('--init_target_distance', type=float, default=None, help='初始目标距离（覆盖配置文件）')
    parser.add_argument('--target_dist_increase', type=float, default=None, help='目标距离增加量（覆盖配置文件）')
    parser.add_argument('--target_reached_delta', type=float, default=None, help='目标到达判断阈值（覆盖配置文件）')
    parser.add_argument('--collision_delta', type=float, default=None, help='碰撞判断阈值（覆盖配置文件）')
    parser.add_argument('--world_size', type=int, default=None, help='世界大小（覆盖配置文件）')
    parser.add_argument('--obs_min_dist', type=float, default=None, help='障碍物最小距离（覆盖配置文件）')
    parser.add_argument('--obs_num', type=int, default=None, help='障碍物数量（覆盖配置文件）')
    
    # 模型参数（命令行参数可以覆盖配置文件）
    parser.add_argument('--model_load_dir', type=str, default=None, help='模型加载目录（覆盖配置文件）')
    parser.add_argument('--hidden_dim', type=int, default=None, help='神经网络隐藏层维度（覆盖配置文件）')
    parser.add_argument('--hidden_depth', type=int, default=None, help='神经网络隐藏层深度（覆盖配置文件）')
    
    # 日志参数（命令行参数可以覆盖配置文件）
    parser.add_argument('--eval_log_dir', type=str, default=None, help='评估日志保存目录（覆盖配置文件，已弃用，使用环境变量LOG_DIR）')
    parser.add_argument('--log_dir', type=str, default=None, help='日志目录（从shell脚本传递，优先使用环境变量LOG_DIR）')
    
    # 算法参数（命令行参数可以覆盖配置文件）
    parser.add_argument('--state_dim', type=int, default=None, help='状态维度（覆盖配置文件）')
    parser.add_argument('--action_dim', type=int, default=None, help='动作维度（覆盖配置文件）')
    parser.add_argument('--max_action', type=float, default=None, help='最大动作值（覆盖配置文件）')
    
    # 奖励函数参数（命令行参数可以覆盖配置文件）
    parser.add_argument('--goal_reward', type=float, default=None, help='到达目标的奖励（覆盖配置文件）')
    parser.add_argument('--base_collision_penalty', type=float, default=None, help='基础碰撞惩罚（覆盖配置文件）')
    parser.add_argument('--angle_base_penalty', type=float, default=None, help='角度偏差基础惩罚（覆盖配置文件）')
    parser.add_argument('--base_linear_penalty', type=float, default=None, help='线速度基础惩罚（覆盖配置文件）')
    
    # 障碍物距离惩罚参数（命令行参数可以覆盖配置文件）
    parser.add_argument('--obs_penalty_threshold', type=float, default=None, help='障碍物距离惩罚阈值（覆盖配置文件）')
    parser.add_argument('--obs_penalty_base', type=float, default=None, help='障碍物距离惩罚基础系数（覆盖配置文件）')
    parser.add_argument('--obs_penalty_power', type=float, default=None, help='障碍物距离惩罚指数（覆盖配置文件）')
    
    # 时间控制参数（命令行参数可以覆盖配置文件）
    parser.add_argument('--step_sleep_time', type=float, default=None, help='step方法中的sleep时间（覆盖配置文件）')
    parser.add_argument('--eval_sleep_time', type=float, default=None, help='eval方法中的sleep时间（覆盖配置文件）')
    parser.add_argument('--reset_step_count', type=int, default=None, help='reset方法中调用step的次数（覆盖配置文件）')
    
    # 动作噪声参数（命令行参数可以覆盖配置文件）
    parser.add_argument('--action_noise_std', type=float, default=None, help='动作噪声标准差（覆盖配置文件）')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 设置多进程启动方法为spawn
    mp.set_start_method('spawn', force=True)
    
    args = parse_args()
    
    # ==================== 加载评估配置文件 ====================
    config_path_env = os.environ.get('EVALUATION_CONFIG_PATH', args.config)
    eval_config = load_config(config_path_env)
    
    # 获取实际使用的配置文件路径用于打印
    actual_config_path = config_path_env if config_path_env else (Path(__file__).parent.parent.parent / "config" / "evaluation.yaml")
    print(f"使用评估配置文件: {actual_config_path}")
    
    # ==================== 从评估配置文件读取评估相关参数 ====================
    # 评估相关参数（这些参数需要手动设置，与评估相关）
    num_envs = args.num_envs if args.num_envs is not None else eval_config.get('num_envs', 4)
    total_episodes = args.total_episodes if hasattr(args, 'total_episodes') and args.total_episodes is not None else eval_config.get('total_episodes', 3000)
    max_target_dist = args.max_target_dist if hasattr(args, 'max_target_dist') and args.max_target_dist is not None else eval_config.get('max_target_dist', 12.0)  # 已弃用，保留用于向后兼容
    # 读取距离区间列表（用于分段评估）
    distance_intervals = eval_config.get('distance_intervals', None)
    if distance_intervals is None:
        # 如果未提供distance_intervals，使用默认值
        distance_intervals = [3.0]
    
    # 从distance_intervals自动生成distance_ranges（用于统计）
    # 例如：[3, 6, 9] -> [(0, 3), (3, 6), (6, 9)]
    distance_ranges = []
    prev_dist = 0.0
    for dist in distance_intervals:
        distance_ranges.append((prev_dist, dist))
        prev_dist = dist
    
    gpu_id = args.gpu_id if hasattr(args, 'gpu_id') and args.gpu_id is not None else eval_config.get('gpu_id', 0)
    
    # ==================== 从模型目录加载训练配置文件 ====================
    # 获取模型加载目录
    model_load_dir = args.model_load_dir if hasattr(args, 'model_load_dir') and args.model_load_dir is not None else eval_config.get('model_load_dir', None)
    if model_load_dir is None:
        raise ValueError("必须指定模型加载目录（通过--model_load_dir参数或evaluation.yaml中的model_load_dir）")
    
    model_load_dir_path = Path(model_load_dir)
    model_config_path = model_load_dir_path / "config_used.yaml"
    
    if not model_config_path.exists():
        print(f"警告: 模型目录中未找到config_used.yaml: {model_config_path}")
        print("将仅使用evaluation.yaml中的参数")
        train_config = {}
    else:
        print(f"从模型目录加载训练配置: {model_config_path}")
        train_config = load_config(model_config_path)
        # 调试：打印训练配置中的网络结构参数
        if train_config:
            print(f"训练配置中的网络结构参数: hidden_dim={train_config.get('hidden_dim', '未设置')}, hidden_depth={train_config.get('hidden_depth', '未设置')}")
            # 调试：打印训练配置中的障碍物数量
            print(f"训练配置中的障碍物数量: obs_num={train_config.get('obs_num', '未设置')}")
    
    # 合并配置：训练配置作为基础，评估配置仅用于评估相关参数
    # 对于非评估相关参数，优先使用训练配置
    config = {}
    config.update(train_config)  # 先加载训练配置（包含所有训练参数）
    # 评估相关参数已从评估配置读取（num_envs, total_episodes, distance_intervals, gpu_id等）
    # distance_ranges已从distance_intervals自动生成
    # 其他参数都从训练配置读取，确保与训练时一致
    
    # 机器人和环境参数
    max_velocity = args.max_velocity if hasattr(args, 'max_velocity') and args.max_velocity is not None else config.get('max_velocity', 1.0)
    neglect_angle = args.neglect_angle if hasattr(args, 'neglect_angle') and args.neglect_angle is not None else config.get('neglect_angle', 0)
    max_yawrate = args.max_yawrate if hasattr(args, 'max_yawrate') and args.max_yawrate is not None else config.get('max_yawrate', 20.0)
    scan_range = args.scan_range if hasattr(args, 'scan_range') and args.scan_range is not None else config.get('scan_range', 5)
    init_target_distance = args.init_target_distance if hasattr(args, 'init_target_distance') and args.init_target_distance is not None else config.get('init_target_distance', 2.0)
    target_dist_increase = args.target_dist_increase if hasattr(args, 'target_dist_increase') and args.target_dist_increase is not None else config.get('target_dist_increase', 0.01)
    target_reached_delta = args.target_reached_delta if hasattr(args, 'target_reached_delta') and args.target_reached_delta is not None else config.get('target_reached_delta', 0.3)
    collision_delta = args.collision_delta if hasattr(args, 'collision_delta') and args.collision_delta is not None else config.get('collision_delta', 0.25)
    world_size = args.world_size if hasattr(args, 'world_size') and args.world_size is not None else config.get('world_size', 15)
    obs_min_dist = args.obs_min_dist if hasattr(args, 'obs_min_dist') and args.obs_min_dist is not None else config.get('obs_min_dist', 2)
    obs_num = args.obs_num if hasattr(args, 'obs_num') and args.obs_num is not None else config.get('obs_num', 20)
    # 调试：打印最终使用的obs_num值
    print(f"最终使用的障碍物数量: obs_num={obs_num} (来源: {'命令行参数' if hasattr(args, 'obs_num') and args.obs_num is not None else ('训练配置' if 'obs_num' in config else '默认值')})")
    
    # max_steps相关参数（从训练配置读取）
    max_steps_ratio = config.get('max_steps_ratio', 0)
    max_steps_fixed = config.get('max_steps_fixed', 100)
    max_steps_min = config.get('max_steps_min', 50)
    
    # 模型参数
    state_dim = args.state_dim if hasattr(args, 'state_dim') and args.state_dim is not None else config.get('state_dim', 25)
    action_dim = args.action_dim if hasattr(args, 'action_dim') and args.action_dim is not None else config.get('action_dim', 2)
    max_action = args.max_action if hasattr(args, 'max_action') and args.max_action is not None else config.get('max_action', 1.0)
    
    # 网络结构参数（必须从训练配置读取，确保与模型结构一致）
    # 调试：打印config中的值
    print(f"调试: config中的hidden_dim={config.get('hidden_dim', '未设置')}, hidden_depth={config.get('hidden_depth', '未设置')}")
    print(f"调试: config中hidden_depth的类型={type(config.get('hidden_depth', None))}, 值={repr(config.get('hidden_depth', None))}")
    
    hidden_dim = args.hidden_dim if hasattr(args, 'hidden_dim') and args.hidden_dim is not None else config.get('hidden_dim', 1024)
    hidden_depth = args.hidden_depth if hasattr(args, 'hidden_depth') and args.hidden_depth is not None else config.get('hidden_depth', 3)
    
    # 验证网络结构参数是否从训练配置正确读取
    if 'hidden_dim' not in config or 'hidden_depth' not in config:
        print(f"警告: 未从训练配置中找到网络结构参数，使用默认值 hidden_dim={hidden_dim}, hidden_depth={hidden_depth}")
    else:
        print(f"从训练配置读取网络结构参数: hidden_dim={hidden_dim}, hidden_depth={hidden_depth}")
        # 如果读取到的值与训练配置不一致，发出警告
        if train_config and train_config.get('hidden_depth') != hidden_depth:
            print(f"错误: 读取到的hidden_depth={hidden_depth}与训练配置中的hidden_depth={train_config.get('hidden_depth')}不一致！")
    
    # 路径参数
    # model_load_dir 已在上面从评估配置读取，这里不再重复读取
    # eval_log_dir 不再需要，日志统一保存到 log/evaluation 目录
    eval_log_dir = None
    
    # 奖励函数参数（优先使用训练配置中的参数名，向后兼容旧参数名）
    goal_reward = args.goal_reward if hasattr(args, 'goal_reward') and args.goal_reward is not None else config.get('goal_reward', 1000.0)
    # collision_penalty_base是训练配置中的参数名，base_collision_penalty是旧参数名（向后兼容）
    base_collision_penalty = args.base_collision_penalty if hasattr(args, 'base_collision_penalty') and args.base_collision_penalty is not None else config.get('collision_penalty_base', config.get('base_collision_penalty', -1000.0))
    # angle_penalty_base是训练配置中的参数名，angle_base_penalty是旧参数名（向后兼容）
    angle_base_penalty = args.angle_base_penalty if hasattr(args, 'angle_base_penalty') and args.angle_base_penalty is not None else config.get('angle_penalty_base', config.get('angle_base_penalty', 0.0))
    # linear_penalty_base是训练配置中的参数名，base_linear_penalty是旧参数名（向后兼容）
    base_linear_penalty = args.base_linear_penalty if hasattr(args, 'base_linear_penalty') and args.base_linear_penalty is not None else config.get('linear_penalty_base', config.get('base_linear_penalty', -1.0))
    yawrate_penalty_base = config.get('yawrate_penalty_base', 0.0)
    
    # 奖励/惩罚开关参数
    def parse_bool(val, default=True):
        if isinstance(val, bool):
            return val
        if val is None:
            return default
        if isinstance(val, str):
            low = val.strip().lower()
            if low in ("true", "1", "yes", "y", "on"):
                return True
            if low in ("false", "0", "no", "n", "off"):
                return False
        return default
    
    enable_obs_penalty = parse_bool(config.get('enable_obs_penalty', True), True)
    enable_yawrate_penalty = parse_bool(config.get('enable_yawrate_penalty', True), True)
    enable_angle_penalty = parse_bool(config.get('enable_angle_penalty', True), True)
    enable_linear_penalty = parse_bool(config.get('enable_linear_penalty', True), True)
    
    # 障碍物距离惩罚参数
    obs_penalty_threshold = args.obs_penalty_threshold if hasattr(args, 'obs_penalty_threshold') and args.obs_penalty_threshold is not None else config.get('obs_penalty_threshold', 1.0)
    min_obs_penalty_threshold = config.get('min_obs_penalty_threshold', 0.5)
    obs_penalty_base = args.obs_penalty_base if hasattr(args, 'obs_penalty_base') and args.obs_penalty_base is not None else config.get('obs_penalty_base', -10.0)
    obs_penalty_power = args.obs_penalty_power if hasattr(args, 'obs_penalty_power') and args.obs_penalty_power is not None else config.get('obs_penalty_power', 2.0)
    obs_penalty_high_weight = config.get('obs_penalty_high_weight', 1.0)
    obs_penalty_low_weight = config.get('obs_penalty_low_weight', 0.5)
    obs_penalty_middle_ratio = config.get('obs_penalty_middle_ratio', 0.4)
    
    # 时间控制参数
    sim_time = config.get('sim_time', 0.1)
    step_sleep_time = args.step_sleep_time if hasattr(args, 'step_sleep_time') and args.step_sleep_time is not None else config.get('step_sleep_time', 0.1)
    eval_sleep_time = args.eval_sleep_time if hasattr(args, 'eval_sleep_time') and args.eval_sleep_time is not None else config.get('eval_sleep_time', 1.0)
    reset_step_count = args.reset_step_count if hasattr(args, 'reset_step_count') and args.reset_step_count is not None else config.get('reset_step_count', 3)
    
    # 动作噪声参数
    action_noise_std = args.action_noise_std if hasattr(args, 'action_noise_std') and args.action_noise_std is not None else config.get('action_noise_std', 0.2)
    
    # 调试参数
    reward_debug = parse_bool(config.get('reward_debug', False), False)
    
    # ==================== 打印完整配置信息（在读取训练配置后）====================
    print("\n" + "="*80)
    print("评估配置信息（从训练配置读取的参数已更新）")
    print("="*80)
    print("评估相关参数:")
    print(f"  - 并行环境数: {num_envs}")
    print(f"  - 总episode数: {total_episodes}")
    print(f"  - 距离区间列表（分段评估）: {distance_intervals}")
    print(f"  - 距离区间（用于统计）: {distance_ranges}")
    print(f"  - GPU ID: {gpu_id}")
    print("环境参数（从训练配置读取）:")
    print(f"  - 最大速度: {max_velocity}")
    print(f"  - 忽略角度: {neglect_angle} 度")
    print(f"  - 最大偏航率: {max_yawrate} 度/秒")
    print(f"  - 扫描范围: {scan_range}")
    print(f"  - 初始目标距离: {init_target_distance}")
    print(f"  - 目标距离增加量: {target_dist_increase}")
    print(f"  - 目标到达判断阈值: {target_reached_delta}")
    print(f"  - 碰撞判断阈值: {collision_delta}")
    print(f"  - 世界大小: {world_size} 米")
    print(f"  - 障碍物最小距离: {obs_min_dist} 米")
    print(f"  - 障碍物数量: {obs_num} (从训练配置读取)")
    print("模型参数（从训练配置读取）:")
    print(f"  - 动作维度: {action_dim}")
    print(f"  - 最大动作值: {max_action}")
    print(f"  - 状态维度: {state_dim}")
    print(f"  - 网络隐藏层维度: {hidden_dim}")
    print(f"  - 网络隐藏层深度: {hidden_depth}")
    print("模型配置:")
    print(f"  - 模型加载目录: {model_load_dir}")
    print("="*80 + "\n")
    
    # 创建评估器
    evaluator = ParallelMultiEnvEvaluator(
        num_envs=num_envs,
        total_episodes=total_episodes,
        max_target_dist=max_target_dist,
        distance_ranges=distance_ranges,
        distance_intervals=distance_intervals,
        max_velocity=max_velocity,
        neglect_angle=neglect_angle,
        max_yawrate=max_yawrate,
        scan_range=scan_range,
        init_target_distance=init_target_distance,
        target_dist_increase=target_dist_increase,
        target_reached_delta=target_reached_delta,
        collision_delta=collision_delta,
        world_size=world_size,
        obs_min_dist=obs_min_dist,
        obs_num=obs_num,
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        model_load_dir=model_load_dir,
        gpu_id=gpu_id,
        hidden_dim=hidden_dim,
        hidden_depth=hidden_depth,
        eval_log_dir=None,  # 不再使用eval_log_dir，日志统一保存到log/evaluation目录
        max_steps_ratio=max_steps_ratio,
        max_steps_fixed=max_steps_fixed,
        max_steps_min=max_steps_min,
        goal_reward=goal_reward,
        base_collision_penalty=base_collision_penalty,
        angle_base_penalty=angle_base_penalty,
        base_linear_penalty=base_linear_penalty,
        yawrate_penalty_base=yawrate_penalty_base,
        enable_obs_penalty=enable_obs_penalty,
        enable_yawrate_penalty=enable_yawrate_penalty,
        enable_angle_penalty=enable_angle_penalty,
        enable_linear_penalty=enable_linear_penalty,
        obs_penalty_threshold=obs_penalty_threshold,
        min_obs_penalty_threshold=min_obs_penalty_threshold,
        obs_penalty_base=obs_penalty_base,
        obs_penalty_power=obs_penalty_power,
        obs_penalty_high_weight=obs_penalty_high_weight,
        obs_penalty_low_weight=obs_penalty_low_weight,
        obs_penalty_middle_ratio=obs_penalty_middle_ratio,
        sim_time=sim_time,
        step_sleep_time=step_sleep_time,
        eval_sleep_time=eval_sleep_time,
        reset_step_count=reset_step_count,
        action_noise_std=action_noise_std,
        reward_debug=reward_debug
    )
    
    # 开始评估
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()

