#!/usr/bin/env python3
"""
多环境并行训练脚本 - 真正的并行架构
数据收集和模型训练同时进行：
- X个环境并行收集数据，使用最新模型
- 独立线程监控缓冲区，当收集到Y个数据时触发训练
- 训练完成后更新模型，环境继续使用新模型收集数据
"""
import argparse
import multiprocessing as mp
import queue
import threading
import time
import copy
import math
from pathlib import Path
import shutil
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
import yaml
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from collections import deque
from typing import Optional

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from SAC.SAC import SAC
from ros_python import ROS_env
from replay_buffer import ReplayBuffer, StratifiedReplayBuffer, NumpyReplayBuffer, NumpyStratifiedReplayBuffer
import utils
from logging_utils import CollectLogger, TrainLogger, EnvLogger, RewardLogger, NodesLogger, multi_env_log_paths, append_timestamped_line


# ==================== Phase定义 ====================
# 统一的阶段控制，替代原来的phase_ref + collection_start_event + stop_collection_event
PHASE_TRAIN_COLLECT = 0   # 训练阶段，允许开始/继续收集训练episode
PHASE_TRAIN_DRAIN = 1     # 训练阶段，本轮"收尾"：不再启动新的训练episode，只让已在跑的episode跑到最小门槛后尽快结束
PHASE_EVAL_COLLECT = 2    # 评估阶段，收集评估episode（无噪声、不写replay，只统计）
PHASE_EVAL_DRAIN = 3      # 评估阶段收尾：不再启动新的评估episode，正在进行的episode满足最小门槛后尽快截断并丢弃（不计入统计）
PHASE_PAUSE = 4           # 完全暂停：既不收集训练episode，也不收集评估episode，所有环境进程只是在while里sleep等待下一次phase切换
PHASE_STOP = 5            # 停止：环境进程看到后直接break退出循环


def load_config(config_path=None):
    """加载配置文件，若未指定则使用默认统一配置"""
    default_path = Path(__file__).parent.parent.parent / "config" / "train.yaml"
    path = Path(config_path) if config_path else default_path
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"[ERROR] 警告: 未找到配置文件 {path}")
        return {}
    except Exception as e:
        print(f"[ERROR] 警告: 读取配置文件失败 {path}: {e}")
        return {}


def _is_valid_experience(exp):
    """检查经验是否包含NaN或Inf值
    
    Args:
        exp: 经验元组 (state, action, reward, done, next_state)
        
    Returns:
        bool: True表示经验有效，False表示包含无效值
    """
    state, action, reward, done, next_state = exp
    
    # 检查state和next_state
    for name, data in [("state", state), ("next_state", next_state), ("action", action)]:
        arr = np.asarray(data)
        if not np.isfinite(arr).all():
            return False
    
    # 检查reward和done
    if not np.isfinite(reward):
        return False
    if not np.isfinite(done):
        return False
    
    return True


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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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


def _is_valid_env_return(latest_scan, distance, distance_raw, cos, sin, collision, goal, reward, current_v, current_w):
    """检查环境 reset/step 返回值是否包含 NaN 或 Inf 值。

    注意：ROS_env.step() 不再返回 last_action，动作由调用方维护，
    因此这里不再校验 last_action。
    
    Args:
        latest_scan: 激光扫描数据（数组）
        distance: 距离（标量）
        distance_raw: 原始距离（标量）
        cos: 余弦值（标量）
        sin: 正弦值（标量）
        collision: 碰撞标志（布尔值）
        goal: 目标到达标志（布尔值）
        reward: 奖励（标量）
        current_v: 当前线速度（标量）
        current_w: 当前角速度（标量）
        
    Returns:
        bool: True表示返回值有效，False表示包含无效值
    """
    def _all_finite(x) -> bool:
        """对标量/数组统一做 isfinite 检查；遇到异常直接判为无效。"""
        try:
            return bool(np.isfinite(np.asarray(x)).all())
        except Exception:
            try:
                return bool(np.isfinite(float(x)))
            except Exception:
                return False

    # latest_scan 为数组（或可转数组）
    if not _all_finite(latest_scan):
        return False

    # 其余为标量/布尔：统一用 _all_finite 覆盖（布尔也会返回 True）
    for value in (distance, distance_raw, cos, sin, reward, current_v, current_w, collision, goal):
        if not _all_finite(value):
            return False

    return True


def _has_nan_inf(x) -> bool:
    """安全判断对象是否包含 NaN/Inf（异常也视为无效）。"""
    try:
        return not bool(np.isfinite(np.asarray(x)).all())
    except Exception:
        return True


def verify_actor_weight_consistency(local_model, latest_model_state, env_id, config):
    """验证actor网络权重一致性（只匹配actor网络）
    
    Args:
        local_model: 本地模型实例
        latest_model_state: 从文件加载的模型状态字典（只包含actor权重）
        env_id: 环境ID（用于日志）
        config: 配置字典（用于检查enable_weight_consistency_check）
    
    Returns:
        bool: True表示权重匹配，False表示权重不匹配（仅在启用检查时返回有意义的值）
    """
    # 只有在配置中启用权重一致性检查时才进行验证
    if not config.get('enable_weight_consistency_check', False):
        return True
    
    print(f"环境 {env_id} 验证加载后的模型权重:")
    # actor_only模式下只返回actor权重
    local_state_dict_for_log = local_model.state_dict(actor_only=True)
    
    # 打印模型结构概览
    print(f"环境 {env_id} 从文件加载的模型结构概览:")
    for key, value in local_state_dict_for_log.items():
        if hasattr(value, 'shape') and value.numel() > 0:
            try:
                print(f"  环境 {env_id} {key} 形状: {value.shape}, 均值: {value.mean().item():.2f}, 标准差: {value.std().item():.2f}")
            except Exception:
                print(f"  环境 {env_id} {key} 形状: {value.shape}, 类型: {type(value)}")
        elif isinstance(value, dict):
            print(f"  环境 {env_id} {key} 子键: {list(value.keys())}")
            for subkey, subvalue in value.items():
                if hasattr(subvalue, 'shape') and subvalue.numel() > 0:
                    try:
                        print(f"    环境 {env_id} {key}.{subkey} 形状: {subvalue.shape}, 均值: {subvalue.mean().item():.2f}, 标准差: {subvalue.std().item():.2f}")
                    except Exception:
                        print(f"    环境 {env_id} {key}.{subkey} 形状: {subvalue.shape}, 类型: {type(subvalue)}")
                else:
                    print(f"    环境 {env_id} {key}.{subkey} 形状: {subvalue.shape}, 类型: {type(subvalue)}")
        else:
            print(f"  环境 {env_id} {key} 值: {value}, 类型: {type(value)}")
    
    # 权重一致性验证（只验证actor权重）
    print(f"环境 {env_id} 权重一致性验证:")
    local_state_dict = local_model.state_dict(actor_only=True)  # actor_only模式下只返回actor权重
    weights_match = True
    
    for key, value in latest_model_state.items():
        if isinstance(value, dict):  # actor权重是一个字典
            # 验证actor的每个子权重
            if key in local_state_dict:
                for subkey, subvalue in value.items():
                    if hasattr(subvalue, 'shape') and subvalue.numel() > 0:
                        local_subvalue = local_state_dict[key][subkey]
                        if torch.allclose(subvalue, local_subvalue, atol=1e-6):
                            print(f"  ✓ {key}.{subkey} 权重完全匹配")
                        else:
                            print(f"  ✗ {key}.{subkey} 权重不匹配!")
                            print(f"    文件权重均值: {subvalue.mean().item():.2f}, 本地权重均值: {local_subvalue.mean().item():.2f}")
                            print(f"    文件权重标准差: {subvalue.std().item():.2f}, 本地权重标准差: {local_subvalue.std().item():.2f}")
                            weights_match = False
            else:
                print(f"  [ERROR] [ERROR] 警告: 键 {key} 不在本地模型状态字典中")
        elif hasattr(value, 'shape') and value.numel() > 0:
            # 直接是tensor的情况（不应该出现在actor_only模式下）
            if key in local_state_dict:
                local_value = local_state_dict[key]
                if torch.allclose(value, local_value, atol=1e-6):
                    print(f"  ✓ {key} 权重完全匹配")
                else:
                    print(f"  ✗ {key} 权重不匹配!")
                    print(f"    文件权重均值: {value.mean().item():.2f}, 本地权重均值: {local_value.mean().item():.2f}")
                    print(f"    文件权重标准差: {value.std().item():.2f}, 本地权重标准差: {local_value.std().item():.2f}")
                    weights_match = False
    
    # 输出汇总结果
    if weights_match:
        print(f"环境 {env_id} ✓ 所有权重完全匹配!")
    else:
        print(f"环境 {env_id} ✗ 发现权重不匹配!")
    
    return weights_match


class LocalReplayBuffer:
    """训练线程本地的高性能重放缓冲区
    
    设计目标：
    - 避免 Python 层 for 循环逐样本/逐字段拆解
    - 使用 NumPy 连续内存按字段存储，采样时一次性切片
    - 保持与旧版 LocalReplayBuffer.sample_batch 接口兼容
    """
    
    def __init__(self, max_size: int, dtype=np.float32, recent_buffer_ratio=0.1, recent_batch_ratio=0.3):
        self.max_size = max_size
        self.count = 0          # 当前有效样本数量
        self.write_pos = 0      # 循环写入指针
        self.initialized = False
        self.dtype = dtype
        self.filtered_count = 0  # 统计被过滤的无效经验数量
        self.recent_buffer_ratio = recent_buffer_ratio  # 最新数据比例
        self.recent_batch_ratio = recent_batch_ratio    # batch中来自最新数据的比例
        
        # 延迟初始化实际存储数组（直到拿到第一条经验，才能知道 state_dim / action_dim）
        self.states = None          # shape: (max_size, state_dim)
        self.actions = None         # shape: (max_size, action_dim)
        self.rewards = None         # shape: (max_size, 1)
        self.dones = None           # shape: (max_size, 1)
        self.next_states = None     # shape: (max_size, state_dim)
    
    def _lazy_init(self, example_exp):
        """根据第一条经验的形状初始化底层 NumPy 存储（字段分离、连续内存）"""
        state_example, action_example, _, _, next_state_example = example_exp
        
        # 将示例转换为 NumPy 数组以获取精确维度，同时统一为指定精度
        state_example = np.asarray(state_example, dtype=self.dtype)
        next_state_example = np.asarray(next_state_example, dtype=self.dtype)
        action_example = np.asarray(action_example, dtype=self.dtype)
        
        state_dim = state_example.shape[0]
        next_state_dim = next_state_example.shape[0]
        
        # 安全性校验：当前设计默认 state 和 next_state 维度一致
        if state_dim != next_state_dim:
            raise ValueError(
                f"LocalReplayBuffer 期望 state_dim == next_state_dim，但得到 "
                f"{state_dim} vs {next_state_dim}"
            )
        
        action_dim = action_example.shape[0]
        
        self.states = np.zeros((self.max_size, state_dim), dtype=self.dtype)
        self.next_states = np.zeros((self.max_size, state_dim), dtype=self.dtype)
        self.actions = np.zeros((self.max_size, action_dim), dtype=self.dtype)
        # reward / done 也用同一 dtype，便于后续统一转换到 torch
        self.rewards = np.zeros((self.max_size, 1), dtype=self.dtype)
        self.dones = np.zeros((self.max_size, 1), dtype=self.dtype)
        
        self.initialized = True
    
    def add_batch(self, experiences):
        """追加一批经验到本地缓冲区，超过容量则循环覆盖最旧数据（环形缓冲区）
        
        experiences: list[ (state, action, reward, done, next_state) ]
        - state / next_state: 序列或数组（一维），长度 ~= 1275 或包含历史后的长度
        - action: 序列或数组
        - reward: 标量
        - done: 标量（0/1 或 bool）
        """
        if not experiences:
            return
        
        # 说明：
        # 1. NaN/Inf 检查已经在采集进程（env进程）中完成，
        #    这里再做一次会造成明显的CPU开销，且重复意义不大，故在本地缓冲区去掉重复检查。
        # 2. 这里对整个 batch 做向量化写入，避免逐样本 Python for 循环，显著降低
        #    “拉取到本地 buffer” 阶段的时间开销。

        # 延迟初始化底层数组（只有第一次有数据时才初始化）
        if not self.initialized:
            self._lazy_init(experiences[0])
        
        # -------------------------
        # 向量化写入：一次性构造批量 NumPy 数组并写入环形缓冲区
        # -------------------------
        batch_size = len(experiences)

        # 将 batch 拆分为各个字段的列表
        states_list = [np.asarray(exp[0], dtype=self.dtype) for exp in experiences]
        actions_list = [np.asarray(exp[1], dtype=self.dtype) for exp in experiences]
        rewards_list = [float(exp[2]) for exp in experiences]
        dones_list = [float(exp[3]) for exp in experiences]
        next_states_list = [np.asarray(exp[4], dtype=self.dtype) for exp in experiences]

        # 叠成批量数组
        states_arr = np.stack(states_list, axis=0)
        actions_arr = np.stack(actions_list, axis=0)
        rewards_arr = np.asarray(rewards_list, dtype=self.dtype).reshape(-1, 1)
        dones_arr = np.asarray(dones_list, dtype=self.dtype).reshape(-1, 1)
        next_states_arr = np.stack(next_states_list, axis=0)

        # 环形写入逻辑：最多拆成两段写入，避免逐样本循环
        start = self.write_pos
        end = start + batch_size

        if end <= self.max_size:
            # 不发生环绕，直接写入一段
            idx_slice = slice(start, end)
            self.states[idx_slice] = states_arr
            self.next_states[idx_slice] = next_states_arr
            self.actions[idx_slice] = actions_arr
            self.rewards[idx_slice] = rewards_arr
            self.dones[idx_slice] = dones_arr
        else:
            # 发生环绕，拆成两段写入
            first_len = self.max_size - start
            second_len = end - self.max_size

            # 第一段：[start, max_size)
            first_slice = slice(start, self.max_size)
            self.states[first_slice] = states_arr[:first_len]
            self.next_states[first_slice] = next_states_arr[:first_len]
            self.actions[first_slice] = actions_arr[:first_len]
            self.rewards[first_slice] = rewards_arr[:first_len]
            self.dones[first_slice] = dones_arr[:first_len]

            # 第二段：[0, end - max_size)
            second_slice = slice(0, second_len)
            self.states[second_slice] = states_arr[first_len:]
            self.next_states[second_slice] = next_states_arr[first_len:]
            self.actions[second_slice] = actions_arr[first_len:]
            self.rewards[second_slice] = rewards_arr[first_len:]
            self.dones[second_slice] = dones_arr[first_len:]

        # 更新写指针与有效样本计数
        self.write_pos = end % self.max_size
        self.count = min(self.count + batch_size, self.max_size)
    
    def sample_batch(self, batch_size):
        """从本地缓冲区采样一个批次；不足 batch_size 时使用所有可用样本
        支持分层采样：batch的recent_batch_ratio比例来自最新的recent_buffer_ratio部分
        
        返回：
            states:      np.ndarray, shape (B, state_dim),  dtype float32
            actions:     np.ndarray, shape (B, action_dim), dtype float32
            rewards:     np.ndarray, shape (B, 1),          dtype float32
            dones:       np.ndarray, shape (B, 1),          dtype float32
            next_states: np.ndarray, shape (B, state_dim),  dtype float32
        """
        if self.count == 0 or not self.initialized:
            return None
        
        buf_len = self.count
        # 实际 batch 大小：不足时用全部样本
        actual_batch_size = min(buf_len, batch_size)
        
        if buf_len <= batch_size:
            # 无放回使用全部样本
            indices = np.arange(buf_len, dtype=np.int64)
        else:
            # 分层采样：batch的recent_batch_ratio比例来自最新的recent_buffer_ratio部分
            # 计算最新数据的范围
            recent_size = max(1, int(buf_len * self.recent_buffer_ratio))
            old_size = buf_len - recent_size
            
            # 计算从两部分采样的数量
            recent_batch_size = max(1, int(actual_batch_size * self.recent_batch_ratio))
            old_batch_size = actual_batch_size - recent_batch_size
            
            # 获取最新数据的逻辑索引（在环形缓冲区中的实际位置）
            if self.count < self.max_size:
                # 缓冲区未满，数据在[0, count)范围内，最新数据在右侧
                recent_start_idx = buf_len - recent_size
                recent_indices_logical = np.arange(recent_start_idx, buf_len, dtype=np.int64)
                old_indices_logical = np.arange(0, recent_start_idx, dtype=np.int64)
            else:
                # 缓冲区已满，需要处理环形
                # 最新数据在write_pos往前推的位置
                # 最旧数据在write_pos位置（下一个要覆盖的位置）
                recent_start_pos = (self.write_pos - recent_size) % self.max_size
                if recent_start_pos < self.write_pos:
                    # 不跨越边界：最新数据在[recent_start_pos, write_pos)
                    recent_indices_logical = np.arange(recent_start_pos, self.write_pos, dtype=np.int64)
                    # 旧数据是[write_pos, max_size)和[0, recent_start_pos)
                    if recent_start_pos > 0:
                        part1 = np.arange(self.write_pos, self.max_size, dtype=np.int64)
                        part2 = np.arange(0, recent_start_pos, dtype=np.int64)
                        old_indices_logical = np.concatenate([part1, part2])
                    else:
                        # recent_start_pos == 0，旧数据就是[write_pos, max_size)
                        old_indices_logical = np.arange(self.write_pos, self.max_size, dtype=np.int64)
                else:
                    # 跨越边界：最新数据跨越了0位置
                    # 最新数据是[recent_start_pos, max_size)和[0, write_pos)
                    part1 = np.arange(recent_start_pos, self.max_size, dtype=np.int64)
                    part2 = np.arange(0, self.write_pos, dtype=np.int64)
                    recent_indices_logical = np.concatenate([part1, part2])
                    # 旧数据是[write_pos, recent_start_pos)
                    old_indices_logical = np.arange(self.write_pos, recent_start_pos, dtype=np.int64)
            
            # 从最新部分采样
            if len(recent_indices_logical) >= recent_batch_size:
                recent_selected_logical = np.random.choice(recent_indices_logical, recent_batch_size, replace=False)
            else:
                recent_selected_logical = recent_indices_logical
            
            # 从旧数据部分采样
            if len(old_indices_logical) >= old_batch_size:
                old_selected_logical = np.random.choice(old_indices_logical, old_batch_size, replace=False)
            else:
                old_selected_logical = old_indices_logical
            
            # 合并两部分索引（逻辑索引就是实际索引，因为使用NumPy数组）
            indices = np.concatenate([recent_selected_logical, old_selected_logical])
            # 随机打乱顺序
            np.random.shuffle(indices)
        
        # 关键优化：一次性在 NumPy 层完成索引，避免 Python for 循环
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]
        next_states = self.next_states[indices]
        
        return states, actions, rewards, dones, next_states
    
    def size(self):
        return self.count


    


class GlobalStatistics:
    """全局统计信息管理器"""
    
    def __init__(self, window_size=100):
        self.total_episodes = mp.Value('i', 0)  # 全局Episode计数器（线程安全）
        self.goal_count = mp.Value('i', 0)      # 目标到达计数（多进程共享）
        self.collision_count = mp.Value('i', 0) # 碰撞计数（多进程共享）
        self.total_reward = mp.Value('d', 0.0)  # 总奖励（多进程共享）
        self.lock = mp.Lock()
        
        # 滑动窗口统计
        self.window_size = window_size
        self.recent_episodes = mp.Manager().list()  # 存储最近X个episode的数据
    
    def add_episode_result(self, goal, collision, timeout, reward, target_dist=None):
        """添加一个episode的结果并返回episode编号
        
        Args:
            goal: 是否到达目标
            collision: 是否碰撞
            timeout: 是否超时
            reward: 奖励值
            target_dist: 期望的目标距离（ros_env.target_dist），即生成该episode时设定的目标距离上限
                        注意：这是期望值，不是实际生成的终点距离（实际距离有随机性）
        """
        with self.lock:
            # 递增episode计数并获取编号
            self.total_episodes.value += 1
            episode_number = self.total_episodes.value
            
            # 更新统计信息
            self.total_reward.value += reward
            if goal:
                self.goal_count.value += 1
            elif collision:
                self.collision_count.value += 1
            # timeout_count可以通过计算得到，不需要单独存储
            
            # 添加到滑动窗口
            episode_data = {
                'goal': goal,
                'collision': collision,
                'timeout': timeout,
                'reward': reward,
                'target_dist': target_dist  # 添加期望的目标距离
            }
            self.recent_episodes.append(episode_data)
            
            # 保持窗口大小
            if len(self.recent_episodes) > self.window_size:
                self.recent_episodes.pop(0)
            
            return episode_number
    
    def get_statistics(self, use_window=True):
        """获取统计信息
        
        Args:
            use_window: 是否使用滑动窗口统计
        """
        with self.lock:
            total_episodes_value = self.total_episodes.value
            if total_episodes_value == 0:
                return {
                    'total_episodes': 0,
                    'goal_rate': 0.0,
                    'collision_rate': 0.0,
                    'timeout_rate': 0.0,
                    'avg_reward': 0.0,
                    'window_size': 0,
                }
            
            if use_window and len(self.recent_episodes) > 0:
                # 使用滑动窗口统计
                window_episodes = len(self.recent_episodes)
                window_goals = sum(1 for ep in self.recent_episodes if ep['goal'])
                window_collisions = sum(1 for ep in self.recent_episodes if ep['collision'])
                window_timeouts = sum(1 for ep in self.recent_episodes if ep['timeout'])
                window_rewards = sum(ep['reward'] for ep in self.recent_episodes)
                
                goal_rate = window_goals / window_episodes
                collision_rate = window_collisions / window_episodes
                timeout_rate = window_timeouts / window_episodes
                avg_reward = window_rewards / window_episodes
                
                return {
                    'total_episodes': total_episodes_value,
                    'goal_rate': goal_rate,
                    'collision_rate': collision_rate,
                    'timeout_rate': timeout_rate,
                    'avg_reward': avg_reward,
                    'window_size': window_episodes,
                }
            else:
                # 使用全部历史统计
                goal_rate = self.goal_count.value / total_episodes_value
                collision_rate = self.collision_count.value / total_episodes_value
                
                # timeout_count通过计算得到：总episode数 - goal_count - collision_count
                timeout_count = total_episodes_value - self.goal_count.value - self.collision_count.value
                timeout_rate = timeout_count / total_episodes_value
                avg_reward = self.total_reward.value / total_episodes_value
                
                # 调试信息：检查计数是否完整
                total_counted = self.goal_count.value + self.collision_count.value + timeout_count
                print(f"调试统计: 总episode={total_episodes_value}, goal={self.goal_count.value}, collision={self.collision_count.value}, timeout={timeout_count}, 合计={total_counted}")
                
                return {
                    'total_episodes': total_episodes_value,
                    'goal_rate': goal_rate,
                    'collision_rate': collision_rate,
                    'timeout_rate': timeout_rate,
                    'avg_reward': avg_reward,
                    'window_size': total_episodes_value,
                }
    
    def reset(self):
        """重置统计信息"""
        with self.lock:
            self.total_episodes.value = 0
            self.goal_count.value = 0
            self.collision_count.value = 0
            self.total_reward.value = 0.0
            self.recent_episodes[:] = []  # 清空滑动窗口


class SharedModelManager:
    """共享模型管理器 - 使用共享字典和锁实现真正的跨进程模型共享"""
    
    def __init__(self, initial_model, shared_model_dict, shared_lock, training_count_ref=None, critic_loss_ref=None, recent_losses_ref=None, is_main_process=True, shared_temp_dir=None):
        self.model = initial_model  # 主进程中的模型实例
        self.shared_model_dict = shared_model_dict  # 共享模型权重字典
        self.shared_lock = shared_lock  # 共享锁
        self.training_count_ref = training_count_ref  # 共享训练次数计数器引用
        self.critic_loss_ref = critic_loss_ref  # 共享critic损失引用
        self.recent_losses_ref = recent_losses_ref  # 共享最近损失列表引用
        
        # 使用传入的共享临时文件目录
        if shared_temp_dir:
            self.temp_dir = shared_temp_dir
        else:
            raise ValueError("shared_temp_dir must be provided")
        
        self.model_file_path = os.path.join(self.temp_dir, "shared_model.pth")
        
        # 只有主进程才在初始化时保存模型到临时文件
        if is_main_process:
            self.update_dict_from_model()
    
    def _convert_to_serializable(self, obj):
        """递归转换对象为可序列化格式"""
        if hasattr(obj, 'items'):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, torch.Tensor):
            # 使用detach()分离梯度信息，然后转换为numpy
            return obj.detach().cpu().numpy()
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        else:
            # 其他类型，尝试转换为字符串
            return str(obj)
    
    def get_model_state_dict_for_inference(self):
        """获取用于推理的模型状态字典（子进程使用）"""
        if not os.path.exists(self.model_file_path):
            return None
        
        # 获取目标设备（数据收集进程的设备）
        # 使用更安全的方法获取设备
        try:
            if hasattr(self.model, 'device'):
                target_device = self.model.device
            elif hasattr(self.model, 'parameters'):
                target_device = next(self.model.parameters()).device
            else:
                # 如果无法获取设备，使用CPU
                target_device = torch.device('cpu')
        except Exception:
            target_device = torch.device('cpu')
        
        # 从文件加载模型状态字典到目标设备（GPU或CPU）
        # 添加异常处理，防止文件损坏或权限问题导致进程崩溃
        try:
            device_model_state = torch.load(self.model_file_path, map_location=target_device, weights_only=True)
            # 验证加载的数据是否为字典类型（state_dict应该是字典）
            if not isinstance(device_model_state, dict):
                print(f"[ERROR] 警告: 从文件 {self.model_file_path} 加载的数据不是字典类型，类型: {type(device_model_state)}")
                return None
            return device_model_state
        except Exception as e:
            print(f"[ERROR] 错误: 从文件 {self.model_file_path} 加载模型权重失败: {e}")
            return None
    
    def _convert_from_serializable(self, obj):
        """递归转换对象从可序列化格式"""
        if hasattr(obj, 'items'):#检查对象是否具有items方法
            return {key: self._convert_from_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, np.ndarray):
            return torch.tensor(obj)
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        else:
            # 其他类型，直接返回
            return obj
    
    def update_dict_from_model(self):
        """训练完成后更新共享字典（只同步actor模型权重）"""
        with self.shared_lock:
            # 只获取actor权重，节省显存和传输时间
            model_state = self.model.state_dict(actor_only=True)
            # 先写入临时文件，再原子替换，避免子进程读取到不完整的模型文件
            tmp_path = self.model_file_path + ".tmp"
            torch.save(model_state, tmp_path)
            os.replace(tmp_path, self.model_file_path)
            # 在共享字典中保存文件路径信息（用于调试）
            self.shared_model_dict['model_file_path'] = self.model_file_path
            self.shared_model_dict['temp_dir'] = self.temp_dir
    
    def get_training_count(self):
        """获取当前训练次数"""
        if self.training_count_ref is not None:
            return self.training_count_ref.value
        return 0
    
    def get_critic_loss(self):
        """获取当前critic损失"""
        if self.critic_loss_ref is not None:
            return self.critic_loss_ref.value
        return float('inf')  # 如果没有损失记录，返回无穷大
    
    def update_critic_loss(self, loss):
        """更新critic损失"""
        if self.critic_loss_ref is not None:
            self.critic_loss_ref.value = loss
    
    def get_average_critic_loss(self, window_size):
        """获取前window_size次训练的平均critic损失"""
        if self.recent_losses_ref is not None and len(self.recent_losses_ref) > 0:
            recent_losses = list(self.recent_losses_ref)
            if len(recent_losses) >= window_size:
                return sum(recent_losses[-window_size:]) / window_size
            else:
                return sum(recent_losses) / len(recent_losses)
        return float('inf')  # 如果没有损失记录，返回无穷大
    
    def add_critic_loss(self, loss):
        """添加critic损失到最近损失列表"""
        if self.recent_losses_ref is not None:
            self.recent_losses_ref.append(loss)
            # 保持列表大小不超过100（避免内存无限增长）
            if len(self.recent_losses_ref) > 100:
                self.recent_losses_ref.pop(0)
    
    def cleanup_temp_files(self):
        """清理临时文件"""
        import shutil
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"已清理临时文件目录: {self.temp_dir}")
        except Exception as e:
            print(f"清理临时文件时出错: {e}")
    


def collect_episode_data(env_id, shared_model_dict, model_lock, experience_queue, total_added_step, global_stats, config, init_complete_counter, total_episodes_counter, training_count_ref, critic_loss_ref, recent_losses_ref, avg_loss_window_size, phase_ref, eval_target_ref, eval_collected_lock, eval_collected_ref, current_buffer_size_ref, check_best_model_ref, round_done_counter, current_round_ref, env_target_dist_list=None):
    """单个环境的数据收集进程 - 真正的并行版本"""
    # 公共配置中的每环境日志根目录与时间戳
    env_logs_dir_cfg = config.get('env_logs_dir')
    log_timestamp = str(config.get('log_timestamp', '') or '')
    from pathlib import Path
    if env_logs_dir_cfg and log_timestamp:
        # 允许 env_logs_dir 是字符串或 Path
        env_logs_dir = Path(env_logs_dir_cfg)
        # 为当前环境构造独立日志目录，例如：.../env_logs/env_7/
        env_log_dir = env_logs_dir / f"env_{env_id}"
        # 具体文件名示例：collect_log_7_20260128_154423.log
        collect_log_path = str(env_log_dir / f"collect_log_{env_id}_{log_timestamp}.log")
        env_log_path = str(env_log_dir / f"env_log_{env_id}_{log_timestamp}.log")
        reward_log_path = str(env_log_dir / f"reward_log_{env_id}_{log_timestamp}.log")
        nodes_log_path = str(env_log_dir / f"nodes_log_{env_id}_{log_timestamp}.log")
    else:
        # 兼容旧配置：退回到单一日志文件（不再仅限 env_id=0）
        collect_log_path = (config.get('collect_log_path') or '').strip()
        env_log_path = (config.get('env_log_path') or '').strip()
        reward_log_path = (config.get('reward_log_path') or '').strip()
        nodes_log_path = (config.get('nodes_log_path') or '').strip()

    clog = CollectLogger(collect_log_path) if collect_log_path else None
    env_logger = EnvLogger(env_log_path) if env_log_path else None
    reward_logger = RewardLogger(reward_log_path) if reward_log_path else None
    nodes_logger = NodesLogger(nodes_log_path) if nodes_log_path else None
    try:
        print(f"环境 {env_id} 开始初始化...")
        
        # 设置正确的ROS域ID，确保与对应的Gazebo环境通信
        # 从配置文件中读取起始ROS_DOMAIN_ID，如果没有则默认为1（向后兼容）
        start_ros_domain_id = config.get('start_ros_domain_id', 1)
        ros_domain_id = start_ros_domain_id + env_id  # 环境0使用start_ros_domain_id，环境1使用start_ros_domain_id+1...
        os.environ['ROS_DOMAIN_ID'] = str(ros_domain_id)
        print(f"环境 {env_id} 设置ROS_DOMAIN_ID={ros_domain_id}")
        if clog:
            clog.log(env_id, "set_ros_domain_id", None, {"ROS_DOMAIN_ID": ros_domain_id})
        if clog:
            clog.log(env_id, "collect_start", {"env_id": env_id, "ros_domain_id": ros_domain_id, "start_ros_domain_id": start_ros_domain_id}, "started")
        
        print(f"环境 {env_id} 开始初始化ROS环境")
        # 调试：打印传感器日志开关配置
        print(f"环境 {env_id} 调试: config 中是否包含 'sensor_log_enable': {'sensor_log_enable' in config}")
        print(f"环境 {env_id} 调试: config 中所有键: {list(config.keys())[:20]}...")  # 只打印前20个键
        sensor_log_config = config.get('sensor_log_enable', {})
        print(f"环境 {env_id} sensor_log_enable 配置类型: {type(sensor_log_config)}, 内容: {sensor_log_config}")
        if isinstance(sensor_log_config, dict) and len(sensor_log_config) == 0:
            print(f"环境 {env_id} [ERROR] [ERROR] 警告: sensor_log_enable 是空字典！检查配置文件格式。")
        
        # 辅助函数：将值转换为布尔值（处理字符串形式的 true/false）
        def to_bool(val, default=False):
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in ('true', '1', 'yes', 'on')
            return bool(val) if val is not None else default
        
        scan_enable_log_val = to_bool(sensor_log_config.get('scan_enable_log', False)) if isinstance(sensor_log_config, dict) else False
        odom_enable_log_val = to_bool(sensor_log_config.get('odom_enable_log', False)) if isinstance(sensor_log_config, dict) else False
        imu_enable_log_val = to_bool(sensor_log_config.get('imu_enable_log', False)) if isinstance(sensor_log_config, dict) else False
        print(f"环境 {env_id} 传感器日志开关配置: scan={scan_enable_log_val} (type={type(scan_enable_log_val)}), odom={odom_enable_log_val} (type={type(odom_enable_log_val)}), imu={imu_enable_log_val} (type={type(imu_enable_log_val)})")
        # 初始化ROS环境（奖励/惩罚参数从 train.yaml 配置读取）
        ros_env = ROS_env(
            env_id=env_id,  # 传递正确的环境ID
            env_logger=env_logger,
            reward_logger=reward_logger,
            nodes_logger=nodes_logger,
            max_velocity=config['max_velocity'],
            init_target_distance=config['init_target_distance'],
            target_dist_increase=config['target_dist_increase'],
            max_target_dist=config['max_target_dist'],
            target_reached_delta=config['target_reached_delta'],
            collision_delta=config['collision_delta'],
            neglect_angle=config['neglect_angle'],
            scan_range=config['scan_range'],
            localization_noise_stddev=config.get('localization_noise_stddev', 0.0),
            # 传感器频率限制参数
            scan_max_freq=config.get('sensor_freq_limit', {}).get('scan_max_freq', 0.0),
            odom_max_freq=config.get('sensor_freq_limit', {}).get('odom_max_freq', 0.0),
            imu_max_freq=config.get('sensor_freq_limit', {}).get('imu_max_freq', 0.0),
            # 传感器日志开关（从 train.yaml 的 sensor_log_enable 配置读取）
            scan_enable_log=scan_enable_log_val,
            odom_enable_log=odom_enable_log_val,
            imu_enable_log=imu_enable_log_val,
            world_size=config['world_size'],
            goals_per_map=config.get('goals_per_map', 1),
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
            enable_step_penalty=config.get('enable_step_penalty', False),
            enable_target_distance_penalty=config.get('enable_target_distance_penalty', False),
            enable_progress_reward=config.get('enable_progress_reward', False),
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
            # 时间步惩罚参数
            step_penalty_base=config.get('step_penalty_base', 0.0),
            # 震荡惩罚参数
            linear_acceleration_oscillation_penalty_base=config.get('linear_acceleration_oscillation_penalty_base', -1.0),
            yawrate_oscillation_penalty_base=config.get('yawrate_oscillation_penalty_base', -1.0),
            # 进度奖惩参数
            progress_reward_base=config.get('progress_reward_base', 1.0),
            # 时间控制
            sim_time=config.get('sim_time', 0.1),
            step_sleep_time=config.get('step_sleep_time', 0.1),
            reset_step_count=config.get('reset_step_count', 3),
            # 连通区域选择偏好
            region_select_bias=config.get('region_select_bias', 1.0),
            # 奖励归一化参数
            reward_scale=config.get('reward_scale', 1.0),  # 奖励缩放因子，用于控制每步整体奖励大小
        )
        print(f"环境 {env_id} ROS环境初始化完成")
        if clog:
            clog.log(env_id, "init_ros_env", {"env_id": env_id, "ros_domain_id": ros_domain_id}, "created")
        
        # 在spawn模式下，需要重新创建模型实例
        # 确保数据收集进程使用与主进程相同的GPU设备
        # 注意：如果通过环境变量 CUDA_VISIBLE_DEVICES 设置了 GPU，PyTorch 视角下 GPU 索引从 0 开始
        gpu_id = config.get('gpu_id', 0)
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if cuda_visible_devices and torch.cuda.is_available():
            # 如果设置了 CUDA_VISIBLE_DEVICES，PyTorch 只能看到指定的 GPU，索引总是从 0 开始
            device = torch.device("cuda")
        else:
            device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        print(f"环境 {env_id} 使用设备: {device}")
        
        # 计算实际使用的state_dim，确保与历史state拼接一致
        state_history_steps = config.get('state_history_steps', 0)
        # 单步 state 维度（base_state_dim）不再在 train.yaml 中手动配置，改为由 bin_num + 非激光特征数动态计算
        bin_num = config.get('bin_num', 72)
        non_lidar_dim = 7  # distance,cos,sin(3) + last_action(2) + current_v,current_w(2)
        base_state_dim = int(bin_num) + non_lidar_dim
        state_dim_effective = base_state_dim * (1 + state_history_steps) if state_history_steps > 0 else base_state_dim

        # 创建本地SAC模型实例（子进程版本）
        # 注意：数据收集进程只需要actor模型进行推理，不需要critic和target_critic，以节省显存
        # 数据收集进程不需要加载已有模型，因为会立即同步主进程的模型权重
        local_model = SAC(
            state_dim=state_dim_effective,
            action_dim=config['action_dim'],
            max_action=config['max_action'],
            device=device,
            discount=config.get('discount_factor', 0.99),  # 传递折扣因子
            actor_update_frequency=config.get('actor_update_frequency', 1),
            critic_target_update_frequency=config.get('critic_target_update_frequency', 4),
            hidden_layers=config.get('hidden_layers', [1024, 512]),
            save_every=0,  # 不自动保存
            load_model=False,  # 数据收集进程不加载模型，使用共享模型状态
            action_noise_std=config.get('action_noise_std', 0.2),
            base_state_dim=base_state_dim,  # 确保prepare_state使用单步状态长度
            bin_num=config.get('bin_num', 72),  # 激光scan分桶数量
            actor_only=True,  # 只创建actor模型，不创建critic和target_critic，节省显存
            actor_grad_clip_value=config.get('actor_grad_clip_value', 0.0),  # 传递Actor梯度裁剪值（虽然actor_only模式下不会训练，但保持参数一致性）
            critic_grad_clip_value=config.get('critic_grad_clip_value', 0.0),  # 传递Critic梯度裁剪值（虽然actor_only模式下不会训练，但保持参数一致性）
        )
        
        # 验证：确认只创建了actor模型，没有创建critic
        has_actor = hasattr(local_model, 'actor') and local_model.actor is not None
        has_critic = hasattr(local_model, 'critic') and local_model.critic is not None
        has_critic_target = hasattr(local_model, 'critic_target') and local_model.critic_target is not None
        if clog:
            clog.log(env_id, "create_local_model", {"state_dim": state_dim_effective, "action_dim": config["action_dim"], "max_action": config["max_action"], "actor_only": True, "bin_num": config.get("bin_num", 72), "base_state_dim": base_state_dim}, {"created": True, "has_actor": has_actor, "has_critic": has_critic})
        print(f"环境 {env_id} 本地SAC副本创建完成 - Actor: {has_actor}, Critic: {has_critic}, CriticTarget: {has_critic_target}")
        if has_critic or has_critic_target:
            print(f"[ERROR] 警告: 环境 {env_id} 在actor_only模式下仍然创建了critic模型！这不应该发生。")
        if not has_actor:
            print(f"[ERROR] 环境 {env_id} 未能创建actor模型！")
        
        # 创建共享模型管理器实例（子进程版本）
        # 从共享字典获取临时目录路径
        shared_temp_dir = shared_model_dict.get('shared_temp_dir', None)
        model_manager = SharedModelManager(local_model, shared_model_dict, model_lock, training_count_ref, critic_loss_ref, recent_losses_ref, is_main_process=False, shared_temp_dir=shared_temp_dir)
        if clog:
            clog.log(env_id, "create_model_manager", {"shared_temp_dir": shared_temp_dir}, "created")
        
        # 等待主进程创建模型文件并同步权重
        print(f"环境 {env_id} 等待主进程模型文件...")
        if clog:
            clog.log(env_id, "model_wait_start", {"max_wait_attempts": 50}, "waiting")
        max_wait_attempts = 50  # 最多等待50次，每次0.1秒
        wait_attempt = 0
        
        while wait_attempt < max_wait_attempts:
            latest_model_state = model_manager.get_model_state_dict_for_inference()
            if latest_model_state and len(latest_model_state) > 0:
                print(f"环境 {env_id} 获取到模型权重，键: {list(latest_model_state.keys())}")
                
                # 加载模型权重到本地模型
                local_model.load_state_dict(latest_model_state)
                print(f"环境 {env_id} 模型权重同步完成")
                if clog:
                    clog.log(env_id, "model_sync", None, {"keys": list(latest_model_state.keys()), "ok": True})
                
                # 验证actor网络权重一致性
                verify_ok = verify_actor_weight_consistency(local_model, latest_model_state, env_id, config)
                if clog:
                    clog.log(env_id, "model_verify", {"sync_type": "initial"}, {"ok": verify_ok})
                
                break
            time.sleep(0.1)  # 等待0.1秒
            wait_attempt += 1
        
        if wait_attempt >= max_wait_attempts:
            print(f"环境 {env_id} [ERROR] [ERROR] 警告: 未能获取到有效的模型权重，使用随机初始化的模型")
            if clog:
                clog.log(env_id, "model_sync", None, {"ok": False, "msg": "timeout"})
        
        
        # 增加初始化完成计数器
        with init_complete_counter.get_lock():
            init_complete_counter.value += 1
            if clog:
                clog.log(env_id, "init_complete", {"env_id": env_id}, {"counter": int(init_complete_counter.value)})
            print(f"环境 {env_id} 初始化完成，开始持续数据收集...")
        
        total_steps = 0
        discount_factor = config.get('discount_factor', 0.99)  # 折扣因子（gamma），从配置文件 train.yaml 读取，统一用于计算总回报和所有 Reward Detail 分项的折扣回报

        # 历史state设置（长度为 state_history_steps + 1，用于存储 [s_{t-k}, ..., s_t]）
        # 即使 state_history_steps 为 0，也保持长度为 1，此时等价于只使用当前 state
        state_history = deque(maxlen=state_history_steps + 1)
        
        eval_model_synced = False  # 评估阶段只同步一次模型（每次进入评估阶段会重置）
        last_is_eval = False

        global_episode_number = -1  # 安全初始化，防止异常时未定义
        local_episode_counter = 0  # 局部episode计数器，用于在step()日志中标记episode编号
        # 关键约束：每个环境在同一轮（current_round_ref）内最多“提交/计入”一次训练episode。
        # 注意：被丢弃的episode（NaN/Inf、过滤规则、phase切换边界丢弃等）不会计入 round_done_counter，
        # 因此不会触发该闸门，仍允许在同一轮内重试，避免单环境因为偶发异常导致整轮卡住。
        last_committed_round = None

        # 用于跟踪上一次 phase_skip 的状态，只在状态变化时记录日志
        last_phase_skip_state = None

        while True:
            current_phase = phase_ref.value
            # STOP: 直接退出
            if current_phase == PHASE_STOP:
                if clog:
                    clog.log(env_id, "collect_exit", {"reason": "stop", "phase": current_phase}, "exiting")
                break
            # PAUSE 或 EVAL_DRAIN: 等待主进程切换phase，不启动新episode
            if current_phase == PHASE_PAUSE or current_phase == PHASE_EVAL_DRAIN:
                current_skip_state = {"phase": current_phase, "reason": "pause_or_eval_drain"}
                if clog and current_skip_state != last_phase_skip_state:
                    clog.log(env_id, "phase_skip", current_skip_state, None)
                    last_phase_skip_state = current_skip_state
                time.sleep(0.1)
                continue
            
            # 判断是否为评估语义（评估收集/评估收尾都视为评估）
            is_eval = (current_phase == PHASE_EVAL_COLLECT or current_phase == PHASE_EVAL_DRAIN)

            # 训练阶段：根据phase决定是否允许开始新episode
            if not is_eval:
                # TRAIN_DRAIN: 不允许开始新episode，等待当前episode结束
                if current_phase == PHASE_TRAIN_DRAIN:
                    current_skip_state = {"phase": current_phase, "reason": "train_drain"}
                    if clog and current_skip_state != last_phase_skip_state:
                        clog.log(env_id, "phase_skip", current_skip_state, None)
                        last_phase_skip_state = current_skip_state
                    time.sleep(0.1)
                    continue
                # TRAIN_COLLECT: 允许开始新episode
                elif current_phase == PHASE_TRAIN_COLLECT:
                    local_round = current_round_ref.value
                    # 每轮每环境最多提交一次：如果本环境已经在该轮提交过一个有效训练episode，则等待进入下一轮
                    if last_committed_round is not None and local_round == last_committed_round:
                        current_skip_state = {"phase": current_phase, "reason": "round_committed", "local_round": local_round, "last_committed_round": last_committed_round}
                        if clog and current_skip_state != last_phase_skip_state:
                            clog.log(env_id, "phase_skip", current_skip_state, None)
                            last_phase_skip_state = current_skip_state
                        time.sleep(0.1)
                        continue
                else:
                    # 未知phase，等待
                    current_skip_state = {"phase": current_phase, "reason": "unknown_phase"}
                    if clog and current_skip_state != last_phase_skip_state:
                        clog.log(env_id, "phase_skip", current_skip_state, None)
                        last_phase_skip_state = current_skip_state
                    time.sleep(0.1)
                    continue
            else:
                local_round = -1

            # 当不再处于 phase_skip 状态时，重置状态跟踪变量
            last_phase_skip_state = None

            # 阶段切换处理：每次进入评估阶段时，允许同步一次最新模型
            if is_eval and not last_is_eval:
                eval_model_synced = False
            last_is_eval = is_eval

            if is_eval:
                # 评估阶段固定使用最大目标距离，避免沿用训练阶段的递增值
                ros_env.target_dist = config['max_target_dist']
                if clog:
                    clog.log(env_id, "eval_target_dist_set", {"max_target_dist": config["max_target_dist"]}, {"target_dist": ros_env.target_dist, "eval_target": eval_target_ref.value})
                if env_id == 0:  # 只在环境0打印，避免刷屏
                    print(f"[环境 {env_id}] 进入评估阶段，target_dist={ros_env.target_dist:.2f}, eval_target={eval_target_ref.value}")

            # 评估阶段：首次进入时同步模型一次，之后不再更新
            if is_eval and not eval_model_synced:
                latest_model_state = model_manager.get_model_state_dict_for_inference()
                if latest_model_state:
                    local_model.load_state_dict(latest_model_state)
                    if clog:
                        clog.log(env_id, "model_sync_eval", None, {"ok": True})
                    print(f"环境 {env_id} 评估阶段模型同步完成")
                    # 验证actor网络权重一致性
                    verify_ok = verify_actor_weight_consistency(local_model, latest_model_state, env_id, config)
                    if clog:
                        clog.log(env_id, "model_verify", {"sync_type": "eval"}, {"ok": verify_ok})
                else:
                    if clog:
                        clog.log(env_id, "model_sync_eval", None, {"ok": False})
                eval_model_synced = True

            try:
                # 训练阶段的模型同步策略；评估阶段不再更新模型
                if not is_eval:
                    latest_model_state = model_manager.get_model_state_dict_for_inference()
                    if latest_model_state:
                        local_model.load_state_dict(latest_model_state)
                        if clog:
                            clog.log(env_id, "model_sync_episode", None, {"ok": True})
                        # 验证actor网络权重一致性
                        verify_ok = verify_actor_weight_consistency(local_model, latest_model_state, env_id, config)
                        if clog:
                            clog.log(env_id, "model_verify", {"sync_type": "episode"}, {"ok": verify_ok})
                    else:
                        print(f"环境 {env_id} [ERROR] [ERROR] 警告: 未能获取到模型权重")
                        if clog:
                            clog.log(env_id, "model_sync_episode", None, {"ok": False})
                
                # 重置环境，循环调用直到成功
                reset_success = False
                while not reset_success:
                    reset_success, latest_scan, distance, distance_raw, cos, sin, collision, goal, last_action, reward, current_v, current_w = ros_env.reset()
                    if not reset_success:
                        if clog:
                            clog.log(env_id, "reset_failed", None, {"reason": "reset returned False, retrying"})
                        # 只写入 env_log，不再额外刷综合训练日志；标记为 [ERROR]
                        if env_logger is not None:
                            env_logger.log(env_id, "[ERROR] reset()失败，重试中...")
                        continue
                local_episode_counter += 1  # 每次reset时递增局部episode计数器
                if clog:
                    clog.log(env_id, "reset", None, {"distance": distance, "cos": cos, "sin": sin, "collision": collision, "goal": goal, "reward": reward, "last_action": last_action, "current_v": current_v, "current_w": current_w})
                
                state, terminal = local_model.prepare_state(
                    latest_scan, distance, cos, sin, collision, goal, last_action, current_v, current_w
                )
                if clog:
                    clog.log(env_id, "prepare_state", {"source": "reset", "distance": distance, "cos": cos, "sin": sin, "collision": collision, "goal": goal, "last_action": last_action, "current_v": current_v, "current_w": current_w}, {"state": state, "state_len": len(state), "terminal": terminal})
                
                episode_reward = 0
                gamma_power = 1.0  # 折扣因子幂次（γ^t），用于统一计算总回报和所有 Reward Detail 分项的折扣回报
                experiences = []
                episode_discarded_due_to_nan_inf = False  # 标记是否因NaN/Inf而丢弃episode
                
                # 折扣后的奖励分项统计（用于 Reward Detail 显示，使用与总回报相同的折扣因子）
                # 直接读取 ros_python.py 中 get_reward() 记录的“本step分量”，避免用episode累计值做差分
                discounted_goal_sum = 0.0
                discounted_collision_sum = 0.0
                discounted_obs_sum = 0.0
                discounted_yawrate_sum = 0.0
                discounted_angle_sum = 0.0
                discounted_linear_sum = 0.0
                discounted_target_distance_sum = 0.0
                discounted_step_sum = 0.0
                discounted_progress_sum = 0.0
                discounted_linear_acc_osc_sum = 0.0
                discounted_yawrate_osc_sum = 0.0
                
                # 清空并用 s0 填满历史队列（state_history_steps+1 个 s0）
                state_history.clear()
                for _ in range(state_history_steps + 1):
                    state_history.append(list(state))  # 修复风险B：避免同一对象重复引用
                # 预先展开得到当前输入 x0，后续循环中复用并逐步更新
                current_state_with_history = []
                for hist_state in state_history:
                    current_state_with_history.extend(hist_state)
                
                # 最大步数策略：
                # - 默认：max_steps = max(distance * max_steps_ratio, max_steps_min)
                # - 当 max_steps_ratio == 0 时，使用固定 max_steps（共享参数）
                if config.get('max_steps_ratio', 0) == 0:
                    max_steps = int(config.get('max_steps', config.get('max_steps_min', 50)))
                else:
                    calculated_max_steps = int(distance * config['max_steps_ratio'])
                    max_steps = max(calculated_max_steps, config['max_steps_min'])
                
                # 收集一个episode的数据；当达到全局停止信号且超过最小门槛时提前结束
                min_step_threshold = max(1, min(max_steps, int(math.ceil(max_steps * config.get('min_collection_ratio', 0.25)))))
                forced_stop_flag = False
                if clog:
                    clog.log(env_id, "episode_start", {"is_eval": is_eval, "max_steps": max_steps, "min_step_threshold": min_step_threshold, "target_dist": ros_env.target_dist, "distance": distance, "state_len": len(state), "state_history_steps": state_history_steps, "discount_factor": discount_factor, "local_round": local_round}, "loop_started")
                while not terminal and ros_env.step_count < max_steps:
                    current_phase_in_loop = phase_ref.value
                    # 训练阶段：如果phase切换到TRAIN_DRAIN或已离开训练阶段
                    if not is_eval:
                        if current_phase_in_loop == PHASE_TRAIN_DRAIN or current_phase_in_loop != PHASE_TRAIN_COLLECT:
                            # 如果满足最小门槛，可以截断（如果是terminal导致的自然结束，也会正常退出循环）
                            if ros_env.step_count >= min_step_threshold:
                                forced_stop_flag = True
                                break
                            # 如果未满足最小门槛，继续运行直到满足min_threshold或发生terminal
                            # （terminal时循环会自然退出，此时ros_env.step_count可能 < min_threshold，但后续会判断terminal允许完成）
                    # 评估阶段：如果phase离开EVAL_COLLECT（例如进入EVAL_DRAIN/PAUSE等），满足最小门槛后尽快截断，
                    # 且该episode后续不会计入统计（见后面的phase判断）
                    else:
                        if current_phase_in_loop != PHASE_EVAL_COLLECT:
                            forced_stop_flag = True
                            break
                    # 1. 构造当前输入 x_t（统一从 current_state_with_history 中读取；
                    #    当 state_history_steps 为 0 时，current_state_with_history 仅包含当前 state）
                    model_action = local_model.get_action(current_state_with_history, add_noise=not is_eval)
                    if clog:
                        clog.log(env_id, "get_action", {"state": current_state_with_history, "state_len": len(current_state_with_history), "add_noise": not is_eval}, {"model_action": model_action})
                    # 手动转换动作：
                    # - 线速度：将 model_action[0] 从 [-1, 1] 重新投影到 [0, max_velocity]
                    # - 角速度：保持与 utils.transfor_action() 相同的转换方式
                    max_velocity = float(config["max_velocity"])
                    lin_velocity = (float(model_action[0]) + 1.0) * (max_velocity / 2.0)
                    lin_velocity = min(max(lin_velocity, 0.0), max_velocity)

                    max_yawrate = float(config["max_yawrate"])
                    ang_velocity = float(model_action[1]) * (max_yawrate / 180.0) * math.pi

                    ros_action = [lin_velocity, ang_velocity]
                    if clog:
                        clog.log(
                            env_id,
                            "manual_action_transform",
                            {"model_action": model_action, "max_velocity": max_velocity, "max_yawrate": max_yawrate},
                            {"lin_velocity": ros_action[0], "ang_velocity": ros_action[1]},
                        )
                    
                    # 执行动作
                    latest_scan, distance, distance_raw, cos, sin, collision, goal, reward, current_v, current_w = ros_env.step(
                        lin_velocity=ros_action[0], ang_velocity=ros_action[1]
                    )
                    if clog:
                        # 为避免语义混淆：step 的 output 里同时记录“上一动作(last_action)”与“本次执行动作(action)”
                        clog.log(
                            env_id,
                            "step",
                            {"lin_velocity": ros_action[0], "ang_velocity": ros_action[1], "episode_number": local_episode_counter, "episode_step": ros_env.step_count},
                            {
                                "distance": distance,
                                "cos": cos,
                                "sin": sin,
                                "reward": reward,
                                "collision": collision,
                                "goal": goal,
                                # step 日志中的 last_action = 上一动作（即执行本次 action 之前的 last_action）
                                "last_action": last_action,
                                # step 日志中的 action = 本次实际下发动作
                                "action": [float(ros_action[0]), float(ros_action[1])],
                                "current_v": current_v,
                                "current_w": current_w,
                            },
                        )
                                        # 检查step返回值是否包含NaN/Inf
                    ros_action_has_nan_inf = _has_nan_inf(ros_action)
                    if ros_action_has_nan_inf or not _is_valid_env_return(
                        latest_scan, distance, distance_raw, cos, sin, collision, goal, reward, current_v, current_w
                    ):
                        latest_scan_has_nan_inf = _has_nan_inf(latest_scan)
                        last_action_has_nan_inf = _has_nan_inf(last_action)
                        # 记录错误日志到控制台
                        print(
                            f"[ERROR] 环境 {env_id} step()返回值包含NaN/Inf，中断并丢弃本次episode（不占用episode编号）。"
                            f"已收集 {len(experiences)} 条样本将被丢弃。"
                            f"latest_scan包含NaN/Inf: {latest_scan_has_nan_inf}, "
                            f"distance: {distance}, distance_raw: {distance_raw}, cos: {cos}, sin: {sin}, reward: {reward}, "
                            f"collision: {collision}, goal: {goal}, current_v: {current_v}, current_w: {current_w}, "
                            f"ros_action包含NaN/Inf: {ros_action_has_nan_inf}, "
                            f"last_action包含NaN/Inf: {last_action_has_nan_inf}"
                        )
                        # 同时追加到 collect_log
                        if clog:
                            clog.log(
                                env_id,
                                "[ERROR] step_nan_inf",
                                {
                                    "episode_number": local_episode_counter,
                                    "episode_step": ros_env.step_count,
                                    "distance": distance,
                                    "cos": cos,
                                    "sin": sin,
                                    "collision": collision,
                                    "goal": goal,
                                    "reward": reward,
                                },
                                {
                                    "latest_scan_has_nan_inf": latest_scan_has_nan_inf,
                                    "ros_action_has_nan_inf": ros_action_has_nan_inf,
                                    "last_action_has_nan_inf": last_action_has_nan_inf,
                                },
                            )
                        # 标记episode因NaN/Inf被丢弃，中断循环
                        episode_discarded_due_to_nan_inf = True
                        terminal = True  # 设置terminal为True以退出内层循环
                        break
                    # 调用方维护 last_action：用于下一时刻的观测拼接
                    last_action = [float(ros_action[0]), float(ros_action[1])]
                    

                    
                    # 计算折扣回报：G_0 = r_0 + γ*r_1 + γ²*r_2 + ...（使用统一的 discount_factor）
                    episode_reward += gamma_power * reward
                    
                    # 计算各奖励分项的折扣累加值（用于 Reward Detail 显示，使用与总回报相同的 discount_factor）
                    parts = getattr(ros_env, "last_step_reward_parts", None) or {}
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
                    total_steps += 1
                    
                    # 准备下一个状态（单步 state）
                    next_state, terminal = local_model.prepare_state(
                        latest_scan, distance, cos, sin, collision, goal, last_action, current_v, current_w
                    )
                    if clog:
                        clog.log(env_id, "prepare_state", {"source": "step", "episode_step": ros_env.step_count, "distance": distance, "cos": cos, "sin": sin, "collision": collision, "goal": goal, "last_action": last_action, "current_v": current_v, "current_w": current_w}, {"state": next_state, "state_len": len(next_state), "terminal": terminal})
                    
                    # 更新历史并构造下一时刻输入 x_{t+1}
                    # 将 next_state 压入历史，形成 [s_{t-k+1}, ..., s_t, s_{t+1}]
                    state_history.append(list(next_state))  # 修复风险B：避免同一对象重复引用
                    next_state_with_history = []
                    for hist_state in state_history:
                        next_state_with_history.extend(hist_state)
                    # 存储 (x_t, action, reward, done, x_{t+1})
                    experiences.append((current_state_with_history, model_action, reward, terminal, next_state_with_history))
                    # 下一个循环直接使用 x_{t+1}
                    current_state_with_history = next_state_with_history
                    
                    # 若phase切换到DRAIN或已离开训练阶段且已超过最小门槛，则提前结束episode
                    # （这个检查已经在循环开头做了，这里保留作为冗余检查）
                                # 如果episode因NaN/Inf被丢弃，跳过后续所有处理，不占用episode编号

                if episode_discarded_due_to_nan_inf:
                    if clog:
                        clog.log(env_id, "episode_end", {"episode_steps": ros_env.step_count, "experiences_len": len(experiences)}, "discarded_step_nan_inf")
                    continue

                # 强制停止：达到全局停止信号并手动截断
                if forced_stop_flag and len(experiences) > 0:
                    # 末条transition置为非终止，避免错误的bootstrap截断
                    last_exp = experiences[-1]
                    if clog:
                        clog.log(env_id, "transition_trim", {"reason": "force_stop", "episode_steps": ros_env.step_count, "experiences_len": len(experiences), "last_done_before": last_exp[3]}, {"last_done_after": False})
                    experiences[-1] = (last_exp[0], last_exp[1], last_exp[2], False, last_exp[4])
                
                # 修复风险A：如果是因为timeout退出（步数达到上限且未发生goal/collision），将最后一个transition的done改为False
                # 注意：
                # 1. timeout时episode确实会结束（退出内层循环，开启新的episode）
                # 2. 但是done标志必须为False，这样在模型更新时会对后续回报的期望进行估计（bootstrap），而不是只使用即时回报
                # 3. 如果同时达到max_steps和goal/collision，terminal已经是True，episode_ending会正确判断为Goal/Collision，无需修复
                if ros_env.step_count >= max_steps and not terminal and len(experiences) > 0:
                    # 将最后一个transition的done标志改为False，确保训练时bootstrap下一个状态的价值估计
                    last_exp = experiences[-1]
                    if clog:
                        clog.log(env_id, "transition_trim", {"reason": "timeout", "episode_steps": ros_env.step_count, "max_steps": max_steps, "experiences_len": len(experiences), "last_done_before": last_exp[3]}, {"last_done_after": False})
                    experiences[-1] = (last_exp[0], last_exp[1], last_exp[2], False, last_exp[4])
                
                # 判断episode结束原因
                if forced_stop_flag:
                    episode_ending = "ForceStop"
                    timeout = False
                elif goal:
                    episode_ending = "Goal"
                    timeout = False
                elif collision:
                    episode_ending = "Collision"
                    timeout = False
                else:
                    episode_ending = "Timeout"
                    timeout = True
                
                # 过滤规则：步数小于10且碰撞的episode不采用
                # should_filter = (ros_env.step_count < 2 and (collision or goal))
                # 
                # if should_filter:
                #     if clog:
                #         clog.log(env_id, "episode_end", {"episode_steps": ros_env.step_count, "ending": episode_ending}, "filtered")
                #     continue

                # 若episode在阶段切换边界被截断完成（或切换后才结束），则直接丢弃：
                # - 训练阶段：
                #   * 如果phase是TRAIN_DRAIN且满足完成条件（step>=min_threshold 或 terminal），则允许完成
                #   * 其他非TRAIN_COLLECT的phase，丢弃
                # - 评估阶段：避免把"超出目标数量"的episode计入评估统计
                current_phase_after_episode = phase_ref.value
                if not is_eval:
                    # 训练阶段：TRAIN_DRAIN阶段需要特殊处理
                    if current_phase_after_episode == PHASE_TRAIN_DRAIN:
                        # 如果满足完成条件（step>=min_threshold 或 terminal），允许完成
                        if ros_env.step_count >= min_step_threshold or terminal:
                            # 允许继续，不丢弃
                            pass
                        else:
                            # step < min_threshold 且没有terminal，丢弃
                            if clog:
                                clog.log(env_id, "episode_end", {"episode_steps": ros_env.step_count, "ending": episode_ending}, "discarded_phase_drain")
                            continue
                    elif current_phase_after_episode != PHASE_TRAIN_COLLECT:
                        # 其他非TRAIN_COLLECT的phase，丢弃
                        if clog:
                            clog.log(env_id, "episode_end", {"episode_steps": ros_env.step_count, "ending": episode_ending}, "discarded_phase")
                        continue
                else:
                    # 评估阶段：只有EVAL_COLLECT阶段的episode才计入统计
                    if current_phase_after_episode != PHASE_EVAL_COLLECT:
                        if clog:
                            clog.log(env_id, "episode_end", {"episode_steps": ros_env.step_count, "ending": episode_ending}, "discarded_eval_phase")
                        continue
                
                # 训练阶段才写入经验队列；评估阶段仅统计
                if not is_eval:
                    # experiences 为本 episode 的一批 transition
                    # 在添加到队列之前，检查是否存在包含NaN/Inf的样本：
                    #  - 若存在任意一条无效样本，则整条episode直接丢弃（不写入队列，也不占用episode编号）
                    #  - 若全部样本有效，则原样写入队列
                    invalid_count = 0
                    for exp in experiences:
                        if not _is_valid_experience(exp):
                            invalid_count += 1
                    
                    if invalid_count > 0:
                        # 本episode存在NaN/Inf，整条episode丢弃
                        print(
                            f"环境 {env_id} 当前episode包含 {invalid_count}/{len(experiences)} 条包含NaN/Inf的经验，"
                            f"整条episode将被丢弃且不写入经验队列，不占用episode编号。"
                        )
                        if clog:
                            clog.log(env_id, "episode_end", {"episode_steps": ros_env.step_count, "invalid_count": invalid_count}, "discarded_nan_inf")
                        continue
                    
                    # 所有样本均为有效数值，直接写入经验队列
                    try:
                        experience_queue.put({"outcome": episode_ending, "experiences": experiences})
                        if clog:
                            clog.log(env_id, "queue_put", {"outcome": episode_ending, "experiences_len": len(experiences)}, "ok")
                        if clog:
                            clog.log(env_id, "episode_end", {"episode_steps": ros_env.step_count, "episode_reward": episode_reward, "ending": episode_ending, "experiences_len": len(experiences)}, "submitted")
                    except Exception as e:
                        print(f"[ERROR] 环境 {env_id} 推送经验到队列失败: {e}")
                        if clog:
                            clog.log(env_id, "episode_end", {"episode_steps": ros_env.step_count}, "submit_failed")
                
                # 以 ros_env.initial_target_distance 为准（reset 后首步 step 的 distance）
                target_distance = getattr(ros_env, "initial_target_distance", None)
                if target_distance is None:
                    if ros_env.episode_start_position is not None and ros_env.target is not None:
                        target_distance = np.linalg.norm([
                            ros_env.target[0] - ros_env.episode_start_position[0],
                            ros_env.target[1] - ros_env.episode_start_position[1]
                        ])
                    else:
                        target_distance = ros_env.target_dist
                
                # episode编号：
                # - 评估阶段：使用 global_stats 作为“评估窗口”的统计与编号来源（会在每次评估开始时 reset）
                # - 训练阶段：不再写入 global_stats，避免被评估 reset 影响；训练episode编号改为独立全局计数器 total_episodes_counter
                if is_eval:
                    # 评估阶段：计入评估统计并返回评估窗口内的episode编号
                    global_episode_number = global_stats.add_episode_result(
                        goal, collision, timeout, episode_reward, target_dist=ros_env.target_dist
                    )
                else:
                    # 训练阶段：全局训练episode计数（严格递增，不受评估reset影响）
                    with total_episodes_counter.get_lock():
                        total_episodes_counter.value += 1
                        global_episode_number = int(total_episodes_counter.value)

                # 写入本环境“最近一次 episode 的 target_dist”，供主进程判断是否“所有采集进程都已达到 max_target_dist”
                if env_target_dist_list is not None:
                    try:
                        env_target_dist_list[env_id] = float(ros_env.target_dist)
                    except Exception:
                        pass
                
                # 更新全局样本计数（用于日志）
                if not is_eval and experiences:
                    with total_added_step.get_lock():
                        total_added_step.value += len(experiences)
                if is_eval:
                    with eval_collected_lock:
                        eval_collected_ref.value += 1
                        current_eval_count = eval_collected_ref.value
                    if clog:
                        clog.log(env_id, "episode_end", {"episode_steps": ros_env.step_count, "episode_reward": episode_reward, "ending": episode_ending}, "eval_recorded")
                    if env_id == 0:  # 只在环境0打印，避免刷屏
                        print(f"[环境 {env_id}] 评估episode完成，eval_collected={current_eval_count}/{eval_target_ref.value}")
                
                # 标记当前轮次的收集完成（或被截断）
                # 训练线程会根据round_done_counter的值来设置phase_ref为TRAIN_DRAIN
                if not is_eval:
                    with round_done_counter.get_lock():
                        round_done_counter.value += 1
                        new_round_done = int(round_done_counter.value)
                    if clog:
                        clog.log(env_id, "round_done", {"local_round": local_round, "round_done_counter": new_round_done}, "incremented")
                    # 记录本环境在该轮已提交过一个有效训练episode，防止同轮再次提交
                    last_committed_round = local_round
                
                # 获取当前时间
                current_time = datetime.now()
                
                
                # 使用折扣后的奖励分项值（已在每个 step 中计算）
                goal_sum = discounted_goal_sum
                collision_sum = discounted_collision_sum
                obs_sum = discounted_obs_sum
                yaw_sum = discounted_yawrate_sum
                angle_sum = discounted_angle_sum
                linear_sum = discounted_linear_sum
                step_penalty_sum = discounted_step_sum
                progress_sum = discounted_progress_sum
                target_distance_sum = discounted_target_distance_sum
                linear_acc_osc_sum = discounted_linear_acc_osc_sum
                yawrate_osc_sum = discounted_yawrate_osc_sum

                # 读取所有奖惩开关状态（确保所有开启的项都被打印）
                enable_obs = getattr(ros_env, "enable_obs_penalty", False)
                enable_yawrate = getattr(ros_env, "enable_yawrate_penalty", False)
                enable_angle = getattr(ros_env, "enable_angle_penalty", False)
                enable_linear = getattr(ros_env, "enable_linear_penalty", False)
                enable_step = getattr(ros_env, "enable_step_penalty", False)
                enable_target_distance = getattr(ros_env, "enable_target_distance_penalty", False)
                enable_linear_acc_osc = getattr(ros_env, "enable_linear_acceleration_oscillation_penalty", False)
                enable_yawrate_osc = getattr(ros_env, "enable_yawrate_oscillation_penalty", False)
                enable_progress = getattr(ros_env, "enable_progress_reward", False)

                detail_parts = [
                    f"goal={goal_sum:.6f}",
                    f"collision={collision_sum:.6f}",
                ]
                # step：将所有“非终止型”的开启分项汇总为 step_total，便于快速对齐 total_reward
                step_total = 0.0
                if enable_obs:
                    step_total += obs_sum
                if enable_yawrate:
                    step_total += yaw_sum
                if enable_angle:
                    step_total += angle_sum
                if enable_linear:
                    step_total += linear_sum
                if enable_step:
                    step_total += step_penalty_sum
                if enable_target_distance:
                    step_total += target_distance_sum
                if enable_linear_acc_osc:
                    step_total += linear_acc_osc_sum
                if enable_yawrate_osc:
                    step_total += yawrate_osc_sum
                if enable_progress:
                    step_total += progress_sum
                detail_parts.append(f"step={step_total:.6f}")
                # 所有开启的奖惩项都要打印（即使值为0也要显示）
                if enable_obs:
                    detail_parts.append(f"obs={obs_sum:.6f}")
                if enable_yawrate:
                    detail_parts.append(f"yawrate={yaw_sum:.6f}")
                if enable_angle:
                    detail_parts.append(f"angle={angle_sum:.6f}")
                if enable_linear:
                    detail_parts.append(f"linear={linear_sum:.6f}")
                if enable_step:
                    detail_parts.append(f"step_penalty={step_penalty_sum:.6f}")
                if enable_target_distance:
                    detail_parts.append(f"target_distance={target_distance_sum:.6f}")
                if enable_linear_acc_osc:
                    detail_parts.append(f"linear_acc_osc={linear_acc_osc_sum:.6f}")
                if enable_yawrate_osc:
                    detail_parts.append(f"yawrate_osc={yawrate_osc_sum:.6f}")
                if enable_progress:
                    detail_parts.append(f"progress={progress_sum:.6f}")

                # 将结束状态放在最前面，然后是总reward，最后是其他奖励分项
                detail_parts_with_total = [f"end={episode_ending}", f"total_reward={episode_reward:.6f}"] + detail_parts
                detail_str = ", ".join(detail_parts_with_total)

                # 输出详细的episode信息（含奖励明细），时间戳放在最前面，便于与训练日志对齐
                # target_dist 是配置的目标距离上限，target_distance 为 ros_env.initial_target_distance（首步 step 的 distance）
                # 这里的 Queue(episodes) 表示当前经验队列中累计的 episode 数量（近似反映待训练数据量），
                # 而本地训练缓冲区的样本数量由训练线程在训练日志中单独打印。
                queue_size_episodes = 0
                try:
                    # multiprocessing.Queue.qsize() 在部分平台上可能不完全精确，但用于日志监控是可以接受的
                    queue_size_episodes = experience_queue.qsize()
                except Exception:
                    queue_size_episodes = -1  # 获取失败时输出 -1 以示区分
                
                mode_str = "EVAL" if is_eval else "TRAIN"
                episode_info = (
                    f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} 环境 {env_id} "
                    f"Mode: {mode_str} Round: {local_round} Episode: {global_episode_number} "
                    f"Target Distance: {ros_env.target_dist:.2f} (actual: {target_distance:.2f}) Steps: {ros_env.step_count}\n"
                    f"  Reward Detail: {detail_str}"
                )
                if clog:
                    clog.log(env_id, "episode_reward_detail", {"episode_number": global_episode_number, "steps": ros_env.step_count, "ending": episode_ending, "mode": mode_str, "round": local_round, "target_dist": ros_env.target_dist, "target_distance": target_distance, "episode_reward": episode_reward}, {"detail": detail_str, "queue_size_episodes": queue_size_episodes})
                print(episode_info)
                
            except Exception as e:
                print(f"环境 {env_id} Episode {global_episode_number} 出错: {e}")
                time.sleep(0.1)
    except Exception as e:
        if clog:
            clog.log(env_id, "collect_exit", {"reason": "exception", "msg": str(e)}, "exiting")
        print(f"环境 {env_id} 初始化失败: {e}")
    finally:
        if clog is not None:
            clog.close()
        if env_logger is not None:
            env_logger.close()


def training_thread(model_manager, env_queues, config, total_added_step, total_episodes_counter, current_buffer_size_ref, round_done_counter, current_round_ref, phase_ref, eval_target_ref, eval_collected_lock, eval_collected_ref, global_stats, env_target_dist_list, max_steps_for_pause, check_best_model_fn, train_logger=None, collect_logger=None):
    """按轮次训练：每轮收集完一定比例的episode后统一训练
    
    周期性评估触发位置：每轮训练完成后、开启下一轮收集之前。
    """
    def _log(msg):
        """通用日志：写入train_log，同时输出到综合日志"""
        if train_logger:
            train_logger.log(msg)
        print(msg)  # 同时输出到综合日志
    
    def _collect_log(msg):
        """收集相关日志：现在直接写入训练日志（train_log），不再单独使用collect_log文件。"""
        _log(msg)
    
    def _train_log(msg):
        """训练相关日志：写入train_log"""
        _log(msg)

    try:
        _log("训练线程启动，按轮次等待数据再启动训练")
        
        training_count = 0
        total_samples_drawn = 0  # 总抽样样本数统计
        recent_actor_losses = []  # 存储最近N次训练的actor损失
        threshold_envs = max(1, math.ceil(config.get('collection_wait_ratio', 0.8) * config.get('num_envs', 1)))
        num_envs = config.get('num_envs', 1)
        # 周期性评估配置
        eval_every_rounds = int(config.get('eval_every_rounds', 0) or 0)
        eval_episodes_per_round = int(config.get('eval_episodes_per_round', 0) or 0)
        eval_start_after_all_max = bool(config.get('eval_start_after_all_target_dist_max', False))
        periodic_eval_enabled = eval_every_rounds > 0 and eval_episodes_per_round > 0
        last_eval_training_count = 0
        max_reached_logged = False
        
        _train_log(f"训练线程初始化: threshold_envs={threshold_envs}, num_envs={num_envs}")
        
        # 本地缓冲区（固定使用 float32 以降低内存占用并提升速度）
        if config.get('enable_stratified_replay', False):
            local_buffer = NumpyStratifiedReplayBuffer(
                buffer_size=config.get('buffer_size', 50000),
            dtype=np.float32,
            recent_buffer_ratio=config.get('recent_buffer_ratio', 0.1),
                recent_batch_ratio=config.get('recent_batch_ratio', 0.3),
                stratified_sampling=config.get('stratified_replay', {}),
            )
        else:
            local_buffer = NumpyReplayBuffer(
                buffer_size=config.get('buffer_size', 50000),
                dtype=np.float32,
                recent_buffer_ratio=config.get('recent_buffer_ratio', 0.1),
                recent_batch_ratio=config.get('recent_batch_ratio', 0.3),
        )
        
        # 初始轮次
        current_round_ref.value = 0
        phase_ref.value = PHASE_TRAIN_COLLECT
        
        _train_log(f"初始轮次: current_round={current_round_ref.value}, phase_ref已设置为TRAIN_COLLECT")
        
        while True:
            try:
                max_rounds_cfg = config.get('max_rounds', config.get('max_training_count', 0))
                if max_rounds_cfg and max_rounds_cfg > 0 and training_count >= max_rounds_cfg:
                    _log(f"达到最大轮次数 {max_rounds_cfg}，训练完成！")
                    break
                
                # 准备新一轮收集
                with round_done_counter.get_lock():
                    round_done_counter.value = 0
                phase_ref.value = PHASE_TRAIN_COLLECT
                round_index = current_round_ref.value
                round_steps = 0
                
                _collect_log(f"开始新一轮收集: round_index={round_index}, round_done_counter已重置为0, phase_ref已设置为TRAIN_COLLECT")
                
                # 等待达到阈值并让所有进程提交本轮数据
                wait_iteration = 0
                last_logged_done = -1  # 记录上一次记录的round_done_counter值
                while True:
                    pulled_any = False
                    queue_sizes = []
                    for q_idx, q in enumerate(env_queues):
                        queue_size = 0
                        while True:
                            try:
                                experiences = q.get_nowait()
                            except queue.Empty:
                                break
                            if experiences:
                                if config.get('enable_stratified_replay', False):
                                    if isinstance(experiences, dict):
                                        experiences_outcome = experiences.get("outcome", "Timeout")
                                        experiences_list = experiences.get("experiences", [])
                                    else:
                                        experiences_outcome = "Timeout"
                                        experiences_list = experiences
                                    local_buffer.add_episode(experiences_list, outcome=experiences_outcome)
                                else:
                                    if isinstance(experiences, dict):
                                        experiences = experiences.get("experiences", [])
                                    local_buffer.add_batch(experiences)
                                round_steps += len(experiences_list if isinstance(experiences, dict) else experiences)
                                pulled_any = True
                                queue_size += 1
                        queue_sizes.append(queue_size)
                    
                    current_done = round_done_counter.value
                    if current_done >= threshold_envs:
                        if phase_ref.value == PHASE_TRAIN_COLLECT:
                            phase_ref.value = PHASE_TRAIN_DRAIN
                            _collect_log(f"达到阈值({threshold_envs}), 设置phase_ref为TRAIN_DRAIN, round_done_counter={current_done}")
                    if current_done >= num_envs:
                        _collect_log(f"所有进程完成({current_done}/{num_envs}), 退出等待循环, round_steps={round_steps}")
                        break
                    
                    # 只在round_done_counter更新时记录日志
                    if current_done != last_logged_done:
                        phase_str = ["TRAIN_COLLECT", "TRAIN_DRAIN", "EVAL_COLLECT", "PAUSE", "STOP"][phase_ref.value]
                        _collect_log(f"等待中: round_done_counter={current_done}/{num_envs} (阈值={threshold_envs}), "
                              f"round_steps={round_steps}, 队列大小={queue_sizes}, "
                              f"phase_ref={phase_str}")
                        last_logged_done = current_done
                    
                    wait_iteration += 1
                    time.sleep(0.1)
                    
                pull_start_time = time.time()
                # 确保队列清空
                final_queue_sizes = []
                for q_idx, q in enumerate(env_queues):
                    queue_size = 0
                    while True:
                        try:
                            experiences = q.get_nowait()
                        except queue.Empty:
                            break
                        if experiences:
                            if config.get('enable_stratified_replay', False):
                                if isinstance(experiences, dict):
                                    experiences_outcome = experiences.get("outcome", "Timeout")
                                    experiences_list = experiences.get("experiences", [])
                                else:
                                    experiences_outcome = "Timeout"
                                    experiences_list = experiences
                                local_buffer.add_episode(experiences_list, outcome=experiences_outcome)
                                round_steps += len(experiences_list)
                            else:
                                if isinstance(experiences, dict):
                                    experiences = experiences.get("experiences", [])
                                local_buffer.add_batch(experiences)
                            round_steps += len(experiences)
                            queue_size += 1
                    final_queue_sizes.append(queue_size)
                pull_time = time.time() - pull_start_time if round_steps > 0 else 0.0
                
                _collect_log(f"队列清空完成: 最终round_steps={round_steps}, 清空时各队列大小={final_queue_sizes}, 耗时={pull_time:.3f}秒")
                
                # 关闭本轮采集，准备训练（设置为PAUSE，防止新episode开始）
                phase_ref.value = PHASE_PAUSE
                
                _collect_log(f"关闭本轮采集: phase_ref已设置为PAUSE, 准备训练")
                
                # 计算缓冲区大小（总）以及子缓冲区大小（若启用分层采样）
                buffer_size = local_buffer.size()
                stratified_enabled = config.get('enable_stratified_replay', False)
                if stratified_enabled:
                    goal_buf_size = local_buffer.goal_buffer.size()
                    coll_buf_size = local_buffer.collision_buffer.size()
                    timeout_buf_size = local_buffer.timeout_buffer.size()
                    # 判断是否满足阈值以真正启用分层采样
                    min_steps_to_enable = int(config.get('stratified_replay', {}).get('min_steps_to_enable', 0))
                    stratified_active = not (
                        min_steps_to_enable > 0 and (
                            goal_buf_size < min_steps_to_enable
                            or coll_buf_size < min_steps_to_enable
                            or timeout_buf_size < min_steps_to_enable
                        )
                    )
                else:
                    goal_buf_size = coll_buf_size = timeout_buf_size = None
                    stratified_active = False
                batch_size = config['batch_size']
                
                if round_steps == 0 or buffer_size == 0:
                    _collect_log(f"本轮未收集到有效step: round_steps={round_steps}, buffer_size={buffer_size}, 跳过训练")
                    _collect_log("本轮未收集到有效step，跳过训练，等待下一轮。")
                    current_round_ref.value = round_index + 1
                    phase_ref.value = PHASE_TRAIN_COLLECT
                    _collect_log(f"重新开启收集: phase_ref已设置为TRAIN_COLLECT, current_round={current_round_ref.value}")
                    time.sleep(0.1)
                    continue
                
                training_iterations = max(1, int(round_steps * config.get('train_n_per_step', 1.0)))
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                
                _train_log(f"开始训练: training_count={training_count+1}, round_steps={round_steps}, "
                      f"training_iterations={training_iterations}, buffer_size={buffer_size}")
                
                message = (
                    f"{current_time} 第{training_count+1}次训练开始："
                    f"本轮step数: {round_steps}，训练迭代数: {training_iterations}，"
                    f"当前总episode数: {total_episodes_counter.value}，"
                    f"当前总step数: {total_added_step.value}，"
                )
                _log(message)
                
                # 训练
                start_time = time.time()
                avg_critic_loss, critic_losses, avg_actor_loss, actor_losses, avg_critic_grad, avg_actor_grad, avg_entropy, avg_alpha_grad = model_manager.model.train(
                    replay_buffer=local_buffer,
                    iterations=training_iterations,
                    batch_size=batch_size,
                    stats=global_stats.get_statistics(use_window=True),
                )
                end_time = time.time()
                
                sample_time = getattr(model_manager.model, "last_sample_time", None)
                update_time = getattr(model_manager.model, "last_update_time", None)
                compute_time = None
                if sample_time is not None and update_time is not None:
                    compute_time = max(update_time - sample_time, 0.0)
                
                samples_this_training = training_iterations * batch_size
                total_samples_drawn += samples_this_training
                total_experiences_added = total_added_step.value
                avg_training_times = total_samples_drawn / max(total_experiences_added, 1)
                
                training_count += 1
                end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                total_train_time = end_time - start_time
                
                buffer_info = (
                    f"当前缓冲区大小: {buffer_size}" if not stratified_enabled else
                    f"当前缓冲区大小: {buffer_size} (Goal: {goal_buf_size}, Collision: {coll_buf_size}, Timeout: {timeout_buf_size})"
                )
                _log(
                    f"{end_time_str} 第{training_count}次训练完成 | {buffer_info} | "
                    f"分层采样启用: {stratified_enabled}, 实际生效: {stratified_active}"
                )
                # 训练完成后再输出分层采样信息
                _log(
                    f"  本轮step: {round_steps} | 总抽样数: {total_samples_drawn} | 总样本数: {total_experiences_added} "
                    f"| 样本平均抽样次数: {avg_training_times:.2f}"
                )
                
                model_manager.add_critic_loss(avg_critic_loss)
                window_avg_critic_loss = model_manager.get_average_critic_loss(config.get('avg_loss_window_size', 10))
                if avg_critic_grad is not None:
                    if isinstance(avg_critic_grad, dict):
                        grad_info = f" | critic全局参数梯度L2范数(裁剪前:{avg_critic_grad['before']:.6f}, 裁剪后:{avg_critic_grad['after']:.6f})"
                    else:
                        grad_info = f" | critic全局参数梯度L2范数: {avg_critic_grad:.6f}"
                else:
                    grad_info = ""
                _log(f"  本次训练的平均critic网络损失: {avg_critic_loss:.6f}{grad_info}")
                
                if avg_actor_loss is not None:
                    recent_actor_losses.append(avg_actor_loss)
                    if len(recent_actor_losses) > 100:
                        recent_actor_losses.pop(0)
                    if avg_actor_grad is not None:
                        if isinstance(avg_actor_grad, dict):
                            actor_grad_info = f" | actor全局参数梯度L2范数(裁剪前:{avg_actor_grad['before']:.6f}, 裁剪后:{avg_actor_grad['after']:.6f})"
                        else:
                            actor_grad_info = f" | actor全局参数梯度L2范数: {avg_actor_grad:.6f}"
                    else:
                        actor_grad_info = ""
                    _log(f"  本次训练的平均actor网络损失: {avg_actor_loss:.6f}{actor_grad_info}")
                
                entropy_info = ""
                if avg_entropy is not None:
                    entropy_info = f" | 熵值: {avg_entropy:.6f}"
                if avg_alpha_grad is not None:
                    entropy_info += f" | alpha梯度L2范数: {avg_alpha_grad:.6f}"
                if entropy_info:
                    _log(f"  熵值统计:{entropy_info}")
                
                if sample_time is not None and update_time is not None:
                    total_sample_time = pull_time + sample_time
                    total_train_log_time = total_sample_time + compute_time if compute_time is not None else total_sample_time
                    _log(f"  训练耗时: {total_train_log_time:.2f}秒")
                    _log(
                        f"  采样耗时总计: {total_sample_time:.2f}秒 "
                        f"(拉取到本地buffer: {pull_time:.2f}秒, 从本地buffer随机采样: {sample_time:.2f}秒)"
                    )
                    _log(f"  前向/反向(网络更新)耗时: {compute_time:.2f}秒")
                else:
                    _log(f"  训练耗时: {total_train_time:.2f}秒")
                
                if hasattr(model_manager, 'training_count_ref') and model_manager.training_count_ref is not None:
                    model_manager.training_count_ref.value = training_count
                model_manager.update_critic_loss(avg_critic_loss)
                current_buffer_size_ref.value = buffer_size
                
                model_manager.update_dict_from_model()
                
                if training_count % config.get('save_every', 50) == 0:
                    save_dir = Path(config['model_save_dir'])
                    save_dir.mkdir(parents=True, exist_ok=True)
                    # 实时模型保存：始终覆盖同名文件，避免训练时间长导致生成过多模型文件
                    # 输出文件名固定为：
                    #   SAC_actor.pth, SAC_critic.pth, SAC_critic_target.pth
                    model_manager.model.save(filename="SAC", directory=save_dir)
                
                
                phase_str = ["TRAIN_COLLECT", "TRAIN_DRAIN", "EVAL_COLLECT", "PAUSE", "STOP"][phase_ref.value]
                _train_log(f"训练完成，准备下一轮: current_round={current_round_ref.value}, phase_ref={phase_str}")
                
                # ==================== 周期性评估（训练线程尾部，开启下一轮前） ====================
                if periodic_eval_enabled:
                    max_target_dist = float(config.get('max_target_dist', 0.0))
                    eps = 1e-6
                    try:
                        all_at_max = all(float(td) >= (max_target_dist - eps) for td in list(env_target_dist_list))
                    except Exception:
                        all_at_max = False
                    if all_at_max and not max_reached_logged:
                        _log(f"[周期性评估] 已检测到所有环境 target_dist 达到 max_target_dist={max_target_dist:.2f}，将开始按轮次触发评估。")
                        max_reached_logged = True
                    
                    if (not eval_start_after_all_max) or all_at_max:
                        if (training_count - last_eval_training_count) >= eval_every_rounds:
                            _log(f"[周期性评估] 触发评估：training_count={training_count}，本轮评估 episode 数={eval_episodes_per_round}（无动作噪声）")
                        
                            
                            # 切换到评估阶段
                            phase_ref.value = PHASE_EVAL_COLLECT
                            eval_target_ref.value = eval_episodes_per_round
                            eval_collected_ref.value = 0
                            global_stats.reset()
                            
                            _collect_log(f"[周期性评估] 已设置phase_ref=EVAL_COLLECT, eval_target={eval_episodes_per_round}, eval_collected=0")
                            
                            # 等待评估完成
                            wait_count = 0
                            while True:
                                with eval_collected_lock:
                                    current_eval = eval_collected_ref.value
                                if current_eval >= eval_episodes_per_round:
                                    # 达到目标后，立即切回训练收集阶段：
                                    # - 训练线程把 phase_ref 设置为 PHASE_TRAIN_COLLECT
                                    # - 此时仍在运行评估 episode 的 worker，会在各自循环中检测到current_phase_in_loop != PHASE_EVAL_COLLECT（见约1139行），
                                    # 从而设置 forced_stop_flag 跳出本次 episode，并在统计阶段被丢弃
                                    phase_ref.value = PHASE_TRAIN_COLLECT
                                    _collect_log(f"[周期性评估] 达到目标episode数({current_eval}/{eval_episodes_per_round})，切回phase_ref=TRAIN_COLLECT")
                                    break
                                wait_count += 1
                                if wait_count % 10 == 0:  # 每5秒打印一次
                                    _collect_log(f"[周期性评估] 等待中: eval_collected={current_eval}/{eval_episodes_per_round}, phase_ref=EVAL_COLLECT")
                                time.sleep(0.1)
                            
                            _log("[周期性评估] 本轮评估完成，输出统计报告")
                            stats = global_stats.get_statistics(use_window=False)
                            _print_statistics_report(stats, train_logger=train_logger)
                            # 按评估终点率保存最好模型
                            check_best_model_fn(stats)
                            
                            # 回到训练阶段（当前已切回TRAIN_COLLECT，这里只更新时间戳和轮次）
                            last_eval_training_count = training_count
                
                # 下一轮（如果当前不是评估阶段，则开启新一轮收集）
                current_round_ref.value = round_index + 1
                # 若当前不在评估收集阶段（包括评估刚结束或训练路径），则切回训练收集阶段
                if phase_ref.value != PHASE_EVAL_COLLECT:
                    phase_ref.value = PHASE_TRAIN_COLLECT
                
                time.sleep(0.1)
                
            except Exception as e:
                _log(f"训练线程出错: {e}")
                if "Broken pipe" in str(e) or "Errno 32" in str(e):
                    _log("检测到Broken pipe错误，训练线程退出")
                    break
                time.sleep(0.1)
                
    except Exception as e:
        _log(f"训练线程初始化失败: {e}")




class ParallelMultiEnvTrainer:
    """并行多环境训练器"""
    
    def __init__(self, config=None, config_path=None):
        cfg = config or {}

        def _to_bool(val, default=True):
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                low = val.strip().lower()
                if low in ("true", "1", "yes", "y", "on"):
                    return True
                if low in ("false", "0", "no", "n", "off"):
                    return False
            return default

        # ==================== 从配置字典读取参数（统一入口） ====================
        # 训练参数
        self.num_envs = cfg.get('num_envs', 4)
        self.batch_size = cfg.get('batch_size', 40)
        self.training_iterations = cfg.get('training_iterations', 200)
        self.train_n_per_step = cfg.get('train_n_per_step', 1.0)
        self.collection_wait_ratio = cfg.get('collection_wait_ratio', 0.8)
        self.min_collection_ratio = cfg.get('min_collection_ratio', 0.25)
        self.save_every = cfg.get('save_every', 50)
        self.buffer_size = cfg.get('buffer_size', 50000)
        self.max_training_count = cfg.get('max_training_count', 1000)
        self.max_rounds = cfg.get('max_rounds', self.max_training_count)
        self.max_steps_ratio = cfg.get('max_steps_ratio', 100)
        self.max_steps = cfg.get('max_steps', 3000)
        self.max_steps_min = cfg.get('max_steps_min', 50)

        # 机器人与环境参数
        self.max_velocity = cfg.get('max_velocity', 1.0)
        self.max_acceleration = cfg.get('max_acceleration', 5.0)
        self.max_deceleration = cfg.get('max_deceleration', -5.0)
        self.neglect_angle = cfg.get('neglect_angle', 0)
        self.max_yawrate = cfg.get('max_yawrate', 20.0)
        self.scan_range = cfg.get('scan_range', 5)
        self.max_target_dist = cfg.get('max_target_dist', 15.0)
        self.init_target_distance = cfg.get('init_target_distance', 2.0)
        self.target_dist_increase = cfg.get('target_dist_increase', 0.01)
        self.target_reached_delta = cfg.get('target_reached_delta', 0.3)
        self.collision_delta = cfg.get('collision_delta', 0.25)
        self.world_size = cfg.get('world_size', 15)
        self.goals_per_map = cfg.get('goals_per_map', 4)
        self.obs_min_dist = cfg.get('obs_min_dist', 2)
        self.obs_num = cfg.get('obs_num', 20)
        self.costmap_resolution = cfg.get('costmap_resolution', 0.3)
        self.obstacle_size = cfg.get('obstacle_size', 0.3)
        self.obs_distribution_mode = cfg.get('obs_distribution_mode', 'uniform')

        # 连通区域/一致性检查
        self.region_select_bias = cfg.get('region_select_bias', 1.0)
        self.enable_weight_consistency_check = _to_bool(cfg.get('enable_weight_consistency_check', False), False)

        # 模型与网络参数
        # 根据bin_num自动计算：base_state_dim = bin_num + 非激光特征(7)
        self.bin_num = cfg.get('bin_num', 72)
        non_lidar_dim = 7
        self.base_state_dim = int(self.bin_num) + non_lidar_dim
        self.state_history_steps = cfg.get('state_history_steps', 0)
        self.hidden_layers = cfg.get('hidden_layers', [1024, 512])
        self.avg_loss_window_size = cfg.get('avg_loss_window_size', 5)

        self.stats_window_size = cfg.get('stats_window_size', 100)
        gpu_id = cfg.get('gpu_id', 0)
        max_action = cfg.get('max_action', 1.0)
        action_dim = cfg.get('action_dim', 2)

        # 训练算法参数
        discount_factor = cfg.get('discount_factor', 0.99)
        actor_update_frequency = cfg.get('actor_update_frequency', 1)
        critic_target_update_frequency = cfg.get('critic_target_update_frequency', 4)

        # 奖励缩放
        self.reward_scale = cfg.get('reward_scale', 1.0)

        # 时间控制参数
        self.sim_time = cfg.get('sim_time', 0.1)
        self.step_sleep_time = cfg.get('step_sleep_time', 0.1)
        self.reset_step_count = cfg.get('reset_step_count', 3)

        # 动作噪声
        self.action_noise_std = cfg.get('action_noise_std', 0.2)

        # 奖励函数参数（兼容旧字段名）
        self.goal_reward = cfg.get('goal_reward', 1000.0)
        self.base_collision_penalty = cfg.get('collision_penalty_base', cfg.get('base_collision_penalty', -1000.0))
        self.angle_base_penalty = cfg.get('angle_penalty_base', cfg.get('angle_base_penalty', 0.0))
        self.base_linear_penalty = cfg.get('linear_penalty_base', cfg.get('base_linear_penalty', -1.0))
        self.yawrate_penalty_base = cfg.get('yawrate_penalty_base', 0.0)

        self.enable_obs_penalty = _to_bool(cfg.get('enable_obs_penalty', True), True)
        self.enable_yawrate_penalty = _to_bool(cfg.get('enable_yawrate_penalty', True), True)
        self.enable_angle_penalty = _to_bool(cfg.get('enable_angle_penalty', True), True)
        self.enable_linear_penalty = _to_bool(cfg.get('enable_linear_penalty', True), True)
        self.enable_step_penalty = _to_bool(cfg.get('enable_step_penalty', False), False)
        self.enable_target_distance_penalty = _to_bool(cfg.get('enable_target_distance_penalty', False), False)
        self.enable_progress_reward = _to_bool(cfg.get('enable_progress_reward', False), False)
        self.enable_linear_acceleration_oscillation_penalty = _to_bool(cfg.get('enable_linear_acceleration_oscillation_penalty', False), False)
        self.enable_yawrate_oscillation_penalty = _to_bool(cfg.get('enable_yawrate_oscillation_penalty', False), False)

        self.progress_reward_base = cfg.get('progress_reward_base', 1.0)
        self.target_distance_penalty_base = cfg.get('target_distance_penalty_base', -1.0)
        self.step_penalty_base = cfg.get('step_penalty_base', 0.0)
        self.linear_acceleration_oscillation_penalty_base = cfg.get('linear_acceleration_oscillation_penalty_base', -1.0)
        self.yawrate_oscillation_penalty_base = cfg.get('yawrate_oscillation_penalty_base', -1.0)

        # 障碍物距离惩罚参数
        self.obs_penalty_threshold = cfg.get('obs_penalty_threshold', 1.0)
        self.obs_penalty_base = cfg.get('obs_penalty_base', -10.0)
        self.obs_penalty_power = cfg.get('obs_penalty_power', 2.0)
        self.min_obs_penalty_threshold = cfg.get('min_obs_penalty_threshold', 0.5)
        self.obs_penalty_high_weight = cfg.get('obs_penalty_high_weight', 1.0)
        self.obs_penalty_low_weight = cfg.get('obs_penalty_low_weight', 0.4)
        self.obs_penalty_middle_ratio = cfg.get('obs_penalty_middle_ratio', 0.4)

        # 路径参数（兼容 load_path/model_load_dir 等）
        model_save_dir = cfg.get('model_save_dir', None)
        model_load_dir = cfg.get('load_path', cfg.get('model_load_dir', None))
        load_model_str = cfg.get('load_model', cfg.get('load_existing_model', True))
        load_model = load_model_str if isinstance(load_model_str, bool) else str(load_model_str).lower() == 'true'

        # stats_window_size 老字段兼容
        self.max_training_count = cfg.get('max_training_count', self.max_training_count)

        # 设备配置
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if False:
            pass
        elif torch.cuda.is_available():
            if cuda_visible_devices:
                self.device = torch.device("cuda")
                print(f"使用GPU: cuda (物理GPU: {cuda_visible_devices})")
            else:
                self.device = torch.device(f"cuda:{gpu_id}")
                print(f"使用GPU: cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
            print("CUDA不可用，使用CPU")

        # 配置文件路径
        self.config_path = config_path

        # 设备配置
        # 注意：如果通过环境变量 CUDA_VISIBLE_DEVICES 设置了 GPU，PyTorch 视角下 GPU 索引从 0 开始
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if device:
            self.device = device
        elif torch.cuda.is_available():
            if cuda_visible_devices:
                # 如果设置了 CUDA_VISIBLE_DEVICES，PyTorch 只能看到指定的 GPU，索引总是从 0 开始
                self.device = torch.device("cuda")
                print(f"使用GPU: cuda (物理GPU: {cuda_visible_devices})")
            else:
                self.device = torch.device(f"cuda:{gpu_id}")
                print(f"使用GPU: cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
            print("CUDA不可用，使用CPU")

        # 先从配置文件中读取完整配置（如果存在），供日志目录等使用
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._loaded_config = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"[ERROR] 警告: 读取配置文件失败: {e}")
                self._loaded_config = {}
        else:
            self._loaded_config = {}

        # ==================== 日志与模型保存目录配置 ====================
        # 统一使用 TRAINING_TIMESTAMP（若未设置则 fallback 为当前时间）
        # 日志目录由 train.yaml 的 multi_env_log_model_dir + 时间戳子目录组合得到：
        #   <multi_env_log_model_dir>/train_<timestamp>/model
        #   <multi_env_log_model_dir>/train_<timestamp>/best_model
        self.timestamp = os.environ.get("TRAINING_TIMESTAMP", datetime.now().strftime("%Y%m%d_%H%M%S"))

        base_log_dir = Path(self._loaded_config.get("multi_env_log_model_dir", "log/multi_env_training"))
        self.log_dir = base_log_dir / f"train_{self.timestamp}"
        self._log_paths = multi_env_log_paths(self.log_dir, self.timestamp)
        # 每环境日志根目录（例如：train_xxx/env_logs/）
        self.env_logs_dir = Path(self._log_paths.get("env_logs_dir", self.log_dir / "env_logs"))
        # 实时模型保存目录
        self.model_save_dir = self.log_dir / "model"
        # 最好模型单独保存到 best_model 子目录，便于区分
        self.best_model_save_dir = self.log_dir / "best_model"

        self.model_load_dir = Path(model_load_dir) if model_load_dir else Path("src/drl_navigation_ros2/models/SAC")
        self.load_model = load_model
        
        # 创建目录
        self._setup_directories()
        # 保存本次训练使用的配置快照：
        #   - 转换为图片格式保存为只读文件，确保完全不可修改
        #   - 若存在日志目录，则保存在日志目录根下，文件名带时间戳
        #   - 否则，保存在模型保存目录下，文件名同样带时间戳
        if self.config_path and Path(self.config_path).exists():
            try:
                if self.log_dir is not None:
                    config_save_dir = self.log_dir
                else:
                    config_save_dir = self.model_save_dir
                config_save_dir.mkdir(parents=True, exist_ok=True)
                config_filename = f"config_{self.timestamp}.png"
                target_config_path = config_save_dir / config_filename
                # 将YAML配置转换为图片格式保存
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f) or {}
                config_to_image(config_data, target_config_path, "DRL Robot Dog Navigation - Training Config")
            except Exception as e:
                print(f"[ERROR] 警告: 保存配置快照到保存目录失败: {e}")
        
        print(f"初始化SAC模型...")
        print(f"使用state_dim={(self.base_state_dim * (1 + self.state_history_steps) if self.state_history_steps > 0 else self.base_state_dim)} (base_state_dim={self.base_state_dim}, state_history_steps={self.state_history_steps})")
        
        # 初始化模型
        actor_grad_clip_value = self._loaded_config.get('actor_grad_clip_value', 0.0)
        critic_grad_clip_value = self._loaded_config.get('critic_grad_clip_value', 0.0)
        # 学习率相关参数（从配置文件读取，若未配置则使用默认值）
        actor_lr = self._loaded_config.get('actor_lr', 1e-4)
        critic_lr = self._loaded_config.get('critic_lr', 1e-4)
        alpha_lr = self._loaded_config.get('alpha_lr', 1e-4)
        self.model = SAC(
            state_dim=self.base_state_dim * (1 + self.state_history_steps) if self.state_history_steps > 0 else self.base_state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=self.device,
            discount=discount_factor,  # 传递折扣因子
            actor_lr=actor_lr,  # 传递Actor学习率
            critic_lr=critic_lr,  # 传递Critic学习率
            alpha_lr=alpha_lr,  # 传递温度参数学习率
            actor_update_frequency=actor_update_frequency,
            critic_target_update_frequency=critic_target_update_frequency,
            hidden_layers=self.hidden_layers,  # 使用实例变量，确保使用更新后的值
            save_every=0,  # 不自动保存
            load_model=self.load_model,
            save_directory=self.model_save_dir,
            load_directory=self.model_load_dir,
            action_noise_std=self.action_noise_std,
            base_state_dim=self.base_state_dim,  # 传递base_state_dim给SAC模型
            bin_num=self._loaded_config.get('bin_num', 72),  # 激光scan分桶数量
            actor_grad_clip_value=actor_grad_clip_value,  # 传递Actor梯度裁剪值
            critic_grad_clip_value=critic_grad_clip_value,  # 传递Critic梯度裁剪值
            train_log_path=self._log_paths["train_log_path"],
        )
        print(f"初始化SAC模型完成")
        
        # 初始化Manager（必须在SharedModelManager之前）
        self.manager = mp.Manager()
        
        # 创建固定的临时文件目录（在当前目录下的tmp文件夹中）
        import tempfile
        # 获取当前工作目录，创建tmp文件夹（如果不存在）
        current_dir = Path.cwd()
        tmp_base_dir = current_dir / "tmp"
        tmp_base_dir.mkdir(exist_ok=True)
        # 在tmp目录中创建临时子目录
        self.shared_temp_dir = tempfile.mkdtemp(prefix="sac_model_sync_", dir=str(tmp_base_dir))
        print(f"创建共享临时目录: {self.shared_temp_dir}")
        
        # 创建共享的模型相关对象
        shared_model_dict = self.manager.dict()  # 共享模型权重字典
        model_lock = self.manager.Lock()  # 共享模型锁
        training_count_ref = self.manager.Value('i', 0)  # 共享训练次数计数器
        critic_loss_ref = self.manager.Value('d', float('inf'))  # 共享critic损失
        recent_losses_ref = self.manager.list()  # 共享最近损失列表
        # 新的同步控制：按轮次收集 episode 后再训练（使用统一的phase_ref）
        self.round_done_counter = mp.Value('i', 0)  # 当前轮完成/被截断的进程计数
        self.current_round_ref = mp.Value('i', 0)  # 当前轮次编号
        
        # 初始化共享模型管理器
        self.model_manager = SharedModelManager(self.model, shared_model_dict, model_lock, training_count_ref, critic_loss_ref, recent_losses_ref, is_main_process=True, shared_temp_dir=self.shared_temp_dir)
        
        # 立即将主进程的模型权重同步到临时文件
        print("将主进程模型权重同步到临时文件...")
        self.model_manager.update_dict_from_model()
        print("模型权重同步完成")
        
        # 将临时目录路径保存到共享字典，供数据收集进程使用
        self.model_manager.shared_model_dict['shared_temp_dir'] = self.model_manager.temp_dir
        
        # 打印主进程模型结构概要（层级与参数形状）
        def _describe_mlp(input_dim, hidden_layers_list, output_dim):
            layers = [("输入层", input_dim)]
            for idx, hidden_size in enumerate(hidden_layers_list):
                layers.append((f"隐含层{idx + 1}", hidden_size))
            layers.append(("输出层", output_dim))
            return layers

        print("主进程模型结构概览:")
        # actor 结构: trunk 输出为 action_dim 的均值和对数方差，故输出维度为 2 * action_dim
        # 注意：这里直接使用主进程创建模型时的输入维度，避免依赖尚未初始化的 self.config
        _state_dim_effective = int(self.base_state_dim * (1 + self.state_history_steps) if self.state_history_steps > 0 else self.base_state_dim)
        actor_layers = _describe_mlp(_state_dim_effective, self.hidden_layers, 2 * action_dim)
        print(f"  主进程 actor 层级神经元: " + " -> ".join([f"{name}:{size}" for name, size in actor_layers]))
        # critic 结构: Q1/Q2 两个 MLP，输入为 state_dim+action_dim，输出为标量 Q 值
        critic_layers = _describe_mlp(_state_dim_effective + action_dim, self.hidden_layers, 1)
        print(f"  主进程 critic(Q1/Q2) 层级神经元: " + " -> ".join([f"{name}:{size}" for name, size in critic_layers]))
        for key, value in self.model.state_dict().items():
            if isinstance(value, dict):
                print(f"  主进程 {key} 子键: {list(value.keys())}")
            elif hasattr(value, 'shape'):
                print(f"  主进程 {key} 形状: {value.shape}")
            else:
                print(f"  主进程 {key} 类型: {type(value)}")

        print(f"主进程模型文件路径: {self.model_manager.model_file_path}")
        print(f"主进程临时目录: {self.model_manager.temp_dir}")
        
        # 初始化共享缓冲区相关结构
        # 使用每个环境一个队列传输经验，由训练线程集中维护本地大缓冲区
        self.env_queues = [mp.Queue() for _ in range(self.num_envs)]
        # 计数器使用 multiprocessing.Value，在 spawn 模式下通过参数传入子进程
        # 而不是使用 Manager().Value（后者返回的 ValueProxy 不支持 get_lock）
        self.total_added_step = mp.Value('i', 0)        # 全局step数量（样本数）
        self.total_episodes_counter = mp.Value('i', 0)  # 全局episode计数器
        self.current_buffer_size = mp.Value('i', 0)     # 当前本地缓冲区大小（用于日志）

        # 训练/评估阶段共享标识（5态phase统一控制）
        self.phase_ref = self.manager.Value('i', PHASE_TRAIN_COLLECT)  # 初始为训练收集阶段
        # eval_target_ref 仅用于“当前一轮评估”的目标episode数，由训练线程在触发周期性评估时写入
        self.eval_target_ref = self.manager.Value('i', 0)
        self.eval_collected_ref = self.manager.Value('i', 0)
        self.eval_collected_lock = self.manager.Lock()

        # 共享：每个环境“最近一次 episode 的 target_dist”（用于判定是否所有采集进程都达到 max_target_dist）
        # 注意：target_dist 以采集进程侧 ROS_env.target_dist 为准（期望目标距离上限），不是 actual 终点距离。
        try:
            init_td = float(self._loaded_config.get('init_target_distance', self.init_target_distance))
        except Exception:
            init_td = float(self.init_target_distance)
        self.env_target_dist_list = self.manager.list([init_td for _ in range(self.num_envs)])
        
        
        # 初始化全局统计
        self.global_stats = GlobalStatistics(window_size=self.stats_window_size)
        
        # 初始化最好模型跟踪变量
        # 注意：最好模型保存从“评估阶段”开始启用；训练阶段只打印统计不保存
        self.best_model_enabled = False
        self.best_goal_rate = -1.0  # 最好的成功率（初始化为-1，表示还没有记录）
        self.best_collision_rate = float('inf')  # 最好的碰撞率（初始化为无穷大，表示还没有记录）
        self.check_best_model_ref = self.manager.Value('i', 0)  # 共享变量：标记是否需要检查最好模型（episode编号）
        self.check_best_model_lock = self.manager.Lock()  # 锁，用于保护检查最好模型的逻辑
        self.last_eligibility_warning_episode = 0  # 记录上次打印资格警告的episode编号，避免频繁打印
        
        # 初始化计数器（不再需要同步事件）
        self.init_complete_counter = mp.Value('i', 0)  # 跟踪初始化完成的环境数量
        
        # 配置字典（供采集子进程初始化 ROS_env 等读取）
        self.config = {
            'num_envs': self.num_envs,
            # 关键：bin_num 必须下发给采集子进程，否则子进程会走默认值(72)导致 state_dim/base_state_dim 与主进程不一致
            'bin_num': int(self.bin_num),
            'state_dim': int(self.base_state_dim * (1 + self.state_history_steps) if self.state_history_steps > 0 else self.base_state_dim),
            'base_state_dim': self.base_state_dim,
            'state_history_steps': self.state_history_steps,
            'action_dim': action_dim,
            'max_action': max_action,
            'max_steps_ratio': self.max_steps_ratio,
            'max_steps': self.max_steps,
            'max_steps_min': self.max_steps_min,
            'batch_size': self.batch_size,
            'training_iterations': self.training_iterations,
            'train_n_per_step': self.train_n_per_step,
            'collection_wait_ratio': self.collection_wait_ratio,
            'min_collection_ratio': self.min_collection_ratio,
            'max_rounds': self.max_rounds,
            'save_every': self.save_every,
            'buffer_size': self.buffer_size,
            'enable_stratified_replay': self._loaded_config.get('enable_stratified_replay', False),
            'stratified_replay': self._loaded_config.get('stratified_replay', {}),
            'min_batches_for_training': self._loaded_config.get('min_batches_for_training', 0),  # 训练开始前需要收集的最少batch数量（0表示只要有数据就训练）
            'model_save_dir': self.model_save_dir,
            # 公共日志路径（不区分环境）
            'collect_log_path': self._log_paths['collect_log_path'],
            'train_log_path': self._log_paths['train_log_path'],
            'env_log_path': self._log_paths['env_log_path'],
            'reward_log_path': self._log_paths['reward_log_path'],
            'nodes_log_path': self._log_paths['nodes_log_path'],
            # 每环境日志根目录与时间戳（供采集进程按 env_id 生成独立日志文件）
            # 注意：这里存成字符串，避免在子进程中被当作 PosixPath 调用 .strip() 出错
            'env_logs_dir': str(self.env_logs_dir),
            'log_timestamp': self.timestamp,
            'max_velocity': self.max_velocity,
            'neglect_angle': self.neglect_angle,
            'max_yawrate': self.max_yawrate,
            'scan_range': self.scan_range,
            'max_target_dist': self.max_target_dist,
            'init_target_distance': self.init_target_distance,
            'target_dist_increase': self.target_dist_increase,
            'target_reached_delta': self.target_reached_delta,
            'collision_delta': self.collision_delta,
            'world_size': self.world_size,
            'goals_per_map': self.goals_per_map,
            'obs_min_dist': self.obs_min_dist,
            'obs_num': self.obs_num,
            'costmap_resolution': self.costmap_resolution,
            'obstacle_size': self.obstacle_size,
            'obs_distribution_mode': self.obs_distribution_mode,
            'max_acceleration': self.max_acceleration,
            'max_deceleration': self.max_deceleration,
            'enable_weight_consistency_check': self.enable_weight_consistency_check,
            'max_training_count': self.max_training_count,
            'actor_update_frequency': cfg.get('actor_update_frequency', 1),
            'critic_target_update_frequency': cfg.get('critic_target_update_frequency', 4),
            'hidden_layers': self.hidden_layers,
            'avg_loss_window_size': self.avg_loss_window_size,
            'gpu_id': cfg.get('gpu_id', 0),
            # 周期性评估参数（从 train.yaml 读取）
            'eval_start_after_all_target_dist_max': self._loaded_config.get('eval_start_after_all_target_dist_max', False),
            'eval_every_rounds': int(self._loaded_config.get('eval_every_rounds', 0) or 0),
            'eval_episodes_per_round': int(self._loaded_config.get('eval_episodes_per_round', 0) or 0),
            # 奖励函数参数
            'goal_reward': self.goal_reward,
            'base_collision_penalty': self.base_collision_penalty,
            'angle_base_penalty': self.angle_base_penalty,
            'base_linear_penalty': self.base_linear_penalty,
            'yawrate_penalty_base': self.yawrate_penalty_base,
            'enable_obs_penalty': self.enable_obs_penalty,
            'enable_yawrate_penalty': self.enable_yawrate_penalty,
            'enable_angle_penalty': self.enable_angle_penalty,
            'enable_linear_penalty': self.enable_linear_penalty,
            'enable_step_penalty': self.enable_step_penalty,
            'enable_target_distance_penalty': self.enable_target_distance_penalty,
            # 进度奖惩参数（必须写入config，供采集子进程初始化ROS_env时读取）
            'enable_progress_reward': self.enable_progress_reward,
            'progress_reward_base': self.progress_reward_base,
            'enable_linear_acceleration_oscillation_penalty': self.enable_linear_acceleration_oscillation_penalty,
            'enable_yawrate_oscillation_penalty': self.enable_yawrate_oscillation_penalty,
            # 障碍物距离惩罚参数
            'obs_penalty_threshold': self.obs_penalty_threshold,
            'obs_penalty_base': self.obs_penalty_base,
            'obs_penalty_power': self.obs_penalty_power,
            'min_obs_penalty_threshold': self.min_obs_penalty_threshold,
            'obs_penalty_high_weight': self.obs_penalty_high_weight,
            'obs_penalty_low_weight': self.obs_penalty_low_weight,
            'obs_penalty_middle_ratio': self.obs_penalty_middle_ratio,
            # 终点距离惩罚参数
            'target_distance_penalty_base': self.target_distance_penalty_base,
            # 时间步惩罚参数
            'step_penalty_base': self.step_penalty_base,
            # 震荡惩罚参数
            'linear_acceleration_oscillation_penalty_base': self.linear_acceleration_oscillation_penalty_base,
            'yawrate_oscillation_penalty_base': self.yawrate_oscillation_penalty_base,
            # 时间控制参数
            'sim_time': self.sim_time,
            'step_sleep_time': self.step_sleep_time,
            'reset_step_count': self.reset_step_count,
            # 动作噪声参数
            'action_noise_std': self.action_noise_std,
            # 连通区域选择参数（从 train.yaml 读取）
            'region_select_bias': self.region_select_bias,
            # 奖励归一化参数
            'reward_scale': self.reward_scale,
            # ROS域ID参数（从配置文件读取）
            'start_ros_domain_id': self._loaded_config.get('start_ros_domain_id', 1),
            # 传感器频率限制参数（从配置文件读取）
            'sensor_freq_limit': self._loaded_config.get('sensor_freq_limit', {}),
            # 传感器日志开关（从配置文件读取）
            'sensor_log_enable': self._loaded_config.get('sensor_log_enable', {}),
            # 定位噪声参数（从配置文件读取）
            'localization_noise_stddev': self._loaded_config.get('localization_noise_stddev', 0.0),
        }
        
                # 打印训练配置汇总（包含 YAML 原始项和运行时派生字段）
        print("\n训练配置汇总（含派生参数）:")
        for k, v in self.config.items():
            print(f"  - {k}: {v}")
        print("\n真正的并行多环境训练器初始化完成\n")
    
    def _setup_directories(self):
        """设置目录"""
        try:
            # 创建实时模型保存目录
            self.model_save_dir.mkdir(parents=True, exist_ok=True)
            print(f"模型保存目录已准备: {self.model_save_dir}")
            # 创建最好模型保存目录
            self.best_model_save_dir = self.log_dir / "best_model"
            self.best_model_save_dir.mkdir(parents=True, exist_ok=True)
            print(f"最好模型保存目录已准备: {self.best_model_save_dir}")
            # 创建每环境日志根目录
            self.env_logs_dir.mkdir(parents=True, exist_ok=True)
            print(f"环境日志根目录已准备: {self.env_logs_dir}")
        except PermissionError:
            print(f"错误: 没有权限创建模型保存目录 {self.model_save_dir}")
            raise
        except OSError as e:
            print(f"错误: 无法创建模型保存目录 {self.model_save_dir}")
            raise
    
    def _check_and_save_best_model_by_eval_goal_rate(self, eval_stats):
        """评估阶段：按终点率(goal_rate)保存最好模型（不做训练阶段资格检查）

        规则：
        - goal_rate 更高则更新最好模型
        - goal_rate 相等时，collision_rate 更低则更新（避免并列）
        """
        with self.check_best_model_lock:
            # 一旦出现评估阶段，就启用最好模型保存
            self.best_model_enabled = True

            current_goal_rate = float(eval_stats.get('goal_rate', 0.0) or 0.0)
            current_collision_rate = float(eval_stats.get('collision_rate', float('inf')))

            is_better = False
            if current_goal_rate > self.best_goal_rate:
                is_better = True
            elif current_goal_rate == self.best_goal_rate and current_collision_rate < self.best_collision_rate:
                is_better = True

            if not is_better:
                return

            self.best_goal_rate = current_goal_rate
            self.best_collision_rate = current_collision_rate

            try:
                # 确保最好模型目录存在
                self.best_model_save_dir.mkdir(parents=True, exist_ok=True)
                # 最好模型保存：覆盖写入同名文件，避免生成过多模型文件
                # 输出文件名固定为：
                #   SAC_actor.pth, SAC_critic.pth, SAC_critic_target.pth
                self.model.save(filename="SAC", directory=self.best_model_save_dir)
                if getattr(self, "train_logger", None):
                    self.train_logger.log(f"\n{'='*60}")
                    self.train_logger.log(f"[评估最好模型] 保存最好模型到: {self.best_model_save_dir}")
                    self.train_logger.log(f"[评估最好模型] 当前最好统计: 终点率={self.best_goal_rate:.4f} ({self.best_goal_rate*100:.2f}%), "
                          f"碰撞率={self.best_collision_rate:.4f} ({self.best_collision_rate*100:.2f}%)")
                    self.train_logger.log(f"{'='*60}\n")
                else:
                    print(f"\n{'='*60}")
                    print(f"[评估最好模型] 保存最好模型到: {self.best_model_save_dir}")
                    print(f"[评估最好模型] 当前最好统计: 终点率={self.best_goal_rate:.4f} ({self.best_goal_rate*100:.2f}%), "
                          f"碰撞率={self.best_collision_rate:.4f} ({self.best_collision_rate*100:.2f}%)")
                    print(f"{'='*60}\n")
            except Exception as e:
                msg = f"[ERROR] 警告: [评估最好模型] 保存最好模型失败: {e}"
                if getattr(self, "train_logger", None):
                    self.train_logger.log(msg)
                else:
                    print(msg)
        
    
    def run_training(self):
        """运行真正的并行多环境训练"""
        train_log_path = (self.config.get("train_log_path") or "").strip()
        train_logger = TrainLogger(train_log_path) if train_log_path else None
        self.train_logger = train_logger
        
        collect_log_path = (self.config.get("collect_log_path") or "").strip()
        collect_logger = CollectLogger(collect_log_path) if collect_log_path else None

        # 启动数据收集进程
        collect_processes = []
        for env_id in range(self.num_envs):
            p = mp.Process(
                target=collect_episode_data,
                args=(
                    env_id,
                    self.model_manager.shared_model_dict,
                    self.model_manager.shared_lock,
                    self.env_queues[env_id],
                    self.total_added_step,
                    self.global_stats,
                    self.config,
                    self.init_complete_counter,
                    self.total_episodes_counter,
                    self.model_manager.training_count_ref,
                    self.model_manager.critic_loss_ref,
                    self.model_manager.recent_losses_ref,
                    self.avg_loss_window_size,
                    self.phase_ref,
                    self.eval_target_ref,
                    self.eval_collected_lock,
                    self.eval_collected_ref,
                    self.current_buffer_size,
                    self.check_best_model_ref,
                    self.round_done_counter,
                    self.current_round_ref,
                    self.env_target_dist_list,
                )
            )
            p.start()
            collect_processes.append(p)
        
        # 启动训练线程
        training_thread_obj = threading.Thread(
            target=training_thread,
            args=(
                self.model_manager,
                self.env_queues,
                self.config,
                self.total_added_step,
                self.total_episodes_counter,
                self.current_buffer_size,
                self.round_done_counter,
                self.current_round_ref,
                self.phase_ref,
                self.eval_target_ref,
                self.eval_collected_lock,
                self.eval_collected_ref,
                self.global_stats,
                self.env_target_dist_list,
                self.max_steps,
                self._check_and_save_best_model_by_eval_goal_rate,
                train_logger,
                collect_logger,
            )
        )
        training_thread_obj.daemon = True
        training_thread_obj.start()

        try:
            # 等待训练线程结束（周期性评估已在训练线程内部完成）
            while training_thread_obj.is_alive():
                time.sleep(0.1)
        except KeyboardInterrupt:
            if train_logger:
                train_logger.log("\n收到中断信号，正在停止训练...")
            else:
                print("\n收到中断信号，正在停止训练...")
        finally:
            # 不做“训练结束后一次性评估”的特殊处理；
            # 若最后一轮训练结束时恰好触发周期性评估，则训练线程已完成评估并输出报告。
            if training_thread_obj.is_alive():
                training_thread_obj.join(timeout=5)

            # 通知子进程退出
            self.phase_ref.value = PHASE_STOP
            # 清理进程
            for p in collect_processes:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        p.kill()

            # 保存最终模型
            self.model_save_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(filename="SAC_final", directory=self.model_save_dir)
            if train_logger:
                train_logger.log(f"最终模型已保存到 {self.model_save_dir}")
            else:
                print(f"最终模型已保存到 {self.model_save_dir}")

            self.model_manager.cleanup_temp_files()

            if train_logger:
                train_logger.log("真正的并行训练已停止")
            else:
                print("真正的并行训练已停止")
            if train_logger is not None:
                train_logger.close()
            if collect_logger is not None:
                collect_logger.close()
    
    
def _print_statistics_report(stats, train_logger=None):
    """打印统计报告；若提供 train_logger 则写入训练日志，否则 print。"""
    from datetime import datetime
    current_time = datetime.now()
    def _out(msg):
        if train_logger:
            train_logger.log(msg)
        else:
            print(msg)
    _out(f"\n{'='*60}")
    _out(f"统计报告 - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    _out(f"Episode总数：{stats['total_episodes']}")
    _out(f"统计窗口大小：{stats['window_size']}")
    _out(f"平均奖励: {stats['avg_reward']:.2f}")
    _out(f"成功率: {stats['goal_rate']:.2f} ({stats['goal_rate']*100:.2f}%)")
    _out(f"碰撞率: {stats['collision_rate']:.2f} ({stats['collision_rate']*100:.2f}%)")
    _out(f"超时率: {stats['timeout_rate']:.2f} ({stats['timeout_rate']*100:.2f}%)")
    _out(f"{'='*60}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='并行多环境训练脚本')
    
    # 配置文件参数
    parser.add_argument('--config', type=str, default=None, help='配置文件路径（默认：config/train.yaml）')
    
    # 训练参数（命令行参数可以覆盖配置文件）
    parser.add_argument('--num_envs', type=int, default=None, help='并行环境数量（覆盖配置文件）')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小（覆盖配置文件）')
    parser.add_argument('--training_iterations', type=int, default=None, help='每轮训练迭代次数（覆盖配置文件）')
    parser.add_argument('--max_steps_ratio', type=int, default=None, help='每Episode最大步数比例（覆盖配置文件，0表示使用固定max_steps）')
    parser.add_argument('--max_steps', type=int, default=None, help='当 max_steps_ratio=0 时使用的固定max_steps（覆盖配置文件）')
    parser.add_argument('--max_steps_min', type=int, default=None, help='每Episode最小步数（覆盖配置文件）')
    parser.add_argument('--save_every', type=int, default=None, help='每多少次训练保存一次模型（覆盖配置文件）')
    parser.add_argument('--buffer_size', type=int, default=None, help='重放缓冲区大小（覆盖配置文件）')
    parser.add_argument('--stats_window_size', type=int, default=None, help='统计窗口大小（覆盖配置文件）')
    parser.add_argument('--gpu_id', type=int, default=None, help='使用的GPU编号（覆盖配置文件）')
    
    # 环境参数（命令行参数可以覆盖配置文件）
    parser.add_argument('--max_velocity', type=float, default=None, help='最大速度（覆盖配置文件）')
    parser.add_argument('--neglect_angle', type=int, default=None, help='前方视野左右两边忽略的角度（覆盖配置文件）')
    parser.add_argument('--max_yawrate', type=float, default=None, help='最大偏航率（覆盖配置文件）')
    parser.add_argument('--scan_range', type=int, default=None, help='扫描范围（覆盖配置文件）')
    parser.add_argument('--max_target_dist', type=float, default=None, help='最大目标距离（覆盖配置文件）')
    parser.add_argument('--init_target_distance', type=float, default=None, help='初始目标距离（覆盖配置文件）')
    parser.add_argument('--target_dist_increase', type=float, default=None, help='目标距离增加量（覆盖配置文件）')
    parser.add_argument('--target_reached_delta', type=float, default=None, help='目标到达判断阈值（覆盖配置文件）')
    parser.add_argument('--collision_delta', type=float, default=None, help='碰撞判断阈值（覆盖配置文件）')
    parser.add_argument('--world_size', type=int, default=None, help='世界大小（覆盖配置文件）')
    parser.add_argument('--obs_min_dist', type=float, default=None, help='障碍物圆心最小距离（覆盖配置文件）')
    parser.add_argument('--obs_num', type=int, default=None, help='障碍物数量（覆盖配置文件）')
    
    # 模型参数（命令行参数可以覆盖配置文件）
    parser.add_argument('--model_save_dir', type=str, default=None, help='模型保存目录（覆盖配置文件）')
    parser.add_argument('--model_load_dir', type=str, default=None, help='模型加载目录（覆盖配置文件）')
    parser.add_argument('--load_model', type=str, default=None, help='是否加载已有模型（覆盖配置文件）')
    
    # 并行训练参数（命令行参数可以覆盖配置文件）
    parser.add_argument('--max_training_count', type=int, default=None, help='最大训练次数（覆盖配置文件）')
    parser.add_argument('--actor_update_frequency', type=int, default=None, help='Actor网络更新频率（覆盖配置文件）')
    parser.add_argument('--critic_target_update_frequency', type=int, default=None, help='Critic目标网络更新频率（覆盖配置文件）')
    parser.add_argument('--hidden_layers', type=str, default=None, help='神经网络隐藏层结构（覆盖配置文件，JSON格式，例如：[1024,512]）')
    parser.add_argument('--avg_loss_window_size', type=int, default=None, help='平均损失计算窗口大小（覆盖配置文件，向后兼容参数名）')
    
    # 算法参数（命令行参数可以覆盖配置文件）
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
    parser.add_argument('--reset_step_count', type=int, default=None, help='reset方法中调用step的次数（覆盖配置文件）')
    
    # 动作噪声参数（命令行参数可以覆盖配置文件）
    parser.add_argument('--action_noise_std', type=float, default=None, help='动作噪声标准差（覆盖配置文件）')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 设置多进程启动方法为spawn，解决CUDA多进程问题
    mp.set_start_method('spawn', force=True)
    
    args = parse_args()

    # ==================== 加载配置文件（仅从 train.yaml 读取参数） ====================
    config_path = args.config
    config = load_config(config_path)
    
    # 获取实际使用的配置文件路径用于打印
    actual_config_path = config_path if config_path else (Path(__file__).parent.parent.parent / "config" / "train.yaml")
    print(f"使用配置文件: {actual_config_path}")
    
    # 调试：打印 sensor_log_enable 配置
    if 'sensor_log_enable' in config:
        print(f"调试: sensor_log_enable 存在，类型: {type(config['sensor_log_enable'])}, 值: {config['sensor_log_enable']}")
    else:
        print(f"调试: sensor_log_enable 不存在于配置中")
        print(f"调试: 配置中的所有键: {list(config.keys())}")
    
    # ==================== 设置TURTLEBOT3_MODEL ====================
    # 从配置文件读取turtlebot3_model，用于指定世界模型文件
    # 注意：此参数由启动脚本（start_multi_env_training.sh）设置到环境变量，launch文件会读取环境变量
    # 这里读取并打印用于验证，确保配置正确
    turtlebot3_model = config.get('turtlebot3_model', 'waffle')
    print(f"TURTLEBOT3_MODEL: {turtlebot3_model} (用于加载世界文件: turtlebot3_drl/{turtlebot3_model}.model)")
    print(f"机器人模型固定为: /root/DRL-Robot-Dog-Navigation/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_waffle/model.sdf")

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
    
    # ==================== 从配置文件读取参数（不再从命令行读取） ====================
    # 训练参数
    num_envs = config.get('num_envs', 4)
    batch_size = config.get('batch_size', 40)
    training_iterations = config.get('training_iterations', 200)
    train_n_per_step = config.get('train_n_per_step', 1.0)
    collection_wait_ratio = config.get('collection_wait_ratio', 0.8)
    min_collection_ratio = config.get('min_collection_ratio', 0.25)
    max_steps_ratio = config.get('max_steps_ratio', 0)
    max_steps = config.get('max_steps', 300)
    max_steps_min = config.get('max_steps_min', 50)
    save_every = config.get('save_every', 50)
    buffer_size = config.get('buffer_size', 50000)
    stats_window_size = config.get('stats_window_size', 20)
    max_training_count = config.get('max_training_count', 1000)
    max_rounds = config.get('max_rounds', max_training_count)
    
    # 机器人和环境参数
    max_velocity = config.get('max_velocity', 1.0)
    max_acceleration = config.get('max_acceleration', 5.0)
    max_deceleration = config.get('max_deceleration', -5.0)
    neglect_angle = config.get('neglect_angle', 0)
    max_yawrate = config.get('max_yawrate', 20.0)
    scan_range = config.get('scan_range', 5)
    localization_noise_stddev = config.get('localization_noise_stddev', 0.0)
    max_target_dist = config.get('max_target_dist', 15.0)
    init_target_distance = config.get('init_target_distance', 2.0)
    target_dist_increase = config.get('target_dist_increase', 0.01)
    target_reached_delta = config.get('target_reached_delta', 0.3)
    collision_delta = config.get('collision_delta', 0.25)
    world_size = config.get('world_size', 15)
    goals_per_map = config.get('goals_per_map', 4)
    obs_min_dist = config.get('obs_min_dist', 2)
    obs_num = config.get('obs_num', 20)
    costmap_resolution = config.get('costmap_resolution', 0.3)
    obstacle_size = config.get('obstacle_size', 0.3)
    obs_distribution_mode = config.get('obs_distribution_mode', 'uniform')
    
    # 模型参数（仅来自配置文件）
    base_state_dim = config.get('base_state_dim', 25)
    state_history_steps = config.get('state_history_steps', 0)
    action_dim = config.get('action_dim', 2)
    max_action = config.get('max_action', 1.0)
    gpu_id = config.get('gpu_id', 0)
    
    # 网络结构参数（仅来自配置文件）
    hidden_layers = config.get('hidden_layers', [1024, 512])
    
    # 训练算法参数（仅来自配置文件）
    discount_factor = config.get('discount_factor', 0.99)  # 折扣因子（gamma），从配置文件 train.yaml 读取，统一用于计算总回报和所有 Reward Detail 分项的折扣回报
    actor_update_frequency = config.get('actor_update_frequency', 1)
    critic_target_update_frequency = config.get('critic_target_update_frequency', 4)
    # 平均损失窗口大小（向后兼容参数名 avg_loss_window_size）
    avg_loss_window_size = config.get('avg_loss_window_size', 10)
    
    # 路径参数（仅来自配置文件）
    model_save_dir = config.get('model_save_dir', None)
    # 优先使用 load_path，如果没有则使用 model_load_dir（向后兼容）
    model_load_dir = config.get('load_path', config.get('model_load_dir', None))
    # 优先使用 load_model，如果没有则使用 load_existing_model（向后兼容）
    load_model_str = config.get('load_model', config.get('load_existing_model', True))
    load_model = load_model_str if isinstance(load_model_str, bool) else str(load_model_str).lower() == 'true'
    
    # 奖励函数参数（仅来自 train.yaml，兼容旧字段名）
    goal_reward = config.get('goal_reward', 1000.0)
    base_collision_penalty = config.get('collision_penalty_base', config.get('base_collision_penalty', -1000.0))
    angle_base_penalty = config.get('angle_penalty_base', config.get('angle_base_penalty', 0.0))
    base_linear_penalty = config.get('linear_penalty_base', config.get('base_linear_penalty', -1.0))
    yawrate_penalty_base = config.get('yawrate_penalty_base', 0.0)
    enable_obs_penalty = parse_bool(config.get('enable_obs_penalty', True), True)
    enable_yawrate_penalty = parse_bool(config.get('enable_yawrate_penalty', True), True)
    enable_angle_penalty = parse_bool(config.get('enable_angle_penalty', True), True)
    enable_linear_penalty = parse_bool(config.get('enable_linear_penalty', True), True)
    enable_step_penalty = parse_bool(config.get('enable_step_penalty', False), False)
    enable_target_distance_penalty = parse_bool(config.get('enable_target_distance_penalty', False), False)
    enable_progress_reward = parse_bool(config.get('enable_progress_reward', False), False)
    enable_linear_acceleration_oscillation_penalty = parse_bool(config.get('enable_linear_acceleration_oscillation_penalty', False), False)
    enable_yawrate_oscillation_penalty = parse_bool(config.get('enable_yawrate_oscillation_penalty', False), False)
    progress_reward_base = config.get('progress_reward_base', 1.0)
    
    # 障碍物距离惩罚参数
    obs_penalty_threshold = config.get('obs_penalty_threshold', 1.0)
    obs_penalty_base = config.get('obs_penalty_base', -10.0)
    obs_penalty_power = config.get('obs_penalty_power', 2.0)
    min_obs_penalty_threshold = config.get('min_obs_penalty_threshold', 0.5)
    obs_penalty_high_weight = config.get('obs_penalty_high_weight', 1.0)
    obs_penalty_low_weight = config.get('obs_penalty_low_weight', 0.4)
    obs_penalty_middle_ratio = config.get('obs_penalty_middle_ratio', 0.4)
    
    # 终点距离惩罚参数
    target_distance_penalty_base = config.get('target_distance_penalty_base', -1.0)
    # 时间步惩罚参数
    step_penalty_base = config.get('step_penalty_base', 0.0)
    
    # 震荡惩罚参数
    linear_acceleration_oscillation_penalty_base = config.get('linear_acceleration_oscillation_penalty_base', -1.0)
    yawrate_oscillation_penalty_base = config.get('yawrate_oscillation_penalty_base', -1.0)
    
    # 奖励归一化参数
    reward_scale = config.get('reward_scale', 1.0)  # 奖励缩放因子，用于控制每步整体奖励大小
    
    # 时间控制参数
    sim_time = config.get('sim_time', 0.1)
    step_sleep_time = config.get('step_sleep_time', 0.1)
    reset_step_count = config.get('reset_step_count', 3)
    
    # 动作噪声参数
    action_noise_std = config.get('action_noise_std', 0.2)
    
    # 连通区域选择参数（仅来自配置文件）
    region_select_bias = config.get('region_select_bias', 1.0)
    
    # 最好模型评比资格检查参数（仅来自配置文件）
    
    # 多环境训练参数（仅来自配置文件）
    enable_weight_consistency_check = parse_bool(config.get('enable_weight_consistency_check', False), False)
    
    # 缓冲区参数（仅来自配置文件）
    
    # 日志目录由 ParallelMultiEnvTrainer 内部统一处理：
    # - 使用环境变量 TRAINING_TIMESTAMP（由 start_multi_env_training.sh 在创建日志目录时导出）
    # - 若未设置则 fallback 为当前时间
    # - 目录结构：<multi_env_log_model_dir>/train_<timestamp>/{model,best_model,...}
    # 创建训练器：仅传递配置字典，所有派生/计算参数在 ParallelMultiEnvTrainer 内部完成
    trainer = ParallelMultiEnvTrainer(config=config, config_path=actual_config_path)
    
    # 开始训练
    trainer.run_training()


if __name__ == "__main__":
    main()
