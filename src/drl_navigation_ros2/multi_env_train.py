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
from pathlib import Path
import shutil
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
import yaml
from datetime import datetime
from collections import deque
from typing import Optional

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
from SAC.SAC import SAC
from ros_python import ROS_env
from replay_buffer import ReplayBuffer
import utils

def setup_warning_redirect():
    """将 warnings 输出重定向到日志文件，避免终端刷屏"""
    logfile = os.environ.get("TRAINING_LOGFILE")
    if not logfile:
        return

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        try:
            with open(logfile, "a", encoding="utf-8") as f:
                f.write(warnings.formatwarning(message, category, filename, lineno, line))
        except Exception:
            sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))

    warnings.showwarning = _showwarning
    # 针对 std() 的自由度警告只写入日志，不干扰终端
    warnings.filterwarnings("ignore", message=r"std\(\): degrees of freedom.*")


# 将权重等冗长信息只写入日志文件，避免终端刷屏
def weight_log(msg: str):
    logfile = os.environ.get("TRAINING_LOGFILE")
    if logfile:
        try:
            with open(logfile, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
            return
        except Exception:
            pass
    print(msg)


setup_warning_redirect()


def load_config(config_path=None):
    """加载配置文件，若未指定则使用默认统一配置"""
    default_path = Path(__file__).parent.parent.parent / "config" / "train.yaml"
    path = Path(config_path) if config_path else default_path
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"警告: 未找到配置文件 {path}")
        return {}
    except Exception as e:
        print(f"警告: 读取配置文件失败 {path}: {e}")
        return {}


class SharedReplayBuffer:
    """多进程共享的重放缓冲区 - 真正的并行版本"""
    
    def __init__(self, shared_list, lock, total_added_step, buffer_size,total_episodes_counter):
        self.shared_list = shared_list
        self.lock = lock
        self.total_added_step = total_added_step  # 改名为total_added_step，记录step数量
        self.buffer_size = buffer_size
        self.total_episodes_counter = total_episodes_counter  # 新增：Episode计数器
        
    def add_batch(self, experiences):
        """批量添加经验 - 使用锁自然阻塞，优化性能"""
        if not experiences:
            return
        
        with self.lock:
            # 批量添加，减少锁持有时间
            for exp in experiences:
                self.shared_list.append(exp)
                self.total_added_step.value += 1  # 改名为total_added_step
            
            # 高效清理超出部分：使用切片操作，避免pop(0)的O(n)复杂度
            # 当缓冲区超过限制时，一次性删除多余的元素
            if len(self.shared_list) > self.buffer_size:
                excess_count = len(self.shared_list) - self.buffer_size
                # 使用切片赋值，比逐个pop(0)快得多
                self.shared_list[:] = self.shared_list[excess_count:]
            
            # 只有在数据成功添加到缓冲区时才增加episode计数
            self.total_episodes_counter.value += 1
    
    def drain_batch(self, max_items=None):
        """从共享缓冲区一次性拉取数据到训练线程本地缓存。
        - max_items 为正时：最多拉取 max_items 条
        - max_items 为 None 或 <=0 时：拉取当前全部数据
        返回列表（可能为空）。"""
        with self.lock:
            available = len(self.shared_list)
            if available == 0:
                return []
            if (max_items is None) or (max_items <= 0):
                take = available
            else:
                take = min(max_items, available)
            # 从末尾截取一段数据返回，然后截断共享列表
            # Manager().list 支持切片操作
            data = self.shared_list[-take:]
            del self.shared_list[-take:]
            return list(data)
    
    def sample_batch(self, batch_size):
        """采样批次的实现逻辑 - 在采样前获取lock"""
        with self.lock:
            if len(self.shared_list) < batch_size:
                return None
            
            # 使用numpy随机采样，更高效
            indices = np.random.choice(len(self.shared_list), batch_size, replace=False)
            batch = [self.shared_list[i] for i in indices]
            
            # 转换为numpy数组，确保形状一致
            states = np.array([exp[0] for exp in batch])
            actions = np.array([exp[1] for exp in batch])
            rewards = np.array([exp[2] for exp in batch]).reshape(-1, 1)  # 强制reshape为列向量
            dones = np.array([exp[3] for exp in batch]).reshape(-1, 1)    # 强制reshape为列向量
            next_states = np.array([exp[4] for exp in batch])
            
            return states, actions, rewards, dones, next_states
    
    def size(self):
        """返回缓冲区当前大小"""
        with self.lock:
            return len(self.shared_list)
    
    def get_total_added_step(self):
        """返回总添加的step数量"""
        return self.total_added_step.value
    
    def get_total_episodes_added(self):
        """返回总添加的episode数量"""
        return self.total_episodes_counter.value
    
    
    
class LocalReplayBuffer:
    """训练线程本地的高性能重放缓冲区
    
    设计目标：
    - 避免 Python 层 for 循环逐样本/逐字段拆解
    - 使用 NumPy 连续内存按字段存储，采样时一次性切片
    - 保持与旧版 LocalReplayBuffer.sample_batch 接口兼容
    """
    
    def __init__(self, max_size: int, dtype=np.float32):
        self.max_size = max_size
        self.count = 0          # 当前有效样本数量
        self.write_pos = 0      # 循环写入指针
        self.initialized = False
        self.dtype = dtype
        
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
        
        # 延迟初始化底层数组（只有第一次有数据时才初始化）
        if not self.initialized:
            self._lazy_init(experiences[0])
        
        for exp in experiences:
            state, action, reward, done, next_state = exp
            
            # 转成指定精度（float32 或 float64）
            s = np.asarray(state, dtype=self.dtype)
            ns = np.asarray(next_state, dtype=self.dtype)
            a = np.asarray(action, dtype=self.dtype)
            
            # 写入当前指针位置
            self.states[self.write_pos] = s
            self.next_states[self.write_pos] = ns
            self.actions[self.write_pos] = a
            self.rewards[self.write_pos, 0] = float(reward)
            # done 统一为 0/1 的 float32，便于后续直接转成 torch.Tensor
            self.dones[self.write_pos, 0] = float(done)
            
            # 更新指针和计数
            self.write_pos = (self.write_pos + 1) % self.max_size
            if self.count < self.max_size:
                self.count += 1
    
    def sample_batch(self, batch_size):
        """从本地缓冲区采样一个批次；不足 batch_size 时使用所有可用样本
        
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
            # 随机无放回采样
            indices = np.random.choice(buf_len, actual_batch_size, replace=False)
        
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
    
    def get_statistics(self, use_window=True, max_target_dist=None, report_every=None):
        """获取统计信息
        
        Args:
            use_window: 是否使用滑动窗口统计
            max_target_dist: 最大目标距离，用于检查最好模型评比资格
                           资格条件：过去 report_every 个 episode 的期望目标距离（ros_env.target_dist）
                           都等于 max_target_dist（注意：比较的是期望值，不是实际生成的终点距离）
            report_every: 报告间隔，用于检查最好模型评比资格
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
                    'eligible_for_best_model': False  # 添加资格标志
                }
            
            # 检查是否有资格参与最好模型评比
            # 资格条件：过去 report_every 个 episode 的期望目标距离（ros_env.target_dist）都等于 max_target_dist
            # 注意：这里比较的是生成episode时的期望目标距离，而非实际生成的终点距离（实际距离有随机性）
            eligible_for_best_model = False
            if max_target_dist is not None and report_every is not None and len(self.recent_episodes) >= report_every:
                # 只要遇到一个不等于 max_target_dist（或缺少 target_dist），就立即判定为不具备资格
                recent_eps = list(self.recent_episodes)[-report_every:]
                all_match = True
                for ep in recent_eps:
                    td = ep.get('target_dist', None)
                    # 旧数据可能没有 target_dist，直接视为不具备资格
                    if td is None or abs(td - max_target_dist) >= 1e-10:
                        all_match = False
                        break
                eligible_for_best_model = all_match
            
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
                    'eligible_for_best_model': eligible_for_best_model  # 添加资格标志
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
                    'eligible_for_best_model': eligible_for_best_model  # 添加资格标志
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
        if os.path.exists(self.model_file_path):
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
            except:
                target_device = torch.device('cpu')
            
            # 直接从文件加载到目标设备（GPU或CPU）
            device_model_state = torch.load(self.model_file_path, map_location=target_device)
            
            return device_model_state
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
    


def collect_episode_data(env_id, shared_model_dict, model_lock, experience_queue, total_added_step, global_stats, config, init_complete_counter, total_episodes_counter, training_count_ref, critic_loss_ref, critic_loss_threshold, recent_losses_ref, avg_loss_window_size, phase_ref, eval_target_ref, eval_collected_lock, eval_collected_ref, current_buffer_size_ref, check_best_model_ref):
    """单个环境的数据收集进程 - 真正的并行版本"""
    try:
        print(f"环境 {env_id} 开始初始化...")
        
        # 设置正确的ROS域ID，确保与对应的Gazebo环境通信
        ros_domain_id = env_id + 1  # 环境0使用域1，环境1使用域2...
        os.environ['ROS_DOMAIN_ID'] = str(ros_domain_id)
        print(f"环境 {env_id} 设置ROS_DOMAIN_ID={ros_domain_id}")
        
        print(f"环境 {env_id} 开始初始化ROS环境")
        # 初始化ROS环境（奖励/惩罚参数从 train.yaml 配置读取）
        ros_env = ROS_env(
            env_id=env_id,  # 传递正确的环境ID
            max_velocity=config['max_velocity'],
            init_target_distance=config['init_target_distance'],
            target_dist_increase=config['target_dist_increase'],
            max_target_dist=config['max_target_dist'],
            target_reached_delta=config['target_reached_delta'],
            collision_delta=config['collision_delta'],
            neglect_angle=config['neglect_angle'],
            scan_range=config['scan_range'],
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
        print(f"环境 {env_id} ROS环境初始化完成")
        
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
        base_state_dim = config.get('base_state_dim', config['state_dim'])
        state_dim_effective = base_state_dim * (1 + state_history_steps) if state_history_steps > 0 else base_state_dim
        if config['state_dim'] != state_dim_effective:
            weight_log(f"环境 {env_id} 警告: config.state_dim={config['state_dim']} 与计算值 {state_dim_effective} 不一致，采用 {state_dim_effective}")

        # 创建本地SAC模型实例（子进程版本）
        # 注意：数据收集进程只需要actor模型进行推理，不需要critic和target_critic，以节省显存
        # 数据收集进程不需要加载已有模型，因为会立即同步主进程的模型权重
        local_model = SAC(
            state_dim=state_dim_effective,
            action_dim=config['action_dim'],
            max_action=config['max_action'],
            device=device,
            actor_update_frequency=config.get('actor_update_frequency', 1),
            critic_target_update_frequency=config.get('critic_target_update_frequency', 4),
            hidden_dim=config.get('hidden_dim', 1024),
            hidden_depth=config.get('hidden_depth', 3),
            save_every=0,  # 不自动保存
            load_model=False,  # 数据收集进程不加载模型，使用共享模型状态
            action_noise_std=config.get('action_noise_std', 0.2),
            base_state_dim=base_state_dim,  # 确保prepare_state使用单步状态长度
            actor_only=True,  # 只创建actor模型，不创建critic和target_critic，节省显存
        )
        
        # 验证：确认只创建了actor模型，没有创建critic
        has_actor = hasattr(local_model, 'actor') and local_model.actor is not None
        has_critic = hasattr(local_model, 'critic') and local_model.critic is not None
        has_critic_target = hasattr(local_model, 'critic_target') and local_model.critic_target is not None
        print(f"环境 {env_id} 本地SAC副本创建完成 - Actor: {has_actor}, Critic: {has_critic}, CriticTarget: {has_critic_target}")
        if has_critic or has_critic_target:
            print(f"警告: 环境 {env_id} 在actor_only模式下仍然创建了critic模型！这不应该发生。")
        if not has_actor:
            print(f"错误: 环境 {env_id} 未能创建actor模型！")
        
        # 创建共享模型管理器实例（子进程版本）
        # 从共享字典获取临时目录路径
        shared_temp_dir = shared_model_dict.get('shared_temp_dir', None)
        model_manager = SharedModelManager(local_model, shared_model_dict, model_lock, training_count_ref, critic_loss_ref, recent_losses_ref, is_main_process=False, shared_temp_dir=shared_temp_dir)
        
        # 等待主进程创建模型文件并同步权重
        print(f"环境 {env_id} 等待主进程模型文件...")
        max_wait_attempts = 50  # 最多等待50次，每次0.1秒
        wait_attempt = 0
        
        while wait_attempt < max_wait_attempts:
            latest_model_state = model_manager.get_model_state_dict_for_inference()
            if latest_model_state and len(latest_model_state) > 0:
                weight_log(f"环境 {env_id} 获取到模型权重，键: {list(latest_model_state.keys())}")
                
                # 仅在明确启用时打印子进程模型结构，默认不打印以减少冗余
                if config.get('log_env_model_structure', False):
                    weight_log(f"环境 {env_id} 从文件加载的模型结构概览:")
                    for key, value in latest_model_state.items():
                        if isinstance(value, dict):
                            weight_log(f"  环境 {env_id} {key} 子键: {list(value.keys())}")
                        elif hasattr(value, 'shape'):
                            weight_log(f"  环境 {env_id} {key} 形状: {value.shape}")
                        else:
                            weight_log(f"  环境 {env_id} {key} 类型: {type(value)}")
                
                # 加载模型权重到本地模型
                local_model.load_state_dict(latest_model_state)
                weight_log(f"环境 {env_id} 模型权重同步完成")
                
                # 验证加载后的模型权重
                # 是否进行详细的权重一致性检验由配置控制
                # 默认认为权重是一致的，只有在开启一致性检查并发现不一致时才置为 False
                weights_match = True
                if config.get('enable_weight_consistency_check', False):
                    weight_log(f"环境 {env_id} 验证加载后的模型权重:")
                    # actor_only模式下只返回actor权重
                    local_state_dict_for_log = local_model.state_dict(actor_only=True)
                    for key, value in local_state_dict_for_log.items():
                        if hasattr(value, 'shape') and value.numel() > 0:
                            try:
                                weight_log(f"  环境 {env_id} {key} 形状: {value.shape}, 均值: {value.mean().item():.2f}, 标准差: {value.std().item():.2f}")
                            except Exception:
                                weight_log(f"  环境 {env_id} {key} 形状: {value.shape}, 类型: {type(value)}")
                        elif isinstance(value, dict):
                            weight_log(f"  环境 {env_id} {key} 子键: {list(value.keys())}")
                            for subkey, subvalue in value.items():
                                if hasattr(subvalue, 'shape') and subvalue.numel() > 0:
                                    try:
                                        weight_log(f"    环境 {env_id} {key}.{subkey} 形状: {subvalue.shape}, 均值: {subvalue.mean().item():.2f}, 标准差: {subvalue.std().item():.2f}")
                                    except Exception:
                                        weight_log(f"    环境 {env_id} {key}.{subkey} 形状: {subvalue.shape}, 类型: {type(subvalue)}")
                                else:
                                    weight_log(f"    环境 {env_id} {key}.{subkey} 形状: {subvalue.shape}, 类型: {type(subvalue)}")
                        else:
                            weight_log(f"  环境 {env_id} {key} 值: {value}, 类型: {type(value)}")
                    
                    # 权重一致性验证（只验证actor权重，因为actor_only模式下只有actor）
                    weight_log(f"环境 {env_id} 权重一致性验证:")
                    local_state_dict = local_model.state_dict(actor_only=True)  # actor_only模式下只返回actor权重
                    for key, value in latest_model_state.items():
                        if isinstance(value, dict):  # actor权重是一个字典
                            # 验证actor的每个子权重
                            if key in local_state_dict:
                                for subkey, subvalue in value.items():
                                    if hasattr(subvalue, 'shape') and subvalue.numel() > 0:
                                        local_subvalue = local_state_dict[key][subkey]
                                        if torch.allclose(subvalue, local_subvalue, atol=1e-6):
                                            weight_log(f"  ✓ {key}.{subkey} 权重完全匹配")
                                        else:
                                            weight_log(f"  ✗ {key}.{subkey} 权重不匹配!")
                                            weight_log(f"    文件权重均值: {subvalue.mean().item():.2f}, 本地权重均值: {local_subvalue.mean().item():.2f}")
                                            weight_log(f"    文件权重标准差: {subvalue.std().item():.2f}, 本地权重标准差: {local_subvalue.std().item():.2f}")
                                            weights_match = False
                            else:
                                weight_log(f"  警告: 键 {key} 不在本地模型状态字典中")
                        elif hasattr(value, 'shape') and value.numel() > 0:
                            # 直接是tensor的情况（不应该出现在actor_only模式下）
                            if key in local_state_dict:
                                local_value = local_state_dict[key]
                                if torch.allclose(value, local_value, atol=1e-6):
                                    weight_log(f"  ✓ {key} 权重完全匹配")
                                else:
                                    weight_log(f"  ✗ {key} 权重不匹配!")
                                    weight_log(f"    文件权重均值: {value.mean().item():.2f}, 本地权重均值: {local_value.mean().item():.2f}")
                                    weight_log(f"    文件权重标准差: {value.std().item():.2f}, 本地权重标准差: {local_value.std().item():.2f}")
                                    weights_match = False
                
                # 只有在启用了权重一致性检查时才输出汇总结果
                if config.get('enable_weight_consistency_check', False):
                    if weights_match:
                        weight_log(f"环境 {env_id} ✓ 所有权重完全匹配!")
                    else:
                        weight_log(f"环境 {env_id} ✗ 发现权重不匹配!")
                
                break
            time.sleep(0.1)  # 等待0.1秒
            wait_attempt += 1
        
        if wait_attempt >= max_wait_attempts:
            print(f"环境 {env_id} 警告: 未能获取到有效的模型权重，使用随机初始化的模型")
        
        
        # 增加初始化完成计数器
        with init_complete_counter.get_lock():
            init_complete_counter.value += 1
            print(f"环境 {env_id} 初始化完成，开始持续数据收集...")
        
        total_steps = 0
        last_training_count = 0  # 记录上次更新时的训练次数

        # 历史state设置（长度为 state_history_steps + 1，用于存储 [s_{t-k}, ..., s_t]）
        # 即使 state_history_steps 为 0，也保持长度为 1，此时等价于只使用当前 state
        state_history = deque(maxlen=state_history_steps + 1)
        
        eval_model_synced = False  # 评估阶段只同步一次模型

        global_episode_number = -1  # 安全初始化，防止异常时未定义

        while True:
            current_phase = phase_ref.value  # 0=train, 1=eval, 2=pause, 3=stop
            if current_phase == 3:
                break
            if current_phase == 2:
                time.sleep(0.05)  # 暂停阶段：等待主进程切换
                continue
            is_eval = current_phase == 1

            if is_eval:
                # 评估阶段固定使用最大目标距离，避免沿用训练阶段的递增值
                ros_env.target_dist = config['max_target_dist']

            # 评估阶段：首次进入时同步模型一次，之后不再更新
            if is_eval and not eval_model_synced:
                latest_model_state = model_manager.get_model_state_dict_for_inference()
                if latest_model_state:
                    local_model.load_state_dict(latest_model_state)
                    weight_log(f"环境 {env_id} 评估阶段模型同步完成")
                eval_model_synced = True

            try:
                # 训练阶段的模型同步策略；评估阶段不再更新模型
                if not is_eval:
                    current_training_count = model_manager.get_training_count()
                    should_update_model = False
                    
                    # 使用前X次训练的平均critic损失阈值判断是否同步模型
                    avg_critic_loss = model_manager.get_average_critic_loss(avg_loss_window_size)
                    
                    # 根据平均critic损失阈值判断是否更新模型
                    # 当 critic_loss_threshold < 0 时，表示忽略阈值，始终允许更新
                    if critic_loss_threshold is not None and critic_loss_threshold >= 0:
                        if avg_critic_loss < critic_loss_threshold:
                            should_update_model = True
                            print(f"环境 {env_id} 基于前{avg_loss_window_size}次平均损失阈值更新模型 (平均损失: {avg_critic_loss:.2f} < {critic_loss_threshold})")
                        else:
                            should_update_model = False
                            # 只在第一次或每10次检查时打印，避免日志过多
                            if not hasattr(collect_episode_data, f'no_update_count_{env_id}'):
                                setattr(collect_episode_data, f'no_update_count_{env_id}', 0)
                            count = getattr(collect_episode_data, f'no_update_count_{env_id}')
                            count += 1
                            setattr(collect_episode_data, f'no_update_count_{env_id}', count)
                            
                            if count == 1 or count % 10 == 0:
                                print(f"环境 {env_id} 暂不更新模型 (前{avg_loss_window_size}次平均损失: {avg_critic_loss:.2f} >= {critic_loss_threshold})")
                    else:
                        # 阈值为负，表示不启用损失阈值，始终允许更新
                        should_update_model = True
                        # 为避免刷屏，这里不打印每次更新原因
                    
                    if should_update_model:
                        latest_model_state = model_manager.get_model_state_dict_for_inference()
                        if latest_model_state:
                            local_model.load_state_dict(latest_model_state)
                            last_training_count = current_training_count
                        else:
                            print(f"环境 {env_id} 警告: 未能获取到模型权重")
                
                # 重置环境
                latest_scan, distance, cos, sin, collision, goal, last_action, reward = ros_env.reset()
                state, terminal = local_model.prepare_state(
                    latest_scan, distance, cos, sin, collision, goal, last_action
                )
                
                episode_reward = 0
                episode_steps = 0
                experiences = []
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
                
                # 收集一个episode的数据
                while not terminal and episode_steps < max_steps:
                    # 1. 构造当前输入 x_t（统一从 current_state_with_history 中读取；
                    #    当 state_history_steps 为 0 时，current_state_with_history 仅包含当前 state）
                    model_action = local_model.get_action(current_state_with_history, add_noise=not is_eval)
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
                    total_steps += 1
                    
                    # 准备下一个状态（单步 state）
                    next_state, terminal = local_model.prepare_state(
                        latest_scan, distance, cos, sin, collision, goal, last_action
                    )
                    
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
                
                # 修复风险A：如果是因为timeout退出（步数达到上限且未发生goal/collision），将最后一个transition的done改为True
                # 注意：如果同时达到max_steps和goal/collision，terminal已经是True，episode_ending会正确判断为Goal/Collision，无需修复
                # 这避免了训练时bootstrap错误（time-limit bug）
                if episode_steps >= max_steps and not terminal and len(experiences) > 0:
                    # 将最后一个transition的done标志改为True
                    last_exp = experiences[-1]
                    experiences[-1] = (last_exp[0], last_exp[1], last_exp[2], True, last_exp[4])
                
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
                
                # 过滤规则：步数小于10且碰撞的episode不采用
                should_filter = (episode_steps < 2 and (collision or goal))
                
                if should_filter:
                    continue
                
                # 训练阶段才写入经验队列；评估阶段仅统计
                if not is_eval:
                    # experiences 为本 episode 的一批 transition
                    # 使用队列将经验推送到训练进程，避免使用 Manager().list 大列表切片
                    try:
                        experience_queue.put(experiences)
                    except Exception as e:
                        print(f"环境 {env_id} 推送经验到队列失败: {e}")
                        experiences = []
                
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
                
                # 更新全局统计并获取episode编号（只有通过过滤的episode才会到这里）
                # 传递期望的目标距离（ros_env.target_dist），而非实际生成的目标距离
                global_episode_number = global_stats.add_episode_result(goal, collision, timeout, episode_reward, target_dist=ros_env.target_dist)
                
                # 更新全局样本/episode 计数（用于日志）
                if not is_eval and experiences:
                    with total_added_step.get_lock():
                        total_added_step.value += len(experiences)
                    with total_episodes_counter.get_lock():
                        total_episodes_counter.value += 1
                if is_eval:
                    with eval_collected_lock:
                        eval_collected_ref.value += 1
                
                # 获取当前时间
                current_time = datetime.now()
                
                
                # 读取本 episode 奖励分项（所有开启的项都要打印，保持与单环境一致的明细格式）
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

                # 输出详细的episode信息（含奖励明细），时间戳放在最前面，便于与训练日志对齐
                # target_dist 是配置的目标距离上限，target_distance 是实际生成的终点距离
                episode_info = (
                    f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} 环境 {env_id} "
                    f"Episode: {global_episode_number} "
                    f"Target Distance: {ros_env.target_dist:.2f} (actual: {target_distance:.2f}) Steps: {episode_steps} "
                    f"Buffer(local): {current_buffer_size_ref.value}\n"
                    f"  Reward Detail: {detail_str}"
                )
                print(episode_info)
                
                # 训练阶段按照 report_every 输出；评估阶段统一由主线程在结束后输出
                # 数据收集进程只负责通知主进程，所有统计、打印、评比都由主进程完成
                if (not is_eval) and global_episode_number % config.get('report_every', 20) == 0:
                    # 设置标记，通知主进程进行统计报告打印、资格检查、最好模型评比
                    check_best_model_ref.value = global_episode_number
                
            except Exception as e:
                print(f"环境 {env_id} Episode {global_episode_number} 出错: {e}")
                time.sleep(1)
    except Exception as e:
        print(f"环境 {env_id} 初始化失败: {e}")


def training_thread(model_manager, env_queues, config, total_added_step, total_episodes_counter, current_buffer_size_ref):
    """持续循环训练线程 - 真正的并行版本"""
    try:
        print("训练线程启动，持续循环从缓存区中抽样训练")
        
        training_count = 0
        total_samples_drawn = 0  # 总抽样样本数统计
        # 本地维护actor损失列表，用于计算窗口平均
        recent_actor_losses = []  # 存储最近N次训练的actor损失
        loss_window_size = config.get('loss_window_size', 10)  # 获取窗口大小配置
        
        # 本地缓冲区，用于避免跨进程随机采样的开销
        # 容量与全局buffer_size一致；精度由配置控制（float32 / float64）
        use_float64 = config.get('use_float64_for_buffer', False)
        buffer_dtype = np.float64 if use_float64 else np.float32
        local_buffer = LocalReplayBuffer(config.get('buffer_size', 50000), dtype=buffer_dtype)
        
        while True:
            try:
                # 检查是否达到最大训练次数
                if training_count >= config['max_training_count']:
                    print(f"达到最大训练次数 {config['max_training_count']}，训练完成！")
                    break
                
                # 从各个环境的队列中拉取当前所有可用数据到本地缓冲区，避免使用 Manager().list
                pull_start_time = time.time()
                pulled_any = False
                for q in env_queues:
                    while True:
                        try:
                            experiences = q.get_nowait()
                        except queue.Empty:
                            break
                        if experiences:
                            local_buffer.add_batch(experiences)
                            pulled_any = True
                pull_time = time.time() - pull_start_time if pulled_any else 0.0
                
                # 检查本地缓冲区是否有数据进行训练（即使不足batch_size也进行训练）
                buffer_size = local_buffer.size()
                if buffer_size > 0:
                    
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(
                        f"{current_time} 第{training_count+1}次训练开始："
                        f"当前总episode数: {total_episodes_counter.value}，"
                        f"当前总step数: {total_added_step.value}，"
                        f"当前缓冲区大小: {buffer_size}"
                    )
                    
                    # 直接使用主进程的模型进行训练
                    # sample_batch方法内部会获取lock，训练完成后自动释放
                    training_iterations = config.get('training_iterations', 200)
                    batch_size = config['batch_size']
                    samples_this_training = training_iterations * batch_size  # 本次训练抽样的样本数
                    
                    start_time = time.time()
                    avg_critic_loss, critic_losses, avg_actor_loss, actor_losses = model_manager.model.train(
                        replay_buffer=local_buffer,
                        iterations=training_iterations,
                        batch_size=batch_size
                    )
                    end_time = time.time()
                    
                    # 采样与训练耗时拆分统计
                    sample_time = getattr(model_manager.model, "last_sample_time", None)  # 从本地buffer随机采样的耗时
                    update_time = getattr(model_manager.model, "last_update_time", None)
                    compute_time = None
                    if sample_time is not None and update_time is not None:
                        compute_time = max(update_time - sample_time, 0.0)
                    
                    # 更新总抽样样本数（使用实际成功抽样的样本数）
                    total_samples_drawn += samples_this_training
                    
                    # 计算平均训练次数
                    total_experiences_added = total_added_step.value
                    avg_training_times = total_samples_drawn / max(total_experiences_added, 1)
                    
                    training_count += 1
                    end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    # 为了便于阅读，将训练耗时与统计信息分行打印，并细化耗时统计
                    total_train_time = end_time - start_time

                    # 行1：训练完成基本信息
                    print(f"{end_time_str} 第{training_count}次训练完成")

                    # 行2：抽样与样本统计
                    print(
                        f"  总抽样数: {total_samples_drawn} | 总样本数: {total_experiences_added} "
                        f"| 样本平均抽样次数: {avg_training_times:.2f}"
                    )
                    
                    # 先将当前损失添加到列表中（这样计算窗口平均时会包括当前这次）
                    model_manager.add_critic_loss(avg_critic_loss)
                    
                    # 行3：Critic损失（本次训练的平均 + 窗口平均）
                    window_avg_critic_loss = model_manager.get_average_critic_loss(loss_window_size)
                    if window_avg_critic_loss != float('inf'):
                        print(f"  本次训练的平均critic网络损失: {avg_critic_loss:.2f} | 前{loss_window_size}次训练的平均critic网络损失: {window_avg_critic_loss:.2f}")
                    else:
                        print(f"  本次训练的平均critic网络损失: {avg_critic_loss:.2f}")
                    
                    # 行4：Actor损失（本次训练的平均 + 窗口平均）
                    if avg_actor_loss is not None:
                        # 更新actor损失列表
                        recent_actor_losses.append(avg_actor_loss)
                        # 保持列表大小不超过100（避免内存无限增长）
                        if len(recent_actor_losses) > 100:
                            recent_actor_losses.pop(0)
                        
                        # 计算actor损失的窗口平均（此时已包括当前这次）
                        if len(recent_actor_losses) > 0:
                            if len(recent_actor_losses) >= loss_window_size:
                                window_avg_actor_loss = sum(recent_actor_losses[-loss_window_size:]) / loss_window_size
                            else:
                                window_avg_actor_loss = sum(recent_actor_losses) / len(recent_actor_losses)
                            print(f"  本次训练的平均actor网络损失: {avg_actor_loss:.2f} | 前{loss_window_size}次训练的平均actor网络损失: {window_avg_actor_loss:.2f}")
                        else:
                            print(f"  本次训练的平均actor网络损失: {avg_actor_loss:.2f}")

                    # 行5：本次训练整体耗时（从调用train前到返回后的总墙钟时间）
                    print(f"  训练耗时: {total_train_time:.2f}秒")

                    # 行6+: 采样与前向/反向耗时的细分
                    if sample_time is not None and update_time is not None:
                        # 采样耗时分解为：拉取到本地buffer的耗时 + 从本地buffer随机采样的耗时
                        total_sample_time = pull_time + sample_time
                        print(
                            f"  采样耗时总计: {total_sample_time:.2f}秒 "
                            f"(拉取到本地buffer: {pull_time:.2f}秒, 从本地buffer随机采样: {sample_time:.2f}秒)"
                        )

                        # compute_time 是模型参数更新（前向+反向+优化）的总时间
                        print(f"  前向/反向(网络更新)耗时: {compute_time:.2f}秒")
                    
                    # 增加共享训练次数计数器
                    if hasattr(model_manager, 'training_count_ref') and model_manager.training_count_ref is not None:
                        model_manager.training_count_ref.value = training_count
                    
                    # 更新critic损失到共享变量（已在上面添加到列表，这里只更新当前值）
                    model_manager.update_critic_loss(avg_critic_loss)
                    
                    # 更新当前本地缓冲区大小（供采集进程打印日志使用）
                    current_buffer_size_ref.value = buffer_size
                    
                    # 训练完成后同步模型权重到共享字典
                    model_manager.update_dict_from_model()
                    # print(f"训练完成后更新模型到临时文件: {model_manager.model_file_path}")
                    
                    # 定期保存模型（基于训练次数）
                    if training_count % config.get('save_every', 50) == 0:
                        model_manager.model.save(filename="SAC", directory=config['model_save_dir'])
                else:
                    # 缓冲区数据不足，短暂等待
                    time.sleep(0.1)
                
                # 短暂休息，避免过度占用CPU
                time.sleep(0.01)
                
            except Exception as e:
                print(f"训练线程出错: {e}")
                # 如果是Broken pipe错误，说明主进程可能已经退出
                if "Broken pipe" in str(e) or "Errno 32" in str(e):
                    print("检测到Broken pipe错误，训练线程退出")
                    break
                time.sleep(5)  # 出错后等待更长时间
                
    except Exception as e:
        print(f"训练线程初始化失败: {e}")




class ParallelMultiEnvTrainer:
    """并行多环境训练器"""
    
    def __init__(self, 
                 num_envs=4,
                 batch_size=40,
                 training_iterations=200,
                 state_dim=25,
                 base_state_dim=25,
                 state_history_steps=0,
                 action_dim=2,
                 max_action=1,
                 max_steps_ratio=100,
                 max_steps=3000,
                 max_steps_min=50,
                 device=None,
                 model_save_dir=None,
                 model_load_dir=None,
                 load_model=True,
                 save_every=50,
                 buffer_size=50000,
                 report_every=50,
                 max_velocity=1.0,
                 neglect_angle=0,
                 max_yawrate=20.0,
                 scan_range=5,
                 max_target_dist=15.0,
                 init_target_distance=2.0,
                 target_dist_increase=0.01,
                 target_reached_delta=0.3,
                 collision_delta=0.25,
                 world_size=15,
                 obs_min_dist=2,
                 obs_num=20,
                 costmap_resolution=0.3,
                 obstacle_size=0.3,
                 obs_distribution_mode='uniform',
                 region_select_bias=1.0,
                 is_code_debug=False,
                 max_training_count=1000,
                 stats_window_size=100,
                 gpu_id=0,
                 critic_loss_threshold=100.0,
                 actor_update_frequency=1,
                 critic_target_update_frequency=4,
                 hidden_dim=1024,
                 hidden_depth=3,
                 avg_loss_window_size=5,
                 total_eval_episodes=0,
                 # 奖励函数参数
                 goal_reward=1000.0,
                 base_collision_penalty=-1000.0,
                 angle_base_penalty=0.0,
                 base_linear_penalty=-1.0,
                 yawrate_penalty_base=0.0,
                 # 障碍物距离惩罚参数
                 obs_penalty_threshold=1.0,
                 obs_penalty_base=-10.0,
                 obs_penalty_power=2.0,
                 # 时间控制参数
                 step_sleep_time=0.1,
                 sim_time=0.1,
                 eval_sleep_time=1.0,
                 reset_step_count=3,
                 # 动作噪声参数
                 action_noise_std=0.2,
                 # 调试
                 reward_debug=False,
                 enable_obs_penalty=True,
                 enable_yawrate_penalty=True,
                 enable_angle_penalty=True,
                 enable_linear_penalty=True,
                 enable_target_distance_penalty=False,
                 enable_linear_acceleration_oscillation_penalty=False,
                 enable_yawrate_oscillation_penalty=False,
                 target_distance_penalty_base=-1.0,
                 linear_acceleration_oscillation_penalty_base=-1.0,
                 yawrate_oscillation_penalty_base=-1.0,
                 config_path=None):
        
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
        
        self.num_envs = num_envs
        self.batch_size = batch_size
        self.training_iterations = training_iterations
        self.max_steps_ratio = max_steps_ratio
        self.max_steps = max_steps
        self.max_steps_min = max_steps_min
        self.save_every = save_every
        self.report_every = report_every
        self.max_velocity = max_velocity
        self.neglect_angle = neglect_angle
        self.max_yawrate = max_yawrate
        self.scan_range = scan_range
        self.max_target_dist = max_target_dist
        self.init_target_distance = init_target_distance
        self.target_dist_increase = target_dist_increase
        self.target_reached_delta = target_reached_delta
        self.collision_delta = collision_delta
        self.world_size = world_size
        self.obs_min_dist = obs_min_dist
        self.obs_num = obs_num
        self.costmap_resolution = costmap_resolution
        self.obstacle_size = obstacle_size
        self.obs_distribution_mode = obs_distribution_mode
        self.region_select_bias = region_select_bias
        self.is_code_debug = is_code_debug
        self.max_training_count = max_training_count
        self.critic_loss_threshold = critic_loss_threshold
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        self.avg_loss_window_size = avg_loss_window_size
        self.reward_debug = reward_debug
        self.base_state_dim = base_state_dim
        self.state_history_steps = state_history_steps
        # 奖励函数参数
        self.goal_reward = goal_reward
        self.base_collision_penalty = base_collision_penalty
        self.angle_base_penalty = angle_base_penalty
        self.base_linear_penalty = base_linear_penalty
        self.yawrate_penalty_base = yawrate_penalty_base
        # 奖励/惩罚开关（规范为布尔）
        self.enable_obs_penalty = _to_bool(enable_obs_penalty, True)
        self.enable_yawrate_penalty = _to_bool(enable_yawrate_penalty, True)
        self.enable_angle_penalty = _to_bool(enable_angle_penalty, True)
        self.enable_linear_penalty = _to_bool(enable_linear_penalty, True)
        self.enable_target_distance_penalty = _to_bool(enable_target_distance_penalty, False)
        self.enable_linear_acceleration_oscillation_penalty = _to_bool(enable_linear_acceleration_oscillation_penalty, False)
        self.enable_yawrate_oscillation_penalty = _to_bool(enable_yawrate_oscillation_penalty, False)
        # 障碍物距离惩罚参数
        self.obs_penalty_threshold = obs_penalty_threshold
        self.obs_penalty_base = obs_penalty_base
        self.obs_penalty_power = obs_penalty_power
        # 终点距离惩罚参数
        self.target_distance_penalty_base = target_distance_penalty_base
        # 震荡惩罚参数
        self.linear_acceleration_oscillation_penalty_base = linear_acceleration_oscillation_penalty_base
        self.yawrate_oscillation_penalty_base = yawrate_oscillation_penalty_base
        # 时间控制参数
        self.sim_time = sim_time
        self.step_sleep_time = step_sleep_time
        self.eval_sleep_time = eval_sleep_time
        self.reset_step_count = reset_step_count
        # 动作噪声参数
        self.action_noise_std = action_noise_std
        
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
        
        # 模型配置
        base_save_dir = Path(model_save_dir) if model_save_dir else Path("src/drl_navigation_ros2/models/SAC")
        timestamp_env = os.environ.get('TRAINING_TIMESTAMP')
        timestamp = timestamp_env if timestamp_env else datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_dir = base_save_dir / timestamp
        self.model_load_dir = Path(model_load_dir) if model_load_dir else Path("src/drl_navigation_ros2/models/SAC")
        self.load_model = load_model
        self.config_path = config_path
        
        
        # 创建目录
        self._setup_directories()
        # 保存本次训练使用的配置快照
        if self.config_path and Path(self.config_path).exists():
            try:
                shutil.copy(self.config_path, self.model_save_dir / "config_used.yaml")
            except Exception as e:
                print(f"警告: 复制配置到保存目录失败: {e}")
        
        # 计算实际使用的state_dim，确保与历史state拼接保持一致
        state_dim_effective = self.base_state_dim * (1 + self.state_history_steps) if self.state_history_steps > 0 else self.base_state_dim
        if state_dim != state_dim_effective:
            print(f"警告: 传入state_dim={state_dim} 与计算值 {state_dim_effective} 不一致，已使用 {state_dim_effective}")
        else:
            print(f"使用state_dim={state_dim_effective}")

        print(f"初始化SAC模型...")
        if self.load_model:
            print(f"加载已有模型: {self.model_load_dir}")
        else:
            print(f"创建新的随机初始化模型")
        # 初始化模型
        self.model = SAC(
            state_dim=state_dim_effective,
            action_dim=action_dim,
            max_action=max_action,
            device=self.device,
            actor_update_frequency=actor_update_frequency,
            critic_target_update_frequency=critic_target_update_frequency,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            save_every=0,  # 不自动保存
            load_model=self.load_model,
            save_directory=self.model_save_dir,
            load_directory=self.model_load_dir,
            action_noise_std=self.action_noise_std,
            base_state_dim=base_state_dim,  # 传递base_state_dim给SAC模型
        )
        print(f"初始化SAC模型完成")
        
        # 初始化Manager（必须在SharedModelManager之前）
        self.manager = mp.Manager()
        
        # 创建固定的临时文件目录
        import tempfile
        self.shared_temp_dir = tempfile.mkdtemp(prefix="sac_model_sync_")
        weight_log(f"创建共享临时目录: {self.shared_temp_dir}")
        
        # 创建共享的模型相关对象
        shared_model_dict = self.manager.dict()  # 共享模型权重字典
        model_lock = self.manager.Lock()  # 共享模型锁
        training_count_ref = self.manager.Value('i', 0)  # 共享训练次数计数器
        critic_loss_ref = self.manager.Value('d', float('inf'))  # 共享critic损失
        recent_losses_ref = self.manager.list()  # 共享最近损失列表
        
        # 初始化共享模型管理器
        self.model_manager = SharedModelManager(self.model, shared_model_dict, model_lock, training_count_ref, critic_loss_ref, recent_losses_ref, is_main_process=True, shared_temp_dir=self.shared_temp_dir)
        
        # 立即将主进程的模型权重同步到临时文件
        weight_log("将主进程模型权重同步到临时文件...")
        self.model_manager.update_dict_from_model()
        weight_log("模型权重同步完成")
        
        # 将临时目录路径保存到共享字典，供数据收集进程使用
        self.model_manager.shared_model_dict['shared_temp_dir'] = self.model_manager.temp_dir
        
        # 打印主进程模型结构概要（层级与参数形状）
        def _describe_mlp(input_dim, hidden_dim_val, output_dim, depth):
            layers = [("输入层", input_dim)]
            if depth == 0:
                layers.append(("输出层", output_dim))
                return layers
            for idx in range(depth):
                layers.append((f"隐含层{idx + 1}", hidden_dim_val))
            layers.append(("输出层", output_dim))
            return layers

        weight_log("主进程模型结构概览:")
        # actor 结构: trunk 输出为 action_dim 的均值和对数方差，故输出维度为 2 * action_dim
        actor_layers = _describe_mlp(state_dim, hidden_dim, 2 * action_dim, hidden_depth)
        weight_log(f"  主进程 actor 层级神经元: " + " -> ".join([f"{name}:{size}" for name, size in actor_layers]))
        # critic 结构: Q1/Q2 两个 MLP，输入为 state_dim+action_dim，输出为标量 Q 值
        critic_layers = _describe_mlp(state_dim + action_dim, hidden_dim, 1, hidden_depth)
        weight_log(f"  主进程 critic(Q1/Q2) 层级神经元: " + " -> ".join([f"{name}:{size}" for name, size in critic_layers]))
        for key, value in self.model.state_dict().items():
            if isinstance(value, dict):
                weight_log(f"  主进程 {key} 子键: {list(value.keys())}")
            elif hasattr(value, 'shape'):
                weight_log(f"  主进程 {key} 形状: {value.shape}")
            else:
                weight_log(f"  主进程 {key} 类型: {type(value)}")

        weight_log(f"主进程模型文件路径: {self.model_manager.model_file_path}")
        weight_log(f"主进程临时目录: {self.model_manager.temp_dir}")
        
        # 初始化共享缓冲区相关结构
        # 使用每个环境一个队列传输经验，由训练线程集中维护本地大缓冲区
        self.env_queues = [mp.Queue() for _ in range(num_envs)]
        # 计数器使用 multiprocessing.Value，在 spawn 模式下通过参数传入子进程
        # 而不是使用 Manager().Value（后者返回的 ValueProxy 不支持 get_lock）
        self.total_added_step = mp.Value('i', 0)        # 全局step数量（样本数）
        self.total_episodes_counter = mp.Value('i', 0)  # 全局episode计数器
        self.current_buffer_size = mp.Value('i', 0)     # 当前本地缓冲区大小（用于日志）

        # 训练/评估阶段共享标识
        self.phase_ref = self.manager.Value('i', 0)  # 0=train,1=eval,2=pause,3=stop
        self.eval_target_ref = self.manager.Value('i', total_eval_episodes)
        self.eval_collected_ref = self.manager.Value('i', 0)
        self.eval_collected_lock = self.manager.Lock()
        
        
        # 初始化全局统计
        self.global_stats = GlobalStatistics(window_size=stats_window_size)
        
        # 初始化最好模型跟踪变量
        self.best_goal_rate = -1.0  # 最好的成功率（初始化为-1，表示还没有记录）
        self.best_collision_rate = float('inf')  # 最好的碰撞率（初始化为无穷大，表示还没有记录）
        self.check_best_model_ref = self.manager.Value('i', 0)  # 共享变量：标记是否需要检查最好模型（episode编号）
        self.check_best_model_lock = self.manager.Lock()  # 锁，用于保护检查最好模型的逻辑
        
        # 初始化计数器（不再需要同步事件）
        self.init_complete_counter = mp.Value('i', 0)  # 跟踪初始化完成的环境数量
        
        # 配置字典
        self.config = {
            'num_envs': num_envs,
            'state_dim': state_dim_effective,
            'base_state_dim': base_state_dim,  # 添加base_state_dim到配置
            'state_history_steps': state_history_steps,  # 添加state_history_steps到配置
            'action_dim': action_dim,
            'max_action': max_action,
            'max_steps_ratio': max_steps_ratio,
            'max_steps': self.max_steps,
            'max_steps_min': max_steps_min,
            'batch_size': batch_size,
            'training_iterations': training_iterations,
            'save_every': save_every,
            'report_every': report_every,
            'model_save_dir': self.model_save_dir,
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
            'costmap_resolution': costmap_resolution,
            'obstacle_size': obstacle_size,
            'obs_distribution_mode': obs_distribution_mode,
            'is_code_debug': is_code_debug,
            'max_training_count': max_training_count,
            'critic_loss_threshold': critic_loss_threshold,
            'actor_update_frequency': actor_update_frequency,
            'critic_target_update_frequency': critic_target_update_frequency,
            'hidden_dim': hidden_dim,
            'hidden_depth': hidden_depth,
            'avg_loss_window_size': avg_loss_window_size,
            'gpu_id': gpu_id,  # 添加GPU ID到配置中
            'reward_debug': self.reward_debug,
            'total_eval_episodes': total_eval_episodes,
            # 奖励函数参数
            'goal_reward': goal_reward,
            'base_collision_penalty': base_collision_penalty,
            'angle_base_penalty': angle_base_penalty,
            'base_linear_penalty': base_linear_penalty,
            'yawrate_penalty_base': self.yawrate_penalty_base,
            'enable_obs_penalty': self.enable_obs_penalty,
            'enable_yawrate_penalty': self.enable_yawrate_penalty,
            'enable_angle_penalty': self.enable_angle_penalty,
            'enable_linear_penalty': self.enable_linear_penalty,
            'enable_target_distance_penalty': self.enable_target_distance_penalty,
            'enable_linear_acceleration_oscillation_penalty': self.enable_linear_acceleration_oscillation_penalty,
            'enable_yawrate_oscillation_penalty': self.enable_yawrate_oscillation_penalty,
            # 障碍物距离惩罚参数
            'obs_penalty_threshold': obs_penalty_threshold,
            'obs_penalty_base': obs_penalty_base,
            'obs_penalty_power': obs_penalty_power,
            # 终点距离惩罚参数
            'target_distance_penalty_base': self.target_distance_penalty_base,
            # 震荡惩罚参数
            'linear_acceleration_oscillation_penalty_base': self.linear_acceleration_oscillation_penalty_base,
            'yawrate_oscillation_penalty_base': self.yawrate_oscillation_penalty_base,
            # 时间控制参数
            'sim_time': self.sim_time,
            'step_sleep_time': step_sleep_time,
            'eval_sleep_time': eval_sleep_time,
            'reset_step_count': reset_step_count,
            # 动作噪声参数
            'action_noise_std': action_noise_std,
            # 连通区域选择参数（从 train.yaml 读取）
            'region_select_bias': self.region_select_bias,
        }
        
        # 初始化信息在 shell 脚本中已经完整打印过，这里只打一行简要提示，避免重复参数配置
        weight_log("真正的并行多环境训练器初始化完成")
        weight_log("")
    
    def _setup_directories(self):
        """设置目录"""
        try:
            self.model_save_dir.mkdir(parents=True, exist_ok=True)
            print(f"模型保存目录已准备: {self.model_save_dir}")
            # 创建best模型保存目录
            self.best_model_save_dir = Path(str(self.model_save_dir) + "_best")
            self.best_model_save_dir.mkdir(parents=True, exist_ok=True)
            print(f"最好模型保存目录已准备: {self.best_model_save_dir}")
        except PermissionError:
            print(f"错误: 没有权限创建模型保存目录 {self.model_save_dir}")
            raise
        except OSError as e:
            print(f"错误: 无法创建模型保存目录 {self.model_save_dir}")
            raise
    
    def _check_and_save_best_model(self, stats):
        """检查当前统计信息是否更好，如果是则保存最好模型
        
        Args:
            stats: 统计信息字典，包含 goal_rate, collision_rate, eligible_for_best_model 等
        """
        with self.check_best_model_lock:
            # 首先检查是否有资格参与最好模型评比
            eligible_for_best_model = stats.get('eligible_for_best_model', False)
            if not eligible_for_best_model:
                # 没有资格，不进行评比
                return
            
            current_goal_rate = stats.get('goal_rate', 0.0)
            current_collision_rate = stats.get('collision_rate', float('inf'))
            
            # 判断是否更好：成功率更高，或成功率相等但碰撞率更低
            is_better = False
            if current_goal_rate > self.best_goal_rate:
                is_better = True
            elif current_goal_rate == self.best_goal_rate and current_collision_rate < self.best_collision_rate:
                is_better = True
            
            if is_better:
                # 更新最好统计信息
                self.best_goal_rate = current_goal_rate
                self.best_collision_rate = current_collision_rate
                
                # 保存最好模型
                try:
                    self.model.save(filename="SAC", directory=self.best_model_save_dir)
                    print(f"\n{'='*60}")
                    print(f"保存最好模型到: {self.best_model_save_dir}")
                    print(f"当前最好统计: 成功率={self.best_goal_rate:.4f} ({self.best_goal_rate*100:.2f}%), "
                          f"碰撞率={self.best_collision_rate:.4f} ({self.best_collision_rate*100:.2f}%)")
                    print(f"{'='*60}\n")
                except Exception as e:
                    print(f"警告: 保存最好模型失败: {e}")
        
    
    def run_training(self):
        """运行真正的并行多环境训练"""
        
        # 创建进程间通信队列
        
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
                    self.critic_loss_threshold,
                    self.model_manager.recent_losses_ref,
                    self.avg_loss_window_size,
                    self.phase_ref,
                    self.eval_target_ref,
                    self.eval_collected_lock,
                    self.eval_collected_ref,
                    self.current_buffer_size,
                    self.check_best_model_ref,
                )
            )
            p.start()
            collect_processes.append(p)
        
        # 启动训练线程
        training_thread_obj = threading.Thread(
            target=training_thread,
            args=(self.model_manager, self.env_queues, self.config, self.total_added_step, self.total_episodes_counter, self.current_buffer_size)
        )
        training_thread_obj.daemon = True
        training_thread_obj.start()
        
        try:
            last_checked_episode = 0  # 记录上次检查的episode编号，避免重复检查
            while True:
                # 检查训练线程是否还在运行
                if not training_thread_obj.is_alive():
                    break
                
                # 检查是否需要打印统计报告和评比最好模型
                # 数据收集进程在每 report_every 个 episode 时设置标记，通知主进程
                current_check_episode = self.check_best_model_ref.value
                if current_check_episode > last_checked_episode:
                    # 获取当前统计信息，传递 max_target_dist 和 report_every 用于资格检查
                    stats = self.global_stats.get_statistics(
                        max_target_dist=self.config.get('max_target_dist'),
                        report_every=self.config.get('report_every', 20)
                    )
                    # 打印统计报告（主进程统一负责）
                    _print_statistics_report(stats)
                    # 检查并保存最好模型（内部会检查资格）
                    self._check_and_save_best_model(stats)
                    last_checked_episode = current_check_episode
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n收到中断信号，正在停止训练...")
        
        finally:
            # 如果需要评估阶段
            total_eval_episodes = self.config.get('total_eval_episodes', 0)
            if training_thread_obj.is_alive():
                training_thread_obj.join(timeout=5)

            if total_eval_episodes > 0:
                print(f"训练完成，进入评估阶段，总计需要 {total_eval_episodes} 个episode")
                # 先暂停收集进程，等待训练尾巴收敛
                self.phase_ref.value = 2  # pause
                pause_seconds = max(int(self.max_steps / 10), 1)
                time.sleep(pause_seconds)

                # 切换到评估阶段
                self.phase_ref.value = 1
                self.eval_collected_ref.value = 0
                self.global_stats.reset()

                # 等待评估完成
                while True:
                    with self.eval_collected_lock:
                        current_eval = self.eval_collected_ref.value
                    if current_eval >= total_eval_episodes:
                        break
                    time.sleep(0.5)

                print("评估阶段完成，准备输出统计报告")
                stats = self.global_stats.get_statistics(use_window=False)
                _print_statistics_report(stats)
            else:
                print("训练完成，未配置评估阶段。")

            # 通知子进程退出
            self.phase_ref.value = 3
            # 清理进程
            for p in collect_processes:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        p.kill()
            
            # 保存最终模型
            self.model.save(filename="SAC", directory=self.model_save_dir)
            print(f"最终模型已保存到 {self.model_save_dir}")
            
            # 清理临时文件
            self.model_manager.cleanup_temp_files()
            
            print("真正的并行训练已停止")
    
    
def _print_statistics_report(stats):
    """打印统计报告"""
    from datetime import datetime
    current_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"统计报告 - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Episode总数：{stats['total_episodes']}")
    print(f"统计窗口大小：{stats['window_size']}")
    print(f"平均奖励: {stats['avg_reward']:.2f}")
    print(f"成功率: {stats['goal_rate']:.2f} ({stats['goal_rate']*100:.2f}%)")
    print(f"碰撞率: {stats['collision_rate']:.2f} ({stats['collision_rate']*100:.2f}%)")
    print(f"超时率: {stats['timeout_rate']:.2f} ({stats['timeout_rate']*100:.2f}%)")
    print(f"{'='*60}")


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
    parser.add_argument('--report_every', type=int, default=None, help='每多少个episode输出一次统计报告（覆盖配置文件）')
    parser.add_argument('--stats_window_size', type=int, default=None, help='统计窗口大小（覆盖配置文件）')
    parser.add_argument('--gpu_id', type=int, default=None, help='使用的GPU编号（覆盖配置文件）')
    parser.add_argument('--total_eval_episodes', type=int, default=None, help='评估阶段采集的episode数量（覆盖配置文件）')
    
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
    
    # 调试参数（命令行参数可以覆盖配置文件）
    parser.add_argument('--is_code_debug', type=str, default=None, help='是否为调试代码（覆盖配置文件）')
    
    # 模型参数（命令行参数可以覆盖配置文件）
    parser.add_argument('--model_save_dir', type=str, default=None, help='模型保存目录（覆盖配置文件）')
    parser.add_argument('--model_load_dir', type=str, default=None, help='模型加载目录（覆盖配置文件）')
    parser.add_argument('--load_model', type=str, default=None, help='是否加载已有模型（覆盖配置文件）')
    
    # 并行训练参数（命令行参数可以覆盖配置文件）
    parser.add_argument('--max_training_count', type=int, default=None, help='最大训练次数（覆盖配置文件）')
    parser.add_argument('--critic_loss_threshold', type=float, default=None, help='critic损失阈值（覆盖配置文件）')
    parser.add_argument('--actor_update_frequency', type=int, default=None, help='Actor网络更新频率（覆盖配置文件）')
    parser.add_argument('--critic_target_update_frequency', type=int, default=None, help='Critic目标网络更新频率（覆盖配置文件）')
    parser.add_argument('--hidden_dim', type=int, default=None, help='神经网络隐藏层维度（覆盖配置文件）')
    parser.add_argument('--hidden_depth', type=int, default=None, help='神经网络隐藏层深度（覆盖配置文件）')
    parser.add_argument('--avg_loss_window_size', type=int, default=None, help='平均损失计算窗口大小（覆盖配置文件，向后兼容参数名）')
    parser.add_argument('--loss_window_size', type=int, default=None, help='训练损失窗口大小（覆盖配置文件）')
    
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
    # 设置多进程启动方法为spawn，解决CUDA多进程问题
    mp.set_start_method('spawn', force=True)
    
    args = parse_args()
    
    # ==================== 加载配置文件 ====================
    config_path_env = os.environ.get('MULTI_ENV_TRAIN_CONFIG_PATH', args.config)
    config = load_config(config_path_env)
    
    # 获取实际使用的配置文件路径用于打印
    actual_config_path = config_path_env if config_path_env else (Path(__file__).parent.parent.parent / "config" / "train.yaml")
    print(f"使用配置文件: {actual_config_path}")

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
    
    # ==================== 从配置文件读取参数（命令行参数优先） ====================
    # 训练参数
    num_envs = args.num_envs if args.num_envs is not None else config.get('num_envs', 4)
    batch_size = args.batch_size if hasattr(args, 'batch_size') and args.batch_size is not None else config.get('batch_size', 40)
    training_iterations = args.training_iterations if hasattr(args, 'training_iterations') and args.training_iterations is not None else config.get('training_iterations', 200)
    max_steps_ratio = args.max_steps_ratio if hasattr(args, 'max_steps_ratio') and args.max_steps_ratio is not None else config.get('max_steps_ratio', 0)
    max_steps = args.max_steps if hasattr(args, 'max_steps') and args.max_steps is not None else config.get('max_steps', 300)
    max_steps_min = args.max_steps_min if hasattr(args, 'max_steps_min') and args.max_steps_min is not None else config.get('max_steps_min', 50)
    save_every = args.save_every if hasattr(args, 'save_every') and args.save_every is not None else config.get('save_every', 50)
    buffer_size = args.buffer_size if hasattr(args, 'buffer_size') and args.buffer_size is not None else config.get('buffer_size', 50000)
    base_report_every = args.report_every if hasattr(args, 'report_every') and args.report_every is not None else config.get('report_every', 20)
    # report_every 由配置决定
    report_every = base_report_every
    # stats_window_size 始终与 report_every 保持一致，忽略配置文件中的 stats_window_size
    stats_window_size = report_every
    # 如果配置文件中显式设置了 stats_window_size，发出警告
    if config.get('stats_window_size') is not None and config.get('stats_window_size') != report_every:
        print(f"警告: 配置文件中的 stats_window_size ({config.get('stats_window_size')}) 将被忽略，将使用 report_every ({report_every}) 的值")
    max_training_count = args.max_training_count if hasattr(args, 'max_training_count') and args.max_training_count is not None else config.get('max_training_count', 1000)
    total_eval_episodes = args.total_eval_episodes if hasattr(args, 'total_eval_episodes') and args.total_eval_episodes is not None else config.get('total_eval_episodes', 0)
    
    # 机器人和环境参数
    max_velocity = args.max_velocity if hasattr(args, 'max_velocity') and args.max_velocity is not None else config.get('max_velocity', 1.0)
    neglect_angle = args.neglect_angle if hasattr(args, 'neglect_angle') and args.neglect_angle is not None else config.get('neglect_angle', 0)
    max_yawrate = args.max_yawrate if hasattr(args, 'max_yawrate') and args.max_yawrate is not None else config.get('max_yawrate', 20.0)
    scan_range = args.scan_range if hasattr(args, 'scan_range') and args.scan_range is not None else config.get('scan_range', 5)
    max_target_dist = args.max_target_dist if hasattr(args, 'max_target_dist') and args.max_target_dist is not None else config.get('max_target_dist', 15.0)
    init_target_distance = args.init_target_distance if hasattr(args, 'init_target_distance') and args.init_target_distance is not None else config.get('init_target_distance', 2.0)
    target_dist_increase = args.target_dist_increase if hasattr(args, 'target_dist_increase') and args.target_dist_increase is not None else config.get('target_dist_increase', 0.01)
    target_reached_delta = args.target_reached_delta if hasattr(args, 'target_reached_delta') and args.target_reached_delta is not None else config.get('target_reached_delta', 0.3)
    collision_delta = args.collision_delta if hasattr(args, 'collision_delta') and args.collision_delta is not None else config.get('collision_delta', 0.25)
    world_size = args.world_size if hasattr(args, 'world_size') and args.world_size is not None else config.get('world_size', 15)
    obs_min_dist = args.obs_min_dist if hasattr(args, 'obs_min_dist') and args.obs_min_dist is not None else config.get('obs_min_dist', 2)
    obs_num = args.obs_num if hasattr(args, 'obs_num') and args.obs_num is not None else config.get('obs_num', 20)
    costmap_resolution = config.get('costmap_resolution', 0.3)
    obstacle_size = config.get('obstacle_size', 0.3)
    obs_distribution_mode = config.get('obs_distribution_mode', 'uniform')
    
    # 模型参数
    base_state_dim = config.get('base_state_dim', 25)
    state_history_steps = config.get('state_history_steps', 0)
    
    # 动态计算state_dim
    if state_history_steps > 0:
        calculated_state_dim = base_state_dim * (1 + state_history_steps)
        print(f"启用历史state模式: base_state_dim={base_state_dim}, state_history_steps={state_history_steps}, 最终state_dim={calculated_state_dim}")
    else:
        calculated_state_dim = base_state_dim
        print(f"未启用历史state模式: state_dim={calculated_state_dim}")
    
    # 命令行参数可以覆盖配置文件中的state_dim（如果指定了）
    state_dim = args.state_dim if hasattr(args, 'state_dim') and args.state_dim is not None else calculated_state_dim
    
    action_dim = args.action_dim if hasattr(args, 'action_dim') and args.action_dim is not None else config.get('action_dim', 2)
    max_action = args.max_action if hasattr(args, 'max_action') and args.max_action is not None else config.get('max_action', 1.0)
    gpu_id = args.gpu_id if hasattr(args, 'gpu_id') and args.gpu_id is not None else config.get('gpu_id', 0)
    
    # 网络结构参数
    hidden_dim = args.hidden_dim if hasattr(args, 'hidden_dim') and args.hidden_dim is not None else config.get('hidden_dim', 1024)
    hidden_depth = args.hidden_depth if hasattr(args, 'hidden_depth') and args.hidden_depth is not None else config.get('hidden_depth', 3)
    
    # 训练算法参数
    critic_loss_threshold = args.critic_loss_threshold if hasattr(args, 'critic_loss_threshold') and args.critic_loss_threshold is not None else config.get('critic_loss_threshold', 100.0)
    actor_update_frequency = args.actor_update_frequency if hasattr(args, 'actor_update_frequency') and args.actor_update_frequency is not None else config.get('actor_update_frequency', 1)
    critic_target_update_frequency = args.critic_target_update_frequency if hasattr(args, 'critic_target_update_frequency') and args.critic_target_update_frequency is not None else config.get('critic_target_update_frequency', 4)
    # 支持向后兼容：优先使用loss_window_size，如果没有则使用avg_loss_window_size
    loss_window_size = config.get('loss_window_size', None) or config.get('avg_loss_window_size', 10)
    # 命令行参数可以覆盖配置文件
    if hasattr(args, 'loss_window_size') and args.loss_window_size is not None:
        loss_window_size = args.loss_window_size
    elif hasattr(args, 'avg_loss_window_size') and args.avg_loss_window_size is not None:
        loss_window_size = args.avg_loss_window_size
    # 为了向后兼容，同时设置avg_loss_window_size
    avg_loss_window_size = loss_window_size
    
    # 路径参数
    model_save_dir = args.model_save_dir if hasattr(args, 'model_save_dir') and args.model_save_dir is not None else config.get('model_save_dir', None)
    # 优先使用 load_path，如果没有则使用 model_load_dir（向后兼容）
    model_load_dir = args.model_load_dir if hasattr(args, 'model_load_dir') and args.model_load_dir is not None else config.get('load_path', config.get('model_load_dir', None))
    # 优先使用 load_model，如果没有则使用 load_existing_model（向后兼容）
    load_model_str = args.load_model if hasattr(args, 'load_model') and args.load_model is not None else config.get('load_model', config.get('load_existing_model', True))
    load_model = load_model_str if isinstance(load_model_str, bool) else load_model_str.lower() == 'true'
    
    # 奖励函数参数（从 train.yaml 读取，兼容旧字段名）
    goal_reward = args.goal_reward if hasattr(args, 'goal_reward') and args.goal_reward is not None else config.get('goal_reward', 1000.0)
    base_collision_penalty = args.base_collision_penalty if hasattr(args, 'base_collision_penalty') and args.base_collision_penalty is not None else config.get('collision_penalty_base', config.get('base_collision_penalty', -1000.0))
    angle_base_penalty = args.angle_base_penalty if hasattr(args, 'angle_base_penalty') and args.angle_base_penalty is not None else config.get('angle_penalty_base', config.get('angle_base_penalty', 0.0))
    base_linear_penalty = args.base_linear_penalty if hasattr(args, 'base_linear_penalty') and args.base_linear_penalty is not None else config.get('linear_penalty_base', config.get('base_linear_penalty', -1.0))
    yawrate_penalty_base = config.get('yawrate_penalty_base', 0.0)
    enable_obs_penalty = parse_bool(config.get('enable_obs_penalty', True), True)
    enable_yawrate_penalty = parse_bool(config.get('enable_yawrate_penalty', True), True)
    enable_angle_penalty = parse_bool(config.get('enable_angle_penalty', True), True)
    enable_linear_penalty = parse_bool(config.get('enable_linear_penalty', True), True)
    enable_target_distance_penalty = parse_bool(config.get('enable_target_distance_penalty', False), False)
    enable_linear_acceleration_oscillation_penalty = parse_bool(config.get('enable_linear_acceleration_oscillation_penalty', False), False)
    enable_yawrate_oscillation_penalty = parse_bool(config.get('enable_yawrate_oscillation_penalty', False), False)
    
    # 障碍物距离惩罚参数
    obs_penalty_threshold = args.obs_penalty_threshold if hasattr(args, 'obs_penalty_threshold') and args.obs_penalty_threshold is not None else config.get('obs_penalty_threshold', 1.0)
    obs_penalty_base = args.obs_penalty_base if hasattr(args, 'obs_penalty_base') and args.obs_penalty_base is not None else config.get('obs_penalty_base', -10.0)
    obs_penalty_power = args.obs_penalty_power if hasattr(args, 'obs_penalty_power') and args.obs_penalty_power is not None else config.get('obs_penalty_power', 2.0)
    
    # 终点距离惩罚参数
    target_distance_penalty_base = config.get('target_distance_penalty_base', -1.0)
    
    # 震荡惩罚参数
    linear_acceleration_oscillation_penalty_base = config.get('linear_acceleration_oscillation_penalty_base', -1.0)
    yawrate_oscillation_penalty_base = config.get('yawrate_oscillation_penalty_base', -1.0)
    
    # 时间控制参数
    sim_time = config.get('sim_time', 0.1)
    step_sleep_time = args.step_sleep_time if hasattr(args, 'step_sleep_time') and args.step_sleep_time is not None else config.get('step_sleep_time', 0.1)
    eval_sleep_time = args.eval_sleep_time if hasattr(args, 'eval_sleep_time') and args.eval_sleep_time is not None else config.get('eval_sleep_time', 1.0)
    reset_step_count = args.reset_step_count if hasattr(args, 'reset_step_count') and args.reset_step_count is not None else config.get('reset_step_count', 3)
    
    # 动作噪声参数
    action_noise_std = args.action_noise_std if hasattr(args, 'action_noise_std') and args.action_noise_std is not None else config.get('action_noise_std', 0.2)
    
    # 调试参数
    is_code_debug_str = args.is_code_debug if hasattr(args, 'is_code_debug') else config.get('is_code_debug', False)
    is_code_debug = is_code_debug_str if isinstance(is_code_debug_str, bool) else is_code_debug_str.lower() == 'true'
    reward_debug = config.get('reward_debug', False)
    
    # 连通区域选择参数（仅来自配置文件）
    region_select_bias = config.get('region_select_bias', 1.0)
    
    # 创建训练器
    trainer = ParallelMultiEnvTrainer(
        num_envs=num_envs,
        batch_size=batch_size,
        training_iterations=training_iterations,
        max_steps_ratio=max_steps_ratio,
        max_steps=max_steps,
        max_steps_min=max_steps_min,
        save_every=save_every,
        buffer_size=buffer_size,
        report_every=report_every,
        max_velocity=max_velocity,
        neglect_angle=neglect_angle,
        max_yawrate=max_yawrate,
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
        region_select_bias=region_select_bias,
        is_code_debug=is_code_debug,
        state_dim=state_dim,
        base_state_dim=base_state_dim,  # 添加base_state_dim参数
        state_history_steps=state_history_steps,  # 添加state_history_steps参数
        action_dim=action_dim,
        max_action=max_action,
        model_save_dir=model_save_dir,
        model_load_dir=model_load_dir,
        load_model=load_model,
        max_training_count=max_training_count,
        stats_window_size=stats_window_size,
        gpu_id=gpu_id,
        critic_loss_threshold=critic_loss_threshold,
        actor_update_frequency=actor_update_frequency,
        critic_target_update_frequency=critic_target_update_frequency,
        hidden_dim=hidden_dim,
        hidden_depth=hidden_depth,
        avg_loss_window_size=avg_loss_window_size,
        goal_reward=goal_reward,
        base_collision_penalty=base_collision_penalty,
        angle_base_penalty=angle_base_penalty,
        base_linear_penalty=base_linear_penalty,
        yawrate_penalty_base=yawrate_penalty_base,
        enable_obs_penalty=enable_obs_penalty,
        enable_yawrate_penalty=enable_yawrate_penalty,
        enable_angle_penalty=enable_angle_penalty,
        enable_linear_penalty=enable_linear_penalty,
        enable_target_distance_penalty=enable_target_distance_penalty,
        enable_linear_acceleration_oscillation_penalty=enable_linear_acceleration_oscillation_penalty,
        enable_yawrate_oscillation_penalty=enable_yawrate_oscillation_penalty,
        obs_penalty_threshold=obs_penalty_threshold,
        obs_penalty_base=obs_penalty_base,
        obs_penalty_power=obs_penalty_power,
        target_distance_penalty_base=target_distance_penalty_base,
        linear_acceleration_oscillation_penalty_base=linear_acceleration_oscillation_penalty_base,
        yawrate_oscillation_penalty_base=yawrate_oscillation_penalty_base,
        sim_time=sim_time,
        step_sleep_time=step_sleep_time,
        eval_sleep_time=eval_sleep_time,
        reset_step_count=reset_step_count,
        action_noise_std=action_noise_std,
        reward_debug=reward_debug,
        total_eval_episodes=total_eval_episodes,
        config_path=actual_config_path,
    )
    
    # 开始训练
    trainer.run_training()


if __name__ == "__main__":
    main()
