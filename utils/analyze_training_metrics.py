#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练指标分析与可视化工具
功能：
1. 读取每次训练的平均critic和actor网络损失值，根据窗口值绘制曲线图
2. 统计所有episode结果，根据窗口值绘制成功率、碰撞率、超时率的曲线图
3. 读取每次训练的样本平均抽样次数，根据窗口值绘制曲线图
4. 根据窗口值绘制reward detail中各项的曲线图
"""

# ============================================================================
# 配置参数 - 可在此处修改默认值
# ============================================================================

# 滑动窗口大小（用于平滑曲线）
DEFAULT_WINDOW_SIZE = 50

# 是否生成曲线图（True: 生成, False: 仅显示统计信息）
DEFAULT_GENERATE_PLOT = True

# 输出目录（None表示使用日志文件所在目录）
DEFAULT_OUTPUT_DIR = None

# 图片DPI（分辨率）
FIGURE_DPI = 300

# 图片尺寸（宽度, 高度）
FIGURE_SIZE = (14, 4)

# 曲线图样式配置
PLOT_CONFIG = {
    'linewidth_raw': 0.5,          # 原始数据线条宽度
    'linewidth_smooth': 2,         # 平滑曲线线条宽度
    'alpha_raw': 0.3,              # 原始数据透明度
    'alpha_smooth': 0.8,            # 平滑曲线透明度
    'grid_alpha': 0.3,             # 网格透明度
    'fontsize_label': 12,          # 坐标轴标签字体大小
    'fontsize_title': 14,          # 标题字体大小
    'fontsize_legend': 9,          # 图例字体大小
}

# Reward Detail各项的颜色配置
REWARD_COLORS = {
    'goal': 'green',
    'collision': 'red',
    'angle': 'blue',
    'linear': 'purple',
    'target_distance': 'orange',
    'obs': 'brown',
    'yawrate': 'pink'
}

# 计入成功率/碰撞率/超时率的 end 类型；ForceStop 不统计，忽略
COUNTED_END_STATUSES = ('Goal', 'Collision', 'Timeout')

# Reward Detail各项的标签配置
REWARD_LABELS = {
    'goal': 'Goal Reward',
    'collision': 'Collision Penalty',
    'angle': 'Angle Penalty',
    'linear': 'Linear Penalty',
    'target_distance': 'Target Distance Penalty',
    'obs': 'Obstacle Penalty',
    'yawrate': 'Yawrate Penalty'
}

# ============================================================================
# 导入库
# ============================================================================

import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from datetime import datetime

# 尝试导入matplotlib，如果失败则只提供统计功能
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
    # 设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    matplotlib.rcParams['axes.unicode_minus'] = False
    # 优化matplotlib性能设置
    matplotlib.rcParams['path.simplify'] = True  # 简化路径以提高渲染速度
    matplotlib.rcParams['path.simplify_threshold'] = 1.0  # 简化阈值（值越大简化越多）
    matplotlib.rcParams['agg.path.chunksize'] = 10000  # 分块处理大数据集
    matplotlib.rcParams['figure.max_open_warning'] = 0  # 禁用警告
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib未安装，将跳过绘图功能")


class TeeOutput:
    """同时输出到终端和文件的类"""
    def __init__(self, file_path: Path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # 确保立即写入文件
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        if self.log_file:
            self.log_file.close()


class TrainingMetricsAnalyzer:
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        # training_records: (training_step, critic_loss, actor_loss, avg_sample_times, 
        #                   critic_grad_before, critic_grad_after, actor_grad_before, actor_grad_after,
        #                   entropy, alpha_grad)
        self.training_records = []
        self.episodes = []  # (episode_num, env_id, end_status, reward_details, steps, timestamp)
        # reward_details: Dict with keys: goal, collision, angle, linear, target_distance, obs, yawrate
        # steps: int, episode的步数
        # timestamp: datetime, episode产生的时间戳
        self.best_model_records = []  # (episode_num, success_rate, collision_rate, save_path)
        # 训练耗时记录
        self.training_durations = []  # 每次训练的耗时（秒）
        self.total_sample_count = []  # 每次训练的总抽样数
        self.total_sample_steps = []  # 每次训练的总样本数（step数）
        # 用于跟踪最高成功率
        self.best_success_rate = 0.0
        self.best_model_info = None  # (episode_num, success_rate, collision_rate, save_path)
        
        # 训练记录解析状态机：用于逐行匹配训练信息的5行
        self.current_training_data = {
            'training_step': None,      # 训练步数
            'line1_data': None,         # 第1行：总抽样数、总样本数、平均抽样次数
            'line2_data': None,         # 第2行：Critic损失和梯度
            'line3_data': None,         # 第3行：Actor损失和梯度
            'line4_data': None,         # 第4行：熵值统计（可选）
            'line5_data': None,         # 第5行：训练耗时
            'lines_collected': set()    # 已收集的行号集合（1-5）
        }
        # 记录已完成的训练步数，避免重复解析
        self.completed_training_steps = set()
        
        # 预编译正则表达式以提高性能
        self._compile_regex_patterns()
    
    def _compile_regex_patterns(self):
        """预编译所有正则表达式以提高性能"""
        # 训练记录相关
        self.regex_training_step = re.compile(r'第(\d+)次训练完成')
        self.regex_sample_times = re.compile(r'样本平均抽样次数:\s+([\d.]+)')
        self.regex_critic_loss = re.compile(r'本次训练的平均critic网络损失:\s+([-\d.]+)')
        self.regex_actor_loss = re.compile(r'本次训练的平均actor网络损失:\s+([-\d.]+)')
        self.regex_critic_grad = re.compile(r'critic全局参数梯度L2范数\(裁剪前:([\d.]+),\s*裁剪后:([\d.]+)\)')
        # 兼容旧格式 "actor梯度(裁剪前:x, 裁剪后:y)" 和新格式
        # "actor全局参数梯度L2范数(裁剪前:x, 裁剪后:y)" 等变体
        self.regex_actor_grad = re.compile(
            r'actor.*?梯度.*?\(裁剪前:([\d.]+),\s*裁剪后:([\d.]+)\)'
        )
        self.regex_entropy = re.compile(r'熵值:\s+([-\d.]+)')
        self.regex_alpha_grad = re.compile(r'alpha梯度L2范数:\s+([\d.]+)')
        self.regex_training_duration = re.compile(r'训练耗时:\s+([\d.]+)秒')
        self.regex_total_sample_count = re.compile(r'总抽样数:\s+(\d+)')
        self.regex_total_sample_steps = re.compile(r'总样本数:\s+(\d+)')
        # 时间戳格式：2026-01-07 11:49:47
        self.regex_timestamp = re.compile(r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})')
        
        # Episode相关：end= 后仅 Goal / Timeout / Collision / ForceStop；ForceStop 不参与三率统计
        self.regex_episode_env = re.compile(r'环境\s+(\d+)\s+.*?Episode:\s+(\d+)')
        self.regex_episode_steps = re.compile(r'Steps:\s+(\d+)')
        self.regex_end_status = re.compile(r'end=(Goal|Collision|Timeout|ForceStop)')
        self.regex_episode_old = re.compile(r'环境\s+(\d+)\s+Episode:\s+(\d+).*?End:\s+(Goal|Collision|Timeout)')
        
        # Reward Detail相关
        self.regex_reward_patterns = {
            'goal': re.compile(r'goal=(-?\d+\.?\d*)'),
            'collision': re.compile(r'collision=(-?\d+\.?\d*)'),
            'angle': re.compile(r'angle=(-?\d+\.?\d*)'),
            'linear': re.compile(r'linear=(-?\d+\.?\d*)'),
            'target_distance': re.compile(r'target_distance=(-?\d+\.?\d*)'),
            'obs': re.compile(r'obs=(-?\d+\.?\d*)'),
            'yawrate': re.compile(r'yawrate=(-?\d+\.?\d*)'),
        }
        
        # 最好模型相关
        self.regex_best_model_path = re.compile(r'保存最好模型到:\s+(.+)')
        self.regex_best_model_success = re.compile(r'成功率=([\d.]+)\s*\(')
        self.regex_best_model_collision = re.compile(r'碰撞率=([\d.]+)\s*\(')
    
    def parse_training_record(self, line: str) -> Optional[Tuple]:
        """解析训练完成记录"""
        # 格式：第X次训练完成
        match = self.regex_training_step.search(line)
        if match:
            training_step = int(match.group(1))
            return ('training_start', training_step)
        return None
    
    def match_training_line1(self, line: str) -> bool:
        """匹配训练信息第1行：总抽样数、总样本数、样本平均抽样次数"""
        line = line.strip()
        sample_match = self.regex_sample_times.search(line)
        if sample_match:
            avg_sample_times = float(sample_match.group(1))
            total_sample_count = None
            total_sample_steps = None
            sample_count_match = self.regex_total_sample_count.search(line)
            if sample_count_match:
                total_sample_count = int(sample_count_match.group(1))
            sample_steps_match = self.regex_total_sample_steps.search(line)
            if sample_steps_match:
                total_sample_steps = int(sample_steps_match.group(1))
            self.current_training_data['line1_data'] = {
                'avg_sample_times': avg_sample_times,
                'total_sample_count': total_sample_count,
                'total_sample_steps': total_sample_steps
            }
            self.current_training_data['lines_collected'].add(1)
            return True
        return False
    
    def match_training_line2(self, line: str) -> bool:
        """匹配训练信息第2行：Critic损失和梯度"""
        line = line.strip()
        critic_match = self.regex_critic_loss.search(line)
        if critic_match:
            critic_loss = float(critic_match.group(1))
            critic_grad_before = None
            critic_grad_after = None
            critic_grad_match = self.regex_critic_grad.search(line)
            if critic_grad_match:
                critic_grad_before = float(critic_grad_match.group(1))
                critic_grad_after = float(critic_grad_match.group(2))
            self.current_training_data['line2_data'] = {
                'critic_loss': critic_loss,
                'critic_grad_before': critic_grad_before,
                'critic_grad_after': critic_grad_after
            }
            self.current_training_data['lines_collected'].add(2)
            return True
        return False
    
    def match_training_line3(self, line: str) -> bool:
        """匹配训练信息第3行：Actor损失和梯度"""
        line = line.strip()
        actor_match = self.regex_actor_loss.search(line)
        if actor_match:
            actor_loss = float(actor_match.group(1))
            actor_grad_before = None
            actor_grad_after = None
            actor_grad_match = self.regex_actor_grad.search(line)
            if actor_grad_match:
                actor_grad_before = float(actor_grad_match.group(1))
                actor_grad_after = float(actor_grad_match.group(2))
            self.current_training_data['line3_data'] = {
                'actor_loss': actor_loss,
                'actor_grad_before': actor_grad_before,
                'actor_grad_after': actor_grad_after
            }
            self.current_training_data['lines_collected'].add(3)
            return True
        return False
    
    def match_training_line4(self, line: str) -> bool:
        """匹配训练信息第4行：熵值统计（可选）"""
        line = line.strip()
        if '熵值统计' in line:
            entropy = None
            alpha_grad = None
            entropy_match = self.regex_entropy.search(line)
            if entropy_match:
                entropy = float(entropy_match.group(1))
            alpha_grad_match = self.regex_alpha_grad.search(line)
            if alpha_grad_match:
                alpha_grad = float(alpha_grad_match.group(1))
            self.current_training_data['line4_data'] = {
                'entropy': entropy,
                'alpha_grad': alpha_grad
            }
            self.current_training_data['lines_collected'].add(4)
            return True
        return False
    
    def match_training_line5(self, line: str) -> bool:
        """匹配训练信息第5行：训练耗时"""
        line = line.strip()
        duration_match = self.regex_training_duration.search(line)
        if duration_match:
            training_duration = float(duration_match.group(1))
            self.current_training_data['line5_data'] = {
                'training_duration': training_duration
            }
            self.current_training_data['lines_collected'].add(5)
            return True
        return False
    
    def try_complete_training_record(self) -> bool:
        """尝试完成当前训练记录（当收集到必要的行时）"""
        # 至少需要第1、2、3行才能完成一个训练记录
        if 1 in self.current_training_data['lines_collected'] and \
           2 in self.current_training_data['lines_collected'] and \
           3 in self.current_training_data['lines_collected']:
            
            # 获取训练步数
            training_step = self.current_training_data['training_step']
            if training_step is None:
                training_step = len(self.training_records) + 1
            
            # 组装训练记录
            line1 = self.current_training_data['line1_data']
            line2 = self.current_training_data['line2_data']
            line3 = self.current_training_data['line3_data']
            line4 = self.current_training_data['line4_data']
            line5 = self.current_training_data['line5_data']
            
            # 检查是否已经完成过这个训练步数（避免重复解析）
            if training_step in self.completed_training_steps:
                # 如果已经完成过，重置状态机但不添加记录
                self.current_training_data = {
                    'training_step': None,
                    'line1_data': None,
                    'line2_data': None,
                    'line3_data': None,
                    'line4_data': None,
                    'line5_data': None,
                    'lines_collected': set()
                }
                return False
            
            self.training_records.append((
                training_step,
                line2['critic_loss'],
                line3['actor_loss'],
                line1['avg_sample_times'],
                line2['critic_grad_before'],
                line2['critic_grad_after'],
                line3['actor_grad_before'],
                line3['actor_grad_after'],
                line4['entropy'] if line4 else None,
                line4['alpha_grad'] if line4 else None
            ))
            
            # 记录已完成的训练步数
            self.completed_training_steps.add(training_step)
            
            # 保存训练耗时、总抽样数、总样本数
            if line5 and line5['training_duration'] is not None:
                self.training_durations.append(line5['training_duration'])
            if line1 and line1['total_sample_count'] is not None:
                self.total_sample_count.append(line1['total_sample_count'])
            if line1 and line1['total_sample_steps'] is not None:
                self.total_sample_steps.append(line1['total_sample_steps'])
            
            # 重置状态机
            self.current_training_data = {
                'training_step': None,
                'line1_data': None,
                'line2_data': None,
                'line3_data': None,
                'line4_data': None,
                'line5_data': None,
                'lines_collected': set()
            }
            return True
        return False
    
    def parse_training_details(self, lines: List[str], start_idx: int) -> Optional[Tuple]:
        """解析训练详细信息（多行）
        直接匹配5行模式，不依赖训练完成行
        第1行：总抽样数、总样本数、样本平均抽样次数
        第2行：Critic损失和梯度
        第3行：Actor损失和梯度
        第4行：熵值统计（可选）
        第5行：训练耗时
        """
        if start_idx + 2 >= len(lines):
            return None
        
        try:
            # 第一行：总抽样数、总样本数、样本平均抽样次数
            # 格式：  总抽样数: X | 总样本数: Y | 样本平均抽样次数: Z.ZZ
            line1 = lines[start_idx].strip()
            sample_match = self.regex_sample_times.search(line1)
            if not sample_match:
                return None
            avg_sample_times = float(sample_match.group(1))
            
            # 尝试从第一行或前面的行中提取训练步数（如果有训练完成行）
            training_step = None
            # 向前查找最多5行，寻找训练完成行
            for lookback in range(1, min(6, start_idx + 1)):
                prev_line = lines[start_idx - lookback].strip()
                training_match = self.regex_training_step.search(prev_line)
                if training_match:
                    training_step = int(training_match.group(1))
                    break
            # 如果找不到，使用已解析的训练记录数量+1作为训练步数
            if training_step is None:
                training_step = len(self.training_records) + 1
            
            # 第二行：critic和actor网络损失，可能包含梯度信息
            # 格式：  本次训练的平均critic网络损失: X.XX | 前10次训练的平均critic网络损失: Y.YY | critic全局参数梯度L2范数(裁剪前:XX.XX, 裁剪后:YY.YY)
            line2 = lines[start_idx + 1].strip()
            critic_match = self.regex_critic_loss.search(line2)
            if not critic_match:
                return None
            critic_loss = float(critic_match.group(1))
            
            # 提取critic梯度信息（裁剪前后）
            critic_grad_before = None
            critic_grad_after = None
            critic_grad_match = self.regex_critic_grad.search(line2)
            if critic_grad_match:
                critic_grad_before = float(critic_grad_match.group(1))
                critic_grad_after = float(critic_grad_match.group(2))
            
            # 第三行：actor网络损失，可能包含梯度信息
            # 格式：  本次训练的平均actor网络损失: X.XX | 前10次训练的平均actor网络损失: Y.YY | actor梯度(裁剪前:XX.XX, 裁剪后:YY.YY)
            line3 = lines[start_idx + 2].strip()
            actor_match = self.regex_actor_loss.search(line3)
            if not actor_match:
                return None
            actor_loss = float(actor_match.group(1))
            
            # 提取actor梯度信息（裁剪前后）
            actor_grad_before = None
            actor_grad_after = None
            actor_grad_match = self.regex_actor_grad.search(line3)
            if actor_grad_match:
                actor_grad_before = float(actor_grad_match.group(1))
                actor_grad_after = float(actor_grad_match.group(2))
            
            # 第四行：熵值统计（可能不存在）
            # 格式：  熵值统计: | 熵值: X.XX | alpha梯度L2范数: Y.YY
            entropy = None
            alpha_grad = None
            if start_idx + 3 < len(lines):
                line4 = lines[start_idx + 3].strip()
                if '熵值统计' in line4:
                    entropy_match = self.regex_entropy.search(line4)
                    if entropy_match:
                        entropy = float(entropy_match.group(1))
                    alpha_grad_match = self.regex_alpha_grad.search(line4)
                    if alpha_grad_match:
                        alpha_grad = float(alpha_grad_match.group(1))
            
            # 解析训练耗时、总抽样数、总样本数
            training_duration = None
            total_sample_count = None
            total_sample_steps = None
            # 从第一行提取总抽样数和总样本数
            sample_count_match = self.regex_total_sample_count.search(line1)
            if sample_count_match:
                total_sample_count = int(sample_count_match.group(1))
            sample_steps_match = self.regex_total_sample_steps.search(line1)
            if sample_steps_match:
                total_sample_steps = int(sample_steps_match.group(1))
            # 查找训练耗时（可能在熵值统计行之后，即第5行或更后面）
            # 检查第4行（如果有熵值统计）或第4行之后
            check_start = start_idx + 3 if (start_idx + 3 < len(lines) and '熵值统计' in lines[start_idx + 3].strip()) else start_idx + 2
            for idx in range(check_start, min(start_idx + 7, len(lines))):
                line_check = lines[idx].strip()
                duration_match = self.regex_training_duration.search(line_check)
                if duration_match:
                    training_duration = float(duration_match.group(1))
                    break
            
            return ('training_details', training_step, avg_sample_times, critic_loss, actor_loss, 
                   critic_grad_before, critic_grad_after, actor_grad_before, actor_grad_after,
                   entropy, alpha_grad, training_duration, total_sample_count, total_sample_steps)
        except (IndexError, ValueError, AttributeError):
            return None
    
    def parse_reward_detail(self, line: str) -> Dict:
        """解析Reward Detail行"""
        # 格式：  Reward Detail: end=Status, total_reward=X.XX, goal=Y.YY, collision=Z.ZZ, angle=W.WW, linear=V.VV, target_distance=U.UU
        detail = {}
        # 使用预编译的正则表达式
        for key, pattern in self.regex_reward_patterns.items():
            match = pattern.search(line)
            if match:
                try:
                    detail[key] = float(match.group(1))
                except ValueError:
                    detail[key] = 0.0  # 如果转换失败，默认为0
            else:
                detail[key] = 0.0  # 如果没有找到，默认为0
        return detail
    
    def parse_best_model_record(self, lines: List[str], start_idx: int) -> Optional[Tuple]:
        """解析保存最好模型的记录"""
        # 格式：
        # ============================================================
        # 保存最好模型到: /path/to/best_model
        # 当前最好统计: 成功率=0.1270 (12.70%), 碰撞率=0.2910 (29.10%)
        # ============================================================
        if start_idx + 2 >= len(lines):
            return None
        
        try:
            # 检查是否是保存最好模型的记录（strip()去除换行符和空格）
            line1 = lines[start_idx + 1].strip() if start_idx + 1 < len(lines) else ""
            if '保存最好模型到:' not in line1:
                return None
            
            # 提取保存路径
            path_match = self.regex_best_model_path.search(line1)
            if not path_match:
                return None
            save_path = path_match.group(1).strip()
            
            # 提取成功率和碰撞率
            stats_line = lines[start_idx + 2].strip() if start_idx + 2 < len(lines) else ""
            success_match = self.regex_best_model_success.search(stats_line)
            collision_match = self.regex_best_model_collision.search(stats_line)
            
            if not success_match or not collision_match:
                return None
            
            success_rate = float(success_match.group(1))
            collision_rate = float(collision_match.group(1))
            
            # 尝试从上下文获取episode编号（从之前的episode记录中推断）
            # 如果找不到，使用当前已解析的最后一个episode编号
            episode_num = 0
            if self.episodes:
                episode_num = self.episodes[-1][0]
            
            return ('best_model', episode_num, success_rate, collision_rate, save_path)
        except (IndexError, ValueError, AttributeError):
            return None
    
    def parse_episode(self, line: str, prev_line: str = "") -> Optional[Tuple]:
        """解析episode信息"""
        # 检查是否是Reward Detail行
        if 'Reward Detail:' in line:
            # 从Reward Detail行提取end状态和reward details
            end_match = self.regex_end_status.search(line)
            reward_detail = self.parse_reward_detail(line)
            
            if end_match and prev_line:
                # 从上一行提取episode编号、环境ID和Steps
                # 匹配格式：环境 X Episode: Y Target Distance: A.AA (actual: B.BB) Steps: N
                match = self.regex_episode_env.search(prev_line)
                steps_match = self.regex_episode_steps.search(prev_line)
                if match:
                    env_id = int(match.group(1))
                    episode_num = int(match.group(2))
                    end_status = end_match.group(1)
                    # 提取Steps信息，如果没有找到则默认为0
                    steps = int(steps_match.group(1)) if steps_match else 0
                    # 提取时间戳
                    timestamp = None
                    timestamp_match = self.regex_timestamp.search(prev_line)
                    if timestamp_match:
                        try:
                            timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            timestamp = None
                    return ('episode', env_id, episode_num, end_status, reward_detail, steps, timestamp)
        
        # 向后兼容：尝试匹配旧格式
        match = self.regex_episode_old.search(line)
        if match:
            env_id = int(match.group(1))
            episode_num = int(match.group(2))
            end_status = match.group(3)
            # 旧格式没有reward detail，使用空字典
            reward_detail = {'goal': 0.0, 'collision': 0.0, 'angle': 0.0, 'linear': 0.0, 
                           'target_distance': 0.0, 'obs': 0.0, 'yawrate': 0.0}
            # 旧格式没有steps信息，默认为0
            steps = 0
            # 提取时间戳
            timestamp = None
            timestamp_match = self.regex_timestamp.search(line)
            if timestamp_match:
                try:
                    timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    timestamp = None
            return ('episode', env_id, episode_num, end_status, reward_detail, steps, timestamp)
        
        return None
    
    def collect_training_lines_forward(self, lines: List[str], training_start_idx: int, max_lookahead: int = 100) -> int:
        """从训练完成行开始，向后收集所有训练信息行
        返回：收集到的最后一个训练信息行的索引（不包括训练完成行本身）
        """
        # 重置状态机
        self.current_training_data = {
            'training_step': None,
            'line1_data': None,
            'line2_data': None,
            'line3_data': None,
            'line4_data': None,
            'line5_data': None,
            'lines_collected': set()
        }
        
        # 设置训练步数
        training_start = self.parse_training_record(lines[training_start_idx])
        if training_start:
            self.current_training_data['training_step'] = training_start[1]
        
        # 从训练完成行的下一行开始，向后搜索最多max_lookahead行
        last_training_line_idx = training_start_idx
        end_idx = min(training_start_idx + max_lookahead + 1, len(lines))
        
        for j in range(training_start_idx + 1, end_idx):
            line = lines[j]
            
            # 如果遇到新的训练完成行，停止收集
            if self.parse_training_record(line):
                break
            
            # 检查是否是训练信息行
            is_training_line = False
            
            # 依次尝试匹配训练信息的5行（跳过已经匹配的行）
            if 1 not in self.current_training_data['lines_collected']:
                if self.match_training_line1(line):
                    is_training_line = True
                    last_training_line_idx = j
            if 2 not in self.current_training_data['lines_collected']:
                if self.match_training_line2(line):
                    is_training_line = True
                    last_training_line_idx = j
            if 3 not in self.current_training_data['lines_collected']:
                if self.match_training_line3(line):
                    is_training_line = True
                    last_training_line_idx = j
            if 4 not in self.current_training_data['lines_collected']:
                if self.match_training_line4(line):
                    is_training_line = True
                    last_training_line_idx = j
            if 5 not in self.current_training_data['lines_collected']:
                if self.match_training_line5(line):
                    is_training_line = True
                    last_training_line_idx = j
            
            # 如果已经收集了所有必要的行（至少1、2、3行和5行），可以提前停止
            if (1 in self.current_training_data['lines_collected'] and 
                2 in self.current_training_data['lines_collected'] and 
                3 in self.current_training_data['lines_collected'] and
                5 in self.current_training_data['lines_collected']):
                # 如果连续几行都不是训练信息行，可以提前停止
                if not is_training_line:
                    # 检查后续几行是否还有训练信息行
                    has_more_training_lines = False
                    for k in range(j + 1, min(j + 3, end_idx)):
                        check_line = lines[k]
                        if (self.regex_sample_times.search(check_line) or 
                            self.regex_critic_loss.search(check_line) or 
                            self.regex_actor_loss.search(check_line) or
                            self.regex_training_duration.search(check_line) or
                            ('熵值统计' in check_line)):
                            has_more_training_lines = True
                            break
                    if not has_more_training_lines:
                        break
        
        return last_training_line_idx
    
    def parse_log(self):
        """解析日志文件（优化版本：匹配到训练完成行后，向后收集所有训练信息行）"""
        print(f"正在解析日志文件: {self.log_file}")
        
        # 读取文件（对于大文件，可以考虑分块读取，但这里一次性读取更简单）
        with open(self.log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"日志文件总行数: {len(lines)}")
        
        # 预分配列表以提高性能（如果可能的话）
        prev_line = ""
        
        # 使用更高效的循环（减少函数调用开销）
        i = 0
        lines_len = len(lines)
        
        while i < lines_len:
            # 直接访问，减少函数调用
            line = lines[i]
            
            # 首先检查是否是训练完成行
            training_start = self.parse_training_record(line)
            if training_start:
                training_step = training_start[1]
                # 检查这个训练步数是否已经完成过
                if training_step in self.completed_training_steps:
                    # 如果已经完成过，跳过这行继续处理
                    i += 1
                    continue
                
                # 匹配到训练完成行后，向后搜索收集所有训练信息行
                last_training_line_idx = self.collect_training_lines_forward(lines, i, max_lookahead=100)
                
                # 尝试完成训练记录
                record_completed = self.try_complete_training_record()
                
                if record_completed:
                    # 训练记录已完成，跳转到训练完成行的下一行继续处理
                    i = i + 1
                    continue
                else:
                    # 训练记录未完成（可能缺少必要的信息），重置状态机并继续
                    self.current_training_data = {
                        'training_step': None,
                        'line1_data': None,
                        'line2_data': None,
                        'line3_data': None,
                        'line4_data': None,
                        'line5_data': None,
                        'lines_collected': set()
                    }
                    i += 1
                    continue
            
            # 检查是否是训练信息行（总抽样数、critic损失、actor损失等）
            # 如果遇到训练信息行但状态机中没有训练步数，说明这些行肯定已经被前面的训练完成行统计过了，直接跳过
            is_training_info_line = (
                self.regex_sample_times.search(line) or 
                self.regex_critic_loss.search(line) or 
                self.regex_actor_loss.search(line) or
                self.regex_training_duration.search(line) or
                ('熵值统计' in line)
            )
            
            if is_training_info_line:
                # 如果状态机中没有训练步数，说明这些训练信息行肯定已经被前面的训练完成行统计过了
                # 直接跳过，避免重复解析
                if self.current_training_data['training_step'] is None:
                    i += 1
                    continue
            
            # 尝试匹配Episode
            if 'Reward Detail:' in line or 'Episode:' in line:
                episode_info = self.parse_episode(line, prev_line)
                if episode_info and episode_info[0] == 'episode':
                    _, env_id, episode_num, end_status, reward_detail, steps, timestamp = episode_info
                    self.episodes.append((episode_num, env_id, end_status, reward_detail, steps, timestamp))
                    i += 1
                    prev_line = line
                    continue
            
            # 尝试匹配最好模型保存记录
            if '============================================================' in line and i + 3 < lines_len:
                best_model_info = self.parse_best_model_record(lines, i)
                if best_model_info and best_model_info[0] == 'best_model':
                    _, episode_num, success_rate, collision_rate, save_path = best_model_info
                    self.best_model_records.append((episode_num, success_rate, collision_rate, save_path))
                    # 更新最高成功率信息
                    if success_rate > self.best_success_rate:
                        self.best_success_rate = success_rate
                        self.best_model_info = (episode_num, success_rate, collision_rate, save_path)
                    i += 4  # 跳过已解析的行（包括分隔线和空行）
                    continue
            
            prev_line = line
            i += 1
        
        # 解析完成后，尝试完成最后一个训练记录（如果有未完成的）
        self.try_complete_training_record()
        
        # 使用numpy进行排序（对于大数据集更快）
        if self.training_records:
            self.training_records.sort(key=lambda x: x[0])
        if self.episodes:
            self.episodes.sort(key=lambda x: x[0])
        if self.best_model_records:
            self.best_model_records.sort(key=lambda x: x[0])
        
        print(f"成功解析 {len(self.training_records)} 个训练记录")
        print(f"成功解析 {len(self.episodes)} 个episode")
        print(f"成功解析 {len(self.best_model_records)} 个最好模型保存记录")
    
    def calculate_sliding_window(self, data: List[float], window_size: int) -> List[float]:
        """计算滑动窗口平均值（使用numpy向量化优化）"""
        if not data:
            return []
        
        # 转换为numpy数组以提高性能
        arr = np.array(data, dtype=np.float64)
        n = len(arr)
        
        if n == 0:
            return []
        
        # 使用numpy的cumsum和数组索引进行向量化计算
        # 对于每个位置i，计算从max(0, i-window_size+1)到i的平均值
        cumsum = np.cumsum(arr, dtype=np.float64)
        
        # 预计算所有窗口长度
        indices = np.arange(n)
        start_indices = np.maximum(0, indices - window_size + 1)
        window_lens = indices - start_indices + 1
        
        # 向量化计算：对于每个位置，使用cumsum计算窗口和
        # 当start_idx > 0时，需要减去前面的累积和
        result = np.zeros(n, dtype=np.float64)
        mask = start_indices > 0
        result[~mask] = cumsum[~mask] / window_lens[~mask]
        if np.any(mask):
            result[mask] = (cumsum[mask] - cumsum[start_indices[mask] - 1]) / window_lens[mask]
        
        return result.tolist()
    
    def downsample_for_plot(self, x_data: List, y_data: List, max_points: int = 5000) -> Tuple[List, List]:
        """对绘图数据进行降采样以提高性能（保留首尾和关键点）"""
        if len(x_data) <= max_points:
            return x_data, y_data
        
        # 转换为numpy数组
        x_arr = np.array(x_data)
        y_arr = np.array(y_data)
        
        # 计算步长
        step = max(1, len(x_data) // max_points)
        
        # 均匀采样
        indices = np.arange(0, len(x_data), step)
        
        # 确保包含最后一个点
        if indices[-1] != len(x_data) - 1:
            indices = np.append(indices, len(x_data) - 1)
        
        return x_arr[indices].tolist(), y_arr[indices].tolist()
    
    def add_statistics_text(self, ax, data: List[float], position: str = 'upper right', 
                           fontsize: int = 9, precision: int = 4):
        """在图上添加统计信息文本（最小值、最大值、平均值）"""
        if not data:
            return
        
        data_arr = np.array(data)
        min_val = np.min(data_arr)
        max_val = np.max(data_arr)
        mean_val = np.mean(data_arr)
        
        # 格式化文本
        stats_text = f'Min: {min_val:.{precision}f}\nMax: {max_val:.{precision}f}\nMean: {mean_val:.{precision}f}'
        
        # 根据位置设置坐标
        if position == 'upper right':
            x_pos = 0.98
            y_pos = 0.98
            ha = 'right'
            va = 'top'
        elif position == 'upper left':
            x_pos = 0.02
            y_pos = 0.98
            ha = 'left'
            va = 'top'
        elif position == 'lower right':
            x_pos = 0.98
            y_pos = 0.02
            ha = 'right'
            va = 'bottom'
        elif position == 'lower left':
            x_pos = 0.02
            y_pos = 0.02
            ha = 'left'
            va = 'bottom'
        else:
            x_pos = 0.98
            y_pos = 0.98
            ha = 'right'
            va = 'top'
        
        # 添加文本框（带背景）
        ax.text(x_pos, y_pos, stats_text, transform=ax.transAxes,
               fontsize=fontsize, verticalalignment=va, horizontalalignment=ha,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               family='monospace')
    
    def calculate_episode_rates(self, window_size: int) -> Tuple[List[int], List[float], List[float], List[float]]:
        """计算episode的成功率、碰撞率、超时率（滑动窗口）。仅统计 end=Goal/Collision/Timeout，ForceStop 忽略。"""
        if not self.episodes:
            return [], [], [], []

        counted = [ep for ep in self.episodes if ep[2] in COUNTED_END_STATUSES]
        if not counted:
            return [], [], [], []

        episodes = np.array([ep[0] for ep in counted], dtype=np.int32)
        successes = np.array([1 if ep[2] == 'Goal' else 0 for ep in counted], dtype=np.float64)
        collisions = np.array([1 if ep[2] == 'Collision' else 0 for ep in counted], dtype=np.float64)
        timeouts = np.array([1 if ep[2] == 'Timeout' else 0 for ep in counted], dtype=np.float64)

        success_rates = self.calculate_sliding_window(successes.tolist(), window_size)
        collision_rates = self.calculate_sliding_window(collisions.tolist(), window_size)
        timeout_rates = self.calculate_sliding_window(timeouts.tolist(), window_size)

        return episodes.tolist(), success_rates, collision_rates, timeout_rates
    
    def calculate_reward_detail_curves(self, window_size: int) -> Tuple[List[int], Dict[str, List[float]]]:
        """计算reward detail各项的滑动窗口平均值。ForceStop 不参与。"""
        counted = [ep for ep in self.episodes if ep[2] in COUNTED_END_STATUSES]
        if not counted:
            return [], {}

        episodes = [ep[0] for ep in counted]
        reward_keys = ['goal', 'collision', 'angle', 'linear', 'target_distance', 'obs', 'yawrate']
        reward_curves = {}
        for key in reward_keys:
            values = [ep[3].get(key, 0.0) for ep in counted]
            reward_curves[key] = self.calculate_sliding_window(values, window_size)
        return episodes, reward_curves
    
    def calculate_per_step_reward_stats(self) -> Dict[str, float]:
        """计算每个step平均的Reward Detail（不包含goal和collision）"""
        if not self.episodes:
            return {}
        
        # 排除goal和collision的reward keys
        reward_keys = ['angle', 'linear', 'target_distance', 'obs', 'yawrate']
        
        # 统计所有episode的总reward和总step数
        total_rewards = {key: 0.0 for key in reward_keys}
        total_steps = 0
        valid_episodes = 0
        
        for ep in self.episodes:
            episode_num, env_id, end_status, reward_detail, steps, timestamp = ep

            if end_status == 'ForceStop':
                continue
            if steps <= 0:
                continue

            valid_episodes += 1
            total_steps += steps
            
            # 累加每个reward项的总值
            for key in reward_keys:
                reward_value = reward_detail.get(key, 0.0)
                total_rewards[key] += reward_value
        
        # 计算每个step的平均reward（总reward除以总step数）
        result = {}
        for key in reward_keys:
            if total_steps > 0:
                result[key] = total_rewards[key] / total_steps
            else:
                result[key] = 0.0
        
        result['total_steps'] = total_steps
        result['valid_episodes'] = valid_episodes
        
        return result
    
    def plot_curves(self, output_dir: Optional[str] = None, window_size: int = 100):
        """绘制训练曲线"""
        if not HAS_MATPLOTLIB:
            print("警告: matplotlib未安装，跳过绘图功能")
            return
        
        if not self.training_records and not self.episodes:
            print("错误：未找到有效的训练数据")
            return
        
        # 准备数据
        num_plots = 0
        has_training = len(self.training_records) > 0
        has_episodes = len(self.episodes) > 0
        counted_eps = [ep for ep in self.episodes if ep[2] in COUNTED_END_STATUSES]
        has_reward_details = bool(counted_eps) and any(ep[3] for ep in counted_eps if any(ep[3].values()))
        
        # 检查是否有梯度数据
        has_gradients = False
        if has_training and len(self.training_records) > 0:
            # 检查是否有梯度数据（gradient字段不为None）
            has_gradients = any(tr[4] is not None for tr in self.training_records)  # critic_grad_before
        
        # 检查是否有熵值和alpha梯度数据
        has_entropy = False
        if has_training and len(self.training_records) > 0:
            # 检查是否有熵值数据（entropy字段不为None）
            has_entropy = any(tr[8] is not None for tr in self.training_records)  # entropy
        
        if has_training:
            num_plots += 3  # critic loss, actor loss, avg_sample_times
        if has_gradients:
            num_plots += 2  # critic gradients, actor gradients
        if has_entropy:
            num_plots += 2  # entropy curve, alpha gradient curve
        if has_episodes:
            num_plots += 1  # episode rates
        if has_reward_details:
            num_plots += 1  # reward detail curves
        
        if num_plots == 0:
            print("错误：没有可绘制的数据")
            return
        
        # 创建图形（优化：减少DPI以提高渲染速度，但保持清晰度）
        # 使用更高效的backend设置
        fig, axes = plt.subplots(num_plots, 1, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1]*num_plots))
        if num_plots == 1:
            axes = [axes]
        
        # 设置绘图优化参数
        MAX_POINTS_PER_LINE = 5000  # 每条线最大点数，超过则降采样
        
        plot_idx = 0
        
        # 1. Critic Loss曲线（滑动窗口）
        if has_training:
            training_steps = [tr[0] for tr in self.training_records]
            critic_losses = [tr[1] for tr in self.training_records]
            critic_losses_smooth = self.calculate_sliding_window(critic_losses, window_size)
            
            ax = axes[plot_idx]
            # 对原始数据进行降采样以提高绘图性能
            if len(training_steps) > MAX_POINTS_PER_LINE:
                steps_raw, losses_raw = self.downsample_for_plot(training_steps, critic_losses, MAX_POINTS_PER_LINE)
                ax.plot(steps_raw, losses_raw, 'b-', 
                       linewidth=PLOT_CONFIG['linewidth_raw'], 
                       alpha=PLOT_CONFIG['alpha_raw'], label='Raw')
            else:
                ax.plot(training_steps, critic_losses, 'b-', 
                       linewidth=PLOT_CONFIG['linewidth_raw'], 
                       alpha=PLOT_CONFIG['alpha_raw'], label='Raw')
            # 平滑曲线通常点数较少，直接绘制
            ax.plot(training_steps, critic_losses_smooth, 'b-', 
                   linewidth=PLOT_CONFIG['linewidth_smooth'], 
                   alpha=PLOT_CONFIG['alpha_smooth'], 
                   label=f'Smoothed (window={window_size})')
            ax.set_xlabel('Training Step', fontsize=PLOT_CONFIG['fontsize_label'])
            ax.set_ylabel('Critic Loss', fontsize=PLOT_CONFIG['fontsize_label'])
            ax.set_title('Critic Loss Curve', fontsize=PLOT_CONFIG['fontsize_title'], fontweight='bold')
            ax.grid(True, alpha=PLOT_CONFIG['grid_alpha'])
            ax.legend(loc='upper right')
            if max(critic_losses) / min([x for x in critic_losses if x > 0]) > 100:
                ax.set_yscale('log')
                ax.set_ylabel('Critic Loss (log scale)', fontsize=PLOT_CONFIG['fontsize_label'])
            # 添加统计信息
            self.add_statistics_text(ax, critic_losses, position='upper left', 
                                    fontsize=PLOT_CONFIG['fontsize_legend'], precision=2)
            plot_idx += 1
            
            # 2. Actor Loss曲线（滑动窗口）
            actor_losses = [tr[2] for tr in self.training_records]
            actor_losses_smooth = self.calculate_sliding_window(actor_losses, window_size)
            
            ax = axes[plot_idx]
            # 对原始数据进行降采样以提高绘图性能
            if len(training_steps) > MAX_POINTS_PER_LINE:
                steps_raw, losses_raw = self.downsample_for_plot(training_steps, actor_losses, MAX_POINTS_PER_LINE)
                ax.plot(steps_raw, losses_raw, 'r-', 
                       linewidth=PLOT_CONFIG['linewidth_raw'], 
                       alpha=PLOT_CONFIG['alpha_raw'], label='Raw')
            else:
                ax.plot(training_steps, actor_losses, 'r-', 
                       linewidth=PLOT_CONFIG['linewidth_raw'], 
                       alpha=PLOT_CONFIG['alpha_raw'], label='Raw')
            ax.plot(training_steps, actor_losses_smooth, 'r-', 
                   linewidth=PLOT_CONFIG['linewidth_smooth'], 
                   alpha=PLOT_CONFIG['alpha_smooth'],
                   label=f'Smoothed (window={window_size})')
            ax.set_xlabel('Training Step', fontsize=PLOT_CONFIG['fontsize_label'])
            ax.set_ylabel('Actor Loss', fontsize=PLOT_CONFIG['fontsize_label'])
            ax.set_title('Actor Loss Curve', fontsize=PLOT_CONFIG['fontsize_title'], fontweight='bold')
            ax.grid(True, alpha=PLOT_CONFIG['grid_alpha'])
            ax.legend(loc='upper right')
            # 添加统计信息
            self.add_statistics_text(ax, actor_losses, position='upper left', 
                                    fontsize=PLOT_CONFIG['fontsize_legend'], precision=2)
            plot_idx += 1
            
            # 3. 样本平均抽样次数曲线（滑动窗口）
            avg_sample_times = [tr[3] for tr in self.training_records]
            avg_sample_times_smooth = self.calculate_sliding_window(avg_sample_times, window_size)
            
            ax = axes[plot_idx]
            # 对原始数据进行降采样以提高绘图性能
            if len(training_steps) > MAX_POINTS_PER_LINE:
                steps_raw, times_raw = self.downsample_for_plot(training_steps, avg_sample_times, MAX_POINTS_PER_LINE)
                ax.plot(steps_raw, times_raw, 'g-', 
                       linewidth=PLOT_CONFIG['linewidth_raw'], 
                       alpha=PLOT_CONFIG['alpha_raw'], label='Raw')
            else:
                ax.plot(training_steps, avg_sample_times, 'g-', 
                       linewidth=PLOT_CONFIG['linewidth_raw'], 
                       alpha=PLOT_CONFIG['alpha_raw'], label='Raw')
            ax.plot(training_steps, avg_sample_times_smooth, 'g-', 
                   linewidth=PLOT_CONFIG['linewidth_smooth'], 
                   alpha=PLOT_CONFIG['alpha_smooth'],
                   label=f'Smoothed (window={window_size})')
            ax.set_xlabel('Training Step', fontsize=PLOT_CONFIG['fontsize_label'])
            ax.set_ylabel('Average Sample Times', fontsize=PLOT_CONFIG['fontsize_label'])
            ax.set_title('Average Sample Times Curve', fontsize=PLOT_CONFIG['fontsize_title'], fontweight='bold')
            ax.grid(True, alpha=PLOT_CONFIG['grid_alpha'])
            ax.legend(loc='upper right')
            # 添加统计信息
            self.add_statistics_text(ax, avg_sample_times, position='upper left', 
                                    fontsize=PLOT_CONFIG['fontsize_legend'], precision=2)
            plot_idx += 1
        
        # 梯度曲线（如果有梯度数据）
        if has_gradients:
            # 提取梯度数据（过滤掉None值）
            gradient_data = []
            gradient_steps = []
            for tr in self.training_records:
                if tr[4] is not None:  # critic_grad_before不为None
                    gradient_steps.append(tr[0])
                    gradient_data.append((
                        tr[4],  # critic_grad_before
                        tr[5] if tr[5] is not None else 0.0,  # critic_grad_after
                        tr[6] if tr[6] is not None else 0.0,  # actor_grad_before
                        tr[7] if tr[7] is not None else 0.0   # actor_grad_after
                    ))
            
            if gradient_data:
                # 4. Critic梯度曲线（裁剪前后）
                critic_grad_before = [gd[0] for gd in gradient_data]
                critic_grad_after = [gd[1] for gd in gradient_data]
                critic_grad_before_smooth = self.calculate_sliding_window(critic_grad_before, window_size)
                critic_grad_after_smooth = self.calculate_sliding_window(critic_grad_after, window_size)
                
                ax = axes[plot_idx]
                # 对原始数据进行降采样以提高绘图性能
                if len(gradient_steps) > MAX_POINTS_PER_LINE:
                    steps_before, grad_before = self.downsample_for_plot(gradient_steps, critic_grad_before, MAX_POINTS_PER_LINE)
                    steps_after, grad_after = self.downsample_for_plot(gradient_steps, critic_grad_after, MAX_POINTS_PER_LINE)
                    ax.plot(steps_before, grad_before, 'b--', 
                           linewidth=PLOT_CONFIG['linewidth_raw'], 
                           alpha=PLOT_CONFIG['alpha_raw'], label='Critic Grad Before Clip (Raw)')
                    ax.plot(steps_after, grad_after, 'b-', 
                           linewidth=PLOT_CONFIG['linewidth_raw'], 
                           alpha=PLOT_CONFIG['alpha_raw'], label='Critic Grad After Clip (Raw)')
                else:
                    ax.plot(gradient_steps, critic_grad_before, 'b--', 
                           linewidth=PLOT_CONFIG['linewidth_raw'], 
                           alpha=PLOT_CONFIG['alpha_raw'], label='Critic Grad Before Clip (Raw)')
                    ax.plot(gradient_steps, critic_grad_after, 'b-', 
                           linewidth=PLOT_CONFIG['linewidth_raw'], 
                           alpha=PLOT_CONFIG['alpha_raw'], label='Critic Grad After Clip (Raw)')
                ax.plot(gradient_steps, critic_grad_before_smooth, 'b--', 
                       linewidth=PLOT_CONFIG['linewidth_smooth'], 
                       alpha=PLOT_CONFIG['alpha_smooth'], 
                       label=f'Critic Grad Before Clip (window={window_size})')
                ax.plot(gradient_steps, critic_grad_after_smooth, 'b-', 
                       linewidth=PLOT_CONFIG['linewidth_smooth'], 
                       alpha=PLOT_CONFIG['alpha_smooth'], 
                       label=f'Critic Grad After Clip (window={window_size})')
                ax.set_xlabel('Training Step', fontsize=PLOT_CONFIG['fontsize_label'])
                ax.set_ylabel('Critic Gradient L2 Norm', fontsize=PLOT_CONFIG['fontsize_label'])
                ax.set_title('Critic Gradient Norm (Before/After Clipping)', 
                           fontsize=PLOT_CONFIG['fontsize_title'], fontweight='bold')
                ax.grid(True, alpha=PLOT_CONFIG['grid_alpha'])
                ax.legend(loc='upper right', fontsize=PLOT_CONFIG['fontsize_legend'])
                # 如果梯度值范围很大，使用对数刻度
                critic_grad_after_positive = [x for x in critic_grad_after if x is not None and x > 0]
                if critic_grad_before and critic_grad_after_positive and max(critic_grad_before) / max(critic_grad_after_positive + [1]) > 10:
                    ax.set_yscale('log')
                    ax.set_ylabel('Critic Gradient L2 Norm (log scale)', fontsize=PLOT_CONFIG['fontsize_label'])
                # 添加统计信息（使用裁剪前的数据）
                self.add_statistics_text(ax, critic_grad_before, position='upper left', 
                                        fontsize=PLOT_CONFIG['fontsize_legend'], precision=4)
                plot_idx += 1
                
                # 5. Actor梯度曲线（裁剪前后）
                actor_grad_before = [gd[2] for gd in gradient_data]
                actor_grad_after = [gd[3] for gd in gradient_data]
                actor_grad_before_smooth = self.calculate_sliding_window(actor_grad_before, window_size)
                actor_grad_after_smooth = self.calculate_sliding_window(actor_grad_after, window_size)
                
                ax = axes[plot_idx]
                # 对原始数据进行降采样以提高绘图性能
                if len(gradient_steps) > MAX_POINTS_PER_LINE:
                    steps_before, grad_before = self.downsample_for_plot(gradient_steps, actor_grad_before, MAX_POINTS_PER_LINE)
                    steps_after, grad_after = self.downsample_for_plot(gradient_steps, actor_grad_after, MAX_POINTS_PER_LINE)
                    ax.plot(steps_before, grad_before, 'r--', 
                           linewidth=PLOT_CONFIG['linewidth_raw'], 
                           alpha=PLOT_CONFIG['alpha_raw'], label='Actor Grad Before Clip (Raw)')
                    ax.plot(steps_after, grad_after, 'r-', 
                           linewidth=PLOT_CONFIG['linewidth_raw'], 
                           alpha=PLOT_CONFIG['alpha_raw'], label='Actor Grad After Clip (Raw)')
                else:
                    ax.plot(gradient_steps, actor_grad_before, 'r--', 
                           linewidth=PLOT_CONFIG['linewidth_raw'], 
                           alpha=PLOT_CONFIG['alpha_raw'], label='Actor Grad Before Clip (Raw)')
                    ax.plot(gradient_steps, actor_grad_after, 'r-', 
                           linewidth=PLOT_CONFIG['linewidth_raw'], 
                           alpha=PLOT_CONFIG['alpha_raw'], label='Actor Grad After Clip (Raw)')
                ax.plot(gradient_steps, actor_grad_before_smooth, 'r--', 
                       linewidth=PLOT_CONFIG['linewidth_smooth'], 
                       alpha=PLOT_CONFIG['alpha_smooth'], 
                       label=f'Actor Grad Before Clip (window={window_size})')
                ax.plot(gradient_steps, actor_grad_after_smooth, 'r-', 
                       linewidth=PLOT_CONFIG['linewidth_smooth'], 
                       alpha=PLOT_CONFIG['alpha_smooth'], 
                       label=f'Actor Grad After Clip (window={window_size})')
                ax.set_xlabel('Training Step', fontsize=PLOT_CONFIG['fontsize_label'])
                ax.set_ylabel('Actor Gradient L2 Norm', fontsize=PLOT_CONFIG['fontsize_label'])
                ax.set_title('Actor Gradient Norm (Before/After Clipping)', 
                           fontsize=PLOT_CONFIG['fontsize_title'], fontweight='bold')
                ax.grid(True, alpha=PLOT_CONFIG['grid_alpha'])
                ax.legend(loc='upper right', fontsize=PLOT_CONFIG['fontsize_legend'])
                # 如果梯度值范围很大，使用对数刻度
                actor_grad_after_positive = [x for x in actor_grad_after if x is not None and x > 0]
                if actor_grad_before and actor_grad_after_positive and max(actor_grad_before) / max(actor_grad_after_positive + [1]) > 10:
                    ax.set_yscale('log')
                    ax.set_ylabel('Actor Gradient L2 Norm (log scale)', fontsize=PLOT_CONFIG['fontsize_label'])
                # 添加统计信息（使用裁剪前的数据）
                self.add_statistics_text(ax, actor_grad_before, position='upper left', 
                                        fontsize=PLOT_CONFIG['fontsize_legend'], precision=4)
                plot_idx += 1
        
        # 熵值曲线和alpha梯度曲线（如果有熵值数据）
        if has_entropy:
            # 提取熵值和alpha梯度数据（过滤掉None值）
            entropy_data = []
            alpha_grad_data = []
            entropy_steps = []
            for tr in self.training_records:
                if tr[8] is not None:  # entropy不为None
                    entropy_steps.append(tr[0])
                    entropy_data.append(tr[8])  # entropy
                    alpha_grad_data.append(tr[9] if tr[9] is not None else 0.0)  # alpha_grad
            
            if entropy_data:
                # 6. 熵值曲线
                entropy_smooth = self.calculate_sliding_window(entropy_data, window_size)
                
                ax = axes[plot_idx]
                # 对原始数据进行降采样以提高绘图性能
                if len(entropy_steps) > MAX_POINTS_PER_LINE:
                    steps_raw, entropy_raw = self.downsample_for_plot(entropy_steps, entropy_data, MAX_POINTS_PER_LINE)
                    ax.plot(steps_raw, entropy_raw, 'm-', 
                           linewidth=PLOT_CONFIG['linewidth_raw'], 
                           alpha=PLOT_CONFIG['alpha_raw'], label='Raw')
                else:
                    ax.plot(entropy_steps, entropy_data, 'm-', 
                           linewidth=PLOT_CONFIG['linewidth_raw'], 
                           alpha=PLOT_CONFIG['alpha_raw'], label='Raw')
                ax.plot(entropy_steps, entropy_smooth, 'm-', 
                       linewidth=PLOT_CONFIG['linewidth_smooth'], 
                       alpha=PLOT_CONFIG['alpha_smooth'], 
                       label=f'Smoothed (window={window_size})')
                ax.set_xlabel('Training Step', fontsize=PLOT_CONFIG['fontsize_label'])
                ax.set_ylabel('Entropy', fontsize=PLOT_CONFIG['fontsize_label'])
                ax.set_title('Entropy Curve', fontsize=PLOT_CONFIG['fontsize_title'], fontweight='bold')
                ax.grid(True, alpha=PLOT_CONFIG['grid_alpha'])
                ax.legend(loc='upper right')
                # 添加统计信息
                self.add_statistics_text(ax, entropy_data, position='upper left', 
                                        fontsize=PLOT_CONFIG['fontsize_legend'], precision=4)
                plot_idx += 1
                
                # 7. Alpha梯度曲线
                alpha_grad_smooth = self.calculate_sliding_window(alpha_grad_data, window_size)
                
                ax = axes[plot_idx]
                # 对原始数据进行降采样以提高绘图性能
                if len(entropy_steps) > MAX_POINTS_PER_LINE:
                    steps_raw, alpha_grad_raw = self.downsample_for_plot(entropy_steps, alpha_grad_data, MAX_POINTS_PER_LINE)
                    ax.plot(steps_raw, alpha_grad_raw, 'c-', 
                           linewidth=PLOT_CONFIG['linewidth_raw'], 
                           alpha=PLOT_CONFIG['alpha_raw'], label='Raw')
                else:
                    ax.plot(entropy_steps, alpha_grad_data, 'c-', 
                           linewidth=PLOT_CONFIG['linewidth_raw'], 
                           alpha=PLOT_CONFIG['alpha_raw'], label='Raw')
                ax.plot(entropy_steps, alpha_grad_smooth, 'c-', 
                       linewidth=PLOT_CONFIG['linewidth_smooth'], 
                       alpha=PLOT_CONFIG['alpha_smooth'], 
                       label=f'Smoothed (window={window_size})')
                ax.set_xlabel('Training Step', fontsize=PLOT_CONFIG['fontsize_label'])
                ax.set_ylabel('Alpha Gradient L2 Norm', fontsize=PLOT_CONFIG['fontsize_label'])
                ax.set_title('Alpha Gradient Norm Curve', fontsize=PLOT_CONFIG['fontsize_title'], fontweight='bold')
                ax.grid(True, alpha=PLOT_CONFIG['grid_alpha'])
                ax.legend(loc='upper right')
                # 如果梯度值范围很大，使用对数刻度
                alpha_grad_positive = [x for x in alpha_grad_data if x is not None and x > 0]
                if alpha_grad_data and alpha_grad_positive and max(alpha_grad_data) / max(alpha_grad_positive + [1]) > 10:
                    ax.set_yscale('log')
                    ax.set_ylabel('Alpha Gradient L2 Norm (log scale)', fontsize=PLOT_CONFIG['fontsize_label'])
                # 添加统计信息
                self.add_statistics_text(ax, alpha_grad_data, position='upper left', 
                                        fontsize=PLOT_CONFIG['fontsize_legend'], precision=4)
                plot_idx += 1
        
        # 8. Episode成功率、碰撞率、超时率曲线（滑动窗口）
        if has_episodes:
            episodes, success_rates, collision_rates, timeout_rates = self.calculate_episode_rates(window_size)
            
            ax = axes[plot_idx]
            # 对数据进行降采样以提高绘图性能（所有三条线使用相同的x轴采样）
            if len(episodes) > MAX_POINTS_PER_LINE:
                eps_down, success_down = self.downsample_for_plot(episodes, success_rates, MAX_POINTS_PER_LINE)
                _, collision_down = self.downsample_for_plot(episodes, collision_rates, MAX_POINTS_PER_LINE)
                _, timeout_down = self.downsample_for_plot(episodes, timeout_rates, MAX_POINTS_PER_LINE)
                ax.plot(eps_down, success_down, 'g-', 
                       linewidth=PLOT_CONFIG['linewidth_smooth'], 
                       alpha=PLOT_CONFIG['alpha_smooth'], 
                       label=f'Success Rate (window={window_size})')
                ax.plot(eps_down, collision_down, 'r-', 
                       linewidth=PLOT_CONFIG['linewidth_smooth'], 
                       alpha=PLOT_CONFIG['alpha_smooth'], 
                       label=f'Collision Rate (window={window_size})')
                ax.plot(eps_down, timeout_down, 'orange', 
                       linewidth=PLOT_CONFIG['linewidth_smooth'], 
                       alpha=PLOT_CONFIG['alpha_smooth'], 
                       label=f'Timeout Rate (window={window_size})')
            else:
                ax.plot(episodes, success_rates, 'g-', 
                       linewidth=PLOT_CONFIG['linewidth_smooth'], 
                       alpha=PLOT_CONFIG['alpha_smooth'], 
                       label=f'Success Rate (window={window_size})')
                ax.plot(episodes, collision_rates, 'r-', 
                       linewidth=PLOT_CONFIG['linewidth_smooth'], 
                       alpha=PLOT_CONFIG['alpha_smooth'], 
                       label=f'Collision Rate (window={window_size})')
                ax.plot(episodes, timeout_rates, 'orange', 
                       linewidth=PLOT_CONFIG['linewidth_smooth'], 
                       alpha=PLOT_CONFIG['alpha_smooth'], 
                       label=f'Timeout Rate (window={window_size})')
            ax.set_xlabel('Episode Number', fontsize=PLOT_CONFIG['fontsize_label'])
            ax.set_ylabel('Rate', fontsize=PLOT_CONFIG['fontsize_label'])
            ax.set_title('Episode Rates (Success/Collision/Timeout)', 
                        fontsize=PLOT_CONFIG['fontsize_title'], fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=PLOT_CONFIG['grid_alpha'])
            ax.legend(loc='upper left')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            # 添加统计信息（使用成功率数据）
            self.add_statistics_text(ax, success_rates, position='upper right', 
                                    fontsize=PLOT_CONFIG['fontsize_legend'], precision=4)
            plot_idx += 1
        
        # 9. Reward Detail各项曲线（滑动窗口）
        if has_reward_details:
            episodes, reward_curves = self.calculate_reward_detail_curves(window_size)
            
            ax = axes[plot_idx]
            # 只绘制有非零数据的项，但跳过固定值（goal和collision通常是固定值）
            plotted_any = False
            skip_keys = ['goal', 'collision']  # 跳过固定值项
            
            for key in reward_curves.keys():
                if key in skip_keys:
                    continue  # 跳过固定值项
                # 检查是否有非零值且不是常数
                values = reward_curves[key]
                if any(abs(v) > 1e-6 for v in values):
                    # 检查是否是常数（所有值都相同）
                    if len(set([round(v, 2) for v in values if abs(v) > 1e-6])) > 1:
                        # 对数据进行降采样以提高绘图性能
                        if len(episodes) > MAX_POINTS_PER_LINE:
                            eps_down, values_down = self.downsample_for_plot(episodes, values, MAX_POINTS_PER_LINE)
                            ax.plot(eps_down, values_down, 
                                   color=REWARD_COLORS.get(key, 'gray'), 
                                   linewidth=PLOT_CONFIG['linewidth_smooth'], 
                                   alpha=PLOT_CONFIG['alpha_smooth'], 
                                   label=f'{REWARD_LABELS.get(key, key)} (window={window_size})')
                        else:
                            ax.plot(episodes, values, 
                                   color=REWARD_COLORS.get(key, 'gray'), 
                                   linewidth=PLOT_CONFIG['linewidth_smooth'], 
                                   alpha=PLOT_CONFIG['alpha_smooth'], 
                                   label=f'{REWARD_LABELS.get(key, key)} (window={window_size})')
                        plotted_any = True
            
            if plotted_any:
                ax.set_xlabel('Episode Number', fontsize=PLOT_CONFIG['fontsize_label'])
                ax.set_ylabel('Reward Value', fontsize=PLOT_CONFIG['fontsize_label'])
                ax.set_title('Reward Detail Components (Variable)', 
                           fontsize=PLOT_CONFIG['fontsize_title'], fontweight='bold')
                ax.grid(True, alpha=PLOT_CONFIG['grid_alpha'])
                ax.legend(loc='best', fontsize=PLOT_CONFIG['fontsize_legend'])
                # 添加统计信息（使用第一个有数据的reward项）
                for key in reward_curves.keys():
                    if key in skip_keys:
                        continue
                    values = reward_curves[key]
                    if any(abs(v) > 1e-6 for v in values) and len(set([round(v, 2) for v in values if abs(v) > 1e-6])) > 1:
                        self.add_statistics_text(ax, values, position='upper right', 
                                                fontsize=PLOT_CONFIG['fontsize_legend'], precision=4)
                        break  # 只显示第一个有数据的项的统计信息
            plot_idx += 1
        
        plt.tight_layout()
        
        # 保存图片
        if output_dir is None:
            output_dir = self.log_file.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = self.log_file.stem
        output_file = output_dir / f'{base_name}_metrics.png'
        plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"\n曲线图已保存至: {output_file}")
        
        plt.close(fig)
    
    def print_statistics(self):
        """打印统计结果"""
        if self.training_records:
            print("\n" + "="*80)
            print("【训练记录统计】")
            print("="*80)
            print(f"总训练次数: {len(self.training_records)}")
            if self.training_records:
                critic_losses = [tr[1] for tr in self.training_records]
                actor_losses = [tr[2] for tr in self.training_records]
                avg_sample_times = [tr[3] for tr in self.training_records]
                print(f"Critic Loss - 最小值: {min(critic_losses):.2f}, 最大值: {max(critic_losses):.2f}, 平均值: {np.mean(critic_losses):.2f}")
                print(f"Actor Loss - 最小值: {min(actor_losses):.2f}, 最大值: {max(actor_losses):.2f}, 平均值: {np.mean(actor_losses):.2f}")
                print(f"平均抽样次数 - 最小值: {min(avg_sample_times):.2f}, 最大值: {max(avg_sample_times):.2f}, 平均值: {np.mean(avg_sample_times):.2f}")
                
                # 梯度统计（如果有梯度数据）
                gradient_records = [tr for tr in self.training_records if tr[4] is not None]
                if gradient_records:
                    critic_grad_before = [tr[4] for tr in gradient_records if tr[4] is not None]
                    critic_grad_after = [tr[5] for tr in gradient_records if tr[5] is not None]
                    actor_grad_before = [tr[6] for tr in gradient_records if tr[6] is not None]
                    actor_grad_after = [tr[7] for tr in gradient_records if tr[7] is not None]
                    if critic_grad_before:
                        print(f"\n梯度统计（共{len(gradient_records)}条记录）:")
                        print(f"Critic梯度（裁剪前） - 最小值: {min(critic_grad_before):.6f}, 最大值: {max(critic_grad_before):.6f}, 平均值: {np.mean(critic_grad_before):.6f}")
                    if critic_grad_after:
                        print(f"Critic梯度（裁剪后） - 最小值: {min(critic_grad_after):.6f}, 最大值: {max(critic_grad_after):.6f}, 平均值: {np.mean(critic_grad_after):.6f}")
                    if actor_grad_before:
                        print(f"Actor梯度（裁剪前） - 最小值: {min(actor_grad_before):.6f}, 最大值: {max(actor_grad_before):.6f}, 平均值: {np.mean(actor_grad_before):.6f}")
                    if actor_grad_after:
                        print(f"Actor梯度（裁剪后） - 最小值: {min(actor_grad_after):.6f}, 最大值: {max(actor_grad_after):.6f}, 平均值: {np.mean(actor_grad_after):.6f}")
                
                # 熵值和alpha梯度统计（如果有熵值数据）
                entropy_records = [tr for tr in self.training_records if tr[8] is not None]
                if entropy_records:
                    entropies = [tr[8] for tr in entropy_records]
                    alpha_grads = [tr[9] for tr in entropy_records if tr[9] is not None]
                    print(f"\n熵值和Alpha梯度统计（共{len(entropy_records)}条记录）:")
                    print(f"熵值 - 最小值: {min(entropies):.6f}, 最大值: {max(entropies):.6f}, 平均值: {np.mean(entropies):.6f}")
                    if alpha_grads:
                        print(f"Alpha梯度L2范数 - 最小值: {min(alpha_grads):.6f}, 最大值: {max(alpha_grads):.6f}, 平均值: {np.mean(alpha_grads):.6f}")
                
                # 训练耗时统计
                if self.training_durations:
                    print(f"\n训练耗时统计（共{len(self.training_durations)}条记录）:")
                    print(f"平均每次训练耗时: {np.mean(self.training_durations):.2f}秒")
                    print(f"训练耗时 - 最小值: {min(self.training_durations):.2f}秒, 最大值: {max(self.training_durations):.2f}秒")
                    total_training_time = sum(self.training_durations)
                    print(f"总训练耗时: {total_training_time:.2f}秒 ({total_training_time/60:.2f}分钟)")
                    
                    # 平均每秒抽样样本step数量（总抽样step数量/总训练耗时）
                    # 注意：日志中的"总抽样数"已经是累加值，所以直接取最后一次的值即可
                    if self.total_sample_count and total_training_time > 0:
                        # 总抽样数是累加的，取最后一次的值
                        total_sample_count_sum = self.total_sample_count[-1] if self.total_sample_count else 0
                        avg_samples_per_second = total_sample_count_sum / total_training_time
                        print(f"平均每秒抽样样本step数量: {avg_samples_per_second:.2f} steps/秒")
                        print(f"总抽样step数量: {total_sample_count_sum}")
        
        # 最好模型统计
        if self.best_model_records:
            print("\n" + "="*80)
            print("【最好模型统计】")
            print("="*80)
            print(f"总共保存了 {len(self.best_model_records)} 次最好模型")
            if self.best_model_info:
                episode_num, success_rate, collision_rate, save_path = self.best_model_info
                print(f"\n最高成功率记录:")
                print(f"  Episode编号: {episode_num}")
                print(f"  成功率: {success_rate:.4f} ({success_rate*100:.2f}%)")
                print(f"  碰撞率: {collision_rate:.4f} ({collision_rate*100:.2f}%)")
                print(f"  模型保存路径: {save_path}")
            else:
                print("未找到最好模型记录")
        
        if self.episodes:
            print("\n" + "="*80)
            print("【Episode统计】")
            print("="*80)
            total = len(self.episodes)
            end_statuses = np.array([ep[2] for ep in self.episodes])
            goal_count = int(np.sum(end_statuses == 'Goal'))
            collision_count = int(np.sum(end_statuses == 'Collision'))
            timeout_count = int(np.sum(end_statuses == 'Timeout'))
            force_stop_count = int(np.sum(end_statuses == 'ForceStop'))
            denom = goal_count + collision_count + timeout_count
            print(f"总episode数: {total}")
            if denom > 0:
                print(f"到达终点: {goal_count} ({goal_count/denom*100:.1f}% 占统计样本)")
                print(f"发生碰撞: {collision_count} ({collision_count/denom*100:.1f}% 占统计样本)")
                print(f"超时结束: {timeout_count} ({timeout_count/denom*100:.1f}% 占统计样本)")
            else:
                print(f"到达终点: {goal_count}")
                print(f"发生碰撞: {collision_count}")
                print(f"超时结束: {timeout_count}")
            if force_stop_count > 0:
                print(f"ForceStop (忽略，不参与三率): {force_stop_count} ({force_stop_count/total*100:.1f}% 占全部)")
            
            # 统计每秒产生的平均样本数（总样本数/从第一条episode数据产生时间开始，到日志中最后一条episode数据产生结束的总时间）
            episode_timestamps = [ep[5] for ep in self.episodes if ep[5] is not None]  # timestamp是第6个元素（索引5）
            if episode_timestamps and len(episode_timestamps) > 1:
                first_timestamp = min(episode_timestamps)
                last_timestamp = max(episode_timestamps)
                total_time_seconds = (last_timestamp - first_timestamp).total_seconds()
                
                if total_time_seconds > 0:
                    # 计算总样本数（所有episode的steps之和）
                    total_sample_steps_from_episodes = sum([ep[4] for ep in self.episodes if ep[4] > 0])  # steps是第5个元素（索引4）
                    avg_samples_per_second = total_sample_steps_from_episodes / total_time_seconds
                    print(f"\n时间范围统计:")
                    print(f"第一条episode时间: {first_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"最后一条episode时间: {last_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"总时间跨度: {total_time_seconds:.2f}秒 ({total_time_seconds/60:.2f}分钟)")
                    print(f"总样本数（所有episode的steps之和）: {total_sample_steps_from_episodes}")
                    print(f"每秒产生的平均样本数: {avg_samples_per_second:.2f} samples/秒")
            
            # 统计每个step平均的Reward Detail（不包含goal和collision）
            per_step_stats = self.calculate_per_step_reward_stats()
            if per_step_stats and per_step_stats.get('valid_episodes', 0) > 0:
                print("\n" + "="*80)
                print("【每Step平均Reward Detail统计】（不包含goal和collision）")
                print("="*80)
                print(f"有效episode数: {per_step_stats['valid_episodes']}")
                print(f"总step数: {per_step_stats['total_steps']}")
                print("\n每个step的平均Reward Detail:")
                reward_keys = ['angle', 'linear', 'target_distance', 'obs', 'yawrate']
                for key in reward_keys:
                    if key in per_step_stats:
                        print(f"  {REWARD_LABELS.get(key, key)}: {per_step_stats[key]:.4f}")
            
            # Reward Detail统计（不含 ForceStop）
            reward_eps = [ep for ep in self.episodes if ep[2] in COUNTED_END_STATUSES]
            if reward_eps and any(ep[3] for ep in reward_eps if any(ep[3].values())):
                print("\n" + "="*80)
                print("【Reward Detail统计】（仅 Goal/Collision/Timeout，不含 ForceStop）")
                print("="*80)
                reward_keys = ['goal', 'collision', 'angle', 'linear', 'target_distance', 'obs', 'yawrate']

                print("所有统计Episode的平均值:")
                for key in reward_keys:
                    all_values = np.array([ep[3].get(key, 0.0) for ep in reward_eps], dtype=np.float64)
                    non_zero_mask = np.abs(all_values) > 1e-6
                    non_zero_values = all_values[non_zero_mask]
                    
                    if len(non_zero_values) > 0:
                        avg_all = np.mean(all_values)
                        avg_non_zero = np.mean(non_zero_values)
                        min_val = np.min(non_zero_values)
                        max_val = np.max(non_zero_values)
                        count_non_zero = len(non_zero_values)
                        total_count = len(all_values)
                        
                        # 检查是否是固定值（使用numpy的unique）
                        unique_vals = np.unique(np.round(non_zero_values, 2))
                        if len(unique_vals) == 1:
                            print(f"  {REWARD_LABELS.get(key, key)}: {non_zero_values[0]:.2f} (固定值, 出现{count_non_zero}次)")
                        else:
                            print(f"  {REWARD_LABELS.get(key, key)}: 平均值={avg_all:.2f}, "
                                  f"范围=[{min_val:.2f}, {max_val:.2f}], 非零次数={count_non_zero}/{total_count}")
                    else:
                        print(f"  {REWARD_LABELS.get(key, key)}: 0.00 (未出现)")
                
                # 按episode类型分组统计（Goal/Collision/Timeout 有明细；ForceStop 仅计数、忽略）
                print("\n按Episode类型分组的平均值:")
                for end_status in ['Goal', 'Collision', 'Timeout']:
                    status_episodes = [ep for ep in self.episodes if ep[2] == end_status]
                    if not status_episodes:
                        continue

                    print(f"\n  {end_status} ({len(status_episodes)}个episode):")
                    for key in reward_keys:
                        # 使用numpy向量化操作
                        values = np.array([ep[3].get(key, 0.0) for ep in status_episodes], dtype=np.float64)
                        non_zero_mask = np.abs(values) > 1e-6
                        non_zero_values = values[non_zero_mask]
                        
                        if len(non_zero_values) > 0:
                            avg_val = np.mean(values)
                            unique_vals = np.unique(np.round(non_zero_values, 2))
                            if len(unique_vals) == 1:
                                print(f"    {REWARD_LABELS.get(key, key)}: {non_zero_values[0]:.2f} (固定值)")
                            else:
                                print(f"    {REWARD_LABELS.get(key, key)}: 平均值={avg_val:.2f}, "
                                      f"范围=[{np.min(non_zero_values):.2f}, {np.max(non_zero_values):.2f}]")

                force_stop_eps = [ep for ep in self.episodes if ep[2] == 'ForceStop']
                if force_stop_eps:
                    print(f"\n  ForceStop ({len(force_stop_eps)}个episode): 忽略，不参与三率与 Reward 统计")

    def run(self, plot: bool = True, output_dir: Optional[str] = None, window_size: int = 100):
        """运行分析"""
        self.parse_log()
        
        if not self.training_records and not self.episodes:
            print("错误：未找到有效的训练数据")
            return
        
        self.print_statistics()
        
        if plot:
            self.plot_curves(output_dir, window_size)


def main():
    parser = argparse.ArgumentParser(
        description='训练指标分析与可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例用法:
  python3 {sys.argv[0]} /path/to/train.log
  python3 {sys.argv[0]} /path/to/train.log --window-size 200
  python3 {sys.argv[0]} /path/to/train.log --no-plot
  python3 {sys.argv[0]} /path/to/train.log --output-dir ./output

注意: 命令行参数会覆盖文件开头的默认配置参数
        """
    )
    parser.add_argument('log_file', type=str, help='训练日志文件路径')
    parser.add_argument('--no-plot', action='store_true', 
                       help=f'不生成曲线图（默认: {"不生成" if not DEFAULT_GENERATE_PLOT else "生成"}）')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR, 
                       help=f'输出目录（默认: {"日志文件所在目录" if DEFAULT_OUTPUT_DIR is None else DEFAULT_OUTPUT_DIR}）')
    parser.add_argument('--window-size', type=int, default=DEFAULT_WINDOW_SIZE, 
                       help=f'滑动窗口大小（默认: {DEFAULT_WINDOW_SIZE}）')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"错误：文件不存在 - {args.log_file}")
        sys.exit(1)
    
    # 解析参数（命令行参数优先）
    generate_plot = DEFAULT_GENERATE_PLOT if not args.no_plot else False
    output_dir = args.output_dir
    window_size = args.window_size
    
    # 确定输出目录（和图片同目录）
    if output_dir is None:
        output_dir = log_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建输出txt文件路径（和图片同目录）
    base_name = log_path.stem
    output_txt_file = output_dir / f'{base_name}_output.txt'
    
    # 设置输出重定向（同时输出到终端和文件）
    original_stdout = sys.stdout
    tee = TeeOutput(output_txt_file)
    sys.stdout = tee
    
    try:
        print(f"配置参数:")
        print(f"  日志文件: {log_path}")
        print(f"  窗口大小: {window_size}")
        print(f"  生成图片: {generate_plot}")
        print(f"  输出目录: {output_dir}")
        print(f"  输出文本文件: {output_txt_file}")
        print()
        
        # 创建分析器
        analyzer = TrainingMetricsAnalyzer(str(log_path))
        analyzer.run(plot=generate_plot, output_dir=str(output_dir), window_size=window_size)
        
        print(f"\n终端输出已保存至: {output_txt_file}")
    finally:
        # 恢复原始输出
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    main()

