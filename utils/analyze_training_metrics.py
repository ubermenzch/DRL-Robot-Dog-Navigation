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
DEFAULT_WINDOW_SIZE = 100

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

# 尝试导入matplotlib，如果失败则只提供统计功能
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
    # 设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    matplotlib.rcParams['axes.unicode_minus'] = False
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib未安装，将跳过绘图功能")


class TrainingMetricsAnalyzer:
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.training_records = []  # (training_step, critic_loss, actor_loss, avg_sample_times)
        self.episodes = []  # (episode_num, env_id, end_status, reward_details)
        # reward_details: Dict with keys: goal, collision, angle, linear, target_distance, obs, yawrate
        
    def parse_training_record(self, line: str) -> Optional[Tuple]:
        """解析训练完成记录"""
        # 格式：第X次训练完成
        match = re.search(r'第(\d+)次训练完成', line)
        if match:
            training_step = int(match.group(1))
            return ('training_start', training_step)
        return None
    
    def parse_training_details(self, lines: List[str], start_idx: int) -> Optional[Tuple]:
        """解析训练详细信息（多行）"""
        if start_idx + 2 >= len(lines):
            return None
        
        try:
            # 第一行：总抽样数、总样本数、样本平均抽样次数
            # 格式：  总抽样数: X | 总样本数: Y | 样本平均抽样次数: Z.ZZ
            line1 = lines[start_idx].strip()
            sample_match = re.search(r'样本平均抽样次数:\s+([\d.]+)', line1)
            if not sample_match:
                return None
            avg_sample_times = float(sample_match.group(1))
            
            # 第二行：critic和actor网络损失
            # 格式：  本次训练的平均critic网络损失: X.XX | 前10次训练的平均critic网络损失: Y.YY
            line2 = lines[start_idx + 1].strip()
            critic_match = re.search(r'本次训练的平均critic网络损失:\s+([-\d.]+)', line2)
            if not critic_match:
                return None
            critic_loss = float(critic_match.group(1))
            
            # 第三行：actor网络损失
            # 格式：  本次训练的平均actor网络损失: X.XX | 前10次训练的平均actor网络损失: Y.YY
            line3 = lines[start_idx + 2].strip()
            actor_match = re.search(r'本次训练的平均actor网络损失:\s+([-\d.]+)', line3)
            if not actor_match:
                return None
            actor_loss = float(actor_match.group(1))
            
            return ('training_details', avg_sample_times, critic_loss, actor_loss)
        except (IndexError, ValueError, AttributeError):
            return None
    
    def parse_reward_detail(self, line: str) -> Dict:
        """解析Reward Detail行"""
        # 格式：  Reward Detail: end=Status, total_reward=X.XX, goal=Y.YY, collision=Z.ZZ, angle=W.WW, linear=V.VV, target_distance=U.UU
        detail = {}
        # 使用更精确的正则表达式，确保匹配的是浮点数格式（包含可选负号和数字+小数点）
        patterns = {
            'goal': r'goal=(-?\d+\.?\d*)',
            'collision': r'collision=(-?\d+\.?\d*)',
            'angle': r'angle=(-?\d+\.?\d*)',
            'linear': r'linear=(-?\d+\.?\d*)',
            'target_distance': r'target_distance=(-?\d+\.?\d*)',
            'obs': r'obs=(-?\d+\.?\d*)',
            'yawrate': r'yawrate=(-?\d+\.?\d*)',
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                try:
                    detail[key] = float(match.group(1))
                except ValueError:
                    detail[key] = 0.0  # 如果转换失败，默认为0
            else:
                detail[key] = 0.0  # 如果没有找到，默认为0
        return detail
    
    def parse_episode(self, line: str, prev_line: str = "") -> Optional[Tuple]:
        """解析episode信息"""
        # 检查是否是Reward Detail行
        if 'Reward Detail:' in line:
            # 从Reward Detail行提取end状态和reward details
            end_match = re.search(r'end=(Goal|Collision|Timeout)', line)
            reward_detail = self.parse_reward_detail(line)
            
            if end_match and prev_line:
                # 从上一行提取episode编号和环境ID
                # 匹配格式：环境 X Episode: Y Target Distance: A.AA (actual: B.BB) Steps: N
                pattern1 = r'环境\s+(\d+)\s+Episode:\s+(\d+)'
                
                match = re.search(pattern1, prev_line)
                if match:
                    env_id = int(match.group(1))
                    episode_num = int(match.group(2))
                    end_status = end_match.group(1)
                    return ('episode', env_id, episode_num, end_status, reward_detail)
        
        # 向后兼容：尝试匹配旧格式
        pattern_old = r'环境\s+(\d+)\s+Episode:\s+(\d+).*?End:\s+(Goal|Collision|Timeout)'
        match = re.search(pattern_old, line)
        if match:
            env_id = int(match.group(1))
            episode_num = int(match.group(2))
            end_status = match.group(3)
            # 旧格式没有reward detail，使用空字典
            reward_detail = {'goal': 0.0, 'collision': 0.0, 'angle': 0.0, 'linear': 0.0, 
                           'target_distance': 0.0, 'obs': 0.0, 'yawrate': 0.0}
            return ('episode', env_id, episode_num, end_status, reward_detail)
        
        return None
    
    def parse_log(self):
        """解析日志文件"""
        print(f"正在解析日志文件: {self.log_file}")
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"日志文件总行数: {len(lines)}")
        
        current_training_step = None
        prev_line = ""
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # 解析训练记录
            training_start = self.parse_training_record(line)
            if training_start:
                current_training_step = training_start[1]
                # 尝试解析接下来的训练详细信息
                if i + 3 < len(lines):
                    training_details = self.parse_training_details(lines, i + 1)
                    if training_details:
                        _, avg_sample_times, critic_loss, actor_loss = training_details
                        self.training_records.append((
                            current_training_step, critic_loss, actor_loss, avg_sample_times
                        ))
                        i += 4  # 跳过已解析的行
                        continue
            
            # 解析Episode
            episode_info = self.parse_episode(line, prev_line)
            if episode_info and episode_info[0] == 'episode':
                _, env_id, episode_num, end_status, reward_detail = episode_info
                self.episodes.append((episode_num, env_id, end_status, reward_detail))
            
            prev_line = line
            i += 1
        
        # 按训练步数和episode编号排序
        self.training_records.sort(key=lambda x: x[0])
        self.episodes.sort(key=lambda x: x[0])
        
        print(f"成功解析 {len(self.training_records)} 个训练记录")
        print(f"成功解析 {len(self.episodes)} 个episode")
    
    def calculate_sliding_window(self, data: List[float], window_size: int) -> List[float]:
        """计算滑动窗口平均值"""
        if not data:
            return []
        
        result = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size + 1)
            window_data = data[start_idx:i+1]
            avg = np.mean(window_data) if window_data else 0
            result.append(avg)
        return result
    
    def calculate_episode_rates(self, window_size: int) -> Tuple[List[int], List[float], List[float], List[float]]:
        """计算episode的成功率、碰撞率、超时率（滑动窗口）"""
        if not self.episodes:
            return [], [], [], []
        
        episodes = [ep[0] for ep in self.episodes]
        successes = [1 if ep[2] == 'Goal' else 0 for ep in self.episodes]
        collisions = [1 if ep[2] == 'Collision' else 0 for ep in self.episodes]
        timeouts = [1 if ep[2] == 'Timeout' else 0 for ep in self.episodes]
        
        success_rates = self.calculate_sliding_window(successes, window_size)
        collision_rates = self.calculate_sliding_window(collisions, window_size)
        timeout_rates = self.calculate_sliding_window(timeouts, window_size)
        
        return episodes, success_rates, collision_rates, timeout_rates
    
    def calculate_reward_detail_curves(self, window_size: int) -> Tuple[List[int], Dict[str, List[float]]]:
        """计算reward detail各项的滑动窗口平均值"""
        if not self.episodes:
            return [], {}
        
        episodes = [ep[0] for ep in self.episodes]
        reward_keys = ['goal', 'collision', 'angle', 'linear', 'target_distance', 'obs', 'yawrate']
        
        reward_curves = {}
        for key in reward_keys:
            values = [ep[3].get(key, 0.0) for ep in self.episodes]
            smoothed = self.calculate_sliding_window(values, window_size)
            reward_curves[key] = smoothed
        
        return episodes, reward_curves
    
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
        has_reward_details = has_episodes and any(ep[3] for ep in self.episodes if any(ep[3].values()))
        
        if has_training:
            num_plots += 3  # critic loss, actor loss, avg_sample_times
        if has_episodes:
            num_plots += 1  # episode rates
        if has_reward_details:
            num_plots += 1  # reward detail curves
        
        if num_plots == 0:
            print("错误：没有可绘制的数据")
            return
        
        # 创建图形
        fig, axes = plt.subplots(num_plots, 1, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1]*num_plots))
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # 1. Critic Loss曲线（滑动窗口）
        if has_training:
            training_steps = [tr[0] for tr in self.training_records]
            critic_losses = [tr[1] for tr in self.training_records]
            critic_losses_smooth = self.calculate_sliding_window(critic_losses, window_size)
            
            ax = axes[plot_idx]
            ax.plot(training_steps, critic_losses, 'b-', 
                   linewidth=PLOT_CONFIG['linewidth_raw'], 
                   alpha=PLOT_CONFIG['alpha_raw'], label='Raw')
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
            plot_idx += 1
            
            # 2. Actor Loss曲线（滑动窗口）
            actor_losses = [tr[2] for tr in self.training_records]
            actor_losses_smooth = self.calculate_sliding_window(actor_losses, window_size)
            
            ax = axes[plot_idx]
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
            plot_idx += 1
            
            # 3. 样本平均抽样次数曲线（滑动窗口）
            avg_sample_times = [tr[3] for tr in self.training_records]
            avg_sample_times_smooth = self.calculate_sliding_window(avg_sample_times, window_size)
            
            ax = axes[plot_idx]
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
            plot_idx += 1
        
        # 4. Episode成功率、碰撞率、超时率曲线（滑动窗口）
        if has_episodes:
            episodes, success_rates, collision_rates, timeout_rates = self.calculate_episode_rates(window_size)
            
            ax = axes[plot_idx]
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
            plot_idx += 1
        
        # 5. Reward Detail各项曲线（滑动窗口）
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
        
        if self.episodes:
            print("\n" + "="*80)
            print("【Episode统计】")
            print("="*80)
            total = len(self.episodes)
            goal_count = sum(1 for ep in self.episodes if ep[2] == 'Goal')
            collision_count = sum(1 for ep in self.episodes if ep[2] == 'Collision')
            timeout_count = sum(1 for ep in self.episodes if ep[2] == 'Timeout')
            print(f"总episode数: {total}")
            print(f"到达终点: {goal_count} ({goal_count/total*100:.1f}%)")
            print(f"发生碰撞: {collision_count} ({collision_count/total*100:.1f}%)")
            print(f"超时结束: {timeout_count} ({timeout_count/total*100:.1f}%)")
            
            # Reward Detail统计
            if any(ep[3] for ep in self.episodes if any(ep[3].values())):
                print("\n" + "="*80)
                print("【Reward Detail统计】")
                print("="*80)
                reward_keys = ['goal', 'collision', 'angle', 'linear', 'target_distance', 'obs', 'yawrate']
                
                # 统计所有episode的平均值
                print("所有Episode的平均值:")
                for key in reward_keys:
                    # 获取所有值（包括0值）
                    all_values = [ep[3].get(key, 0.0) for ep in self.episodes]
                    non_zero_values = [v for v in all_values if abs(v) > 1e-6]
                    
                    if non_zero_values:
                        avg_all = np.mean(all_values)
                        avg_non_zero = np.mean(non_zero_values)
                        min_val = min(non_zero_values)
                        max_val = max(non_zero_values)
                        count_non_zero = len(non_zero_values)
                        total_count = len(all_values)
                        
                        # 检查是否是固定值
                        unique_vals = set([round(v, 2) for v in non_zero_values])
                        if len(unique_vals) == 1:
                            print(f"  {REWARD_LABELS.get(key, key)}: {non_zero_values[0]:.2f} (固定值, 出现{count_non_zero}次)")
                        else:
                            print(f"  {REWARD_LABELS.get(key, key)}: 平均值={avg_all:.2f}, "
                                  f"范围=[{min_val:.2f}, {max_val:.2f}], 非零次数={count_non_zero}/{total_count}")
                    else:
                        print(f"  {REWARD_LABELS.get(key, key)}: 0.00 (未出现)")
                
                # 按episode类型分组统计
                print("\n按Episode类型分组的平均值:")
                for end_status in ['Goal', 'Collision', 'Timeout']:
                    status_episodes = [ep for ep in self.episodes if ep[2] == end_status]
                    if not status_episodes:
                        continue
                    
                    print(f"\n  {end_status} ({len(status_episodes)}个episode):")
                    for key in reward_keys:
                        values = [ep[3].get(key, 0.0) for ep in status_episodes]
                        non_zero_values = [v for v in values if abs(v) > 1e-6]
                        
                        if non_zero_values:
                            avg_val = np.mean(values)
                            if len(set([round(v, 2) for v in non_zero_values])) == 1:
                                print(f"    {REWARD_LABELS.get(key, key)}: {non_zero_values[0]:.2f} (固定值)")
                            else:
                                print(f"    {REWARD_LABELS.get(key, key)}: 平均值={avg_val:.2f}, "
                                      f"范围=[{min(non_zero_values):.2f}, {max(non_zero_values):.2f}]")
    
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
    
    print(f"配置参数:")
    print(f"  日志文件: {log_path}")
    print(f"  窗口大小: {window_size}")
    print(f"  生成图片: {generate_plot}")
    print(f"  输出目录: {output_dir if output_dir else '日志文件所在目录'}")
    print()
    
    analyzer = TrainingMetricsAnalyzer(str(log_path))
    analyzer.run(plot=generate_plot, output_dir=output_dir, window_size=window_size)


if __name__ == "__main__":
    main()

