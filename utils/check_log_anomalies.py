#!/usr/bin/env python3
"""
检查日志文件中的异常值
检查项包括：
- position的x、y轴是否超过阈值（默认160）
- 速度是否异常
- 加速度是否异常
- 频率是否异常
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


class LogAnomalyChecker:
    def __init__(self, log_file: str, position_threshold: float = 160.0):
        """
        初始化日志异常检查器
        
        Args:
            log_file: 日志文件路径
            position_threshold: position的x、y轴阈值，默认160
        """
        self.log_file = Path(log_file)
        self.position_threshold = position_threshold
        self.anomalies = []
        
    def parse_position(self, line: str) -> Tuple[float, float, float]:
        """解析position值"""
        match = re.search(r'position=\(([-\d.]+),([-\d.]+),([-\d.]+)\)', line)
        if match:
            return float(match.group(1)), float(match.group(2)), float(match.group(3))
        return None
    
    def parse_linear_vel(self, line: str) -> Tuple[float, float, float]:
        """解析linear_vel值"""
        match = re.search(r'linear_vel=\(([-\d.]+),([-\d.]+),([-\d.]+)\)', line)
        if match:
            return float(match.group(1)), float(match.group(2)), float(match.group(3))
        return None
    
    def parse_angular_vel(self, line: str) -> Tuple[float, float, float]:
        """解析angular_vel值"""
        match = re.search(r'angular_vel=\(([-\d.]+),([-\d.]+),([-\d.]+)\)', line)
        if match:
            return float(match.group(1)), float(match.group(2)), float(match.group(3))
        return None
    
    def parse_linear_acc(self, line: str) -> Tuple[float, float, float]:
        """解析linear_acc值"""
        match = re.search(r'linear_acc=\(([-\d.]+),([-\d.]+),([-\d.]+)\)', line)
        if match:
            return float(match.group(1)), float(match.group(2)), float(match.group(3))
        return None
    
    def parse_frequency(self, line: str) -> Tuple[float, float]:
        """解析频率值"""
        match = re.search(r'callback_freq=([\d.]+)Hz.*update_freq=([\d.]+)Hz.*limit=([\d.]+)Hz', line)
        if match:
            return float(match.group(1)), float(match.group(2)), float(match.group(3))
        return None
    
    def check_position(self, line: str, line_num: int) -> bool:
        """检查position是否异常"""
        position = self.parse_position(line)
        if position is None:
            return False
        
        x, y, z = position
        has_anomaly = False
        
        if abs(x) > self.position_threshold:
            self.anomalies.append({
                'line': line_num,
                'type': 'position_x_exceeded',
                'value': x,
                'threshold': self.position_threshold,
                'message': f'Position x={x:.4f} 超过阈值 {self.position_threshold}',
                'raw_line': line.strip()
            })
            has_anomaly = True
        
        if abs(y) > self.position_threshold:
            self.anomalies.append({
                'line': line_num,
                'type': 'position_y_exceeded',
                'value': y,
                'threshold': self.position_threshold,
                'message': f'Position y={y:.4f} 超过阈值 {self.position_threshold}',
                'raw_line': line.strip()
            })
            has_anomaly = True
        
        # 检查z轴是否异常（通常z应该接近0）
        if abs(z) > 1.0:
            self.anomalies.append({
                'line': line_num,
                'type': 'position_z_abnormal',
                'value': z,
                'threshold': 1.0,
                'message': f'Position z={z:.4f} 异常（通常应接近0）',
                'raw_line': line.strip()
            })
            has_anomaly = True
        
        return has_anomaly
    
    def check_velocity(self, line: str, line_num: int) -> bool:
        """检查速度是否异常"""
        has_anomaly = False
        
        linear_vel = self.parse_linear_vel(line)
        if linear_vel:
            vx, vy, vz = linear_vel
            speed = (vx**2 + vy**2 + vz**2)**0.5
            
            # 检查线速度是否异常大（假设正常速度不超过5 m/s）
            if speed > 5.0:
                self.anomalies.append({
                    'line': line_num,
                    'type': 'linear_velocity_exceeded',
                    'value': speed,
                    'threshold': 5.0,
                    'message': f'线速度 {speed:.4f} m/s 异常大（正常应<5 m/s）',
                    'raw_line': line.strip()
                })
                has_anomaly = True
        
        angular_vel = self.parse_angular_vel(line)
        if angular_vel:
            wx, wy, wz = angular_vel
            angular_speed = (wx**2 + wy**2 + wz**2)**0.5
            
            # 检查角速度是否异常大（假设正常角速度不超过2 rad/s）
            if angular_speed > 2.0:
                self.anomalies.append({
                    'line': line_num,
                    'type': 'angular_velocity_exceeded',
                    'value': angular_speed,
                    'threshold': 2.0,
                    'message': f'角速度 {angular_speed:.4f} rad/s 异常大（正常应<2 rad/s）',
                    'raw_line': line.strip()
                })
                has_anomaly = True
        
        return has_anomaly
    
    def check_acceleration(self, line: str, line_num: int) -> bool:
        """检查加速度是否异常"""
        linear_acc = self.parse_linear_acc(line)
        if linear_acc is None:
            return False
        
        ax, ay, az = linear_acc
        acc_magnitude = (ax**2 + ay**2 + az**2)**0.5
        
        # 检查加速度是否异常（重力加速度约9.8，加上运动加速度，正常应<20 m/s²）
        if acc_magnitude > 20.0:
            self.anomalies.append({
                'line': line_num,
                'type': 'acceleration_exceeded',
                'value': acc_magnitude,
                'threshold': 20.0,
                'message': f'加速度 {acc_magnitude:.4f} m/s² 异常大（正常应<20 m/s²）',
                'raw_line': line.strip()
            })
            return True
        
        return False
    
    def check_frequency(self, line: str, line_num: int) -> bool:
        """检查频率是否异常"""
        freq_data = self.parse_frequency(line)
        if freq_data is None:
            return False
        
        callback_freq, update_freq, limit = freq_data
        has_anomaly = False
        
        # 检查回调频率是否过低（低于限制的10%视为异常）
        if callback_freq < limit * 0.1:
            self.anomalies.append({
                'line': line_num,
                'type': 'callback_frequency_too_low',
                'value': callback_freq,
                'threshold': limit * 0.1,
                'message': f'回调频率 {callback_freq:.2f} Hz 过低（限制为 {limit} Hz）',
                'raw_line': line.strip()
            })
            has_anomaly = True
        
        # 检查更新频率是否超过限制
        if update_freq > limit:
            self.anomalies.append({
                'line': line_num,
                'type': 'update_frequency_exceeded',
                'value': update_freq,
                'threshold': limit,
                'message': f'更新频率 {update_freq:.2f} Hz 超过限制 {limit} Hz',
                'raw_line': line.strip()
            })
            has_anomaly = True
        
        return has_anomaly
    
    def check_line(self, line: str, line_num: int):
        """检查单行日志"""
        # 只检查包含SUBSCRIBER的行
        if '[SUBSCRIBER]' not in line:
            return
        
        self.check_position(line, line_num)
        self.check_velocity(line, line_num)
        self.check_acceleration(line, line_num)
        self.check_frequency(line, line_num)
    
    def analyze(self) -> Dict:
        """分析整个日志文件"""
        if not self.log_file.exists():
            print(f"错误：日志文件不存在: {self.log_file}")
            return {}
        
        print(f"正在分析日志文件: {self.log_file}")
        print(f"Position阈值: ±{self.position_threshold}")
        print(f"忽略前100行的异常检测")
        print("=" * 80)
        
        total_lines = 0
        checked_lines = 0
        skip_lines = 100  # 忽略前100行
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    total_lines += 1
                    
                    # 跳过前100行
                    if line_num <= skip_lines:
                        continue
                    
                    if '[SUBSCRIBER]' in line:
                        checked_lines += 1
                        self.check_line(line, line_num)
                    
                    # 每处理100万行显示一次进度
                    if total_lines % 1000000 == 0:
                        print(f"已处理 {total_lines:,} 行，发现 {len(self.anomalies)} 个异常...")
        
        except Exception as e:
            print(f"读取文件时出错: {e}")
            return {}
        
        # 统计结果
        anomaly_types = defaultdict(int)
        for anomaly in self.anomalies:
            anomaly_types[anomaly['type']] += 1
        
        return {
            'total_lines': total_lines,
            'checked_lines': checked_lines,
            'total_anomalies': len(self.anomalies),
            'anomaly_types': dict(anomaly_types),
            'anomalies': self.anomalies
        }
    
    def print_report(self, stats: Dict):
        """打印检查报告"""
        print("\n" + "=" * 80)
        print("检查结果统计")
        print("=" * 80)
        print(f"总行数: {stats['total_lines']:,}")
        print(f"检查行数: {stats['checked_lines']:,}")
        print(f"异常总数: {stats['total_anomalies']}")
        print("\n异常类型统计:")
        for anomaly_type, count in stats['anomaly_types'].items():
            print(f"  {anomaly_type}: {count}")
        
        if self.anomalies:
            print("\n" + "=" * 80)
            print("异常详情（显示前50个）:")
            print("=" * 80)
            for i, anomaly in enumerate(self.anomalies[:50], 1):
                print(f"\n[{i}] 第 {anomaly['line']} 行")
                print(f"    类型: {anomaly['type']}")
                print(f"    值: {anomaly['value']:.4f}")
                print(f"    阈值: {anomaly['threshold']}")
                print(f"    信息: {anomaly['message']}")
                print(f"    原始日志: {anomaly.get('raw_line', 'N/A')}")
            
            if len(self.anomalies) > 50:
                print(f"\n... 还有 {len(self.anomalies) - 50} 个异常未显示")
            
            # 保存所有异常到文件
            output_file = self.log_file.parent / f"{self.log_file.stem}_anomalies.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                # 写入报告头部
                f.write("=" * 80 + "\n")
                f.write("异常检查报告\n")
                f.write("=" * 80 + "\n\n")
                
                # 写入整体统计信息（放在最前面，更突出）
                f.write("=" * 80 + "\n")
                f.write("【整体统计】\n")
                f.write("=" * 80 + "\n")
                f.write(f"日志文件: {self.log_file}\n")
                f.write(f"Position阈值: ±{self.position_threshold}\n")
                f.write(f"忽略前100行的异常检测\n")
                f.write("-" * 80 + "\n")
                f.write(f"总行数:           {stats['total_lines']:>15,}\n")
                f.write(f"检查行数:         {stats['checked_lines']:>15,}\n")
                f.write(f"总异常数:         {stats['total_anomalies']:>15,}\n")
                if stats['checked_lines'] > 0:
                    anomaly_rate = stats['total_anomalies'] / stats['checked_lines'] * 100
                    f.write(f"异常率:           {anomaly_rate:>15.2f}%\n")
                else:
                    f.write(f"异常率:           {0.00:>15.2f}%\n")
                
                # 在整体统计中包含异常类型汇总
                if stats['total_anomalies'] > 0:
                    f.write("-" * 80 + "\n")
                    f.write("异常类型分布:\n")
                    sorted_types = sorted(stats['anomaly_types'].items(), key=lambda x: x[1], reverse=True)
                    for anomaly_type, count in sorted_types:
                        percentage = count / stats['total_anomalies'] * 100
                        f.write(f"  {anomaly_type:30s}: {count:>8,} ({percentage:>5.2f}%)\n")
                
                f.write("=" * 80 + "\n\n")
                
                # 写入异常类型详细说明（整体统计中已有汇总，这里提供详细说明）
                if stats['total_anomalies'] > 0:
                    f.write("【异常类型说明】\n")
                    f.write("-" * 80 + "\n")
                    type_descriptions = {
                        'position_x_exceeded': 'Position X轴超过阈值',
                        'position_y_exceeded': 'Position Y轴超过阈值',
                        'position_z_abnormal': 'Position Z轴异常（通常应接近0）',
                        'linear_velocity_exceeded': '线速度异常大（>5 m/s）',
                        'angular_velocity_exceeded': '角速度异常大（>2 rad/s）',
                        'acceleration_exceeded': '加速度异常大（>20 m/s²）',
                        'callback_frequency_too_low': '回调频率过低（<限制的10%）',
                        'update_frequency_exceeded': '更新频率超过限制'
                    }
                    # 按数量排序
                    sorted_types = sorted(stats['anomaly_types'].items(), key=lambda x: x[1], reverse=True)
                    for anomaly_type, count in sorted_types:
                        desc = type_descriptions.get(anomaly_type, '未知类型')
                        f.write(f"  {anomaly_type:30s}: {desc}\n")
                    f.write("\n")
                
                # 写入异常详情
                f.write("=" * 80 + "\n")
                f.write("【异常详情】\n")
                f.write("=" * 80 + "\n\n")
                
                for i, anomaly in enumerate(self.anomalies, 1):
                    f.write(f"[{i}] 第 {anomaly['line']} 行\n")
                    f.write(f"    类型: {anomaly['type']}\n")
                    f.write(f"    值: {anomaly['value']:.4f}\n")
                    f.write(f"    阈值: {anomaly['threshold']}\n")
                    f.write(f"    信息: {anomaly['message']}\n")
                    f.write(f"    原始日志: {anomaly.get('raw_line', 'N/A')}\n\n")
            
            print(f"\n所有异常已保存到: {output_file}")
        else:
            print("\n✓ 未发现异常值")
            # 即使没有异常，也保存统计报告
            output_file = self.log_file.parent / f"{self.log_file.stem}_anomalies.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("异常检查报告\n")
                f.write("=" * 80 + "\n\n")
                
                # 写入整体统计信息（格式与有异常时一致）
                f.write("=" * 80 + "\n")
                f.write("【整体统计】\n")
                f.write("=" * 80 + "\n")
                f.write(f"日志文件: {self.log_file}\n")
                f.write(f"Position阈值: ±{self.position_threshold}\n")
                f.write(f"忽略前100行的异常检测\n")
                f.write("-" * 80 + "\n")
                f.write(f"总行数:           {stats['total_lines']:>15,}\n")
                f.write(f"检查行数:         {stats['checked_lines']:>15,}\n")
                f.write(f"总异常数:         {0:>15,}\n")
                f.write(f"异常率:           {0.00:>15.2f}%\n")
                f.write("=" * 80 + "\n\n")
                f.write("✓ 未发现异常值\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='检查日志文件中的异常值')
    parser.add_argument('log_file', type=str, help='日志文件路径')
    parser.add_argument('--position-threshold', type=float, default=160.0,
                       help='position的x、y轴阈值（默认160.0）')
    
    args = parser.parse_args()
    
    checker = LogAnomalyChecker(args.log_file, args.position_threshold)
    stats = checker.analyze()
    
    if stats:
        checker.print_report(stats)


if __name__ == '__main__':
    main()
