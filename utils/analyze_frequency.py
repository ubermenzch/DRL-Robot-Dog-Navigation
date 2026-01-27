#!/usr/bin/env python3
"""
分析日志中的callback频率和数据更新频率
"""
import re
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_timestamp(line: str) -> float:
    """从日志行中解析时间戳，返回秒数（浮点数）"""
    match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]', line)
    if match:
        ts_str = match.group(1)
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
        return dt.timestamp()
    return None


def analyze_callback_frequency(log_path: str, start_line: int, end_line: int):
    """分析指定行范围的callback频率和数据更新频率"""
    
    # 存储每个topic/env_id的callback时间戳
    callback_times: Dict[str, List[float]] = defaultdict(list)
    # 存储数据更新的时间戳
    update_times: Dict[str, List[float]] = defaultdict(list)
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # 分析指定行范围
        for i in range(start_line - 1, min(end_line, len(lines))):
            line = lines[i]
            timestamp = parse_timestamp(line)
            if timestamp is None:
                continue
            
            # 解析SUBSCRIBER callback
            sub_match = re.search(r'env_id=(\d+)\s+\[SUBSCRIBER\]\s+([/\w]+)\s+received', line)
            if sub_match:
                env_id = sub_match.group(1)
                topic = sub_match.group(2)
                key = f"env_{env_id}_{topic}"
                callback_times[key].append(timestamp)
            
            # 解析数据更新
            update_match = re.search(r'env_id=(\d+)\s+(\w+)\s+updated', line)
            if update_match:
                env_id = update_match.group(1)
                data_type = update_match.group(2)
                key = f"env_{env_id}_{data_type}"
                update_times[key].append(timestamp)
    
    # 计算真实的callback频率
    print("=" * 80)
    print("真实的Callback频率（基于时间戳计算）")
    print("=" * 80)
    
    callback_freqs = {}
    for key, times in callback_times.items():
        if len(times) < 2:
            continue
        
        # 计算时间间隔
        intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
        avg_interval = sum(intervals) / len(intervals) if intervals else 0
        real_freq = 1.0 / avg_interval if avg_interval > 0 else 0
        
        # 从日志中提取报告的频率
        reported_freq_match = re.search(r'freq=([\d.]+)Hz', lines[start_line - 1 + callback_times[key].index(times[-1])] if start_line - 1 + callback_times[key].index(times[-1]) < len(lines) else "")
        reported_freq = float(reported_freq_match.group(1)) if reported_freq_match else None
        
        callback_freqs[key] = {
            'real_freq': real_freq,
            'reported_freq': reported_freq,
            'count': len(times),
            'time_span': times[-1] - times[0] if len(times) > 1 else 0
        }
        
        print(f"\n{key}:")
        print(f"  Callback次数: {len(times)}")
        print(f"  时间跨度: {callback_freqs[key]['time_span']:.3f}秒")
        print(f"  真实频率: {real_freq:.2f} Hz")
        if reported_freq:
            print(f"  日志中报告频率: {reported_freq:.2f} Hz")
            print(f"  频率差异: {abs(real_freq - reported_freq):.2f} Hz")
    
    # 计算实际的数据更新频率
    print("\n" + "=" * 80)
    print("实际的数据更新频率")
    print("=" * 80)
    
    update_freqs = {}
    for key, times in update_times.items():
        if len(times) < 2:
            continue
        
        intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
        avg_interval = sum(intervals) / len(intervals) if intervals else 0
        update_freq = 1.0 / avg_interval if avg_interval > 0 else 0
        
        update_freqs[key] = {
            'freq': update_freq,
            'count': len(times),
            'time_span': times[-1] - times[0] if len(times) > 1 else 0
        }
        
        print(f"\n{key}:")
        print(f"  更新次数: {len(times)}")
        print(f"  时间跨度: {update_freqs[key]['time_span']:.3f}秒")
        print(f"  更新频率: {update_freq:.4f} Hz")
    
    # 如果没有找到更新记录，提示用户
    if not update_times:
        print("\n未找到数据更新记录（'updated'关键字）")
        print("可能数据更新记录不在指定行范围内，或者使用了不同的日志格式")
    
    return callback_freqs, update_freqs


if __name__ == "__main__":
    import sys
    
    log_path = "/root/DRL-Robot-Dog-Navigation/log/multi_env_training/train_20260126_183342/env_log_20260126_183342.log"
    start_line = 2128
    end_line = 2140
    
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    if len(sys.argv) > 2:
        start_line = int(sys.argv[2])
    if len(sys.argv) > 3:
        end_line = int(sys.argv[3])
    
    print(f"分析日志文件: {log_path}")
    print(f"行范围: {start_line} - {end_line}")
    
    analyze_callback_frequency(log_path, start_line, end_line)
