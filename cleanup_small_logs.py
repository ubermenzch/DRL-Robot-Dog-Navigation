#!/usr/bin/env python3
"""
清理小于指定大小的日志文件夹脚本

用法:
    python3 cleanup_small_logs.py [选项]

选项:
    --log-dir: 日志目录路径（默认: log/multi_env_training）
    --size-threshold: 大小阈值，单位MB（默认: 1）
    --dry-run: 仅显示将要删除的文件夹，不实际删除
    --interactive: 交互模式，删除前询问确认
"""

import os
import sys
import argparse
from pathlib import Path


def get_dir_size(path):
    """计算目录的总大小（字节）"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
    except (OSError, PermissionError):
        pass
    return total_size


def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def cleanup_small_logs(log_dir, size_threshold_mb, dry_run=False, interactive=False):
    """
    清理小于指定大小的日志文件夹
    
    Args:
        log_dir: 日志目录路径
        size_threshold_mb: 大小阈值（MB）
        dry_run: 是否仅预览，不实际删除
        interactive: 是否交互式确认
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"错误: 日志目录不存在: {log_dir}")
        return
    
    if not log_path.is_dir():
        print(f"错误: 不是目录: {log_dir}")
        return
    
    size_threshold_bytes = size_threshold_mb * 1024 * 1024
    small_dirs = []
    
    print(f"扫描日志目录: {log_dir}")
    print(f"大小阈值: {size_threshold_mb} MB ({format_size(size_threshold_bytes)})")
    print(f"模式: {'预览模式（不会实际删除）' if dry_run else '删除模式'}")
    print("-" * 60)
    
    # 扫描所有子目录
    for item in log_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            dir_size = get_dir_size(item)
            if dir_size < size_threshold_bytes:
                small_dirs.append((item, dir_size))
    
    if not small_dirs:
        print("没有找到小于阈值的文件夹。")
        return
    
    # 按大小排序
    small_dirs.sort(key=lambda x: x[1])
    
    print(f"\n找到 {len(small_dirs)} 个小于阈值的文件夹:\n")
    total_size = 0
    for dir_path, dir_size in small_dirs:
        total_size += dir_size
        print(f"  {dir_path.name:50s} {format_size(dir_size):>12s}")
    
    print("-" * 60)
    print(f"总计: {len(small_dirs)} 个文件夹, 总大小: {format_size(total_size)}")
    
    if dry_run:
        print("\n[预览模式] 以上文件夹将被删除。使用 --no-dry-run 来实际执行删除。")
        return
    
    # 确认删除
    if interactive:
        response = input(f"\n确定要删除这 {len(small_dirs)} 个文件夹吗？(yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("取消删除。")
            return
    
    # 执行删除
    print("\n开始删除...")
    deleted_count = 0
    failed_count = 0
    
    for dir_path, dir_size in small_dirs:
        try:
            import shutil
            shutil.rmtree(dir_path)
            deleted_count += 1
            print(f"  ✓ 已删除: {dir_path.name} ({format_size(dir_size)})")
        except Exception as e:
            failed_count += 1
            print(f"  ✗ 删除失败: {dir_path.name} - {e}")
    
    print("-" * 60)
    print(f"删除完成: 成功 {deleted_count} 个, 失败 {failed_count} 个")
    print(f"释放空间: {format_size(total_size)}")


def main():
    parser = argparse.ArgumentParser(
        description='清理小于指定大小的日志文件夹',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='log/multi_env_training',
        help='日志目录路径（默认: log/multi_env_training）'
    )
    
    parser.add_argument(
        '--size-threshold',
        type=float,
        default=1.0,
        help='大小阈值，单位MB（默认: 1.0）'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='预览模式，仅显示将要删除的文件夹，不实际删除'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='交互模式，删除前询问确认'
    )
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    log_dir = os.path.abspath(args.log_dir)
    
    cleanup_small_logs(
        log_dir=log_dir,
        size_threshold_mb=args.size_threshold,
        dry_run=args.dry_run,
        interactive=args.interactive
    )


if __name__ == '__main__':
    main()

