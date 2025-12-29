#!/usr/bin/env python3
"""
将PyTorch SAC模型转换为ONNX格式
支持导出actor模型用于推理
"""
import argparse
import yaml
import torch
import torch.onnx
import numpy as np
from pathlib import Path
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'drl_navigation_ros2'))

from SAC.SAC import SAC
from SAC.SAC_actor import DiagGaussianActor


class ActorWrapper(torch.nn.Module):
    """Actor模型包装器，用于ONNX导出
    
    将SAC的actor模型包装为直接输出确定性动作（mean）的模型
    """
    def __init__(self, actor):
        super().__init__()
        self.actor = actor
    
    def forward(self, obs):
        """
        Args:
            obs: 输入状态，shape为 (batch_size, state_dim) 或 (state_dim,)
        
        Returns:
            action: 确定性动作，shape为 (batch_size, action_dim) 或 (action_dim,)
        """
        # 确保输入是2D tensor
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # 获取分布
        dist = self.actor(obs)
        
        # 返回确定性动作（mean）
        action = dist.mean
        
        # 如果输入是1D，输出也应该是1D
        if obs.shape[0] == 1 and obs.dim() == 2:
            action = action.squeeze(0)
        
        return action


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_dummy_input(state_dim, batch_size=1):
    """创建虚拟输入用于ONNX导出"""
    return torch.randn(batch_size, state_dim, dtype=torch.float32)


def convert_to_onnx(
    model_dir,
    config_path=None,
    output_path=None,
    opset_version=11,
    device='cpu',
    batch_size=1,
    dynamic_axes=True
):
    """
    将PyTorch模型转换为ONNX格式
    
    Args:
        model_dir: 模型目录路径（包含SAC_actor.pth和config_used.yaml）
        config_path: 配置文件路径（如果为None，则使用model_dir中的config_used.yaml）
        output_path: 输出ONNX文件路径（如果为None，则在model_dir中创建）
        opset_version: ONNX opset版本
        device: 设备（'cpu'或'cuda'）
        batch_size: 批次大小（用于静态导出，如果dynamic_axes=True则会被忽略）
        dynamic_axes: 是否使用动态轴（支持可变batch size）
    """
    model_dir = Path(model_dir)
    
    # 检查模型文件是否存在
    actor_path = model_dir / "SAC_actor.pth"
    if not actor_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {actor_path}")
    
    # 加载配置文件
    if config_path is None:
        config_path = model_dir / "config_used.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    config = load_config(config_path)
    
    # 从配置中读取参数
    base_state_dim = config.get('base_state_dim', config.get('state_dim', 25))
    state_history_steps = config.get('state_history_steps', 0)
    state_dim = base_state_dim * (1 + state_history_steps) if state_history_steps > 0 else base_state_dim
    action_dim = config.get('action_dim', 2)
    hidden_dim = config.get('hidden_dim', 1024)
    hidden_depth = config.get('hidden_depth', 3)
    max_action = config.get('max_action', 1)
    
    print(f"模型参数:")
    print(f"  - state_dim: {state_dim}")
    print(f"  - base_state_dim: {base_state_dim}")
    print(f"  - state_history_steps: {state_history_steps}")
    print(f"  - action_dim: {action_dim}")
    print(f"  - hidden_dim: {hidden_dim}")
    print(f"  - hidden_depth: {hidden_depth}")
    print(f"  - max_action: {max_action}")
    
    # 设置设备
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"使用设备: {device}")
    
    # 创建SAC模型（只使用actor）
    sac_model = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        hidden_dim=hidden_dim,
        hidden_depth=hidden_depth,
        base_state_dim=base_state_dim,
        actor_only=True,  # 只创建actor模型
    )
    
    # 加载模型权重
    print(f"加载模型权重: {actor_path}")
    sac_model.load(filename="SAC", directory=str(model_dir))
    
    # 设置模型为评估模式
    sac_model.actor.eval()
    
    # 创建包装器
    wrapped_actor = ActorWrapper(sac_model.actor)
    wrapped_actor.eval()
    
    # 创建虚拟输入
    dummy_input = create_dummy_input(state_dim, batch_size=batch_size)
    dummy_input = dummy_input.to(device)
    
    # 测试模型输出
    print("测试模型输出...")
    with torch.no_grad():
        output = wrapped_actor(dummy_input)
        print(f"模型输出shape: {output.shape}")
        print(f"模型输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # 设置输出路径
    if output_path is None:
        output_path = model_dir / "SAC_actor.onnx"
    else:
        output_path = Path(output_path)
    
    # 准备动态轴配置
    if dynamic_axes:
        dynamic_axes_config = {
            'input': {0: 'batch_size'},  # 第一个维度是batch size
            'output': {0: 'batch_size'}
        }
    else:
        dynamic_axes_config = None
    
    # 导出ONNX模型
    print(f"正在导出ONNX模型到: {output_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapped_actor,
            dummy_input,
            str(output_path),
            input_names=['state'],
            output_names=['action'],
            dynamic_axes=dynamic_axes_config,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False
        )
    
    print(f"✓ ONNX模型导出成功: {output_path}")
    
    # 验证ONNX模型
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX模型验证通过")
        
        # 打印模型信息
        print(f"\nONNX模型信息:")
        print(f"  - 输入: {[input.name for input in onnx_model.graph.input]}")
        print(f"  - 输出: {[output.name for output in onnx_model.graph.output]}")
        print(f"  - Opset版本: {onnx_model.opset_import[0].version}")
    except ImportError:
        print("警告: 未安装onnx包，跳过模型验证")
    except Exception as e:
        print(f"警告: ONNX模型验证失败: {e}")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description='将PyTorch SAC模型转换为ONNX格式')
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='模型目录路径（包含SAC_actor.pth和config_used.yaml）'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径（可选，默认使用model_dir中的config_used.yaml）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出ONNX文件路径（可选，默认在model_dir中创建SAC_actor.onnx）'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=11,
        help='ONNX opset版本（默认: 11）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'auto'],
        help='设备选择（默认: cpu）'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='批次大小（用于静态导出，如果使用动态轴则会被忽略，默认: 1）'
    )
    parser.add_argument(
        '--no_dynamic_axes',
        action='store_true',
        help='禁用动态轴（使用固定batch size）'
    )
    
    args = parser.parse_args()
    
    try:
        output_path = convert_to_onnx(
            model_dir=args.model_dir,
            config_path=args.config,
            output_path=args.output,
            opset_version=args.opset,
            device=args.device,
            batch_size=args.batch_size,
            dynamic_axes=not args.no_dynamic_axes
        )
        print(f"\n转换完成！ONNX模型保存在: {output_path}")
    except Exception as e:
        print(f"转换失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

