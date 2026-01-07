#!/usr/bin/env python3
"""
将PyTorch SAC模型转换为ONNX格式
支持导出actor模型用于推理

输入输出格式说明：
==================

1. PyTorch模型（DiagGaussianActor.forward()）：
   - 输入: obs, shape为 (state_dim,)
   - 输出: dist (SquashedNormal分布对象)
   - 获取动作: dist.mean 或 dist.sample()

2. ONNX模型（导出的模型）：
   - 输入: state, shape为 (state_dim,) - 单个状态向量
   - 输出: action, shape为 (action_dim,) - 单个动作向量
   - 注意：ONNX模型直接输出确定性动作（相当于dist.mean），已经过clamp处理
   - 注意：仅支持单个向量输入输出，不支持批量处理

3. 使用方式：
   - 输入: 1D numpy数组或tensor，shape为 (state_dim,)
   - 输出: 1D numpy数组或tensor，shape为 (action_dim,)
   - 示例: action = onnx_model(state)  # state是1D，action也是1D

4. 与SAC.act()方法的对应关系：
   - SAC.act()方法: 输入1D数组，输出1D数组
   - ONNX模型: 输入1D数组，输出1D数组
   - 两者输入输出格式完全一致，可以直接替换使用
"""
import argparse
import yaml
import torch
import torch.onnx
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime
import logging

# 添加项目路径
# 获取脚本所在目录的父目录（项目根目录）
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # utils的父目录是项目根目录
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src', 'drl_navigation_ros2'))

# 如果logger未初始化，创建一个默认的logger（用于在setup_logging之前可能调用的地方）
logger = logging.getLogger(__name__)
if not logger.handlers:
    # 如果没有handler，创建一个临时的控制台handler
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

# 设置日志
def setup_logging(model_dir_name=None):
    """设置日志系统，同时输出到控制台和文件"""
    global logger
    
    log_dir = Path("/home/zc/DRL-Robot-Navigation-ROS2/log/convert_to_onnx_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成日志文件名（使用时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_dir_name:
        # 从模型目录名中提取有用信息
        model_name = Path(model_dir_name).name
        log_filename = f"convert_{model_name}_{timestamp}.log"
    else:
        log_filename = f"convert_{timestamp}.log"
    
    log_file = log_dir / log_filename
    
    # 清除现有的handlers
    logger.handlers.clear()
    
    # 创建新的handlers
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler(sys.stdout)
    
    # 设置格式
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 设置级别
    logger.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.INFO)  # 控制台只显示INFO及以上级别
    
    # 添加handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"日志文件保存在: {log_file}")
    logger.info(f"开始转换ONNX模型...")
    
    return logger

# 全局logger（在main函数中重新初始化）

from SAC.SAC import SAC
from SAC.SAC_actor import DiagGaussianActor


class ActorWrapper(torch.nn.Module):
    """Actor模型包装器，用于ONNX导出
    
    将SAC的actor模型包装为直接输出确定性动作（mean）的模型
    仅支持1D输入（单个向量），输出也是1D向量
    """
    def __init__(self, actor, max_action=1.0):
        super().__init__()
        self.actor = actor
        self.max_action = max_action
    
    def forward(self, obs):
        """
        Args:
            obs: 输入状态，shape为 (state_dim,)
        
        Returns:
            action: 确定性动作，shape为 (action_dim,)
        """
        # 确保输入是2D tensor（内部处理需要，添加batch维度）
        obs = obs.unsqueeze(0)  # (state_dim,) -> (1, state_dim)
        
        # 获取分布
        dist = self.actor(obs)
        
        # 返回确定性动作（mean），并clamp到动作范围
        action = dist.mean  # (1, action_dim)
        action = action.clamp(-self.max_action, self.max_action)
        
        # 移除batch维度，返回1D向量
        action = action.squeeze(0)  # (1, action_dim) -> (action_dim,)
        
        return action


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_dummy_input(state_dim):
    """创建虚拟输入用于ONNX导出（1D向量）"""
    return torch.randn(state_dim, dtype=torch.float32)


def create_realistic_test_input(config, base_state_dim, state_history_steps):
    """
    根据训练参数配置，构造符合实际场景的随机输入用于测试ONNX模型
    
    Args:
        config: 配置字典
        base_state_dim: 基础状态维度
        state_history_steps: 历史状态步数
    
    Returns:
        state: 构造的状态向量，shape为 (state_dim,)
    """
    import random
    
    # 从配置中读取参数
    max_target_dist = config.get('max_target_dist', 3.0)
    target_reached_delta = config.get('target_reached_delta', 0.4)
    collision_delta = config.get('collision_delta', 0.3)
    scan_range = config.get('scan_range', 10.0)
    
    # 1. target_distance: 随机在 (target_reached_delta, max_target_dist] 范围
    target_distance = random.uniform(target_reached_delta, max_target_dist)
    
    # 2. target_sin和target_cos: 随机生成（需要满足 sin^2 + cos^2 = 1）
    # 随机生成一个角度，然后计算sin和cos
    angle = random.uniform(0, 2 * np.pi)
    target_sin = np.sin(angle)
    target_cos = np.cos(angle)
    
    # 3. obs_min: 随机生成max_bins个分区，每个分区的随机范围为(collision_delta, scan_range]
    max_bins = base_state_dim - 5  # 激光扫描分箱数
    obs_min = []
    for _ in range(max_bins):
        # 每个分区的最小距离在 (collision_delta, scan_range] 范围
        min_dist = random.uniform(collision_delta, scan_range)
        obs_min.append(min_dist)
    
    # 4. 上一线速度、角速度: 随机在 [-1, 1] 范围
    prev_linear_velocity = random.uniform(-1.0, 1.0)
    prev_angular_velocity = random.uniform(-1.0, 1.0)
    
    # 5. 构造基础状态向量 (base_state_dim个元素)
    base_state = obs_min + [target_distance, target_cos, target_sin, prev_linear_velocity, prev_angular_velocity]
    
    # 6. 如果使用历史状态，将state复制state_history_steps+1份
    if state_history_steps > 0:
        # 复制当前state state_history_steps+1次（包括当前）
        state = base_state * (state_history_steps + 1)
    else:
        state = base_state
    
    return np.array(state, dtype=np.float32)


def convert_to_onnx(
    model_dir,
    config_path=None,
    output_path=None,
    opset_version=11,
    device='cpu'
):
    """
    将PyTorch模型转换为ONNX格式（仅支持单个向量输入输出）
    
    Args:
        model_dir: 模型目录路径（包含SAC_actor.pth和配置文件）
        config_path: 配置文件路径（如果为None，则自动查找目录下第一个名称包含"config"的yaml文件）
        output_path: 输出ONNX文件路径（如果为None，则在model_dir中创建）
        opset_version: ONNX opset版本
        device: 设备（'cpu'或'cuda'）
    """
    global logger
    model_dir = Path(model_dir)
    logger.info(f"模型目录: {model_dir}")
    
    # 检查模型文件是否存在
    actor_path = model_dir / "SAC_actor.pth"
    logger.debug(f"检查模型文件: {actor_path}")
    if not actor_path.exists():
        logger.error(f"模型文件不存在: {actor_path}")
        raise FileNotFoundError(f"模型文件不存在: {actor_path}")
    logger.info(f"找到模型文件: {actor_path}")
    
    # 加载配置文件
    if config_path is None:
        # 自动查找目录下第一个名称包含"config"的yaml文件
        logger.debug(f"自动查找配置文件，搜索目录: {model_dir}")
        config_files = list(model_dir.glob("*config*.yaml")) + list(model_dir.glob("*config*.yml"))
        logger.debug(f"找到的配置文件: {[f.name for f in config_files]}")
        if not config_files:
            logger.error(f"在目录 {model_dir} 中未找到包含'config'的yaml配置文件")
            raise FileNotFoundError(f"在目录 {model_dir} 中未找到包含'config'的yaml配置文件")
        config_path = config_files[0]  # 使用第一个找到的配置文件
        logger.info(f"自动找到配置文件: {config_path.name}")
        print(f"自动找到配置文件: {config_path.name}")
    else:
        config_path = Path(config_path)
        logger.info(f"使用指定的配置文件: {config_path}")
    
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    logger.debug(f"加载配置文件: {config_path}")
    config = load_config(config_path)
    logger.debug(f"配置文件加载成功，包含 {len(config)} 个配置项")
    
    # 从配置中读取参数
    base_state_dim = config.get('base_state_dim', config.get('state_dim', 25))
    state_history_steps = config.get('state_history_steps', 0)
    state_dim = base_state_dim * (1 + state_history_steps) if state_history_steps > 0 else base_state_dim
    action_dim = config.get('action_dim', 2)
    
    # 处理hidden_layers：优先使用hidden_layers，如果没有则从hidden_dim和hidden_depth构造
    # 网络结构：输入层(state_dim) -> 隐含层[hidden_dim] * hidden_depth -> 输出层(action_dim * 2)
    if 'hidden_layers' in config:
        hidden_layers = config.get('hidden_layers')
        if isinstance(hidden_layers, str):
            # 如果是字符串，尝试解析为列表
            import json
            try:
                hidden_layers = json.loads(hidden_layers)
            except:
                # 解析失败，尝试从hidden_dim和hidden_depth构造
                if 'hidden_dim' in config and 'hidden_depth' in config:
                    hidden_dim = config.get('hidden_dim', 1024)
                    hidden_depth = config.get('hidden_depth', 2)
                    hidden_layers = [hidden_dim] * hidden_depth
                else:
                    hidden_layers = [1024, 512]
    elif 'hidden_dim' in config and 'hidden_depth' in config:
        # 从hidden_dim和hidden_depth构造hidden_layers
        # hidden_layers表示所有隐含层的宽度，每个隐含层都是hidden_dim宽度，共hidden_depth层
        hidden_dim = config.get('hidden_dim', 1024)
        hidden_depth = config.get('hidden_depth', 2)
        hidden_layers = [hidden_dim] * hidden_depth
        print(f"  从hidden_dim={hidden_dim}和hidden_depth={hidden_depth}构造hidden_layers={hidden_layers}")
    else:
        hidden_layers = [1024, 512]  # 默认值
        print(f"  使用默认hidden_layers={hidden_layers}")
    
    max_action = config.get('max_action', 1)
    
    logger.info("="*60)
    logger.info("模型参数配置:")
    logger.info(f"  - state_dim: {state_dim} (输入层宽度，手动计算)")
    logger.info(f"  - base_state_dim: {base_state_dim}")
    logger.info(f"  - state_history_steps: {state_history_steps}")
    logger.info(f"  - action_dim: {action_dim} (输出层宽度=2)")
    logger.info(f"  - hidden_layers: {hidden_layers} (隐含层结构)")
    if 'hidden_dim' in config and 'hidden_depth' in config:
        logger.info(f"  - hidden_dim: {config.get('hidden_dim')} (隐含层宽度)")
        logger.info(f"  - hidden_depth: {config.get('hidden_depth')} (隐含层深度)")
    logger.info(f"  - max_action: {max_action}")
    logger.info(f"  网络结构: 输入({state_dim}) -> 隐含层{hidden_layers} -> 输出({action_dim * 2})")
    logger.info("="*60)
    
    print(f"模型参数:")
    print(f"  - state_dim: {state_dim} (输入层宽度，手动计算)")
    print(f"  - base_state_dim: {base_state_dim}")
    print(f"  - state_history_steps: {state_history_steps}")
    print(f"  - action_dim: {action_dim} (输出层宽度=2)")
    print(f"  - hidden_layers: {hidden_layers} (隐含层结构)")
    if 'hidden_dim' in config and 'hidden_depth' in config:
        print(f"  - hidden_dim: {config.get('hidden_dim')} (隐含层宽度)")
        print(f"  - hidden_depth: {config.get('hidden_depth')} (隐含层深度)")
    print(f"  - max_action: {max_action}")
    print(f"  网络结构: 输入({state_dim}) -> 隐含层{hidden_layers} -> 输出({action_dim * 2})")
    
    # 设置设备
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logger.info(f"使用设备: {device}")
    print(f"使用设备: {device}")
    
    # 创建SAC模型（只使用actor）
    logger.info("创建SAC模型（actor_only模式）...")
    logger.debug(f"创建参数: state_dim={state_dim}, action_dim={action_dim}, max_action={max_action}, "
                 f"hidden_layers={hidden_layers}, base_state_dim={base_state_dim}")
    sac_model = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        hidden_layers=hidden_layers,
        base_state_dim=base_state_dim,
        actor_only=True,  # 只创建actor模型
    )
    logger.info("SAC模型创建完成")
    
    # 检查模型结构
    logger.debug("检查模型结构...")
    total_params = sum(p.numel() for p in sac_model.actor.parameters())
    trainable_params = sum(p.numel() for p in sac_model.actor.parameters() if p.requires_grad)
    logger.info(f"Actor模型参数统计: 总参数={total_params}, 可训练参数={trainable_params}")
    
    # 加载模型权重
    logger.info(f"加载模型权重: {actor_path}")
    print(f"加载模型权重: {actor_path}")
    try:
        sac_model.load(filename="SAC", directory=str(model_dir))
        logger.info("✓ 模型权重加载成功")
        print("✓ 模型权重加载成功")
        
        # 验证加载后的权重
        logger.debug("验证加载后的模型权重...")
        loaded_params = sum(p.numel() for p in sac_model.actor.parameters())
        logger.info(f"加载后模型参数数量: {loaded_params}")
        
        # 检查权重是否全0
        zero_params = sum((p.abs() < 1e-6).sum().item() for p in sac_model.actor.parameters())
        zero_ratio = zero_params / total_params * 100 if total_params > 0 else 0
        logger.info(f"权重统计: 零参数={zero_params}, 零参数比例={zero_ratio:.2f}%")
        if zero_ratio > 50:
            logger.warning(f"警告: 模型权重中零参数比例过高 ({zero_ratio:.2f}%)，可能权重未正确加载")
    except Exception as e:
        logger.error(f"模型权重加载失败: {e}", exc_info=True)
        raise RuntimeError(f"模型权重加载失败: {e}")
    
    # 设置模型为评估模式
    logger.debug("设置模型为评估模式...")
    sac_model.actor.eval()
    
    # 创建包装器（传入max_action用于clamp）
    logger.debug(f"创建ActorWrapper，max_action={max_action}")
    wrapped_actor = ActorWrapper(sac_model.actor, max_action=max_action)
    wrapped_actor.eval()
    logger.info("ActorWrapper创建完成")
    
    # 创建虚拟输入（1D向量，符合实际使用场景）
    logger.debug(f"创建虚拟输入，state_dim={state_dim}")
    dummy_input = create_dummy_input(state_dim).to(device)
    logger.debug(f"虚拟输入shape: {dummy_input.shape}, 范围: [{dummy_input.min().item():.4f}, {dummy_input.max().item():.4f}]")
    
    # 测试包装器输出（使用1D输入，模拟实际使用场景）
    logger.info("="*60)
    logger.info("测试包装器模型（1D输入 -> 1D输出）...")
    print("\n测试包装器模型（1D输入 -> 1D输出）...")
    with torch.no_grad():
        logger.debug("执行包装器前向传播...")
        output = wrapped_actor(dummy_input)
        logger.info(f"输入shape: {dummy_input.shape} (应该是1D: ({state_dim},))")
        logger.info(f"输出shape: {output.shape} (应该是1D: ({action_dim},))")
        logger.info(f"输出线速度与角速度: [{output.min().item():.4f}, {output.max().item():.4f}]")
        logger.info(f"输出均值: {output.mean().item():.4f}")
        logger.info(f"输出是否全0: {(output.abs() < 1e-6).all().item()}")
        
        print(f"输入shape: {dummy_input.shape} (应该是1D: ({state_dim},))")
        print(f"输出shape: {output.shape} (应该是1D: ({action_dim},))")
        print(f"输出线速度与角速度: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"输出均值: {output.mean().item():.4f}")
        print(f"输出是否全0: {(output.abs() < 1e-6).all().item()}")
        
        # 验证输出维度
        if output.dim() != 1 or output.shape[0] != action_dim:
            logger.error(f"包装器输出维度错误: 期望({action_dim},)，实际{output.shape}")
            raise RuntimeError(f"包装器输出维度错误: 期望({action_dim},)，实际{output.shape}")
        
        # 如果输出全0，检查模型参数
        if (output.abs() < 1e-6).all().item():
            logger.warning("⚠️  警告: 包装器模型输出全0，检查模型参数...")
            print("⚠️  警告: 包装器模型输出全0，检查模型参数...")
            total_params = 0
            zero_params = 0
            for param in sac_model.actor.parameters():
                total_params += param.numel()
                zero_params += (param.abs() < 1e-6).sum().item()
            zero_ratio = zero_params / total_params * 100 if total_params > 0 else 0
            logger.warning(f"模型参数统计: 总参数={total_params}, 零参数={zero_params}, 零参数比例={zero_ratio:.2f}%")
            print(f"模型参数统计: 总参数={total_params}, 零参数={zero_params}, 零参数比例={zero_ratio:.2f}%")
            raise RuntimeError("包装器模型输出全0，可能是权重未正确加载！")
        
        logger.info("✓ 包装器模型测试通过（1D输入 -> 1D输出）")
        print("✓ 包装器模型测试通过（1D输入 -> 1D输出）")
    
    # 设置输出路径
    if output_path is None:
        output_path = model_dir / "SAC_actor.onnx"
    else:
        output_path = Path(output_path)
    
    # 导出ONNX模型（使用1D输入，符合实际使用场景）
    logger.info("="*60)
    logger.info(f"正在导出ONNX模型到: {output_path}")
    logger.info(f"使用1D输入进行导出: shape={dummy_input.shape}")
    logger.info(f"ONNX opset版本: {opset_version}")
    print(f"\n正在导出ONNX模型到: {output_path}")
    print(f"使用1D输入进行导出: shape={dummy_input.shape}")
    try:
        with torch.no_grad():
            logger.debug("开始torch.onnx.export...")
            torch.onnx.export(
                wrapped_actor,
                dummy_input,  # 使用1D输入
                str(output_path),
                input_names=['state'],
                output_names=['action'],
                dynamic_axes=None,  # 固定输入输出维度，不支持动态batch
                opset_version=opset_version,
                do_constant_folding=True,
                verbose=False,
                export_params=True,  # 确保导出参数
            )
            logger.debug("torch.onnx.export完成")
    except Exception as e:
        logger.error(f"ONNX导出失败: {e}", exc_info=True)
        raise RuntimeError(f"ONNX导出失败: {e}")
    
    logger.info(f"✓ ONNX模型导出成功: {output_path}")
    print(f"✓ ONNX模型导出成功: {output_path}")
    
    # 检查文件大小
    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    logger.info(f"ONNX模型文件大小: {file_size:.2f} MB")
    
    # 验证ONNX模型
    logger.info("="*60)
    logger.info("验证ONNX模型...")
    try:
        import onnx
        logger.debug(f"加载ONNX模型文件: {output_path}")
        onnx_model = onnx.load(str(output_path))
        logger.debug("执行ONNX模型检查...")
        onnx.checker.check_model(onnx_model)
        logger.info("✓ ONNX模型验证通过")
        print("✓ ONNX模型验证通过")
        
        # 打印模型信息
        logger.info("ONNX模型详细信息:")
        input_names = [input.name for input in onnx_model.graph.input]
        output_names = [output.name for output in onnx_model.graph.output]
        opset_version = onnx_model.opset_import[0].version
        
        logger.info(f"  - 输入: {input_names}")
        logger.info(f"  - 输出: {output_names}")
        logger.info(f"  - Opset版本: {opset_version}")
        
        # 打印输入输出形状信息
        for inp in onnx_model.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else '?' for dim in inp.type.tensor_type.shape.dim]
            logger.info(f"  - 输入 '{inp.name}' 形状: {shape}")
        for out in onnx_model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else '?' for dim in out.type.tensor_type.shape.dim]
            logger.info(f"  - 输出 '{out.name}' 形状: {shape}")
        
        print(f"\nONNX模型信息:")
        print(f"  - 输入: {input_names}")
        print(f"  - 输出: {output_names}")
        print(f"  - Opset版本: {opset_version}")
    except ImportError:
        logger.warning("未安装onnx包，跳过ONNX模型验证")
        print("警告: 未安装onnx包，跳过ONNX模型验证")
    except Exception as e:
        logger.warning(f"ONNX模型验证失败: {e}", exc_info=True)
        print(f"警告: ONNX模型验证失败: {e}")
    
    # 使用ONNX Runtime测试模型输出（如果可用）
    logger.info("="*60)
    logger.info("使用ONNX Runtime测试模型（1D输入）...")
    try:
        import onnxruntime as ort
        
        print("\n使用ONNX Runtime测试模型（1D输入）...")
        logger.debug("创建ONNX Runtime InferenceSession...")
        session = ort.InferenceSession(str(output_path))
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        logger.debug(f"ONNX Runtime输入名称: {input_name}, 输出名称: {output_name}")
        
        # 准备测试输入（1D numpy格式）
        test_input = dummy_input.cpu().numpy()
        logger.debug(f"准备测试输入，shape: {test_input.shape}, dtype: {test_input.dtype}")
        logger.debug(f"测试输入范围: [{test_input.min():.4f}, {test_input.max():.4f}]")
        
        logger.debug("执行ONNX Runtime推理...")
        onnx_output = session.run([output_name], {input_name: test_input})[0]
        logger.debug(f"ONNX Runtime推理完成，输出shape: {onnx_output.shape}")
        
        logger.info(f"ONNX Runtime输入shape: {test_input.shape} (应该是1D: ({state_dim},))")
        logger.info(f"ONNX Runtime输出shape: {onnx_output.shape} (应该是1D: ({action_dim},))")
        logger.info(f"ONNX Runtime输出线速度与角速度: [{onnx_output.min():.4f}, {onnx_output.max():.4f}]")
        logger.info(f"ONNX Runtime输出均值: {onnx_output.mean():.4f}")
        logger.info(f"ONNX Runtime输出是否全0: {(np.abs(onnx_output) < 1e-6).all()}")
        
        print(f"ONNX Runtime输入shape: {test_input.shape} (应该是1D: ({state_dim},))")
        print(f"ONNX Runtime输出shape: {onnx_output.shape} (应该是1D: ({action_dim},))")
        print(f"ONNX Runtime输出线速度与角速度: [{onnx_output.min():.4f}, {onnx_output.max():.4f}]")
        print(f"ONNX Runtime输出均值: {onnx_output.mean():.4f}")
        print(f"ONNX Runtime输出是否全0: {(np.abs(onnx_output) < 1e-6).all()}")
        
        # 验证输出维度
        if onnx_output.ndim != 1 or onnx_output.shape[0] != action_dim:
            logger.warning(f"ONNX输出维度错误: 期望({action_dim},)，实际{onnx_output.shape}")
            print(f"⚠️  警告: ONNX输出维度错误: 期望({action_dim},)，实际{onnx_output.shape}")
        
        # 对比PyTorch和ONNX输出（都是1D）
        pytorch_output = output.cpu().numpy()
        max_diff = np.abs(pytorch_output - onnx_output).max()
        mean_diff = np.abs(pytorch_output - onnx_output).mean()
        logger.info(f"PyTorch vs ONNX最大差异: {max_diff:.6f}")
        logger.info(f"PyTorch vs ONNX平均差异: {mean_diff:.6f}")
        print(f"PyTorch vs ONNX最大差异: {max_diff:.6f}")
        print(f"PyTorch vs ONNX平均差异: {mean_diff:.6f}")
        
        if (np.abs(onnx_output) < 1e-6).all():
            logger.warning("⚠️  警告: ONNX模型输出全0！请检查模型权重是否正确加载。")
            print("⚠️  警告: ONNX模型输出全0！请检查模型权重是否正确加载。")
        elif max_diff > 1e-3:
            logger.warning(f"⚠️  警告: PyTorch和ONNX输出差异较大 (>{1e-3})")
            print(f"⚠️  警告: PyTorch和ONNX输出差异较大 (>{1e-3})")
        else:
            logger.info("✓ PyTorch和ONNX输出一致（1D输入 -> 1D输出）")
            print("✓ PyTorch和ONNX输出一致（1D输入 -> 1D输出）")
            
    except ImportError:
        logger.warning("未安装onnxruntime包，跳过运行时测试（建议安装以验证模型）")
        logger.info("安装onnxruntime: pip install onnxruntime")
        print("提示: 未安装onnxruntime包，跳过运行时测试（建议安装以验证模型）")
        print("安装onnxruntime: pip install onnxruntime")
    except Exception as e:
        logger.warning(f"ONNX Runtime测试失败: {e}", exc_info=True)
        print(f"警告: ONNX Runtime测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 使用符合实际场景的随机输入进行额外测试
    logger.info("="*60)
    logger.info("使用符合实际场景的随机输入进行测试...")
    print("\n" + "="*60)
    print("使用符合实际场景的随机输入进行测试...")
    print("="*60)
    
    # 从配置中读取参数用于显示
    max_target_dist = config.get('max_target_dist', 3.0)
    target_reached_delta = config.get('target_reached_delta', 0.4)
    collision_delta = config.get('collision_delta', 0.3)
    scan_range = config.get('scan_range', 10.0)
    max_bins = base_state_dim - 5
    logger.debug(f"测试参数: max_target_dist={max_target_dist}, target_reached_delta={target_reached_delta}, "
                f"collision_delta={collision_delta}, scan_range={scan_range}, max_bins={max_bins}")
    
    # 构造符合实际场景的随机输入
    logger.debug("构造符合实际场景的随机输入...")
    realistic_input = create_realistic_test_input(config, base_state_dim, state_history_steps)
    realistic_input_tensor = torch.from_numpy(realistic_input).to(device)
    logger.debug(f"构造的随机输入shape: {realistic_input.shape}, 范围: [{realistic_input.min():.4f}, {realistic_input.max():.4f}]")
    
    print(f"\n构造的随机输入参数:")
    print(f"  - target_distance: {realistic_input[base_state_dim-5]:.4f} (范围: [{target_reached_delta}, {max_target_dist}])")
    print(f"  - target_cos: {realistic_input[base_state_dim-4]:.4f}")
    print(f"  - target_sin: {realistic_input[base_state_dim-3]:.4f}")
    print(f"  - prev_linear_velocity: {realistic_input[base_state_dim-2]:.4f} (范围: [-1, 1])")
    print(f"  - prev_angular_velocity: {realistic_input[base_state_dim-1]:.4f} (范围: [-1, 1])")
    print(f"  - obs_min范围: [{realistic_input[:max_bins].min():.4f}, {realistic_input[:max_bins].max():.4f}] (范围: ({collision_delta}, {scan_range}])")
    print(f"  - state_history_steps: {state_history_steps}")
    print(f"  - 输入shape: {realistic_input.shape}")
    
    # 测试PyTorch模型（不依赖onnxruntime）
    logger.info("测试PyTorch模型（符合实际场景的随机输入）...")
    print("\n测试PyTorch模型...")
    with torch.no_grad():
        logger.debug("执行PyTorch模型推理...")
        pytorch_output_realistic = wrapped_actor(realistic_input_tensor)
        logger.info(f"PyTorch输出shape: {pytorch_output_realistic.shape}")
        logger.info(f"PyTorch输出线速度与角速度: [{pytorch_output_realistic.min().item():.4f}, {pytorch_output_realistic.max().item():.4f}]")
        logger.info(f"PyTorch输出均值: {pytorch_output_realistic.mean().item():.4f}")
        logger.info(f"PyTorch输出是否全0: {(pytorch_output_realistic.abs() < 1e-6).all().item()}")
        print(f"PyTorch输出shape: {pytorch_output_realistic.shape}")
        print(f"PyTorch输出线速度与角速度: [{pytorch_output_realistic.min().item():.4f}, {pytorch_output_realistic.max().item():.4f}]")
        print(f"PyTorch输出均值: {pytorch_output_realistic.mean().item():.4f}")
        print(f"PyTorch输出是否全0: {(pytorch_output_realistic.abs() < 1e-6).all().item()}")
        if (pytorch_output_realistic.abs() < 1e-6).all().item():
            logger.warning("⚠️  警告: PyTorch模型输出全0！")
            print("⚠️  警告: PyTorch模型输出全0！")
    
    # 进行多次PyTorch随机测试（不依赖onnxruntime）
    logger.info("进行多次PyTorch随机测试（10次）...")
    print("\n进行多次PyTorch随机测试（10次）...")
    all_pytorch_passed = True
    for i in range(10):
        logger.debug(f"PyTorch随机测试 {i+1}/10: 构造测试输入...")
        test_input = create_realistic_test_input(config, base_state_dim, state_history_steps)
        test_input_tensor = torch.from_numpy(test_input).to(device)
        
        with torch.no_grad():
            pytorch_out = wrapped_actor(test_input_tensor).cpu().numpy()
        
        is_zero = (np.abs(pytorch_out) < 1e-6).all()
        
        if is_zero:
            logger.warning(f"PyTorch随机测试 {i+1}/10: ⚠️  失败 (输出全0)")
            print(f"  测试 {i+1}/10: ⚠️  失败 (输出全0)")
            all_pytorch_passed = False
        else:
            logger.debug(f"PyTorch随机测试 {i+1}/10: ✓ 通过 (输出线速度与角速度: [{pytorch_out.min():.4f}, {pytorch_out.max():.4f}])")
            print(f"  测试 {i+1}/10: ✓ 通过 (输出线速度与角速度: [{pytorch_out.min():.4f}, {pytorch_out.max():.4f}])")
    
    if all_pytorch_passed:
        logger.info("✓ 所有PyTorch随机测试通过！")
        print("\n✓ 所有PyTorch随机测试通过！")
    else:
        logger.warning("⚠️  部分PyTorch随机测试失败，请检查模型。")
        print("\n⚠️  部分PyTorch随机测试失败，请检查模型。")
    
    # 如果安装了onnxruntime，进行ONNX对比测试
    try:
        import onnxruntime as ort
        
        logger.info("检测到onnxruntime，进行ONNX模型对比测试...")
        print("\n检测到onnxruntime，进行ONNX模型对比测试...")
        
        # 测试ONNX模型
        logger.info("测试ONNX模型（符合实际场景的随机输入）...")
        print("\n测试ONNX模型...")
        session = ort.InferenceSession(str(output_path))
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        logger.debug(f"ONNX Runtime输入名称: {input_name}, 输出名称: {output_name}")
        
        logger.debug("执行ONNX Runtime推理...")
        onnx_output_realistic = session.run([output_name], {input_name: realistic_input})[0]
        logger.info(f"ONNX输出shape: {onnx_output_realistic.shape}")
        logger.info(f"ONNX输出线速度与角速度: [{onnx_output_realistic.min():.4f}, {onnx_output_realistic.max():.4f}]")
        logger.info(f"ONNX输出均值: {onnx_output_realistic.mean():.4f}")
        logger.info(f"ONNX输出是否全0: {(np.abs(onnx_output_realistic) < 1e-6).all()}")
        print(f"ONNX输出shape: {onnx_output_realistic.shape}")
        print(f"ONNX输出线速度与角速度: [{onnx_output_realistic.min():.4f}, {onnx_output_realistic.max():.4f}]")
        print(f"ONNX输出均值: {onnx_output_realistic.mean():.4f}")
        print(f"ONNX输出是否全0: {(np.abs(onnx_output_realistic) < 1e-6).all()}")
        if (np.abs(onnx_output_realistic) < 1e-6).all():
            logger.warning("⚠️  警告: ONNX模型输出全0！")
            print("⚠️  警告: ONNX模型输出全0！")
        
        # 对比输出
        max_diff_realistic = np.abs(pytorch_output_realistic.cpu().numpy() - onnx_output_realistic).max()
        mean_diff_realistic = np.abs(pytorch_output_realistic.cpu().numpy() - onnx_output_realistic).mean()
        logger.info(f"PyTorch vs ONNX最大差异: {max_diff_realistic:.6f}")
        logger.info(f"PyTorch vs ONNX平均差异: {mean_diff_realistic:.6f}")
        print(f"\nPyTorch vs ONNX最大差异: {max_diff_realistic:.6f}")
        print(f"PyTorch vs ONNX平均差异: {mean_diff_realistic:.6f}")
        
        if (np.abs(onnx_output_realistic) < 1e-6).all():
            logger.warning("⚠️  警告: ONNX模型输出全0！请检查模型权重是否正确加载。")
            print("⚠️  警告: ONNX模型输出全0！请检查模型权重是否正确加载。")
        elif max_diff_realistic > 1e-3:
            logger.warning(f"⚠️  警告: PyTorch和ONNX输出差异较大 (>{1e-3})")
            print(f"⚠️  警告: PyTorch和ONNX输出差异较大 (>{1e-3})")
        else:
            logger.info("✓ PyTorch和ONNX输出一致（符合实际场景的随机输入测试通过）")
            print("✓ PyTorch和ONNX输出一致（符合实际场景的随机输入测试通过）")
        
        # 进行多次随机测试（PyTorch vs ONNX对比）
        logger.info("进行多次随机测试（10次，PyTorch vs ONNX对比）...")
        print("\n进行多次随机测试（10次，PyTorch vs ONNX对比）...")
        all_passed = True
        for i in range(10):
            logger.debug(f"随机测试 {i+1}/10: 构造测试输入...")
            test_input = create_realistic_test_input(config, base_state_dim, state_history_steps)
            test_input_tensor = torch.from_numpy(test_input).to(device)
            
            with torch.no_grad():
                pytorch_out = wrapped_actor(test_input_tensor).cpu().numpy()
            onnx_out = session.run([output_name], {input_name: test_input})[0]
            
            max_diff = np.abs(pytorch_out - onnx_out).max()
            is_zero = (np.abs(onnx_out) < 1e-6).all()
            
            if is_zero or max_diff > 1e-3:
                logger.warning(f"随机测试 {i+1}/10: ⚠️  失败 (全0: {is_zero}, 最大差异: {max_diff:.6f})")
                print(f"  测试 {i+1}/10: ⚠️  失败 (全0: {is_zero}, 最大差异: {max_diff:.6f})")
                all_passed = False
            else:
                logger.debug(f"随机测试 {i+1}/10: ✓ 通过 (最大差异: {max_diff:.6f})")
                print(f"  测试 {i+1}/10: ✓ 通过 (最大差异: {max_diff:.6f})")
        
        if all_passed:
            logger.info("✓ 所有随机测试通过！")
            print("\n✓ 所有随机测试通过！")
        else:
            logger.warning("⚠️  部分随机测试失败，请检查模型。")
            print("\n⚠️  部分随机测试失败，请检查模型。")
    except ImportError:
        logger.warning("未安装onnxruntime包，跳过ONNX对比测试（PyTorch测试已完成）")
        logger.info("安装onnxruntime: pip install onnxruntime")
        print("提示: 未安装onnxruntime包，跳过ONNX对比测试（PyTorch测试已完成）")
        print("安装onnxruntime: pip install onnxruntime")
    
    # 使用get_action方法进行验证（模拟实际使用场景，使用prepare_state构造state）
    # 这部分不依赖onnxruntime，始终执行
    try:
        logger.info("="*60)
        logger.info("使用get_action方法进行验证（模拟multi_env_train.py的调用方式）...")
        logger.info("使用prepare_state构造state，然后复制state_history_steps+1份作为输入")
        print("\n" + "="*60)
        print("使用get_action方法进行验证（模拟multi_env_train.py的调用方式）...")
        print("使用prepare_state构造state，然后复制state_history_steps+1份作为输入")
        print("="*60)
        
        # 构造prepare_state所需的参数
        import random
        max_target_dist = config.get('max_target_dist', 3.0)
        target_reached_delta = config.get('target_reached_delta', 0.4)
        collision_delta = config.get('collision_delta', 0.3)
        scan_range = config.get('scan_range', 10.0)
        neglect_angle = config.get('neglect_angle', 0)
        logger.debug(f"prepare_state参数: max_target_dist={max_target_dist}, target_reached_delta={target_reached_delta}, "
                    f"collision_delta={collision_delta}, scan_range={scan_range}, neglect_angle={neglect_angle}")
        
        # 构造latest_scan（180度激光扫描，通常有180个点）
        # 考虑neglect_angle的影响，实际使用的扫描点会减少
        scan_points = 180  # 默认180个扫描点
        neglect_scan = int(np.ceil((neglect_angle / 180) * scan_points))
        actual_scan_len = scan_points - 2 * neglect_scan
        logger.debug(f"激光扫描参数: scan_points={scan_points}, neglect_scan={neglect_scan}, actual_scan_len={actual_scan_len}")
        
        # 构造latest_scan：每个点的距离在 (collision_delta, scan_range] 范围
        latest_scan = [random.uniform(collision_delta, scan_range) for _ in range(actual_scan_len)]
        logger.debug(f"构造latest_scan: 长度={len(latest_scan)}, 范围=[{min(latest_scan):.4f}, {max(latest_scan):.4f}]")
        
        # 构造其他参数
        distance = random.uniform(target_reached_delta, max_target_dist)
        angle = random.uniform(0, 2 * np.pi)
        cos = np.cos(angle)
        sin = np.sin(angle)
        collision = False  # 既不到达终点也不碰撞
        goal = False  # 既不到达终点也不碰撞
        last_action = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]  # [linear_velocity, angular_velocity]
        logger.debug(f"构造其他参数: distance={distance:.4f}, angle={angle:.4f}, cos={cos:.4f}, sin={sin:.4f}, "
                    f"collision={collision}, goal={goal}, last_action={last_action}")
        
        # 使用prepare_state构造state（返回base_state_dim长度的state）
        logger.info("构造prepare_state参数:")
        logger.info(f"  - latest_scan长度: {len(latest_scan)}")
        logger.info(f"  - distance: {distance:.4f}")
        logger.info(f"  - cos: {cos:.4f}, sin: {sin:.4f}")
        logger.info(f"  - collision: {collision} (False，表示不碰撞)")
        logger.info(f"  - goal: {goal} (False，表示不到达终点)")
        logger.info(f"  - last_action: {last_action}")
        print(f"\n构造prepare_state参数:")
        print(f"  - latest_scan长度: {len(latest_scan)}")
        print(f"  - distance: {distance:.4f}")
        print(f"  - cos: {cos:.4f}, sin: {sin:.4f}")
        print(f"  - collision: {collision} (False，表示不碰撞)")
        print(f"  - goal: {goal} (False，表示不到达终点)")
        print(f"  - last_action: {last_action}")
        
        logger.debug("调用prepare_state构造state...")
        state, terminal = sac_model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, last_action
        )
        logger.info(f"prepare_state返回的state长度: {len(state)} (应该是base_state_dim={base_state_dim})")
        logger.debug(f"prepare_state返回的terminal: {terminal}")
        print(f"  - prepare_state返回的state长度: {len(state)} (应该是base_state_dim={base_state_dim})")
        
        # 将state复制state_history_steps+1份作为输入（模拟multi_env_train.py的方式）
        if state_history_steps > 0:
            state_with_history = list(state) * (state_history_steps + 1)
            logger.info(f"复制后state长度: {len(state_with_history)} (state_history_steps={state_history_steps})")
            print(f"  - 复制后state长度: {len(state_with_history)} (state_history_steps={state_history_steps})")
        else:
            state_with_history = list(state)
            logger.info(f"未使用历史state，state长度: {len(state_with_history)}")
            print(f"  - 未使用历史state，state长度: {len(state_with_history)}")
        
        test_input_get_action = np.array(state_with_history, dtype=np.float32)
        logger.debug(f"最终测试输入shape: {test_input_get_action.shape}, dtype: {test_input_get_action.dtype}")
        
        # 使用PyTorch的get_action方法（add_noise=False，使用确定性动作）
        logger.info("使用PyTorch的get_action方法（add_noise=False）...")
        print("\n使用PyTorch的get_action方法（add_noise=False）...")
        logger.debug("执行PyTorch get_action...")
        pytorch_get_action_output = sac_model.get_action(test_input_get_action, add_noise=False)
        logger.info(f"PyTorch get_action输出shape: {pytorch_get_action_output.shape}")
        logger.info(f"PyTorch get_action输出: {pytorch_get_action_output}")
        logger.info(f"PyTorch get_action输出线速度与角速度: [{pytorch_get_action_output.min():.4f}, {pytorch_get_action_output.max():.4f}]")
        logger.info(f"PyTorch get_action输出是否全0: {(np.abs(pytorch_get_action_output) < 1e-6).all()}")
        print(f"PyTorch get_action输出shape: {pytorch_get_action_output.shape}")
        print(f"PyTorch get_action输出: {pytorch_get_action_output}")
        print(f"PyTorch get_action输出线速度与角速度: [{pytorch_get_action_output.min():.4f}, {pytorch_get_action_output.max():.4f}]")
        print(f"PyTorch get_action输出是否全0: {(np.abs(pytorch_get_action_output) < 1e-6).all()}")
        
        # 如果安装了onnxruntime，进行PyTorch和ONNX交替对比测试
        try:
            import onnxruntime as ort
            
            logger.info("检测到onnxruntime，进行PyTorch和ONNX交替对比测试...")
            print("\n检测到onnxruntime，进行PyTorch和ONNX交替对比测试...")
            
            # 创建ONNX Runtime会话
            session = ort.InferenceSession(str(output_path))
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            
            # 使用ONNX模型进行第一次推理（使用test_input_get_action）
            logger.info("使用ONNX模型进行推理（第一次测试）...")
            print("\n使用ONNX模型进行推理（第一次测试）...")
            logger.debug("执行ONNX Runtime推理...")
            onnx_get_action_output = session.run([output_name], {input_name: test_input_get_action})[0]
            logger.info(f"ONNX输出shape: {onnx_get_action_output.shape}")
            logger.info(f"ONNX输出: {onnx_get_action_output}")
            logger.info(f"ONNX输出线速度与角速度: [{onnx_get_action_output.min():.4f}, {onnx_get_action_output.max():.4f}]")
            logger.info(f"ONNX输出是否全0: {(np.abs(onnx_get_action_output) < 1e-6).all()}")
            print(f"ONNX输出shape: {onnx_get_action_output.shape}")
            print(f"ONNX输出: {onnx_get_action_output}")
            print(f"ONNX输出线速度与角速度: [{onnx_get_action_output.min():.4f}, {onnx_get_action_output.max():.4f}]")
            print(f"ONNX输出是否全0: {(np.abs(onnx_get_action_output) < 1e-6).all()}")
            
            # 对比第一次测试的输出
            max_diff_get_action = np.abs(pytorch_get_action_output - onnx_get_action_output).max()
            mean_diff_get_action = np.abs(pytorch_get_action_output - onnx_get_action_output).mean()
            logger.info(f"PyTorch get_action vs ONNX最大差异: {max_diff_get_action:.6f}")
            logger.info(f"PyTorch get_action vs ONNX平均差异: {mean_diff_get_action:.6f}")
            print(f"\nPyTorch get_action vs ONNX最大差异: {max_diff_get_action:.6f}")
            print(f"PyTorch get_action vs ONNX平均差异: {mean_diff_get_action:.6f}")
            
            if (np.abs(onnx_get_action_output) < 1e-6).all():
                logger.warning("⚠️  警告: ONNX模型输出全0！请检查模型权重是否正确加载。")
                print("⚠️  警告: ONNX模型输出全0！请检查模型权重是否正确加载。")
            elif max_diff_get_action > 1e-3:
                logger.warning(f"⚠️  警告: PyTorch get_action和ONNX输出差异较大 (>{1e-3})")
                print(f"⚠️  警告: PyTorch get_action和ONNX输出差异较大 (>{1e-3})")
            else:
                logger.info("✓ PyTorch get_action和ONNX输出一致（get_action验证通过）")
                print("✓ PyTorch get_action和ONNX输出一致（get_action验证通过）")
            
            # 进行多次get_action测试（PyTorch和ONNX交替进行，每次对比）
            logger.info("进行多次get_action测试（10次，PyTorch和ONNX交替对比）...")
            print("\n进行多次get_action测试（10次，PyTorch和ONNX交替对比）...")
            all_get_action_passed = True
            for i in range(10):
                logger.debug(f"get_action测试 {i+1}/10: 构造随机参数...")
                # 构造随机参数
                latest_scan = [random.uniform(collision_delta, scan_range) for _ in range(actual_scan_len)]
                distance = random.uniform(target_reached_delta, max_target_dist)
                angle = random.uniform(0, 2 * np.pi)
                cos = np.cos(angle)
                sin = np.sin(angle)
                collision = False  # 既不到达终点也不碰撞
                goal = False  # 既不到达终点也不碰撞
                last_action = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
                
                # 使用prepare_state构造state（collision和goal都设置为False）
                state, _ = sac_model.prepare_state(
                    latest_scan, distance, cos, sin, collision, goal, last_action
                )
                
                # 复制state_history_steps+1份
                if state_history_steps > 0:
                    state_with_history = list(state) * (state_history_steps + 1)
                else:
                    state_with_history = list(state)
                
                test_input = np.array(state_with_history, dtype=np.float32)
                
                # PyTorch推理
                logger.debug(f"get_action测试 {i+1}/10: 执行PyTorch推理...")
                pytorch_out = sac_model.get_action(test_input, add_noise=False)
                
                # ONNX推理
                logger.debug(f"get_action测试 {i+1}/10: 执行ONNX推理...")
                onnx_out = session.run([output_name], {input_name: test_input})[0]
                
                # 对比输出
                max_diff = np.abs(pytorch_out - onnx_out).max()
                mean_diff = np.abs(pytorch_out - onnx_out).mean()
                is_zero = (np.abs(onnx_out) < 1e-6).all()
                
                # 显示详细信息
                logger.info(f"get_action测试 {i+1}/10:")
                logger.info(f"  PyTorch输出: [{pytorch_out[0]:.6f}, {pytorch_out[1]:.6f}]")
                logger.info(f"  ONNX输出: [{onnx_out[0]:.6f}, {onnx_out[1]:.6f}]")
                logger.info(f"  最大差异: {max_diff:.6f}, 平均差异: {mean_diff:.6f}")
                
                if is_zero or max_diff > 1e-3:
                    logger.warning(f"get_action测试 {i+1}/10: ⚠️  失败 (全0: {is_zero}, 最大差异: {max_diff:.6f})")
                    print(f"  测试 {i+1}/10: ⚠️  失败")
                    print(f"    PyTorch输出: [{pytorch_out[0]:.6f}, {pytorch_out[1]:.6f}]")
                    print(f"    ONNX输出: [{onnx_out[0]:.6f}, {onnx_out[1]:.6f}]")
                    print(f"    最大差异: {max_diff:.6f}, 平均差异: {mean_diff:.6f}")
                    all_get_action_passed = False
                else:
                    logger.debug(f"get_action测试 {i+1}/10: ✓ 通过 (最大差异: {max_diff:.6f})")
                    print(f"  测试 {i+1}/10: ✓ 通过")
                    print(f"    PyTorch输出: [{pytorch_out[0]:.6f}, {pytorch_out[1]:.6f}]")
                    print(f"    ONNX输出: [{onnx_out[0]:.6f}, {onnx_out[1]:.6f}]")
                    print(f"    最大差异: {max_diff:.6f}, 平均差异: {mean_diff:.6f}")
            
            if all_get_action_passed:
                logger.info("✓ 所有get_action测试通过！")
                print("\n✓ 所有get_action测试通过！")
            else:
                logger.warning("⚠️  部分get_action测试失败，请检查模型。")
                print("\n⚠️  部分get_action测试失败，请检查模型。")
        except ImportError:
            # 如果没有onnxruntime，只进行PyTorch测试
            logger.info("未安装onnxruntime包，仅进行PyTorch get_action测试...")
            print("\n未安装onnxruntime包，仅进行PyTorch get_action测试...")
            logger.info("安装onnxruntime: pip install onnxruntime")
            print("提示: 安装onnxruntime后可进行PyTorch和ONNX对比测试")
            print("安装onnxruntime: pip install onnxruntime")
            
            # 进行多次PyTorch get_action测试（不依赖onnxruntime）
            logger.info("进行多次PyTorch get_action测试（10次，使用prepare_state构造state）...")
            print("\n进行多次PyTorch get_action测试（10次，使用prepare_state构造state）...")
            all_pytorch_get_action_passed = True
            for i in range(10):
                logger.debug(f"PyTorch get_action测试 {i+1}/10: 构造随机参数...")
                # 构造随机参数
                latest_scan = [random.uniform(collision_delta, scan_range) for _ in range(actual_scan_len)]
                distance = random.uniform(target_reached_delta, max_target_dist)
                angle = random.uniform(0, 2 * np.pi)
                cos = np.cos(angle)
                sin = np.sin(angle)
                collision = False  # 既不到达终点也不碰撞
                goal = False  # 既不到达终点也不碰撞
                last_action = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
                
                # 使用prepare_state构造state（collision和goal都设置为False）
                state, _ = sac_model.prepare_state(
                    latest_scan, distance, cos, sin, collision, goal, last_action
                )
                
                # 复制state_history_steps+1份
                if state_history_steps > 0:
                    state_with_history = list(state) * (state_history_steps + 1)
                else:
                    state_with_history = list(state)
                
                test_input = np.array(state_with_history, dtype=np.float32)
                
                logger.debug(f"PyTorch get_action测试 {i+1}/10: 执行推理...")
                pytorch_out = sac_model.get_action(test_input, add_noise=False)
                
                is_zero = (np.abs(pytorch_out) < 1e-6).all()
                
                if is_zero:
                    logger.warning(f"PyTorch get_action测试 {i+1}/10: ⚠️  失败 (输出全0)")
                    print(f"  PyTorch get_action测试 {i+1}/10: ⚠️  失败 (输出全0)")
                    all_pytorch_get_action_passed = False
                else:
                    logger.debug(f"PyTorch get_action测试 {i+1}/10: ✓ 通过 (输出线速度与角速度: [{pytorch_out.min():.4f}, {pytorch_out.max():.4f}])")
                    print(f"  PyTorch get_action测试 {i+1}/10: ✓ 通过 (输出线速度与角速度: [{pytorch_out.min():.4f}, {pytorch_out.max():.4f}])")
            
            if all_pytorch_get_action_passed:
                logger.info("✓ 所有PyTorch get_action测试通过！")
                print("\n✓ 所有PyTorch get_action测试通过！")
            else:
                logger.warning("⚠️  部分PyTorch get_action测试失败，请检查模型。")
                print("\n⚠️  部分PyTorch get_action测试失败，请检查模型。")
    except Exception as e:
        logger.warning(f"get_action测试失败: {e}", exc_info=True)
        print(f"警告: get_action测试失败: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"警告: 实际场景测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 输出ONNX模型结构信息
    logger.info("="*60)
    logger.info("ONNX模型结构信息:")
    print("\n" + "="*60)
    print("ONNX模型结构信息:")
    print("="*60)
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        
        # 基本信息
        logger.info(f"模型文件: {output_path}")
        logger.info(f"文件大小: {output_path.stat().st_size / (1024 * 1024):.2f} MB")
        print(f"模型文件: {output_path}")
        print(f"文件大小: {output_path.stat().st_size / (1024 * 1024):.2f} MB")
        
        # 输入输出信息
        logger.info("\n输入输出信息:")
        print("\n输入输出信息:")
        for inp in onnx_model.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else '?' for dim in inp.type.tensor_type.shape.dim]
            dtype = inp.type.tensor_type.elem_type
            dtype_name = onnx.TensorProto.DataType.Name(dtype) if dtype else "UNKNOWN"
            logger.info(f"  输入 '{inp.name}': shape={shape}, dtype={dtype_name}")
            print(f"  输入 '{inp.name}': shape={shape}, dtype={dtype_name}")
        
        for out in onnx_model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else '?' for dim in out.type.tensor_type.shape.dim]
            dtype = out.type.tensor_type.elem_type
            dtype_name = onnx.TensorProto.DataType.Name(dtype) if dtype else "UNKNOWN"
            logger.info(f"  输出 '{out.name}': shape={shape}, dtype={dtype_name}")
            print(f"  输出 '{out.name}': shape={shape}, dtype={dtype_name}")
        
        # 图结构信息
        logger.info(f"\n图结构信息:")
        print(f"\n图结构信息:")
        logger.info(f"  节点数量: {len(onnx_model.graph.node)}")
        logger.info(f"  初始值数量: {len(onnx_model.graph.initializer)}")
        logger.info(f"  Opset版本: {onnx_model.opset_import[0].version}")
        print(f"  节点数量: {len(onnx_model.graph.node)}")
        print(f"  初始值数量: {len(onnx_model.graph.initializer)}")
        print(f"  Opset版本: {onnx_model.opset_import[0].version}")
        
        # 统计操作类型
        op_types = {}
        for node in onnx_model.graph.node:
            op_type = node.op_type
            op_types[op_type] = op_types.get(op_type, 0) + 1
        
        logger.info(f"\n操作类型统计 (前10个):")
        print(f"\n操作类型统计 (前10个):")
        sorted_ops = sorted(op_types.items(), key=lambda x: x[1], reverse=True)[:10]
        for op_type, count in sorted_ops:
            logger.info(f"  {op_type}: {count}")
            print(f"  {op_type}: {count}")
        
        if len(op_types) > 10:
            logger.info(f"  ... 共 {len(op_types)} 种操作类型")
            print(f"  ... 共 {len(op_types)} 种操作类型")
        
        logger.info("="*60)
        print("="*60)
        
    except ImportError:
        logger.warning("未安装onnx包，跳过模型结构输出")
        print("警告: 未安装onnx包，跳过模型结构输出")
    except Exception as e:
        logger.warning(f"输出模型结构失败: {e}", exc_info=True)
        print(f"警告: 输出模型结构失败: {e}")
    
    return str(output_path)


def main():
    global logger
    
    parser = argparse.ArgumentParser(description='将PyTorch SAC模型转换为ONNX格式')
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='模型目录路径（包含SAC_actor.pth和配置文件）'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径（可选，默认自动查找目录下第一个名称包含"config"的yaml文件）'
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
    
    args = parser.parse_args()
    
    # 初始化日志系统
    model_dir_name = args.model_dir
    logger = setup_logging(model_dir_name)
    
    try:
        logger.info(f"命令行参数: model_dir={args.model_dir}, config={args.config}, "
                   f"output={args.output}, opset={args.opset}, device={args.device}")
        
        output_path = convert_to_onnx(
            model_dir=args.model_dir,
            config_path=args.config,
            output_path=args.output,
            opset_version=args.opset,
            device=args.device
        )
        logger.info("="*60)
        logger.info(f"转换完成！ONNX模型保存在: {output_path}")
        logger.info("="*60)
        print(f"\n转换完成！ONNX模型保存在: {output_path}")
    except Exception as e:
        logger.error(f"转换失败: {e}", exc_info=True)
        print(f"转换失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

