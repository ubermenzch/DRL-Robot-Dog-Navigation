import numpy as np
import torch
from torch import nn
from torch import distributions as pyd
import torch.nn.functional as F
import os
from collections import deque
import random
import math


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(
        self, input_dim, hidden_layers, output_dim, output_mod=None
    ):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_layers, output_dim, output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_layers, output_dim, output_mod=None):
    """
    构建多层感知机网络
    
    Args:
        input_dim: 输入维度
        hidden_layers: 隐藏层结构，列表形式，例如 [1024, 512] 表示两层，第一层1024个神经元，第二层512个神经元
        output_dim: 输出维度
        output_mod: 可选的输出层修改（如激活函数等）
    """
    mods = []
    if len(hidden_layers) == 0:
        # 如果没有隐藏层，直接从输入到输出
        mods.append(nn.Linear(input_dim, output_dim))
    else:
        # 第一层：从输入到第一个隐藏层
        mods.append(nn.Linear(input_dim, hidden_layers[0]))
        mods.append(nn.ReLU(inplace=True))
        # 中间隐藏层
        for i in range(len(hidden_layers) - 1):
            mods.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            mods.append(nn.ReLU(inplace=True))
        # 最后一层：从最后一个隐藏层到输出
        mods.append(nn.Linear(hidden_layers[-1], output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()
