# Multi-Thread-DRL-Robot-Navigation（多线程训练深度强化学习机器人导航）

基于 [reiniscimurs/DRL-Robot-Navigation-ROS2](https://github.com/reiniscimurs/DRL-Robot-Navigation-ROS2) 项目，添加了多实例 Gazebo 并行训练等功能。

训练示例：
<p align="center">
    <img width=100% src="https://github.com/reiniscimurs/DRL-Robot-Navigation-ROS2/blob/main/gif.gif">
</p>

## 项目来源

- ROS2 adapted from: https://github.com/tomasvr/turtlebot3_drlnav
- SAC adapted from: https://github.com/denisyarats/pytorch_sac

---

## 目录

- [环境要求](#环境要求)
- [安装教程](#安装教程)
- [训练方法](#训练方法)
- [服务器连接](#服务器连接)
- [常见问题](#常见问题)

---

## 环境要求

### 主要依赖

- **操作系统**: Ubuntu 20.04
- **ROS2**: Foxy
- **深度学习框架**: PyTorch
- **仿真环境**: Gazebo 11

---

## 安装教程

### 1. Clone 仓库

```shell
cd ~
git clone https://github.com/reiniscimurs/DRL-Robot-Navigation-ROS2.git
cd DRL-Robot-Navigation-ROS2
```

如果 HTTPS 克隆失败，可以使用 SSH 方式：

```shell
# 生成 SSH 密钥（如果还没有）
ssh-keygen -t ed25519 -C "your_email@example.com"
# 查看公钥内容
cat ~/.ssh/id_ed25519.pub
# 将公钥添加到 GitHub 账户后，使用 SSH 克隆
git clone git@github.com:reiniscimurs/DRL-Robot-Navigation-ROS2.git
```

### 2. 安装 ROS2 Foxy

在 Ubuntu 20.04 系统中，推荐使用鱼香 ROS 一键安装：

```shell
wget http://fishros.com/install -O fishros && . fishros
```

这将自动完成 ROS2 Foxy 安装、源配置、ROS 环境配置和 rosdep 配置。

### 3. 安装编译工具

```shell
cd ~/DRL-Robot-Navigation-ROS2
sudo apt update
sudo apt install python3-colcon-common-extensions
colcon build
```

### 4. 配置环境变量

将以下内容添加到 `~/.bashrc` 文件中：

```shell
export DRLNAV_BASE_PATH=~/DRL-Robot-Navigation-ROS2
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/DRL-Robot-Navigation-ROS2/src/turtlebot3_simulations/turtlebot3_gazebo/models
export TURTLEBOT3_MODEL=waffle
source /opt/ros/foxy/setup.bash
source ~/DRL-Robot-Navigation-ROS2/install/setup.bash
```

然后执行：

```shell
source ~/.bashrc
```

**注意**: `ROS_DOMAIN_ID` 不需要手动设置，启动脚本会自动处理。如果使用启动脚本（推荐方式），无需额外配置。

### 5. 安装 Gazebo 11

```shell
sudo apt install gazebo11
sudo apt install ros-foxy-gazebo-ros-pkgs
sudo apt install ros-foxy-gazebo-*
```

### 6. 安装 PyTorch

根据你的 Python 版本和 CUDA 版本选择合适的 PyTorch 版本。以 CUDA 12.1、Python 3.8 为例：

```shell
# 先安装 networkx（兼容性要求）
pip install "networkx<3.0" --no-deps
# 安装 PyTorch 及匹配的 CUDA 工具包
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

如果使用 CPU 版本：

```shell
pip3 install torch torchvision torchaudio
```

### 7. 安装其他 Python 依赖

```shell
pip install squaternion tqdm
```

### 8. 下载 Gazebo 模型库

原项目自带的模型可能不全，需要下载完整的模型库：

```shell
# 下载模型库（如果网络可行）
wget https://github.com/osrf/gazebo_models/archive/refs/heads/master.zip -O gazebo_models-master.zip

# 或者手动下载后上传到服务器，然后解压
unzip gazebo_models-master.zip

# 将模型移动到项目模型文件夹
mv ~/gazebo_models-master/* ~/DRL-Robot-Navigation-ROS2/src/turtlebot3_simulations/turtlebot3_gazebo/models
```

---

## 训练方法

项目支持两种训练模式：**单环境训练**和**多环境并行训练**。

### 单环境训练

单环境训练适合调试和可视化，使用一个 Gazebo 环境进行训练。

#### 使用启动脚本 `start_training.sh`

`start_training.sh` 是单环境训练的启动脚本，会自动处理 Gazebo 启动、环境配置等操作。

**基本用法：**

```shell
cd ~/DRL-Robot-Navigation-ROS2
./start_training.sh
```

**运行模式：**

脚本支持两种运行模式（在 `config/train.yaml` 中配置 `run_mode` 参数）：

- `run_mode: 1` - **后台模式**（默认）：所有输出保存到日志文件，终端关闭后训练继续运行
- `run_mode: 2` - **可视化模式**：Gazebo 和 RViz 会显示图形界面，适合调试和可视化

**命令行参数：**

```shell
# 启用调试模式（显示所有执行的指令）
./start_training.sh --debug

# 通过命令行参数覆盖配置文件中的运行模式
./start_training.sh --run_mode=2
```

**脚本功能：**

- 自动检测并设置 DISPLAY 环境变量
- 自动加载 ROS2 和工作空间环境
- 自动启动 Gazebo（根据运行模式选择 GUI 或无头模式）
- 自动启动 RViz（仅在可视化模式下）
- 自动启动训练脚本
- 自动处理信号和清理（Ctrl+C 时会清理所有进程）

**日志文件位置：**

```shell
log/single_env_training/train_<时间戳>/train_<时间戳>.log
```

**停止训练：**

- 前台运行：按 `Ctrl+C` 停止
- 后台运行：使用 `ps aux | grep train.py` 查找进程，然后 `kill <PID>`

#### 手动启动（不推荐）

如果需要手动启动，可以按照以下步骤：

1. **启动 Gazebo**（在一个终端）：

```shell
export ROS_DOMAIN_ID=1
source /opt/ros/foxy/setup.bash
source ~/DRL-Robot-Navigation-ROS2/install/setup.bash
ros2 launch turtlebot3_gazebo ros2_drl.launch.py
```

2. **启动训练脚本**（在另一个终端）：

```shell
cd ~/DRL-Robot-Navigation-ROS2
python3 src/drl_navigation_ros2/train.py
```

**注意**: 手动启动需要手动设置环境变量和清理进程，推荐使用启动脚本。

### 多环境并行训练

多环境并行训练可以同时运行多个 Gazebo 环境，大幅提升数据收集效率。

#### 使用启动脚本 `start_multi_env_training.sh`

`start_multi_env_training.sh` 是多环境并行训练的启动脚本，会自动启动多个 Gazebo 环境并管理训练过程。

**基本用法：**

```shell
cd ~/DRL-Robot-Navigation-ROS2
./start_multi_env_training.sh
```

**命令行参数：**

```shell
# 启用调试模式（显示所有执行的指令）
./start_multi_env_training.sh --debug
```

**注意**: 多环境训练脚本**强制使用后台模式**，无论是否传入参数，所有输出都会保存到日志文件。终端关闭后训练会继续运行。

**配置训练参数：**

编辑 `config/train.yaml` 文件，主要参数包括：

- `num_envs`: 并行环境数量（一个环境约占用 800MB 显存）
- `gpu_id`: 使用的 GPU 编号（设置为 `-1` 可自动选择最佳 GPU）
- `batch_size`: 训练批次大小
- `training_iterations`: 每次训练的迭代次数
- `buffer_size`: 经验回放缓冲区大小
- `critic_loss_threshold`: Critic 损失阈值，低于此值时才更新模型
- 其他训练和环境参数

**脚本功能：**

- 自动启动多个独立的 Gazebo 环境（每个环境使用不同的 `ROS_DOMAIN_ID`）
- 自动清理和重新挂载共享内存（避免共享内存不足）
- 自动选择最佳 GPU（如果 `gpu_id` 设置为 `-1`）
- 并行收集数据
- 当收集到足够数据时自动触发训练
- 训练完成后更新模型，环境继续使用新模型收集数据
- 自动处理信号和清理（Ctrl+C 时会清理所有进程）

**日志文件位置：**

```shell
log/multi_env_training/train_<时间戳>/train_<时间戳>.log
```

**查看训练状态：**

```shell
# 实时查看训练日志
tail -f log/multi_env_training/train_*/train_*.log

# 查看后台进程
ps aux | grep multi_env_train

# 查看 PID 文件
cat tmp/multi_env_training.pid
```

**停止训练：**

多环境训练运行在后台，停止方法：

- 使用停止脚本（如果存在）：`./stop_multi_env_training.sh`
- 手动停止：查找进程 PID 后使用 `kill <PID>`，或使用 `pkill -f multi_env_train`

**注意事项：**

1. **显存占用**：每个环境约占用 800MB 显存，请根据 GPU 显存合理设置 `num_envs`
2. **共享内存**：脚本会自动将共享内存重新挂载为 2GB，如果仍然不足，可能需要手动调整
3. **ROS_DOMAIN_ID**：脚本会自动为每个环境分配不同的 `ROS_DOMAIN_ID`（从 1 开始），无需手动设置
4. **模型更新策略**：只有当 Critic 损失低于阈值时，数据收集进程才会同步训练线程的最新模型，这有助于训练稳定性

### 训练配置说明

所有训练参数都在 `config/train.yaml` 文件中配置，包括：

- **环境参数**: 最大速度、扫描范围、目标距离等
- **模型参数**: 网络结构、状态维度、动作维度等
- **训练参数**: 学习率、批次大小、缓冲区大小等
- **奖励函数参数**: 各种奖励和惩罚的权重

详细参数说明请参考 `config/train.yaml` 文件中的注释。

### 查看训练日志

```shell
# 单环境训练日志
tail -f log/single_env_training/train_*/train_*.log

# 多环境训练日志
tail -f log/multi_env_training/train_*/train_*.log
```

---

## 服务器连接

### SSH 连接

```shell
# 基本连接（请替换为你的服务器信息）
ssh -p <端口号> <用户名>@<服务器IP地址>
# 示例：ssh -p 22 user@192.168.1.100
```

### VNC 连接 Docker 图形化界面

如果需要图形化界面，可以设置 VNC。

#### 1. 安装 TurboVNC 和 VirtualGL

```shell
echo 'deb [signed-by=/etc/apt/trusted.gpg.d/VirtualGL.gpg] https://packagecloud.io/dcommander/virtualgl/any/ any main' \
  | sudo tee /etc/apt/sources.list.d/VirtualGL.list > /dev/null

echo 'deb [signed-by=/etc/apt/trusted.gpg.d/TurboVNC.gpg] https://packagecloud.io/dcommander/turbovnc/any/ any main' \
  | sudo tee /etc/apt/sources.list.d/TurboVNC.list > /dev/null

sudo apt-get update
sudo apt-get install -y turbovnc virtualgl libjpeg-turbo8 libjpeg-turbo-progs
```

#### 2. 安装 Xfce 会话

```shell
sudo apt-get update
sudo apt-get install -y xfce4 xfce4-session xfce4-terminal dbus-x11
mkdir -p ~/.vnc
cat > ~/.vnc/turbovncserver.conf << 'EOF'
$wm = "xfce";
EOF
```

#### 3. 启动 VNC 服务器

```shell
# 在本地机器上启动 SSH 隧道（保持终端不关闭）
# 请替换为你的服务器信息
ssh -p <端口号> -L <本地端口>:127.0.0.1:<远程端口> <用户名>@<服务器IP地址>
# 示例：ssh -p 22 -L 5901:127.0.0.1:5901 user@192.168.1.100

# 在服务器/容器中启动 VNC 服务器
# 请替换 <显示编号> 为你想要的显示编号（例如 :1, :2 等）
/opt/TurboVNC/bin/vncserver :<显示编号> -localhost -geometry 1920x1080 -depth 24 -xstartup ~/.vnc/xstartup.zh

# 检查运行情况（请替换 <显示编号> 为你使用的显示编号）
tail -n 40 ~/.vnc/*:<显示编号>.log || true
```

#### 4. 连接 VNC

使用 TurboVNC Viewer 连接到 `localhost:59XX`（端口号对应显示编号，例如使用 `:1` 则端口为 `5901`，使用 `:2` 则端口为 `5902`）。

#### 5. 停止 VNC 服务器

```shell
# 请替换 <显示编号> 为你使用的显示编号
/opt/TurboVNC/bin/vncserver -kill :<显示编号> || true
# 示例：/opt/TurboVNC/bin/vncserver -kill :1 || true
```

### GPU 监控

查看 GPU 使用情况：

```shell
# 查看所有 GPU
nvidia-smi

# 查看指定 GPU（请替换 <GPU编号> 为实际的 GPU 编号），每 1 秒刷新一次
nvidia-smi -i <GPU编号> -l 1
# 示例：nvidia-smi -i 0 -l 1
```

---

## 常见问题

### 1. 训练启动失败

- 检查 Gazebo 模型库是否完整下载
- 检查环境变量是否正确配置
- 检查 ROS2 环境是否正确加载

### 2. 多环境训练显存不足

- 减少 `num_envs` 参数（每个环境约占用 800MB 显存）
- 减少 `batch_size` 参数
- 使用 `use_float64_for_buffer: false` 以降低内存占用

### 3. Gazebo 无法启动

- 检查 DISPLAY 环境变量（如果使用 GUI 模式）
- 检查 Gazebo 模型路径是否正确
- 检查 ROS_DOMAIN_ID 是否冲突

### 4. 训练过程中断

- 检查系统资源（内存、显存、磁盘空间）
- 查看训练日志文件排查错误
- 检查是否有其他进程占用资源

---

## 项目结构

```
DRL-Robot-Navigation-ROS2/
├── config/                 # 配置文件
│   └── train.yaml         # 训练配置文件
├── src/
│   └── drl_navigation_ros2/
│       ├── train.py       # 单环境训练脚本
│       ├── multi_env_train.py  # 多环境并行训练脚本
│       ├── drl_navigation.py  # 导航脚本
│       ├── evaluate.py    # 评估脚本
│       └── SAC/           # SAC 算法实现
├── models/                # 模型保存目录
├── log/                   # 日志保存目录
├── start_training.sh      # 单环境训练启动脚本
└── start_multi_env_training.sh  # 多环境训练启动脚本
```

---

## 许可证

本项目基于原 [DRL-Robot-Navigation-ROS2](https://github.com/reiniscimurs/DRL-Robot-Navigation-ROS2) 项目，请参考原项目的许可证。

---

## 贡献

欢迎提交 Issue 和 Pull Request！
