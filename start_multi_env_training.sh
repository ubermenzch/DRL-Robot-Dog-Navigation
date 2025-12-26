#!/bin/bash
# 多环境并行训练启动脚本

# ===================== 后台运行支持 =====================
# 使用方法：
# 前台运行：./start_multi_env_training.sh
# 后台运行：./start_multi_env_training.sh --daemon
# 查看后台进程：ps aux | grep multi_env_train
# 停止后台进程：./stop_multi_env_training.sh

# ===================== 模型更新模式 =====================
# 功能说明：
# 数据收集进程不停地收集数据，但并不每次开始收集数据前都更新模型，
# 而是直到"前X次训练的平均损失小于阈值时，数据收集进程才同步训练线程的模型"
# 
# 配置参数：
# CRITIC_LOSS_THRESHOLD=50.0   # 前X次平均损失<50时才更新模型
# AVG_LOSS_WINDOW_SIZE=5       # 计算前5次训练的平均损失

DEBUG_MODE=false
DAEMON_MODE=false
# CLI 参数解析
for arg in "$@"; do
    if [ "$arg" = "--daemon" ]; then
        DAEMON_MODE=true
        echo "启动后台模式..."
    elif [ "$arg" = "--debug" ]; then
        DEBUG_MODE=true
    fi
done

# ===================== 环境变量初始化 =====================
# 清除可能干扰的外部环境变量，确保脚本自包含，不依赖外部环境
# 注意：这些unset操作不会影响外部终端，因为脚本在子shell中运行
unset DISPLAY ROS_DOMAIN_ID CUDA_VISIBLE_DEVICES GAZEBO_IP GAZEBO_MASTER_URI GAZEBO_GUI 2>/dev/null || true
unset GAZEBO_MODEL_PATH GAZEBO_RESOURCE_PATH TURTLEBOT3_MODEL 2>/dev/null || true
unset ROS_PACKAGE_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH 2>/dev/null || true

# ===================== 项目路径与配置文件 =====================
# 将所有临时/生成文件固定在当前项目目录，避免权限问题
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config/train.yaml"

parse_yaml_value() {
    local key="$1"
    python3 - "$key" <<PY
import yaml, sys
cfg_path = "$CONFIG_FILE"
key = sys.argv[1] if len(sys.argv) > 1 else ""
try:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    val = cfg.get(key, "")
    if isinstance(val, bool):
        print("true" if val else "false")
    else:
        print(val)
except FileNotFoundError:
    print("")
except Exception:
    print("")
PY
}

# ===================== 信号处理 =====================
# 存储所有启动的进程PID
GAZEBO_PIDS=()
TRAINING_PID=""
ALREADY_CLEANED=false

# 信号处理清理函数
cleanup_all() {
    # 避免重复清理
    if [ "$ALREADY_CLEANED" = true ]; then
        exit 0
    fi
    ALREADY_CLEANED=true
    echo -e "\n收到中断信号，正在清理所有进程..."
    
    # 调用 clean.sh 进行清理
    local clean_script="$SCRIPT_DIR/clean.sh"
    if [ -f "$clean_script" ] && [ -x "$clean_script" ]; then
        bash "$clean_script" || true
    else
        echo "警告: clean.sh 不存在，使用内置清理逻辑"
    # 停止训练进程
    if [ ! -z "$TRAINING_PID" ] && kill -0 "$TRAINING_PID" 2>/dev/null; then
        kill -TERM "$TRAINING_PID" 2>/dev/null
        sleep 2
            kill -0 "$TRAINING_PID" 2>/dev/null && kill -KILL "$TRAINING_PID" 2>/dev/null
        fi
    # 停止所有Gazebo进程
    for pid in "${GAZEBO_PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null
    done
        sleep 2
        pkill -f "ros2 launch turtlebot3_gazebo|gazebo|multi_env_train.py" 2>/dev/null
    fi
    
    echo "所有进程已清理完成"
    exit 0
}

# 注册信号处理（包含 EXIT，确保异常或正常退出都清理）
trap cleanup_all SIGINT SIGTERM EXIT

# ===================== 清理共享内存 =====================
# 重新挂载共享内存为2GB，避免共享内存不够
echo "重新挂载共享内存为2GB..."
sudo mount -o remount,size=2G /dev/shm
echo "共享内存重新挂载完成"

echo "清理ROS2共享内存段..."
sudo rm -f /dev/shm/fastrtps_*
echo "共享内存清理完成"
echo

# ===================== 从YAML读取配置（为缺省做兜底） =====================
NUM_ENVS=$(parse_yaml_value "num_envs")
MAX_STEPS_RATIO=$(parse_yaml_value "max_steps_ratio")
MAX_STEPS_MIN=$(parse_yaml_value "max_steps_min")
BATCH_SIZE=$(parse_yaml_value "batch_size")
TRAINING_ITERATIONS=$(parse_yaml_value "training_iterations")
SAVE_EVERY=$(parse_yaml_value "save_every")
BUFFER_SIZE=$(parse_yaml_value "buffer_size")
REPORT_EVERY=$(parse_yaml_value "report_every")
STATS_WINDOW_SIZE=$(parse_yaml_value "stats_window_size")
MAX_TRAINING_COUNT=$(parse_yaml_value "max_training_count")
CRITIC_LOSS_THRESHOLD=$(parse_yaml_value "critic_loss_threshold")
ACTOR_UPDATE_FREQUENCY=$(parse_yaml_value "actor_update_frequency")
CRITIC_TARGET_UPDATE_FREQUENCY=$(parse_yaml_value "critic_target_update_frequency")
HIDDEN_DIM=$(parse_yaml_value "hidden_dim")
HIDDEN_DEPTH=$(parse_yaml_value "hidden_depth")
AVG_LOSS_WINDOW_SIZE=$(parse_yaml_value "avg_loss_window_size")
GPU_ID=$(parse_yaml_value "gpu_id")

MAX_VELOCITY=$(parse_yaml_value "max_velocity")
NEGLECT_ANGLE=$(parse_yaml_value "neglect_angle")
MAX_YAWRATE=$(parse_yaml_value "max_yawrate")
SCAN_RANGE=$(parse_yaml_value "scan_range")
MAX_TARGET_DIST=$(parse_yaml_value "max_target_dist")
INIT_TARGET_DISTANCE=$(parse_yaml_value "init_target_distance")
TARGET_DIST_INCREASE=$(parse_yaml_value "target_dist_increase")
TARGET_REACHED_DELTA=$(parse_yaml_value "target_reached_delta")
COLLISION_DELTA=$(parse_yaml_value "collision_delta")
WORLD_SIZE=$(parse_yaml_value "world_size")
OBS_MIN_DIST=$(parse_yaml_value "obs_min_dist")
OBS_NUM=$(parse_yaml_value "obs_num")

ACTION_DIM=$(parse_yaml_value "action_dim")
MAX_ACTION=$(parse_yaml_value "max_action")
STATE_DIM=$(parse_yaml_value "state_dim")

IS_CODE_DEBUG=$(parse_yaml_value "is_code_debug")

MODEL_SAVE_DIR=$(parse_yaml_value "model_save_dir")
# 优先使用 load_path，如果没有则使用 model_load_dir（向后兼容）
MODEL_LOAD_DIR=$(parse_yaml_value "load_path")
if [ -z "$MODEL_LOAD_DIR" ]; then
MODEL_LOAD_DIR=$(parse_yaml_value "model_load_dir")
fi
LOAD_MODEL=$(parse_yaml_value "load_model")

LOG_DIR=$(parse_yaml_value "multi_env_log_dir")
GAZEBO_WAIT_TIME=$(parse_yaml_value "gazebo_wait_time")

# 兜底默认值（若YAML缺项）
NUM_ENVS=${NUM_ENVS:-4}
MAX_STEPS_RATIO=${MAX_STEPS_RATIO:-100}
MAX_STEPS_MIN=${MAX_STEPS_MIN:-50}
BATCH_SIZE=${BATCH_SIZE:-40}
TRAINING_ITERATIONS=${TRAINING_ITERATIONS:-500}
SAVE_EVERY=${SAVE_EVERY:-50}
BUFFER_SIZE=${BUFFER_SIZE:-100000}
REPORT_EVERY=${REPORT_EVERY:-20}
STATS_WINDOW_SIZE=${STATS_WINDOW_SIZE:-100}
MAX_TRAINING_COUNT=${MAX_TRAINING_COUNT:-1000}
CRITIC_LOSS_THRESHOLD=${CRITIC_LOSS_THRESHOLD:-100.0}
ACTOR_UPDATE_FREQUENCY=${ACTOR_UPDATE_FREQUENCY:-1}
CRITIC_TARGET_UPDATE_FREQUENCY=${CRITIC_TARGET_UPDATE_FREQUENCY:-4}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
HIDDEN_DEPTH=${HIDDEN_DEPTH:-2}
AVG_LOSS_WINDOW_SIZE=${AVG_LOSS_WINDOW_SIZE:-5}
GPU_ID=${GPU_ID:-0}

MAX_VELOCITY=${MAX_VELOCITY:-2.5}
NEGLECT_ANGLE=${NEGLECT_ANGLE:-30}
MAX_YAWRATE=${MAX_YAWRATE:-30.0}
SCAN_RANGE=${SCAN_RANGE:-10}
MAX_TARGET_DIST=${MAX_TARGET_DIST:-3.0}
INIT_TARGET_DISTANCE=${INIT_TARGET_DISTANCE:-3.0}
TARGET_DIST_INCREASE=${TARGET_DIST_INCREASE:-0.001}
TARGET_REACHED_DELTA=${TARGET_REACHED_DELTA:-0.3}
COLLISION_DELTA=${COLLISION_DELTA:-0.25}
WORLD_SIZE=${WORLD_SIZE:-10}
OBS_MIN_DIST=${OBS_MIN_DIST:-2.0}
OBS_NUM=${OBS_NUM:-8}

ACTION_DIM=${ACTION_DIM:-2}
MAX_ACTION=${MAX_ACTION:-1}
STATE_DIM=${STATE_DIM:-25}

IS_CODE_DEBUG=${IS_CODE_DEBUG:-false}

MODEL_SAVE_DIR=${MODEL_SAVE_DIR:-"/home/zc/DRL-Robot-Navigation-ROS2/models/multi_env_SAC"}
MODEL_LOAD_DIR=${MODEL_LOAD_DIR:-"/home/zc/DRL-Robot-Navigation-ROS2/src/drl_navigation_ros2/models/SAC"}
LOAD_MODEL=${LOAD_MODEL:-true}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_DIR_BASE=${LOG_DIR:-"$SCRIPT_DIR/log/multi_env_training"}
LOG_DIR="$LOG_DIR_BASE/train_${TIMESTAMP}"
LOGFILE="$LOG_DIR/train_${TIMESTAMP}.log"
MODEL_SAVE_DIR_TS="${MODEL_SAVE_DIR%/}/$TIMESTAMP"
GAZEBO_WAIT_TIME=${GAZEBO_WAIT_TIME:-15}
export TRAINING_TIMESTAMP="$TIMESTAMP"

# 定义临时目录（使用项目目录）
TMP_DIR="$SCRIPT_DIR/tmp"
PID_FILE="$TMP_DIR/multi_env_training.pid"

# 预先创建日志、模型和临时目录（在首次 tee 之前）
mkdir -p "$LOG_DIR"
mkdir -p "$MODEL_SAVE_DIR_TS"
mkdir -p "$TMP_DIR"

# ===================== 输出函数 =====================
log_output() {
    if [ "$DAEMON_MODE" = true ]; then
        # 后台模式：只输出到日志（使用 stderr 避免被命令替换捕获）
        echo "$@" >> "$LOGFILE" 2>&1
    else
        # 前台模式：输出到终端和日志
        echo "$@" | tee -a "$LOGFILE"
    fi
}

# 强制后台模式（多环境训练只支持后台模式）
    DAEMON_MODE=true

# ===================== GPU自动选择 =====================
select_best_gpu() {
    # 当 gpu_id 为 -1 时，自动选择最佳 GPU
    # 注意：所有日志输出只写入日志文件，避免重复输出
    echo "开始自动选择 GPU（监控 5 秒）..." >> "$LOGFILE" 2>&1
    
    # 检查 nvidia-smi 是否可用
    if ! command -v nvidia-smi &> /dev/null; then
        echo "警告: nvidia-smi 不可用，无法自动选择 GPU，使用默认 GPU 0" >> "$LOGFILE" 2>&1
        echo "0"
        return
    fi
    
    # 检查 bc 是否可用（用于浮点数计算）
    if ! command -v bc &> /dev/null; then
        echo "警告: bc 不可用，无法自动选择 GPU，使用默认 GPU 0" >> "$LOGFILE" 2>&1
        echo "0"
        return
    fi
    
    # 获取所有 GPU 数量
    local num_gpus=$(nvidia-smi --list-gpus | wc -l)
    if [ "$num_gpus" -eq 0 ]; then
        echo "警告: 未检测到 GPU，使用默认 GPU 0" >> "$LOGFILE" 2>&1
        echo "0"
        return
    fi
    
    echo "检测到 $num_gpus 个 GPU，开始监控..." >> "$LOGFILE" 2>&1
    
    # 使用临时文件存储统计数据
    local tmp_file="/tmp/gpu_stats_$$.txt"
    > "$tmp_file"  # 清空文件
    
    # 监控 5 秒，每秒采样一次
    local monitor_duration=5
    for sample in $(seq 1 $monitor_duration); do
        # 使用 nvidia-smi 查询 GPU 利用率和显存利用率
        # 格式: index,utilization.gpu,memory.used,memory.total
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read -r idx util mem_used mem_total; do
            # 去除空格
            idx=$(echo "$idx" | xargs)
            util=$(echo "$util" | xargs)
            mem_used=$(echo "$mem_used" | xargs)
            mem_total=$(echo "$mem_total" | xargs)
            
            # 计算显存利用率百分比
            if [ "$mem_total" -gt 0 ] && [ -n "$mem_used" ] && [ -n "$mem_total" ]; then
                mem_util=$(echo "scale=2; $mem_used * 100 / $mem_total" | bc 2>/dev/null || echo "0")
            else
                mem_util=0
            fi
            
            # 写入临时文件（追加模式）
            echo "$idx|$util|$mem_util" >> "$tmp_file"
        done
        sleep 1
    done
    
    # 汇总统计数据（使用 Python 处理，更可靠）
    local python_output=$(python3 << PYEOF
import sys
import os

tmp_file = "$tmp_file"
log_messages = []

if not os.path.exists(tmp_file):
    log_messages.append("警告: 临时文件不存在，使用默认 GPU 0")
    print("\\n".join(log_messages), file=sys.stderr)
    print("0")
    sys.exit(0)

# 读取所有样本数据
gpu_data = {}
with open(tmp_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('|')
        if len(parts) != 3:
            continue
        try:
            idx = int(parts[0])
            util = float(parts[1])
            mem_util = float(parts[2])
            
            if idx not in gpu_data:
                gpu_data[idx] = {'util_sum': 0.0, 'mem_sum': 0.0, 'count': 0}
            
            gpu_data[idx]['util_sum'] += util
            gpu_data[idx]['mem_sum'] += mem_util
            gpu_data[idx]['count'] += 1
        except (ValueError, IndexError):
            continue

# 计算平均值并筛选
best_gpu = None
best_util = None
best_mem_util = None

log_messages.append("GPU 监控结果：")
for idx in sorted(gpu_data.keys()):
    data = gpu_data[idx]
    if data['count'] > 0:
        avg_util = data['util_sum'] / data['count']
        avg_mem_util = data['mem_sum'] / data['count']
        
        log_messages.append(f"  GPU {idx}: 平均利用率={avg_util:.2f}%, 平均显存利用率={avg_mem_util:.2f}%")
        
        # 去除平均利用率 90% 以上的 GPU
        if avg_util >= 90.0:
            log_messages.append(f"    -> GPU {idx} 利用率过高（{avg_util:.2f}% >= 90%），已排除")
            continue
        
        # 选择最佳 GPU（利用率最低，如果相同则显存利用率最低）
        if best_gpu is None:
            best_gpu = idx
            best_util = avg_util
            best_mem_util = avg_mem_util
        else:
            if avg_util < best_util:
                best_gpu = idx
                best_util = avg_util
                best_mem_util = avg_mem_util
            elif abs(avg_util - best_util) < 0.01 and avg_mem_util < best_mem_util:
                best_gpu = idx
                best_util = avg_util
                best_mem_util = avg_mem_util

if best_gpu is None:
    log_messages.append("警告: 所有 GPU 利用率都超过 90%，使用 GPU 0")
    print("\\n".join(log_messages), file=sys.stderr)
    print("0")
else:
    log_messages.append(f"已选择 GPU {best_gpu} (平均利用率={best_util:.2f}%, 平均显存利用率={best_mem_util:.2f}%)")
    print("\\n".join(log_messages), file=sys.stderr)
    print(str(best_gpu))

# 清理临时文件
try:
    os.remove(tmp_file)
except:
    pass
PYEOF
    2>&1)
    
    # 将 Python 的输出分离：stderr（日志）和 stdout（GPU ID）
    local best_gpu=$(echo "$python_output" | tail -1)
    local log_msg=$(echo "$python_output" | head -n -1)
    
    # 输出日志信息（只写入日志文件，避免重复输出）
    if [ -n "$log_msg" ]; then
        echo "$log_msg" >> "$LOGFILE" 2>&1
    fi
    
    # 清理临时文件（如果 Python 没有清理）
    rm -f "$tmp_file" 2>/dev/null || true
    
    # 只输出 GPU ID 到 stdout（避免被日志污染）
    echo "$best_gpu"
}

# ===================== 设置CUDA =====================
setup_cuda() {
    # 如果 gpu_id 为 -1，自动选择最佳 GPU
    if [ "$GPU_ID" = "-1" ]; then
        GPU_ID=$(select_best_gpu)
        log_output "自动选择 GPU: $GPU_ID"
    fi
    
    # 设置 CUDA_VISIBLE_DEVICES（如果提供了gpu_id），并为 PyTorch 映射到本地索引
    TORCH_GPU_ID="$GPU_ID"
    if [ -n "$GPU_ID" ] && [ "$GPU_ID" != "-1" ]; then
        export CUDA_VISIBLE_DEVICES="$GPU_ID"
        # 设置后，PyTorch 视角下可见 GPU 从 0 开始，单卡时应使用 0
        TORCH_GPU_ID=0
        log_output "已设置 CUDA_VISIBLE_DEVICES=$GPU_ID（物理卡），PyTorch 使用 cuda:$TORCH_GPU_ID"
    else
        # 如果没有指定GPU_ID，确保CUDA_VISIBLE_DEVICES未设置，使用系统默认
        unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
        log_output "未指定 GPU_ID，使用默认 CUDA 设备可见性"
    fi
    export TORCH_GPU_ID
}

# ===================== 初始化日志 =====================
init_logging() {
    # 创建日志文件夹（如果不存在）
    mkdir -p "$LOG_DIR"
    # 如果日志文件不存在或为空，初始化它
    if [ ! -f "$LOGFILE" ] || [ ! -s "$LOGFILE" ]; then
        > "$LOGFILE"
        echo "===== 多环境并行训练启动 - $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOGFILE"
    fi
    echo "GPU 选择:" >> "$LOGFILE"
    echo "  - 配置文件 gpu_id (物理卡): ${GPU_ID:-未设置}" >> "$LOGFILE"
    echo "  - CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-未设置}" >> "$LOGFILE"
    echo "  - PyTorch 使用的 cuda 索引: ${TORCH_GPU_ID:-未设置}" >> "$LOGFILE"
    # 记录用户信息（用于排查 Docker 用户映射问题）
    echo "用户信息:" >> "$LOGFILE"
    echo "  - 容器内用户名: $(whoami)" >> "$LOGFILE"
    echo "  - 容器内 UID: $(id -u)" >> "$LOGFILE"
    echo "  - 容器内 GID: $(id -g)" >> "$LOGFILE"
    echo "  - 容器内完整用户信息: $(id)" >> "$LOGFILE"
    echo "  - 环境变量 USER: ${USER:-未设置}" >> "$LOGFILE"
    echo "  - 环境变量 HOME: ${HOME:-未设置}" >> "$LOGFILE"
    echo "运行模式: 后台模式（多环境并行训练）" >> "$LOGFILE"
    echo "配置参数:" >> "$LOGFILE"
    echo "  - 并行环境数: $NUM_ENVS" >> "$LOGFILE"
    echo "  - 批次大小: $BATCH_SIZE" >> "$LOGFILE"
    echo "  - 每轮训练迭代次数: $TRAINING_ITERATIONS" >> "$LOGFILE"
    echo "  - 每Episode最大步数比例: $MAX_STEPS_RATIO (max_steps = target_distance * max_steps_ratio)" >> "$LOGFILE"
    echo "  - 每Episode最小步数: $MAX_STEPS_MIN (max_steps不会小于此值)" >> "$LOGFILE"
    echo "  - 模型保存频率: 每 $SAVE_EVERY 次训练" >> "$LOGFILE"
    echo "  - 重放缓冲区大小: $BUFFER_SIZE" >> "$LOGFILE"
    echo "  - 统计报告频率: 每 $REPORT_EVERY 个episode" >> "$LOGFILE"
    echo "  - 训练模式: 持续循环训练（真正的并行训练）" >> "$LOGFILE"
    echo "  - 模型更新模式: 基于前${AVG_LOSS_WINDOW_SIZE}次训练的平均损失阈值更新 (${CRITIC_LOSS_THRESHOLD})" >> "$LOGFILE"
    echo "环境参数:" >> "$LOGFILE"
    echo "  - 最大速度: $MAX_VELOCITY" >> "$LOGFILE"
    echo "  - 忽略角度: $NEGLECT_ANGLE 度" >> "$LOGFILE"
    echo "  - 最大偏航率: $MAX_YAWRATE 度/秒" >> "$LOGFILE"
    echo "  - 扫描范围: $SCAN_RANGE" >> "$LOGFILE"
    echo "  - 最大目标距离: $MAX_TARGET_DIST" >> "$LOGFILE"
    echo "  - 初始目标距离: $INIT_TARGET_DISTANCE" >> "$LOGFILE"
    echo "  - 目标距离增加量: $TARGET_DIST_INCREASE" >> "$LOGFILE"
    echo "  - 目标到达判断阈值: $TARGET_REACHED_DELTA" >> "$LOGFILE"
    echo "  - 碰撞判断阈值: $COLLISION_DELTA" >> "$LOGFILE"
    echo "  - 世界大小: $WORLD_SIZE 米" >> "$LOGFILE"
    echo "  - 障碍物最小距离: $OBS_MIN_DIST 米" >> "$LOGFILE"
    echo "  - 障碍物数量: $OBS_NUM" >> "$LOGFILE"
    echo "模型参数:" >> "$LOGFILE"
    echo "  - 动作维度: $ACTION_DIM" >> "$LOGFILE"
    echo "  - 最大动作值: $MAX_ACTION" >> "$LOGFILE"
    echo "  - 状态维度: $STATE_DIM" >> "$LOGFILE"
    echo "调试参数:" >> "$LOGFILE"
    echo "  - 调试模式: $IS_CODE_DEBUG" >> "$LOGFILE"
    echo "模型配置:" >> "$LOGFILE"
    echo "  - 模型保存目录: $MODEL_SAVE_DIR" >> "$LOGFILE"
    echo "  - 模型加载目录: $MODEL_LOAD_DIR" >> "$LOGFILE"
    echo "  - 加载已有模型: $LOAD_MODEL" >> "$LOGFILE"
    echo "  - 模型文件名: SAC_actor.pth, SAC_critic.pth, SAC_critic_target.pth" >> "$LOGFILE"
    echo "  - Actor更新频率: $ACTOR_UPDATE_FREQUENCY" >> "$LOGFILE"
    echo "  - Critic目标更新频率: $CRITIC_TARGET_UPDATE_FREQUENCY" >> "$LOGFILE"
    echo "  - 网络隐藏层维度: $HIDDEN_DIM" >> "$LOGFILE"
    echo "  - 网络隐藏层深度: $HIDDEN_DEPTH" >> "$LOGFILE"
    echo "  - 平均损失窗口大小: $AVG_LOSS_WINDOW_SIZE" >> "$LOGFILE"
}

# ===================== 启动多个Gazebo环境 =====================
start_gazebo_instances() {
    log_output "启动 $NUM_ENVS 个独立的Gazebo环境..."
    
    for i in $(seq 0 $((NUM_ENVS-1))); do
        # 为每个环境创建独立的启动脚本（无头模式）
        cat > "$TMP_DIR/start_gazebo_env_${i}.sh" << EOF
#!/bin/bash
# 环境 $i 的独立启动脚本

# 清除可能干扰的环境变量（确保脚本自包含）
unset DISPLAY ROS_DOMAIN_ID 2>/dev/null || true

# 设置环境变量
export ROS_DOMAIN_ID=$((i + 1))

# 设置ROS2环境
source /opt/ros/foxy/setup.bash
source $SCRIPT_DIR/install/setup.bash

# 加载 Gazebo 环境
[ -f "/usr/share/gazebo/setup.bash" ] && source /usr/share/gazebo/setup.bash 2>/dev/null
[ -f "/usr/share/gazebo-11/setup.bash" ] && [ -z "\$GAZEBO_RESOURCE_PATH" ] && \
    source /usr/share/gazebo-11/setup.bash 2>/dev/null

# 设置 Gazebo 路径
orig_path="\${GAZEBO_RESOURCE_PATH:-}"
share_dir=""
for path in /usr/share/gazebo-11 /usr/share/gazebo /opt/ros/foxy/share/gazebo_plugins; do
    [ -d "\$path" ] && { share_dir="\$path"; break; }
done

if [ -z "\$GAZEBO_MODEL_PATH" ]; then
    export GAZEBO_MODEL_PATH="$SCRIPT_DIR/src/turtlebot3_simulations/turtlebot3_gazebo/models"
else
    export GAZEBO_MODEL_PATH="\$GAZEBO_MODEL_PATH:$SCRIPT_DIR/src/turtlebot3_simulations/turtlebot3_gazebo/models"
fi

if [ -n "\$share_dir" ]; then
    [ -n "\$orig_path" ] && orig_path="\$orig_path:"
    export GAZEBO_RESOURCE_PATH="\${orig_path}\${share_dir}:\$GAZEBO_MODEL_PATH"
else
    export GAZEBO_RESOURCE_PATH="\${orig_path}\${orig_path:+:}\$GAZEBO_MODEL_PATH"
fi

export TURTLEBOT3_MODEL=waffle

# 强制无头模式环境变量
export GAZEBO_IP=127.0.0.1
export GAZEBO_MASTER_URI=http://127.0.0.1:$((11345 + i)) # 不同的gazebo环境绑定不同的端口
    export GAZEBO_GUI=0

# 启动无头 Gazebo
ros2 launch turtlebot3_gazebo ros2_drl_headless.launch.py
EOF
        
        chmod +x "$TMP_DIR/start_gazebo_env_${i}.sh"
        
        # 在后台启动Gazebo环境
        bash "$TMP_DIR/start_gazebo_env_${i}.sh" >> "$LOGFILE" 2>&1 &
        GAZEBO_PID=$!
        GAZEBO_PIDS+=($GAZEBO_PID)
        
            log_output "启动无头Gazebo环境 $i (ROS_DOMAIN_ID=$((i + 1)), PID: $GAZEBO_PID)"
        sleep 2
    done
    
    # 等待所有Gazebo环境初始化
    log_output "等待Gazebo环境初始化 ($GAZEBO_WAIT_TIME秒)..."
    for i in $(seq 1 $GAZEBO_WAIT_TIME); do
            echo -n "."
        sleep 1
    done
    echo ""
    
    # 检查ROS服务是否可用
    log_output "检查ROS服务状态..."
    for i in $(seq 0 $((NUM_ENVS-1))); do
        log_output "检查环境 $i (ROS_DOMAIN_ID=$((i + 1))) 的ROS服务..."
        
        # 创建临时检查脚本，确保在正确的ROS域中检查
        cat > "$TMP_DIR/check_services_env_${i}.sh" << EOF
#!/bin/bash
export ROS_DOMAIN_ID=$((i + 1))
source /opt/ros/foxy/setup.bash
source $SCRIPT_DIR/install/setup.bash

# 检查关键服务是否可用
if timeout 10 ros2 service list | grep -q "gazebo"; then
    echo "  - Gazebo服务正常"
else
    echo "  - Gazebo服务异常"
fi

if timeout 10 ros2 service list | grep -q "set_entity_state"; then
    echo "  - SetModelState服务正常"
else
    echo "  - SetModelState服务异常"
fi
EOF
        
        chmod +x "$TMP_DIR/check_services_env_${i}.sh"
            bash "$TMP_DIR/check_services_env_${i}.sh" >> "$LOGFILE" 2>&1
    done
}



# ===================== 启动多环境训练脚本 =====================
start_multi_env_training() {
    log_output "启动多环境并行训练脚本..."
    cd "$SCRIPT_DIR"
    
    # 设置训练脚本所需的环境变量（仅在当前函数作用域内）
    # 让 Python 脚本将权重等冗长信息只写入日志文件
    export TRAINING_LOGFILE="$LOGFILE"
    
    # 确保训练进程不受外部ROS环境变量影响
    # 训练进程不需要ROS环境，只处理数据
    unset ROS_DOMAIN_ID ROS_PACKAGE_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH 2>/dev/null || true
    
    # 启动训练脚本
    unbuffer python3 -u src/drl_navigation_ros2/multi_env_train.py \
        --num_envs $NUM_ENVS \
        --batch_size $BATCH_SIZE \
        --training_iterations $TRAINING_ITERATIONS \
        --max_steps_ratio $MAX_STEPS_RATIO \
        --max_steps_min $MAX_STEPS_MIN \
        --save_every $SAVE_EVERY \
        --buffer_size $BUFFER_SIZE \
        --report_every $REPORT_EVERY \
        --stats_window_size $STATS_WINDOW_SIZE \
        --gpu_id ${TORCH_GPU_ID:-$GPU_ID} \
        --max_training_count $MAX_TRAINING_COUNT \
        --critic_loss_threshold $CRITIC_LOSS_THRESHOLD \
        --actor_update_frequency $ACTOR_UPDATE_FREQUENCY \
        --critic_target_update_frequency $CRITIC_TARGET_UPDATE_FREQUENCY \
        --hidden_dim $HIDDEN_DIM \
        --hidden_depth $HIDDEN_DEPTH \
        --avg_loss_window_size $AVG_LOSS_WINDOW_SIZE \
        --max_velocity $MAX_VELOCITY \
        --neglect_angle $NEGLECT_ANGLE \
        --max_yawrate $MAX_YAWRATE \
        --scan_range $SCAN_RANGE \
        --max_target_dist $MAX_TARGET_DIST \
        --init_target_distance $INIT_TARGET_DISTANCE \
        --target_dist_increase $TARGET_DIST_INCREASE \
        --target_reached_delta $TARGET_REACHED_DELTA \
        --collision_delta $COLLISION_DELTA \
        --world_size $WORLD_SIZE \
        --obs_min_dist $OBS_MIN_DIST \
        --obs_num $OBS_NUM \
        --action_dim $ACTION_DIM \
        --max_action $MAX_ACTION \
        --state_dim $STATE_DIM \
        --is_code_debug $IS_CODE_DEBUG \
        --model_save_dir "$MODEL_SAVE_DIR" \
        --model_load_dir "$MODEL_LOAD_DIR" \
        --load_model $LOAD_MODEL \
        >> "$LOGFILE" 2>&1 &
    
    TRAINING_PID=$!
    log_output "训练脚本已启动 (PID: $TRAINING_PID)"
    
    # 等待训练完成
    wait $TRAINING_PID
    TRAINING_EXIT_CODE=$?
    
    log_output "训练脚本已结束 (退出码: $TRAINING_EXIT_CODE)"
}

# ===================== 清理函数 =====================
cleanup() {
    log_output "执行正常清理..."
    echo "===== 多环境并行训练结束 - $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOGFILE"
    
    # 调用 clean.sh 进行清理
    local clean_script="$SCRIPT_DIR/clean.sh"
    if [ -f "$clean_script" ] && [ -x "$clean_script" ]; then
        log_output "调用 clean.sh 进行清理..."
        bash "$clean_script" >> "$LOGFILE" 2>&1 || true
    else
        log_output "警告: clean.sh 不存在，使用内置清理逻辑"
        # 停止训练进程
    if [ ! -z "$TRAINING_PID" ] && kill -0 "$TRAINING_PID" 2>/dev/null; then
        kill -TERM "$TRAINING_PID" 2>/dev/null
            sleep 2
            kill -0 "$TRAINING_PID" 2>/dev/null && kill -KILL "$TRAINING_PID" 2>/dev/null
    fi
    # 停止所有Gazebo进程
    for pid in "${GAZEBO_PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null
    done
    sleep 2
        pkill -f "ros2 launch turtlebot3_gazebo|gazebo|multi_env_train.py" 2>/dev/null
        rm -f "$TMP_DIR/start_gazebo_env_"*.sh "$TMP_DIR/check_services_env_"*.sh "$PID_FILE" 2>/dev/null
    fi
}

# ===================== 主执行流程 =====================
main() {
    # 开启 bash 调试打印
    if [ "$DEBUG_MODE" = true ]; then
        set -x
        log_output "DEBUG MODE: 打印执行的所有指令"
    fi

    # 确保工作目录正确（不依赖外部环境）
    cd "$SCRIPT_DIR" || {
        echo "错误: 无法切换到脚本目录 $SCRIPT_DIR" >&2
        exit 1
    }
    
    # 先创建日志目录和文件（但不写入内容，避免清空 GPU 选择的日志）
    mkdir -p "$LOG_DIR"
    > "$LOGFILE"
    echo "===== 多环境并行训练启动 - $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOGFILE"
    
    # 设置 CUDA_VISIBLE_DEVICES（在脚本作用域内，不影响外部终端）
    # 这会在日志中记录 GPU 选择过程
    setup_cuda
    
    # 初始化日志（记录 GPU 选择结果）
    init_logging
    
    # 启动Gazebo环境（每个环境在独立的子进程中设置自己的环境变量）
    start_gazebo_instances
    
    # 启动训练脚本（在脚本作用域内设置所需环境变量）
    start_multi_env_training
    
    # 训练正常结束后执行清理
    log_output "训练正常结束，执行清理..."
    cleanup
}

# ===================== 后台运行处理 =====================
if [ "$DAEMON_MODE" = true ]; then
    # 后台模式：使用nohup和重定向
    echo "启动后台训练进程..."
    echo "日志文件：$LOGFILE"
    echo "PID文件：$PID_FILE"
    
    # 创建PID文件
    echo $$ > "$PID_FILE"
    
    # 使用nohup在后台运行，重定向所有输出到日志文件
    # 导出必要的变量和函数到子shell
    export LOGFILE
    export NUM_ENVS MAX_STEPS_RATIO MAX_STEPS_MIN BATCH_SIZE TRAINING_ITERATIONS
    export SAVE_EVERY BUFFER_SIZE REPORT_EVERY STATS_WINDOW_SIZE
    export MAX_TRAINING_COUNT CRITIC_LOSS_THRESHOLD ACTOR_UPDATE_FREQUENCY
    export CRITIC_TARGET_UPDATE_FREQUENCY HIDDEN_DIM HIDDEN_DEPTH AVG_LOSS_WINDOW_SIZE
    export GPU_ID TORCH_GPU_ID MAX_VELOCITY
    export NEGLECT_ANGLE MAX_YAWRATE SCAN_RANGE MAX_TARGET_DIST INIT_TARGET_DISTANCE
    export TARGET_DIST_INCREASE TARGET_REACHED_DELTA COLLISION_DELTA WORLD_SIZE
    export OBS_MIN_DIST OBS_NUM ACTION_DIM MAX_ACTION STATE_DIM IS_CODE_DEBUG
    export MODEL_SAVE_DIR MODEL_LOAD_DIR LOAD_MODEL LOG_DIR
    export GAZEBO_WAIT_TIME TRAINING_TIMESTAMP
    export SCRIPT_DIR TIMESTAMP LOG_DIR_BASE MODEL_SAVE_DIR_TS DAEMON_MODE
    export TMP_DIR PID_FILE CONFIG_FILE
    # 外层直接重定向到日志，避免 nohup 提示输出到 nohup.out
    nohup bash -c "
        $(declare -f log_output)
        $(declare -f select_best_gpu)
        $(declare -f init_logging)
        $(declare -f setup_cuda)
        $(declare -f start_gazebo_instances)
        $(declare -f start_multi_env_training)
        $(declare -f cleanup)
        $(declare -f cleanup_all)
        $(declare -f main)
        
        # 后台进程设置自己的信号处理（只处理中断信号，不处理EXIT）
        trap cleanup_all SIGINT SIGTERM
        
        # 重新定义输出重定向
        exec 1>> \"\$LOGFILE\" 2>&1
        echo \"===== 后台训练进程启动 - \$(date '+%Y-%m-%d %H:%M:%S') =====\"
        main
    " >> "$LOGFILE" 2>&1 &
    
    # 获取后台进程PID
    DAEMON_PID=$!
    echo $DAEMON_PID > "$PID_FILE"
    
    echo "后台训练进程已启动，PID: $DAEMON_PID"
    echo "使用以下命令查看状态："
    echo "  tail -f $LOGFILE"
    echo "  ps aux | grep $DAEMON_PID"
    
    # 等待一下确保进程启动
    sleep 2
    
    # 检查进程是否还在运行
    if kill -0 $DAEMON_PID 2>/dev/null; then
        echo "后台训练进程运行正常"
        # 取消 EXIT trap，避免正常退出时触发清理（后台进程会自己处理清理）
        trap - EXIT
        exit 0
    else
        echo "后台训练进程启动失败"
        # 启动失败时保留清理功能
        exit 1
    fi
else
    # 前台模式：直接运行
    main
fi
