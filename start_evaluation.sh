#!/bin/bash
# 多环境并行评估启动脚本

# ===================== 运行模式说明 =====================
# 默认使用后台模式（run_mode=2），日志输出到文件，可通过 tail -f 实时查看
# 使用方法：
# 直接运行：./start_evaluation.sh（默认后台模式，日志输出到文件）
# 可视化模式：在 evaluation.yaml 中设置 run_mode: 3（仅第1个环境可视化）
# Debug模式：./start_evaluation.sh --debug（显示所有执行的指令）
# 查看日志：tail -f log/evaluation/eval_*/eval_*.log
# 停止评估：./stop_evaluation.sh

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

# ===================== 配置文件路径与解析 =====================
CONFIG_FILE="config/evaluation.yaml"

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
EVAL_PID=""
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
    # 停止评估进程
    if [ ! -z "$EVAL_PID" ] && kill -0 "$EVAL_PID" 2>/dev/null; then
        kill -TERM "$EVAL_PID" 2>/dev/null
        sleep 2
            kill -0 "$EVAL_PID" 2>/dev/null && kill -KILL "$EVAL_PID" 2>/dev/null
        fi
    # 停止所有Gazebo进程
    for pid in "${GAZEBO_PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null
    done
    sleep 3
        pkill -f "ros2 launch turtlebot3_gazebo|gazebo|evaluate.py|gzclient" 2>/dev/null
    # 清理临时文件
        [ -d "$TMP_SCRIPT_DIR" ] && rm -rf "$TMP_SCRIPT_DIR" 2>/dev/null || true
        rm -f "$TMP_DIR/start_gazebo_eval_env_"*.sh "$TMP_DIR/check_services_eval_env_"*.sh "$PID_FILE" 2>/dev/null
    fi
    
    echo "所有进程已清理完成"
    exit 0
}

# 注册信号处理（包含 EXIT，确保异常或正常退出都清理）
trap cleanup_all SIGINT SIGTERM EXIT

# ===================== 清理共享内存 =====================
echo "清理ROS2共享内存段..."
sudo rm -f /dev/shm/fastrtps_*
echo "共享内存清理完成"
echo

# ===================== 从YAML读取配置（为缺省做兜底） =====================
NUM_ENVS=$(parse_yaml_value "num_envs")
TOTAL_EPISODES=$(parse_yaml_value "total_episodes")
MAX_TARGET_DIST=$(parse_yaml_value "max_target_dist")
DISTANCE_RANGES=$(parse_yaml_value "distance_ranges")
GPU_ID=$(parse_yaml_value "gpu_id")
GAZEBO_WAIT_TIME=$(parse_yaml_value "gazebo_wait_time")
RUN_MODE=$(parse_yaml_value "run_mode")

MAX_VELOCITY=$(parse_yaml_value "max_velocity")
NEGLECT_ANGLE=$(parse_yaml_value "neglect_angle")
MAX_YAWRATE=$(parse_yaml_value "max_yawrate")
SCAN_RANGE=$(parse_yaml_value "scan_range")
INIT_TARGET_DISTANCE=$(parse_yaml_value "init_target_distance")
TARGET_DIST_INCREASE=$(parse_yaml_value "target_dist_increase")
TARGET_REACHED_DELTA=$(parse_yaml_value "target_reached_delta")
COLLISION_DELTA=$(parse_yaml_value "collision_delta")
WORLD_SIZE=$(parse_yaml_value "world_size")
OBS_MIN_DIST=$(parse_yaml_value "obs_min_dist")
# obs_num 不应该从 evaluation.yaml 读取，应该从模型目录的 config_used.yaml 读取
# 如果 evaluation.yaml 中明确指定了 obs_num，则使用该值；否则不设置，让Python脚本从训练配置读取
OBS_NUM=$(parse_yaml_value "obs_num")
# 如果 OBS_NUM 为空，则不设置默认值，让Python脚本从训练配置读取

ACTION_DIM=$(parse_yaml_value "action_dim")
MAX_ACTION=$(parse_yaml_value "max_action")
STATE_DIM=$(parse_yaml_value "state_dim")
HIDDEN_DIM=$(parse_yaml_value "hidden_dim")
# HIDDEN_DEPTH 不再从evaluation.yaml读取，由Python脚本从模型目录的config_used.yaml读取
# HIDDEN_DEPTH=$(parse_yaml_value "hidden_depth")

MODEL_LOAD_DIR=$(parse_yaml_value "model_load_dir")
# eval_log_dir 不再需要，日志统一保存到 log/evaluation 目录

# 兜底默认值（若YAML缺项）
NUM_ENVS=${NUM_ENVS:-32}
TOTAL_EPISODES=${TOTAL_EPISODES:-3000}
MAX_TARGET_DIST=${MAX_TARGET_DIST:-12.0}
DISTANCE_RANGES=${DISTANCE_RANGES:-"0-3,3-6,6-9,9-12"}
GPU_ID=${GPU_ID:-0}
GAZEBO_WAIT_TIME=${GAZEBO_WAIT_TIME:-20}
RUN_MODE=${RUN_MODE:-2}  # 默认使用后台模式

MAX_VELOCITY=${MAX_VELOCITY:-1.0}
NEGLECT_ANGLE=${NEGLECT_ANGLE:-0}
MAX_YAWRATE=${MAX_YAWRATE:-20.0}
SCAN_RANGE=${SCAN_RANGE:-5}
INIT_TARGET_DISTANCE=${INIT_TARGET_DISTANCE:-2.0}
TARGET_DIST_INCREASE=${TARGET_DIST_INCREASE:-0.01}
TARGET_REACHED_DELTA=${TARGET_REACHED_DELTA:-0.3}
COLLISION_DELTA=${COLLISION_DELTA:-0.25}
WORLD_SIZE=${WORLD_SIZE:-15}
OBS_MIN_DIST=${OBS_MIN_DIST:-2}
# OBS_NUM 不再设置默认值，如果 evaluation.yaml 中没有指定，则不传递该参数，让Python脚本从训练配置读取
# 注意：这里不设置 OBS_NUM 的默认值，如果 evaluation.yaml 中没有指定，则 OBS_NUM 为空
# Python脚本会从模型目录的 config_used.yaml 中读取正确的 obs_num 值

ACTION_DIM=${ACTION_DIM:-2}
MAX_ACTION=${MAX_ACTION:-1.0}
STATE_DIM=${STATE_DIM:-25}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
# HIDDEN_DEPTH 不再设置默认值，由Python脚本从训练配置读取
# HIDDEN_DEPTH=${HIDDEN_DEPTH:-2}

MODEL_LOAD_DIR=${MODEL_LOAD_DIR:-"/home/zc/DRL-Robot-Navigation-ROS2/models/single_env_SAC_15m"}
# eval_log_dir 不再需要，日志统一保存到 log/evaluation 目录

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 使用脚本目录下的临时目录，避免 /tmp 的权限问题
TMP_DIR="${SCRIPT_DIR}/tmp"
TMP_SCRIPT_DIR="${TMP_DIR}/eval_scripts"
PID_FILE="$TMP_DIR/evaluation.pid"
mkdir -p "$TMP_SCRIPT_DIR"
mkdir -p "$TMP_DIR"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_DIR_BASE="$SCRIPT_DIR/log/evaluation"
LOG_DIR="$LOG_DIR_BASE/eval_${TIMESTAMP}"
LOGFILE="$LOG_DIR/eval_${TIMESTAMP}.log"

# 预先创建日志目录（在首次 tee 之前）
mkdir -p "$LOG_DIR"

# ===================== 输出函数（根据模式选择输出方式）=====================
# 模式1：输出到终端和日志；模式2：只输出到日志
log_output() {
    if [ "$DAEMON_MODE" = true ]; then
        # 模式2：只输出到日志（后台模式下标准输出已重定向，直接echo即可）
        echo "$@"
    else
        # 模式1：输出到终端和日志
        echo "$@" | tee -a "$LOGFILE"
    fi
}

# 规范化 run_mode（防止解析异常）并输出
# 运行模式：2=后台（不向终端输出，终端关闭评估不停止）；3=可视化（仅第1个环境可视化）
RUN_MODE=$(echo "$RUN_MODE" | tr -d '[:space:]')
case "$RUN_MODE" in
    2|3) ;;  # 合法值：2=后台模式，3=可视化模式
    *) RUN_MODE=2 ;;  # 默认使用后台模式
esac
log_output "run_mode: $RUN_MODE"

# 后台模式：run_mode=2 时启用后台模式
if [ "$RUN_MODE" -eq 2 ]; then
    DAEMON_MODE=true
fi

# ===================== Debug模式处理 =====================
# 如果启用debug模式，在脚本早期就开启 set -x（这样可以看到所有命令）
if [ "$DEBUG_MODE" = true ]; then
    set -x
    echo "DEBUG MODE: 已启用，将显示所有执行的指令"
fi

# ===================== 初始化日志 =====================
init_logging() {
    # 创建日志文件夹（如果不存在）
    mkdir -p "$LOG_DIR"
    > "$LOGFILE"
    echo "===== 多环境并行评估启动 - $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOGFILE"
    echo "运行模式: $RUN_MODE（2=后台，3=仅可视化首环境）" >> "$LOGFILE"
    echo "评估相关参数（从evaluation.yaml读取）:" >> "$LOGFILE"
    echo "  - 并行环境数: $NUM_ENVS" >> "$LOGFILE"
    echo "  - 总episode数: $TOTAL_EPISODES" >> "$LOGFILE"
    echo "  - 固定max_target_dist: ${MAX_TARGET_DIST}m" >> "$LOGFILE"
    echo "  - 距离区间: ${DISTANCE_RANGES}m" >> "$LOGFILE"
    echo "  - GPU ID: $GPU_ID" >> "$LOGFILE"
    echo "模型配置:" >> "$LOGFILE"
    echo "  - 模型加载目录: $MODEL_LOAD_DIR" >> "$LOGFILE"
    echo "  - 评估日志目录: $LOG_DIR（统一保存到log/evaluation目录）" >> "$LOGFILE"
    echo "" >> "$LOGFILE"
    echo "注意: 环境参数、模型参数等将从模型目录的 config_used.yaml 读取，详见后续Python脚本输出" >> "$LOGFILE"
}

# ===================== 启动多个Gazebo环境 =====================
start_gazebo_instances() {
    # 确保 TMP_SCRIPT_DIR 已设置（在后台模式下可能需要重新设置）
    if [ -z "${TMP_SCRIPT_DIR:-}" ]; then
        # 如果 SCRIPT_DIR 未设置，尝试从当前脚本位置获取
        if [ -z "${SCRIPT_DIR:-}" ]; then
            SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        fi
        TMP_DIR="${SCRIPT_DIR}/tmp"
        TMP_SCRIPT_DIR="${TMP_DIR}/eval_scripts"
        mkdir -p "$TMP_SCRIPT_DIR"
        mkdir -p "$TMP_DIR"
    fi
    
    log_output "启动 $NUM_ENVS 个独立的Gazebo环境..."
    
    for i in $(seq 0 $((NUM_ENVS-1))); do
        # 为每个环境创建独立的启动脚本
        cat > "${TMP_SCRIPT_DIR}/start_gazebo_eval_env_${i}.sh" << EOF
#!/bin/bash
# 评估环境 $i 的独立启动脚本

# 设置环境变量
export ROS_DOMAIN_ID=$((i + 1))
if [ "$RUN_MODE" -eq 3 ] && [ "$i" -eq 0 ]; then
    # 仅首环境可视化，直接继承当前终端的 DISPLAY；这里不修改 DISPLAY
    # 为提升稳定性，强制使用软件渲染，避免显卡/GL 相关崩溃
    export LIBGL_ALWAYS_SOFTWARE=1
    export MESA_GL_VERSION_OVERRIDE=3.3
fi

# 设置ROS2环境
source /opt/ros/foxy/setup.bash
source /home/zc/DRL-Robot-Navigation-ROS2/install/setup.bash

# 设置Gazebo无头模式环境变量
export GAZEBO_MODEL_PATH=\$GAZEBO_MODEL_PATH:/home/zc/DRL-Robot-Navigation-ROS2/src/turtlebot3_simulations/turtlebot3_gazebo/models
export TURTLEBOT3_MODEL=waffle

# 强制无头模式环境变量
export GAZEBO_IP=127.0.0.1
export GAZEBO_MASTER_URI=http://127.0.0.1:$((11345 + i)) # 不同的gazebo环境绑定不同的端口
export GAZEBO_RESOURCE_PATH=\$GAZEBO_MODEL_PATH
# 对后台环境显式禁用 GUI，防止 gzclient 启动
if [ ! \( "$RUN_MODE" -eq 3 -a "$i" -eq 0 \) ]; then
    export GAZEBO_GUI=0
fi

# 选择正确的启动文件
HEADLESS_LAUNCH="ros2_drl_headless.launch.py"
VISUAL_LAUNCH="ros2_drl.launch.py"

if [ "$RUN_MODE" -eq 3 ] && [ "$i" -eq 0 ]; then
    # 第一个环境可视化
    ros2 launch turtlebot3_gazebo \$VISUAL_LAUNCH
else
    # 其他环境后台运行（无GUI）
    ros2 launch turtlebot3_gazebo \$HEADLESS_LAUNCH
fi
EOF
        
        # 添加执行权限
        chmod +x "${TMP_SCRIPT_DIR}/start_gazebo_eval_env_${i}.sh"
        
        # 在后台启动Gazebo环境（使用bash执行，不需要执行权限）
        if [ "$DAEMON_MODE" = true ]; then
            bash "${TMP_SCRIPT_DIR}/start_gazebo_eval_env_${i}.sh" >> "$LOGFILE" 2>&1 &
        else
            bash "${TMP_SCRIPT_DIR}/start_gazebo_eval_env_${i}.sh" >> "$LOG_DIR/eval_gazebo_${i}.log" 2>&1 &
        fi
        GAZEBO_PID=$!
        GAZEBO_PIDS+=($GAZEBO_PID)
        
        if [ "$RUN_MODE" -eq 3 ] && [ "$i" -eq 0 ]; then
            log_output "启动可视化Gazebo环境 $i (ROS_DOMAIN_ID=$((i + 1)), DISPLAY=${DISPLAY:-<未设置>}, PID: $GAZEBO_PID)"
        else
            log_output "启动后台Gazebo环境 $i (ROS_DOMAIN_ID=$((i + 1)), PID: $GAZEBO_PID)"
        fi
        sleep 2
    done
    
    # 等待所有Gazebo环境初始化
    log_output "等待Gazebo环境初始化 ($GAZEBO_WAIT_TIME秒)..."
    for i in $(seq 1 $GAZEBO_WAIT_TIME); do
        if [ "$DAEMON_MODE" = true ]; then
            echo -n "." >> "$LOGFILE"
        else
            echo -n "."
            echo -n "." >> "$LOGFILE"
        fi
        sleep 1
    done
    if [ "$DAEMON_MODE" = true ]; then
        echo "" >> "$LOGFILE"
    fi
    
    # 检查ROS服务是否可用
    # 确保 TMP_SCRIPT_DIR 已设置（在后台模式下可能需要重新设置）
    if [ -z "${TMP_SCRIPT_DIR:-}" ]; then
        # 如果 SCRIPT_DIR 未设置，尝试从当前脚本位置获取
        if [ -z "${SCRIPT_DIR:-}" ]; then
            SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        fi
        TMP_DIR="${SCRIPT_DIR}/tmp"
        TMP_SCRIPT_DIR="${TMP_DIR}/eval_scripts"
        mkdir -p "$TMP_SCRIPT_DIR"
        mkdir -p "$TMP_DIR"
    fi
    
    log_output "检查ROS服务状态..."
    for i in $(seq 0 $((NUM_ENVS-1))); do
        log_output "检查环境 $i (ROS_DOMAIN_ID=$((i + 1))) 的ROS服务..."
        
        # 创建临时检查脚本，确保在正确的ROS域中检查
        cat > "${TMP_SCRIPT_DIR}/check_services_eval_env_${i}.sh" << EOF
#!/bin/bash
export ROS_DOMAIN_ID=$((i + 1))
source /opt/ros/foxy/setup.bash
source /home/zc/DRL-Robot-Navigation-ROS2/install/setup.bash

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
        
        # 添加执行权限
        chmod +x "${TMP_SCRIPT_DIR}/check_services_eval_env_${i}.sh"
        if [ "$DAEMON_MODE" = true ]; then
            bash "${TMP_SCRIPT_DIR}/check_services_eval_env_${i}.sh" >> "$LOGFILE" 2>&1
        else
            bash "${TMP_SCRIPT_DIR}/check_services_eval_env_${i}.sh" | tee -a "$LOGFILE"
        fi
    done
}

# ===================== 启动评估脚本 =====================
start_evaluation() {
    log_output "启动多环境并行评估脚本..."
    cd /home/zc/DRL-Robot-Navigation-ROS2
    
    # 检查模型文件是否存在
    if [ ! -f "${MODEL_LOAD_DIR}/SAC_actor.pth" ]; then
        log_output "错误：未找到模型文件: ${MODEL_LOAD_DIR}/SAC_actor.pth"
        exit 1
    fi
    
    # 启动评估脚本
    # 注意：obs_num 不应该从 evaluation.yaml 读取，应该从模型目录的 config_used.yaml 读取
    # 如果 evaluation.yaml 中没有明确指定 obs_num，则不传递该参数，让Python脚本从训练配置读取
    EVAL_CMD="unbuffer python3 -u src/drl_navigation_ros2/evaluate.py \
        --num_envs $NUM_ENVS \
        --total_episodes $TOTAL_EPISODES \
        --max_target_dist $MAX_TARGET_DIST \
        --distance_ranges $DISTANCE_RANGES \
        --gpu_id $GPU_ID \
        --max_velocity $MAX_VELOCITY \
        --neglect_angle $NEGLECT_ANGLE \
        --max_yawrate $MAX_YAWRATE \
        --scan_range $SCAN_RANGE \
        --init_target_distance $INIT_TARGET_DISTANCE \
        --target_dist_increase $TARGET_DIST_INCREASE \
        --target_reached_delta $TARGET_REACHED_DELTA \
        --collision_delta $COLLISION_DELTA \
        --world_size $WORLD_SIZE \
        --obs_min_dist $OBS_MIN_DIST"
    
    # 只有在 evaluation.yaml 中明确指定了 obs_num 时才传递该参数
    # 否则让Python脚本从训练配置中读取（从模型目录的 config_used.yaml）
    if [ -n "$OBS_NUM" ]; then
        EVAL_CMD="$EVAL_CMD --obs_num $OBS_NUM"
    fi
    
    EVAL_CMD="$EVAL_CMD --state_dim $STATE_DIM \
        --action_dim $ACTION_DIM \
        --max_action $MAX_ACTION \
        --model_load_dir \"$MODEL_LOAD_DIR\" \
        --hidden_dim $HIDDEN_DIM"
    
    eval $EVAL_CMD >> "$LOGFILE" 2>&1 &
    
    EVAL_PID=$!
    log_output "评估脚本已启动 (PID: $EVAL_PID)"
    
    # 等待评估完成
    wait $EVAL_PID
    EVAL_EXIT_CODE=$?
    
    log_output "评估脚本已结束 (退出码: $EVAL_EXIT_CODE)"
}

# ===================== 清理函数 =====================
cleanup() {
    log_output "执行正常清理..."
    echo "===== 多环境并行评估结束 - $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOGFILE"
    
    # 调用 clean.sh 进行清理
    local clean_script="$SCRIPT_DIR/clean.sh"
    if [ -f "$clean_script" ] && [ -x "$clean_script" ]; then
        log_output "调用 clean.sh 进行清理..."
        bash "$clean_script" >> "$LOGFILE" 2>&1 || true
    else
        log_output "警告: clean.sh 不存在，使用内置清理逻辑"
        # 停止评估进程
    if [ ! -z "$EVAL_PID" ] && kill -0 "$EVAL_PID" 2>/dev/null; then
        kill -TERM "$EVAL_PID" 2>/dev/null
    fi
    # 停止所有Gazebo进程
    for pid in "${GAZEBO_PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null
    done
    sleep 2
        pkill -f "ros2 launch turtlebot3_gazebo|gazebo|gzclient|evaluate.py" 2>/dev/null
    # 清理临时文件
    [ -d "$TMP_SCRIPT_DIR" ] && rm -rf "$TMP_SCRIPT_DIR" 2>/dev/null || true
        rm -f "$PID_FILE" 2>/dev/null
    fi
}

# ===================== 主执行流程 =====================
main() {
    # 开启 bash 调试打印
    if [ "$DEBUG_MODE" = true ]; then
        set -x
        log_output "DEBUG MODE: 打印执行的所有指令"
    fi

    init_logging
    
    log_output "======================================"
    log_output "评估启动"
    log_output "======================================"
    log_output "日志文件: $LOGFILE"
    log_output "并行环境数: ${NUM_ENVS}"
    log_output "总episode数: ${TOTAL_EPISODES}"
    log_output "固定max_target_dist: ${MAX_TARGET_DIST}m"
    log_output "距离区间: ${DISTANCE_RANGES}m"
    log_output "模型加载路径: ${MODEL_LOAD_DIR}"
    log_output "GPU ID: ${GPU_ID}"
    log_output "运行模式: ${RUN_MODE}（2=后台，3=仅可视化首环境）"
    log_output "======================================"
    echo ""
    
    start_gazebo_instances
    
    log_output "开始评估..."
    echo ""
    
    start_evaluation
    
    # 评估正常结束后执行清理
    log_output "评估正常结束，执行清理..."
    cleanup
}

# ===================== 后台运行处理 =====================
if [ "$DAEMON_MODE" = true ]; then
    # 后台模式：使用nohup和重定向
    echo "启动后台评估进程..."
    echo "日志文件：$LOGFILE"
    echo "PID文件：$PID_FILE"
    
    # 创建PID文件
    echo $$ > "$PID_FILE"
    
    # 使用nohup在后台运行，重定向所有输出到日志文件
    # 导出必要的变量和函数到子shell
    export LOGFILE
    export NUM_ENVS TOTAL_EPISODES MAX_TARGET_DIST DISTANCE_RANGES GPU_ID
    export GAZEBO_WAIT_TIME RUN_MODE MAX_VELOCITY NEGLECT_ANGLE MAX_YAWRATE
    export SCAN_RANGE INIT_TARGET_DISTANCE TARGET_DIST_INCREASE
    export TARGET_REACHED_DELTA COLLISION_DELTA WORLD_SIZE OBS_MIN_DIST OBS_NUM
    export ACTION_DIM MAX_ACTION STATE_DIM HIDDEN_DIM
    export MODEL_LOAD_DIR SCRIPT_DIR TIMESTAMP
    export LOG_DIR_BASE LOG_DIR DAEMON_MODE
    export TMP_DIR TMP_SCRIPT_DIR PID_FILE  # 导出临时目录和PID文件路径，供后台进程使用
    export DEBUG_MODE  # 导出DEBUG_MODE，供后台进程使用
    # 使用nohup在后台运行，明确指定输出重定向到日志文件，避免输出到nohup.out
    nohup bash -c "
        $(declare -f log_output)
        $(declare -f init_logging)
        $(declare -f start_gazebo_instances)
        $(declare -f start_evaluation)
        $(declare -f cleanup)
        $(declare -f cleanup_all)
        $(declare -f main)
        
        # 后台进程设置自己的信号处理（只处理中断信号，不处理EXIT）
        trap cleanup_all SIGINT SIGTERM
        
        # 如果启用debug模式，在后台子shell中也启用 set -x
        if [ \"\$DEBUG_MODE\" = true ]; then
            set -x
        fi
        
        # 重新定义输出重定向
        exec 1>> \"\$LOGFILE\" 2>&1
        echo \"===== 后台评估进程启动 - \$(date '+%Y-%m-%d %H:%M:%S') =====\"
        if [ \"\$DEBUG_MODE\" = true ]; then
            echo \"DEBUG MODE: 后台进程已启用调试模式，将显示所有执行的指令\"
        fi
        main
    " >> "$LOGFILE" 2>&1 &
    
    # 获取后台进程PID
    DAEMON_PID=$!
    echo $DAEMON_PID > "$PID_FILE"
    
    echo "后台评估进程已启动，PID: $DAEMON_PID"
    echo "使用以下命令查看状态："
    echo "  tail -f $LOGFILE"
    echo "  ps aux | grep $DAEMON_PID"
    echo "  ./stop_evaluation.sh"
    
    # 等待一下确保进程启动
    sleep 2
    
    # 检查进程是否还在运行
    if kill -0 $DAEMON_PID 2>/dev/null; then
        echo "后台评估进程运行正常"
        # 取消 EXIT trap，避免正常退出时触发清理（后台进程会自己处理清理）
        trap - EXIT
        exit 0
    else
        echo "后台评估进程启动失败"
        # 启动失败时保留清理功能
        exit 1
    fi
else
    # 前台模式：直接运行
    # 初始化日志输出重定向
    > "$LOGFILE"
    exec 1> >(tee -a "$LOGFILE")
    exec 2> >(tee -a "$LOGFILE" >&2)
    main
fi
