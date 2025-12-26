#!/bin/bash
# start_training.sh - 增强版一键启动训练脚本
#
# 重要说明：
#   1. 此脚本必须在子shell中运行（使用 ./start_training.sh 或 bash start_training.sh）
#   2. 脚本中的所有环境变量操作都是临时的，只影响当前脚本执行环境
#   3. 脚本执行完毕后，所有环境变量修改不会影响调用它的shell或新开的终端
#   4. 请勿使用 source 或 . 命令执行此脚本，否则环境变量可能污染当前shell

# 检查脚本是否被source执行
[ "${BASH_SOURCE[0]}" != "${0}" ] && {
    echo "警告: 检测到脚本被 source 执行，环境变量修改可能会影响当前shell" >&2
    echo "建议: 请使用 ./start_training.sh 或 bash start_training.sh 直接执行" >&2
    echo "继续执行..." >&2
}

# 解析命令行参数
DEBUG_MODE=false
CLI_RUN_MODE=""
for arg in "$@"; do
    case "$arg" in
        --debug) DEBUG_MODE=true ;;
        --run_mode=*|--run-mode=*) CLI_RUN_MODE="${arg#*=}" ;;
    esac
done

# ===================== 获取脚本所在目录 =====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===================== 环境变量初始化 =====================
# 清除可能干扰的外部环境变量，确保脚本自包含，不依赖外部环境
# 注意：这些unset操作不会影响外部终端，因为脚本在子shell中运行
unset DISPLAY ROS_DOMAIN_ID CUDA_VISIBLE_DEVICES GAZEBO_IP GAZEBO_MASTER_URI GAZEBO_GUI 2>/dev/null || true
unset GAZEBO_MODEL_PATH GAZEBO_RESOURCE_PATH TURTLEBOT3_MODEL 2>/dev/null || true
unset ROS_PACKAGE_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH 2>/dev/null || true
unset QT_X11_NO_MITSHM QT_SESSION_MANAGER XDG_RUNTIME_DIR 2>/dev/null || true

# ===================== 初始化 ROS2 环境 =====================
setup_ros2_env() {
    # 加载 ROS2 环境（在脚本作用域内，不影响外部终端）
    local ros2_setup=""
    [ -f "/opt/ros/foxy/setup.bash" ] && ros2_setup="/opt/ros/foxy/setup.bash"
    [ -z "$ros2_setup" ] && [ -f "/opt/ros/humble/setup.bash" ] && ros2_setup="/opt/ros/humble/setup.bash"
    [ -z "$ros2_setup" ] && ros2_setup=$(ls -d /opt/ros/*/setup.bash 2>/dev/null | head -1)
    
    if [ -n "$ros2_setup" ] && [ -f "$ros2_setup" ]; then
        echo "正在加载 ROS2 环境: $ros2_setup"
        source "$ros2_setup"
        echo "ROS2 环境已加载 (ROS_DISTRO=${ROS_DISTRO:-未设置})"
    else
        echo "警告: 未找到 ROS2 setup.bash 文件" >&2
    fi
    
    # 加载工作空间环境
    local workspace_setup="$SCRIPT_DIR/install/setup.bash"
    [ -f "$workspace_setup" ] && {
        source "$workspace_setup"
        echo "工作空间环境已加载: $workspace_setup"
    } || echo "警告: 未找到工作空间 setup.bash 文件: $workspace_setup" >&2
    
    # 加载 Gazebo 环境
    [ -f "/usr/share/gazebo/setup.bash" ] && source /usr/share/gazebo/setup.bash 2>/dev/null && echo "Gazebo 环境已加载"
    [ -f "/usr/share/gazebo-11/setup.bash" ] && \
        source /usr/share/gazebo-11/setup.bash 2>/dev/null && echo "Gazebo 环境已加载"
}

setup_ros2_env

# ===================== DISPLAY 设置 =====================
setup_display() {
    local candidates=(":77.0" ":0.0" ":0" ":1.0" ":1")
    local found=false
    
    for candidate in "${candidates[@]}"; do
        DISPLAY="$candidate" xdpyinfo >/dev/null 2>&1 && {
            export DISPLAY="$candidate"
            echo "检测到可用的 DISPLAY=$DISPLAY"
            found=true
            break
        }
    done
    
    # 尝试从 X11 socket 查找
    [ "$found" = false ] && while IFS= read -r x_socket; do
        [ -S "$x_socket" ] || continue
        local display_num=$(basename "$x_socket" | sed 's/X//')
        local candidate=":$display_num.0"
        DISPLAY="$candidate" xdpyinfo >/dev/null 2>&1 && {
            export DISPLAY="$candidate"
            echo "从 X11 socket 检测到 DISPLAY=$DISPLAY"
            found=true
            break
        }
    done < <(find /tmp/.X11-unix -maxdepth 1 -name "X*" -type s 2>/dev/null)
    
    [ "$found" = false ] && {
        export DISPLAY=:77.0
        echo "警告: 未检测到可用的 DISPLAY，默认使用 DISPLAY=:77.0" >&2
    }
    
    xdpyinfo >/dev/null 2>&1 && echo "DISPLAY=$DISPLAY 验证通过" || \
        echo "警告: DISPLAY=$DISPLAY 不可用，GUI 模式可能无法工作" >&2
    
    # 设置 Qt/X11 环境变量
    export QT_X11_NO_MITSHM=1
    export QT_SESSION_MANAGER=""
}

setup_display

# ===================== 从 yaml 配置文件读取参数 =====================
CONFIG_FILE="$SCRIPT_DIR/config/train.yaml"
parse_yaml_value() {
    local key="$1"
    grep -E "^[[:space:]]*$key:" "$CONFIG_FILE" 2>/dev/null | head -1 \
        | sed -E "s/^[[:space:]]*$key:[[:space:]]*//;s/[[:space:]]*#.*$//;s/^[[:space:]]*//;s/[[:space:]]*$//" \
        | sed -E "s/^[\"']//;s/[\"']$//"
}

if [ -f "$CONFIG_FILE" ]; then
    RUN_MODE=$(parse_yaml_value "run_mode")
    GPU_ID=$(parse_yaml_value "gpu_id")
    LOG_DIR=$(parse_yaml_value "single_env_log_dir")
fi
ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-77}
RUN_MODE=${RUN_MODE:-1}
[ -n "$CLI_RUN_MODE" ] && RUN_MODE="$CLI_RUN_MODE"

# 规范化运行模式（1=后台；2=可视化）
case "$RUN_MODE" in
    1|2) ;;
    *) RUN_MODE=1 ;;
esac

# ===================== 全局变量初始化 =====================
GAZEBO_PID=""
RVIZ_PID=""
TRAINING_PID=""
INTERRUPTED=false
CLEANUP_DONE=false

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
# 如果配置文件中没有指定log_dir，使用默认值
LOG_DIR=${LOG_DIR:-"$SCRIPT_DIR/log/single_env_training"}
LOG_FOLDER="$LOG_DIR/train_${TIMESTAMP}"
LOGFILE="$LOG_FOLDER/train_${TIMESTAMP}.log"
GAZEBO_WAIT_TIME=5

# ===================== 辅助函数 =====================
setup_xdg_runtime() {
    [ -z "$XDG_RUNTIME_DIR" ] && {
            export XDG_RUNTIME_DIR="${HOME}/.runtime"
        [ -d "$XDG_RUNTIME_DIR" ] && [ ! -O "$XDG_RUNTIME_DIR" ] && rm -rf "$XDG_RUNTIME_DIR" 2>/dev/null || true
            mkdir -p "$XDG_RUNTIME_DIR"
            chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null || true
    }
}

setup_gazebo_paths() {
    local orig_path="${GAZEBO_RESOURCE_PATH:-}"
    local share_dir=""
    
    for path in /usr/share/gazebo-11 /usr/share/gazebo /opt/ros/foxy/share/gazebo_plugins; do
        [ -d "$path" ] && { share_dir="$path"; break; }
    done
    
    if [ -z "$GAZEBO_MODEL_PATH" ]; then
        export GAZEBO_MODEL_PATH="$SCRIPT_DIR/src/turtlebot3_simulations/turtlebot3_gazebo/models"
    else
        export GAZEBO_MODEL_PATH="$GAZEBO_MODEL_PATH:$SCRIPT_DIR/src/turtlebot3_simulations/turtlebot3_gazebo/models"
    fi
    
    if [ -n "$share_dir" ]; then
        [ -n "$orig_path" ] && orig_path="$orig_path:"
        export GAZEBO_RESOURCE_PATH="${orig_path}${share_dir}:$GAZEBO_MODEL_PATH"
    else
        export GAZEBO_RESOURCE_PATH="${orig_path}${orig_path:+:}$GAZEBO_MODEL_PATH"
    fi
    
    export TURTLEBOT3_MODEL=waffle
}

# ===================== 初始化日志 =====================
init_logging() {
    > "$LOGFILE"
    echo "===== 训练启动 - $(date '+%Y-%m-%d %H:%M:%S') =====" | tee -a "$LOGFILE"
    echo "日志文件: $LOGFILE" | tee -a "$LOGFILE"
    echo "ROS_DOMAIN_ID: $ROS_DOMAIN_ID (默认值: 77)" >> "$LOGFILE"
}

# ===================== 启动 Gazebo =====================
start_gazebo() {
    local launch_file
    [ "$RUN_MODE" = "2" ] && launch_file="ros2_drl.launch.py" || launch_file="ros2_drl_headless.launch.py"
    
    echo "启动Gazebo（ROS_DOMAIN_ID=$ROS_DOMAIN_ID，使用 $launch_file）..." | tee -a "$LOGFILE"
    
    export ROS_DOMAIN_ID=$ROS_DOMAIN_ID
    setup_gazebo_paths
    
    if [ "$RUN_MODE" = "1" ]; then
        [ -z "$DISPLAY" ] && export DISPLAY=:77.0
        xdpyinfo >/dev/null 2>&1 || {
            echo "错误: DISPLAY=$DISPLAY 不可用，Gazebo GUI 无法启动" | tee -a "$LOGFILE"
            return 1
        }
        setup_xdg_runtime
        echo "启动 Gazebo GUI，DISPLAY=$DISPLAY" | tee -a "$LOGFILE"
    fi
    
    # 启动 Gazebo
    if [ "$RUN_MODE" = "1" ]; then
        nohup ros2 launch turtlebot3_gazebo $launch_file >> "$LOGFILE" 2>&1 &
    else
        ros2 launch turtlebot3_gazebo $launch_file >> "$LOGFILE" 2>&1 &
    fi
    GAZEBO_PID=$!
    echo "Gazebo已启动 (PID=$GAZEBO_PID)" | tee -a "$LOGFILE"
    
    # 等待初始化
    echo "等待Gazebo初始化 ($GAZEBO_WAIT_TIME秒)..." | tee -a "$LOGFILE"
    for i in $(seq 1 $GAZEBO_WAIT_TIME); do
        [ "$INTERRUPTED" = true ] && return 1
        echo -n "." | tee -a "$LOGFILE"
        (sleep 1) & wait $! 2>/dev/null
    done
    echo -e "\nGazebo初始化完成" | tee -a "$LOGFILE"
}

# ===================== 启动 RViz =====================
start_rviz() {
    [ "$RUN_MODE" != "2" ] && return
    
    setup_xdg_runtime
        echo "启动RViz2（本机显示）..." | tee -a "$LOGFILE"
    nohup rviz2 >> "$LOGFILE" 2>&1 &
        RVIZ_PID=$!
        echo "RViz2已启动 (PID=$RVIZ_PID)" | tee -a "$LOGFILE"
}

# ===================== 启动训练脚本 =====================
start_training() {
    echo "启动训练脚本（参数从train.yaml读取，ROS_DOMAIN_ID=$ROS_DOMAIN_ID）..." | tee -a "$LOGFILE"
    cd "$SCRIPT_DIR" || {
        echo "错误: 无法切换到脚本目录 $SCRIPT_DIR" | tee -a "$LOGFILE"
        return 1
    }
    export TRAINING_TIMESTAMP=$TIMESTAMP
    
    [ "$RUN_MODE" = "4" ] && setup_xdg_runtime
    
    if [ "$RUN_MODE" = "1" ]; then
        nohup unbuffer python3 -u src/drl_navigation_ros2/train.py >> "$LOGFILE" 2>&1 &
        TRAINING_PID=$!
        echo "训练已后台启动 (PID=$TRAINING_PID)" | tee -a "$LOGFILE"
    else
        unbuffer python3 -u src/drl_navigation_ros2/train.py 2>&1 | tee -a "$LOGFILE" &
        TRAINING_PID=$!
        wait $TRAINING_PID 2>/dev/null
    TRAINING_EXIT_CODE=$?
    echo "训练脚本已结束 (退出码: $TRAINING_EXIT_CODE)" | tee -a "$LOGFILE"
    fi
}

# ===================== 信号处理和清理 =====================
handle_interrupt() {
    INTERRUPTED=true
    echo -e "\n\n检测到中断信号 (Ctrl+C)，正在清理..." | tee -a "$LOGFILE"
    cleanup
    exit 130
}

cleanup() {
    [ "$CLEANUP_DONE" = true ] && return
    CLEANUP_DONE=true
    
    echo "清理进程..." | tee -a "$LOGFILE"
    echo "===== 训练结束 - $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOGFILE"
    
    # 调用 clean.sh 进行清理
    local clean_script="$SCRIPT_DIR/clean.sh"
    if [ -f "$clean_script" ] && [ -x "$clean_script" ]; then
        echo "调用 clean.sh 进行清理..." | tee -a "$LOGFILE"
        bash "$clean_script" >> "$LOGFILE" 2>&1 || true
    else
        # 如果 clean.sh 不存在，使用原有清理逻辑
        echo "警告: clean.sh 不存在，使用内置清理逻辑" | tee -a "$LOGFILE"
        for pid in "$GAZEBO_PID" "$RVIZ_PID" "$TRAINING_PID"; do
            [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null
        done
        pkill -f "train.py|ros2 launch turtlebot3_gazebo|gzserver|gazebo" 2>/dev/null
    sleep 1
    fi
}

# ===================== 主执行流程 =====================
main() {
    # 确保工作目录正确（不依赖外部环境）
    cd "$SCRIPT_DIR" || {
        echo "错误: 无法切换到脚本目录 $SCRIPT_DIR" >&2
        exit 1
    }
    
    mkdir -p "$LOG_DIR" "$LOG_FOLDER"
    trap handle_interrupt SIGINT SIGTERM
    
    [ "$DEBUG_MODE" = true ] && set -x
    
    # 设置 CUDA_VISIBLE_DEVICES（在脚本作用域内，不影响外部终端）
    if [ -n "$GPU_ID" ]; then
        export CUDA_VISIBLE_DEVICES="$GPU_ID"
        echo "已设置 CUDA_VISIBLE_DEVICES=$GPU_ID（来自train.yaml）" | tee -a "$LOGFILE"
    else
        # 如果没有指定GPU_ID，确保CUDA_VISIBLE_DEVICES未设置，使用系统默认
        unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
    fi
    
    # 初始化日志
    init_logging

    if [ "$RUN_MODE" = "1" ]; then
        shopt -s expand_aliases
        alias tee="cat >> $LOGFILE"
        exec >>"$LOGFILE" 2>&1
    fi
    trap cleanup EXIT
    
    [ "$INTERRUPTED" = true ] && return 1
    start_gazebo || return 1
    
    [ "$INTERRUPTED" = true ] && return 1
    start_rviz
    
    [ "$INTERRUPTED" = true ] && return 1
    start_training
}

main
