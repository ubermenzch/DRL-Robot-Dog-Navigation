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
# 注意：TURTLEBOT3_MODEL 会在 setup_gazebo_paths 中根据配置文件重新设置
# 保存原始DISPLAY值（如果存在），用于TurboVNC等虚拟桌面环境
ORIGINAL_DISPLAY="${DISPLAY:-}"
unset ROS_DOMAIN_ID CUDA_VISIBLE_DEVICES GAZEBO_IP GAZEBO_MASTER_URI GAZEBO_GUI 2>/dev/null || true
unset GAZEBO_MODEL_PATH GAZEBO_RESOURCE_PATH TURTLEBOT3_MODEL 2>/dev/null || true
unset ROS_PACKAGE_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH 2>/dev/null || true
unset QT_X11_NO_MITSHM QT_SESSION_MANAGER XDG_RUNTIME_DIR 2>/dev/null || true

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
    ROS_DOMAIN_ID_CONFIG=$(parse_yaml_value "single_env_ros_domain_id")
    TURTLEBOT3_MODEL_CONFIG=$(parse_yaml_value "turtlebot3_model")
    GAZEBO_PORT_CONFIG=$(parse_yaml_value "gazebo_port")
    # 强制输出调试信息，不依赖 DEBUG_MODE
    echo "[配置读取] TURTLEBOT3_MODEL_CONFIG='$TURTLEBOT3_MODEL_CONFIG'" >&2
    [ "$DEBUG_MODE" = true ] && echo "[DEBUG] 从配置文件读取: TURTLEBOT3_MODEL_CONFIG='$TURTLEBOT3_MODEL_CONFIG'" >&2
else
    echo "[配置读取] 警告: 配置文件不存在: $CONFIG_FILE" >&2
fi
# 优先使用配置文件中的值，其次使用环境变量，最后使用默认值0
ROS_DOMAIN_ID=${ROS_DOMAIN_ID_CONFIG:-${ROS_DOMAIN_ID:-0}}
RUN_MODE=${RUN_MODE:-1}
[ -n "$CLI_RUN_MODE" ] && RUN_MODE="$CLI_RUN_MODE"
# Gazebo Master端口：优先使用配置文件中的gazebo_port，否则使用默认11345
GAZEBO_PORT=${GAZEBO_PORT_CONFIG:-11345}

# 规范化运行模式（1=后台；2=可视化）
case "$RUN_MODE" in
    1|2) ;;
    *) RUN_MODE=1 ;;
esac

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
    
    # 加载 Gazebo 环境（按优先级检查，只加载第一个找到的）
    if [ -f "/usr/share/gazebo/setup.bash" ]; then
        source /usr/share/gazebo/setup.bash 2>/dev/null && echo "Gazebo 环境已加载"
    elif [ -f "/usr/share/gazebo-11/setup.bash" ]; then
        source /usr/share/gazebo-11/setup.bash 2>/dev/null && echo "Gazebo 环境已加载"
    fi
}

setup_ros2_env

# ===================== DISPLAY 设置 =====================
setup_display() {
    local found=false
    local xauth_file=""
    
    # 查找可能的Xauthority文件（用于X11认证）
    find_xauthority() {
        # 优先使用环境变量中的XAUTHORITY
        [ -n "$XAUTHORITY" ] && [ -f "$XAUTHORITY" ] && {
            echo "$XAUTHORITY"
            return
        }
        # 查找VNC进程对应的用户的Xauthority
        local vnc_user=$(ps aux 2>/dev/null | grep -E "[X]vnc|Xtightvnc|Xtigervnc" | head -1 | awk '{print $1}')
        [ -n "$vnc_user" ] && {
            local user_xauth="/home/$vnc_user/.Xauthority"
            [ -f "$user_xauth" ] && {
                echo "$user_xauth"
                return
            }
        }
        # 查找所有用户的Xauthority文件
        local found_xauth=$(find /home -maxdepth 2 -name ".Xauthority" -type f 2>/dev/null | head -1)
        [ -n "$found_xauth" ] && echo "$found_xauth"
    }
    
    # 允许所有用户访问指定的DISPLAY（用于VNC等虚拟桌面）
    enable_display_access() {
        local target_display="$1"
        local vnc_user=$(ps aux 2>/dev/null | grep -E "[X]vnc|Xtightvnc|Xtigervnc.*$target_display" | head -1 | awk '{print $1}')
        
        # 方法1: 尝试直接使用xhost（优先，最简单）
        if command -v xhost >/dev/null 2>&1; then
            # 先尝试使用VNC用户的Xauthority来执行xhost
            if [ -n "$vnc_user" ]; then
                local vnc_xauth="/home/$vnc_user/.Xauthority"
                if [ -f "$vnc_xauth" ] && [ -r "$vnc_xauth" ]; then
                    DISPLAY="$target_display" XAUTHORITY="$vnc_xauth" xhost +local: >/dev/null 2>&1 && {
                        echo "已使用xhost允许所有本地用户访问 DISPLAY=$target_display"
                        return 0
                    }
                fi
                # 如果使用VNC用户的Xauthority失败，尝试使用sudo
                if command -v sudo >/dev/null 2>&1; then
                    DISPLAY="$target_display" sudo -u "$vnc_user" xhost +local: >/dev/null 2>&1 && {
                        echo "已使用xhost允许所有本地用户访问 DISPLAY=$target_display（通过sudo）"
                        return 0
                    }
                fi
            fi
            # 直接尝试xhost（如果当前用户有权限）
            DISPLAY="$target_display" xhost +local: >/dev/null 2>&1 && {
                echo "已使用xhost允许所有本地用户访问 DISPLAY=$target_display"
                return 0
            }
        fi
        
        # 方法2: 如果xhost不可用或失败，尝试复制Xauthority文件到当前用户
        if [ -n "$vnc_user" ] && command -v xauth >/dev/null 2>&1; then
            local vnc_xauth="/home/$vnc_user/.Xauthority"
            if [ -f "$vnc_xauth" ] && [ -r "$vnc_xauth" ]; then
                local current_xauth="${HOME}/.Xauthority"
                # 提取VNC display的认证信息并合并到当前用户的Xauthority
                DISPLAY="$target_display" XAUTHORITY="$vnc_xauth" xauth extract - "$target_display" 2>/dev/null | \
                    xauth merge - 2>/dev/null && {
                    echo "已复制X11认证信息以访问 DISPLAY=$target_display"
                    return 0
                }
            fi
        fi
        
        return 1
    }
    
    xauth_file=$(find_xauthority)
    [ -n "$xauth_file" ] && export XAUTHORITY="$xauth_file"
    
    # 测试DISPLAY是否可用的辅助函数
    test_display() {
        local test_display="$1"
        local test_xauth="${2:-$XAUTHORITY}"
        if [ -n "$test_xauth" ] && [ -f "$test_xauth" ]; then
            DISPLAY="$test_display" XAUTHORITY="$test_xauth" xdpyinfo >/dev/null 2>&1
        else
            DISPLAY="$test_display" xdpyinfo >/dev/null 2>&1
        fi
    }
    
    # 优先使用原始DISPLAY值（来自TurboVNC等虚拟桌面环境）
    if [ -n "$ORIGINAL_DISPLAY" ]; then
        test_display "$ORIGINAL_DISPLAY" "$xauth_file" && {
            export DISPLAY="$ORIGINAL_DISPLAY"
            [ -n "$xauth_file" ] && export XAUTHORITY="$xauth_file"
            echo "使用原始 DISPLAY=$DISPLAY（来自环境变量）"
            found=true
        }
    fi
    
    # 如果原始DISPLAY不可用，尝试从 X11 socket 查找所有可用的display
    [ "$found" = false ] && while IFS= read -r x_socket; do
        [ -S "$x_socket" ] || continue
        local display_num=$(basename "$x_socket" | sed 's/X//')
        # 尝试多种格式
        for format in ":$display_num" ":$display_num.0"; do
            # 先尝试直接测试
            if test_display "$format" "$xauth_file"; then
                export DISPLAY="$format"
                [ -n "$xauth_file" ] && export XAUTHORITY="$xauth_file"
                echo "从 X11 socket 检测到 DISPLAY=$DISPLAY"
                found=true
                break 2
            else
                # 如果测试失败，尝试启用访问权限（可能是VNC display）
                enable_display_access "$format" && test_display "$format" && {
                    export DISPLAY="$format"
                    unset XAUTHORITY  # 使用xhost后可能不需要XAUTHORITY
                    echo "从 X11 socket 检测到 DISPLAY=$DISPLAY（已启用多用户访问）"
                    found=true
                    break 2
                }
            fi
        done
    done < <(find /tmp/.X11-unix -maxdepth 1 -name "X*" -type s 2>/dev/null | sort -V)
    
    # 如果还是没找到，尝试常见的候选值
    [ "$found" = false ] && {
        local candidates=(":1" ":1.0" ":2" ":2.0" ":0" ":0.0")
        for candidate in "${candidates[@]}"; do
            test_display "$candidate" "$xauth_file" && {
                export DISPLAY="$candidate"
                [ -n "$xauth_file" ] && export XAUTHORITY="$xauth_file"
                echo "检测到可用的 DISPLAY=$DISPLAY"
                found=true
                break
            }
        done
    }
    
    # 最后尝试：检查VNC相关的环境变量或进程
    [ "$found" = false ] && {
        # 检查是否有VNC进程在运行
        local vnc_display=$(ps aux 2>/dev/null | grep -E "[X]vnc|Xtightvnc|Xtigervnc" | grep -oE ":[0-9]+" | head -1)
        if [ -n "$vnc_display" ]; then
            # 先尝试直接测试
            if test_display "$vnc_display" "$xauth_file"; then
                export DISPLAY="$vnc_display"
                [ -n "$xauth_file" ] && export XAUTHORITY="$xauth_file"
                echo "从VNC进程检测到 DISPLAY=$DISPLAY"
                found=true
            else
                # 如果测试失败，尝试启用访问权限
                enable_display_access "$vnc_display" && test_display "$vnc_display" && {
                    export DISPLAY="$vnc_display"
                    unset XAUTHORITY  # 使用xhost后可能不需要XAUTHORITY
                    echo "从VNC进程检测到 DISPLAY=$DISPLAY（已启用多用户访问）"
                    found=true
                }
            fi
        fi
    }
    
    [ "$found" = false ] && {
        export DISPLAY=:0.0
        echo "警告: 未检测到可用的 DISPLAY，默认使用 DISPLAY=:0.0" >&2
    }
    
    # 验证DISPLAY是否可用
    if xdpyinfo >/dev/null 2>&1; then
        echo "DISPLAY=$DISPLAY 验证通过"
        [ -n "$XAUTHORITY" ] && echo "使用 XAUTHORITY=$XAUTHORITY"
    else
        echo "警告: DISPLAY=$DISPLAY 不可用，GUI 模式可能无法工作" >&2
        echo "提示: 请确保TurboVNC viewer已连接，或检查DISPLAY环境变量设置" >&2
        [ -n "$xauth_file" ] && echo "提示: 已尝试使用 XAUTHORITY=$xauth_file，但仍无法连接" >&2
    fi
    
    # 设置 Qt/X11 环境变量
    export QT_X11_NO_MITSHM=1
    export QT_SESSION_MANAGER=""
}

setup_display

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
    
    # 强制将项目模型路径放在最前面，避免多环境运行时使用系统/共享缓存中的模型
    # 这确保单环境始终使用项目内的模型（如 100by100, obstacle_cube_5m），而不是多环境已加载的模型（如 10by10, obstacle5）
    local project_models_path="$SCRIPT_DIR/src/turtlebot3_simulations/turtlebot3_gazebo/models"
    
    # 清除可能存在的用户模型缓存路径（~/.gazebo/models），避免使用缓存的错误模型
    local user_gazebo_models="${HOME}/.gazebo/models"
    if [ -z "$GAZEBO_MODEL_PATH" ]; then
        export GAZEBO_MODEL_PATH="$project_models_path"
    else
        # 移除可能存在的项目路径和用户缓存路径，然后前置项目路径（避免重复和缓存干扰）
        GAZEBO_MODEL_PATH=$(echo "$GAZEBO_MODEL_PATH" | sed "s|:$project_models_path||g; s|$project_models_path:||g; s|^$project_models_path$||; s|:$user_gazebo_models||g; s|$user_gazebo_models:||g; s|^$user_gazebo_models$||")
        export GAZEBO_MODEL_PATH="$project_models_path${GAZEBO_MODEL_PATH:+:}$GAZEBO_MODEL_PATH"
    fi
    
    if [ -n "$share_dir" ]; then
        [ -n "$orig_path" ] && orig_path="$orig_path:"
        export GAZEBO_RESOURCE_PATH="${orig_path}${share_dir}:$GAZEBO_MODEL_PATH"
    else
        export GAZEBO_RESOURCE_PATH="${orig_path}${orig_path:+:}$GAZEBO_MODEL_PATH"
    fi
    
    # 从配置文件读取 turtlebot3_model，如果没有则使用默认值 waffle
    # 强制重新读取配置文件，确保使用最新值
    # 使用 SCRIPT_DIR 构建配置文件路径，不依赖外部变量
    local config_file_path="$SCRIPT_DIR/config/train.yaml"
    if [ -f "$config_file_path" ]; then
        # 直接在函数内解析，不依赖外部函数或变量
        TURTLEBOT3_MODEL_CONFIG=$(grep -E "^[[:space:]]*turtlebot3_model:" "$config_file_path" 2>/dev/null | head -1 \
            | sed -E "s/^[[:space:]]*turtlebot3_model:[[:space:]]*//;s/[[:space:]]*#.*$//;s/^[[:space:]]*//;s/[[:space:]]*$//" \
            | sed -E "s/^[\"']//;s/[\"']$//")
    else
        echo "[setup_gazebo_paths] 警告: 配置文件不存在: $config_file_path" >&2
    fi
    # 设置环境变量，优先使用配置文件中的值
    export TURTLEBOT3_MODEL=${TURTLEBOT3_MODEL_CONFIG:-waffle}
    # 输出到日志文件（如果存在）和 stderr（用于调试）
    if [ -f "$LOGFILE" ]; then
        echo "已设置 TURTLEBOT3_MODEL=$TURTLEBOT3_MODEL（来自train.yaml: $config_file_path）" | tee -a "$LOGFILE"
    else
        echo "已设置 TURTLEBOT3_MODEL=$TURTLEBOT3_MODEL（来自train.yaml: $config_file_path）" >&2
    fi
}

# ===================== 初始化日志 =====================
init_logging() {
    > "$LOGFILE"
    echo "===== 训练启动 - $(date '+%Y-%m-%d %H:%M:%S') =====" | tee -a "$LOGFILE"
    echo "日志文件: $LOGFILE" | tee -a "$LOGFILE"
    echo "ROS_DOMAIN_ID: $ROS_DOMAIN_ID (默认值: 0，可从train.yaml中配置)" >> "$LOGFILE"
    echo "GAZEBO_PORT: $GAZEBO_PORT (默认值: 11345，可从train.yaml中配置 gazebo_port)" >> "$LOGFILE"
}

# ===================== 启动 Gazebo =====================
start_gazebo() {
    local launch_file
    [ "$RUN_MODE" = "2" ] && launch_file="ros2_drl.launch.py" || launch_file="ros2_drl_headless.launch.py"
    
    echo "启动Gazebo（ROS_DOMAIN_ID=$ROS_DOMAIN_ID，GAZEBO_PORT=$GAZEBO_PORT，使用 $launch_file）..." | tee -a "$LOGFILE"
    
    export ROS_DOMAIN_ID=$ROS_DOMAIN_ID
    # 设置 Gazebo Master URI（从配置文件读取 gazebo_port）
    export GAZEBO_MASTER_URI="http://127.0.0.1:$GAZEBO_PORT"
    export GAZEBO_IP=127.0.0.1
    # 清除可能残留的 TURTLEBOT3_MODEL（如先跑多环境再跑单环境时），确保仅用 train.yaml
    unset TURTLEBOT3_MODEL 2>/dev/null || true
    # 确保 TURTLEBOT3_MODEL 环境变量在启动 Gazebo 前已设置（来自 train.yaml）
    setup_gazebo_paths
    # 再次确认环境变量已设置（用于调试）
    [ "$DEBUG_MODE" = true ] && echo "[DEBUG] start_gazebo: TURTLEBOT3_MODEL=$TURTLEBOT3_MODEL" | tee -a "$LOGFILE"
    
    if [ "$RUN_MODE" = "1" ]; then
        # 后台模式：使用无头 Gazebo（仅 gzserver），不依赖 DISPLAY
        export GAZEBO_GUI=0
        unset DISPLAY 2>/dev/null || true
        echo "启动无头 Gazebo（ros2_drl_headless），DISPLAY 未使用" | tee -a "$LOGFILE"
    else
        # 可视化模式：需要 DISPLAY
        [ -z "$DISPLAY" ] && export DISPLAY=:0.0
        xdpyinfo >/dev/null 2>&1 || {
            echo "错误: DISPLAY=$DISPLAY 不可用，Gazebo GUI 无法启动" | tee -a "$LOGFILE"
            return 1
        }
        setup_xdg_runtime
        echo "启动 Gazebo GUI，DISPLAY=$DISPLAY" | tee -a "$LOGFILE"
    fi
    
    # 启动 Gazebo：显式传入 TURTLEBOT3_MODEL、GAZEBO_MODEL_PATH 和 GAZEBO_MASTER_URI，避免被多环境等残留 env 覆盖
    # 确保模型路径优先使用项目路径，防止多环境运行时加载错误的模型（如 10by10 而非 100by100）
    # 使用配置文件中的 gazebo_port 作为 Gazebo Master 端口，避免与多环境训练端口冲突
    # TURTLEBOT3_ROBOT_MODEL 固定为 waffle（用于机器人 URDF），TURTLEBOT3_MODEL 用于世界文件
    export TURTLEBOT3_ROBOT_MODEL=waffle
    if [ "$RUN_MODE" = "1" ]; then
        nohup env TURTLEBOT3_MODEL="$TURTLEBOT3_MODEL" TURTLEBOT3_ROBOT_MODEL="$TURTLEBOT3_ROBOT_MODEL" GAZEBO_MODEL_PATH="$GAZEBO_MODEL_PATH" GAZEBO_RESOURCE_PATH="$GAZEBO_RESOURCE_PATH" GAZEBO_MASTER_URI="$GAZEBO_MASTER_URI" GAZEBO_IP="$GAZEBO_IP" ros2 launch turtlebot3_gazebo $launch_file >> "$LOGFILE" 2>&1 &
    else
        env TURTLEBOT3_MODEL="$TURTLEBOT3_MODEL" TURTLEBOT3_ROBOT_MODEL="$TURTLEBOT3_ROBOT_MODEL" GAZEBO_MODEL_PATH="$GAZEBO_MODEL_PATH" GAZEBO_RESOURCE_PATH="$GAZEBO_RESOURCE_PATH" GAZEBO_MASTER_URI="$GAZEBO_MASTER_URI" GAZEBO_IP="$GAZEBO_IP" ros2 launch turtlebot3_gazebo $launch_file >> "$LOGFILE" 2>&1 &
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
    export LOG_DIR="$LOG_FOLDER"  # 导出日志目录，供train.py使用，确保模型保存到日志目录
    
    [ "$RUN_MODE" = "4" ] && setup_xdg_runtime
    
    if [ "$RUN_MODE" = "1" ]; then
        # 后台模式：只重定向到日志，不输出到终端
        nohup unbuffer python3 -u src/drl_navigation_ros2/train.py >> "$LOGFILE" 2>&1 &
        TRAINING_PID=$!
        echo "训练已后台启动 (PID=$TRAINING_PID)" | tee -a "$LOGFILE"
    else
        # 可视化模式：同时输出到终端和日志
        unbuffer python3 -u src/drl_navigation_ros2/train.py 2>&1 | tee -a "$LOGFILE"
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

    if [ "$RUN_MODE" = "1" ]; then
        # 后台模式：不清理，让无头 Gazebo 与训练继续运行；仅 Ctrl+C 时清理
        trap - EXIT
        exit 0
    fi
}

main
