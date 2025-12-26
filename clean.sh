#!/bin/bash
# 统一清理脚本 - 清理所有训练和评估相关进程
# 合并了 stop_training.sh、stop_multi_env_training.sh 和 stop_evaluation.sh 的功能

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DIR="$SCRIPT_DIR/tmp"

echo "=========================================="
echo "清理所有训练和评估相关进程..."
echo "=========================================="

# 标记是否有进程被停止
PROCESSES_STOPPED=0

# 清理过程中忽略 Ctrl+C / SIGTERM，避免中断
trap 'echo "clean.sh: 正在清理，忽略中断信号..."' INT TERM

# ===================== 辅助函数 =====================
# 优雅停止进程（先TERM，再KILL）
stop_process() {
    local pid="$1"
    local name="${2:-进程}"
    local kill_pgrp="${3:-false}"
    
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    
    echo "  停止 $name (PID: $pid)..."
    kill -TERM "$pid" 2>/dev/null
    sleep 2
    
    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null
        sleep 1
    fi
    
    if [ "$kill_pgrp" = "true" ]; then
        local pgrp=$(ps -o pgid= "$pid" 2>/dev/null | tr -d ' ')
        [ -n "$pgrp" ] && kill -9 -$pgrp 2>/dev/null
    fi
    
    PROCESSES_STOPPED=1
    return 0
}

# 通过PID文件停止进程
stop_by_pid_file() {
    local pid_file="$1"
    local name="$2"
    local kill_pgrp="${3:-false}"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        stop_process "$pid" "$name" "$kill_pgrp"
        rm -f "$pid_file"
    fi
}

# 停止匹配模式的进程
stop_by_pattern() {
    local pattern="$1"
    local name="$2"
    local pids=$(pgrep -f "$pattern" 2>/dev/null)
    
    if [ -z "$pids" ]; then
        return 0
    fi
    
    echo "  停止 $name..."
    for pid in $pids; do
        kill -TERM $pid 2>/dev/null
    done
    sleep 2
    
    pids=$(pgrep -f "$pattern" 2>/dev/null)
    if [ -n "$pids" ]; then
        for pid in $pids; do
            kill -9 $pid 2>/dev/null
        done
        PROCESSES_STOPPED=1
    fi
}

# 停止进程名列表
stop_process_list() {
    local pattern="$1"
    local name="$2"
    local pids=$(pgrep -f "$pattern" 2>/dev/null)
    
    if [ -n "$pids" ]; then
        echo "  停止 $name..."
        pkill -TERM -f "$pattern" 2>/dev/null
        sleep 2
        pkill -9 -f "$pattern" 2>/dev/null
        PROCESSES_STOPPED=1
    fi
}

# ===================== 1. 停止主后台进程（通过PID文件）=====================
echo "[1/8] 停止主后台进程..."
stop_by_pid_file "$TMP_DIR/multi_env_training.pid" "多环境训练主进程" "true"
stop_by_pid_file "$TMP_DIR/evaluation.pid" "评估主进程"

# ===================== 2. 停止启动脚本进程 =====================
echo "[2/8] 停止启动脚本进程..."
for script in "start_training.sh" "start_multi_env_training.sh" "start_evaluation.sh"; do
    pids=$(pgrep -f "$script" 2>/dev/null)
    if [ -n "$pids" ]; then
        for pid in $pids; do
            if ps -p "$pid" >/dev/null 2>&1; then
                stop_process "$pid" "$script" "true"
            fi
        done
    fi
done

# ===================== 3. 停止Python脚本 =====================
echo "[3/8] 停止Python脚本..."
stop_by_pattern "train\.py" "train.py"
stop_by_pattern "multi_env_train\.py" "multi_env_train.py"
stop_by_pattern "evaluate\.py" "evaluate.py"

# ===================== 4. 停止Gazebo相关进程 =====================
echo "[4/8] 停止Gazebo进程..."

# 停止Gazebo进程名列表
GAZEBO_PROCESSES=("gazebo" "gzserver" "gzclient" "gz" "gzsim" "gazebo_ros")
for proc in "${GAZEBO_PROCESSES[@]}"; do
    pids=$(pgrep -x "$proc" 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "  停止 $proc 进程..."
        killall -TERM "$proc" 2>/dev/null
        sleep 1
        killall -9 "$proc" 2>/dev/null
        PROCESSES_STOPPED=1
    fi
done

# 通过进程名匹配停止Gazebo相关进程
stop_process_list "ros2 launch turtlebot3_gazebo" "Gazebo启动进程"
stop_process_list "gazebo" "Gazebo进程"

# ===================== 5. 停止ROS2和RViz进程 =====================
echo "[5/8] 停止ROS2和RViz进程..."
stop_process_list "ros2" "ROS2进程"
stop_process_list "rviz2" "RViz2进程"

# ===================== 6. 停止环境控制脚本进程 =====================
echo "[6/8] 停止环境控制脚本进程..."
stop_by_pattern "start_gazebo_env_.*\.sh" "Gazebo环境启动脚本"
stop_by_pattern "start_gazebo_eval_env_.*\.sh" "评估环境启动脚本"

# ===================== 7. 停止其他相关进程 =====================
echo "[7/8] 停止其他相关进程..."
stop_process_list "tensorboard" "TensorBoard进程"

# Xvfb进程
pids=$(pgrep -x "Xvfb" 2>/dev/null)
if [ -n "$pids" ]; then
    echo "  停止Xvfb进程..."
    killall -9 Xvfb 2>/dev/null
    PROCESSES_STOPPED=1
fi

# unbuffer进程
stop_process_list "unbuffer.*\(train\|multi_env_train\|evaluate\)" "unbuffer进程"

# ===================== 8. 清理临时文件和共享内存 =====================
echo "[8/8] 清理临时文件和共享内存..."

# 清理临时脚本文件
rm -f "$TMP_DIR"/start_gazebo_env_*.sh \
      "$TMP_DIR"/check_services_env_*.sh \
      "$TMP_DIR"/start_gazebo_eval_env_*.sh \
      "$TMP_DIR"/check_services_eval_env_*.sh \
      "$TMP_DIR"/eval_gazebo_*.log \
      "$TMP_DIR"/*.pid 2>/dev/null

# 清理评估脚本目录
[ -d "$TMP_DIR/eval_scripts" ] && rm -rf "$TMP_DIR/eval_scripts" 2>/dev/null || true

# 清理ROS2共享内存段
if [ -d "/dev/shm" ]; then
    echo "  清理ROS2共享内存段..."
    # 先尝试不使用sudo删除（通常/dev/shm所有用户都有写权限）
    rm -f /dev/shm/fastrtps_* 2>/dev/null || true
    # 如果仍有残留文件，尝试删除当前用户拥有的文件
    shopt -s nullglob
    for file in /dev/shm/fastrtps_*; do
        [ -f "$file" ] && [ -O "$file" ] && rm -f "$file" 2>/dev/null || true
    done
    shopt -u nullglob
fi

# 如果临时目录为空，则删除目录
[ -d "$TMP_DIR" ] && [ -z "$(ls -A "$TMP_DIR" 2>/dev/null)" ] && rmdir "$TMP_DIR" 2>/dev/null || true

# ===================== 最终确认 =====================
echo ""
echo "最终检查残留进程..."
PATTERN="train\.py|multi_env_train\.py|evaluate\.py|gazebo|ros2 launch turtlebot3_gazebo|start_gazebo_env|start_gazebo_eval_env|start_training\.sh|start_multi_env_training\.sh|start_evaluation\.sh"
REMAINING=$(pgrep -f "$PATTERN" 2>/dev/null | wc -l)

if [ "$REMAINING" -gt 0 ]; then
    echo "  ⚠ 警告: 仍有 $REMAINING 个相关进程在运行"
    echo "  进程列表:"
    pgrep -f "$PATTERN" 2>/dev/null | xargs ps -p 2>/dev/null || true
else
    echo "  ✓ 确认: 没有残留进程"
fi

echo ""
echo "=========================================="
if [ $PROCESSES_STOPPED -eq 1 ]; then
    echo "✓ 所有训练和评估相关进程已停止"
else
    echo "- 未发现需要停止的进程（可能已经停止）"
fi
echo "=========================================="
