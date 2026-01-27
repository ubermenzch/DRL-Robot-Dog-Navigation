"""
统一日志工具：收集 / 训练 / 环境 专用日志，避免重复实现。

日志文件归类：
- collect_log_<timestamp>.log：数据收集进程操作，含 env_id、input/output。记录操作包括：collect_start, set_ros_domain_id, init_ros_env, create_local_model, create_model_manager, model_wait_start, model_sync, model_verify, init_complete, phase_skip, eval_target_dist_set, model_sync_eval, model_sync_episode, reset, reset_discard, prepare_state, episode_start, get_action, transfor_action, step, transition_trim, episode_end（含多种 ending）, queue_put, episode_reward_detail, round_done, collect_exit。
- train_log_<timestamp>.log：训练线程及 SAC 训练调试（轮次、损失、评估、保存等）。
- env_log_<timestamp>.log：环境/ROS 调试（ros_python、ros_nodes），含 env_id、时间戳。

所有日志行统一带 [YYYY-MM-DD HH:MM:SS.mmm] 时间戳；Collect/Env 含 env_id。

单环境训练（train.py）可复用 multi_env_log_paths、EnvLogger、TrainLogger 等，将 env_log/train_log
写入同一 log 目录，以实现与多环境一致的归类。
"""

__all__ = [
    "fmt_io",
    "fmt_io_detailed",
    "append_timestamped_line",
    "multi_env_log_paths",
    "TimestampedFileLogger",
    "CollectLogger",
    "TrainLogger",
    "EnvLogger",
    "RewardLogger",
    "NodesLogger",
]

from pathlib import Path
from datetime import datetime
import numpy as np


def multi_env_log_paths(log_dir, timestamp):
    """返回多环境训练所用的日志路径。log_dir 为 train_<timestamp> 目录，timestamp 与 sh 脚本一致。"""
    d = Path(log_dir) if isinstance(log_dir, str) else log_dir
    ts = str(timestamp)
    return {
        "collect_log_path": str(d / f"collect_log_{ts}.log"),
        "train_log_path": str(d / f"train_log_{ts}.log"),
        "env_log_path": str(d / f"env_log_{ts}.log"),
        "reward_log_path": str(d / f"reward_log_{ts}.log"),
        "nodes_log_path": str(d / f"nodes_log_{ts}.log"),
    }


def fmt_io(obj, max_len=200):
    """将输入/输出格式化为简短字符串，用于收集日志。截断过长内容。"""
    if obj is None:
        return ""
    if isinstance(obj, bool):
        return str(obj)
    if isinstance(obj, (int, float)):
        return str(obj) if isinstance(obj, int) else f"{obj:.6g}"
    if isinstance(obj, (list, tuple)):
        n = len(obj)
        if n == 0:
            return "[]"
        try:
            flat = [float(x) for x in obj]
            if n <= 4:
                return str(flat)
            return f"len={n},sample=[{flat[0]:.3g},{flat[1]:.3g},...]"
        except (TypeError, ValueError):
            return f"len={n}"
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        arr = np.asarray(obj)
        if arr.size == 1:
            try:
                return f"{float(arr.flat[0]):.6g}"
            except Exception:
                pass
        sh = getattr(obj, "shape", ())
        return f"shape={sh},dtype={getattr(obj, 'dtype', '')}"
    if isinstance(obj, dict):
        parts = []
        for k, v in list(obj.items())[:10]:
            vstr = fmt_io(v, max_len=50)
            if len(vstr) > 50:
                vstr = vstr[:47] + "..."
            parts.append(f"{k}={vstr}")
        s = ",".join(parts)
        return s[:max_len] + ("..." if len(s) > max_len else "")
    s = str(obj)
    return s[:max_len] + ("..." if len(s) > max_len else "")


def fmt_io_detailed(obj, max_value_len=4000, max_list_len=500, max_list_sample=50):
    """将输入/输出格式化为详细字符串，用于 collect_log。保留完整标量、小数组与字典各键值。"""
    if obj is None:
        return ""
    if isinstance(obj, bool):
        return str(obj)
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        return f"{obj:.10g}"
    if isinstance(obj, (list, tuple)):
        n = len(obj)
        if n == 0:
            return "[]"
        try:
            flat = [float(x) for x in obj]
            if n <= max_list_len:
                return "[" + ",".join(f"{x:.8g}" for x in flat) + "]"
            head = ",".join(f"{x:.8g}" for x in flat[:max_list_sample])
            tail = ",".join(f"{x:.8g}" for x in flat[-max_list_sample:])
            return f"len={n},first{max_list_sample}=[{head}],last{max_list_sample}=[{tail}]"
        except (TypeError, ValueError):
            return f"len={n}"
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        arr = np.asarray(obj).flatten()
        n = arr.size
        if n == 0:
            return "[]"
        if n == 1:
            return f"{float(arr.flat[0]):.10g}"
        try:
            flat = arr.tolist()
            if n <= max_list_len:
                return "[" + ",".join(f"{x:.8g}" for x in flat) + "]"
            head = ",".join(f"{x:.8g}" for x in flat[:max_list_sample])
            tail = ",".join(f"{x:.8g}" for x in flat[-max_list_sample:])
            return f"shape={arr.shape},len={n},first{max_list_sample}=[{head}],last{max_list_sample}=[{tail}]"
        except Exception:
            return f"shape={arr.shape},dtype={getattr(obj, 'dtype', '')}"
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            vstr = fmt_io_detailed(v, max_value_len=max_value_len, max_list_len=max_list_len, max_list_sample=max_list_sample)
            if len(vstr) > max_value_len:
                vstr = vstr[: max_value_len - 3] + "..."
            parts.append(f"{k}={vstr}")
        return ",".join(parts)
    s = str(obj)
    return s if len(s) <= max_value_len else s[: max_value_len - 3] + "..."


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def append_timestamped_line(log_path, msg):
    """一次性追加一条带时间戳的日志行（如 SAC 训练侧零星写入）。"""
    if not (log_path and str(log_path).strip()):
        return
    p = Path(log_path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"[{_ts()}] {msg}\n")
            f.flush()
    except Exception:
        pass


class TimestampedFileLogger:
    """带时间戳的文件日志基类：统一管理路径、打开、写入、关闭。"""

    def __init__(self, log_path, tag="Logger"):
        self.log_path = (log_path or "").strip() or None
        self._file = None
        self._tag = tag

    def _ensure_open(self):
        if self._file is None and self.log_path:
            try:
                Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
                self._file = open(self.log_path, "a", encoding="utf-8")
            except Exception as e:
                self._file = None
                print(f"[{self._tag}] 无法打开日志 {self.log_path}: {e}")

    def _write_line(self, line: str):
        if not self.log_path or self._file is None:
            return
        try:
            self._file.write(line if line.endswith("\n") else line + "\n")
            self._file.flush()
        except Exception as e:
            print(f"[{self._tag}] 写入失败: {e}")

    def close(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None


class CollectLogger(TimestampedFileLogger):
    """数据收集进程专用：collect_log_<timestamp>.log，记录操作、env_id、input、output。"""

    def __init__(self, log_path):
        super().__init__(log_path, tag="CollectLogger")

    def log(self, env_id, operation, input_data=None, output_data=None):
        # 只记录0号环境的日志
        if env_id != 0:
            return
        if not self.log_path:
            return
        self._ensure_open()
        if self._file is None:
            return
        inp = fmt_io_detailed(input_data)
        out = fmt_io_detailed(output_data)
        line = f"[{_ts()}] env_id={env_id} op={operation}"
        if inp:
            line += f" input={inp}"
        if out:
            line += f" output={out}"
        self._write_line(line)


class TrainLogger(TimestampedFileLogger):
    """训练线程及 SAC 训练调试：train_log_<timestamp>.log。"""

    def __init__(self, log_path):
        super().__init__(log_path, tag="TrainLogger")

    def log(self, msg: str):
        if not self.log_path:
            return
        self._ensure_open()
        if self._file is None:
            return
        self._write_line(f"[{_ts()}] {msg}")


class EnvLogger(TimestampedFileLogger):
    """环境/ROS 调试：env_log_<timestamp>.log，每行含 env_id。"""

    def __init__(self, log_path):
        super().__init__(log_path, tag="EnvLogger")

    def log(self, env_id, msg: str):
        # 只记录0号环境的日志
        if env_id != 0:
            return
        if not self.log_path:
            return
        self._ensure_open()
        if self._file is None:
            return
        self._write_line(f"[{_ts()}] env_id={env_id} {msg}")


class RewardLogger(TimestampedFileLogger):
    """奖励调试专用：reward_log_<timestamp>.log，每行含 env_id，记录每步的 reward 详细信息。"""

    def __init__(self, log_path):
        super().__init__(log_path, tag="RewardLogger")

    def log(self, env_id, msg: str):
        # 只记录0号环境的日志
        if env_id != 0:
            return
        if not self.log_path:
            return
        self._ensure_open()
        if self._file is None:
            return
        self._write_line(f"[{_ts()}] env_id={env_id} {msg}")


class NodesLogger(TimestampedFileLogger):
    """ROS节点调试专用：nodes_log_<timestamp>.log，每行含 env_id，记录 ros_nodes.py 中所有调试信息。"""

    def __init__(self, log_path):
        super().__init__(log_path, tag="NodesLogger")

    def log(self, env_id, msg: str):
        # 只记录0号环境的日志
        if env_id != 0:
            return
        if not self.log_path:
            return
        self._ensure_open()
        if self._file is None:
            return
        self._write_line(f"[{_ts()}] env_id={env_id} {msg}")
