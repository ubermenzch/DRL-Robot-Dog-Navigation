import math
import random
from collections import deque

import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123, recent_buffer_ratio=0.1, recent_batch_ratio=0.3):
        """
        The right side of the deque contains the most recent experiences
        
        Args:
            buffer_size: 缓冲区大小
            random_seed: 随机种子
            recent_buffer_ratio: 最新数据比例（0-1之间，例如0.1表示最新的10%数据）
            recent_batch_ratio: batch中来自最新数据的比例（0-1之间，例如0.3表示batch的30%来自最新的recent_buffer_ratio部分）
        """
        self.buffer_size = int(buffer_size)
        self.count = 0
        self.buffer = deque()
        self.recent_buffer_ratio = float(recent_buffer_ratio)
        self.recent_batch_ratio = float(recent_batch_ratio)
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def add_episode(self, experiences):
        for exp in experiences:
            if len(exp) != 5:
                raise ValueError(f"experience must have 5 elements, got {len(exp)}")
            self.add(exp[0], exp[1], exp[2], exp[3], exp[4])

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count <= 0:
            return None

        actual_batch_size = min(self.count, batch_size)

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            buffer_list = list(self.buffer)
            total_size = len(buffer_list)

            recent_size = max(1, int(total_size * self.recent_buffer_ratio))

            recent_batch_size = int(actual_batch_size * self.recent_batch_ratio)
            recent_batch_size = max(0, min(recent_batch_size, actual_batch_size))
            old_batch_size = actual_batch_size - recent_batch_size

            recent_part = buffer_list[-recent_size:]
            if recent_batch_size > 0:
                if len(recent_part) >= recent_batch_size:
                    recent_samples = random.sample(recent_part, recent_batch_size)
                else:
                    recent_samples = list(recent_part)
            else:
                recent_samples = []

            old_part = buffer_list[:-recent_size] if recent_size > 0 else buffer_list
            if old_batch_size > 0:
                if len(old_part) >= old_batch_size:
                    old_samples = random.sample(old_part, old_batch_size)
                else:
                    old_samples = list(old_part)
            else:
                old_samples = []

            batch = recent_samples + old_samples
            random.shuffle(batch)

        if not batch:
            return None

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
        t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def return_buffer(self):
        s = np.array([_[0] for _ in self.buffer])
        a = np.array([_[1] for _ in self.buffer])
        r = np.array([_[2] for _ in self.buffer]).reshape(-1, 1)
        t = np.array([_[3] for _ in self.buffer]).reshape(-1, 1)
        s2 = np.array([_[4] for _ in self.buffer])

        return s, a, r, t, s2

    def clear(self):
        self.buffer.clear()
        self.count = 0


class NumpyReplayBuffer:
    def __init__(self, buffer_size, dtype=np.float32, recent_buffer_ratio=0.1, recent_batch_ratio=0.3):
        self.max_size = int(buffer_size)
        self.count = 0
        self.write_pos = 0
        self.initialized = False
        self.dtype = dtype
        self.recent_buffer_ratio = float(recent_buffer_ratio)
        self.recent_batch_ratio = float(recent_batch_ratio)

        self.states = None
        self.actions = None
        self.rewards = None
        self.dones = None
        self.next_states = None

    def _lazy_init(self, example_exp):
        state_example, action_example, _, _, next_state_example = example_exp

        state_example = np.asarray(state_example, dtype=self.dtype)
        next_state_example = np.asarray(next_state_example, dtype=self.dtype)
        action_example = np.asarray(action_example, dtype=self.dtype)

        state_dim = state_example.shape[0]
        next_state_dim = next_state_example.shape[0]
        if state_dim != next_state_dim:
            raise ValueError(
                f"NumpyReplayBuffer expects state_dim == next_state_dim, got {state_dim} vs {next_state_dim}"
            )

        action_dim = action_example.shape[0]

        self.states = np.zeros((self.max_size, state_dim), dtype=self.dtype)
        self.next_states = np.zeros((self.max_size, state_dim), dtype=self.dtype)
        self.actions = np.zeros((self.max_size, action_dim), dtype=self.dtype)
        self.rewards = np.zeros((self.max_size, 1), dtype=self.dtype)
        self.dones = np.zeros((self.max_size, 1), dtype=self.dtype)

        self.initialized = True

    def add(self, s, a, r, done, s2):
        exp = (s, a, r, done, s2)
        if not self.initialized:
            self._lazy_init(exp)

        idx = self.write_pos
        self.states[idx] = np.asarray(s, dtype=self.dtype)
        self.actions[idx] = np.asarray(a, dtype=self.dtype)
        self.rewards[idx, 0] = float(r)
        self.dones[idx, 0] = float(done)
        self.next_states[idx] = np.asarray(s2, dtype=self.dtype)

        self.write_pos = (self.write_pos + 1) % self.max_size
        self.count = min(self.count + 1, self.max_size)

    def add_batch(self, experiences):
        if not experiences:
            return

        if not self.initialized:
            self._lazy_init(experiences[0])

        batch_size = len(experiences)

        states_arr = np.stack([np.asarray(exp[0], dtype=self.dtype) for exp in experiences], axis=0)
        actions_arr = np.stack([np.asarray(exp[1], dtype=self.dtype) for exp in experiences], axis=0)
        rewards_arr = np.asarray([float(exp[2]) for exp in experiences], dtype=self.dtype).reshape(-1, 1)
        dones_arr = np.asarray([float(exp[3]) for exp in experiences], dtype=self.dtype).reshape(-1, 1)
        next_states_arr = np.stack([np.asarray(exp[4], dtype=self.dtype) for exp in experiences], axis=0)

        start = self.write_pos
        end = start + batch_size

        if end <= self.max_size:
            sl = slice(start, end)
            self.states[sl] = states_arr
            self.actions[sl] = actions_arr
            self.rewards[sl] = rewards_arr
            self.dones[sl] = dones_arr
            self.next_states[sl] = next_states_arr
        else:
            first_len = self.max_size - start
            second_len = end - self.max_size

            first_sl = slice(start, self.max_size)
            self.states[first_sl] = states_arr[:first_len]
            self.actions[first_sl] = actions_arr[:first_len]
            self.rewards[first_sl] = rewards_arr[:first_len]
            self.dones[first_sl] = dones_arr[:first_len]
            self.next_states[first_sl] = next_states_arr[:first_len]

            second_sl = slice(0, second_len)
            self.states[second_sl] = states_arr[first_len:]
            self.actions[second_sl] = actions_arr[first_len:]
            self.rewards[second_sl] = rewards_arr[first_len:]
            self.dones[second_sl] = dones_arr[first_len:]
            self.next_states[second_sl] = next_states_arr[first_len:]

        self.write_pos = end % self.max_size
        self.count = min(self.count + batch_size, self.max_size)

    def add_episode(self, experiences):
        self.add_batch(experiences)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count == 0 or not self.initialized:
            return None

        buf_len = self.count
        actual_batch_size = min(buf_len, batch_size)

        if buf_len <= batch_size:
            indices = np.arange(buf_len, dtype=np.int64)
        else:
            recent_size = max(1, int(buf_len * self.recent_buffer_ratio))

            recent_batch_size = int(actual_batch_size * self.recent_batch_ratio)
            recent_batch_size = max(0, min(recent_batch_size, actual_batch_size))
            old_batch_size = actual_batch_size - recent_batch_size

            if recent_batch_size > 0:
                if self.count < self.max_size:
                    recent_start_idx = buf_len - recent_size
                    recent_indices = np.arange(recent_start_idx, buf_len, dtype=np.int64)
                else:
                    recent_start_pos = (self.write_pos - recent_size) % self.max_size
                    if recent_start_pos < self.write_pos:
                        recent_indices = np.arange(recent_start_pos, self.write_pos, dtype=np.int64)
                    else:
                        recent_indices = np.concatenate(
                            [np.arange(recent_start_pos, self.max_size, dtype=np.int64), np.arange(0, self.write_pos, dtype=np.int64)]
                        )
                if len(recent_indices) >= recent_batch_size:
                    recent_selected = np.random.choice(recent_indices, recent_batch_size, replace=False)
                else:
                    recent_selected = recent_indices
            else:
                recent_selected = np.empty((0,), dtype=np.int64)

            if old_batch_size > 0:
                if self.count < self.max_size:
                    recent_start_idx = buf_len - recent_size
                    old_indices = np.arange(0, recent_start_idx, dtype=np.int64)
                else:
                    recent_start_pos = (self.write_pos - recent_size) % self.max_size
                    if recent_start_pos < self.write_pos:
                        if recent_start_pos > 0:
                            old_indices = np.concatenate(
                                [np.arange(self.write_pos, self.max_size, dtype=np.int64), np.arange(0, recent_start_pos, dtype=np.int64)]
                            )
                        else:
                            old_indices = np.arange(self.write_pos, self.max_size, dtype=np.int64)
                    else:
                        old_indices = np.arange(self.write_pos, recent_start_pos, dtype=np.int64)

                if len(old_indices) >= old_batch_size:
                    old_selected = np.random.choice(old_indices, old_batch_size, replace=False)
                else:
                    old_selected = old_indices
            else:
                old_selected = np.empty((0,), dtype=np.int64)

            indices = np.concatenate([recent_selected, old_selected])
            np.random.shuffle(indices)

        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]
        next_states = self.next_states[indices]

        return states, actions, rewards, dones, next_states

    def clear(self):
        self.count = 0
        self.write_pos = 0
        self.initialized = False
        self.states = None
        self.actions = None
        self.rewards = None
        self.dones = None
        self.next_states = None


class StratifiedReplayBuffer:
    def __init__(
        self,
        buffer_size,
        random_seed=123,
        recent_buffer_ratio=0.1,
        recent_batch_ratio=0.3,
        stratified_sampling=None,
    ):
        self.goal_buffer = ReplayBuffer(
            buffer_size=buffer_size,
            random_seed=random_seed,
            recent_buffer_ratio=recent_buffer_ratio,
            recent_batch_ratio=recent_batch_ratio,
        )
        self.collision_buffer = ReplayBuffer(
            buffer_size=buffer_size,
            random_seed=random_seed + 1,
            recent_buffer_ratio=recent_buffer_ratio,
            recent_batch_ratio=recent_batch_ratio,
        )
        self.timeout_buffer = ReplayBuffer(
            buffer_size=buffer_size,
            random_seed=random_seed + 2,
            recent_buffer_ratio=recent_buffer_ratio,
            recent_batch_ratio=recent_batch_ratio,
        )

        self.stratified_sampling = stratified_sampling or {}

    def add_episode(self, experiences, outcome):
        if outcome == "Goal":
            self.goal_buffer.add_episode(experiences)
        elif outcome == "Collision":
            self.collision_buffer.add_episode(experiences)
        else:
            self.timeout_buffer.add_episode(experiences)

    def size(self):
        return self.goal_buffer.size() + self.collision_buffer.size() + self.timeout_buffer.size()

    def _compute_counts(self, batch_size, stats):
        goal_cfg = self.stratified_sampling.get("goal", {})
        coll_cfg = self.stratified_sampling.get("collision", {})
        min_timeout_prop = float(self.stratified_sampling.get("min_timeout_prop", 0.0))

        p_goal = float(stats.get("goal_rate", 0.0) if stats is not None else 0.0)
        p_coll = float(stats.get("collision_rate", 0.0) if stats is not None else 0.0)

        goal_boost = float(goal_cfg.get("boost", 0.0))
        goal_max = float(goal_cfg.get("max_prop", 1.0))
        coll_boost = float(coll_cfg.get("boost", 0.0))
        coll_max = float(coll_cfg.get("max_prop", 1.0))

        target_goal = min(max(p_goal + goal_boost, 0.0), goal_max)
        target_coll = min(max(p_coll + coll_boost, 0.0), coll_max)

        n_goal = int(math.floor(batch_size * target_goal))
        n_coll = int(math.floor(batch_size * target_coll))

        max_non_timeout = int(math.floor(batch_size * (1.0 - min_timeout_prop)))
        if n_goal + n_coll > max_non_timeout and (n_goal + n_coll) > 0:
            scale = max_non_timeout / float(n_goal + n_coll)
            n_goal = int(math.floor(n_goal * scale))
            n_coll = int(math.floor(n_coll * scale))

        n_timeout = batch_size - n_goal - n_coll
        if n_timeout < 0:
            n_timeout = 0

        return n_goal, n_coll, n_timeout

    def sample_batch(self, batch_size, stats=None):
        if self.size() <= 0:
            return None

        min_steps_to_enable = int(self.stratified_sampling.get("min_steps_to_enable", 0))
        if (
            min_steps_to_enable > 0
            and (
                self.goal_buffer.size() < min_steps_to_enable
                or self.collision_buffer.size() < min_steps_to_enable
                or self.timeout_buffer.size() < min_steps_to_enable
            )
        ):
            total = float(self.size())
            if total <= 0:
                return None
            goal_prop = self.goal_buffer.size() / total
            coll_prop = self.collision_buffer.size() / total
            n_goal = int(math.floor(batch_size * goal_prop))
            n_coll = int(math.floor(batch_size * coll_prop))
            n_timeout = batch_size - n_goal - n_coll
        else:
            n_goal, n_coll, n_timeout = self._compute_counts(batch_size, stats)

        parts = []
        for buf, n in ((self.goal_buffer, n_goal), (self.collision_buffer, n_coll), (self.timeout_buffer, n_timeout)):
            if n <= 0:
                continue
            sampled = buf.sample_batch(n)
            if sampled is None:
                continue
            parts.append(sampled)

        if not parts:
            fallback = None
            for buf in (self.timeout_buffer, self.goal_buffer, self.collision_buffer):
                if buf.size() > 0:
                    fallback = buf
                    break
            return fallback.sample_batch(batch_size) if fallback is not None else None

        s_list, a_list, r_list, t_list, s2_list = [], [], [], [], []
        for s, a, r, t, s2 in parts:
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
            t_list.append(t)
            s2_list.append(s2)

        s_batch = np.concatenate(s_list, axis=0)
        a_batch = np.concatenate(a_list, axis=0)
        r_batch = np.concatenate(r_list, axis=0)
        t_batch = np.concatenate(t_list, axis=0)
        s2_batch = np.concatenate(s2_list, axis=0)

        cur = s_batch.shape[0]
        if cur < batch_size:
            missing = batch_size - cur
            for buf in (self.timeout_buffer, self.goal_buffer, self.collision_buffer):
                if missing <= 0:
                    break
                extra = buf.sample_batch(missing)
                if extra is None:
                    continue
                es, ea, er, et, es2 = extra
                s_batch = np.concatenate([s_batch, es], axis=0)
                a_batch = np.concatenate([a_batch, ea], axis=0)
                r_batch = np.concatenate([r_batch, er], axis=0)
                t_batch = np.concatenate([t_batch, et], axis=0)
                s2_batch = np.concatenate([s2_batch, es2], axis=0)
                missing = batch_size - s_batch.shape[0]

        idx = np.arange(s_batch.shape[0])
        np.random.shuffle(idx)
        s_batch = s_batch[idx]
        a_batch = a_batch[idx]
        r_batch = r_batch[idx]
        t_batch = t_batch[idx]
        s2_batch = s2_batch[idx]

        if s_batch.shape[0] > batch_size:
            s_batch = s_batch[:batch_size]
            a_batch = a_batch[:batch_size]
            r_batch = r_batch[:batch_size]
            t_batch = t_batch[:batch_size]
            s2_batch = s2_batch[:batch_size]

        return s_batch, a_batch, r_batch, t_batch, s2_batch


class NumpyStratifiedReplayBuffer:
    def __init__(
        self,
        buffer_size,
        dtype=np.float32,
        recent_buffer_ratio=0.1,
        recent_batch_ratio=0.3,
        stratified_sampling=None,
    ):
        self.goal_buffer = NumpyReplayBuffer(
            buffer_size=buffer_size,
            dtype=dtype,
            recent_buffer_ratio=recent_buffer_ratio,
            recent_batch_ratio=recent_batch_ratio,
        )
        self.collision_buffer = NumpyReplayBuffer(
            buffer_size=buffer_size,
            dtype=dtype,
            recent_buffer_ratio=recent_buffer_ratio,
            recent_batch_ratio=recent_batch_ratio,
        )
        self.timeout_buffer = NumpyReplayBuffer(
            buffer_size=buffer_size,
            dtype=dtype,
            recent_buffer_ratio=recent_buffer_ratio,
            recent_batch_ratio=recent_batch_ratio,
        )

        self.stratified_sampling = stratified_sampling or {}

    def add_episode(self, experiences, outcome):
        if outcome == "Goal":
            self.goal_buffer.add_episode(experiences)
        elif outcome == "Collision":
            self.collision_buffer.add_episode(experiences)
        else:
            self.timeout_buffer.add_episode(experiences)

    def add(self, s, a, r, done, s2, outcome=None):
        out = outcome or "Timeout"
        if out == "Goal":
            self.goal_buffer.add(s, a, r, done, s2)
        elif out == "Collision":
            self.collision_buffer.add(s, a, r, done, s2)
        else:
            self.timeout_buffer.add(s, a, r, done, s2)

    def size(self):
        return self.goal_buffer.size() + self.collision_buffer.size() + self.timeout_buffer.size()

    def _compute_counts(self, batch_size, stats):
        goal_cfg = self.stratified_sampling.get("goal", {})
        coll_cfg = self.stratified_sampling.get("collision", {})
        min_timeout_prop = float(self.stratified_sampling.get("min_timeout_prop", 0.0))

        p_goal = float(stats.get("goal_rate", 0.0) if stats is not None else 0.0)
        p_coll = float(stats.get("collision_rate", 0.0) if stats is not None else 0.0)

        goal_boost = float(goal_cfg.get("boost", 0.0))
        goal_max = float(goal_cfg.get("max_prop", 1.0))
        coll_boost = float(coll_cfg.get("boost", 0.0))
        coll_max = float(coll_cfg.get("max_prop", 1.0))

        target_goal = min(max(p_goal + goal_boost, 0.0), goal_max)
        target_coll = min(max(p_coll + coll_boost, 0.0), coll_max)

        n_goal = int(math.floor(batch_size * target_goal))
        n_coll = int(math.floor(batch_size * target_coll))

        max_non_timeout = int(math.floor(batch_size * (1.0 - min_timeout_prop)))
        if n_goal + n_coll > max_non_timeout and (n_goal + n_coll) > 0:
            scale = max_non_timeout / float(n_goal + n_coll)
            n_goal = int(math.floor(n_goal * scale))
            n_coll = int(math.floor(n_coll * scale))

        n_timeout = batch_size - n_goal - n_coll
        if n_timeout < 0:
            n_timeout = 0

        return n_goal, n_coll, n_timeout

    def sample_batch(self, batch_size, stats=None):
        if self.size() <= 0:
            return None

        min_steps_to_enable = int(self.stratified_sampling.get("min_steps_to_enable", 0))
        if (
            min_steps_to_enable > 0
            and (
                self.goal_buffer.size() < min_steps_to_enable
                or self.collision_buffer.size() < min_steps_to_enable
                or self.timeout_buffer.size() < min_steps_to_enable
            )
        ):
            total = float(self.size())
            if total <= 0:
                return None
            goal_prop = self.goal_buffer.size() / total
            coll_prop = self.collision_buffer.size() / total
            n_goal = int(math.floor(batch_size * goal_prop))
            n_coll = int(math.floor(batch_size * coll_prop))
            n_timeout = batch_size - n_goal - n_coll
        else:
            n_goal, n_coll, n_timeout = self._compute_counts(batch_size, stats)

        parts = []
        for buf, n in ((self.goal_buffer, n_goal), (self.collision_buffer, n_coll), (self.timeout_buffer, n_timeout)):
            if n <= 0:
                continue
            sampled = buf.sample_batch(n)
            if sampled is None:
                continue
            parts.append(sampled)

        if not parts:
            fallback = None
            for buf in (self.timeout_buffer, self.goal_buffer, self.collision_buffer):
                if buf.size() > 0:
                    fallback = buf
                    break
            return fallback.sample_batch(batch_size) if fallback is not None else None

        s_list, a_list, r_list, t_list, s2_list = [], [], [], [], []
        for s, a, r, t, s2 in parts:
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
            t_list.append(t)
            s2_list.append(s2)

        s_batch = np.concatenate(s_list, axis=0)
        a_batch = np.concatenate(a_list, axis=0)
        r_batch = np.concatenate(r_list, axis=0)
        t_batch = np.concatenate(t_list, axis=0)
        s2_batch = np.concatenate(s2_list, axis=0)

        cur = s_batch.shape[0]
        if cur < batch_size:
            missing = batch_size - cur
            for buf in (self.timeout_buffer, self.goal_buffer, self.collision_buffer):
                if missing <= 0:
                    break
                extra = buf.sample_batch(missing)
                if extra is None:
                    continue
                es, ea, er, et, es2 = extra
                s_batch = np.concatenate([s_batch, es], axis=0)
                a_batch = np.concatenate([a_batch, ea], axis=0)
                r_batch = np.concatenate([r_batch, er], axis=0)
                t_batch = np.concatenate([t_batch, et], axis=0)
                s2_batch = np.concatenate([s2_batch, es2], axis=0)
                missing = batch_size - s_batch.shape[0]

        idx = np.arange(s_batch.shape[0])
        np.random.shuffle(idx)
        s_batch = s_batch[idx]
        a_batch = a_batch[idx]
        r_batch = r_batch[idx]
        t_batch = t_batch[idx]
        s2_batch = s2_batch[idx]

        if s_batch.shape[0] > batch_size:
            s_batch = s_batch[:batch_size]
            a_batch = a_batch[:batch_size]
            r_batch = r_batch[:batch_size]
            t_batch = t_batch[:batch_size]
            s2_batch = s2_batch[:batch_size]

        return s_batch, a_batch, r_batch, t_batch, s2_batch
