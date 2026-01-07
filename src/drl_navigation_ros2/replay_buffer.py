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
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        self.recent_buffer_ratio = recent_buffer_ratio
        self.recent_batch_ratio = recent_batch_ratio
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            # 数据不足时，使用全部数据
            batch = random.sample(self.buffer, self.count)
        else:
            # 分层采样：batch的recent_batch_ratio比例来自最新的recent_buffer_ratio部分
            # 其余来自除了最新recent_buffer_ratio外的部分
            buffer_list = list(self.buffer)
            total_size = len(buffer_list)
            
            # 计算最新数据的范围（deque右侧是最新数据）
            recent_size = max(1, int(total_size * self.recent_buffer_ratio))
            old_size = total_size - recent_size
            
            # 计算从两部分采样的数量
            recent_batch_size = max(1, int(batch_size * self.recent_batch_ratio))
            old_batch_size = batch_size - recent_batch_size
            
            # 从最新部分采样
            recent_part = buffer_list[-recent_size:]  # 最新的recent_size个
            if len(recent_part) >= recent_batch_size:
                recent_samples = random.sample(recent_part, recent_batch_size)
            else:
                recent_samples = recent_part
            
            # 从旧数据部分采样
            old_part = buffer_list[:-recent_size] if recent_size > 0 else buffer_list
            if len(old_part) >= old_batch_size:
                old_samples = random.sample(old_part, old_batch_size)
            else:
                old_samples = old_part
            
            # 合并两部分样本
            batch = recent_samples + old_samples
            # 随机打乱顺序
            random.shuffle(batch)

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
