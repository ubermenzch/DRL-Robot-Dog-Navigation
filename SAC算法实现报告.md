# SAC算法实现报告

## 1. 项目概述

本项目基于Soft Actor-Critic (SAC)算法实现了机器人导航的深度强化学习系统。SAC是一种基于最大熵强化学习的off-policy算法，适用于连续动作空间的控制任务。本项目将SAC算法应用于ROS2环境下的TurtleBot3机器人导航任务，支持单环境和多环境并行训练两种模式。

## 2. SAC算法原理

### 2.1 核心思想

SAC算法的核心思想是在最大化累积奖励的同时，最大化策略的熵（entropy），从而在探索和利用之间取得平衡。SAC使用以下优化目标：

\[
\pi^* = \arg\max_\pi \mathbb{E}_{\tau \sim \pi} \left[ \sum_t r(s_t, a_t) + \alpha H(\pi(\cdot|s_t)) \right]
\]

其中：
- \(r(s_t, a_t)\) 是即时奖励
- \(H(\pi(\cdot|s_t))\) 是策略熵
- \(\alpha\) 是温度参数，用于平衡奖励最大化和熵最大化

### 2.2 算法特点

- **Off-policy算法**：可以使用历史经验数据，提高样本效率
- **连续动作空间**：使用重参数化技巧（reparameterization trick）处理连续动作
- **自动熵调节**：温度参数\(\alpha\)可以自动学习，无需手动调节
- **双Q网络**：使用两个Q网络（Q1和Q2）取最小值，减少过估计问题

## 3. 代码实现架构

### 3.1 项目结构

项目的SAC实现主要包含以下核心文件：

```
src/drl_navigation_ros2/SAC/
├── SAC.py              # SAC算法主类
├── SAC_actor.py        # Actor网络实现
├── SAC_critic.py       # Critic网络实现
└── SAC_utils.py        # 工具函数
```

### 3.2 整体架构

SAC算法实现采用模块化设计：

1. **Actor网络（DiagGaussianActor）**：输出动作的均值和方差，使用SquashedNormal分布
2. **Critic网络（DoubleQCritic）**：双Q网络结构，输出Q1和Q2值
3. **SAC主类**：整合Actor和Critic，实现训练逻辑
4. **经验回放缓冲区**：存储和采样训练数据

## 4. 核心组件实现

### 4.1 Actor网络实现

Actor网络位于`SAC_actor.py`文件中，实现了对角高斯策略（DiagGaussianActor）。

#### 4.1.1 网络结构

```python
class DiagGaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds):
        self.trunk = utils.mlp(obs_dim, hidden_dim, 2 * action_dim, hidden_depth)
```

网络输出维度为`2 * action_dim`，分别表示动作的均值（mu）和标准差的对数（log_std）。

#### 4.1.2 SquashedNormal分布

实现中使用`SquashedNormal`分布，这是经过Tanh变换的正态分布，确保动作被限制在[-1, 1]范围内：

```python
class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)
```

Tanh变换的雅可比行列式计算考虑了数值稳定性：

```python
def log_abs_det_jacobian(self, x, y):
    return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))
```

#### 4.1.3 标准差约束

标准差的对数被约束在指定范围内，避免策略过早收敛：

```python
log_std = torch.tanh(log_std)
log_std_min, log_std_max = self.log_std_bounds
log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
```

### 4.2 Critic网络实现

Critic网络位于`SAC_critic.py`文件中，采用双Q网络结构。

#### 4.2.1 网络结构

```python
class DoubleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
```

两个Q网络结构相同，但参数独立，输入为状态和动作的拼接。

#### 4.2.2 前向传播

```python
def forward(self, obs, action):
    obs_action = torch.cat([obs, action], dim=-1)
    q1 = self.Q1(obs_action)
    q2 = self.Q2(obs_action)
    return q1, q2
```

### 4.3 SAC主类实现

SAC主类位于`SAC.py`文件中，整合了所有组件并实现训练逻辑。

#### 4.3.1 初始化

初始化时创建Actor、Critic和Critic目标网络：

```python
def __init__(self, state_dim, action_dim, device, max_action, ...):
    self.critic = critic_model(...).to(self.device)
    self.critic_target = critic_model(...).to(self.device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.actor = actor_model(...).to(self.device)
    
    self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
    self.log_alpha.requires_grad = True
    self.target_entropy = -action_dim
```

关键参数：
- `target_entropy`：目标熵值，设为`-action_dim`
- `log_alpha`：温度参数的对数，可训练
- `critic_target`：Critic目标网络，使用软更新

#### 4.3.2 Critic更新

Critic网络使用以下损失函数：

```python
def update_critic(self, obs, action, reward, next_obs, done, step):
    dist = self.actor(next_obs)
    next_action = dist.rsample()
    log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
    target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
    target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
    target_Q = reward + ((1 - done) * self.discount * target_V)
    
    current_Q1, current_Q2 = self.critic(obs, action)
    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
```

目标Q值计算包含熵项：
\[
Q_{target} = r + \gamma (1 - done) \left( \min(Q_1(s', a'), Q_2(s', a')) - \alpha \log \pi(a'|s') \right)
\]

#### 4.3.3 Actor更新

Actor网络优化以下目标：

```python
def update_actor_and_alpha(self, obs, step):
    dist = self.actor(obs)
    action = dist.rsample()
    log_prob = dist.log_prob(action).sum(-1, keepdim=True)
    actor_Q1, actor_Q2 = self.critic(obs, action)
    actor_Q = torch.min(actor_Q1, actor_Q2)
    actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
```

Actor损失函数：
\[
J(\pi) = \mathbb{E}_{s \sim D, a \sim \pi} \left[ \alpha \log \pi(a|s) - Q(s, a) \right]
\]

#### 4.3.4 温度参数更新

如果启用可学习的温度参数，同时更新\(\alpha\)：

```python
if self.learnable_temperature:
    alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
    alpha_loss.backward()
    self.log_alpha_optimizer.step()
```

温度参数的损失函数：
\[
J(\alpha) = \mathbb{E}_{a \sim \pi} \left[ -\alpha (\log \pi(a|s) + H_{target}) \right]
\]

#### 4.3.5 目标网络软更新

使用软更新（soft update）更新目标网络：

```python
if step % self.critic_target_update_frequency == 0:
    utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)
```

软更新公式：
\[
\theta_{target} \leftarrow \tau \theta + (1 - \tau) \theta_{target}
\]

其中\(\tau\)通常取0.005。

### 4.4 经验回放缓冲区

项目实现了两种经验回放缓冲区：

#### 4.4.1 单环境版本（ReplayBuffer）

位于`replay_buffer.py`，使用deque实现：

```python
class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        self.buffer_size = buffer_size
        self.buffer = deque()
    
    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
```

#### 4.4.2 多环境版本（SharedReplayBuffer和LocalReplayBuffer）

位于`multi_env_train.py`，支持多进程并行训练：

- **SharedReplayBuffer**：使用multiprocessing.Manager().list()实现跨进程共享
- **LocalReplayBuffer**：使用NumPy数组实现高性能本地缓冲区

LocalReplayBuffer使用连续内存存储，提升采样效率：

```python
class LocalReplayBuffer:
    def __init__(self, max_size: int, dtype=np.float32):
        self.states = np.zeros((max_size, state_dim), dtype=self.dtype)
        self.actions = np.zeros((max_size, action_dim), dtype=self.dtype)
        self.rewards = np.zeros((max_size, 1), dtype=self.dtype)
        self.dones = np.zeros((max_size, 1), dtype=self.dtype)
        self.next_states = np.zeros((max_size, state_dim), dtype=self.dtype)
```

### 4.5 状态处理

SAC类实现了`prepare_state`方法，将ROS环境的状态转换为神经网络输入：

```python
def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action):
    latest_scan = np.array(latest_scan)
    
    # 将激光扫描数据分箱处理
    scan_len = len(latest_scan)
    max_bins = self.base_state_dim - 5
    bin_size = int(np.ceil(scan_len / max_bins))
    
    min_obs_distance = []
    for i in range(0, scan_len, bin_size):
        bin = latest_scan[i : i + min(bin_size, scan_len - i)]
        min_obs_distance.append(min(bin))
    
    state = min_obs_distance + [distance, cos, sin] + [action[0], action[1]]
    return state, terminal
```

状态向量包括：
- 激光扫描数据的最小值（分箱处理）
- 到目标的距离
- 目标方向的余弦和正弦值
- 上一时刻的动作（线速度和角速度）

## 5. 训练流程

### 5.1 单环境训练流程

1. 初始化SAC算法和环境
2. 在每个episode中：
   - 重置环境
   - 执行动作（探索或确定性）
   - 收集经验（state, action, reward, next_state, done）
   - 将经验存入缓冲区
   - 定期从缓冲区采样批次进行训练
3. 训练时：
   - 更新Critic网络
   - 按频率更新Actor网络
   - 按频率更新Critic目标网络
   - 按频率更新温度参数

### 5.2 多环境并行训练流程

多环境训练采用数据收集和模型训练并行的架构：

1. **数据收集进程**：多个环境并行运行，使用最新模型收集数据
2. **训练线程**：监控缓冲区，当数据足够时触发训练
3. **模型共享**：使用SharedModelManager实现跨进程模型权重共享
4. **训练循环**：
   - 从共享缓冲区批量拉取数据到本地缓冲区
   - 执行多次迭代训练
   - 更新共享模型权重

### 5.3 训练方法

```python
def train(self, replay_buffer, iterations, batch_size):
    critic_losses = []
    actor_losses = []
    
    for _ in range(iterations):
        critic_loss, actor_loss, sample_dt, total_dt = self.update(
            replay_buffer=replay_buffer, step=self.step, batch_size=batch_size
        )
        if critic_loss is not None:
            critic_losses.append(critic_loss)
        if actor_loss is not None:
            actor_losses.append(actor_loss)
    
    self.step += 1
    avg_critic_loss = sum(critic_losses) / len(critic_losses) if critic_losses else 0.0
    avg_actor_loss = sum(actor_losses) / len(actor_losses) if actor_losses else None
    
    return avg_critic_loss, critic_losses, avg_actor_loss, actor_losses
```

## 6. 关键特性

### 6.1 自动熵调节

SAC实现支持可学习的温度参数，自动平衡探索和利用。温度参数的更新频率与Actor网络相同。

### 6.2 双Q网络

使用两个独立的Q网络，取最小值作为目标Q值，减少过估计问题，提高训练稳定性。

### 6.3 重参数化技巧

Actor网络使用`rsample()`方法（重参数化采样），使得梯度可以通过随机性部分反向传播，实现策略梯度的有效计算。

### 6.4 软更新机制

Critic目标网络使用软更新而非硬复制，使训练过程更加稳定。

### 6.5 状态历史支持

支持包含历史状态信息，通过`state_history_steps`参数控制。当启用历史状态时，状态维度会动态扩展为`base_state_dim * (1 + state_history_steps)`。

### 6.6 多环境并行训练

实现了真正并行化的多环境训练架构，显著提升数据收集效率。

## 7. 配置参数说明

主要配置参数位于`config/train.yaml`文件中：

### 7.1 模型参数

- `action_dim`: 动作维度（2：线速度和角速度）
- `state_dim`: 状态维度（基础为25）
- `base_state_dim`: 基础状态维度（25）
- `state_history_steps`: 历史状态步数（0表示不使用历史）
- `hidden_dim`: 隐藏层维度（默认1024）
- `hidden_depth`: 隐藏层深度（默认3）
- `max_action`: 动作最大值（默认1）

### 7.2 训练算法参数

- `actor_update_frequency`: Actor更新频率（默认1）
- `critic_target_update_frequency`: Critic目标网络更新频率（默认2）
- `training_iterations`: 每次训练的迭代次数（默认100）
- `batch_size`: 批次大小（默认4096）
- `buffer_size`: 经验回放缓冲区大小（默认50000）
- `discount`: 折扣因子（默认0.99，在代码中硬编码）
- `critic_tau`: 软更新系数（默认0.005，在代码中硬编码）

### 7.3 优化器参数

- `actor_lr`: Actor学习率（默认1e-4，在代码中硬编码）
- `critic_lr`: Critic学习率（默认1e-4，在代码中硬编码）
- `alpha_lr`: 温度参数学习率（默认1e-4，在代码中硬编码）

### 7.4 温度参数

- `init_temperature`: 初始温度值（默认0.1，在代码中硬编码）
- `learnable_temperature`: 是否可学习（默认True，在代码中硬编码）

## 8. 性能优化

### 8.1 本地缓冲区优化

多环境训练中使用LocalReplayBuffer，采用NumPy连续内存存储，避免Python列表的逐个索引开销。

### 8.2 批量采样优化

采样时使用NumPy的向量化操作，一次性完成索引，避免Python循环。

### 8.3 模型共享优化

使用临时文件实现跨进程模型权重共享，避免序列化开销，并通过文件原子替换确保一致性。

### 8.4 训练耗时统计

SAC类记录了采样和更新的耗时，便于性能分析和优化：

```python
self.last_sample_time = sample_time_total
self.last_update_time = update_time_total
```

## 9. 地图生成机制

地图生成是机器人导航训练环境的重要组成部分，本项目实现了基于costmap的智能地图生成系统，确保生成的训练环境具有合理的连通性和可导航性。

### 9.1 地图生成流程

地图生成采用三步流程：

1. **障碍物生成**：根据配置生成指定数量的障碍物，支持均匀分布和随机分布两种模式
2. **Costmap构建**：基于障碍物位置构建栅格代价地图，标记占用和自由区域
3. **位置采样**：在连通自由区域内采样机器人和目标位置，确保路径可达性

### 9.2 障碍物生成策略

#### 9.2.1 均匀分布模式（uniform）

使用改进的Farthest Point采样算法实现障碍物的均匀分布：

```python
def set_random_position(self, name):
    # 生成均匀网格候选点
    if not hasattr(self, 'candidate_points'):
        self.generate_uniform_candidates(bias)
    
    # 选择离已有障碍物最远的点
    if self.element_positions:
        existing_points = np.array(self.element_positions)
        candidate_array = np.array(self.candidate_points)
        dists = distance.cdist(candidate_array, existing_points).min(axis=1)
        selected_idx = np.argmax(dists)
        x, y = self.candidate_points.pop(selected_idx)
```

特点：
- 生成约300个均匀网格候选点
- 添加随机扰动避免网格对齐
- 每次选择距离已有障碍物最远的候选点
- 保证障碍物之间的最小距离约束（`obs_min_dist`）

#### 9.2.2 随机分布模式（random）

直接在世界范围内随机采样障碍物位置：

```python
def fallback_random_position(self, name, bias, angle):
    x = np.random.uniform(-bias, bias)
    y = np.random.uniform(-bias, bias)
    if self.check_position(x, y, self.obs_min_dist):
        return True
```

### 9.3 Costmap构建

Costmap是栅格化的代价地图，用于表示环境的占用状态。

#### 9.3.1 Costmap初始化

```python
def _init_costmap(self):
    self.costmap = np.zeros((self.grid_height, self.grid_width), dtype=bool)
```

Costmap参数：
- `costmap_resolution`：栅格分辨率（米/格），默认1.0米
- `grid_width` 和 `grid_height`：根据`world_size`和分辨率计算
- 地图原点：`(-world_size/2, -world_size/2)`

#### 9.3.2 障碍物标记

采用"任意相交即占用"的规则标记障碍物：

```python
def _mark_obstacle_on_costmap(self, cx, cy):
    half = self.obstacle_size / 2.0
    x_min, x_max = cx - half, cx + half
    y_min, y_max = cy - half, cy + half
    
    # 计算障碍物覆盖的栅格索引范围
    ix_min = int(np.floor((x_min - self.map_origin_x) / res))
    ix_max = int(np.floor((x_max - self.map_origin_x) / res))
    iy_min = int(np.floor((y_min - self.map_origin_y) / res))
    iy_max = int(np.floor((y_max - self.map_origin_y) / res))
    
    # 批量标记占用
    self.costmap[iy_min:iy_max+1, ix_min:ix_max+1] = True
```

规则：
- 只要障碍物的外接正方形与某个栅格有任意面积重叠，该栅格即视为占用
- 使用向量化操作批量标记，提高效率

#### 9.3.3 Costmap构建

```python
def _build_costmap_from_obstacles(self):
    self._init_costmap()
    for pos in self.obstacle_positions:
        self._mark_obstacle_on_costmap(pos[0], pos[1])
```

### 9.4 连通区域查找

使用广度优先搜索（BFS）算法查找自由区域的连通分量：

```python
def _find_largest_free_region(self):
    free = ~self.costmap
    visited = np.zeros_like(free, dtype=bool)
    regions = []  # 存储 (mask, size)
    
    # BFS遍历所有连通区域
    for iy in range(height):
        for ix in range(width):
            if not free[iy, ix] or visited[iy, ix]:
                continue
            # BFS标记连通区域
            queue = [(iy, ix)]
            visited[iy, ix] = True
            current_cells = []
            while queue:
                cy, cx = queue.pop(0)
                current_cells.append((cy, cx))
                # 4-邻接搜索
                for dy, dx in neighbors:
                    # ... 扩展邻居节点
    
    # 只保留格子数量 >= 2 的连通区域
    valid_regions = [r for r in regions if r[1] >= 2]
```

连通区域选择策略：
- 使用`region_select_bias`参数控制选择概率
- 以概率`region_select_bias`选择最大连通区域
- 否则从剩余区域中随机选择
- 确保至少能放下机器人和目标两个不同格子

### 9.5 机器人和目标位置采样

在选定的连通区域内采样机器人和目标位置：

```python
def _sample_robot_and_target_on_region(self, region_mask):
    # 在区域内随机采样机器人位置
    indices = np.argwhere(region_mask)
    ridx = np.random.randint(len(indices))
    ry, rx = indices[ridx]
    robot_x, robot_y = self._grid_to_world_center(rx, ry)
    
    # 在连通区域内选择符合距离约束且距离最远的目标位置
    min_dist = self.target_reached_delta + 0.1
    max_dist = self.target_dist
    
    for tidx in range(len(indices)):
        ty, tx = indices[tidx]
        target_x_center, target_y_center = self._grid_to_world_center(tx, ty)
        dist = np.linalg.norm([target_x_center - robot_x, target_y_center - robot_y])
        if min_dist <= dist <= max_dist and dist > best_dist:
            best_dist = dist
            best_target_grid = (tx, ty)
    
    # 在最佳格子的范围内随机生成终点位置
    target_x, target_y = self._grid_to_world_random(tx, ty)
    return (robot_x, robot_y), (target_x, target_y)
```

采样策略：
- 机器人位置：在连通区域内随机选择栅格中心
- 目标位置：遍历连通区域，选择距离机器人最远且满足距离约束的格子
- 距离约束：目标距离在`[target_reached_delta + 0.1, target_dist]`范围内
- 目标位置在选定格子的范围内随机生成，增加多样性

### 9.6 地图复用机制

为了提高训练效率，实现了地图复用功能：

```python
def reset(self):
    # 根据goals_per_map参数决定是否重新生成地图
    should_regenerate_map = False
    
    if self.goals_per_map <= 1:
        should_regenerate_map = True
    else:
        if self.goals_count_for_current_map == 0:
            should_regenerate_map = True
        elif self.goals_count_for_current_map >= self.goals_per_map:
            should_regenerate_map = True
    
    if should_regenerate_map:
        # 重新生成地图（包括障碍物、机器人和目标）
        self.world_reset.reset_world()
        self.set_positions()
        self.goals_count_for_current_map = 1
    else:
        # 只重置机器人到最近的空格子中心，并重新生成目标点
        robot_only_success = self.reset_robot_only()
        if robot_only_success:
            self.goals_count_for_current_map += 1
```

地图复用逻辑：
- `goals_per_map`参数控制每张地图的目标点数量
- 当目标点计数达到`goals_per_map`时，重新生成完整地图
- 否则只重置机器人位置和目标位置，保持障碍物不变
- 机器人重置到当前位置最近的空格子中心

### 9.7 配置参数

地图生成相关的主要配置参数：

- `world_size`：世界尺寸（米），默认10米
- `obs_num`：障碍物数量，默认30个
- `obs_min_dist`：障碍物圆心最小距离（米），默认0.5米
- `obs_distribution_mode`：障碍物生成方式，"uniform"或"random"，默认"random"
- `costmap_resolution`：Costmap分辨率（米/格），默认1.0米
- `obstacle_size`：障碍物在costmap中的等效边长（米），默认1.0米
- `goals_per_map`：每张地图的目标点数量，默认4个
- `region_select_bias`：选择最大连通区域的概率，默认0.6
- `target_dist`：初始目标距离，默认6.0米
- `target_reached_delta`：判定到达目标的阈值，默认0.4米

### 9.8 地图生成的优势

1. **保证路径可达性**：通过连通区域分析，确保机器人和目标在同一连通区域内，避免生成不可达的目标
2. **提高训练效率**：地图复用机制减少了频繁重建环境的开销
3. **灵活的障碍物分布**：支持均匀分布和随机分布，适应不同训练需求
4. **可配置性强**：丰富的配置参数支持不同场景的实验需求
5. **鲁棒性**：包含重试机制和失败处理，确保地图生成的稳定性

## 10. 模型保存与加载

### 10.1 保存

```python
def save(self, filename, directory):
    torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
    torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))
    torch.save(self.critic_target.state_dict(), "%s/%s_critic_target.pth" % (directory, filename))
```

### 10.2 加载

```python
def load(self, filename, directory):
    self.actor.load_state_dict(torch.load("%s/%s_actor.pth" % (directory, filename)))
    self.critic.load_state_dict(torch.load("%s/%s_critic.pth" % (directory, filename)))
    self.critic_target.load_state_dict(torch.load("%s/%s_critic_target.pth" % (directory, filename)))
```

## 11. 总结

本项目基于SAC算法实现了完整的机器人导航深度强化学习系统，包含Actor、Critic、目标网络、温度参数等所有核心组件，采用模块化设计使代码结构清晰、易于维护和扩展。针对多环境并行训练场景，实现了高性能优化策略，包括本地缓冲区优化、批量采样优化和模型共享优化等。系统支持丰富的配置参数，能够灵活适应不同训练需求，同时充分考虑了模型保存、加载、状态处理等实际工程应用问题。该实现为机器人导航任务提供了稳定、高效的强化学习解决方案，支持单环境和多环境两种训练模式，能够有效训练机器人导航策略。

