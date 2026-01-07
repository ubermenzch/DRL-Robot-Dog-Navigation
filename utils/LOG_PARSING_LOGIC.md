# 日志解析逻辑说明

## 整体解析流程

### 主解析方法：`parse_log()`

采用**单次遍历**的方式，逐行解析日志文件，按优先级顺序匹配不同类型的记录：

```
1. 读取整个日志文件到内存
2. 逐行遍历（使用索引 i）
3. 按优先级匹配：
   a) 训练记录（最高优先级）
   b) 最好模型保存记录
   c) Episode记录
4. 解析完成后对所有记录按训练步数/Episode编号排序
```

## 1. 训练记录解析逻辑

### 1.1 训练记录识别

**触发条件**：匹配到 `"第X次训练完成"` 模式

**日志格式示例**：
```
2026-01-07 11:49:48 第1次训练完成 | 当前缓冲区大小: 71
  总抽样数: 3200 | 总样本数: 171 | 样本平均抽样次数: 18.71
  本次训练的平均critic网络损失: 3.910649 | critic全局参数梯度L2范数(裁剪前:131.983139, 裁剪后:131.983139)
  本次训练的平均actor网络损失: 5.472245 | actor全局参数梯度L2范数(裁剪前:6.433953, 裁剪后:6.433953)
  熵值统计: | 熵值: -11.051837 | alpha梯度L2范数: 0.909522
  训练耗时: 2.71秒
```

### 1.2 解析步骤

1. **识别训练开始** (`parse_training_record`)
   - 正则：`r'第(\d+)次训练完成'`
   - 提取：训练步数（training_step）

2. **解析训练详细信息** (`parse_training_details`)
   - 从 `i+1` 行开始解析（训练完成行的下一行）
   - 需要至少3行数据：
     - **第1行** (`i+1`)：总抽样数、总样本数、样本平均抽样次数
     - **第2行** (`i+2`)：Critic损失和梯度信息
     - **第3行** (`i+3`)：Actor损失和梯度信息
     - **第4行** (`i+4`)：熵值统计（可选）
     - **第5行** (`i+5`)：训练耗时（可能在熵值统计行之后）

3. **提取的数据字段**：
   ```python
   training_records = [
       (training_step,           # 训练步数
        critic_loss,             # Critic损失
        actor_loss,              # Actor损失
        avg_sample_times,        # 平均抽样次数
        critic_grad_before,      # Critic梯度（裁剪前）
        critic_grad_after,       # Critic梯度（裁剪后）
        actor_grad_before,       # Actor梯度（裁剪前）
        actor_grad_after,        # Actor梯度（裁剪后）
        entropy,                 # 熵值
        alpha_grad)              # Alpha梯度L2范数
   ]
   ```

4. **额外存储的数据**：
   - `training_durations[]`：每次训练的耗时（秒）
   - `total_sample_count[]`：每次训练的总抽样数（累加值）
   - `total_sample_steps[]`：每次训练的总样本数

5. **跳行逻辑**：
   - 如果解析成功：
     - 有熵值统计：跳过5行（训练完成行 + 4行详细信息）
     - 无熵值统计：跳过4行（训练完成行 + 3行详细信息）
   - 如果解析失败（格式不完整）：
     - 只记录训练步数，使用默认值填充其他字段
     - 只跳过1行（训练完成行），继续解析

### 1.3 正则表达式模式

```python
regex_training_step = r'第(\d+)次训练完成'
regex_sample_times = r'样本平均抽样次数:\s+([\d.]+)'
regex_critic_loss = r'本次训练的平均critic网络损失:\s+([-\d.]+)'
regex_actor_loss = r'本次训练的平均actor网络损失:\s+([-\d.]+)'
regex_critic_grad = r'critic全局参数梯度L2范数\(裁剪前:([\d.]+),\s*裁剪后:([\d.]+)\)'
regex_actor_grad = r'actor梯度\(裁剪前:([\d.]+),\s*裁剪后:([\d.]+)\)'
regex_entropy = r'熵值:\s+([-\d.]+)'
regex_alpha_grad = r'alpha梯度L2范数:\s+([\d.]+)'
regex_training_duration = r'训练耗时:\s+([\d.]+)秒'
regex_total_sample_count = r'总抽样数:\s+(\d+)'
regex_total_sample_steps = r'总样本数:\s+(\d+)'
```

## 2. Episode记录解析逻辑

### 2.1 Episode记录识别

**触发条件**：行中包含 `"Reward Detail:"` 或 `"Episode:"`

**日志格式示例**：
```
2026-01-07 11:49:47 环境 1 Episode: 2 Target Distance: 1.00 (actual: 1.20) Steps: 100 Queue(episodes): 1
  Reward Detail: end=Timeout, total_reward=-3.054887, goal=0.000000, collision=0.000000, angle=-0.857238, linear=-0.929714, target_distance=-1.267935
```

### 2.2 解析步骤

1. **识别Reward Detail行**
   - 当前行包含 `"Reward Detail:"`
   - 从上一行（`prev_line`）提取Episode基本信息

2. **从上一行提取**：
   - 环境ID：`环境 (\d+)`
   - Episode编号：`Episode: (\d+)`
   - Steps：`Steps: (\d+)`
   - 时间戳：`^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})`

3. **从当前行提取**：
   - 结束状态：`end=(Goal|Collision|Timeout)`
   - Reward详情：使用多个正则表达式提取各项reward值
     - goal, collision, angle, linear, target_distance, obs, yawrate

4. **存储的数据结构**：
   ```python
   episodes = [
       (episode_num,      # Episode编号
        env_id,           # 环境ID
        end_status,       # 结束状态（Goal/Collision/Timeout）
        reward_detail,    # Reward详情字典
        steps,            # Episode步数
        timestamp)        # 时间戳（datetime对象）
   ]
   ```

5. **向后兼容**：
   - 如果Reward Detail格式解析失败，尝试匹配旧格式
   - 旧格式：`环境 X Episode: Y ... End: (Goal|Collision|Timeout)`

### 2.3 Reward Detail解析

使用预编译的正则表达式提取各项reward值：
```python
reward_patterns = {
    'goal': r'goal=(-?\d+\.?\d*)',
    'collision': r'collision=(-?\d+\.?\d*)',
    'angle': r'angle=(-?\d+\.?\d*)',
    'linear': r'linear=(-?\d+\.?\d*)',
    'target_distance': r'target_distance=(-?\d+\.?\d*)',
    'obs': r'obs=(-?\d+\.?\d*)',
    'yawrate': r'yawrate=(-?\d+\.?\d*)',
}
```

如果某项未找到，默认值为 `0.0`。

## 3. 最好模型保存记录解析逻辑

### 3.1 识别条件

**触发条件**：行中包含 `"============================================================"`（分隔线）

**日志格式示例**：
```
============================================================
保存最好模型到: /path/to/best_model
当前最好统计: 成功率=0.1270 (12.70%), 碰撞率=0.2910 (29.10%)
============================================================
```

### 3.2 解析步骤

1. **检查格式**：
   - 第 `i` 行：分隔线
   - 第 `i+1` 行：包含 `"保存最好模型到:"`
   - 第 `i+2` 行：包含成功率和碰撞率
   - 第 `i+3` 行：分隔线

2. **提取信息**：
   - 保存路径：从 `i+1` 行提取
   - 成功率：从 `i+2` 行提取 `成功率=([\d.]+)`
   - 碰撞率：从 `i+2` 行提取 `碰撞率=([\d.]+)`
   - Episode编号：从最近的episode记录推断

3. **存储的数据结构**：
   ```python
   best_model_records = [
       (episode_num,      # Episode编号
        success_rate,     # 成功率
        collision_rate,   # 碰撞率
        save_path)        # 模型保存路径
   ]
   ```

4. **跟踪最高成功率**：
   - 每次解析时更新 `best_success_rate` 和 `best_model_info`

5. **跳行逻辑**：跳过4行（包括分隔线和空行）

## 4. 解析优先级和跳行机制

### 4.1 优先级顺序

```
1. 训练记录（最高优先级）
   ↓ 如果匹配，解析并跳过多行，continue
   
2. 最好模型保存记录
   ↓ 如果匹配，解析并跳过多行，continue
   
3. Episode记录
   ↓ 如果匹配，添加到列表，继续下一行
   
4. 其他行
   ↓ 更新 prev_line，继续下一行
```

### 4.2 跳行机制

- **训练记录**：
  - 成功解析：跳过4-5行（取决于是否有熵值统计）
  - 失败解析：只跳过1行（训练完成行）
  
- **最好模型记录**：跳过4行

- **Episode记录**：不跳行，继续逐行解析

- **其他行**：正常递增 `i += 1`

### 4.3 关键变量

- `i`：当前行索引
- `prev_line`：上一行的内容（用于Episode解析）
- `current_training_step`：当前训练步数（用于关联数据）

## 5. 数据存储结构

### 5.1 训练记录

```python
self.training_records = [
    (training_step, critic_loss, actor_loss, avg_sample_times,
     critic_grad_before, critic_grad_after, actor_grad_before, actor_grad_after,
     entropy, alpha_grad)
]

self.training_durations = [duration1, duration2, ...]  # 秒
self.total_sample_count = [count1, count2, ...]        # 累加的总抽样数
self.total_sample_steps = [steps1, steps2, ...]        # 总样本数
```

### 5.2 Episode记录

```python
self.episodes = [
    (episode_num, env_id, end_status, reward_detail, steps, timestamp)
]
# reward_detail = {
#     'goal': float,
#     'collision': float,
#     'angle': float,
#     'linear': float,
#     'target_distance': float,
#     'obs': float,
#     'yawrate': float
# }
```

### 5.3 最好模型记录

```python
self.best_model_records = [
    (episode_num, success_rate, collision_rate, save_path)
]

self.best_success_rate = float  # 最高成功率
self.best_model_info = (episode_num, success_rate, collision_rate, save_path)
```

## 6. 解析后的处理

### 6.1 排序

所有记录解析完成后，按关键字段排序：
- `training_records`：按 `training_step` 排序
- `episodes`：按 `episode_num` 排序
- `best_model_records`：按 `episode_num` 排序

### 6.2 统计输出

解析完成后输出统计信息：
- 成功解析的训练记录数
- 成功解析的episode数
- 成功解析的最好模型保存记录数

## 7. 特殊处理

### 7.1 格式不完整的训练记录

如果训练记录格式不完整（缺少详细信息）：
- 仍然记录训练步数
- 其他字段使用默认值（0.0 或 None）
- 确保训练次数统计的完整性

### 7.2 时间戳解析

- 从Episode行提取时间戳
- 格式：`YYYY-MM-DD HH:MM:SS`
- 转换为 `datetime` 对象
- 用于计算时间跨度统计

### 7.3 向后兼容

- Episode解析支持旧格式
- 如果新格式解析失败，尝试旧格式
- 旧格式缺少的信息使用默认值

## 8. 性能优化

1. **预编译正则表达式**：所有正则表达式在初始化时预编译
2. **单次遍历**：只遍历日志文件一次
3. **索引访问**：使用列表索引而非迭代器
4. **提前跳过**：匹配成功后立即跳过多行，避免重复检查
5. **向量化排序**：使用numpy进行排序（对于大数据集）


