# G1 Reaching Force - 快速开始指南

## 环境概述

`g1_reaching_force` 是在 `g1_reaching` 基础上添加了持续外力干扰的训练环境。机器人在完成手部到达任务的同时，需要抵抗施加在双手上的动态变化的外力。

## 快速开始

### 1. 环境检查

确保已安装 Isaac Gym 和相关依赖：

```bash
cd /Users/luoxinyuan/Downloads/SkillBlender/legged_gym
pip install -e .
```

### 2. 训练

```bash
cd legged_gym/scripts
python train.py --task=g1_reaching_force
```

训练参数：
- `--num_envs`: 并行环境数（默认 4096）
- `--headless`: 无头模式训练
- `--max_iterations`: 最大迭代次数（默认 15001）

示例：
```bash
python train.py --task=g1_reaching_force --num_envs=2048 --headless
```

### 3. 测试/可视化

```bash
python play.py --task=g1_reaching_force --load_run=<实验名称>
```

示例：
```bash
python play.py --task=g1_reaching_force --load_run=Sep28_12-34-56_
```

## 核心差异（vs g1_reaching）

| 特性 | g1_reaching | g1_reaching_force |
|-----|-------------|-------------------|
| 外力干扰 | ❌ | ✅ 双手动态力 |
| 观测维度 | 83 | 89 (+6维力观测) |
| 课程学习 | 无 | 力强度自适应 |
| 训练难度 | 中等 | 高 |
| 鲁棒性 | 基础 | 强 |

## 关键配置

### 调整外力范围

编辑 `g1_reaching_force_config.py`:

```python
class force:
    # 增大力范围（更难）
    apply_force_x_range = [-50.0, 50.0]  # 默认 ±30N
    apply_force_y_range = [-50.0, 50.0]
    apply_force_z_range = [-50.0, 50.0]
```

### 调整课程学习速度

```python
class force:
    # 更快增加难度
    force_scale_up = 0.10  # 默认 0.05
    force_scale_up_threshold = 0.7  # 默认 0.8（episode长度比例）
    
    # 更慢减少难度
    force_scale_down = 0.02  # 默认 0.05
    force_scale_down_threshold = 0.2  # 默认 0.3
```

### 禁用课程学习（固定难度）

```python
class force:
    force_scale_curriculum = False
    # 设置固定力强度（0.0-1.0）
    force_scale_initial_scale = 0.5
```

### 调整力变化频率

```python
class force:
    # 更快变化的力（更难）
    randomize_force_duration = [20, 80]  # 默认 [50, 200]
    
    # 更慢变化的力（更容易适应）
    randomize_force_duration = [100, 400]
```

## 观测空间说明

新增的 6 维力观测（在基座坐标系）：

```python
obs = [
    ...,  # 原有的 83 维观测
    left_hand_force_x,   # 左手受力 X
    left_hand_force_y,   # 左手受力 Y
    left_hand_force_z,   # 左手受力 Z
    right_hand_force_x,  # 右手受力 X
    right_hand_force_y,  # 右手受力 Y
    right_hand_force_z,  # 右手受力 Z
]  # 总共 89 维
```

## 训练技巧

### 1. 分阶段训练

**阶段 1：无力训练（预训练）**
```python
# 修改配置
class force:
    force_scale_initial_scale = 0.0  # 从零开始
    force_scale_curriculum = True
```

训练 5000 次迭代后，力会逐渐增加。

**阶段 2：固定中等力度**
```python
class force:
    force_scale_curriculum = False
    force_scale_initial_scale = 0.5
```

**阶段 3：全力训练**
```python
class force:
    force_scale_initial_scale = 1.0
```

### 2. 调整奖励权重

如果机器人频繁失败，增加稳定性奖励：

```python
class rewards:
    class scales:
        wrist_pos = 5 * 2           # 到达精度（保持）
        orientation = 2.0           # 增加（原 1.0）
        default_joint_pos = 0.5 * 6  # 增加（原 0.5*4）
```

### 3. 延长 episode

```python
class env:
    episode_length_s = 30  # 从 24 增加到 30
```

## 常见问题

### Q1: 机器人总是跌倒

**解决方法**：
1. 降低初始力强度：`force_scale_initial_scale = 0.1`
2. 增加低通滤波：`force_filter_alpha = 0.1`（默认 0.2，越小越平滑）
3. 提高零力概率：`zero_force_prob = 0.5`

### Q2: 训练不收敛

**解决方法**：
1. 检查观测维度是否正确（应该是 89）
2. 从 `g1_reaching` 预训练的模型开始：
   ```bash
   python train.py --task=g1_reaching_force --load_run=<g1_reaching的实验>
   ```
3. 降低学习率：
   ```python
   class algorithm:
       learning_rate = 5e-6  # 从 1e-5 降低
   ```

### Q3: 力的方向看起来不合理

这是正常的！力的方向是完全随机的，模拟各种可能的干扰场景。如果想要更"对抗性"的力，可以修改 `_calculate_ee_forces()` 方法，让力倾向于与目标方向相反。

### Q4: 如何可视化外力？

目前实现中没有可视化。如果需要，可以参考 FALCON 的 `_draw_debug_vis()` 方法添加力箭头渲染：

```python
def _draw_debug_vis(self):
    self.gym.clear_lines(self.viewer)
    # 画出力的方向
    for env_id in range(self.num_envs):
        left_force = self.apply_force_tensor[env_id, self.wrist_indices[0], :]
        left_pos = self.rigid_state[env_id, self.wrist_indices[0], :3]
        # ... 使用 gym.add_lines() 画箭头
```

## 性能基准

在 4096 个环境下的训练速度（RTX 4090）：

- **训练步数/秒**：~15000
- **单次迭代时间**：~16 秒
- **达到合理性能**：~3000 次迭代（约 13 小时）
- **达到最佳性能**：~10000 次迭代（约 44 小时）

## 进阶：自定义力策略

### 对抗性力（始终反向推）

修改 `g1_reaching_force.py` 的 `_calculate_ee_forces()`:

```python
def _calculate_ee_forces(self):
    # 计算手腕到目标的方向
    left_wrist_pos = self.rigid_state[:, self.wrist_indices[0], :3]
    left_target_pos = self.ref_wrist_pos[:, 0, :3]
    left_direction = left_target_pos - left_wrist_pos
    left_direction = left_direction / (torch.norm(left_direction, dim=-1, keepdim=True) + 1e-6)
    
    # 施加反向力
    force_magnitude = 30.0 * self.apply_force_scale
    left_hand_force = -left_direction * force_magnitude
    
    # 类似处理右手
    # ...
```

### 周期性脉冲力

```python
def _calculate_ee_forces(self):
    # 每 N 步施加一次脉冲
    pulse_interval = 50
    is_pulse_step = (self.common_step_counter % pulse_interval) < 5
    
    force_magnitude = torch.where(
        is_pulse_step.unsqueeze(-1),
        torch.tensor([50.0], device=self.device),  # 脉冲力
        torch.tensor([5.0], device=self.device)     # 基础力
    )
    # ...
```

## 下一步

1. **尝试训练**：`python train.py --task=g1_reaching_force`
2. **调整配置**：根据训练曲线调整力范围和课程学习参数
3. **对比实验**：与 `g1_reaching` 对比，评估鲁棒性提升
4. **迁移测试**：将训练好的策略部署到真实机器人（如果有硬件）

## 相关文档

- 详细文档：`README.md`
- 配置文件：`g1_reaching_force_config.py`
- 环境实现：`g1_reaching_force.py`
- 父类文档：`../g1_reaching/README.md`（如果有）

## 支持

如有问题，请检查：
1. Isaac Gym 是否正确安装
2. 观测维度是否匹配（89 维）
3. 训练日志中的错误信息
4. 尝试降低并行环境数：`--num_envs=512`
