# G1 Reaching with Force Disturbance

## 概述

`g1_reaching_force` 是一个基于 `g1_reaching` 的扩展环境，在机器人手部到达目标位置的同时，施加持续变化的外力干扰。该环境模拟了真实场景中可能遇到的外界力扰动，提高策略的鲁棒性。

## 主要特性

### 1. 外力干扰机制

- **双手独立施力**：左右手分别施加独立的三维力（X, Y, Z 方向）
- **动态力变化**：力的大小随时间以三角波形式变化（phase 从 0 到 1）
- **可配置力范围**：可设置每个轴向的力范围（默认 ±30N）
- **力持续时间随机化**：每次 episode 随机采样力变化周期（50-200 步）
- **零力概率**：每个轴向可设置概率为零（默认 20%），增加多样性

### 2. 力课程学习（Curriculum Learning）

- 根据 episode 表现自动调整力的强度
- Episode 长度超过 80% → 增加力强度（+0.05）
- Episode 长度低于 30% → 减少力强度（-0.05）
- 力强度范围：[0.0, 1.0]，初始值：0.3

### 3. 低通滤波

- 可选的低通滤波器平滑力的变化
- 滤波系数：0.2（可配置）
- 避免力突变导致的不稳定

## 配置参数

### 外力配置 (`cfg.force`)

```python
class force:
    # 力范围（牛顿）
    apply_force_x_range = [-30.0, 30.0]
    apply_force_y_range = [-30.0, 30.0]
    apply_force_z_range = [-30.0, 30.0]
    
    # 力持续时间（仿真步数）
    randomize_force_duration = [50, 200]
    
    # 力施加位置比例（0=手掌中心，1=指尖）
    apply_force_pos_ratio_range = [0.0, 1.0]
    
    # 零力概率
    zero_force_prob = 0.2
    
    # 随机力概率
    random_force_prob = 0.1
    
    # 低通滤波
    use_lpf = True
    force_filter_alpha = 0.2
    
    # 力课程学习
    force_scale_curriculum = True
    force_scale_initial_scale = 0.3
    force_scale_min = 0.0
    force_scale_max = 1.0
    force_scale_up = 0.05
    force_scale_down = 0.05
    force_scale_up_threshold = 0.8
    force_scale_down_threshold = 0.3
    
    # 持续更新力相位
    update_apply_force_phase = True
```

## 观测空间

相比 `g1_reaching`，增加了 **6 维力观测**：

- `left_ee_apply_force` (3D)：左手受力（基座坐标系）
- `right_ee_apply_force` (3D)：右手受力（基座坐标系）

总观测维度：`num_single_obs = 3 * 21 + 6 + 14 + 6 = 89`

## 关键方法说明

### `_init_force_settings()`
初始化所有力相关的缓冲区和参数：
- 力张量、力施加位置
- 力范围、相位、持续时间
- 零力掩码、滤波器状态
- 课程学习参数

### `_update_apply_force_phase()`
更新力的相位（三角波）：
```python
phase = abs(remainder(timestamp, 2.0) - 1.0)  # [0, 1, 0, 1, ...]
```

### `_calculate_ee_forces()`
计算当前时刻施加的力：
1. 根据 phase 插值得到当前力
2. 乘以力强度系数（curriculum）
3. 应用零力掩码
4. 添加小噪声
5. 裁剪到范围内
6. 转换到基座坐标系供观测

### `_update_force_application_pos()`
更新力施加位置（当前实现：手掌中心）

### `post_physics_step()`
在每个物理步之后：
1. 更新力相位
2. 计算新的力
3. 更新施加位置
4. 通过 Isaac Gym API 施加力
5. 调用父类的 post_physics_step

### `_resample_force_settings(env_ids)`
为重置的环境重新采样力参数：
- 力方向分布（Dirichlet 分布）
- 力持续时间
- 初始相位
- 零力掩码
- 力施加位置比例

### `_update_force_scale_curriculum(env_ids)`
根据 episode 表现更新力强度：
- 表现好（episode 长）→ 增加力
- 表现差（episode 短）→ 减少力

### `reset_idx(env_ids)`
重置环境时：
1. 更新力课程学习参数
2. 重新采样力设置
3. 调用父类 reset

### `compute_observations()`
计算观测时：
1. 调用父类方法
2. 添加力观测（带噪声）
3. 同时添加到 privileged obs

## 使用方法

### 1. 训练

```bash
cd legged_gym/scripts
python train.py --task=g1_reaching_force
```

### 2. 测试

```bash
python play.py --task=g1_reaching_force --load_run=<run_name>
```

### 3. 自定义配置

修改 `g1_reaching_force_config.py` 中的参数，例如：

```python
# 增大力范围
class force:
    apply_force_x_range = [-50.0, 50.0]
    apply_force_y_range = [-50.0, 50.0]
    apply_force_z_range = [-50.0, 50.0]
    
# 禁用课程学习
    force_scale_curriculum = False
```

## 与参考环境的对比

### 参考：`LeggedRobotDecoupledLocomotionStanceHeightWBCForce`

本实现参考了 FALCON 项目中的力控制环境，主要相似之处：

1. **力计算流程**：phase → 插值 → 缩放 → 掩码 → 裁剪
2. **课程学习**：根据 episode 表现自适应调整力强度
3. **低通滤波**：平滑力变化
4. **Dirichlet 分布**：采样力在 XYZ 轴的分布

### 主要差异

| 特性 | FALCON | g1_reaching_force |
|-----|--------|-------------------|
| 任务类型 | 步态控制（walking/stance） | 手部到达（reaching） |
| 力施加部位 | 双手 | 双手（腕关节） |
| 力方向控制 | 考虑行走方向的阻力 | 全方向独立 |
| Jacobian 计算 | 根据关节极限计算最大可承受力 | 使用配置的固定范围 |
| 观测空间 | 复杂的全身状态 | 简化的到达任务观测 |

### 简化之处

为了适配 `g1_reaching`，本实现进行了以下简化：

1. **移除 Jacobian 计算**：不基于关节力矩限制动态计算最大力，使用配置的固定范围
2. **移除步态相关逻辑**：不区分 stance/walking 环境，统一施加力
3. **简化力施加位置**：直接在手掌中心施加，不使用球面采样
4. **移除腰部约束**：不考虑力对腰部关节的扭矩约束

## 扩展建议

### 1. 添加 Jacobian 约束

参考 FALCON 的实现，根据手臂关节的力矩限制动态计算可承受的最大力：

```python
def _calculate_max_ee_forces(self):
    jacobian = self.rigid_body_state.jacobian
    j_left_ee = jacobian[:, self.wrist_indices[0], :, :]
    # ... 计算基于关节极限的最大力
    return max_force
```

### 2. 力方向智能化

根据手腕当前位置和目标位置，施加更"对抗性"的力：

```python
# 计算从手腕到目标的方向
target_dir = self.ref_wrist_pos - self.wrist_pos
target_dir_norm = target_dir / torch.norm(target_dir, dim=-1, keepdim=True)

# 施加反向干扰力
force = -target_dir_norm * force_magnitude
```

### 3. 非对称力

为左右手设置不同的力范围或相位，模拟更复杂的干扰场景。

### 4. 可视化

添加力箭头可视化（参考 FALCON 的 `_draw_debug_vis`）。

## 注意事项

1. **力单位**：配置中的力单位是牛顿（N）
2. **坐标系**：
   - `apply_force_tensor` 使用世界坐标系
   - `left/right_ee_apply_force` 观测使用基座坐标系
3. **力施加 API**：使用 Isaac Gym 的 `apply_rigid_body_force_tensors`
4. **观测维度**：确保 `num_single_obs` 配置正确（增加了 6 维）
5. **性能**：力计算在每个 physics step 执行，注意计算开销

## 故障排查

### 问题：机器人不稳定/频繁跌倒

- 降低初始力强度：`force_scale_initial_scale = 0.1`
- 增加力滤波：`force_filter_alpha = 0.1`
- 提高零力概率：`zero_force_prob = 0.5`

### 问题：学习进度慢

- 禁用课程学习初期：`force_scale_initial_scale = 0.0`
- 延长 episode：`episode_length_s = 30`
- 调整奖励权重，增加 `wrist_pos` 的权重

### 问题：力观测维度不匹配

- 检查 `env.num_single_obs` 是否包含 `force_obs_dim = 6`
- 确认 `compute_observations()` 正确拼接了力观测

## 参考文献

- SkillBlender: https://github.com/Humanoid-SkillBlender/SkillBlender
- FALCON: Decoupled Locomotion with Force Control
- Isaac Gym: https://developer.nvidia.com/isaac-gym
