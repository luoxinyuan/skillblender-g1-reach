# G1 Reaching Force - 创建总结

## 已创建的文件

### 1. 核心文件

#### `/Users/luoxinyuan/Downloads/SkillBlender/legged_gym/legged_gym/envs/g1/g1_reaching_force/`

```
g1_reaching_force/
├── __init__.py                    # 模块初始化文件
├── g1_reaching_force.py           # 主环境实现（约 400 行）
├── g1_reaching_force_config.py    # 配置文件（约 270 行）
├── README.md                      # 详细技术文档
└── QUICKSTART.md                  # 快速开始指南
```

### 2. 注册文件修改

修改了 `/Users/luoxinyuan/Downloads/SkillBlender/legged_gym/legged_gym/envs/__init__.py`：
- 添加了 `g1_reaching_force` 的导入
- 添加了任务注册

## 文件说明

### 1. `__init__.py`
简单的模块初始化文件，导出主类和配置类。

### 2. `g1_reaching_force.py` - 主环境实现

**核心特性**：
- 继承自 `G1Reaching`
- 添加双手外力干扰机制
- 实现力课程学习
- 增加力观测到观测空间

**关键方法**：

| 方法名 | 功能 | 调用时机 |
|-------|-----|---------|
| `__init__()` | 初始化环境 | 环境创建时 |
| `_init_force_settings()` | 初始化力相关缓冲区 | 初始化时 |
| `_update_apply_force_phase()` | 更新力相位 | 每个控制步 |
| `_calculate_ee_forces()` | 计算当前力 | 每个物理步 |
| `_update_force_application_pos()` | 更新力施加位置 | 每个物理步 |
| `post_physics_step()` | 物理步后处理 | 每个物理步 |
| `_resample_force_settings()` | 重采样力参数 | 环境重置时 |
| `_update_force_scale_curriculum()` | 更新力课程 | 环境重置时 |
| `reset_idx()` | 重置环境 | 环境结束时 |
| `compute_observations()` | 计算观测 | 每个控制步 |

**数据流**：
```
初始化 → _init_force_settings()
    ↓
每个控制步：
    post_physics_step()
        ↓
    _update_apply_force_phase()
        ↓
    _calculate_ee_forces()
        ↓
    _update_force_application_pos()
        ↓
    gym.apply_rigid_body_force_tensors()
        ↓
    compute_observations() (添加力观测)
    
环境重置：
    reset_idx()
        ↓
    _update_force_scale_curriculum()
        ↓
    _resample_force_settings()
```

### 3. `g1_reaching_force_config.py` - 配置文件

**新增配置类**：`G1ReachingForceCfg.force`

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `apply_force_x/y/z_range` | list[2] | [-30, 30] | 力范围（N） |
| `randomize_force_duration` | list[2] | [50, 200] | 力持续时间（步） |
| `apply_force_pos_ratio_range` | list[2] | [0.0, 1.0] | 力施加位置 |
| `zero_force_prob` | float | 0.2 | 零力概率 |
| `random_force_prob` | float | 0.1 | 随机力概率 |
| `use_lpf` | bool | True | 使用低通滤波 |
| `force_filter_alpha` | float | 0.2 | 滤波系数 |
| `force_scale_curriculum` | bool | True | 启用课程学习 |
| `force_scale_initial_scale` | float | 0.3 | 初始力强度 |
| `force_scale_min/max` | float | 0.0/1.0 | 力强度范围 |
| `force_scale_up/down` | float | 0.05 | 调整步长 |
| `update_apply_force_phase` | bool | True | 持续更新相位 |

**观测维度调整**：
```python
# 原 g1_reaching: 83 维
num_single_obs = 3 * 21 + 6 + 14

# g1_reaching_force: 89 维
force_obs_dim = 6  # 左右手各 3 维力
num_single_obs = 3 * 21 + 6 + 14 + 6
```

### 4. `README.md` - 详细技术文档

包含内容：
- 环境概述和特性
- 配置参数详解
- 关键方法说明（带输入输出）
- 与参考环境对比
- 简化之处说明
- 扩展建议（Jacobian、智能力方向等）
- 注意事项和故障排查

### 5. `QUICKSTART.md` - 快速开始指南

包含内容：
- 快速开始命令
- 核心差异对比表
- 关键配置修改示例
- 训练技巧（分阶段训练）
- 常见问题 Q&A
- 性能基准
- 进阶自定义示例

## 实现参考

### 参考环境：`LeggedRobotDecoupledLocomotionStanceHeightWBCForce`

**借鉴的设计**：

1. **力计算流程**
   ```python
   phase → 插值(min, max) → 缩放(curriculum) → 掩码(zero_force) → 裁剪 → 转换坐标系
   ```

2. **课程学习机制**
   - 根据 episode 长度自适应调整力强度
   - 表现好 → 增加难度
   - 表现差 → 降低难度

3. **低通滤波**
   ```python
   filtered = alpha * new_value + (1 - alpha) * filtered
   ```

4. **Dirichlet 分布**
   用于采样力在 XYZ 轴的分布比例

5. **相位机制**
   使用三角波实现周期性力变化

### 简化之处

| 特性 | FALCON 实现 | g1_reaching_force |
|-----|-------------|-------------------|
| 最大力计算 | 基于 Jacobian 和关节极限 | 固定配置范围 |
| 步态区分 | stance/walking 不同策略 | 统一施加 |
| 力施加位置 | 球面随机采样 | 手掌中心 |
| 腰部约束 | 考虑力对腰部扭矩 | 不考虑 |
| 行走方向 | 施加阻力 | 全方向独立 |

**为什么简化？**
- `g1_reaching` 是静态到达任务，不涉及步态
- 不需要复杂的 Jacobian 计算（可后续扩展）
- 保持与父类一致的简洁性
- 便于理解和调试

## 使用流程

### 1. 验证安装

```bash
cd /Users/luoxinyuan/Downloads/SkillBlender
python -c "from legged_gym.envs import task_registry; print('g1_reaching_force' in task_registry.task_classes)"
# 应输出: True
```

### 2. 快速测试（无头模式，512 环境）

```bash
cd legged_gym/scripts
python train.py --task=g1_reaching_force --num_envs=512 --headless --max_iterations=100
```

### 3. 完整训练

```bash
python train.py --task=g1_reaching_force
```

### 4. 可视化测试

```bash
python play.py --task=g1_reaching_force --load_run=<实验名称>
```

## 扩展方向

### 1. 添加可视化

参考 FALCON 的 `_draw_debug_vis()`，在 Isaac Gym 中渲染力箭头：

```python
def _draw_debug_vis(self):
    self.gym.clear_lines(self.viewer)
    for env_id in range(self.num_envs):
        left_force = self.apply_force_tensor[env_id, self.wrist_indices[0], :]
        left_pos = self.apply_force_pos_tensor[env_id, self.wrist_indices[0], :]
        # 画力箭头
        end_pos = left_pos + left_force * 0.01  # 缩放显示
        self.gym.add_lines(self.viewer, self.envs[env_id], 1,
                          [left_pos[0], left_pos[1], left_pos[2],
                           end_pos[0], end_pos[1], end_pos[2]],
                          [1.0, 0.0, 0.0])  # 红色
```

### 2. 基于 Jacobian 的最大力计算

```python
def _calculate_max_ee_forces(self):
    # 获取手臂 Jacobian
    jacobian = self.gym.acquire_jacobian_tensor(self.sim, "g1")
    j_left_arm = jacobian[:, self.wrist_indices[0], :, self.left_arm_dof_indices]
    
    # 计算基于关节力矩限制的最大力
    joint_effort_limits = self.torque_limits[self.left_arm_dof_indices]
    max_forces = torch.min(joint_effort_limits / (torch.abs(j_left_arm) + 1e-6), dim=-1)[0]
    
    return max_forces
```

### 3. 智能对抗力

```python
def _calculate_ee_forces(self):
    # 计算手腕到目标的单位向量
    wrist_to_target = self.ref_wrist_pos - self.rigid_state[:, self.wrist_indices, :3]
    wrist_to_target_unit = wrist_to_target / (torch.norm(wrist_to_target, dim=-1, keepdim=True) + 1e-6)
    
    # 施加一定比例的对抗力（反向）
    adversarial_ratio = 0.7  # 70% 对抗，30% 随机
    random_force = torch.randn_like(wrist_to_target) * 10.0
    adversarial_force = -wrist_to_target_unit * self.force_magnitude * adversarial_ratio
    
    total_force = adversarial_force + random_force * (1 - adversarial_ratio)
    # ...
```

### 4. 非对称力策略

```python
class force:
    # 左手强力干扰，右手弱干扰
    left_apply_force_range = [-50.0, 50.0]
    right_apply_force_range = [-20.0, 20.0]
    
    # 左手低频变化，右手高频变化
    left_randomize_force_duration = [100, 300]
    right_randomize_force_duration = [20, 60]
```

## 测试清单

- [x] 文件创建完成
- [x] 任务注册完成
- [ ] 环境加载测试
- [ ] 短时训练测试（100 iterations）
- [ ] 观测维度验证（89 维）
- [ ] 力施加验证（通过日志或可视化）
- [ ] 课程学习验证（力强度变化）
- [ ] 完整训练（3000+ iterations）
- [ ] 性能对比（vs g1_reaching）

## 下一步建议

1. **立即测试**：运行快速测试验证环境可用性
   ```bash
   cd legged_gym/scripts
   python train.py --task=g1_reaching_force --num_envs=256 --headless --max_iterations=10
   ```

2. **观测维度检查**：确保观测缓冲区大小正确
   ```python
   # 在环境初始化后打印
   print(f"Obs shape: {env.obs_buf.shape}")  # 应该是 [N, 89]
   ```

3. **添加日志**：在 `_calculate_ee_forces()` 中添加日志
   ```python
   if self.common_step_counter % 1000 == 0:
       print(f"Force scale: {self.apply_force_scale[0].item():.3f}")
       print(f"Left force: {self.left_ee_apply_force[0]}")
   ```

4. **可视化测试**：训练 1000 次后可视化
   ```bash
   python play.py --task=g1_reaching_force
   ```

5. **性能分析**：对比训练曲线
   - 同时训练 `g1_reaching` 和 `g1_reaching_force`
   - 对比收敛速度、最终奖励、成功率

## 技术亮点

1. **模块化设计**：所有力相关逻辑封装在独立方法中
2. **课程学习**：自适应难度调整，提高训练效率
3. **灵活配置**：通过配置文件轻松调整所有参数
4. **向后兼容**：继承自 `g1_reaching`，可复用所有奖励函数
5. **详细文档**：README 和 QUICKSTART 覆盖所有使用场景

## 潜在问题和解决方案

### 问题 1：观测维度不匹配

**现象**：
```
RuntimeError: Expected tensor of size [4096, 83] but got [4096, 89]
```

**原因**：网络输入维度未更新

**解决**：
检查 `cfg.env.num_observations` 是否正确计算（应该是 89）

### 问题 2：力没有生效

**现象**：机器人表现与 `g1_reaching` 完全一样

**调试**：
```python
# 在 post_physics_step() 中添加
if self.common_step_counter % 100 == 0:
    print(f"Max force in tensor: {self.apply_force_tensor.abs().max().item()}")
```

如果输出为 0，检查 `apply_force_scale` 是否为 0。

### 问题 3：训练不稳定

**现象**：奖励曲线剧烈波动或持续下降

**解决**：
1. 降低初始力强度：`force_scale_initial_scale = 0.1`
2. 增加 episode 长度：`episode_length_s = 30`
3. 调整奖励权重，增加稳定性项：`orientation = 2.0`

### 问题 4：课程学习过快/过慢

**调整阈值**：
```python
# 过快：提高阈值
force_scale_up_threshold = 0.9  # 从 0.8 提高

# 过慢：降低阈值
force_scale_up_threshold = 0.6  # 从 0.8 降低
```

## 总结

成功创建了完整的 `g1_reaching_force` 环境，包括：
- ✅ 核心实现（400 行）
- ✅ 配置文件（270 行）
- ✅ 详细文档（README + QUICKSTART）
- ✅ 任务注册
- ✅ 参考 FALCON 的设计模式
- ✅ 保持与 SkillBlender 项目风格一致

**准备就绪，可以开始训练！** 🚀
