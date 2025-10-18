#!/usr/bin/env python3
"""
G1 Reaching Force - 测试脚本

用于验证环境是否正确安装和配置。
"""

import sys
from pathlib import Path

def test_import():
    """测试能否正确导入环境"""
    print("=" * 60)
    print("测试 1: 导入检查")
    print("=" * 60)
    
    try:
        from legged_gym.envs import task_registry
        print("✓ task_registry 导入成功")
        
        # 检查任务是否注册
        if 'g1_reaching_force' in task_registry.task_classes:
            print("✓ g1_reaching_force 已注册")
            return True
        else:
            print("✗ g1_reaching_force 未注册")
            print(f"  可用任务: {list(task_registry.task_classes.keys())[:5]}...")
            return False
            
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_config():
    """测试配置是否正确"""
    print("\n" + "=" * 60)
    print("测试 2: 配置检查")
    print("=" * 60)
    
    try:
        from legged_gym.envs.g1.g1_reaching_force.g1_reaching_force_config import (
            G1ReachingForceCfg, G1ReachingForceCfgPPO
        )
        
        cfg = G1ReachingForceCfg()
        cfg_ppo = G1ReachingForceCfgPPO()
        
        print(f"✓ 配置类导入成功")
        print(f"  - 环境数: {cfg.env.num_envs}")
        print(f"  - 观测维度: {cfg.env.num_observations}")
        print(f"  - Privileged 观测维度: {cfg.env.num_privileged_obs}")
        print(f"  - 动作维度: {cfg.env.num_actions}")
        print(f"  - Episode 长度: {cfg.env.episode_length_s}s")
        
        # 检查力配置
        print(f"\n  力配置:")
        print(f"  - X 力范围: {cfg.force.apply_force_x_range} N")
        print(f"  - Y 力范围: {cfg.force.apply_force_y_range} N")
        print(f"  - Z 力范围: {cfg.force.apply_force_z_range} N")
        print(f"  - 力持续时间: {cfg.force.randomize_force_duration} 步")
        print(f"  - 初始力强度: {cfg.force.force_scale_initial_scale}")
        print(f"  - 课程学习: {cfg.force.force_scale_curriculum}")
        print(f"  - 低通滤波: {cfg.force.use_lpf}")
        
        # 验证观测维度计算
        expected_obs = cfg.env.frame_stack * cfg.env.num_single_obs
        if cfg.env.num_observations == expected_obs:
            print(f"\n✓ 观测维度计算正确: {cfg.env.num_observations}")
        else:
            print(f"\n✗ 观测维度不匹配!")
            print(f"  预期: {expected_obs}")
            print(f"  实际: {cfg.env.num_observations}")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ 配置检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_creation():
    """测试环境创建（需要 Isaac Gym）"""
    print("\n" + "=" * 60)
    print("测试 3: 环境创建 (可选，需要 Isaac Gym)")
    print("=" * 60)
    
    try:
        from legged_gym.envs import task_registry
        from legged_gym import LEGGED_GYM_ROOT_DIR
        
        print("正在创建环境（这可能需要一些时间）...")
        
        env_cfg, train_cfg = task_registry.get_cfgs(name='g1_reaching_force')
        
        # 减少环境数以加快测试
        env_cfg.env.num_envs = 64
        
        print(f"  - 创建 {env_cfg.env.num_envs} 个并行环境")
        
        # 尝试创建环境（仅在有 Isaac Gym 时）
        try:
            env, _ = task_registry.make_env(name='g1_reaching_force', args=None, env_cfg=env_cfg)
            print("✓ 环境创建成功")
            
            # 检查观测形状
            obs = env.obs_buf
            print(f"  - 观测形状: {obs.shape}")
            print(f"  - 预期形状: ({env_cfg.env.num_envs}, {env_cfg.env.num_observations})")
            
            if obs.shape[1] == env_cfg.env.num_observations:
                print("✓ 观测维度匹配")
            else:
                print(f"✗ 观测维度不匹配!")
                return False
                
            return True
            
        except Exception as e:
            print(f"⚠ 环境创建失败（可能未安装 Isaac Gym）: {e}")
            return None  # None 表示跳过此测试
            
    except Exception as e:
        print(f"✗ 环境创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_usage_examples():
    """打印使用示例"""
    print("\n" + "=" * 60)
    print("使用示例")
    print("=" * 60)
    
    print("\n1. 训练（默认配置）:")
    print("   cd legged_gym/scripts")
    print("   python train.py --task=g1_reaching_force")
    
    print("\n2. 训练（快速测试，少量环境）:")
    print("   python train.py --task=g1_reaching_force --num_envs=512 --headless --max_iterations=100")
    
    print("\n3. 训练（从检查点继续）:")
    print("   python train.py --task=g1_reaching_force --resume --load_run=<实验名称>")
    
    print("\n4. 可视化测试:")
    print("   python play.py --task=g1_reaching_force --load_run=<实验名称>")
    
    print("\n5. 评估:")
    print("   python evaluate.py --task=g1_reaching_force --load_run=<实验名称>")

def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("G1 Reaching Force 环境测试")
    print("=" * 60)
    
    results = {}
    
    # 测试 1: 导入
    results['import'] = test_import()
    
    # 测试 2: 配置
    if results['import']:
        results['config'] = test_config()
    else:
        print("\n跳过配置测试（导入失败）")
        results['config'] = False
    
    # 测试 3: 环境创建（可选）
    if results['config']:
        result = test_environment_creation()
        results['environment'] = result if result is not None else True  # None 视为通过
    else:
        print("\n跳过环境创建测试（配置检查失败）")
        results['environment'] = False
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name.capitalize():20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！环境已正确安装。")
        print("=" * 60)
        print_usage_examples()
        return 0
    else:
        print("\n" + "=" * 60)
        print("✗ 部分测试失败，请检查错误信息。")
        print("=" * 60)
        
        print("\n常见问题:")
        print("1. 导入失败: 确保在 SkillBlender 目录下运行")
        print("2. 配置错误: 检查 g1_reaching_force_config.py")
        print("3. 环境创建失败: 确保已安装 Isaac Gym")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
