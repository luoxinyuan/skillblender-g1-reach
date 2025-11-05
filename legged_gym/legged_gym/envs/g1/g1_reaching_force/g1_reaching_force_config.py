# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class G1ReachingForceCfg(LeggedRobotCfg):
    """
    Configuration class for the G1 humanoid robot with force disturbance.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim (added 6 for left/right hand force observation)
        num_actions = 21
        frame_stack = 1
        c_frame_stack = 3
        command_dim = 14
        force_obs_dim = 6  # 3 for left hand force + 3 for right hand force
        num_single_obs = 3 * num_actions + 6 + command_dim + force_obs_dim
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 3 * num_actions + 60 + force_obs_dim
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        
        num_envs = 4096
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = False

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_29dof_lock_waist_rev_1_0_modified_situp.urdf'

        name = "g1"
        foot_name = "ankle_roll"
        knee_name = "knee"
        elbow_name = "elbow"
        torso_name = "torso_link"
        wrist_name = "rubber_hand"

        terminate_after_contacts_on = ['pelvis', 'torso', 'waist', 'shoulder', 'elbow', 'wrist']
        penalize_contacts_on = ["hip", 'knee']
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False
        collapse_fixed_joints = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20
        num_cols = 20
        max_init_terrain_level = 10
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = {
            'left_hip_pitch_joint' : -0.4,   
            'left_hip_roll_joint' : 0,   
            'left_hip_yaw_joint' : 0. ,                     
            'left_knee_joint' : 0.8,       
            'left_ankle_pitch_joint' : -0.4,     
            'left_ankle_roll_joint' : 0,   
            
            'right_hip_pitch_joint' : -0.4,    
            'right_hip_roll_joint' : 0,   
            'right_hip_yaw_joint' : 0.,                                    
            'right_knee_joint' : 0.8,                                             
            'right_ankle_pitch_joint': -0.4,                              
            'right_ankle_roll_joint' : 0,      
             
            'waist_yaw_joint' : 0.,
            
            "left_shoulder_pitch_joint": 0.,
            "left_shoulder_roll_joint": 0.,
            "left_shoulder_yaw_joint": 0.,
            "left_elbow_joint": 0.,
            
            "right_shoulder_pitch_joint": 0.,
            "right_shoulder_roll_joint": 0.,
            "right_shoulder_yaw_joint": 0.,
            "right_elbow_joint": 0.,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'hip_yaw': 150,
                     'hip_roll': 150,
                     'hip_pitch': 200,
                     'knee': 200,
                     'ankle': 20,
                     'waist': 200,
                     'shoulder': 40,
                     'elbow': 40,
                     }
        damping = {  'hip_yaw': 5,
                     'hip_roll': 5,
                     'hip_pitch': 5,
                     'knee': 5,
                     'ankle': 4,
                     'waist': 5,
                     'shoulder': 10,
                     'elbow': 10,
                     }
        action_scale = 0.25
        decimation = 10  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.1
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2

    class commands(LeggedRobotCfg.commands):
        num_commands = 4
        resampling_time = 8.
        heading_command = True
        curriculum = False

        class ranges:
            lin_vel_x = [-0, 0]
            lin_vel_y = [-0, 0]
            ang_vel_yaw = [-0, 0]
            heading = [-0, 0]
            # wrist pos command ranges
            wrist_max_radius = 0.15
            l_wrist_pos_x = [-0.05, 0.15]
            l_wrist_pos_y = [-0.05, 0.15]
            l_wrist_pos_z = [-0.15, 0.15] # [-0.15, 0.15]
            r_wrist_pos_x = [-0.05, 0.15]
            r_wrist_pos_y = [-0.15, 0.05]
            r_wrist_pos_z = [-0.15, 0.15] # [-0.15, 0.15]

    # External force configuration
    class force:
        # Force ranges for X, Y, Z axes (in Newtons, in world frame)
        apply_force_x_range = [-30.0, 30.0]
        apply_force_y_range = [-30.0, 30.0]
        apply_force_z_range = [-30.0, 30.0]
        
        # Force duration (in simulation steps)
        randomize_force_duration = [100, 150]  # min and max steps
        
        # Force application position on hand link
        apply_force_pos_ratio_range = [0.0, 1.0]  # 0 = hand center, 1 = fingertip
        
        # Zero force probability for each axis
        zero_force_prob = 0.2  # probability of having zero force on each axis
        
        # Random force probability
        random_force_prob = 0.1  # probability of applying completely random force
        
        # Low pass filter for force
        use_lpf = True
        force_filter_alpha = 0.2  # filter coefficient
        
        # Force scale curriculum
        force_scale_curriculum = True
        force_scale_initial_scale = 0.3
        force_scale_min = 0.0
        force_scale_max = 1.0
        force_scale_up = 0.05
        force_scale_down = 0.05
        force_scale_up_threshold = 0.8  # episode length ratio to trigger scale up
        force_scale_down_threshold = 0.3  # episode length ratio to trigger scale down
        
        # Update force phase continuously
        update_apply_force_phase = True
        # If true use impedance-based reward (placeholder) for wrist tracking.
        # When False, use the standard positional tracking reward (same as parent).
        if_impedance = True
        # Impedance stiffness (N/m) used to convert external force to an equivalent
        # displacement: delta_x = F_ext / K. Can be tuned per task.
        impedance_K = 200.0

    class rewards:
        base_height_target = 0.728
        min_dist = 0.05
        max_dist = 0.25
        target_joint_pos_scale = 0.17
        target_feet_height = 0.06
        cycle_time = 0.64
        only_positive_rewards = True
        tracking_sigma = 5
        max_contact_force = 700

        class scales:
            # wrist tracking
            wrist_pos = 5 * 2
            # stability
            feet_distance = 0.5
            default_joint_pos = 0.5 * 4
            upper_body_pos = 0.5
            orientation = 1.
            # energy
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7


class G1ReachingForceCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60
        max_iterations = 15001

        save_interval = 1000
        experiment_name = 'g1_reaching_force'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
