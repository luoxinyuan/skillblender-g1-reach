from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class H1BasicCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_actions = 19
        frame_stack = 1
        c_frame_stack = 3
        # keep a small, common command_dim (most h1 envs use 1..14)
        command_dim = 0
        num_single_obs = 3 * num_actions + 6 + command_dim
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 3 * num_actions + 18 # 25
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

        num_envs = 4096
        episode_length_s = 24
        use_ref_actions = False

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1_wrist.urdf'
        name = "h1"
        foot_name = "ankle"
        knee_name = "knee"
        elbow_name = "elbow"
        torso_name = "torso"
        wrist_name = "wrist"
        terminate_after_contacts_on = ['pelvis', 'torso', 'shoulder', 'elbow']
        penalize_contacts_on = ["hip", 'knee']
        self_collisions = 0
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.0]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_hip_yaw_joint' : 0. ,
            'left_hip_roll_joint' : 0,
            'left_hip_pitch_joint' : -0.4,
            'left_knee_joint' : 0.8,
            'left_ankle_joint' : -0.4,
            'right_hip_yaw_joint' : 0.,
            'right_hip_roll_joint' : 0,
            'right_hip_pitch_joint' : -0.4,
            'right_knee_joint' : 0.8,
            'right_ankle_joint' : -0.4,
            'torso_joint' : 0.,
            'left_shoulder_pitch_joint' : 0.,
            'left_shoulder_roll_joint' : 0,
            'left_shoulder_yaw_joint' : 0.,
            'left_elbow_joint'  : 0.,
            'right_shoulder_pitch_joint' : 0.,
            'right_shoulder_roll_joint' : 0.0,
            'right_shoulder_yaw_joint' : 0.,
            'right_elbow_joint' : 0.,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'hip_yaw': 200,
                     'hip_roll': 200,
                     'hip_pitch': 200,
                     'knee': 300,
                     'ankle': 40,
                     'torso': 300,
                     'shoulder': 100,
                     "elbow":100,
                     }
        damping = {  'hip_yaw': 5,
                     'hip_roll': 5,
                     'hip_pitch': 5,
                     'knee': 6,
                     'ankle': 2,
                     'torso': 6,
                     'shoulder': 2,
                     "elbow":2,
                     }
        action_scale = 0.25
        decimation = 10

    class sim(LeggedRobotCfg.sim):
        dt = 0.001
        substeps = 1
        up_axis = 1

    class commands(LeggedRobotCfg.commands):
        num_commands = 4
        resampling_time = 8.
        heading_command = True
        curriculum = False

    class rewards:
        base_height_target = 0.89
        min_dist = 0.2
        max_dist = 0.5
        target_joint_pos_scale = 0.17
        target_feet_height = 0.06
        cycle_time = 0.64
        only_positive_rewards = True
        tracking_sigma = 5
        max_contact_force = 700

        # punch-specific reward tuning
        punch_forward_coeff = 1.0
        punch_extension_coeff = 0.5
        punch_extension_baseline = 0.05
        punch_impact_coeff = 0.01
        punch_torque_coeff = 1e-6

        class scales:
            feet_distance = 0.5
            knee_distance = 0.2
            default_joint_pos = 0.5
            upper_body_pos = 0.5
            orientation = 1.
            punch = 1.0
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7


class H1BasicCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

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
        num_steps_per_env = 60  # per iteration
        max_iterations = 15001 # 3001  # number of policy updates

        # logging
        save_interval = 1000  # check for potential saves every this many iterations
        experiment_name = 'h1_squatting'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and ckpt