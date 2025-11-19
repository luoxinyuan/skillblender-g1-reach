from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class H1BasicCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_actions = 19
        frame_stack = 1
        c_frame_stack = 3
        # keep a small, common command_dim (most h1 envs use 1..14)
        command_dim = 3
        num_single_obs = 3 * num_actions + 6 + command_dim
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 3 * num_actions + 25
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
        default_joint_angles = {
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
        tracking_sigma = 5
        max_contact_force = 700


class H1BasicCfgPPO(LeggedRobotCfgPPO):
    pass
