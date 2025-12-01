from legged_gym.envs.g1.g1_reaching_force.g1_reaching_force_config import G1ReachingForceCfg, G1ReachingForceCfgPPO


class G1BoxLiftCfg(G1ReachingForceCfg):
    class asset(G1ReachingForceCfg.asset):
        # inherit robot file from reaching_force but add box params
        box_size = [0.5, 0.1, 0.7]
        box_offset_xy = [1.0, 0.0]
        box_range_x = [-0.5, -0.3]
        box_range_y = [-0.05, 0.05]
        box_range_mass = [0.1, 2.0]

    class commands(G1ReachingForceCfg.commands):
        class ranges(G1ReachingForceCfg.commands.ranges):
            box_pos_z = [0.2, 0.4]


class G1BoxLiftCfgPPO(G1ReachingForceCfgPPO):
    pass
