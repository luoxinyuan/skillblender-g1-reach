# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.g1.g1_reaching_force.g1_reaching_force import G1ReachingForce
from .g1_box_lift_config import G1BoxLiftCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
import numpy as np
import os
import math
import random
from collections import deque

from legged_gym.envs.base.legged_robot import LeggedRobot, get_euler_xyz_tensor, quat_rotate_inverse, LEGGED_GYM_ROOT_DIR


class G1BoxLift(G1ReachingForce):
    """
    G1BoxLift: same action/observation/behaviour as G1ReachingForce but with
    a box actor added to each environment. The robot model and force settings
    are kept identical to `g1_reaching_force`. The box is spawned and basic
    box state/buffers are exposed (box_root_states, box_idxs, box_goal_pos).
    """

    def __init__(self, cfg: G1BoxLiftCfg, sim_params, physics_engine, sim_device, headless):
        # delegate construction to parent (G1ReachingForce -> G1Reaching)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # box-related placeholders
        if not hasattr(self, 'box_goal_pos'):
            self.box_goal_pos = torch.zeros(self.num_envs, 3, device=self.device)

    def create_sim(self):
        # We'll reimplement _create_envs here to load robot and box assets together
        self.up_axis_idx = 2  # z-up for the box positioning (keeps same semantics as TaskLift)
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            # reuse parent's terrain creation if available
            try:
                from legged_gym.utils.terrain import XBotTerrain
                self.terrain = XBotTerrain(self.cfg.terrain, self.num_envs)
            except Exception:
                pass
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised")
        self._create_envs()

        # we'll largely mirror the box creation code from G1TaskLift
    def _create_envs(self):
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        # gather asset counts and properties
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body and dof names
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(self.body_names)
        self.num_dofs = len(self.dof_names)

        feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]
        knee_names = [s for s in self.body_names if self.cfg.asset.knee_name in s]

        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.env_frictions = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)

        self.body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)

        # create box asset
        box_size = self.cfg.asset.box_size
        box_asset_options = gymapi.AssetOptions()
        box_asset = self.gym.create_box(self.sim, box_size[0], box_size[1], box_size[2], box_asset_options)
        box_pose = gymapi.Transform()
        self.box_idxs = []
        self.humanoid_idxs = []
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.humanoid_idxs.append(self.gym.get_actor_index(env_handle, actor_handle, gymapi.DOMAIN_SIM))

            # add box
            box_pose.p = gymapi.Vec3(*pos[:3])
            box_pose.p.x += self.cfg.asset.box_offset_xy[0] + np.random.uniform(*self.cfg.asset.box_range_x)
            box_pose.p.y += self.cfg.asset.box_offset_xy[1] + np.random.uniform(*self.cfg.asset.box_range_y)
            box_pose.p.z = 0.5 * box_size[2]
            box_handle = self.gym.create_actor(env_handle, box_asset, box_pose, "box", i, 0)
            # change box rigid properties
            box_rigid_body_props = self.gym.get_actor_rigid_body_properties(env_handle, box_handle)
            for prop in box_rigid_body_props:
                prop.mass = random.uniform(*self.cfg.asset.box_range_mass) # change mass here!
            self.gym.set_actor_rigid_body_properties(env_handle, box_handle, box_rigid_body_props, recomputeInertia=True)
            box_rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, box_handle)
            for prop in box_rigid_shape_props:
                prop.friction = 5. # change friction here!
            self.gym.set_actor_rigid_shape_properties(env_handle, box_handle, box_rigid_shape_props)
            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(env_handle, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            self.box_idxs.append(self.gym.get_actor_index(env_handle, box_handle, gymapi.DOMAIN_SIM))

        self._create_sensors_all()
        self.humanoid_idxs = torch.tensor(self.humanoid_idxs, device=self.device)
        self.box_idxs = torch.tensor(self.box_idxs, device=self.device)

        # body part indices
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        # other body indices
        elbow_names = [s for s in self.body_names if self.cfg.asset.elbow_name in s]
        self.elbow_indices = torch.zeros(len(elbow_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(elbow_names)):
            self.elbow_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], elbow_names[i])
        torso_names = [s for s in self.body_names if self.cfg.asset.torso_name in s]
        self.torso_indices = torch.zeros(len(torso_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(torso_names)):
            self.torso_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], torso_names[i])
        wrist_names = [s for s in self.body_names if self.cfg.asset.wrist_name in s]
        self.wrist_indices = torch.zeros(len(wrist_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(wrist_names)):
            self.wrist_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], wrist_names[i])

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        (copied/aligned with G1TaskLift implementation so buffers are consistent)
        """
        self._init_visual_buffers()

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        # actor-wise root states viewed per-env: (num_envs, n_actors, 13)
        actor_rs = self.root_states.view(int(self.num_envs), -1, 13)

        # humanoid_root_states per-environment: handle both scalar-index and per-env index
        try:
            self.humanoid_idxs = self.humanoid_idxs.to(dtype=torch.long, device=self.device)
        except Exception:
            if not hasattr(self, 'humanoid_idxs'):
                self.humanoid_idxs = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        if self.humanoid_idxs.numel() == 1:
            h_idx = int(self.humanoid_idxs.view(-1)[0].item())
            self.humanoid_root_states = actor_rs[:, h_idx]
        elif self.humanoid_idxs.numel() == int(self.num_envs):
            h_idx = self.humanoid_idxs.to(dtype=torch.long, device=self.device)
            env_idx = torch.arange(int(self.num_envs), dtype=torch.long, device=self.device)
            self.humanoid_root_states = actor_rs[env_idx, h_idx]
        else:
            h_idx = int(self.humanoid_idxs.view(-1)[0].item())
            self.humanoid_root_states = actor_rs[:, h_idx]

        # box_root_states per-environment: handle both scalar-index and per-env index
        # ensure box_idxs is long tensor on correct device
        try:
            self.box_idxs = self.box_idxs.to(dtype=torch.long, device=self.device)
        except Exception:
            # if box_idxs isn't set yet, create a default (fallback)
            if not hasattr(self, 'box_idxs'):
                self.box_idxs = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        if self.box_idxs.numel() == 1:
            idx = int(self.box_idxs.view(-1)[0].item())
            self.box_root_states = actor_rs[:, idx]
        elif self.box_idxs.numel() == int(self.num_envs):
            idx = self.box_idxs.to(dtype=torch.long, device=self.device)
            env_idx = torch.arange(int(self.num_envs), dtype=torch.long, device=self.device)
            self.box_root_states = actor_rs[env_idx, idx]
        else:
            idx = int(self.box_idxs.view(-1)[0].item())
            self.box_root_states = actor_rs[:, idx]

    # keep `self.root_states` as the full actor-wise tensor (needed by
    # simulator-indexed setters). Use `self.humanoid_root_states` and
    # `self.box_root_states` for per-actor convenience without overwriting
    # the simulator's expected layout.

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.humanoid_root_states[:, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)

        # update num_bodies to include all actors (robot + box) so force tensors
        # and other per-body buffers use the correct length
        try:
            self.num_bodies = int(self.rigid_state.shape[1])
        except Exception:
            pass

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_rigid_state = torch.zeros_like(self.rigid_state)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.humanoid_root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.humanoid_root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.humanoid_root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[:, i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[:, i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[:, i] = 0.
                self.d_gains[:, i] = 0.
                print(f"PD gain of joint {name} were not defined, setting them to zero")

        self.rand_push_force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.rand_push_torque = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.default_joint_pd_target = self.default_dof_pos.clone()
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(self.num_envs, self.cfg.env.num_single_obs, dtype=torch.float, device=self.device))
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(torch.zeros(self.num_envs, self.cfg.env.single_num_privileged_obs, dtype=torch.float, device=self.device))

    def _reset_box_and_goal(self, env_ids):
        pos = self.env_origins[env_ids].clone()
        # ensure box_goal_pos exists (may be missing during parent init order)
        if not hasattr(self, 'box_goal_pos'):
            self.box_goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.box_root_states[env_ids, 0] = pos[:, 0] + self.cfg.asset.box_offset_xy[0] + torch.FloatTensor(len(env_ids)).uniform_(*self.cfg.asset.box_range_x).to(self.device)
        self.box_root_states[env_ids, 1] = pos[:, 1] + self.cfg.asset.box_offset_xy[1] + torch.FloatTensor(len(env_ids)).uniform_(*self.cfg.asset.box_range_y).to(self.device)
        self.box_root_states[env_ids, 2] = 0.5 * self.cfg.asset.box_size[2]
        self.box_root_states[env_ids, 3] = 1
        self.box_root_states[env_ids, 4:] = 0

        self.box_goal_pos[env_ids, 2] = self.box_root_states[env_ids, 2] + torch.FloatTensor(len(env_ids)).uniform_(*self.cfg.commands.ranges.box_pos_z).to(self.device)
        self.box_goal_pos[env_ids, 0] = self.box_root_states[env_ids, 0]
        self.box_goal_pos[env_ids, 1] = self.box_root_states[env_ids, 1]

    def _reset_root_states(self, env_ids):
        # call parent reset then reset box
        super()._reset_root_states(env_ids)
        self._reset_box_and_goal(env_ids)

    def post_physics_step(self):
        """Custom post_physics_step for multi-actor envs:
        Refresh sim tensors, build per-env humanoid/box root-state views,
        compute base quantities from humanoid view, apply forces (if any),
        then run the usual callbacks, rewards, resets and observations.
        """
        # refresh tensors coming from the simulator
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # advance counters
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # actor-wise root states viewed per-env: (num_envs, n_actors, 13)
        actor_rs = self.root_states.view(int(self.num_envs), -1, 13)

        # recompute humanoid_root_states per-environment
        try:
            h_idx = self.humanoid_idxs.to(dtype=torch.long, device=self.device)
        except Exception:
            if not hasattr(self, 'humanoid_idxs'):
                h_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            else:
                h_idx = self.humanoid_idxs

        if h_idx.numel() == 1:
            h_i = int(h_idx.view(-1)[0].item())
            self.humanoid_root_states = actor_rs[:, h_i]
        elif h_idx.numel() == int(self.num_envs):
            env_idx = torch.arange(int(self.num_envs), dtype=torch.long, device=self.device)
            self.humanoid_root_states = actor_rs[env_idx, h_idx]
        else:
            h_i = int(h_idx.view(-1)[0].item())
            self.humanoid_root_states = actor_rs[:, h_i]

        # recompute box_root_states per-environment
        try:
            b_idx = self.box_idxs.to(dtype=torch.long, device=self.device)
        except Exception:
            if not hasattr(self, 'box_idxs'):
                b_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            else:
                b_idx = self.box_idxs

        if b_idx.numel() == 1:
            bi = int(b_idx.view(-1)[0].item())
            self.box_root_states = actor_rs[:, bi]
        elif b_idx.numel() == int(self.num_envs):
            env_idx = torch.arange(int(self.num_envs), dtype=torch.long, device=self.device)
            self.box_root_states = actor_rs[env_idx, b_idx]
        else:
            bi = int(b_idx.view(-1)[0].item())
            self.box_root_states = actor_rs[:, bi]

        # compute base quantities using humanoid view
        self.base_quat[:] = self.humanoid_root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.humanoid_root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.humanoid_root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        # If force settings exist, update/apply them (mirrors G1ReachingForce)
        if hasattr(self, 'update_apply_force_phase') and getattr(self, 'update_apply_force_phase'):
            try:
                self._update_apply_force_phase()
            except Exception:
                pass

        if hasattr(self, '_calculate_ee_forces'):
            try:
                self._calculate_ee_forces()
                self._update_force_application_pos()
                # apply forces (world frame) if tensors are present
                if hasattr(self, 'apply_force_tensor') and hasattr(self, 'apply_force_pos_tensor'):
                    self.gym.apply_rigid_body_force_at_pos_tensors(
                        self.sim,
                        gymtorch.unwrap_tensor(self.apply_force_tensor),
                        gymtorch.unwrap_tensor(self.apply_force_pos_tensor),
                        gymapi.ENV_SPACE,
                    )
            except Exception:
                # if force application fails, continue without crashing here; higher-level code will detect mismatches
                pass

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        # run the common post-physics pipeline (terminations, rewards, observations)
        self._post_physics_step_callback()

        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # reset envs that need resetting (will also reset box states)
        self.reset_idx(env_ids)
        # update target waypoints for reset envs
        # self.update_target_wp(env_ids)
        # self.update_target_wp_incremental(env_ids)
        self.update_target_wp_from_csv(env_ids)
        # compute observations (some sensors depend on refreshed rigid_state)
        self.compute_observations()

        # update history and last-state buffers
        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.humanoid_root_states[:, 7:13]
        self.last_rigid_state[:] = self.rigid_state[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # nothing extra for now; box reset handled in _reset_root_states
