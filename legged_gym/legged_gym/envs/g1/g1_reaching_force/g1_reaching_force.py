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


from legged_gym.envs.g1.g1_reaching.g1_reaching import G1Reaching
from .g1_reaching_force_config import G1ReachingForceCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch


class G1ReachingForce(G1Reaching):
    '''
    G1 Reaching task with external force disturbances applied to the hands.
    
    This class extends G1Reaching by adding continuous external forces to the robot's hands,
    simulating realistic disturbances during reaching tasks. The forces vary in magnitude,
    direction, and duration according to configurable parameters.
    
    Additional Attributes:
        left_ee_apply_force (torch.Tensor): Force applied to left hand in base frame [N, 3]
        right_ee_apply_force (torch.Tensor): Force applied to right hand in base frame [N, 3]
        apply_force_tensor (torch.Tensor): Force tensor for all rigid bodies [N, num_bodies, 3]
        apply_force_pos_tensor (torch.Tensor): Force application positions [N, num_bodies, 3]
        apply_force_scale (torch.Tensor): Current force scale for curriculum [N, 1]
        left/right_ee_apply_force_phase (torch.Tensor): Current force phase [0,1] for each hand
        
    Additional Methods:
        _init_force_settings(): Initializes force-related buffers and parameters
        _calculate_ee_forces(): Computes forces to apply based on current phase and settings
        _update_apply_force_phase(): Updates force phase for continuous variation
        _resample_force_settings(env_ids): Resamples force parameters for reset environments
        _update_force_scale_curriculum(env_ids): Updates force scale based on performance
    '''
    
    def __init__(self, cfg: G1ReachingForceCfg, sim_params, physics_engine, sim_device, headless):
        # Initialize force settings before parent init
        self.cfg: G1ReachingForceCfg
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # Initialize force-specific buffers
        self._init_force_settings()
        
    def _init_force_settings(self):
        """Initialize all force-related buffers and parameters"""
        # Force application tensors
        self.left_ee_apply_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_ee_apply_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.apply_force_tensor = torch.zeros(
            self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.apply_force_pos_tensor = torch.zeros(
            self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        
        # Force position settings
        self.apply_force_pos_ratio_range = self.cfg.force.apply_force_pos_ratio_range
        self.left_ee_apply_force_pos_ratio = torch.ones(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        ) * (self.apply_force_pos_ratio_range[0] + self.apply_force_pos_ratio_range[1]) / 2.0
        self.right_ee_apply_force_pos_ratio = torch.ones(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        ) * (self.apply_force_pos_ratio_range[0] + self.apply_force_pos_ratio_range[1]) / 2.0
        
        # Force ranges
        self.force_xyz_scale = torch.distributions.Dirichlet(
            torch.tensor([1.0, 1.0, 1.0], device=self.device)
        ).sample((self.num_envs, ))
        self.force_range_low = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.force_range_high = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.force_range_low[:, 0] = self.cfg.force.apply_force_x_range[0]
        self.force_range_high[:, 0] = self.cfg.force.apply_force_x_range[1]
        self.force_range_low[:, 1] = self.cfg.force.apply_force_y_range[0]
        self.force_range_high[:, 1] = self.cfg.force.apply_force_y_range[1]
        self.force_range_low[:, 2] = self.cfg.force.apply_force_z_range[0]
        self.force_range_high[:, 2] = self.cfg.force.apply_force_z_range[1]
        
        # Force duration and phase
        self.apply_force_duration = torch.randint(
            self.cfg.force.randomize_force_duration[0],
            self.cfg.force.randomize_force_duration[1] + 1,
            (self.num_envs, 1), device=self.device
        )
        self.left_ee_apply_force_phase = torch.rand((self.num_envs, 1), device=self.device)
        self.right_ee_apply_force_phase = torch.rand((self.num_envs, 1), device=self.device)
        self.left_ee_apply_force_phase_ts = torch.zeros((self.num_envs, 1), device=self.device)
        self.right_ee_apply_force_phase_ts = torch.zeros((self.num_envs, 1), device=self.device)
        
        # Zero force probability
        zero_force_prob = self.cfg.force.zero_force_prob
        if isinstance(zero_force_prob, float):
            zero_force_prob = [zero_force_prob] * 3
        self.zero_force_prob = torch.tensor(zero_force_prob, device=self.device)
        self.left_zero_force = (torch.rand((self.num_envs, 3), device=self.device) < self.zero_force_prob).float()
        self.right_zero_force = (torch.rand((self.num_envs, 3), device=self.device) < self.zero_force_prob).float()
        
        # Random force probability
        self.random_force_prob = self.cfg.force.random_force_prob
        self.random_force = (torch.rand((self.num_envs, 1), device=self.device) < self.random_force_prob).float()
        
        # Low pass filter for applied force
        self.use_lpf = self.cfg.force.use_lpf
        self.filtered_left_force_min = torch.zeros((self.num_envs, 3), device=self.device)
        self.filtered_left_force_max = torch.zeros((self.num_envs, 3), device=self.device)
        self.filtered_right_force_min = torch.zeros((self.num_envs, 3), device=self.device)
        self.filtered_right_force_max = torch.zeros((self.num_envs, 3), device=self.device)
        self.force_filter_alpha = self.cfg.force.force_filter_alpha
        
        # Force scale curriculum
        if self.cfg.force.force_scale_curriculum:
            self.apply_force_scale = torch.ones(
                self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
            ) * self.cfg.force.force_scale_initial_scale
        else:
            self.apply_force_scale = torch.ones(
                self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
            )
        
        self.update_apply_force_phase = self.cfg.force.update_apply_force_phase
        
    def _update_apply_force_phase(self):
        """Update force phase for continuous variation"""
        # Update phase timestamp
        self.left_ee_apply_force_phase_ts += 1.0 / self.apply_force_duration
        self.right_ee_apply_force_phase_ts += 1.0 / self.apply_force_duration
        
        # Compute triangular wave phase [0, 1]
        self.left_ee_apply_force_phase = torch.abs(
            torch.remainder(self.left_ee_apply_force_phase_ts, 2.0) - 1.0
        )
        self.right_ee_apply_force_phase = torch.abs(
            torch.remainder(self.right_ee_apply_force_phase_ts, 2.0) - 1.0
        )
    
    def _calculate_ee_forces(self):
        """Calculate forces to apply to end effectors based on current phase"""
        # Compute phased force (interpolate between min and max based on phase)
        left_ee_force_phased = self.force_range_low + (
            self.force_range_high - self.force_range_low
        ) * self.left_ee_apply_force_phase
        right_ee_force_phased = self.force_range_low + (
            self.force_range_high - self.force_range_low
        ) * self.right_ee_apply_force_phase
        
        # Apply force scale and add small noise
        left_hand_force = left_ee_force_phased * self.apply_force_scale + \
                         torch.randn((self.num_envs, 3), device=self.device) * 0.5
        right_hand_force = right_ee_force_phased * self.apply_force_scale + \
                          torch.randn((self.num_envs, 3), device=self.device) * 0.5
        
        # Zero the force if zero force probability is met
        left_hand_force *= (1 - self.left_zero_force)
        right_hand_force *= (1 - self.right_zero_force)
        
        # Clip using the min/max as bounds
        left_hand_force = torch.clip(left_hand_force, self.force_range_low, self.force_range_high)
        right_hand_force = torch.clip(right_hand_force, self.force_range_low, self.force_range_high)
        
        # Store forces in base frame for observation
        self.left_ee_apply_force = quat_rotate_inverse(self.base_quat, left_hand_force.clone())
        self.right_ee_apply_force = quat_rotate_inverse(self.base_quat, right_hand_force.clone())
        
        # Apply the force to the hand links (in world frame)
        self.apply_force_tensor[:, self.wrist_indices[0], :] = left_hand_force
        self.apply_force_tensor[:, self.wrist_indices[1], :] = right_hand_force
        
    def _update_force_application_pos(self):
        """Update the position where forces are applied on the hands"""
        # Apply force at interpolated position between hand center and fingertip
        left_ee_pos = self.rigid_state[:, self.wrist_indices[0], :3]
        right_ee_pos = self.rigid_state[:, self.wrist_indices[1], :3]
        
        # For simplicity, apply at hand center (can be extended to fingertip if needed)
        self.apply_force_pos_tensor[:, self.wrist_indices[0], :] = left_ee_pos
        self.apply_force_pos_tensor[:, self.wrist_indices[1], :] = right_ee_pos
        
    def post_physics_step(self):
        """Override to add force updates before observations"""
        # Update force phase if enabled
        if self.update_apply_force_phase:
            self._update_apply_force_phase()
        
        # Calculate and apply forces
        self._calculate_ee_forces()
        self._update_force_application_pos()
        
        # Apply forces to simulation at positions on the rigid bodies
        # Use apply_rigid_body_force_at_pos_tensors to apply force vectors at specific points
        self.gym.apply_rigid_body_force_at_pos_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.apply_force_tensor),
            gymtorch.unwrap_tensor(self.apply_force_pos_tensor),
            gymapi.ENV_SPACE
        )
        print(f'Applied left hand force: {self.left_ee_apply_force[0].cpu().numpy()} at position {self.apply_force_pos_tensor[0, self.wrist_indices[0]].cpu().numpy()}')
        print(f'Applied right hand force: {self.right_ee_apply_force[0].cpu().numpy()} at position {self.apply_force_pos_tensor[0, self.wrist_indices[1]].cpu().numpy()}')
        print('---')
        
        # Call parent post_physics_step
        super().post_physics_step()

    def _draw_debug_vis(self):
        """Draw force vectors for left and right hands in the viewer."""
        # follow parent behavior: only draw when viewer is available and debug enabled
        if not (hasattr(self, 'viewer') and self.viewer and self.enable_viewer_sync and self.debug_viz):
            return

        # ensure rigid state is refreshed
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        for i in range(self.num_envs):
            try:
                left_idx = int(self.wrist_indices[0])
                right_idx = int(self.wrist_indices[1])
            except Exception:
                # fallback if wrist_indices not set as expected
                continue

            # get force and position (numpy)
            pos_l = self.apply_force_pos_tensor[i, left_idx].cpu().numpy()
            pos_r = self.apply_force_pos_tensor[i, right_idx].cpu().numpy()
            force_l = self.apply_force_tensor[i, left_idx].cpu().numpy()
            force_r = self.apply_force_tensor[i, right_idx].cpu().numpy()

            # draw a line from pos to pos + scaled force
            scale = 0.02
            start_l = gymapi.Vec3(float(pos_l[0]), float(pos_l[1]), float(pos_l[2]))
            end_l = gymapi.Vec3(float(pos_l[0] + force_l[0] * scale), float(pos_l[1] + force_l[1] * scale), float(pos_l[2] + force_l[2] * scale))
            start_r = gymapi.Vec3(float(pos_r[0]), float(pos_r[1]), float(pos_r[2]))
            end_r = gymapi.Vec3(float(pos_r[0] + force_r[0] * scale), float(pos_r[1] + force_r[1] * scale), float(pos_r[2] + force_r[2] * scale))

            color = gymapi.Vec3(0.851, 0.144, 0.07)
            # draw lines
            gymutil.draw_line(start_l, end_l, color, self.gym, self.viewer, self.envs[i])
            gymutil.draw_line(start_r, end_r, color, self.gym, self.viewer, self.envs[i])
        
    def _resample_force_settings(self, env_ids):
        """Resample force parameters for reset environments"""
        if len(env_ids) == 0:
            return
            
        # Resample force scales
        self.force_xyz_scale[env_ids] = torch.distributions.Dirichlet(
            torch.tensor([1.0, 1.0, 1.0], device=self.device)
        ).sample((len(env_ids), ))
        
        # Resample duration
        self.apply_force_duration[env_ids] = torch.randint(
            self.cfg.force.randomize_force_duration[0],
            self.cfg.force.randomize_force_duration[1] + 1,
            (len(env_ids), 1), device=self.device
        )
        
        # Resample phase
        self.left_ee_apply_force_phase_ts[env_ids] = torch.rand((len(env_ids), 1), device=self.device)
        self.right_ee_apply_force_phase_ts[env_ids] = torch.rand((len(env_ids), 1), device=self.device)
        self.left_ee_apply_force_phase[env_ids] = torch.rand((len(env_ids), 1), device=self.device)
        self.right_ee_apply_force_phase[env_ids] = torch.rand((len(env_ids), 1), device=self.device)
        
        # Resample zero force mask
        self.left_zero_force[env_ids] = (
            torch.rand((len(env_ids), 3), device=self.device) < self.zero_force_prob
        ).float()
        self.right_zero_force[env_ids] = (
            torch.rand((len(env_ids), 3), device=self.device) < self.zero_force_prob
        ).float()
        
        # Resample random force mask
        self.random_force[env_ids] = (
            torch.rand((len(env_ids), 1), device=self.device) < self.random_force_prob
        ).float()
        
        # Resample force position ratio
        self.left_ee_apply_force_pos_ratio[env_ids] = torch.rand((len(env_ids), 1), device=self.device) * \
            (self.apply_force_pos_ratio_range[1] - self.apply_force_pos_ratio_range[0]) + \
            self.apply_force_pos_ratio_range[0]
        self.right_ee_apply_force_pos_ratio[env_ids] = torch.rand((len(env_ids), 1), device=self.device) * \
            (self.apply_force_pos_ratio_range[1] - self.apply_force_pos_ratio_range[0]) + \
            self.apply_force_pos_ratio_range[0]
        
        # Reset filtered forces
        self.filtered_left_force_max[env_ids] = 0.0
        self.filtered_left_force_min[env_ids] = 0.0
        self.filtered_right_force_max[env_ids] = 0.0
        self.filtered_right_force_min[env_ids] = 0.0
        
    def _update_force_scale_curriculum(self, env_ids):
        """Update force scale based on episode performance"""
        if len(env_ids) == 0:
            return
            
        # Scale up if episode lasted long enough (good performance)
        episode_length_ratio = self.episode_length_buf[env_ids].float() / self.max_episode_length
        env_ids_scale_up_mask = episode_length_ratio > self.cfg.force.force_scale_up_threshold
        env_ids_scale_up = env_ids[env_ids_scale_up_mask]
        
        # Scale down if episode was too short (poor performance)
        env_ids_scale_down_mask = episode_length_ratio < self.cfg.force.force_scale_down_threshold
        env_ids_scale_down = env_ids[env_ids_scale_down_mask]
        
        # Update scales
        self.apply_force_scale[env_ids_scale_up] += self.cfg.force.force_scale_up
        self.apply_force_scale[env_ids_scale_down] -= self.cfg.force.force_scale_down
        
        # Clip the scale
        self.apply_force_scale[env_ids] = torch.clip(
            self.apply_force_scale[env_ids],
            self.cfg.force.force_scale_min,
            self.cfg.force.force_scale_max
        )
        
    def reset_idx(self, env_ids):
        """Override reset to include force curriculum and resampling"""
        # If force buffers haven't been initialized yet (this method can be
        # called from the parent __init__), delegate to the parent reset and
        # skip force-specific updates. The child __init__ will initialize the
        # force buffers right after the parent constructor returns.
        if not hasattr(self, 'apply_force_scale'):
            super().reset_idx(env_ids)
            return

        # Update force curriculum if enabled
        if self.cfg.force.force_scale_curriculum:
            self._update_force_scale_curriculum(env_ids)

        # Resample force settings
        self._resample_force_settings(env_ids)

        # Call parent reset
        super().reset_idx(env_ids)
        
    def compute_observations(self):
        """Override to add force observations"""
        # Call parent compute_observations
        super().compute_observations()
        # Add force observations to obs_buf. During parent initialization the
        # force-related buffers might not be initialized yet, so guard against
        # missing attributes and use zero tensors in that case.
        if not hasattr(self, 'left_ee_apply_force') or not hasattr(self, 'right_ee_apply_force'):
            left_force = torch.zeros((self.num_envs, 3), device=self.device)
            right_force = torch.zeros((self.num_envs, 3), device=self.device)
        else:
            left_force = self.left_ee_apply_force
            right_force = self.right_ee_apply_force

        force_obs = torch.cat([
            left_force,   # 3D
            right_force,  # 3D
        ], dim=-1)  # Total: 6D
        
        # Append to observation buffer
        # Note: This assumes obs_buf is [N, obs_dim]. We concatenate force at the end.
        if self.add_noise:
            force_obs_noisy = force_obs + torch.randn_like(force_obs) * self.obs_scales.dof_pos * 0.1
        else:
            force_obs_noisy = force_obs

        # Insert force observation into history entries so stacking remains consistent
        # Update obs_history last entry (if present) and rebuild obs_buf from history
        if hasattr(self, 'obs_history') and len(self.obs_history) > 0:
            # obs_history elements have shape [num_envs, K]
            self.obs_history[-1] = torch.cat([self.obs_history[-1], force_obs_noisy], dim=-1)
            obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)
            self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)
        else:
            # fallback: append directly to current obs_buf
            self.obs_buf = torch.cat([self.obs_buf, force_obs_noisy], dim=-1)

        # privileged obs: mirror same guarding as above and update critic_history
        if not hasattr(self, 'left_ee_apply_force') or not hasattr(self, 'right_ee_apply_force'):
            privileged_force_obs = force_obs
        else:
            privileged_force_obs = torch.cat([
                self.left_ee_apply_force,
                self.right_ee_apply_force,
            ], dim=-1)

        if hasattr(self, 'critic_history') and len(self.critic_history) > 0:
            self.critic_history[-1] = torch.cat([self.critic_history[-1], privileged_force_obs], dim=-1)
            # rebuild privileged_obs_buf from the (possibly) updated history
            self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)
        else:
            self.privileged_obs_buf = torch.cat([self.privileged_obs_buf, privileged_force_obs], dim=-1)
