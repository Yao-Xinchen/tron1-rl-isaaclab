# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from isaaclab.utils.math import (
    quat_inv,
    quat_mul,
    compute_pose_error,
    quat_from_euler_xyz,
    quat_unique,
    quat_apply_inverse,
    quat_from_matrix,
    sample_uniform,
    axis_angle_from_quat,

)

from isaaclab.envs.mdp.commands import UniformPoseCommandCfg
from isaaclab.envs.mdp import UniformPoseCommand
from isaaclab.envs import ManagerBasedEnv

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def generate_sigmoid_scale(mu: float, decay_length: float, x: torch.Tensor):
    sigmoid_z = 5 / decay_length * (x - mu)
    return torch.sigmoid(sigmoid_z)

def compute_rotation_distance(input_quat, target_quat):
    Ee_target_R = quaternion_to_matrix(target_quat)
    Ee_R = quaternion_to_matrix(input_quat)

    # Calculate the rotation distance (Frobenius norm of log(R1^T * R2))

    R_rel = torch.matmul(torch.transpose(Ee_target_R, 1, 2), Ee_R)
    trace_R_rel = torch.einsum("bii->b", R_rel)

    # Clamping to avoid numerical issues with arccos
    trace_clamped = torch.clamp((trace_R_rel - 1) / 2, -1.0, 1.0)
    rotation_distance = torch.acos(trace_clamped)
    return rotation_distance


class UniformWorldPoseCommand(UniformPoseCommand):

    def __init__(self, cfg: UniformWorldPoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.decrease_vel = torch.zeros(self.num_envs, device=self.device)
        self.se3_distance_ref = torch.ones(self.num_envs, device=self.device) * 5.0
        self.decrease_vel_range = cfg.se3_decrease_vel_range
        self._env._loco_mani_scale = torch.ones(self.num_envs, device=self.device)  # type: ignore
        self.resampling_time_scale = cfg.resampling_time_scale
        self.resample_time_range = cfg.resampling_time_range

        self.optim_pos_distance = torch.zeros(self.num_envs, device=self.device)
        self.optim_orient_distance = torch.zeros(self.num_envs, device=self.device)
        self.pos_improvement = torch.zeros(self.num_envs, device=self.device)
        self.orient_improvement = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        # refresh the pose_command_b
        self.pose_command_b[:, :3] = quat_apply_inverse(
            self.robot.data.root_link_quat_w, self.pose_command_w[:, :3] - self.robot.data.root_link_pos_w
        )
        self.pose_command_b[:, 3:] = quat_unique(
            quat_mul(quat_inv(self.robot.data.root_link_quat_w), self.pose_command_w[:, 3:])
        )

        # compute the error
        pos_error = self.pose_command_w[:, :3] - self.robot.data.root_link_pos_w
        rot_error_angle = compute_rotation_distance(
            quat_unique(self.robot.data.root_link_quat_w), self.pose_command_w[:, 3:]
        )

        self.metrics["position_error"] = torch.norm(pos_error[:, :2], dim=-1)
        self.metrics["orientation_error"] = rot_error_angle
        self.se3_distance_ref -= self.decrease_vel * self._env.step_dt
        self.se3_distance_ref = torch.clamp(self.se3_distance_ref, min=0.0)
        self.pos_improvement = (self.optim_pos_distance - self.metrics["position_error"]).clip(min=0.0)
        self.orient_improvement = (self.optim_orient_distance - self.metrics["orientation_error"]).clip(min=0.0)
        self.optim_pos_distance[:] = torch.minimum(self.metrics["position_error"], self.optim_pos_distance)
        self.optim_orient_distance[:] = torch.minimum(self.metrics["orientation_error"], self.optim_orient_distance)
        self._env._loco_mani_scale = generate_sigmoid_scale(mu=1.0, decay_length=1.0,
                                                            x=self.se3_distance_ref)  # type: ignore

    def _update_se3_ref(self, env_ids: Sequence[int]):
        self.pose_command_b[:, :3] = quat_apply_inverse(
            self.robot.data.root_link_quat_w, self.pose_command_w[:, :3] - self.robot.data.root_link_pos_w
        )
        self.pose_command_b[:, 3:] = quat_unique(
            quat_mul(quat_inv(self.robot.data.root_link_quat_w), self.pose_command_w[:, 3:])
        )

        # compute the error
        pos_error = self.pose_command_w[:, :3] - self.robot.data.root_link_pos_w
        rot_error_angle = compute_rotation_distance(
            quat_unique(self.robot.data.root_link_quat_w), self.pose_command_w[:, 3:7]
        )

        self.metrics["position_error"] = torch.norm(pos_error[:, :2], dim=-1)
        self.metrics["orientation_error"] = rot_error_angle

        self.se3_distance_ref[env_ids] = (
                2 * self.metrics["position_error"][env_ids] + self.metrics["orientation_error"][env_ids]
        )
        self.optim_pos_distance[env_ids] = self.metrics["position_error"][env_ids]
        self.optim_orient_distance[env_ids] = self.metrics["orientation_error"][env_ids]

        self.pos_improvement[env_ids] = 0.0
        self.orient_improvement[env_ids] = 0.0

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # -- position
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_w[env_ids, 0] = self.robot.data.root_link_pos_w[env_ids, 0] + r.uniform_(
            *self.cfg.ranges.pos_x
        )
        self.pose_command_w[env_ids, 1] = self.robot.data.root_link_pos_w[env_ids, 1] + r.uniform_(
            *self.cfg.ranges.pos_y
        )
        self.pose_command_w[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_w[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # make sure the quaternion has real part as positive
        self.pose_command_w[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat
        self.decrease_vel[env_ids] = sample_uniform(
            self.decrease_vel_range[0], self.decrease_vel_range[1], len(env_ids), device=self.device
        )

    def _resample(self, env_ids):
        """Resample the command.

        This function resamples the command and time for which the command is applied for the
        specified environment indices.

        Args:
            env_ids: The list of environment IDs to resample.
        """
        # resample the time left before resampling
        if len(env_ids) != 0:
            # self.metrics["contact_time"][env_ids] = 0.0
            self._resample_command(env_ids)
            self._update_se3_ref(env_ids)
            # self._update_metrics()
            se3_error = 2 * self.metrics["position_error"][env_ids] + self.metrics["orientation_error"][env_ids]
            random_scale = sample_uniform(
                self.resampling_time_scale[0], self.resampling_time_scale[1], len(env_ids), device=self.device
            )
            self.time_left[env_ids] = (se3_error * random_scale).clip(
                min=self.resample_time_range[0], max=self.resample_time_range[1]
            )
            # increment the command counter
            self.command_counter[env_ids] += 1
            # resample the command
