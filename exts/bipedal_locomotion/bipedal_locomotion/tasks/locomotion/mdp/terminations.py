from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm

def bad_orientation_stochastic(
    env: ManagerBasedRLEnv, limit_angle: float, probability: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    bad_orientation = torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle
    random_values = torch.rand(bad_orientation.shape, device=bad_orientation.device)
    return bad_orientation & (random_values < probability)

def bad_height_stochastic(
    env: ManagerBasedRLEnv, limit_height: float, probability: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    foot_position = asset.data.body_pos_w[:, env._wheels_link_ids, :]
    height = asset.data.root_link_pos_w[:, 2] - foot_position[:, :, 2].mean(dim=-1) + env._foot_radius
    bad_height = height < limit_height
    random_values = torch.rand(bad_height.shape, device=bad_height.device)
    return bad_height & (random_values < probability)
