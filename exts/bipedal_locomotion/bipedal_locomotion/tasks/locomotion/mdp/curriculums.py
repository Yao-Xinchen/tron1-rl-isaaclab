from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from exts.bipedal_locomotion.bipedal_locomotion.tasks.locomotion import mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def modify_event_parameter(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    param_name: str,
    value: Any | SceneEntityCfg,
    num_steps: int,
) -> torch.Tensor:
    """Curriculum that modifies a parameter of an event at a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the event term.
        param_name: The name of the event term parameter.
        value: The new value for the event term parameter.
        num_steps: The number of steps after which the change should be applied.

    Returns:
        torch.Tensor: Whether the parameter has already been modified or not.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.event_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.params[param_name] = value
        env.event_manager.set_term_cfg(term_name, term_cfg)
        return torch.ones(1)
    return torch.zeros(1)


def disable_termination(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    num_steps: int,
) -> torch.Tensor:
    """Curriculum that modifies the push velocity range at a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the termination term.
        num_steps: The number of steps after which the change should be applied.

    Returns:
        torch.Tensor: Whether the parameter has already been modified or not.
    """
    env.command_manager.num_envs
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.termination_manager.get_term_cfg(term_name)
        # Remove term settings
        term_cfg.params = dict()
        term_cfg.func = lambda env: torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env.termination_manager.set_term_cfg(term_name, term_cfg)
        return torch.ones(1)
    return torch.zeros(1)


def velocity_commands_ranges_level(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    max_range: dict[str, tuple[float, float]],
    update_interval: int = 50 * 24,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_twist",
) -> torch.Tensor:
    """Curriculum that progressively increases velocity command ranges.

    Returns:
        The current maximum velocity range.
    """
    # extract the used quantities (to enable type-hinting)
    command_cfg: mdp.UniformVelocityCommandCfg = env.command_manager.get_term(command_name).cfg
    current_vx = command_cfg.ranges.lin_vel_x[1]
    
    if env.common_step_counter % update_interval == 0:
        new_vx = command_cfg.ranges.lin_vel_x[1] + 0.1  # Increase by 0.1 m/s
        new_vy = command_cfg.ranges.lin_vel_y[1] + 0.04  # Increase by 0.04 m/s
        new_wz = command_cfg.ranges.ang_vel_z[1] + 0.1   # Increase by 0.1 rad/s
        
        # Clamp to maximum ranges
        new_vx = min(new_vx, max_range["lin_vel_x"][1])
        new_vy = min(new_vy, max_range["lin_vel_y"][1])
        new_wz = min(new_wz, max_range["ang_vel_z"][1])
        
        # Update ranges symmetrically
        command_cfg.ranges.lin_vel_x = (-new_vx, new_vx)
        command_cfg.ranges.lin_vel_y = (-new_vy, new_vy)
        command_cfg.ranges.ang_vel_z = (-new_wz, new_wz)
        current_vx = new_vx

    return torch.ones(1, dtype=torch.float) * current_vx
