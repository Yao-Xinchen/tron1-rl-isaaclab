import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.envs.mdp.commands import UniformPoseCommandCfg
from isaaclab.utils import configclass

from .gait_command import GaitCommand  # Import the GaitCommand class
from .pose_command import UniformWorldPoseCommand

@configclass
class UniformGaitCommandCfg(CommandTermCfg):
    """Configuration for the gait command generator."""

    class_type: type = GaitCommand  # Specify the class type for dynamic instantiation

    @configclass
    class Ranges:
        """Uniform distribution ranges for the gait parameters."""

        frequencies: tuple[float, float] = MISSING
        """Range for gait frequencies [Hz]."""
        offsets: tuple[float, float] = MISSING
        """Range for phase offsets [0-1]."""
        durations: tuple[float, float] = MISSING
        """Range for contact durations [0-1]."""
        swing_height: tuple[float, float] = MISSING
        """Range for contact durations [0-1]."""

    ranges: Ranges = MISSING
    """Distribution ranges for the gait parameters."""

    resampling_time_range: tuple[float, float] = MISSING
    """Time interval for resampling the gait (in seconds)."""


@configclass
class UniformWorldPoseCommandCfg(UniformPoseCommandCfg):
    se3_decrease_vel_range: tuple[float, float] = (0.5, 1.4)
    resampling_time_scale: tuple[float, float] = (6.0, 15.0)

    class_type: type = UniformWorldPoseCommand
