"""Script to evaluate velocity tracking performance across different command directions."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from isaaclab.app import AppLauncher

# local imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate velocity tracking across directions.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")

# experiment-specific arguments
parser.add_argument("--warmup_steps", type=int, default=200, help="Number of warmup steps before data collection.")
parser.add_argument("--collection_steps", type=int, default=600, help="Number of steps for data collection.")
parser.add_argument("--num_angle_bins", type=int, default=128, help="Number of angular bins for analysis.")
parser.add_argument("--output_dir", type=str, default="experiments/tracking_error/twist", help="Directory to save results.")
parser.add_argument("--velocity_magnitude", type=float, default=1.0, help="Magnitude of velocity commands on the circle.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

from rsl_rl.runner import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Import extensions to set up environment tasks
import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg


def sample_unit_circle_commands(num_envs: int, magnitude: float = 1.0, device: str = "cuda"):
    """Sample velocity commands uniformly on a circle.

    Args:
        num_envs: Number of environments (robots)
        magnitude: Magnitude of velocity (radius of circle)
        device: Device to create tensors on

    Returns:
        commands: Tensor of shape (num_envs, 3) with [vx, vy, angular_z]
        angles: Tensor of shape (num_envs,) with angle of each command
    """
    # Sample angles uniformly from [0, 2*pi)
    angles = torch.rand(num_envs, device=device) * 2 * np.pi

    # Convert to cartesian coordinates
    vx = magnitude * torch.cos(angles)
    vy = magnitude * torch.sin(angles)
    ang_vel_z = torch.zeros(num_envs, device=device)

    # Stack into commands tensor
    commands = torch.stack([vx, vy, ang_vel_z], dim=1)

    return commands, angles


def override_velocity_commands(env, commands: torch.Tensor):
    """Override the velocity commands in the environment's command manager.

    Args:
        env: The wrapped environment
        commands: Tensor of shape (num_envs, 3) with [vx, vy, angular_z]
    """
    # Access the command term and set commands directly
    command_term = env.unwrapped.command_manager._terms["base_twist"]
    command_term.command[:] = commands


def calculate_velocity_error(env, target_commands: torch.Tensor):
    """Calculate velocity tracking error in body frame.

    Args:
        env: The wrapped environment
        target_commands: Tensor of shape (num_envs, 3) with [vx, vy, angular_z]

    Returns:
        error: Tensor of shape (num_envs,) with L2 error magnitude
    """
    # Get actual velocity in world frame and robot orientation
    robot = env.unwrapped.scene["robot"]
    actual_vel_w = robot.data.root_vel_w  # Shape: (num_envs, 6) - [vx, vy, vz, wx, wy, wz]
    root_quat_w = robot.data.root_quat_w  # Shape: (num_envs, 4) - [w, x, y, z]

    # Transform linear velocity from world to body frame
    actual_lin_vel_b = quat_apply_inverse(root_quat_w, actual_vel_w[:, :3])

    # Extract linear velocities (x, y) in body frame
    actual_linear = actual_lin_vel_b[:, :2]
    target_linear = target_commands[:, :2]

    # Calculate L2 error
    error = torch.norm(actual_linear - target_linear, dim=1)

    return error


def run_experiment():
    """Run the velocity tracking experiment."""

    # Parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg.seed = agent_cfg.seed if args_cli.seed is None else args_cli.seed

    # Specify directory for logging experiments
    if args_cli.checkpoint_path is None:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = args_cli.checkpoint_path
    log_dir = os.path.dirname(resume_path)

    # Create output directory with datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args_cli.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Results will be saved to: {output_dir}")

    # Create isaac environment
    print(f"[INFO] Creating environment with {args_cli.num_envs} robots...")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # Convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # Load previously trained model
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # Obtain the trained policy for inference
    student_policy = ppo_runner.get_inference_policy_student(device=env.unwrapped.device)

    # Sample velocity commands on unit circle
    print(f"[INFO] Sampling velocity commands on circle with magnitude {args_cli.velocity_magnitude}")
    commands, command_angles = sample_unit_circle_commands(
        args_cli.num_envs,
        magnitude=args_cli.velocity_magnitude,
        device=env.unwrapped.device
    )

    # Override the commands
    override_velocity_commands(env, commands)

    # Reset environment
    obs, obs_dict = env.get_observations()
    obs_history = obs_dict["observations"].get("obsHistory")
    obs_history = obs_history.flatten(start_dim=1)
    critic_obs = obs_dict["observations"].get("critic")
    commands_obs = obs_dict["observations"].get("commands")

    print(f"\n[INFO] Starting experiment...")
    print(f"  Warmup steps: {args_cli.warmup_steps}")
    print(f"  Collection steps: {args_cli.collection_steps}")

    # Initialize data collection
    error_accumulator = torch.zeros(args_cli.num_envs, device=env.unwrapped.device)
    error_count = 0

    # Run simulation
    total_steps = args_cli.warmup_steps + args_cli.collection_steps

    for step in range(total_steps):
        # Override commands at each step to prevent resampling
        override_velocity_commands(env, commands)

        # Run policy inference
        with torch.inference_mode():
            # Agent stepping
            actions = student_policy(obs, obs_history, commands_obs)

            # Env stepping
            obs, _, _, infos = env.step(actions)
            obs_history = infos["observations"].get("obsHistory")
            obs_history = obs_history.flatten(start_dim=1)
            critic_obs = infos["observations"].get("critic")
            commands_obs = infos["observations"].get("commands")

        # Override commands again after step (in case command manager resamples)
        override_velocity_commands(env, commands)

        # Start collecting after warmup
        if step >= args_cli.warmup_steps:
            # Calculate and accumulate errors
            errors = calculate_velocity_error(env, commands)
            error_accumulator += errors
            error_count += 1

            # Print progress
            if (step - args_cli.warmup_steps + 1) % 100 == 0:
                current_mean_error = (error_accumulator / error_count).mean().item()
                print(f"  Collection step {step - args_cli.warmup_steps + 1}/{args_cli.collection_steps}, "
                      f"Current mean error: {current_mean_error:.4f} m/s")

    # Calculate average errors
    avg_errors = error_accumulator / error_count

    print(f"\n[INFO] Data collection complete!")
    print(f"  Overall mean error: {avg_errors.mean().item():.4f} m/s")
    print(f"  Overall std error: {avg_errors.std().item():.4f} m/s")
    print(f"  Min error: {avg_errors.min().item():.4f} m/s")
    print(f"  Max error: {avg_errors.max().item():.4f} m/s")

    # Convert to numpy for analysis
    avg_errors_np = avg_errors.cpu().numpy()
    command_angles_np = command_angles.cpu().numpy()
    commands_np = commands.cpu().numpy()

    # Perform analysis and visualization
    print(f"\n[INFO] Generating visualizations...")
    analyze_and_visualize(
        avg_errors_np,
        command_angles_np,
        commands_np,
        output_dir,
        args_cli.num_angle_bins
    )

    # Save raw data
    save_raw_data(avg_errors_np, command_angles_np, commands_np, output_dir)

    print(f"\n[INFO] Experiment complete! Results saved to {output_dir}")

    # Close the environment
    env.close()


def analyze_and_visualize(errors, angles, commands, output_dir, num_bins):
    """Analyze data and create visualizations.

    Args:
        errors: Array of shape (num_envs,) with average errors
        angles: Array of shape (num_envs,) with command angles
        commands: Array of shape (num_envs, 3) with command velocities
        output_dir: Path to save figures
        num_bins: Number of angular bins for analysis
    """

    # Calculate percentile-based limits for better visualization
    error_95th = np.percentile(errors, 95)
    error_max_display = min(error_95th * 1.2, 2.0)  # Cap at 2.0 m/s for readability

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))

    # 1. Polar scatter plot with limited radius
    ax1 = fig.add_subplot(2, 3, 1, projection='polar')
    ax1.set_theta_zero_location('N')  # Set 0 degrees to point upward
    # Filter data for visualization (but keep all for statistics)
    display_mask = errors <= error_max_display
    scatter = ax1.scatter(angles[display_mask], errors[display_mask], c=errors[display_mask],
                         cmap='RdYlBu_r', alpha=0.6, s=10, vmin=errors.min(), vmax=error_max_display)
    ax1.set_ylim(0, error_max_display)
    ax1.set_title(f'Tracking Error vs Direction\n(showing errors ≤ {error_max_display:.2f} m/s, {display_mask.sum()}/{len(errors)} robots)',
                  fontsize=11, pad=20)
    ax1.set_ylabel('Error (m/s)', labelpad=30)
    plt.colorbar(scatter, ax=ax1, label='Error (m/s)')

    # 2. Cartesian scatter plot with limited y-axis
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(np.rad2deg(angles), errors, alpha=0.3, s=10, c='steelblue')
    ax2.set_xlabel('Command Angle (degrees)')
    ax2.set_ylabel('Tracking Error (m/s)')
    ax2.set_title('Error vs Angle (All Data)')
    ax2.set_ylim(0, error_max_display)
    ax2.grid(True, alpha=0.3)

    # Add reference lines for key directions
    for angle, label, color in [(0, 'Forward', 'green'), (90, 'Right', 'orange'),
                                  (180, 'Backward', 'red'), (270, 'Left', 'orange')]:
        ax2.axvline(angle, color=color, linestyle='--', alpha=0.5, linewidth=1)
        ax2.text(angle, error_max_display * 0.95, label, rotation=90,
                verticalalignment='top', fontsize=8, color=color)

    # 3. Histogram by angle bins with error bars and smoothed line
    ax3 = fig.add_subplot(2, 3, 3)
    bin_edges = np.linspace(0, 2*np.pi, num_bins + 1)
    bin_indices = np.digitize(angles, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    bin_means = []
    bin_stds = []
    bin_centers = []

    for i in range(num_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means.append(errors[mask].mean())
            bin_stds.append(errors[mask].std())
            bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)

    bin_centers_deg = np.rad2deg(bin_centers)
    ax3.bar(bin_centers_deg, bin_means, width=360/num_bins, alpha=0.6,
            yerr=bin_stds, capsize=2, color='steelblue', edgecolor='black', linewidth=0.5)

    # Add smoothed line
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(bin_means, size=max(3, num_bins//20), mode='wrap')
    ax3.plot(bin_centers_deg, smoothed, 'r-', linewidth=2, label='Smoothed trend')

    # Add mean line
    ax3.axhline(errors.mean(), color='green', linestyle='--', linewidth=2, label=f'Overall mean: {errors.mean():.3f}')

    # Add direction markers
    for angle, label in [(0, 'Fwd'), (90, 'Right'), (180, 'Back'), (270, 'Left')]:
        ax3.axvline(angle, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    ax3.set_xlabel('Command Angle (degrees)')
    ax3.set_ylabel('Mean Tracking Error (m/s)')
    ax3.set_title(f'Mean Error by Angular Bin (n={num_bins})')
    ax3.set_xlim(0, 360)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Error distribution histogram
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax4.axvline(errors.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.4f}')
    ax4.axvline(np.median(errors), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.4f}')
    ax4.set_xlabel('Tracking Error (m/s)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Error Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Command vector visualization
    ax5 = fig.add_subplot(2, 3, 5)
    scatter = ax5.scatter(commands[:, 0], commands[:, 1], c=errors, cmap='viridis', alpha=0.6, s=10)
    ax5.set_xlabel('Command Vx (m/s)')
    ax5.set_ylabel('Command Vy (m/s)')
    ax5.set_title('Command Distribution with Errors')
    ax5.axis('equal')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Error (m/s)')

    # 6. Statistics table
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    stats_text = f"""
    EXPERIMENT STATISTICS

    Overall Statistics:
    " Number of robots:        {len(errors)}
    " Mean error:              {errors.mean():.4f} m/s
    " Std error:               {errors.std():.4f} m/s
    " Median error:            {np.median(errors):.4f} m/s
    " Min error:               {errors.min():.4f} m/s
    " Max error:               {errors.max():.4f} m/s
    " 25th percentile:         {np.percentile(errors, 25):.4f} m/s
    " 75th percentile:         {np.percentile(errors, 75):.4f} m/s

    Directional Analysis:
    " Number of angle bins:    {num_bins}
    " Best direction (angle):  {np.rad2deg(angles[errors.argmin()]):.1f}
    " Worst direction (angle): {np.rad2deg(angles[errors.argmax()]):.1f}
    " Angular bin mean error:  {np.array(bin_means).mean():.4f} m/s
    " Angular bin std:         {np.array(bin_means).std():.4f} m/s
    """

    ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'velocity_tracking_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved visualization to: {output_path}")

    # Create separate polar plot with better resolution and limited scale
    fig2, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.set_theta_zero_location('N')  # Set 0 degrees to point upward
    # Use same filtering for consistency
    scatter = ax.scatter(angles[display_mask], errors[display_mask], c=errors[display_mask],
                        cmap='RdYlBu_r', alpha=0.6, s=20, vmin=errors.min(), vmax=error_max_display)
    ax.set_ylim(0, error_max_display)
    ax.set_title(f'Velocity Tracking Error by Command Direction\n(showing {display_mask.sum()}/{len(errors)} robots with errors ≤ {error_max_display:.2f} m/s)',
                fontsize=13, pad=20)
    ax.set_ylabel('Error (m/s)', labelpad=40)
    plt.colorbar(scatter, ax=ax, label='Error (m/s)', pad=0.1)

    # Calculate directional mean and median using angular bins
    polar_bins = 72  # Higher resolution for smoother curves
    polar_bin_edges = np.linspace(0, 2*np.pi, polar_bins + 1)
    polar_bin_indices = np.digitize(angles, polar_bin_edges) - 1
    polar_bin_indices = np.clip(polar_bin_indices, 0, polar_bins - 1)

    polar_bin_means = []
    polar_bin_medians = []
    polar_bin_centers = []

    for i in range(polar_bins):
        mask = polar_bin_indices == i
        if mask.sum() > 0:
            polar_bin_means.append(errors[mask].mean())
            polar_bin_medians.append(np.median(errors[mask]))
            polar_bin_centers.append((polar_bin_edges[i] + polar_bin_edges[i+1]) / 2)
        else:
            # If no data in bin, interpolate from neighbors
            polar_bin_means.append(np.nan)
            polar_bin_medians.append(np.nan)
            polar_bin_centers.append((polar_bin_edges[i] + polar_bin_edges[i+1]) / 2)

    # Convert to arrays and interpolate any missing values
    polar_bin_means = np.array(polar_bin_means)
    polar_bin_medians = np.array(polar_bin_medians)
    polar_bin_centers = np.array(polar_bin_centers)

    # Fill NaN values with interpolation
    if np.any(np.isnan(polar_bin_means)):
        valid_mask = ~np.isnan(polar_bin_means)
        polar_bin_means = np.interp(polar_bin_centers, polar_bin_centers[valid_mask], polar_bin_means[valid_mask])
        polar_bin_medians = np.interp(polar_bin_centers, polar_bin_centers[valid_mask], polar_bin_medians[valid_mask])

    # Smooth the curves for better visualization
    from scipy.ndimage import uniform_filter1d
    smooth_means = uniform_filter1d(polar_bin_means, size=max(3, polar_bins//15), mode='wrap')
    smooth_medians = uniform_filter1d(polar_bin_medians, size=max(3, polar_bins//15), mode='wrap')

    # Close the curve by appending first point at the end
    theta_closed = np.append(polar_bin_centers, polar_bin_centers[0])
    means_closed = np.append(smooth_means, smooth_means[0])
    medians_closed = np.append(smooth_medians, smooth_medians[0])

    # Plot directional curves
    ax.plot(theta_closed, means_closed, 'r-', linewidth=2.5,
            label=f'Directional mean: {errors.mean():.3f} m/s (avg)')
    ax.plot(theta_closed, medians_closed, 'g-', linewidth=2.5,
            label=f'Directional median: {np.median(errors):.3f} m/s (avg)')

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    output_path2 = output_dir / 'velocity_tracking_polar.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"  Saved polar plot to: {output_path2}")

    plt.close('all')


def save_raw_data(errors, angles, commands, output_dir):
    """Save raw data to CSV file.

    Args:
        errors: Array of shape (num_envs,) with average errors
        angles: Array of shape (num_envs,) with command angles
        commands: Array of shape (num_envs, 3) with command velocities
        output_dir: Path to save CSV
    """
    import csv

    output_path = output_dir / 'tracking_errors.csv'

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['robot_id', 'command_angle_rad', 'command_angle_deg',
                        'command_vx', 'command_vy', 'command_wz', 'avg_error'])

        for i in range(len(errors)):
            writer.writerow([
                i,
                angles[i],
                np.rad2deg(angles[i]),
                commands[i, 0],
                commands[i, 1],
                commands[i, 2],
                errors[i]
            ])

    print(f"  Saved raw data to: {output_path}")

    # Also save summary statistics
    summary_path = output_dir / 'summary_statistics.txt'
    with open(summary_path, 'w') as f:
        f.write("VELOCITY TRACKING EXPERIMENT SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Number of robots:          {len(errors)}\n")
        f.write(f"Velocity magnitude:        {args_cli.velocity_magnitude} m/s\n")
        f.write(f"Warmup steps:              {args_cli.warmup_steps}\n")
        f.write(f"Collection steps:          {args_cli.collection_steps}\n")
        f.write(f"Angular bins:              {args_cli.num_angle_bins}\n\n")
        f.write("ERROR STATISTICS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Mean error:                {errors.mean():.6f} m/s\n")
        f.write(f"Std error:                 {errors.std():.6f} m/s\n")
        f.write(f"Median error:              {np.median(errors):.6f} m/s\n")
        f.write(f"Min error:                 {errors.min():.6f} m/s\n")
        f.write(f"Max error:                 {errors.max():.6f} m/s\n")
        f.write(f"25th percentile:           {np.percentile(errors, 25):.6f} m/s\n")
        f.write(f"75th percentile:           {np.percentile(errors, 75):.6f} m/s\n\n")
        f.write("DIRECTIONAL ANALYSIS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Best direction (angle):    {np.rad2deg(angles[errors.argmin()]):.2f}�\n")
        f.write(f"Best direction error:      {errors.min():.6f} m/s\n")
        f.write(f"Worst direction (angle):   {np.rad2deg(angles[errors.argmax()]):.2f}�\n")
        f.write(f"Worst direction error:     {errors.max():.6f} m/s\n")

    print(f"  Saved summary statistics to: {summary_path}")


if __name__ == "__main__":
    # Run the experiment
    run_experiment()
    # Close sim app
    simulation_app.close()
