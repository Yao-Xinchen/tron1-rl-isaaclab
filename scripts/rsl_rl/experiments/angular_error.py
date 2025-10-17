"""Script to analyze angular velocity tracking error vs command angular velocity magnitude."""

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
parser = argparse.ArgumentParser(description="Analyze angular velocity tracking error vs angular velocity magnitude.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")

# experiment-specific arguments
parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps before data collection.")
parser.add_argument("--collection_steps", type=int, default=500, help="Number of steps for data collection.")
parser.add_argument("--dt", type=float, default=0.02, help="Simulation timestep in seconds.")
parser.add_argument("--output_dir", type=str, default="experiments/angular_error", help="Directory to save results.")
parser.add_argument("--angular_vel_range", type=float, default=2.0, help="Angular velocity command range in rad/s (symmetric).")
parser.add_argument("--num_bins", type=int, default=10, help="Number of bins for angular velocity binning.")
parser.add_argument("--linear_vel", type=float, default=0.0, help="Fixed linear velocity in m/s (default 0 for pure rotation).")

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


def sample_angular_velocity_commands(num_envs, angular_vel_range, linear_vel, device):
    """Sample angular velocity commands with magnitude-based sampling.

    Args:
        num_envs: Number of environments
        angular_vel_range: Maximum angular velocity magnitude in rad/s
        linear_vel: Fixed linear velocity in m/s
        device: Torch device

    Returns:
        vel_commands: Tensor of shape (num_envs, 3) with [vel_x, vel_y, vel_yaw]
        angular_vel_magnitudes: Tensor of shape (num_envs,) with angular velocity magnitudes
    """
    # Sample angular velocity magnitude uniformly from [-angular_vel_range, angular_vel_range] rad/s
    # Use full range including negative values for rotation in both directions
    angular_vel_magnitudes = torch.rand(num_envs, device=device) * 2 * angular_vel_range - angular_vel_range

    # Set fixed linear velocity (default 0 for pure rotation)
    vel_x = torch.full((num_envs,), linear_vel, device=device)
    vel_y = torch.zeros(num_envs, device=device)

    # Set angular velocity
    vel_yaw = angular_vel_magnitudes

    vel_commands = torch.stack([vel_x, vel_y, vel_yaw], dim=1)

    return vel_commands, angular_vel_magnitudes


def override_velocity_commands(env, vel_commands):
    """Override pose commands to achieve velocity control for each robot.

    Sets target pose to robot's current pose (zero relative distance)
    and sets velocity command in the command frame.

    Args:
        env: The wrapped environment
        vel_commands: Tensor of shape (num_envs, 3) with [vel_x, vel_y, vel_yaw]
    """
    command_term = env.unwrapped.command_manager._terms["base_pose"]
    robot = env.unwrapped.scene["robot"]

    # Set target pose to robot's current pose (zero relative distance)
    command_term.pose_command_w[:, :3] = robot.data.root_link_pos_w.clone()
    command_term.pose_command_w[:, 3:] = robot.data.root_link_quat_w.clone()

    # Set velocity commands in command frame (body frame when pose is at robot)
    command_term.pose_command_vel_c[:, 0] = vel_commands[:, 0]
    command_term.pose_command_vel_c[:, 1] = vel_commands[:, 1]
    command_term.pose_command_vel_c[:, 2] = vel_commands[:, 2]


def calculate_angular_velocity_error(env, target_vel_commands):
    """Calculate angular velocity tracking error in body frame for all environments.

    Args:
        env: The wrapped environment
        target_vel_commands: Tensor of shape (num_envs, 3) with [vel_x, vel_y, vel_yaw]

    Returns:
        error: Tensor of shape (num_envs,) with absolute angular velocity error
    """
    robot = env.unwrapped.scene["robot"]

    # Get actual angular velocity in world frame and robot orientation
    actual_vel_w = robot.data.root_vel_w  # Shape: (num_envs, 6) - [vx, vy, vz, wx, wy, wz]
    root_quat_w = robot.data.root_quat_w  # Shape: (num_envs, 4) - [w, x, y, z]

    # Transform angular velocity from world to body frame
    actual_ang_vel_b = quat_apply_inverse(root_quat_w, actual_vel_w[:, 3:])

    # Extract yaw angular velocity (z-axis) in body frame
    actual_angular = actual_ang_vel_b[:, 2]
    target_angular = target_vel_commands[:, 2]

    # Calculate absolute error
    error = torch.abs(actual_angular - target_angular)

    return error


def run_experiment():
    """Run the angular velocity error vs angular velocity magnitude experiment."""

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

    # Sample angular velocity commands for each robot
    print(f"\n[INFO] Sampling angular velocity commands...")
    vel_commands, angular_vel_magnitudes = sample_angular_velocity_commands(
        args_cli.num_envs, args_cli.angular_vel_range, args_cli.linear_vel, env.unwrapped.device
    )

    print(f"[INFO] Assigned velocity commands to {len(vel_commands)} robots")
    print(f"  Angular velocity range: [{angular_vel_magnitudes.min():.3f}, {angular_vel_magnitudes.max():.3f}] rad/s")
    print(f"  Linear velocity (fixed): {args_cli.linear_vel:.3f} m/s")

    # Load previously trained model
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # Obtain the trained policy for inference
    student_policy = ppo_runner.get_inference_policy_student(device=env.unwrapped.device)

    # Reset environment
    obs, obs_dict = env.get_observations()
    obs_history = obs_dict["observations"].get("obsHistory")
    obs_history = obs_history.flatten(start_dim=1)
    critic_obs = obs_dict["observations"].get("critic")
    commands_obs = obs_dict["observations"].get("commands")

    # Calculate total steps
    total_steps = args_cli.warmup_steps + args_cli.collection_steps

    print(f"\n[INFO] Starting experiment...")
    print(f"  Warmup steps: {args_cli.warmup_steps}")
    print(f"  Collection steps: {args_cli.collection_steps}")
    print(f"  Total steps: {total_steps}")
    print(f"  Expected duration: {total_steps * args_cli.dt:.2f} seconds")

    # Initialize accumulators for average tracking error
    error_sum = torch.zeros(args_cli.num_envs, device=env.unwrapped.device)
    error_count = 0

    # Run simulation
    for step in range(total_steps):
        # Override velocity commands at each step to maintain velocity control
        override_velocity_commands(env, vel_commands)

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

        # Override velocity commands again after step
        override_velocity_commands(env, vel_commands)

        # Start recording after warmup
        if step >= args_cli.warmup_steps:
            # Get angular velocity error using real robot angular velocity
            angular_velocity_errors = calculate_angular_velocity_error(env, vel_commands)

            # Accumulate errors
            error_sum += angular_velocity_errors
            error_count += 1

            # Print progress
            if (step - args_cli.warmup_steps) % 50 == 0:
                print(f"  Collection step {step - args_cli.warmup_steps}/{args_cli.collection_steps}, "
                      f"mean angular velocity error: {angular_velocity_errors.mean().item():.4f} rad/s")

    # Calculate average tracking error per robot
    avg_tracking_errors = error_sum / error_count

    print(f"\n[INFO] Data collection complete!")
    print(f"  Collection steps: {error_count}")
    print(f"  Overall mean tracking error: {avg_tracking_errors.mean().item():.4f} rad/s")
    print(f"  Overall std tracking error: {avg_tracking_errors.std().item():.4f} rad/s")

    # Move data to CPU for processing
    angular_vel_magnitudes_cpu = angular_vel_magnitudes.cpu().numpy()
    vel_commands_cpu = vel_commands.cpu().numpy()
    avg_tracking_errors_cpu = avg_tracking_errors.cpu().numpy()

    # Process and visualize data
    print(f"\n[INFO] Generating visualizations...")
    visualize_results(angular_vel_magnitudes_cpu, vel_commands_cpu, avg_tracking_errors_cpu, output_dir)

    # Save data
    save_data(angular_vel_magnitudes_cpu, vel_commands_cpu, avg_tracking_errors_cpu, output_dir)

    print(f"\n[INFO] Experiment complete! Results saved to {output_dir}")

    # Close the environment
    env.close()


def visualize_results(angular_vel_magnitudes, vel_commands, avg_errors, output_dir):
    """Create visualizations for angular velocity magnitude vs tracking error.

    Args:
        angular_vel_magnitudes: Array of shape (num_envs,) with angular velocity magnitudes
        vel_commands: Array of shape (num_envs, 3) with [vel_x, vel_y, vel_yaw]
        avg_errors: Array of shape (num_envs,) with average tracking errors
        output_dir: Directory to save plots
    """

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Scatter plot
    scatter = ax1.scatter(angular_vel_magnitudes, avg_errors, alpha=0.3, s=10,
                         c=np.abs(angular_vel_magnitudes), cmap='viridis')
    ax1.set_xlabel('Command Angular Velocity (rad/s)', fontsize=12)
    ax1.set_ylabel('Average Angular Velocity Tracking Error (rad/s)', fontsize=12)
    ax1.set_title('Angular Velocity Tracking Error vs Command Angular Velocity', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('|Angular Velocity| (rad/s)', fontsize=10)

    # Add trendline (polynomial fit)
    if len(angular_vel_magnitudes) > 10:
        # Sort by angular velocity for better visualization
        sort_idx = np.argsort(angular_vel_magnitudes)
        ang_vel_sorted = angular_vel_magnitudes[sort_idx]
        errors_sorted = avg_errors[sort_idx]

        # Use absolute value for fitting
        z = np.polyfit(np.abs(ang_vel_sorted), errors_sorted, 2)
        p = np.poly1d(z)
        ang_vel_plot = np.linspace(angular_vel_magnitudes.min(), angular_vel_magnitudes.max(), 100)
        ax1.plot(ang_vel_plot, p(np.abs(ang_vel_plot)), "r--", linewidth=2, alpha=0.8, label='Quadratic Fit')
        ax1.legend(fontsize=10)

    # Plot 2: Binned violin plot (using absolute angular velocity for binning)
    num_bins = args_cli.num_bins
    abs_angular_vel = np.abs(angular_vel_magnitudes)
    bin_edges = np.linspace(0, args_cli.angular_vel_range, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Collect data for each bin
    bin_data = []
    bin_counts = []
    valid_positions = []

    for i in range(num_bins):
        mask = (abs_angular_vel >= bin_edges[i]) & (abs_angular_vel < bin_edges[i + 1])
        if i == num_bins - 1:  # Include right edge for last bin
            mask = (abs_angular_vel >= bin_edges[i]) & (abs_angular_vel <= bin_edges[i + 1])

        if mask.sum() > 0:
            bin_data.append(avg_errors[mask])
            bin_counts.append(mask.sum())
            valid_positions.append(bin_centers[i])

    # Create violin plot
    if len(bin_data) > 0:
        parts = ax2.violinplot(bin_data, positions=valid_positions,
                              widths=args_cli.angular_vel_range / num_bins * 0.6,
                              showmeans=True, showmedians=False, showextrema=False)

        # Customize violin plot colors
        for pc in parts['bodies']:
            pc.set_facecolor('darkorange')
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)

        # Customize mean line
        parts['cmeans'].set_color('red')
        parts['cmeans'].set_linewidth(2)

    ax2.set_xlabel('|Command Angular Velocity| (rad/s)', fontsize=12)
    ax2.set_ylabel('Average Angular Velocity Tracking Error (rad/s)', fontsize=12)
    ax2.set_title(f'Distribution of Tracking Error (n={num_bins} bins)', fontsize=13)
    ax2.set_xlim(-0.05 * args_cli.angular_vel_range, args_cli.angular_vel_range * 1.05)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'angular_velocity_vs_error.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved visualization to: {output_path}")

    plt.close('all')


def save_data(angular_vel_magnitudes, vel_commands, avg_errors, output_dir):
    """Save experiment data to CSV and summary file.

    Args:
        angular_vel_magnitudes: Array of shape (num_envs,) with angular velocity magnitudes
        vel_commands: Array of shape (num_envs, 3) with [vel_x, vel_y, vel_yaw]
        avg_errors: Array of shape (num_envs,) with average tracking errors
        output_dir: Directory to save data
    """
    import csv

    # Save detailed data
    csv_path = output_dir / 'angular_velocity_error_data.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['robot_id', 'vel_x', 'vel_y', 'vel_yaw', 'abs_vel_yaw', 'avg_tracking_error'])

        for robot_id in range(len(angular_vel_magnitudes)):
            writer.writerow([
                robot_id,
                vel_commands[robot_id, 0],
                vel_commands[robot_id, 1],
                vel_commands[robot_id, 2],
                np.abs(angular_vel_magnitudes[robot_id]),
                avg_errors[robot_id],
            ])

    print(f"  Saved detailed data to: {csv_path}")

    # Calculate correlation (using absolute angular velocity)
    abs_angular_vel = np.abs(angular_vel_magnitudes)
    correlation = np.corrcoef(abs_angular_vel, avg_errors)[0, 1]

    # Calculate binned statistics
    num_bins = args_cli.num_bins
    bin_edges = np.linspace(0, args_cli.angular_vel_range, num_bins + 1)
    bin_stats = []

    for i in range(num_bins):
        mask = (abs_angular_vel >= bin_edges[i]) & (abs_angular_vel < bin_edges[i + 1])
        if i == num_bins - 1:  # Include right edge for last bin
            mask = (abs_angular_vel >= bin_edges[i]) & (abs_angular_vel <= bin_edges[i + 1])

        if mask.sum() > 0:
            bin_stats.append({
                'range': f'[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}]',
                'count': mask.sum(),
                'mean': avg_errors[mask].mean(),
                'std': avg_errors[mask].std(),
                'min': avg_errors[mask].min(),
                'max': avg_errors[mask].max(),
            })

    # Save summary statistics
    summary_path = output_dir / 'summary_statistics.txt'

    with open(summary_path, 'w') as f:
        f.write("ANGULAR VELOCITY ERROR VS ANGULAR VELOCITY MAGNITUDE EXPERIMENT SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Number of robots:           {len(angular_vel_magnitudes)}\n")
        f.write(f"Warmup steps:               {args_cli.warmup_steps}\n")
        f.write(f"Collection steps:           {args_cli.collection_steps}\n")
        f.write(f"Timestep:                   {args_cli.dt} s\n")
        f.write(f"Linear velocity (fixed):    {args_cli.linear_vel} m/s\n\n")

        f.write("ANGULAR VELOCITY COMMAND STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Angular velocity range:     [{-args_cli.angular_vel_range}, {args_cli.angular_vel_range}] rad/s\n")
        f.write(f"Mean angular velocity:      {angular_vel_magnitudes.mean():.4f} rad/s\n")
        f.write(f"Std angular velocity:       {angular_vel_magnitudes.std():.4f} rad/s\n")
        f.write(f"Mean |angular velocity|:    {abs_angular_vel.mean():.4f} rad/s\n\n")

        f.write("ANGULAR VELOCITY TRACKING ERROR STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mean tracking error:        {avg_errors.mean():.6f} rad/s\n")
        f.write(f"Std tracking error:         {avg_errors.std():.6f} rad/s\n")
        f.write(f"Min tracking error:         {avg_errors.min():.6f} rad/s\n")
        f.write(f"Max tracking error:         {avg_errors.max():.6f} rad/s\n\n")

        f.write("CORRELATION ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Pearson correlation coef:   {correlation:.4f}\n")
        f.write(f"  (correlation between |angular velocity| and tracking error)\n\n")

        f.write("BINNED STATISTICS (by |angular velocity|)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Number of bins:             {num_bins}\n\n")

        for i, stats in enumerate(bin_stats):
            f.write(f"Bin {i+1}: |Angular Velocity| {stats['range']} rad/s\n")
            f.write(f"  Count:      {stats['count']}\n")
            f.write(f"  Mean error: {stats['mean']:.6f} rad/s\n")
            f.write(f"  Std error:  {stats['std']:.6f} rad/s\n")
            f.write(f"  Min error:  {stats['min']:.6f} rad/s\n")
            f.write(f"  Max error:  {stats['max']:.6f} rad/s\n\n")

    print(f"  Saved summary statistics to: {summary_path}")


if __name__ == "__main__":
    # Run the experiment
    run_experiment()
    # Close sim app
    simulation_app.close()
