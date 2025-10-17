"""Script to analyze velocity tracking error vs command velocity magnitude."""

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
parser = argparse.ArgumentParser(description="Analyze velocity tracking error vs velocity magnitude.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")

# experiment-specific arguments
parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps before data collection.")
parser.add_argument("--collection_steps", type=int, default=500, help="Number of steps for data collection.")
parser.add_argument("--dt", type=float, default=0.02, help="Simulation timestep in seconds.")
parser.add_argument("--output_dir", type=str, default="experiments/velocity_error", help="Directory to save results.")
parser.add_argument("--vel_range", type=float, default=1.0, help="Velocity command range in m/s (symmetric).")
parser.add_argument("--num_bins", type=int, default=10, help="Number of bins for velocity binning.")

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


def sample_velocity_commands(num_envs, vel_range, device):
    """Sample velocity commands with magnitude-based sampling.

    Args:
        num_envs: Number of environments
        vel_range: Maximum velocity magnitude in m/s
        device: Torch device

    Returns:
        vel_commands: Tensor of shape (num_envs, 3) with [vel_x, vel_y, vel_yaw]
        vel_magnitudes: Tensor of shape (num_envs,) with 2D velocity magnitudes
    """
    # Sample velocity magnitude uniformly from [0, vel_range] m/s
    vel_magnitudes = torch.rand(num_envs, device=device) * vel_range

    # Sample random direction uniformly from [0, 2pi]
    angles = torch.rand(num_envs, device=device) * 2 * np.pi

    # Convert to vel_x, vel_y
    vel_x = vel_magnitudes * torch.cos(angles)
    vel_y = vel_magnitudes * torch.sin(angles)

    # Sample vel_yaw independently from [-1, 1] rad/s
    vel_yaw = torch.rand(num_envs, device=device) * 2.0 - 1.0

    vel_commands = torch.stack([vel_x, vel_y, vel_yaw], dim=1)

    return vel_commands, vel_magnitudes


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


def calculate_velocity_error(env, target_vel_commands):
    """Calculate velocity tracking error in body frame for all environments.

    Args:
        env: The wrapped environment
        target_vel_commands: Tensor of shape (num_envs, 3) with [vel_x, vel_y, vel_yaw]

    Returns:
        error: Tensor of shape (num_envs,) with L2 velocity error magnitude
    """
    robot = env.unwrapped.scene["robot"]

    # Get actual velocity in world frame and robot orientation
    actual_vel_w = robot.data.root_vel_w  # Shape: (num_envs, 6) - [vx, vy, vz, wx, wy, wz]
    root_quat_w = robot.data.root_quat_w  # Shape: (num_envs, 4) - [w, x, y, z]

    # Transform linear velocity from world to body frame
    actual_lin_vel_b = quat_apply_inverse(root_quat_w, actual_vel_w[:, :3])

    # Extract linear velocities (x, y) in body frame
    actual_linear = actual_lin_vel_b[:, :2]
    target_linear = target_vel_commands[:, :2]

    # Calculate L2 error
    error = torch.norm(actual_linear - target_linear, dim=1)

    return error


def run_experiment():
    """Run the velocity error vs velocity magnitude experiment."""

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

    # Sample velocity commands for each robot
    print(f"\n[INFO] Sampling velocity commands...")
    vel_commands, vel_magnitudes = sample_velocity_commands(
        args_cli.num_envs, args_cli.vel_range, env.unwrapped.device
    )

    print(f"[INFO] Assigned velocity commands to {len(vel_commands)} robots")
    print(f"  Velocity magnitude range: [{vel_magnitudes.min():.3f}, {vel_magnitudes.max():.3f}] m/s")
    print(f"  Velocity X range: [{vel_commands[:, 0].min():.3f}, {vel_commands[:, 0].max():.3f}] m/s")
    print(f"  Velocity Y range: [{vel_commands[:, 1].min():.3f}, {vel_commands[:, 1].max():.3f}] m/s")
    print(f"  Velocity yaw range: [{vel_commands[:, 2].min():.3f}, {vel_commands[:, 2].max():.3f}] rad/s")

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
            # Get velocity error using real robot velocity
            velocity_errors = calculate_velocity_error(env, vel_commands)

            # Accumulate errors
            error_sum += velocity_errors
            error_count += 1

            # Print progress
            if (step - args_cli.warmup_steps) % 50 == 0:
                print(f"  Collection step {step - args_cli.warmup_steps}/{args_cli.collection_steps}, "
                      f"mean velocity error: {velocity_errors.mean().item():.4f} m/s")

    # Calculate average tracking error per robot
    avg_tracking_errors = error_sum / error_count

    print(f"\n[INFO] Data collection complete!")
    print(f"  Collection steps: {error_count}")
    print(f"  Overall mean tracking error: {avg_tracking_errors.mean().item():.4f} m/s")
    print(f"  Overall std tracking error: {avg_tracking_errors.std().item():.4f} m/s")

    # Move data to CPU for processing
    vel_magnitudes_cpu = vel_magnitudes.cpu().numpy()
    vel_commands_cpu = vel_commands.cpu().numpy()
    avg_tracking_errors_cpu = avg_tracking_errors.cpu().numpy()

    # Process and visualize data
    print(f"\n[INFO] Generating visualizations...")
    visualize_results(vel_magnitudes_cpu, vel_commands_cpu, avg_tracking_errors_cpu, output_dir)

    # Save data
    save_data(vel_magnitudes_cpu, vel_commands_cpu, avg_tracking_errors_cpu, output_dir)

    print(f"\n[INFO] Experiment complete! Results saved to {output_dir}")

    # Close the environment
    env.close()


def visualize_results(vel_magnitudes, vel_commands, avg_errors, output_dir):
    """Create visualizations for velocity magnitude vs tracking error.

    Args:
        vel_magnitudes: Array of shape (num_envs,) with velocity magnitudes
        vel_commands: Array of shape (num_envs, 3) with [vel_x, vel_y, vel_yaw]
        avg_errors: Array of shape (num_envs,) with average tracking errors
        output_dir: Directory to save plots
    """

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Scatter plot
    scatter = ax1.scatter(vel_magnitudes, avg_errors, alpha=0.3, s=10, c=vel_magnitudes, cmap='viridis')
    ax1.set_xlabel('Command Velocity Magnitude (m/s)', fontsize=12)
    ax1.set_ylabel('Average Velocity Tracking Error (m/s)', fontsize=12)
    ax1.set_title('Velocity Tracking Error vs Command Velocity Magnitude', fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Velocity Magnitude (m/s)', fontsize=10)

    # Add trendline (polynomial fit)
    if len(vel_magnitudes) > 10:
        z = np.polyfit(vel_magnitudes, avg_errors, 2)
        p = np.poly1d(z)
        vel_sorted = np.linspace(vel_magnitudes.min(), vel_magnitudes.max(), 100)
        ax1.plot(vel_sorted, p(vel_sorted), "r--", linewidth=2, alpha=0.8, label='Quadratic Fit')
        ax1.legend(fontsize=10)

    # Plot 2: Binned violin plot
    num_bins = args_cli.num_bins
    bin_edges = np.linspace(0, args_cli.vel_range, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Collect data for each bin
    bin_data = []
    bin_counts = []
    valid_positions = []

    for i in range(num_bins):
        mask = (vel_magnitudes >= bin_edges[i]) & (vel_magnitudes < bin_edges[i + 1])
        if i == num_bins - 1:  # Include right edge for last bin
            mask = (vel_magnitudes >= bin_edges[i]) & (vel_magnitudes <= bin_edges[i + 1])

        if mask.sum() > 0:
            bin_data.append(avg_errors[mask])
            bin_counts.append(mask.sum())
            valid_positions.append(bin_centers[i])

    # Create violin plot
    if len(bin_data) > 0:
        parts = ax2.violinplot(bin_data, positions=valid_positions,
                              widths=args_cli.vel_range / num_bins * 0.6,
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

    ax2.set_xlabel('Command Velocity Magnitude (m/s)', fontsize=12)
    ax2.set_ylabel('Average Velocity Tracking Error (m/s)', fontsize=12)
    ax2.set_title(f'Distribution of Tracking Error (n={num_bins} bins)', fontsize=13)
    ax2.set_xlim(-0.05 * args_cli.vel_range, args_cli.vel_range * 1.05)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'position_error.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved visualization to: {output_path}")

    plt.close('all')


def save_data(vel_magnitudes, vel_commands, avg_errors, output_dir):
    """Save experiment data to CSV and summary file.

    Args:
        vel_magnitudes: Array of shape (num_envs,) with velocity magnitudes
        vel_commands: Array of shape (num_envs, 3) with [vel_x, vel_y, vel_yaw]
        avg_errors: Array of shape (num_envs,) with average tracking errors
        output_dir: Directory to save data
    """
    import csv

    # Save detailed data
    csv_path = output_dir / 'velocity_error_data.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['robot_id', 'vel_x', 'vel_y', 'vel_yaw', 'vel_magnitude', 'avg_tracking_error'])

        for robot_id in range(len(vel_magnitudes)):
            writer.writerow([
                robot_id,
                vel_commands[robot_id, 0],
                vel_commands[robot_id, 1],
                vel_commands[robot_id, 2],
                vel_magnitudes[robot_id],
                avg_errors[robot_id],
            ])

    print(f"  Saved detailed data to: {csv_path}")

    # Calculate correlation
    correlation = np.corrcoef(vel_magnitudes, avg_errors)[0, 1]

    # Calculate binned statistics
    num_bins = args_cli.num_bins
    bin_edges = np.linspace(0, args_cli.vel_range, num_bins + 1)
    bin_stats = []

    for i in range(num_bins):
        mask = (vel_magnitudes >= bin_edges[i]) & (vel_magnitudes < bin_edges[i + 1])
        if i == num_bins - 1:  # Include right edge for last bin
            mask = (vel_magnitudes >= bin_edges[i]) & (vel_magnitudes <= bin_edges[i + 1])

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
        f.write("VELOCITY ERROR VS VELOCITY MAGNITUDE EXPERIMENT SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Number of robots:           {len(vel_magnitudes)}\n")
        f.write(f"Warmup steps:               {args_cli.warmup_steps}\n")
        f.write(f"Collection steps:           {args_cli.collection_steps}\n")
        f.write(f"Timestep:                   {args_cli.dt} s\n\n")

        f.write("VELOCITY COMMAND STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Velocity magnitude range:   [0.0, {args_cli.vel_range}] m/s\n")
        f.write(f"Mean velocity magnitude:    {vel_magnitudes.mean():.4f} m/s\n")
        f.write(f"Std velocity magnitude:     {vel_magnitudes.std():.4f} m/s\n\n")

        f.write("VELOCITY TRACKING ERROR STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mean tracking error:        {avg_errors.mean():.6f} m/s\n")
        f.write(f"Std tracking error:         {avg_errors.std():.6f} m/s\n")
        f.write(f"Min tracking error:         {avg_errors.min():.6f} m/s\n")
        f.write(f"Max tracking error:         {avg_errors.max():.6f} m/s\n\n")

        f.write("CORRELATION ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Pearson correlation coef:   {correlation:.4f}\n\n")

        f.write("BINNED STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Number of bins:             {num_bins}\n\n")

        for i, stats in enumerate(bin_stats):
            f.write(f"Bin {i+1}: Velocity {stats['range']} m/s\n")
            f.write(f"  Count:      {stats['count']}\n")
            f.write(f"  Mean error: {stats['mean']:.6f} m/s\n")
            f.write(f"  Std error:  {stats['std']:.6f} m/s\n")
            f.write(f"  Min error:  {stats['min']:.6f} m/s\n")
            f.write(f"  Max error:  {stats['max']:.6f} m/s\n\n")

    print(f"  Saved summary statistics to: {summary_path}")


if __name__ == "__main__":
    # Run the experiment
    run_experiment()
    # Close sim app
    simulation_app.close()