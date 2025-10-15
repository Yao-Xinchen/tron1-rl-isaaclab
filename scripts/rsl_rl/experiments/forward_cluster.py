"""Script to evaluate how well the policy resists terrain disturbances during forward motion."""

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
parser = argparse.ArgumentParser(description="Evaluate trajectory clustering during forward motion.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")

# experiment-specific arguments
parser.add_argument("--warmup_steps", type=int, default=200, help="Number of warmup steps before data collection.")
parser.add_argument("--duration", type=float, default=5.0, help="Duration of trajectory recording in seconds.")
parser.add_argument("--dt", type=float, default=0.02, help="Simulation timestep in seconds.")
parser.add_argument("--output_dir", type=str, default="experiments/forward_cluster/pose", help="Directory to save results.")
parser.add_argument("--forward_velocity", type=float, default=0.5, help="Forward velocity command (m/s).")

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


def quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw angle from quaternion.

    Args:
        quat: Quaternion tensor of shape (N, 4) in [w, x, y, z] format

    Returns:
        yaw: Yaw angles in radians, shape (N,)
    """
    # Extract components (assuming [w, x, y, z] format)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Calculate yaw using atan2
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    return yaw


def rotate_trajectory(x: float, y: float, yaw: float) -> tuple:
    """Rotate trajectory point by -yaw to align with global X-axis.

    Args:
        x: X position
        y: Y position
        yaw: Initial heading angle in radians

    Returns:
        x_rot, y_rot: Rotated coordinates
    """
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)

    x_rot = x * cos_yaw - y * sin_yaw
    y_rot = x * sin_yaw + y * cos_yaw

    return x_rot, y_rot


def create_forward_command(num_envs: int, velocity: float, device: str = "cuda"):
    """Create forward velocity commands for all robots.

    Args:
        num_envs: Number of environments (robots)
        velocity: Forward velocity (m/s)
        device: Device to create tensors on

    Returns:
        commands: Tensor of shape (num_envs, 3) with [vx, vy, angular_z]
    """
    vx = torch.full((num_envs,), velocity, device=device)
    vy = torch.zeros(num_envs, device=device)
    ang_vel_z = torch.zeros(num_envs, device=device)

    # Stack into commands tensor
    commands = torch.stack([vx, vy, ang_vel_z], dim=1)

    return commands


def override_pose_commands(env, commands: torch.Tensor):
    """Override the pose commands to achieve velocity control.

    Sets the target pose to the robot's current pose (making relative pose zero)
    and sets the velocity in the command frame to the desired velocity.

    Args:
        env: The wrapped environment
        commands: Tensor of shape (num_envs, 3) with [vx, vy, angular_z] velocities
    """
    # Access the base_pose command term
    command_term = env.unwrapped.command_manager._terms["base_pose"]
    robot = env.unwrapped.scene["robot"]

    # Set target pose to robot's current pose (zero relative distance)
    # This makes the robot "believe" it has reached the target
    command_term.pose_command_w[:, :3] = robot.data.root_link_pos_w.clone()
    command_term.pose_command_w[:, 3:] = robot.data.root_link_quat_w.clone()

    # Set velocity commands in command frame (which equals body frame when pose is at robot)
    # pose_command_vel_c is [vel_x, vel_y, vel_yaw] in target frame
    command_term.pose_command_vel_c[:] = commands


def run_experiment():
    """Run the forward clustering experiment."""

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

    # Create forward velocity commands
    print(f"[INFO] Setting forward velocity command: {args_cli.forward_velocity} m/s")
    commands = create_forward_command(
        args_cli.num_envs,
        velocity=args_cli.forward_velocity,
        device=env.unwrapped.device
    )

    # Override the commands
    override_pose_commands(env, commands)

    # Reset environment
    obs, obs_dict = env.get_observations()
    obs_history = obs_dict["observations"].get("obsHistory")
    obs_history = obs_history.flatten(start_dim=1)
    critic_obs = obs_dict["observations"].get("critic")
    commands_obs = obs_dict["observations"].get("commands")

    # Calculate number of steps
    collection_steps = int(args_cli.duration / args_cli.dt)
    total_steps = args_cli.warmup_steps + collection_steps

    print(f"\n[INFO] Starting experiment...")
    print(f"  Warmup steps: {args_cli.warmup_steps}")
    print(f"  Collection duration: {args_cli.duration}s ({collection_steps} steps)")
    print(f"  Simulation dt: {args_cli.dt}s")

    # Initialize trajectory storage
    trajectory_data = []

    # Record initial positions and headings after warmup
    robot = env.unwrapped.scene["robot"]
    initial_positions = None
    initial_yaws = None

    # Run simulation
    for step in range(total_steps):
        # Override commands at each step to prevent resampling
        override_pose_commands(env, commands)

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
        override_pose_commands(env, commands)

        # Start collecting after warmup
        if step == args_cli.warmup_steps:
            # Record initial positions and headings
            initial_positions = robot.data.root_link_pos_w.clone()
            initial_quats = robot.data.root_link_quat_w.clone()
            initial_yaws = quat_to_yaw(initial_quats).cpu().numpy()
            print(f"\n[INFO] Starting trajectory recording from step {step}")
            print(f"[INFO] Initial heading range: [{np.min(np.degrees(initial_yaws)):.1f}, {np.max(np.degrees(initial_yaws)):.1f}] degrees")

        if step >= args_cli.warmup_steps:
            # Record current positions
            current_positions = robot.data.root_link_pos_w.clone()

            # Calculate relative positions from start
            relative_positions = current_positions - initial_positions

            # Store trajectory data with heading alignment
            current_time = (step - args_cli.warmup_steps) * args_cli.dt
            for robot_id in range(args_cli.num_envs):
                # Rotate trajectory to align with global X-axis
                x_aligned, y_aligned = rotate_trajectory(
                    relative_positions[robot_id, 0].item(),
                    relative_positions[robot_id, 1].item(),
                    initial_yaws[robot_id]
                )

                trajectory_data.append({
                    'time': current_time,
                    'robot_id': robot_id,
                    'x': x_aligned,
                    'y': y_aligned,
                    'z': relative_positions[robot_id, 2].item(),
                })

            # Print progress
            if (step - args_cli.warmup_steps + 1) % 100 == 0:
                print(f"  Collection step {step - args_cli.warmup_steps + 1}/{collection_steps} "
                      f"(t={current_time:.2f}s)")

    print(f"\n[INFO] Data collection complete!")
    print(f"  Total trajectory points recorded: {len(trajectory_data)}")

    # Convert to structured format for analysis
    print(f"\n[INFO] Processing trajectory data...")
    trajectories = process_trajectories(trajectory_data, args_cli.num_envs)

    # Analyze and visualize
    print(f"\n[INFO] Generating visualizations...")
    analyze_and_visualize(trajectories, output_dir, args_cli.forward_velocity, args_cli.duration)

    # Save raw data
    save_trajectory_data(trajectory_data, trajectories, output_dir)

    print(f"\n[INFO] Experiment complete! Results saved to {output_dir}")

    # Close the environment
    env.close()


def process_trajectories(trajectory_data, num_envs):
    """Process raw trajectory data into per-robot trajectories.

    Args:
        trajectory_data: List of trajectory points
        num_envs: Number of robots

    Returns:
        trajectories: Dict with robot_id as key, each containing arrays of times and positions
    """
    trajectories = {i: {'time': [], 'x': [], 'y': [], 'z': []} for i in range(num_envs)}

    for point in trajectory_data:
        robot_id = point['robot_id']
        trajectories[robot_id]['time'].append(point['time'])
        trajectories[robot_id]['x'].append(point['x'])
        trajectories[robot_id]['y'].append(point['y'])
        trajectories[robot_id]['z'].append(point['z'])

    # Convert to numpy arrays
    for robot_id in range(num_envs):
        trajectories[robot_id]['time'] = np.array(trajectories[robot_id]['time'])
        trajectories[robot_id]['x'] = np.array(trajectories[robot_id]['x'])
        trajectories[robot_id]['y'] = np.array(trajectories[robot_id]['y'])
        trajectories[robot_id]['z'] = np.array(trajectories[robot_id]['z'])

    return trajectories


def analyze_and_visualize(trajectories, output_dir, forward_velocity, duration):
    """Analyze trajectories and create visualizations.

    Args:
        trajectories: Dict of per-robot trajectory data
        output_dir: Path to save figures
        forward_velocity: Commanded forward velocity
        duration: Duration of experiment
    """
    num_robots = len(trajectories)

    # Calculate statistics
    final_x_positions = np.array([trajectories[i]['x'][-1] for i in range(num_robots)])
    final_y_positions = np.array([trajectories[i]['y'][-1] for i in range(num_robots)])

    lateral_deviations = np.abs(final_y_positions)
    mean_lateral_deviation = np.mean(lateral_deviations)
    std_lateral_deviation = np.std(lateral_deviations)
    max_lateral_deviation = np.max(lateral_deviations)

    expected_forward_distance = forward_velocity * duration
    forward_errors = np.abs(final_x_positions - expected_forward_distance)
    mean_forward_error = np.mean(forward_errors)

    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 12))

    # 1. Top-down view of all trajectories
    ax1 = fig.add_subplot(2, 3, 1)
    for robot_id in range(num_robots):
        ax1.plot(trajectories[robot_id]['x'], trajectories[robot_id]['y'],
                color='steelblue', alpha=0.2, linewidth=1.0)

    # Add reference line for perfect forward motion
    ax1.plot([0, expected_forward_distance], [0, 0], 'k--', linewidth=2,
            label=f'Ideal path ({forward_velocity} m/s)')

    # Set axis limits based on expected distance (not actual data)
    x_limit = expected_forward_distance * 1.2  # 20% margin
    y_limit = expected_forward_distance * 0.8  # Lateral deviation limit proportional to forward distance

    ax1.set_xlabel('Forward Distance (m)')
    ax1.set_ylabel('Lateral Deviation (m)')
    ax1.set_title(f'Trajectory Clustering - Top View\n{num_robots} robots, {duration}s duration')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Set limits AFTER axis('equal') to prevent them from being overridden
    ax1.set_xlim(-0.5, x_limit)
    ax1.set_ylim(-y_limit, y_limit)

    ax1.legend()

    # 2. Final position scatter plot
    ax2 = fig.add_subplot(2, 3, 2)
    scatter = ax2.scatter(final_x_positions, final_y_positions,
                         c=lateral_deviations, cmap='viridis', alpha=0.6, s=50)
    ax2.plot(expected_forward_distance, 0, 'ro', markersize=5,
            label='Expected final position')
    ax2.set_xlabel('Final X Position (m)')
    ax2.set_ylabel('Final Y Position (m)')
    ax2.set_title('Final Robot Positions')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    ax2.legend()
    plt.colorbar(scatter, ax=ax2, label='Lateral Deviation (m)')

    # 3. Lateral deviation over time
    ax3 = fig.add_subplot(2, 3, 3)
    for robot_id in range(num_robots):
        lateral_dev = np.abs(trajectories[robot_id]['y'])
        ax3.plot(trajectories[robot_id]['time'], lateral_dev, alpha=0.3, linewidth=0.8)

    # Calculate mean and std over time
    times = trajectories[0]['time']
    mean_lat_dev_over_time = np.mean([np.abs(trajectories[i]['y']) for i in range(num_robots)], axis=0)
    std_lat_dev_over_time = np.std([np.abs(trajectories[i]['y']) for i in range(num_robots)], axis=0)

    ax3.plot(times, mean_lat_dev_over_time, 'r-', linewidth=2, label='Mean')
    ax3.fill_between(times,
                     mean_lat_dev_over_time - std_lat_dev_over_time,
                     mean_lat_dev_over_time + std_lat_dev_over_time,
                     alpha=0.3, color='red', label='+/- 1 std')

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Lateral Deviation (m)')
    ax3.set_title('Lateral Deviation Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Histogram of lateral deviations
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist(lateral_deviations, bins=30, alpha=0.7, edgecolor='black')
    ax4.axvline(mean_lateral_deviation, color='r', linestyle='--', linewidth=2,
               label=f'Mean: {mean_lateral_deviation:.4f} m')
    ax4.set_xlabel('Lateral Deviation (m)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Lateral Deviations')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Forward distance histogram
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(final_x_positions, bins=30, alpha=0.7, edgecolor='black')
    ax5.axvline(expected_forward_distance, color='r', linestyle='--', linewidth=2,
               label=f'Expected: {expected_forward_distance:.2f} m')
    ax5.axvline(np.mean(final_x_positions), color='g', linestyle='--', linewidth=2,
               label=f'Actual mean: {np.mean(final_x_positions):.2f} m')
    ax5.set_xlabel('Final Forward Distance (m)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Distribution of Forward Distances')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Statistics table
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    stats_text = f"""
    FORWARD CLUSTERING EXPERIMENT STATISTICS

    Configuration:
    " Number of robots:          {num_robots}
    " Commanded velocity:        {forward_velocity} m/s forward
    " Duration:                  {duration} s
    " Expected distance:         {expected_forward_distance:.3f} m

    Forward Motion:
    " Mean final X position:     {np.mean(final_x_positions):.4f} m
    " Std final X position:      {np.std(final_x_positions):.4f} m
    " Mean forward error:        {mean_forward_error:.4f} m

    Lateral Deviation:
    " Mean lateral deviation:    {mean_lateral_deviation:.4f} m
    " Std lateral deviation:     {std_lateral_deviation:.4f} m
    " Max lateral deviation:     {max_lateral_deviation:.4f} m
    " Min lateral deviation:     {np.min(lateral_deviations):.4f} m

    Clustering Quality:
    " Y spread (95th percentile): {np.percentile(lateral_deviations, 95):.4f} m
    " X spread (std):            {np.std(final_x_positions):.4f} m
    """

    ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'trajectory_clustering_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved visualization to: {output_path}")

    # Create separate detailed trajectory plot
    fig2, ax = plt.subplots(figsize=(12, 8))

    # Plot all trajectories with same color and low opacity
    for robot_id in range(num_robots):
        ax.plot(trajectories[robot_id]['x'], trajectories[robot_id]['y'],
               color='steelblue', alpha=0.15, linewidth=0.8)

    # Add reference line
    ax.plot([0, expected_forward_distance], [0, 0], 'k--', linewidth=1,
           label=f'Ideal forward path ({forward_velocity} m/s)')

    # Mark start and expected end
    ax.plot(0, 0, 'go', markersize=5, label='Start position', zorder=5)
    ax.plot(expected_forward_distance, 0, 'ro', markersize=5,
           label='Expected final position', zorder=5)

    ax.set_xlabel('Forward Distance (m)', fontsize=12)
    ax.set_ylabel('Lateral Deviation (m)', fontsize=12)
    ax.set_title(f'Forward Motion Trajectory Clustering\n{num_robots} robots, {duration}s @ {forward_velocity} m/s\n'
                f'Mean lateral deviation: {mean_lateral_deviation:.4f} +/- {std_lateral_deviation:.4f} m',
                fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Set axis limits based on expected distance (same as comprehensive plot)
    # Must be set AFTER axis('equal') to prevent them from being overridden
    ax.set_xlim(-0.5, x_limit)
    ax.set_ylim(-y_limit, y_limit)

    ax.legend(fontsize=11)

    output_path2 = output_dir / 'trajectory_clustering_detailed.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"  Saved detailed plot to: {output_path2}")

    plt.close('all')


def save_trajectory_data(trajectory_data, trajectories, output_dir):
    """Save trajectory data to CSV files.

    Args:
        trajectory_data: Raw trajectory data list
        trajectories: Processed per-robot trajectories
        output_dir: Path to save CSV
    """
    import csv

    # Save raw trajectory data
    output_path = output_dir / 'trajectories_raw.csv'

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'robot_id', 'x', 'y', 'z'])

        for point in trajectory_data:
            writer.writerow([
                point['time'],
                point['robot_id'],
                point['x'],
                point['y'],
                point['z']
            ])

    print(f"  Saved raw trajectory data to: {output_path}")

    # Save summary statistics per robot
    summary_path = output_dir / 'robot_summary.csv'

    num_robots = len(trajectories)
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['robot_id', 'final_x', 'final_y', 'final_z',
                        'lateral_deviation', 'path_length'])

        for robot_id in range(num_robots):
            final_x = trajectories[robot_id]['x'][-1]
            final_y = trajectories[robot_id]['y'][-1]
            final_z = trajectories[robot_id]['z'][-1]
            lateral_dev = abs(final_y)

            # Calculate path length
            dx = np.diff(trajectories[robot_id]['x'])
            dy = np.diff(trajectories[robot_id]['y'])
            dz = np.diff(trajectories[robot_id]['z'])
            path_length = np.sum(np.sqrt(dx**2 + dy**2 + dz**2))

            writer.writerow([
                robot_id,
                final_x,
                final_y,
                final_z,
                lateral_dev,
                path_length
            ])

    print(f"  Saved robot summary to: {summary_path}")

    # Save overall statistics
    stats_path = output_dir / 'summary_statistics.txt'

    final_x_positions = np.array([trajectories[i]['x'][-1] for i in range(num_robots)])
    final_y_positions = np.array([trajectories[i]['y'][-1] for i in range(num_robots)])
    lateral_deviations = np.abs(final_y_positions)

    expected_distance = args_cli.forward_velocity * args_cli.duration

    with open(stats_path, 'w') as f:
        f.write("FORWARD CLUSTERING EXPERIMENT SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Number of robots:          {num_robots}\n")
        f.write(f"Forward velocity:          {args_cli.forward_velocity} m/s\n")
        f.write(f"Duration:                  {args_cli.duration} s\n")
        f.write(f"Expected distance:         {expected_distance:.4f} m\n")
        f.write(f"Warmup steps:              {args_cli.warmup_steps}\n\n")

        f.write("FORWARD MOTION STATISTICS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Mean final X:              {np.mean(final_x_positions):.6f} m\n")
        f.write(f"Std final X:               {np.std(final_x_positions):.6f} m\n")
        f.write(f"Min final X:               {np.min(final_x_positions):.6f} m\n")
        f.write(f"Max final X:               {np.max(final_x_positions):.6f} m\n")
        f.write(f"Mean forward error:        {np.mean(np.abs(final_x_positions - expected_distance)):.6f} m\n\n")

        f.write("LATERAL DEVIATION STATISTICS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Mean lateral deviation:    {np.mean(lateral_deviations):.6f} m\n")
        f.write(f"Std lateral deviation:     {np.std(lateral_deviations):.6f} m\n")
        f.write(f"Min lateral deviation:     {np.min(lateral_deviations):.6f} m\n")
        f.write(f"Max lateral deviation:     {np.max(lateral_deviations):.6f} m\n")
        f.write(f"Median lateral deviation:  {np.median(lateral_deviations):.6f} m\n")
        f.write(f"95th percentile:           {np.percentile(lateral_deviations, 95):.6f} m\n\n")

        f.write("CLUSTERING QUALITY\n")
        f.write("-" * 60 + "\n")
        f.write(f"Y spread (range):          {np.max(final_y_positions) - np.min(final_y_positions):.6f} m\n")
        f.write(f"Y spread (std):            {np.std(final_y_positions):.6f} m\n")
        f.write(f"X spread (std):            {np.std(final_x_positions):.6f} m\n")

    print(f"  Saved summary statistics to: {stats_path}")


if __name__ == "__main__":
    # Run the experiment
    run_experiment()
    # Close sim app
    simulation_app.close()
