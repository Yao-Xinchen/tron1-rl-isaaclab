"""Script to evaluate wheel contact patterns and pose tracking during pose control."""

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
parser = argparse.ArgumentParser(description="Evaluate wheel contact patterns during pose tracking.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")

# experiment-specific arguments
parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps before data collection.")
parser.add_argument("--duration", type=float, default=3.0, help="Duration of data collection in seconds.")
parser.add_argument("--dt", type=float, default=0.02, help="Simulation timestep in seconds.")
parser.add_argument("--output_dir", type=str, default="experiments/contact_pattern", help="Directory to save results.")
parser.add_argument("--num_sampled_robots", type=int, default=3, help="Number of robots to visualize in detail.")
parser.add_argument("--pos_range", type=float, default=1.0, help="Position command range in meters (symmetric).")
parser.add_argument("--contact_threshold", type=float, default=1.0, help="Contact force threshold in Newtons.")

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
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Import extensions to set up environment tasks
import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg


def override_command_ranges(env):
    """Override the command ranges to focus on pose tracking with zero velocity.

    Args:
        env: The wrapped environment
    """
    command_term = env.unwrapped.command_manager._terms["base_pose"]

    # Set position ranges to [-pos_range, pos_range]
    command_term.cfg.ranges.pos_x = (-args_cli.pos_range, args_cli.pos_range)
    command_term.cfg.ranges.pos_y = (-args_cli.pos_range, args_cli.pos_range)

    # Set velocity ranges to zero (no velocity commands)
    if hasattr(command_term.cfg.ranges, 'vel_x'):
        command_term.cfg.ranges.vel_x = (0.0, 0.0)
        command_term.cfg.ranges.vel_y = (0.0, 0.0)
        command_term.cfg.ranges.vel_yaw = (0.0, 0.0)

    print(f"[INFO] Command ranges overridden:")
    print(f"  Position XY: [{-args_cli.pos_range}, {args_cli.pos_range}] m")
    print(f"  Velocities: [0.0, 0.0] m/s")


def get_wheel_contacts(env, threshold: float = 1.0):
    """Get binary contact states for left and right wheels.

    Args:
        env: The wrapped environment
        threshold: Force threshold in Newtons to consider as contact

    Returns:
        contacts: Tensor of shape (num_envs, 2) with [left_contact, right_contact]
    """
    # Access contact sensor from scene.sensors dictionary
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]

    # Get force magnitudes for each wheel
    # contact_sensor.data.net_forces_w has shape (num_envs, num_bodies, 3)
    # We need to find the indices for wheel_L and wheel_R

    # Find wheel body indices
    body_names = contact_sensor.body_names

    # Find indices for left and right wheels
    wheel_l_idx = None
    wheel_r_idx = None

    for idx, name in enumerate(body_names):
        if 'wheel_L' in name or 'wheel_l' in name:
            wheel_l_idx = idx
        elif 'wheel_R' in name or 'wheel_r' in name:
            wheel_r_idx = idx

    if wheel_l_idx is None or wheel_r_idx is None:
        raise ValueError(f"Could not find wheel bodies in contact sensor. Available bodies: {body_names}")

    # Get contact forces for wheels
    wheel_l_force = torch.norm(contact_sensor.data.net_forces_w[:, wheel_l_idx, :], dim=-1)
    wheel_r_force = torch.norm(contact_sensor.data.net_forces_w[:, wheel_r_idx, :], dim=-1)

    # Binary contact detection
    wheel_l_contact = (wheel_l_force > threshold).float()
    wheel_r_contact = (wheel_r_force > threshold).float()

    contacts = torch.stack([wheel_l_contact, wheel_r_contact], dim=1)

    return contacts


def get_pose_error(env):
    """Get position error from pose target.

    Args:
        env: The wrapped environment

    Returns:
        position_error: Tensor of shape (num_envs,) with L2 position error
    """
    command_term = env.unwrapped.command_manager.get_term("base_pose")
    position_error = command_term.metrics["position_error"]
    return position_error


def run_experiment():
    """Run the contact pattern experiment."""

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

    # Override command ranges
    override_command_ranges(env)

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

    # Calculate number of collection steps from duration
    collection_steps = int(args_cli.duration / args_cli.dt)
    total_steps = args_cli.warmup_steps + collection_steps

    print(f"\n[INFO] Starting experiment...")
    print(f"  Warmup steps: {args_cli.warmup_steps}")
    print(f"  Collection duration: {args_cli.duration}s ({collection_steps} steps)")
    print(f"  Total steps: {total_steps}")

    # Initialize data collection
    data_collection = []

    # Run simulation
    for step in range(total_steps):
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

        # Start recording after warmup
        if step >= args_cli.warmup_steps:
            # Get contact states and pose errors
            contacts = get_wheel_contacts(env, threshold=args_cli.contact_threshold)
            pose_errors = get_pose_error(env)

            # Store data for all robots
            data_collection.append({
                'step': step - args_cli.warmup_steps,
                'contacts': contacts.cpu().numpy(),  # shape: (num_envs, 2)
                'pose_errors': pose_errors.cpu().numpy(),  # shape: (num_envs,)
            })

            # Print progress
            if (step - args_cli.warmup_steps) % 50 == 0:
                current_time = (step - args_cli.warmup_steps) * args_cli.dt
                print(f"  Recording step {step - args_cli.warmup_steps}/{collection_steps}, "
                      f"t={current_time:.2f}s, mean pose error: {pose_errors.mean().item():.4f} m")

    if len(data_collection) == 0:
        print(f"\n[ERROR] No data collected!")
        env.close()
        return

    print(f"\n[INFO] Data collection complete!")
    print(f"  Total frames recorded: {len(data_collection)}")

    # Process and visualize data
    print(f"\n[INFO] Processing data...")
    processed_data = process_data(data_collection, args_cli.num_envs)

    # Sample robots for visualization
    sampled_robot_ids = sample_robots(args_cli.num_envs, args_cli.num_sampled_robots)

    print(f"\n[INFO] Generating visualizations for {len(sampled_robot_ids)} robots...")
    visualize_contact_patterns(processed_data, sampled_robot_ids, output_dir, env.unwrapped.step_dt)

    # Save data
    save_data(processed_data, output_dir)

    print(f"\n[INFO] Experiment complete! Results saved to {output_dir}")

    # Close the environment
    env.close()


def process_data(data_collection, num_envs):
    """Process collected data into per-robot time series.

    Args:
        data_collection: List of data frames
        num_envs: Number of environments

    Returns:
        processed_data: Dict with time series for each robot
    """
    num_frames = len(data_collection)

    # Initialize arrays
    steps = np.array([frame['step'] for frame in data_collection])
    contacts = np.stack([frame['contacts'] for frame in data_collection], axis=0)  # (frames, envs, 2)
    pose_errors = np.stack([frame['pose_errors'] for frame in data_collection], axis=0)  # (frames, envs)

    # Organize by robot
    processed_data = {
        'steps': steps,
        'contacts': contacts,  # (frames, envs, 2)
        'pose_errors': pose_errors,  # (frames, envs)
        'num_envs': num_envs,
        'num_frames': num_frames,
    }

    return processed_data


def sample_robots(num_envs, num_samples):
    """Sample robot IDs for visualization.

    Args:
        num_envs: Total number of robots
        num_samples: Number of robots to sample

    Returns:
        sampled_ids: List of sampled robot IDs
    """
    if num_samples >= num_envs:
        return list(range(num_envs))

    # Sample evenly distributed robots
    step = num_envs // num_samples
    sampled_ids = list(range(0, num_envs, step))[:num_samples]

    return sampled_ids


def visualize_contact_patterns(data, sampled_robot_ids, output_dir, dt):
    """Create visualization similar to the reference image.

    Args:
        data: Processed data dict
        sampled_robot_ids: List of robot IDs to visualize
        output_dir: Directory to save plots
        dt: Simulation timestep
    """
    steps = data['steps']
    times = steps * dt
    contacts = data['contacts']  # (frames, envs, 2)
    pose_errors = data['pose_errors']  # (frames, envs)

    num_sampled = len(sampled_robot_ids)

    # Create figure with subplots for each sampled robot
    fig = plt.figure(figsize=(16, 4 * num_sampled))

    for idx, robot_id in enumerate(sampled_robot_ids):
        # Get data for this robot
        robot_pose_error = pose_errors[:, robot_id]
        robot_contacts = contacts[:, robot_id, :]  # (frames, 2)

        # Create subplots: position error (top) and contact pattern (bottom)
        ax_pos = fig.add_subplot(num_sampled, 1, idx + 1)

        # Plot position error
        ax_pos.plot(times, robot_pose_error, linewidth=2, color='darkorange', label='Position Error')
        ax_pos.set_ylabel('Position Error (m)', fontsize=10)
        ax_pos.set_title(f'Robot {robot_id} - Pose Tracking and Wheel Contact Pattern', fontsize=11)
        ax_pos.grid(True, alpha=0.3)
        ax_pos.legend(loc='upper right')

        # Create second y-axis for contact visualization
        ax_contact = ax_pos.twinx()

        # Plot contact patterns as filled regions (thin bars)
        # Bars centered at 2/3 (top) and 1/3 (bottom) height
        bar_height = 0.1  # Half-height of each bar
        left_center = 2/3  # Left wheel at 2/3 height
        right_center = 1/3  # Right wheel at 1/3 height

        # Left wheel (centered at 2/3)
        left_contacts = robot_contacts[:, 0]
        ax_contact.fill_between(times,
                               left_center - bar_height,
                               left_center + left_contacts * bar_height,
                               where=left_contacts > 0.5,
                               color='steelblue', alpha=0.3,
                               label='Left Wheel Contact', step='mid')

        # Right wheel (centered at 1/3)
        right_contacts = robot_contacts[:, 1]
        ax_contact.fill_between(times,
                               right_center - bar_height,
                               right_center + right_contacts * bar_height,
                               where=right_contacts > 0.5,
                               color='orange', alpha=0.3,
                               label='Right Wheel Contact', step='mid')

        ax_contact.set_ylim(0, 1)
        ax_contact.set_yticks([right_center, left_center])
        ax_contact.set_yticklabels(['Right', 'Left'])
        ax_contact.set_ylabel('Wheel Contact', fontsize=10)

        if idx == num_sampled - 1:
            ax_pos.set_xlabel('Time (s)', fontsize=10)

        # Combine legends
        lines1, labels1 = ax_pos.get_legend_handles_labels()
        lines2, labels2 = ax_contact.get_legend_handles_labels()
        ax_pos.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'contact_patterns.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved contact pattern visualization to: {output_path}")

    # Create summary statistics plot
    create_summary_plot(data, output_dir, dt)

    plt.close('all')


def create_summary_plot(data, output_dir, dt):
    """Create summary statistics plot across all robots.

    Args:
        data: Processed data dict
        output_dir: Directory to save plots
        dt: Simulation timestep
    """
    steps = data['steps']
    times = steps * dt
    contacts = data['contacts']  # (frames, envs, 2)
    pose_errors = data['pose_errors']  # (frames, envs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Mean pose error over time
    mean_pose_error = pose_errors.mean(axis=1)
    std_pose_error = pose_errors.std(axis=1)

    ax1.plot(times, mean_pose_error, linewidth=2, color='darkorange', label='Mean Position Error')
    ax1.fill_between(times,
                     mean_pose_error - std_pose_error,
                     mean_pose_error + std_pose_error,
                     alpha=0.3, color='orange', label='+/- 1 std')
    ax1.set_ylabel('Position Error (m)', fontsize=11)
    ax1.set_title('Position Tracking Performance Across All Robots', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Plot 2: Contact statistics
    left_contact_ratio = contacts[:, :, 0].mean(axis=1) * 100  # percentage
    right_contact_ratio = contacts[:, :, 1].mean(axis=1) * 100

    ax2.plot(times, left_contact_ratio, linewidth=2, color='steelblue', label='Left Wheel Contact %')
    ax2.plot(times, right_contact_ratio, linewidth=2, color='orange', label='Right Wheel Contact %')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Contact Percentage (%)', fontsize=11)
    ax2.set_title('Wheel Contact Statistics Across All Robots', fontsize=12)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()

    output_path = output_dir / 'summary_statistics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved summary statistics to: {output_path}")


def save_data(data, output_dir):
    """Save data to CSV files.

    Args:
        data: Processed data dict
        output_dir: Directory to save CSV
    """
    import csv

    steps = data['steps']
    contacts = data['contacts']
    pose_errors = data['pose_errors']
    num_envs = data['num_envs']

    # Save detailed data
    output_path = output_dir / 'contact_data.csv'

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'robot_id', 'pose_error', 'left_contact', 'right_contact'])

        for frame_idx, step in enumerate(steps):
            for robot_id in range(num_envs):
                writer.writerow([
                    step,
                    robot_id,
                    pose_errors[frame_idx, robot_id],
                    contacts[frame_idx, robot_id, 0],
                    contacts[frame_idx, robot_id, 1],
                ])

    print(f"  Saved detailed contact data to: {output_path}")

    # Save summary statistics
    summary_path = output_dir / 'summary_statistics.txt'

    with open(summary_path, 'w') as f:
        f.write("CONTACT PATTERN EXPERIMENT SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Number of robots:          {num_envs}\n")
        f.write(f"Recording frames:          {len(steps)}\n")
        f.write(f"Position range:            [{-args_cli.pos_range}, {args_cli.pos_range}] m\n")
        f.write(f"Contact threshold:         {args_cli.contact_threshold} N\n\n")

        f.write("POSE TRACKING STATISTICS\n")
        f.write("-" * 60 + "\n")
        mean_error = pose_errors.mean()
        f.write(f"Mean position error:       {mean_error:.6f} m\n")
        f.write(f"Std position error:        {pose_errors.std():.6f} m\n")
        f.write(f"Min position error:        {pose_errors.min():.6f} m\n")
        f.write(f"Max position error:        {pose_errors.max():.6f} m\n\n")

        f.write("CONTACT STATISTICS\n")
        f.write("-" * 60 + "\n")
        left_contact_pct = contacts[:, :, 0].mean() * 100
        right_contact_pct = contacts[:, :, 1].mean() * 100
        f.write(f"Left wheel contact time:   {left_contact_pct:.2f}%\n")
        f.write(f"Right wheel contact time:  {right_contact_pct:.2f}%\n")

        # Calculate periods where both wheels in contact
        both_contact = (contacts[:, :, 0] * contacts[:, :, 1]).mean() * 100
        f.write(f"Both wheels contact time:  {both_contact:.2f}%\n")

        # Calculate periods where no contact
        no_contact = ((1 - contacts[:, :, 0]) * (1 - contacts[:, :, 1])).mean() * 100
        f.write(f"No contact time:           {no_contact:.2f}%\n")

    print(f"  Saved summary statistics to: {summary_path}")


if __name__ == "__main__":
    # Run the experiment
    run_experiment()
    # Close sim app
    simulation_app.close()
