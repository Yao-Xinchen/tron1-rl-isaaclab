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
parser = argparse.ArgumentParser(description="Analyze position-controlled motion with wheel contact and rolling patterns.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (fixed to 1 for detailed analysis).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")

# experiment-specific arguments
parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps before data collection.")
parser.add_argument("--collection_steps", type=int, default=125, help="Number of steps for data collection.")
parser.add_argument("--dt", type=float, default=0.02, help="Simulation timestep in seconds.")
parser.add_argument("--output_dir", type=str, default="experiments/position_motion", help="Directory to save results.")
parser.add_argument("--pos_range", type=float, default=1.0, help="Position command range in meters (symmetric).")
parser.add_argument("--contact_threshold", type=float, default=1.0, help="Contact force threshold in Newtons.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force single environment
args_cli.num_envs = 1

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
        left_contact: Binary contact state (0 or 1)
        right_contact: Binary contact state (0 or 1)
    """
    # Access contact sensor from scene.sensors dictionary
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]

    # Get force magnitudes for each wheel
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
    left_contact = (wheel_l_force > threshold).float().item()
    right_contact = (wheel_r_force > threshold).float().item()

    return left_contact, right_contact


def get_pose_error(env):
    """Get position error from pose target.

    Args:
        env: The wrapped environment

    Returns:
        position_error: Scalar with L2 position error for robot 0
    """
    command_term = env.unwrapped.command_manager.get_term("base_pose")
    position_error = command_term.metrics["position_error"]
    return position_error.item()  # Return scalar for single env


def get_wheel_velocities(env):
    """Get angular velocities of left and right wheels.

    Args:
        env: The wrapped environment

    Returns:
        left_vel: Left wheel angular velocity (rad/s)
        right_vel: Right wheel angular velocity (rad/s)
    """
    robot = env.unwrapped.scene["robot"]

    # Find wheel joint indices
    wheel_l_idx, _ = robot.find_joints("wheel_L_Joint")
    wheel_r_idx, _ = robot.find_joints("wheel_R_Joint")

    # Get joint velocities
    left_vel = robot.data.joint_vel[:, wheel_l_idx[0]].item()
    right_vel = robot.data.joint_vel[:, wheel_r_idx[0]].item()

    return left_vel, right_vel


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

    # Calculate total steps
    total_steps = args_cli.warmup_steps + args_cli.collection_steps

    print(f"\n[INFO] Starting experiment...")
    print(f"  Warmup steps: {args_cli.warmup_steps}")
    print(f"  Collection steps: {args_cli.collection_steps}")
    print(f"  Total steps: {total_steps}")
    print(f"  Expected duration: {total_steps * args_cli.dt:.2f} seconds")

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
            # Get position error
            pos_error = get_pose_error(env)

            # Get wheel contacts
            left_contact, right_contact = get_wheel_contacts(env, threshold=args_cli.contact_threshold)

            # Get wheel velocities
            left_vel, right_vel = get_wheel_velocities(env)

            # Store data
            data_collection.append({
                'step': step - args_cli.warmup_steps,
                'pos_error': pos_error,
                'left_contact': left_contact,
                'right_contact': right_contact,
                'left_vel': left_vel,
                'right_vel': right_vel,
            })

            # Print progress
            if (step - args_cli.warmup_steps) % 50 == 0:
                current_time = (step - args_cli.warmup_steps) * args_cli.dt
                print(f"  Recording step {step - args_cli.warmup_steps}/{args_cli.collection_steps}, "
                      f"t={current_time:.2f}s, pos_error: {pos_error:.4f} m")

    if len(data_collection) == 0:
        print(f"\n[ERROR] No data collected!")
        env.close()
        return

    print(f"\n[INFO] Data collection complete!")
    print(f"  Total frames recorded: {len(data_collection)}")

    # Generate visualization
    print(f"\n[INFO] Generating visualization...")
    visualize_pos_motion(data_collection, output_dir, args_cli.dt)

    # Save data
    save_data(data_collection, output_dir)

    print(f"\n[INFO] Experiment complete! Results saved to {output_dir}")

    # Close the environment
    env.close()


def visualize_pos_motion(data_collection, output_dir, dt):
    """Create visualization with position error and wheel patterns.

    Args:
        data_collection: List of data frames
        output_dir: Directory to save plots
        dt: Simulation timestep
    """
    # Extract data
    steps = np.array([frame['step'] for frame in data_collection])
    times = steps * dt
    pos_errors = np.array([frame['pos_error'] for frame in data_collection])
    left_contacts = np.array([frame['left_contact'] for frame in data_collection])
    right_contacts = np.array([frame['right_contact'] for frame in data_collection])
    left_vels = np.array([frame['left_vel'] for frame in data_collection])
    right_vels = np.array([frame['right_vel'] for frame in data_collection])

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Subplot 1: Position Tracking Error
    ax1.plot(times, pos_errors, linewidth=2, color='darkorange', label='Position Error')
    ax1.set_ylabel('Position Error (m)', fontsize=11)
    ax1.set_title('Position Controlled Motion Analysis', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)

    # Subplot 2: Left Wheel
    # Wheel velocity on primary y-axis (left)
    ax2_vel = ax2
    ax2_vel.plot(times, left_vels, linewidth=1.5, color='steelblue', label='Rolling Speed')
    ax2_vel.set_ylabel('Angular Velocity (rad/s)', fontsize=10, color='steelblue')
    # ax2_vel.set_ylim(-5, 10)
    ax2_vel.tick_params(axis='y', labelcolor='steelblue')
    ax2_vel.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2_vel.grid(True, alpha=0.3)

    # Contact pattern on secondary y-axis (right)
    ax2_contact = ax2_vel.twinx()
    ax2_contact.fill_between(times, 0, left_contacts,
                             where=left_contacts > 0.5,
                             color='orange', alpha=0.3,
                             label='Contact', step='mid')
    ax2_contact.set_ylim(-0.1, 1.1)
    ax2_contact.set_yticks([0, 1])
    ax2_contact.set_yticklabels([])

    # Combine legends
    lines1, labels1 = ax2_vel.get_legend_handles_labels()
    lines2, labels2 = ax2_contact.get_legend_handles_labels()
    ax2_vel.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    # Subplot 3: Right Wheel
    # Wheel velocity on primary y-axis (left)
    ax3_vel = ax3
    ax3_vel.plot(times, right_vels, linewidth=1.5, color='steelblue', label='Rolling Speed')
    ax3_vel.set_ylabel('Angular Velocity (rad/s)', fontsize=10, color='steelblue')
    ax3_vel.set_xlabel('Time (s)', fontsize=11)
    # ax3_vel.set_ylim(-5, 10)
    ax3_vel.tick_params(axis='y', labelcolor='steelblue')
    ax3_vel.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3_vel.grid(True, alpha=0.3)

    # Contact pattern on secondary y-axis (right)
    ax3_contact = ax3_vel.twinx()
    ax3_contact.fill_between(times, 0, right_contacts,
                             where=right_contacts > 0.5,
                             color='orange', alpha=0.3,
                             label='Contact', step='mid')
    ax3_contact.set_ylim(-0.1, 1.1)
    ax3_contact.set_yticks([0, 1])
    ax3_contact.set_yticklabels([])

    # Combine legends
    lines1, labels1 = ax3_vel.get_legend_handles_labels()
    lines2, labels2 = ax3_contact.get_legend_handles_labels()
    ax3_vel.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'position_motion.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved visualization to: {output_path}")

    plt.close('all')


def save_data(data_collection, output_dir):
    """Save data to CSV and summary files.

    Args:
        data_collection: List of data frames
        output_dir: Directory to save CSV
    """
    import csv

    # Save detailed data
    csv_path = output_dir / 'position_motion_data.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'time', 'position_error', 'left_contact', 'right_contact',
                        'left_wheel_vel', 'right_wheel_vel'])

        for frame in data_collection:
            writer.writerow([
                frame['step'],
                frame['step'] * args_cli.dt,
                frame['pos_error'],
                frame['left_contact'],
                frame['right_contact'],
                frame['left_vel'],
                frame['right_vel'],
            ])

    print(f"  Saved detailed data to: {csv_path}")

    # Save summary statistics
    summary_path = output_dir / 'summary_statistics.txt'

    # Calculate statistics
    pos_errors = np.array([frame['pos_error'] for frame in data_collection])
    left_contacts = np.array([frame['left_contact'] for frame in data_collection])
    right_contacts = np.array([frame['right_contact'] for frame in data_collection])
    left_vels = np.array([frame['left_vel'] for frame in data_collection])
    right_vels = np.array([frame['right_vel'] for frame in data_collection])

    with open(summary_path, 'w') as f:
        f.write("POSITION CONTROLLED MOTION EXPERIMENT SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Position range:             [{-args_cli.pos_range}, {args_cli.pos_range}] m\n")
        f.write(f"Warmup steps:               {args_cli.warmup_steps}\n")
        f.write(f"Collection steps:           {args_cli.collection_steps}\n")
        f.write(f"Timestep:                   {args_cli.dt} s\n")
        f.write(f"Contact threshold:          {args_cli.contact_threshold} N\n\n")

        f.write("POSITION TRACKING STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mean position error:        {pos_errors.mean():.6f} m\n")
        f.write(f"Std position error:         {pos_errors.std():.6f} m\n")
        f.write(f"Min position error:         {pos_errors.min():.6f} m\n")
        f.write(f"Max position error:         {pos_errors.max():.6f} m\n\n")

        f.write("CONTACT STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Left wheel contact time:    {left_contacts.mean() * 100:.2f}%\n")
        f.write(f"Right wheel contact time:   {right_contacts.mean() * 100:.2f}%\n")
        both_contact = (left_contacts * right_contacts).mean() * 100
        f.write(f"Both wheels contact time:   {both_contact:.2f}%\n")
        no_contact = ((1 - left_contacts) * (1 - right_contacts)).mean() * 100
        f.write(f"No contact time:            {no_contact:.2f}%\n\n")

        f.write("WHEEL VELOCITY STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Left wheel mean velocity:   {left_vels.mean():.4f} rad/s\n")
        f.write(f"Left wheel std velocity:    {left_vels.std():.4f} rad/s\n")
        f.write(f"Right wheel mean velocity:  {right_vels.mean():.4f} rad/s\n")
        f.write(f"Right wheel std velocity:   {right_vels.std():.4f} rad/s\n")

    print(f"  Saved summary statistics to: {summary_path}")


if __name__ == "__main__":
    # Run the experiment
    run_experiment()
    # Close sim app
    simulation_app.close()
