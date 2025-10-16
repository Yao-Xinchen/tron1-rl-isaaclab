"""Visualization script for contact pattern experiment data."""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def find_latest_data_dir(base_dir: str = "experiments/contact_pattern"):
    """Find the latest data directory.

    Args:
        base_dir: Base directory containing timestamped subdirectories

    Returns:
        Path to latest data directory
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_path}")

    # Find all subdirectories with timestamp format (YYYYMMDD_HHMMSS)
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]

    if not subdirs:
        raise FileNotFoundError(f"No data directories found in: {base_path}")

    # Sort by name (which is timestamp) and get latest
    latest_dir = sorted(subdirs)[-1]

    return latest_dir


def load_contact_data(data_dir: Path):
    """Load contact pattern data from CSV.

    Args:
        data_dir: Directory containing contact_data.csv

    Returns:
        DataFrame with contact data
    """
    csv_path = data_dir / "contact_data.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Contact data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    print(f"Loaded data from: {csv_path}")
    print(f"  Total records: {len(df)}")
    print(f"  Robot IDs: {df['robot_id'].min()} to {df['robot_id'].max()}")
    print(f"  Steps: {df['step'].min()} to {df['step'].max()}")

    return df


def visualize_robots(df, robot_ids, dt, output_path=None, xlim=None):
    """Visualize contact patterns for specified robots.

    Args:
        df: DataFrame with contact data
        robot_ids: List of robot IDs to visualize
        dt: Simulation timestep in seconds
        output_path: Optional path to save figure
        xlim: Optional tuple (xmin, xmax) to set x-axis limits
    """
    num_robots = len(robot_ids)

    # Create figure with subplots for each robot
    fig = plt.figure(figsize=(16, 4 * num_robots))

    for idx, robot_id in enumerate(robot_ids):
        # Filter data for this robot
        robot_data = df[df['robot_id'] == robot_id].sort_values('step')

        if len(robot_data) == 0:
            print(f"Warning: No data found for robot {robot_id}")
            continue

        # Extract data
        steps = robot_data['step'].values
        times = steps * dt
        pose_errors = robot_data['pose_error'].values
        left_contacts = robot_data['left_contact'].values
        right_contacts = robot_data['right_contact'].values

        # Create subplot for this robot
        ax_pos = fig.add_subplot(num_robots, 1, idx + 1)

        # Plot position error
        ax_pos.plot(times, pose_errors, linewidth=2, color='darkorange', label='Position Error')
        ax_pos.set_ylabel('Position Error (m)', fontsize=10)
        ax_pos.set_title(f'Robot {robot_id} - Pose Tracking and Wheel Contact Pattern', fontsize=11)
        ax_pos.grid(True, alpha=0.3)

        # Create second y-axis for contact visualization
        ax_contact = ax_pos.twinx()

        # Plot contact patterns as filled regions (thin bars)
        # Bars centered at 2/3 (top) and 1/3 (bottom) height
        bar_height = 0.1  # Half-height of each bar
        left_center = 2/3  # Left wheel at 2/3 height
        right_center = 1/3  # Right wheel at 1/3 height

        # Left wheel (centered at 2/3)
        ax_contact.fill_between(times,
                               left_center - bar_height,
                               left_center + left_contacts * bar_height,
                               where=left_contacts > 0.5,
                               color='steelblue', alpha=0.3,
                               label='Left Wheel Contact', step='mid')

        # Right wheel (centered at 1/3)
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

        # Set x-axis limits if specified
        if xlim is not None:
            ax_pos.set_xlim(xlim)

        if idx == num_robots - 1:
            ax_pos.set_xlabel('Time (s)', fontsize=10)

        # Combine legends
        lines1, labels1 = ax_pos.get_legend_handles_labels()
        lines2, labels2 = ax_contact.get_legend_handles_labels()
        ax_pos.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()

    plt.close('all')


def create_summary_plot(df, dt, output_path=None):
    """Create summary statistics plot across all robots.

    Args:
        df: DataFrame with contact data
        dt: Simulation timestep in seconds
        output_path: Optional path to save figure
    """
    # Get unique steps
    steps = df['step'].unique()
    steps.sort()
    times = steps * dt

    # Calculate statistics across all robots for each step
    mean_pose_error = []
    std_pose_error = []
    left_contact_pct = []
    right_contact_pct = []

    for step in steps:
        step_data = df[df['step'] == step]
        mean_pose_error.append(step_data['pose_error'].mean())
        std_pose_error.append(step_data['pose_error'].std())
        left_contact_pct.append(step_data['left_contact'].mean() * 100)
        right_contact_pct.append(step_data['right_contact'].mean() * 100)

    mean_pose_error = np.array(mean_pose_error)
    std_pose_error = np.array(std_pose_error)
    left_contact_pct = np.array(left_contact_pct)
    right_contact_pct = np.array(right_contact_pct)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Mean pose error over time
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
    ax2.plot(times, left_contact_pct, linewidth=2, color='steelblue', label='Left Wheel Contact %')
    ax2.plot(times, right_contact_pct, linewidth=2, color='orange', label='Right Wheel Contact %')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Contact Percentage (%)', fontsize=11)
    ax2.set_title('Wheel Contact Statistics Across All Robots', fontsize=12)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary plot to: {output_path}")
    else:
        plt.show()

    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description="Visualize contact pattern experiment data.")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Path to data directory. If not specified, uses latest.")
    parser.add_argument("--base_dir", type=str, default="experiments/contact_pattern",
                       help="Base directory containing experiment results.")
    parser.add_argument("--robot_ids", type=int, nargs='+', default=None,
                       help="Robot IDs to visualize. If not specified, samples 6 robots evenly.")
    parser.add_argument("--dt", type=float, default=0.02,
                       help="Simulation timestep in seconds.")
    parser.add_argument("--xlim", type=float, nargs=2, default=None,
                       help="X-axis limits as two floats: xmin xmax (e.g., --xlim 0 2.5)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save output plots. If not specified, displays interactively.")
    parser.add_argument("--summary", action="store_true",
                       help="Also create summary plot across all robots.")

    args = parser.parse_args()

    # Find data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = find_latest_data_dir(args.base_dir)

    print(f"Using data directory: {data_dir}")

    # Load data
    df = load_contact_data(data_dir)

    # Determine robot IDs to visualize
    if args.robot_ids:
        robot_ids = args.robot_ids
    else:
        # Sample 6 robots evenly
        all_robot_ids = df['robot_id'].unique()
        num_robots = len(all_robot_ids)
        num_samples = min(6, num_robots)
        step = num_robots // num_samples
        robot_ids = sorted(all_robot_ids)[::step][:num_samples]

    print(f"Visualizing robots: {robot_ids}")

    # Prepare output paths
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        contact_output = output_dir / "contact_patterns_custom.png"
        summary_output = output_dir / "summary_statistics_custom.png"
    else:
        contact_output = None
        summary_output = None

    # Create visualizations
    xlim = tuple(args.xlim) if args.xlim else None
    visualize_robots(df, robot_ids, args.dt, contact_output, xlim)

    if args.summary:
        create_summary_plot(df, args.dt, summary_output)

    print("Visualization complete!")


if __name__ == "__main__":
    main()
