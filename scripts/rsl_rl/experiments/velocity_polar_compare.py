"""Script to compare velocity tracking experiments from two different branches."""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import uniform_filter1d


def load_experiment_data(experiment_dir):
    """Load tracking data from experiment directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        errors: Array of tracking errors
        angles: Array of command angles in radians
    """
    csv_path = Path(experiment_dir) / 'tracking_errors.csv'
    df = pd.read_csv(csv_path)

    errors = df['avg_error'].values
    angles = df['command_angle_rad'].values

    return errors, angles


def calculate_directional_statistics(errors, angles, num_bins=72):
    """Calculate directional mean and median using angular bins.

    Args:
        errors: Array of tracking errors
        angles: Array of command angles in radians
        num_bins: Number of angular bins

    Returns:
        theta_closed: Angles for plotting (closed curve)
        means_closed: Smoothed mean errors per direction
        medians_closed: Smoothed median errors per direction
    """
    polar_bin_edges = np.linspace(0, 2*np.pi, num_bins + 1)
    polar_bin_indices = np.digitize(angles, polar_bin_edges) - 1
    polar_bin_indices = np.clip(polar_bin_indices, 0, num_bins - 1)

    polar_bin_means = []
    polar_bin_medians = []
    polar_bin_centers = []

    for i in range(num_bins):
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

    # Convert to arrays
    polar_bin_means = np.array(polar_bin_means)
    polar_bin_medians = np.array(polar_bin_medians)
    polar_bin_centers = np.array(polar_bin_centers)

    # Fill NaN values with interpolation
    if np.any(np.isnan(polar_bin_means)):
        valid_mask = ~np.isnan(polar_bin_means)
        polar_bin_means = np.interp(polar_bin_centers, polar_bin_centers[valid_mask], polar_bin_means[valid_mask])
        polar_bin_medians = np.interp(polar_bin_centers, polar_bin_centers[valid_mask], polar_bin_medians[valid_mask])

    # Smooth the curves for better visualization
    smooth_means = uniform_filter1d(polar_bin_means, size=max(3, num_bins//15), mode='wrap')
    smooth_medians = uniform_filter1d(polar_bin_medians, size=max(3, num_bins//15), mode='wrap')

    # Close the curve by appending first point at the end
    theta_closed = np.append(polar_bin_centers, polar_bin_centers[0])
    means_closed = np.append(smooth_means, smooth_means[0])
    medians_closed = np.append(smooth_medians, smooth_medians[0])

    return theta_closed, means_closed, medians_closed


def create_comparison_plot(baseline_dir, ours_dir, output_path=None, polar_radius=None):
    """Create side-by-side comparison of velocity tracking experiments.

    Args:
        baseline_dir: Path to baseline (twist rewarded) experiment
        ours_dir: Path to our (pose rewarded) experiment
        output_path: Path to save output figure (optional)
        polar_radius: Maximum radius for polar plots (auto-computed if None)
    """
    # Load data from both experiments
    print(f"[INFO] Loading baseline data from: {baseline_dir}")
    baseline_errors, baseline_angles = load_experiment_data(baseline_dir)

    print(f"[INFO] Loading our data from: {ours_dir}")
    ours_errors, ours_angles = load_experiment_data(ours_dir)

    # Calculate statistics
    print(f"\n[INFO] Baseline (twist rewarded) statistics:")
    print(f"  Mean error: {baseline_errors.mean():.4f} m/s")
    print(f"  Median error: {np.median(baseline_errors):.4f} m/s")
    print(f"  Std error: {baseline_errors.std():.4f} m/s")

    print(f"\n[INFO] Ours (pose rewarded) statistics:")
    print(f"  Mean error: {ours_errors.mean():.4f} m/s")
    print(f"  Median error: {np.median(ours_errors):.4f} m/s")
    print(f"  Std error: {ours_errors.std():.4f} m/s")

    # Calculate improvement
    mean_improvement = (baseline_errors.mean() - ours_errors.mean()) / baseline_errors.mean() * 100
    print(f"\n[INFO] Mean error improvement: {mean_improvement:.1f}%")

    # Determine display radius
    if polar_radius is not None:
        error_max_display = polar_radius
    else:
        # Use 95th percentile of the worse experiment
        baseline_95th = np.percentile(baseline_errors, 95)
        ours_95th = np.percentile(ours_errors, 95)
        error_max_display = min(max(baseline_95th, ours_95th) * 1.2, 2.0)

    # Calculate directional statistics
    baseline_theta, baseline_means, baseline_medians = calculate_directional_statistics(baseline_errors, baseline_angles)
    ours_theta, ours_means, ours_medians = calculate_directional_statistics(ours_errors, ours_angles)

    # Create side-by-side polar plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), subplot_kw=dict(projection='polar'))

    # Baseline plot (left)
    ax1.set_theta_zero_location('N')
    display_mask_baseline = baseline_errors <= error_max_display
    scatter1 = ax1.scatter(baseline_angles[display_mask_baseline], baseline_errors[display_mask_baseline],
                          c=baseline_errors[display_mask_baseline], cmap='RdYlBu_r', alpha=1.0, s=20,
                          vmin=min(baseline_errors.min(), ours_errors.min()),
                          vmax=error_max_display)
    ax1.set_ylim(0, error_max_display)
    ax1.plot(baseline_theta, baseline_means, 'k-', linewidth=1.5,
            label=f'Mean: {baseline_errors.mean():.3f} m/s')
    ax1.plot(baseline_theta, baseline_medians, 'k--', linewidth=1.5,
            label=f'Median: {np.median(baseline_errors):.3f} m/s')
    ax1.set_title('Twist Rewarded (Baseline)', fontsize=14, pad=40, fontweight='bold')
    ax1.set_ylabel('Error (m/s)', labelpad=40)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    # Ours plot (right)
    ax2.set_theta_zero_location('N')
    display_mask_ours = ours_errors <= error_max_display
    scatter2 = ax2.scatter(ours_angles[display_mask_ours], ours_errors[display_mask_ours],
                          c=ours_errors[display_mask_ours], cmap='RdYlBu_r', alpha=1.0, s=20,
                          vmin=min(baseline_errors.min(), ours_errors.min()),
                          vmax=error_max_display)
    ax2.set_ylim(0, error_max_display)
    ax2.plot(ours_theta, ours_means, 'k-', linewidth=1.5,
            label=f'Mean: {ours_errors.mean():.3f} m/s')
    ax2.plot(ours_theta, ours_medians, 'k--', linewidth=1.5,
            label=f'Median: {np.median(ours_errors):.3f} m/s')
    ax2.set_title('Pose Rewarded (Ours)', fontsize=14, pad=40, fontweight='bold')
    ax2.set_ylabel('Error (m/s)', labelpad=40)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    # Add overall title with improvement
    fig.suptitle(f'Velocity Tracking Error Comparison\n'
                f'Mean Error Improvement: {mean_improvement:.1f}% '
                f'({baseline_errors.mean():.3f} -> {ours_errors.mean():.3f} m/s)',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure
    if output_path is None:
        output_path = Path('experiments/tracking_error/comparison.png')
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[INFO] Saved comparison plot to: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare velocity tracking experiments from two branches.")
    parser.add_argument("--baseline_dir", type=str, default="experiments/velocity_polar/twist/20251017_173014",
                       help="Path to baseline (twist rewarded) experiment directory")
    parser.add_argument("--ours_dir", type=str, default="experiments/velocity_polar/pose/20251017_172342",
                       help="Path to our (pose rewarded) experiment directory")
    parser.add_argument("--output_path", type=str, default="experiments/velocity_polar/comparison_plane.png",
                       help="Path to save comparison plot")
    parser.add_argument("--polar_radius", type=float, default=None,
                       help="Maximum radius for polar plots (m/s). If not specified, auto-computed from data.")

    args = parser.parse_args()

    create_comparison_plot(args.baseline_dir, args.ours_dir, args.output_path, args.polar_radius)


if __name__ == "__main__":
    main()