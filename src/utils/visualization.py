"""
Visualization utilities for training and evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple
import torch


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Optional path to save figure
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Training Loss', marker='o')
    ax.plot(epochs, val_losses, label='Validation Loss', marker='s')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_metric_curves(
    metrics_history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot training metrics over epochs.
    
    Args:
        metrics_history: Dictionary mapping metric names to lists of values
        save_path: Optional path to save figure
        
    Returns:
        fig: Matplotlib figure
    """
    num_metrics = len(metrics_history)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6*num_metrics, 5))
    
    if num_metrics == 1:
        axes = [axes]
    
    for idx, (metric_name, values) in enumerate(metrics_history.items()):
        ax = axes[idx]
        epochs = np.arange(1, len(values) + 1)
        ax.plot(epochs, values, marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_error_distribution(
    trans_errors: np.ndarray,
    rot_errors: np.ndarray,
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot distribution of translation and rotation errors.
    
    Args:
        trans_errors: Array of translation errors (%)
        rot_errors: Array of rotation errors (degrees)
        save_path: Optional path to save figure
        
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Translation error histogram
    ax = axes[0]
    ax.hist(trans_errors, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(trans_errors), color='red', linestyle='--', 
               label=f'Mean: {np.mean(trans_errors):.2f}%')
    ax.axvline(np.median(trans_errors), color='green', linestyle='--',
               label=f'Median: {np.median(trans_errors):.2f}%')
    ax.set_xlabel('Translation Error (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Translation Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rotation error histogram
    ax = axes[1]
    ax.hist(rot_errors, bins=50, alpha=0.7, edgecolor='black', color='orange')
    ax.axvline(np.mean(rot_errors), color='red', linestyle='--',
               label=f'Mean: {np.mean(rot_errors):.2f}°')
    ax.axvline(np.median(rot_errors), color='green', linestyle='--',
               label=f'Median: {np.median(rot_errors):.2f}°')
    ax.set_xlabel('Rotation Error (degrees)')
    ax.set_ylabel('Frequency')
    ax.set_title('Rotation Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_trajectory_3d(
    pred_trajectory: np.ndarray,
    gt_trajectory: np.ndarray,
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot 3D trajectories (predicted vs ground truth).
    
    Args:
        pred_trajectory: Predicted positions (N, 3) - [Tx, Ty, Tz]
        gt_trajectory: Ground truth positions (N, 3)
        save_path: Optional path to save figure
        
    Returns:
        fig: Matplotlib figure
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2],
            'b-', label='Ground Truth', linewidth=2, alpha=0.7)
    ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2],
            'r--', label='Predicted', linewidth=2, alpha=0.7)
    
    # Mark start and end points
    ax.scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], gt_trajectory[0, 2],
               c='green', s=100, marker='o', label='Start')
    ax.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], gt_trajectory[-1, 2],
               c='red', s=100, marker='x', label='End')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Spacecraft Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_voxel_grid(
    voxel_grid: np.ndarray,
    save_path: Optional[str] = None
) -> Figure:
    """
    Visualize a 5-channel voxel grid.
    
    Args:
        voxel_grid: Voxel grid (5, H, W) or (num_bins, H, W)
        save_path: Optional path to save figure
        
    Returns:
        fig: Matplotlib figure
    """
    num_bins = voxel_grid.shape[0]
    
    fig, axes = plt.subplots(1, num_bins, figsize=(4*num_bins, 4))
    
    if num_bins == 1:
        axes = [axes]
    
    for i in range(num_bins):
        ax = axes[i]
        im = ax.imshow(voxel_grid[i], cmap='RdBu_r', aspect='auto')
        ax.set_title(f'Temporal Bin {i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_pose_comparison(
    pred_poses: np.ndarray,
    gt_poses: np.ndarray,
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot pose component comparisons (translation and quaternion).
    
    Args:
        pred_poses: Predicted poses (N, 7) - [Tx, Ty, Tz, Qw, Qx, Qy, Qz]
        gt_poses: Ground truth poses (N, 7)
        save_path: Optional path to save figure
        
    Returns:
        fig: Matplotlib figure
    """
    N = pred_poses.shape[0]
    indices = np.arange(N)
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    
    # Translation components
    trans_labels = ['Tx', 'Ty', 'Tz']
    for i in range(3):
        ax = axes[0, i]
        ax.plot(indices, gt_poses[:, i], 'b-', label='GT', alpha=0.7)
        ax.plot(indices, pred_poses[:, i], 'r--', label='Pred', alpha=0.7)
        ax.set_xlabel('Frame')
        ax.set_ylabel(f'{trans_labels[i]} (m)')
        ax.set_title(f'Translation {trans_labels[i]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Quaternion components
    quat_labels = ['Qw', 'Qx', 'Qy', 'Qz']
    for i in range(4):
        ax = axes[0 if i == 3 else 1, (i+3) % 4]
        ax.plot(indices, gt_poses[:, i+3], 'b-', label='GT', alpha=0.7)
        ax.plot(indices, pred_poses[:, i+3], 'r--', label='Pred', alpha=0.7)
        ax.set_xlabel('Frame')
        ax.set_ylabel(quat_labels[i])
        ax.set_title(f'Rotation {quat_labels[i]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_batch_predictions(
    model: torch.nn.Module,
    voxels: torch.Tensor,
    poses: torch.Tensor,
    device: torch.device,
    num_samples: int = 2,
    save_dir: Optional[str] = None
):
    """
    Visualize model predictions for a batch.
    
    Args:
        model: Trained model
        voxels: Input voxel grids (batch_size, seq_len, num_channels, H, W)
        poses: Ground truth poses (batch_size, seq_len, 7)
        device: Device to run on
        num_samples: Number of samples to visualize
        save_dir: Directory to save figures (if None, displays only)
    """
    model.eval()
    voxels = voxels.to(device)
    poses = poses.to(device)
    
    with torch.no_grad():
        pred_trans, pred_rot = model(voxels)
    
    # Concatenate predictions
    pred_poses = torch.cat([pred_trans, pred_rot], dim=-1)  # (batch, seq, 7)
    
    # Convert to numpy
    pred_poses_np = pred_poses.cpu().numpy()
    gt_poses_np = poses.cpu().numpy()
    
    # Visualize first few samples
    for i in range(min(num_samples, voxels.shape[0])):
        save_path = f"{save_dir}/sample_{i}_comparison.png" if save_dir else None
        plot_pose_comparison(pred_poses_np[i], gt_poses_np[i], save_path)
        
        # 3D trajectory
        save_path = f"{save_dir}/sample_{i}_trajectory.png" if save_dir else None
        plot_trajectory_3d(pred_poses_np[i, :, :3], gt_poses_np[i, :, :3], save_path)
    
    plt.show()


if __name__ == "__main__":
    print("Visualization utilities loaded successfully.")
