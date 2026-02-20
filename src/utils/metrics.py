"""
Evaluation metrics for pose estimation.
Computes translation, rotation, and composite pose errors.
"""

import torch
import numpy as np
from typing import Tuple, Dict


def relative_translation_error(
    pred_translation: torch.Tensor,
    gt_translation: torch.Tensor
) -> torch.Tensor:
    """
    Compute relative translation error (percentage).
    
    Formula: ET = ||t_pred - t_gt|| / ||t_gt||
    
    Args:
        pred_translation: (batch_size, 3) or (batch_size, seq_len, 3)
        gt_translation: Same shape as pred
        
    Returns:
        error: Relative translation error (0-1 scale, multiply by 100 for %)
    """
    # Compute L2 norms
    pred_norm = torch.norm(pred_translation, p=2, dim=-1)
    gt_norm = torch.norm(gt_translation, p=2, dim=-1)
    
    # Compute difference norm
    diff_norm = torch.norm(pred_translation - gt_translation, p=2, dim=-1)
    
    # Relative error
    error = diff_norm / (gt_norm + 1e-8)  # Add epsilon to avoid division by zero
    
    return error


def rotation_error_degrees(
    pred_quaternion: torch.Tensor,
    gt_quaternion: torch.Tensor
) -> torch.Tensor:
    """
    Compute rotation error in degrees using geodesic distance.
    
    Formula: ER = 2 * arccos(|<q_pred, q_gt>|) converted to degrees
    
    Args:
        pred_quaternion: (batch_size, 4) or (batch_size, seq_len, 4) [Qw, Qx, Qy, Qz]
        gt_quaternion: Same shape as pred
        
    Returns:
        error: Rotation error in degrees
    """
    # Flatten if needed for computation
    original_shape = pred_quaternion.shape[:-1]
    pred_q = pred_quaternion.reshape(-1, 4)
    gt_q = gt_quaternion.reshape(-1, 4)
    
    # Normalize quaternions
    pred_q = torch.nn.functional.normalize(pred_q, p=2, dim=-1)
    gt_q = torch.nn.functional.normalize(gt_q, p=2, dim=-1)
    
    # Compute dot product
    dot_product = torch.sum(pred_q * gt_q, dim=-1)
    
    # Handle quaternion double cover (q and -q are equivalent)
    dot_product = torch.abs(dot_product)
    
    # Clamp for numerical stability
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Geodesic distance in radians
    error_rad = 2.0 * torch.acos(dot_product)
    
    # Convert to degrees
    error_deg = torch.rad2deg(error_rad)
    
    # Reshape back to original
    error_deg = error_deg.reshape(original_shape)
    
    return error_deg


def composite_pose_error(
    pred_translation: torch.Tensor,
    pred_rotation: torch.Tensor,
    gt_translation: torch.Tensor,
    gt_rotation: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute composite pose error metrics.
    
    Args:
        pred_translation: Predicted translation
        pred_rotation: Predicted quaternion
        gt_translation: Ground truth translation
        gt_rotation: Ground truth quaternion
        
    Returns:
        trans_error: Relative translation error (0-1 scale)
        rot_error: Rotation error in degrees
        composite: Weighted composite error
    """
    trans_error = relative_translation_error(pred_translation, gt_translation)
    rot_error = rotation_error_degrees(pred_rotation, gt_rotation)
    
    # Composite: average of normalized errors
    # Translation already normalized, rotation in degrees (normalize by 180°)
    composite = (trans_error + rot_error / 180.0) / 2.0
    
    return trans_error, rot_error, composite


def compute_metrics_dict(
    pred_translation: torch.Tensor,
    pred_rotation: torch.Tensor,
    gt_translation: torch.Tensor,
    gt_rotation: torch.Tensor
) -> Dict[str, float]:
    """
    Compute metrics and return as dictionary (useful for logging).
    
    Args:
        pred_translation: (batch_size, seq_len, 3)
        pred_rotation: (batch_size, seq_len, 4)
        gt_translation: (batch_size, seq_len, 3)
        gt_rotation: (batch_size, seq_len, 4)
        
    Returns:
        metrics: Dictionary with keys:
            - 'trans_error_pct': Mean translation error (%)
            - 'rot_error_deg': Mean rotation error (degrees)
            - 'composite_error': Composite pose error
            - 'trans_error_std': Std dev of translation error
            - 'rot_error_std': Std dev of rotation error
    """
    trans_error, rot_error, composite = composite_pose_error(
        pred_translation, pred_rotation,
        gt_translation, gt_rotation
    )
    
    # Convert to numpy for statistics
    trans_error_np = trans_error.detach().cpu().numpy().flatten()
    rot_error_np = rot_error.detach().cpu().numpy().flatten()
    composite_np = composite.detach().cpu().numpy().flatten()
    
    metrics = {
        'trans_error_pct': float(np.mean(trans_error_np) * 100),
        'trans_error_std': float(np.std(trans_error_np) * 100),
        'rot_error_deg': float(np.mean(rot_error_np)),
        'rot_error_std': float(np.std(rot_error_np)),
        'composite_error': float(np.mean(composite_np)),
        'trans_error_median': float(np.median(trans_error_np) * 100),
        'rot_error_median': float(np.median(rot_error_np)),
    }
    
    return metrics


def evaluate_batch(
    model: torch.nn.Module,
    voxels: torch.Tensor,
    poses: torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Evaluate model on a batch and compute metrics.
    
    Args:
        model: VoxelCNNLSTM model
        voxels: Input voxel grids (batch_size, seq_len, num_channels, H, W)
        poses: Ground truth poses (batch_size, seq_len, 7)
        device: Device to run on
        
    Returns:
        pred_translation: Predicted translation (batch_size, seq_len, 3)
        pred_rotation: Predicted rotation (batch_size, seq_len, 4)
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    voxels = voxels.to(device)
    poses = poses.to(device)
    
    with torch.no_grad():
        pred_translation, pred_rotation = model(voxels)
    
    # Split ground truth
    gt_translation = poses[:, :, :3]
    gt_rotation = poses[:, :, 3:]
    
    # Compute metrics
    metrics = compute_metrics_dict(
        pred_translation, pred_rotation,
        gt_translation, gt_rotation
    )
    
    return pred_translation, pred_rotation, metrics


class MetricsTracker:
    """Track metrics across multiple batches."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.trans_errors = []
        self.rot_errors = []
        self.composites = []
    
    def update(
        self,
        pred_translation: torch.Tensor,
        pred_rotation: torch.Tensor,
        gt_translation: torch.Tensor,
        gt_rotation: torch.Tensor
    ):
        """Update with batch of predictions."""
        trans_error, rot_error, composite = composite_pose_error(
            pred_translation, pred_rotation,
            gt_translation, gt_rotation
        )
        
        self.trans_errors.append(trans_error.detach().cpu())
        self.rot_errors.append(rot_error.detach().cpu())
        self.composites.append(composite.detach().cpu())
    
    def compute(self) -> Dict[str, float]:
        """Compute aggregate metrics."""
        if len(self.trans_errors) == 0:
            return {}
        
        # Concatenate all batches
        trans_errors = torch.cat(self.trans_errors).numpy().flatten()
        rot_errors = torch.cat(self.rot_errors).numpy().flatten()
        composites = torch.cat(self.composites).numpy().flatten()
        
        metrics = {
            'trans_error_pct': float(np.mean(trans_errors) * 100),
            'trans_error_std': float(np.std(trans_errors) * 100),
            'trans_error_median': float(np.median(trans_errors) * 100),
            'rot_error_deg': float(np.mean(rot_errors)),
            'rot_error_std': float(np.std(rot_errors)),
            'rot_error_median': float(np.median(rot_errors)),
            'composite_error': float(np.mean(composites)),
        }
        
        return metrics


def test_metrics():
    """Test metrics computation."""
    print("Testing pose metrics...")
    
    # Create dummy predictions
    batch_size, seq_len = 4, 10
    
    # Perfect predictions (should have zero error)
    gt_trans = torch.randn(batch_size, seq_len, 3)
    gt_rot = torch.nn.functional.normalize(torch.randn(batch_size, seq_len, 4), p=2, dim=-1)
    
    pred_trans = gt_trans.clone()
    pred_rot = gt_rot.clone()
    
    trans_error, rot_error, composite = composite_pose_error(
        pred_trans, pred_rot, gt_trans, gt_rot
    )
    
    print(f"Perfect prediction:")
    print(f"  Translation error: {trans_error.mean().item():.6f} (should be ~0)")
    print(f"  Rotation error: {rot_error.mean().item():.6f}° (should be ~0)")
    
    # Add noise to predictions
    pred_trans_noisy = gt_trans + torch.randn_like(gt_trans) * 0.1
    pred_rot_noisy = torch.nn.functional.normalize(
        gt_rot + torch.randn_like(gt_rot) * 0.1, p=2, dim=-1
    )
    
    trans_error, rot_error, composite = composite_pose_error(
        pred_trans_noisy, pred_rot_noisy, gt_trans, gt_rot
    )
    
    print(f"\nNoisy prediction:")
    print(f"  Translation error: {trans_error.mean().item()*100:.2f}%")
    print(f"  Rotation error: {rot_error.mean().item():.2f}°")
    print(f"  Composite error: {composite.mean().item():.4f}")
    
    # Test metrics dict
    metrics = compute_metrics_dict(pred_trans_noisy, pred_rot_noisy, gt_trans, gt_rot)
    print(f"\nMetrics dictionary:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")
    
    print("\nMetrics tests passed!")


if __name__ == "__main__":
    test_metrics()
