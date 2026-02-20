"""
Loss functions for pose estimation.
Includes composite loss with geodesic distance for rotation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CompositePoseLoss(nn.Module):
    """
    Composite loss for pose estimation combining translation and rotation losses.
    
    Components:
        - Translation: MSE loss on [Tx, Ty, Tz]
        - Rotation: Geodesic distance on SO(3) for quaternions
    
    Args:
        lambda_translation: Weight for translation loss (default: 1.0)
        lambda_rotation: Weight for rotation loss (default: 1.0)
        reduction: Loss reduction ('mean' or 'sum', default: 'mean')
    """
    
    def __init__(
        self,
        lambda_translation: float = 1.0,
        lambda_rotation: float = 1.0,
        reduction: str = 'mean'
    ):
        super(CompositePoseLoss, self).__init__()
        self.lambda_t = lambda_translation
        self.lambda_r = lambda_rotation
        self.reduction = reduction
        
    def translation_loss(
        self,
        pred_translation: torch.Tensor,
        gt_translation: torch.Tensor
    ) -> torch.Tensor:
        """
        MSE loss for translation.
        
        Args:
            pred_translation: (batch_size, seq_len, 3) or (batch_size*seq_len, 3)
            gt_translation: Same shape as pred_translation
            
        Returns:
            loss: Scalar translation loss
        """
        loss = F.mse_loss(pred_translation, gt_translation, reduction=self.reduction)
        return loss
    
    def rotation_loss(
        self,
        pred_quaternion: torch.Tensor,
        gt_quaternion: torch.Tensor
    ) -> torch.Tensor:
        """
        Geodesic distance loss for rotation (quaternions).
        
        Formula: d(q1, q2) = 2 * arccos(|<q1, q2>|)
        
        Args:
            pred_quaternion: (batch_size, seq_len, 4) or (batch_size*seq_len, 4)
                            Format: [Qw, Qx, Qy, Qz]
            gt_quaternion: Same shape as pred_quaternion
            
        Returns:
            loss: Scalar rotation loss in radians
        """
        # Flatten if needed
        pred_q = pred_quaternion.view(-1, 4)
        gt_q = gt_quaternion.view(-1, 4)
        
        # Ensure quaternions are normalized
        pred_q = F.normalize(pred_q, p=2, dim=-1)
        gt_q = F.normalize(gt_q, p=2, dim=-1)
        
        # Compute dot product: <q_pred, q_gt>
        dot_product = torch.sum(pred_q * gt_q, dim=-1)
        
        # Take absolute value to handle quaternion double cover (q and -q represent same rotation)
        dot_product = torch.abs(dot_product)
        
        # Clamp to valid range [-1, 1] for numerical stability
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        
        # Geodesic distance: 2 * arccos(|dot_product|)
        geodesic_distance = 2.0 * torch.acos(dot_product)
        
        # Reduction
        if self.reduction == 'mean':
            loss = torch.mean(geodesic_distance)
        elif self.reduction == 'sum':
            loss = torch.sum(geodesic_distance)
        else:
            loss = geodesic_distance
        
        return loss
    
    def forward(
        self,
        pred_translation: torch.Tensor,
        pred_rotation: torch.Tensor,
        gt_translation: torch.Tensor,
        gt_rotation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute composite pose loss.
        
        Args:
            pred_translation: (batch_size, seq_len, 3)
            pred_rotation: (batch_size, seq_len, 4)
            gt_translation: (batch_size, seq_len, 3)
            gt_rotation: (batch_size, seq_len, 4)
            
        Returns:
            total_loss: Weighted sum of translation and rotation losses
            trans_loss: Translation loss component
            rot_loss: Rotation loss component
        """
        # Compute individual losses
        trans_loss = self.translation_loss(pred_translation, gt_translation)
        rot_loss = self.rotation_loss(pred_rotation, gt_rotation)
        
        # Weighted composite loss
        total_loss = self.lambda_t * trans_loss + self.lambda_r * rot_loss
        
        return total_loss, trans_loss, rot_loss


class TranslationLoss(nn.Module):
    """Standalone translation loss (MSE)."""
    
    def __init__(self, reduction: str = 'mean'):
        super(TranslationLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, gt, reduction=self.reduction)


class RotationLoss(nn.Module):
    """Standalone rotation loss (geodesic distance)."""
    
    def __init__(self, reduction: str = 'mean'):
        super(RotationLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, pred_q: torch.Tensor, gt_q: torch.Tensor) -> torch.Tensor:
        # Flatten
        pred_q = pred_q.view(-1, 4)
        gt_q = gt_q.view(-1, 4)
        
        # Normalize
        pred_q = F.normalize(pred_q, p=2, dim=-1)
        gt_q = F.normalize(gt_q, p=2, dim=-1)
        
        # Dot product
        dot_product = torch.sum(pred_q * gt_q, dim=-1)
        dot_product = torch.abs(dot_product)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        
        # Geodesic distance
        geodesic_distance = 2.0 * torch.acos(dot_product)
        
        # Reduction
        if self.reduction == 'mean':
            return torch.mean(geodesic_distance)
        elif self.reduction == 'sum':
            return torch.sum(geodesic_distance)
        else:
            return geodesic_distance


def test_losses():
    """Test loss functions with dummy data."""
    print("Testing pose losses...")
    
    # Dummy predictions and ground truth
    batch_size = 4
    seq_len = 10
    
    pred_trans = torch.randn(batch_size, seq_len, 3)
    gt_trans = torch.randn(batch_size, seq_len, 3)
    
    # Random quaternions (normalized)
    pred_rot = F.normalize(torch.randn(batch_size, seq_len, 4), p=2, dim=-1)
    gt_rot = F.normalize(torch.randn(batch_size, seq_len, 4), p=2, dim=-1)
    
    # Test composite loss
    criterion = CompositePoseLoss(lambda_translation=1.0, lambda_rotation=1.0)
    total_loss, trans_loss, rot_loss = criterion(pred_trans, pred_rot, gt_trans, gt_rot)
    
    print(f"Translation loss: {trans_loss.item():.4f}")
    print(f"Rotation loss (radians): {rot_loss.item():.4f}")
    print(f"Rotation loss (degrees): {torch.rad2deg(rot_loss).item():.2f}°")
    print(f"Total loss: {total_loss.item():.4f}")
    
    # Test with identical quaternions (should be ~0)
    zero_rot_loss = criterion.rotation_loss(gt_rot, gt_rot)
    print(f"Rotation loss (identical): {zero_rot_loss.item():.6f} (should be ~0)")
    
    # Test with opposite quaternions (q and -q, should also be ~0)
    opposite_rot_loss = criterion.rotation_loss(-gt_rot, gt_rot)
    print(f"Rotation loss (opposite sign): {opposite_rot_loss.item():.6f} (should be ~0)")
    
    print("Loss tests passed!")


if __name__ == "__main__":
    test_losses()
