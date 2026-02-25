#!/usr/bin/env python3
"""
Train KeypointPoseNet to regress 8 satellite keypoints from event frames.

Uses L1 loss on visible keypoints only. Keypoints are normalized [0,1].
After training, use inference.py with the saved checkpoint — it auto-detects
the keypoint model and uses PnP to solve for 6-DoF pose.

Usage:
    # Generate labels first (one-time, ~1 min on CPU):
    python scripts/generate_keypoint_labels.py --h5-dir h5 --output-dir keypoint_labels --visualize

    # Verify keypoint_validation_RT*.png look correct, then train:
    python train_keypoints.py \
        --preprocessed-dir preprocessed_voxels_100pct \
        --keypoint-label-dir keypoint_labels \
        --amp

    # Inference with PnP:
    python inference.py \
        --checkpoint checkpoints/keypoint_best.pth \
        --test-dir SPARK_stream2_test_data \
        --output-dir submission_keypoint_pnp \
        --test-timestamp-scale 1.0 \
        --template SPARK_stream2_test_data/template.csv
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.keypoint_net import KeypointPoseNet
from src.data.keypoint_dataset import KeypointDataset
from src.data.dataset import select_stratified_subset, train_val_split

# ── 3D keypoints — must match scripts/generate_keypoint_labels.py ────────────
KEYPOINTS_3D = np.array([
    [+0.80, +0.30, +0.52],
    [+0.80, +0.30, -0.52],
    [+0.80, -0.30, +0.52],
    [+0.80, -0.30, -0.52],
    [-0.80, +0.30, +0.52],
    [-0.80, +0.30, -0.52],
    [-0.80, -0.30, +0.52],
    [-0.80, -0.30, -0.52],
], dtype=np.float64)

IMG_W, IMG_H = 1280, 720


# ─────────────────────────────────────────────────────────────────────────────
# Loss & metrics
# ─────────────────────────────────────────────────────────────────────────────

def masked_l1_loss(pred: torch.Tensor, gt: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
    """L1 loss on visible keypoints only. pred, gt: (B,8,2), vis: (B,8) bool."""
    mask = vis.unsqueeze(-1).expand_as(pred)
    if mask.sum() == 0:
        return torch.zeros(1, device=pred.device, requires_grad=True).squeeze()
    return F.l1_loss(pred[mask], gt[mask])


def mean_pixel_error(pred: torch.Tensor, gt: torch.Tensor, vis: torch.Tensor) -> float:
    """Mean Euclidean pixel error on visible keypoints."""
    scale = torch.tensor([IMG_W, IMG_H], device=pred.device, dtype=pred.dtype)
    pred_px = pred * scale
    gt_px   = gt   * scale
    dist    = torch.norm(pred_px - gt_px, dim=-1)   # (B, 8)
    if vis.sum() == 0:
        return 0.0
    return dist[vis].mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Train / val loops
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss, total_px, n = 0.0, 0.0, 0

    for voxels, kp_gt, vis in loader:
        voxels = voxels.to(device)       # (B, C, H, W)
        kp_gt  = kp_gt.to(device)        # (B, 8, 2)  normalized
        vis    = vis.to(device)           # (B, 8)     bool

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                kp_pred = model(voxels).view(-1, 8, 2)
                loss = masked_l1_loss(kp_pred, kp_gt, vis)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            kp_pred = model(voxels).view(-1, 8, 2)
            loss = masked_l1_loss(kp_pred, kp_gt, vis)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            px = mean_pixel_error(kp_pred.detach(), kp_gt, vis)

        total_loss += loss.item()
        total_px   += px
        n          += 1

    return total_loss / max(n, 1), total_px / max(n, 1)


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total_loss, total_px, n = 0.0, 0.0, 0

    for voxels, kp_gt, vis in loader:
        voxels = voxels.to(device)
        kp_gt  = kp_gt.to(device)
        vis    = vis.to(device)

        kp_pred = model(voxels).view(-1, 8, 2)
        loss    = masked_l1_loss(kp_pred, kp_gt, vis)
        px      = mean_pixel_error(kp_pred, kp_gt, vis)

        total_loss += loss.item()
        total_px   += px
        n          += 1

    return total_loss / max(n, 1), total_px / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train KeypointPoseNet')
    parser.add_argument('--preprocessed-dir',    default='preprocessed_voxels_100pct')
    parser.add_argument('--keypoint-label-dir',  default='keypoint_labels')
    parser.add_argument('--data-subset',         default='100pct',
                        choices=['25pct', '100pct'])
    parser.add_argument('--checkpoint-dir',      default='checkpoints')
    parser.add_argument('--checkpoint-name',     default='keypoint_best.pth')
    parser.add_argument('--resume',              default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--device',              default='cuda')
    parser.add_argument('--amp',                 action='store_true',
                        help='Mixed precision (AMP)')
    parser.add_argument('--num-workers',         type=int, default=8)
    # Hyperparameters
    parser.add_argument('--batch-size',          type=int,   default=32)
    parser.add_argument('--lr',                  type=float, default=1e-4)
    parser.add_argument('--weight-decay',        type=float, default=1e-5)
    parser.add_argument('--epochs',              type=int,   default=40)
    parser.add_argument('--patience',            type=int,   default=10)
    parser.add_argument('--min-visible',         type=int,   default=4,
                        help='Min visible keypoints per frame (default: 4)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    subset_pct = 0.25 if args.data_subset == '25pct' else 1.0
    all_ids    = select_stratified_subset(total_sequences=300, subset_pct=subset_pct)
    train_ids, val_ids = train_val_split(all_ids, val_ratio=0.1)

    train_ds = KeypointDataset(
        preprocessed_dir   = args.preprocessed_dir,
        keypoint_label_dir = args.keypoint_label_dir,
        sequence_ids       = train_ids,
        min_visible        = args.min_visible,
    )
    val_ds = KeypointDataset(
        preprocessed_dir   = args.preprocessed_dir,
        keypoint_label_dir = args.keypoint_label_dir,
        sequence_ids       = val_ids,
        min_visible        = args.min_visible,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = KeypointPoseNet(
        num_input_channels=3,
        dropout=0.2,
        pretrained_backbone=True,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scaler = (torch.cuda.amp.GradScaler()
              if (args.amp and device.type == 'cuda') else None)

    start_epoch    = 0
    best_val_loss  = float('inf')
    patience_counter = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch   = ckpt.get('epoch', -1) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTraining KeypointPoseNet on {device}")
    print(f"  Params:     {total_params:,}")
    print(f"  Train:      {len(train_ds):,} frames | Val: {len(val_ds):,} frames")
    print(f"  Batch size: {args.batch_size} | LR: {args.lr} | AMP: {args.amp}")
    print(f"  Checkpoint: {os.path.join(args.checkpoint_dir, args.checkpoint_name)}")
    print()

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_px = train_epoch(model, train_loader, optimizer, device, scaler)
        val_loss,   val_px   = val_epoch(model, val_loader, device)

        scheduler.step(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss    = val_loss
            patience_counter = 0
            ckpt_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss':        best_val_loss,
                'val_px_error':         val_px,
                # Embed 3D keypoints so inference.py can load them without extra config
                'keypoints_3d':         KEYPOINTS_3D,
                'model_type':           'KeypointPoseNet',
            }, ckpt_path)
            marker = " ✓ BEST"
        else:
            patience_counter += 1
            marker = f" (patience {patience_counter}/{args.patience})"

        print(
            f"Epoch [{epoch+1:3d}/{args.epochs}]"
            f"  Train loss: {train_loss:.4f}  px: {train_px:5.1f}"
            f"  Val loss: {val_loss:.4f}  px: {val_px:5.1f}"
            f"{marker}"
        )

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print(f"\n{'─'*60}")
    print(f"Training complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint:    {os.path.join(args.checkpoint_dir, args.checkpoint_name)}")
    print()
    print("Decision gate for 3D keypoint correctness:")
    print(f"  val px < 15 px  → keypoints accurate, submit PnP result")
    print(f"  val px 15-30 px → marginal — compare PnP submission vs CNN+GRU")
    print(f"  val px > 30 px  → poor regression — check KEYPOINTS_3D dimensions")


if __name__ == '__main__':
    main()
