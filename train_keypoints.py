#!/usr/bin/env python3
"""
Train keypoint models to predict satellite keypoints for PnP-based pose estimation.

Supports two model architectures:
  --model-type heatmap  (default, recommended)
      KeypointHeatmapNet: ResNet-50 + ConvTranspose decoder → K heatmaps (H/4 × W/4).
      Trained with MSE loss against 2D Gaussian targets.  Soft-argmax extracts coords.
      Removes GlobalAvgPool bottleneck — expected val px: 20–40 px.

  --model-type direct
      KeypointPoseNet: ResNet-50 → GlobalAvgPool → FC → K×2.
      Trained with masked L1 loss.  Original approach.  Expected val px: 100–160 px.

Uses masked loss on visible keypoints only. Keypoints are normalized [0,1].
After training, use inference.py with the saved checkpoint — it auto-detects
the model type from the checkpoint and uses PnP to solve for 6-DoF pose.

Usage:
    # Generate labels first (one-time, ~1 min on CPU):
    python scripts/generate_keypoint_labels.py --h5-dir h5 --output-dir keypoint_labels --visualize

    # Stage 1 — validate heatmap architecture at 256×448 (fast):
    HDF5_USE_FILE_LOCKING=FALSE python train_keypoints.py \\
        --preprocessed-dir preprocessed_voxels_256x448 \\
        --keypoint-label-dir keypoint_labels \\
        --amp --num-workers 16 --batch-size 256

    # Stage 2 — full resolution + 14 keypoints:
    HDF5_USE_FILE_LOCKING=FALSE python train_keypoints.py \\
        --preprocessed-dir preprocessed_voxels_100pct \\
        --keypoint-label-dir keypoint_labels_14 \\
        --num-keypoints 14 --batch-size 32 --num-workers 8 --amp

    # Inference with PnP:
    python inference.py \\
        --checkpoint checkpoints/keypoint_best.pth \\
        --test-dir SPARK_stream2_test_data \\
        --output-dir submission_keypoint_pnp \\
        --test-timestamp-scale 1.0 \\
        --template SPARK_stream2_test_data/template.csv
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.keypoint_net import KeypointPoseNet
from src.models.keypoint_heatmap_net import (
    KeypointHeatmapNet, heatmap_loss, soft_argmax_2d,
)
from src.data.keypoint_dataset import KeypointDataset
from src.data.dataset import select_stratified_subset, train_val_split


def _train_augment(voxel: np.ndarray, kp_2d: np.ndarray, vis: np.ndarray):
    """Joint voxel + keypoint augmentation for training.

    Spatial augmentations (rotation, translation) consistently transform
    both the voxel grid and the 2D keypoint coordinates so labels stay valid.
    Non-spatial augmentations (noise, lines) modify only the voxel.

    Based on Jawaid et al. (SEENIC): RandomEventNoise, RandomEventLines,
    random rotation, random translation.

    Args:
        voxel:  (C, H, W) float32 numpy, at training resolution (256×448)
        kp_2d:  (8, 2)   float32 numpy, pixel coords in original 1280×720 space
        vis:    (8,)     bool numpy

    Returns: (voxel, kp_2d, vis) — kp_2d still in 1280×720 pixel space
    """
    C, H, W = voxel.shape
    voxel = voxel.copy()

    # Normalize kp to [0,1] for spatial transform math
    kp_norm = kp_2d / np.array([1280.0, 720.0], dtype=np.float32)  # (8, 2)

    # 1. Random rotation ±15° around image centre
    if np.random.rand() < 0.5:
        angle_deg = np.random.uniform(-15.0, 15.0)
        angle_rad = angle_deg * (np.pi / 180.0)
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle_deg, 1.0)
        for c in range(C):
            voxel[c] = cv2.warpAffine(voxel[c], M, (W, H),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0.0)
        # Same rotation in normalised [0,1] space around (0.5, 0.5)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        du = kp_norm[:, 0] - 0.5
        dv = kp_norm[:, 1] - 0.5
        kp_norm[:, 0] = 0.5 + cos_a * du + sin_a * dv
        kp_norm[:, 1] = 0.5 - sin_a * du + cos_a * dv
        vis = vis & (kp_norm[:, 0] >= 0) & (kp_norm[:, 0] < 1.0) & \
                    (kp_norm[:, 1] >= 0) & (kp_norm[:, 1] < 1.0)

    # 2. Random translation ±10% of image size
    if np.random.rand() < 0.5:
        dx_norm = np.random.uniform(-0.1, 0.1)
        dy_norm = np.random.uniform(-0.1, 0.1)
        M = np.float32([[1, 0, dx_norm * W], [0, 1, dy_norm * H]])
        for c in range(C):
            voxel[c] = cv2.warpAffine(voxel[c], M, (W, H),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0.0)
        kp_norm[:, 0] += dx_norm
        kp_norm[:, 1] += dy_norm
        vis = vis & (kp_norm[:, 0] >= 0) & (kp_norm[:, 0] < 1.0) & \
                    (kp_norm[:, 1] >= 0) & (kp_norm[:, 1] < 1.0)

    # Convert back to original 1280×720 pixel space for dataset normalisation
    kp_2d = kp_norm * np.array([1280.0, 720.0], dtype=np.float32)

    # 3. RandomEventNoise — intensity scale + hot/dead pixels
    voxel *= float(np.random.uniform(0.8, 1.2))
    if np.random.rand() < 0.7:
        n = max(1, int(0.01 * H * W))
        mv = float(np.abs(voxel).max()) + 0.5
        ys = np.random.randint(0, H, n)
        xs = np.random.randint(0, W, n)
        voxel[:, ys, xs] = mv                # salt (hot pixels)
        ys = np.random.randint(0, H, n)
        xs = np.random.randint(0, W, n)
        voxel[:, ys, xs] = 0.0              # pepper (dead pixels)

    # 4. RandomEventLines — horizontal/vertical artifact lines
    if np.random.rand() < 0.3:
        mv = float(np.abs(voxel).max())
        for _ in range(np.random.randint(1, 4)):
            if np.random.rand() < 0.5:
                voxel[:, np.random.randint(0, H), :] = mv   # horizontal
            else:
                voxel[:, :, np.random.randint(0, W)] = mv   # vertical

    return voxel, kp_2d, vis

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

def _forward(model, voxels, kp_gt, vis, is_heatmap: bool):
    """Unified forward pass for both model types.

    Returns:
        loss:    scalar training loss
        kp_pred: (B, K, 2) normalized [0,1] keypoint coords (for pixel error metric)
    """
    if is_heatmap:
        pred_hm = model(voxels)                                    # (B, K, H_hm, W_hm)
        loss    = heatmap_loss(pred_hm, kp_gt, vis)
        kp_pred = soft_argmax_2d(pred_hm.detach().float())        # (B, K, 2) for metric
    else:
        K       = kp_gt.shape[1]
        kp_pred = model(voxels).view(-1, K, 2)
        loss    = masked_l1_loss(kp_pred, kp_gt, vis)
    return loss, kp_pred


def train_epoch(model, loader, optimizer, device, scaler=None, resize_size=None,
                is_heatmap: bool = False):
    model.train()
    total_loss, total_px, n = 0.0, 0.0, 0
    for batch_idx, (voxels, kp_gt, vis) in enumerate(loader):
        voxels = voxels.to(device, non_blocking=True)
        if resize_size is not None:
            voxels = F.interpolate(voxels, size=resize_size, mode='bilinear', align_corners=False)
        kp_gt  = kp_gt.to(device, non_blocking=True)
        vis    = vis.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss, kp_pred = _forward(model, voxels, kp_gt, vis, is_heatmap)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, kp_pred = _forward(model, voxels, kp_gt, vis, is_heatmap)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            px = mean_pixel_error(kp_pred.detach(), kp_gt, vis)

        total_loss += loss.item()
        total_px   += px
        n          += 1

        if batch_idx % 200 == 0 and batch_idx > 0:
            print(f"  batch {batch_idx}/{len(loader)}", flush=True)

    return total_loss / max(n, 1), total_px / max(n, 1)


@torch.no_grad()
def val_epoch(model, loader, device, resize_size=None, is_heatmap: bool = False):
    model.eval()
    total_loss, total_px, n = 0.0, 0.0, 0

    for voxels, kp_gt, vis in loader:
        voxels = voxels.to(device, non_blocking=True)
        if resize_size is not None:
            voxels = F.interpolate(voxels, size=resize_size, mode='bilinear', align_corners=False)
        kp_gt  = kp_gt.to(device, non_blocking=True)
        vis    = vis.to(device, non_blocking=True)

        loss, kp_pred = _forward(model, voxels, kp_gt, vis, is_heatmap)
        px = mean_pixel_error(kp_pred.float(), kp_gt, vis)

        total_loss += loss.item()
        total_px   += px
        n          += 1

    return total_loss / max(n, 1), total_px / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train keypoint model (heatmap or direct)')
    parser.add_argument('--preprocessed-dir',    default='preprocessed_voxels_256x448')
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
    # Model type
    parser.add_argument('--model-type',          default='heatmap',
                        choices=['heatmap', 'direct'],
                        help='heatmap=KeypointHeatmapNet (recommended), direct=KeypointPoseNet')
    parser.add_argument('--num-keypoints',       type=int, default=8,
                        help='Number of keypoints K — must match keypoint label files (default: 8)')
    # Hyperparameters
    parser.add_argument('--batch-size',          type=int,   default=32)
    parser.add_argument('--lr',                  type=float, default=1e-4)
    parser.add_argument('--weight-decay',        type=float, default=1e-5)
    parser.add_argument('--epochs',              type=int,   default=40)
    parser.add_argument('--patience',            type=int,   default=10)
    parser.add_argument('--min-visible',         type=int,   default=4,
                        help='Min visible keypoints per frame (default: 4)')
    parser.add_argument('--input-size',          type=int,   nargs=2,
                        default=None, metavar=('H', 'W'),
                        help='Resize voxel to H×W in workers (default: no resize — use native resolution of --preprocessed-dir)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    subset_pct = 0.25 if args.data_subset == '25pct' else 1.0
    all_ids    = select_stratified_subset(total_sequences=300, subset_pct=subset_pct)
    train_ids, val_ids = train_val_split(all_ids, val_ratio=0.1)

    # Optional CPU resize in workers (reduces NFS bandwidth for full-res dirs).
    # If --input-size not set, use the native resolution of the preprocessed dir.
    if args.input_size is not None:
        in_h, in_w = args.input_size
        def resize_voxel(voxel):
            t = torch.from_numpy(np.array(voxel, dtype=np.float32)).unsqueeze(0)
            t = F.interpolate(t, size=(in_h, in_w), mode='bilinear', align_corners=False)
            return t.squeeze(0).numpy()
        transform_fn = resize_voxel
    else:
        in_h, in_w   = None, None   # determined at runtime from first batch
        transform_fn = None

    train_ds = KeypointDataset(
        preprocessed_dir   = args.preprocessed_dir,
        keypoint_label_dir = args.keypoint_label_dir,
        sequence_ids       = train_ids,
        min_visible        = args.min_visible,
        transform          = transform_fn,
        augment            = _train_augment,
    )
    val_ds = KeypointDataset(
        preprocessed_dir   = args.preprocessed_dir,
        keypoint_label_dir = args.keypoint_label_dir,
        sequence_ids       = val_ids,
        min_visible        = args.min_visible,
        transform          = transform_fn,
    )

    # Use fork (default) + open/close h5 per __getitem__ — same pattern as
    # SPADESVoxelDataset which works reliably on this server with num_workers=8.
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    is_heatmap = (args.model_type == 'heatmap')
    if is_heatmap:
        model = KeypointHeatmapNet(
            num_keypoints      = args.num_keypoints,
            num_input_channels = 3,
            pretrained_backbone= True,
        ).to(device)
        model_type_str = 'KeypointHeatmapNet'
    else:
        model = KeypointPoseNet(
            num_input_channels = 3,
            dropout            = 0.2,
            pretrained_backbone= True,
        ).to(device)
        model_type_str = 'KeypointPoseNet'

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    scaler = (torch.cuda.amp.GradScaler()
              if (args.amp and device.type == 'cuda') else None)

    start_epoch    = 0
    best_val_loss  = float('inf')
    patience_counter = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch   = ckpt.get('epoch', -1) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    input_desc = f"{in_h}×{in_w} (resized)" if in_h else "native resolution"
    print(f"\nTraining {model_type_str} on {device}")
    print(f"  Params:      {total_params:,}")
    print(f"  Keypoints:   {args.num_keypoints}")
    print(f"  Train:       {len(train_ds):,} frames | Val: {len(val_ds):,} frames")
    print(f"  Input size:  {input_desc}")
    print(f"  Batch size:  {args.batch_size} | LR: {args.lr} | AMP: {args.amp}")
    print(f"  Checkpoint:  {os.path.join(args.checkpoint_dir, args.checkpoint_name)}")
    print()

    # ── Training loop ─────────────────────────────────────────────────────────
    print("Starting training loop...", flush=True)
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1} starting...", flush=True)
        train_loss, train_px = train_epoch(
            model, train_loader, optimizer, device, scaler, is_heatmap=is_heatmap)
        val_loss,   val_px   = val_epoch(
            model, val_loader, device, is_heatmap=is_heatmap)

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
                'model_type':           model_type_str,
                'num_keypoints':        args.num_keypoints,
                # Embed input size — None means native resolution of preprocessed dir
                'input_size':           [in_h, in_w] if in_h else None,
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
    if is_heatmap:
        print("Decision gates (heatmap model):")
        print(f"  val px < 20 px  → excellent — PnP submission should beat CNN+GRU")
        print(f"  val px 20-40 px → good — try Stage 2 (full res + 14 keypoints)")
        print(f"  val px 40-80 px → marginal — check sigma (try 1.5 or 3.0)")
        print(f"  val px > 80 px  → bug — verify Gaussian target generation")
    else:
        print("Decision gates (direct regression model):")
        print(f"  val px < 20 px  → accurate — submit PnP result")
        print(f"  val px 20-50 px → marginal — compare vs CNN+GRU baseline")
        print(f"  val px > 80 px  → poor — switch to --model-type heatmap")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)
    main()
