"""
Main training script for SPADES pose estimation.
Supports 25% and 100% data subset training.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
from typing import Dict

from src.data.dataset import (SPADESVoxelDataset, DomainVoxelDataset, train_val_split,
                               ComposeTransforms, RandomIntensityScale, RandomFrameDropout,
                               SaltPepperNoise, RandomErasing)
from src.data.event_representations import SignedVoxelGridGenerator, ThreeChannelEventFrame
from src.models.cnn_lstm_voxel import DirectPoseCNN, count_parameters
from src.models.domain_adaptive_pose_net import DomainAdaptivePoseNet
from src.losses.pose_loss import CompositePoseLoss
from src.utils.metrics import compute_metrics_dict, MetricsTracker


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DirectPoseCNN for pose estimation')
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--data-subset', type=str, default='25pct',
                        choices=['25pct', '100pct'],
                        help='Data subset to use (25pct or 100pct)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on (cuda or cpu)')
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed precision training (AMP)')
    parser.add_argument('--preprocessed-dir', type=str, default=None,
                        help='Directory with preprocessed voxel H5 files. '
                             'Defaults to preprocessed_voxels_<data-subset>.')
    parser.add_argument('--dann', action='store_true',
                        help='Enable DANN domain adaptation (requires sequence_length > 1)')
    parser.add_argument('--dann-test-dir', type=str, default=None,
                        help='Directory with preprocessed real test voxels for DANN')

    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: Dict, device: str, preprocessed_dir: str = None):
    """Create training and validation dataloaders."""
    
    # Load sequence IDs based on subset
    subset_file = config['data'][f'sequence_ids_{config["data_subset"]}']
    with open(subset_file, 'r') as f:
        sequence_ids = [line.strip() for line in f.readlines()]
    
    print(f"Loaded {len(sequence_ids)} sequence IDs from {subset_file}")
    
    # Split into train/val
    train_ids, val_ids = train_val_split(
        sequence_ids,
        val_ratio=config['data']['val_ratio'],
        seed=config['seed']
    )
    
    # Create voxel/event frame generator based on num_bins
    if config['data']['num_bins'] == 3:
        # 3-channel exponential decay representation
        voxel_generator = ThreeChannelEventFrame(
            height=config['data']['height'],
            width=config['data']['width'],
            window_size_us=config['data']['window_size_us']
        )
    else:
        # Standard voxel grid representation
        voxel_generator = SignedVoxelGridGenerator(
            height=config['data']['height'],
            width=config['data']['width'],
            num_bins=config['data']['num_bins'],
            window_size_us=config['data']['window_size_us']
        )
    
    # Create augmentation transforms (including event-camera specific augmentations)
    train_transform = ComposeTransforms([
        RandomIntensityScale(scale_range=(0.8, 1.2)),
        RandomFrameDropout(drop_prob=0.1),
        SaltPepperNoise(amount=0.02, prob=0.7),  # Simulate hot/dead pixels
        RandomErasing(prob=0.5, scale_range=(0.1, 0.3))  # Simulate dropped packets
    ])
    
    # Create datasets
    train_dataset = SPADESVoxelDataset(
        data_dir=config['data']['h5_dir'],
        sequence_ids=train_ids,
        sequence_length=config['model']['sequence_length'],
        sequence_stride=config['data']['sequence_stride'],
        voxel_generator=voxel_generator,
        min_events=config['data']['min_events'],
        transform=train_transform,
        preprocessed_dir=preprocessed_dir
    )
    
    val_dataset = SPADESVoxelDataset(
        data_dir=config['data']['h5_dir'],
        sequence_ids=val_ids,
        sequence_length=config['model']['sequence_length'],
        sequence_stride=config['data']['sequence_stride'],
        voxel_generator=voxel_generator,
        min_events=config['data']['min_events'],
        transform=None,
        preprocessed_dir=preprocessed_dir
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=(device == 'cuda'),
        prefetch_factor=4,  # Load 4 batches ahead per worker
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=(device == 'cuda'),
        prefetch_factor=4,
        persistent_workers=True
    )
    
    return train_loader, val_loader


def compute_grl_alpha(epoch: int, total_epochs: int) -> float:
    """Standard DANN schedule: sigmoid ramp from 0 → 1 over training."""
    import math
    p = epoch / max(total_epochs, 1)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


def _model_forward(model, voxels):
    """Call model and always return (trans, rot, domain_logits|None)."""
    out = model(voxels)
    if isinstance(out, tuple) and len(out) == 3:
        return out[0], out[1], out[2]
    return out[0], out[1], None


def _get_last_pose(poses):
    """Return the last pose in a sequence batch.

    poses: (batch, seq_len, 7)  or  (batch, 1, 7) → (batch, 3), (batch, 4)
    """
    last = poses[:, -1]  # (batch, 7)
    return last[:, :3], last[:, 3:]


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    scaler: GradScaler = None,
    dann_loader=None,
    lambda_domain: float = 0.0,
) -> Dict[str, float]:
    """Train for one epoch with optional mixed precision (AMP)."""
    
    model.train()
    metrics_tracker = MetricsTracker()

    total_loss = 0.0
    total_trans_loss = 0.0
    total_rot_loss = 0.0
    num_batches = 0

    # DANN: cycle over real test voxels alongside synthetic training batches
    domain_iter = iter(dann_loader) if (dann_loader is not None and lambda_domain > 0) else None
    domain_criterion = nn.CrossEntropyLoss()

    for batch_idx, (voxels, poses) in enumerate(dataloader):
        voxels = voxels.to(device)  # (batch, seq_len, C, H, W)
        poses = poses.to(device)    # (batch, seq_len, 7)

        gt_translation, gt_rotation = _get_last_pose(poses)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                pred_translation, pred_rotation, domain_logits = _model_forward(model, voxels)
                loss, trans_loss, rot_loss = criterion(
                    pred_translation, pred_rotation, gt_translation, gt_rotation
                )
                if domain_logits is not None and lambda_domain > 0 and domain_iter is not None:
                    # Synthetic domain label = 0
                    syn_labels = torch.zeros(voxels.size(0), dtype=torch.long, device=device)
                    d_loss = domain_criterion(domain_logits, syn_labels)
                    # Real domain: label = 1
                    try:
                        real_voxels = next(domain_iter).to(device)
                    except StopIteration:
                        domain_iter = iter(dann_loader)
                        real_voxels = next(domain_iter).to(device)
                    _, _, real_domain_logits = _model_forward(model, real_voxels)
                    real_labels = torch.ones(real_voxels.size(0), dtype=torch.long, device=device)
                    d_loss = (d_loss + domain_criterion(real_domain_logits, real_labels)) / 2.0
                    loss = loss + lambda_domain * d_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_translation, pred_rotation, domain_logits = _model_forward(model, voxels)
            loss, trans_loss, rot_loss = criterion(
                pred_translation, pred_rotation, gt_translation, gt_rotation
            )
            if domain_logits is not None and lambda_domain > 0 and domain_iter is not None:
                syn_labels = torch.zeros(voxels.size(0), dtype=torch.long, device=device)
                d_loss = domain_criterion(domain_logits, syn_labels)
                try:
                    real_voxels = next(domain_iter).to(device)
                except StopIteration:
                    domain_iter = iter(dann_loader)
                    real_voxels = next(domain_iter).to(device)
                _, _, real_domain_logits = _model_forward(model, real_voxels)
                real_labels = torch.ones(real_voxels.size(0), dtype=torch.long, device=device)
                d_loss = (d_loss + domain_criterion(real_domain_logits, real_labels)) / 2.0
                loss = loss + lambda_domain * d_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        metrics_tracker.update(pred_translation, pred_rotation, gt_translation, gt_rotation)

        total_loss += loss.item()
        total_trans_loss += trans_loss.item()
        total_rot_loss += rot_loss.item()
        num_batches += 1

        if (batch_idx + 1) % 10 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Batch/Loss', loss.item(), global_step)
            writer.add_scalar('Train/Batch/TransLoss', trans_loss.item(), global_step)
            writer.add_scalar('Train/Batch/RotLoss', rot_loss.item(), global_step)
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}] - "
                  f"Loss: {loss.item():.4f}, Trans: {trans_loss.item():.4f}, Rot: {rot_loss.item():.4f}")
    
    # Compute epoch metrics
    avg_loss = total_loss / num_batches
    avg_trans_loss = total_trans_loss / num_batches
    avg_rot_loss = total_rot_loss / num_batches
    
    pose_metrics = metrics_tracker.compute()
    
    metrics = {
        'loss': avg_loss,
        'trans_loss': avg_trans_loss,
        'rot_loss': avg_rot_loss,
        **pose_metrics
    }
    
    return metrics


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validate model."""
    
    model.eval()
    metrics_tracker = MetricsTracker()
    
    total_loss = 0.0
    total_trans_loss = 0.0
    total_rot_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for voxels, poses in dataloader:
            voxels = voxels.to(device)
            poses = poses.to(device)

            gt_translation, gt_rotation = _get_last_pose(poses)

            pred_translation, pred_rotation, _ = _model_forward(model, voxels)

            loss, trans_loss, rot_loss = criterion(
                pred_translation, pred_rotation,
                gt_translation, gt_rotation
            )
            
            # Track metrics
            metrics_tracker.update(pred_translation, pred_rotation, gt_translation, gt_rotation)
            
            total_loss += loss.item()
            total_trans_loss += trans_loss.item()
            total_rot_loss += rot_loss.item()
            num_batches += 1
    
    # Compute metrics
    avg_loss = total_loss / num_batches
    avg_trans_loss = total_trans_loss / num_batches
    avg_rot_loss = total_rot_loss / num_batches
    
    pose_metrics = metrics_tracker.compute()
    
    metrics = {
        'loss': avg_loss,
        'trans_loss': avg_trans_loss,
        'rot_loss': avg_rot_loss,
        **pose_metrics
    }
    
    return metrics


def main():
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    config['data_subset'] = args.data_subset
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # --- ADD THIS FOR A100 OPTIMIZATION ---
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # auto-tune cuDNN kernels for fixed input sizes
        print("Enabled TF32 + cudnn.benchmark for Ampere GPU acceleration")
    
    # Set random seed
    torch.manual_seed(config['seed'])
    
    # Resolve preprocessed directory
    preprocessed_dir = args.preprocessed_dir or f"preprocessed_voxels_{args.data_subset}"
    print(f"Preprocessed voxels dir: {preprocessed_dir}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(config, args.device, preprocessed_dir)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # DANN domain dataloader (real test voxels, no labels)
    dann_loader = None
    if args.dann:
        if not args.dann_test_dir:
            raise ValueError("--dann requires --dann-test-dir pointing to preprocessed real test voxels")
        seq_len = config['model']['sequence_length']
        dann_dataset = DomainVoxelDataset(
            preprocessed_dir=args.dann_test_dir,
            sequence_length=seq_len,
        )
        dann_loader = DataLoader(
            dann_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=2,        # 2 workers — minimal footprint alongside 16 train workers
            prefetch_factor=1,    # 2 batches queued total (2w×1) — ~3.5 GB vs 28 GB at default
            pin_memory=False,     # don't pin domain data at native 720×1280 res
        )
        print(f"DANN domain loader: {len(dann_loader)} batches from {args.dann_test_dir}")

    # Create model — branch on sequence_length
    print("\nCreating model...")
    backbone = config['model'].get('backbone', 'resnet50')
    seq_len = config['model']['sequence_length']
    if seq_len > 1 or args.dann:
        model = DomainAdaptivePoseNet(
            num_input_channels=config['model']['num_input_channels'],
            backbone=backbone,
            dropout=config['model']['dropout'],
            pretrained_backbone=config['model']['pretrained_backbone'],
            grl_alpha=0.0,
        )
        model_name = f"DomainAdaptivePoseNet (GRU, seq={seq_len}, backbone={backbone})"
    else:
        model = DirectPoseCNN(
            num_input_channels=config['model']['num_input_channels'],
            dropout=config['model']['dropout'],
            pretrained_backbone=config['model']['pretrained_backbone'],
            backbone=backbone,
        )
        model_name = f"DirectPoseCNN (backbone={backbone})"
    model = model.to(device)
    num_params = count_parameters(model)
    print(f"Model: {model_name}")
    print(f"Model parameters: {num_params:,}")

    # Compile model for ~20-30% additional throughput on A100
    if device.type == 'cuda':
        try:
            model = torch.compile(model)
            print("torch.compile() applied")
        except Exception as e:
            print(f"torch.compile() skipped: {e}")
    
    # Create loss
    criterion = CompositePoseLoss(
        lambda_translation=config['loss']['lambda_translation'],
        lambda_rotation=config['loss']['lambda_rotation']
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['training']['scheduler_factor'],
        patience=config['training']['scheduler_patience']
    )
    
    # Create tensorboard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(config['logging']['tensorboard_dir'], 
                           f"{args.data_subset}_{timestamp}")
    writer = SummaryWriter(log_dir)
    print(f"Tensorboard logs: {log_dir}")
    
    # Create GradScaler for AMP if enabled
    scaler = None
    if args.amp:
        scaler = GradScaler()
        print("Mixed Precision Training (AMP) enabled")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Override lr from current config (handles batch-size changes between runs)
            for pg in optimizer.param_groups:
                pg['lr'] = config['training']['learning_rate']
        else:
            print("  Warning: optimizer state not in checkpoint, starting optimizer fresh")
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler and checkpoint.get('scaler_state_dict') is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        start_epoch = checkpoint.get('epoch', -1) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"  Resumed epoch {start_epoch}, best_val_loss={best_val_loss:.4f}, "
              f"patience_counter={epochs_without_improvement}")
    else:
        epochs_without_improvement = 0

    # Training loop
    print(f"\nStarting training for {config['training']['num_epochs']} epochs...")
    
    total_epochs = config['training']['num_epochs']
    for epoch in range(start_epoch, total_epochs):
        print(f"\nEpoch {epoch+1}/{total_epochs}")

        # Ramp GRL alpha (DANN schedule: 0 → 1 over all epochs)
        lambda_domain = 0.0
        if dann_loader is not None and hasattr(model, 'set_grl_alpha'):
            alpha = compute_grl_alpha(epoch, total_epochs)
            model.set_grl_alpha(alpha)
            lambda_domain = alpha
            print(f"  GRL alpha: {alpha:.4f}, lambda_domain: {lambda_domain:.4f}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, scaler,
            dann_loader=dann_loader, lambda_domain=lambda_domain,
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Log metrics
        writer.add_scalar('Epoch/Train/Loss', train_metrics['loss'], epoch)
        writer.add_scalar('Epoch/Train/TransError', train_metrics['trans_error_pct'], epoch)
        writer.add_scalar('Epoch/Train/RotError', train_metrics['rot_error_deg'], epoch)
        
        writer.add_scalar('Epoch/Val/Loss', val_metrics['loss'], epoch)
        writer.add_scalar('Epoch/Val/TransError', val_metrics['trans_error_pct'], epoch)
        writer.add_scalar('Epoch/Val/RotError', val_metrics['rot_error_deg'], epoch)
        
        writer.add_scalar('Epoch/LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print summary
        print(f"\n  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Trans: {train_metrics['trans_error_pct']:.2f}%, "
              f"Rot: {train_metrics['rot_error_deg']:.2f}°")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Trans: {val_metrics['trans_error_pct']:.2f}%, "
              f"Rot: {val_metrics['rot_error_deg']:.2f}°")
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Save checkpoint
        checkpoint_path = os.path.join(config['logging']['checkpoint_dir'],
                                       f"{args.data_subset}_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'epochs_without_improvement': epochs_without_improvement,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_val_loss': best_val_loss,
            'config': config
        }, checkpoint_path)
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_path = os.path.join(config['logging']['checkpoint_dir'],
                                     f"{args.data_subset}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, best_path)
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= config['training']['early_stopping_patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    writer.close()
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {best_path}")


if __name__ == "__main__":
    main()
