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
from datetime import datetime
from typing import Dict

from src.data.dataset import SPADESVoxelDataset, train_val_split, ComposeTransforms, RandomIntensityScale, RandomFrameDropout
from src.data.event_representations import SignedVoxelGridGenerator, ThreeChannelEventFrame
from src.models.cnn_lstm_voxel import DirectPoseCNN, count_parameters
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
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: Dict, device: str):
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
    
    # Create augmentation transforms
    train_transform = ComposeTransforms([
        RandomIntensityScale(scale_range=(0.8, 1.2)),
        RandomFrameDropout(drop_prob=0.1)
    ])
    
    # Create datasets
    train_dataset = SPADESVoxelDataset(
        data_dir=config['data']['h5_dir'],
        sequence_ids=train_ids,
        sequence_length=config['model']['sequence_length'],
        sequence_stride=config['data']['sequence_stride'],
        voxel_generator=voxel_generator,
        min_events=config['data']['min_events'],
        transform=train_transform
    )
    
    val_dataset = SPADESVoxelDataset(
        data_dir=config['data']['h5_dir'],
        sequence_ids=val_ids,
        sequence_length=config['model']['sequence_length'],
        sequence_stride=config['data']['sequence_stride'],
        voxel_generator=voxel_generator,
        min_events=config['data']['min_events'],
        transform=None  # No augmentation for validation
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=(device == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=(device == 'cuda')
    )
    
    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter
) -> Dict[str, float]:
    """Train for one epoch."""
    
    model.train()
    metrics_tracker = MetricsTracker()
    
    total_loss = 0.0
    total_trans_loss = 0.0
    total_rot_loss = 0.0
    num_batches = 0
    
    for batch_idx, (voxels, poses) in enumerate(dataloader):
        voxels = voxels.to(device)  # (batch, seq_len, num_channels, H, W)
        poses = poses.to(device)    # (batch, seq_len, 7)
        
        # Split poses
        gt_translation = poses[:, :, :3]
        gt_rotation = poses[:, :, 3:]
        
        # Forward pass
        pred_translation, pred_rotation = model(voxels)
        
        # Compute loss
        loss, trans_loss, rot_loss = criterion(
            pred_translation, pred_rotation,
            gt_translation, gt_rotation
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        metrics_tracker.update(pred_translation, pred_rotation, gt_translation, gt_rotation)
        
        total_loss += loss.item()
        total_trans_loss += trans_loss.item()
        total_rot_loss += rot_loss.item()
        num_batches += 1
        
        # Log every 10 batches
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
            
            # Split poses
            gt_translation = poses[:, :, :3]
            gt_rotation = poses[:, :, 3:]
            
            # Forward pass
            pred_translation, pred_rotation = model(voxels)
            
            # Compute loss
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
    
    # Set random seed
    torch.manual_seed(config['seed'])
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(config, args.device)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = DirectPoseCNN(
        num_input_channels=config['model']['num_input_channels'],
        dropout=config['model']['dropout'],
        pretrained_backbone=config['model']['pretrained_backbone']
    )
    model = model.to(device)
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
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
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
    
    # Training loop
    print(f"\nStarting training for {config['training']['num_epochs']} epochs...")
    epochs_without_improvement = 0
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        
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
