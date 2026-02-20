"""
Inference script for test set prediction.
Generates submission CSV files for competition.
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

from load_h5 import load_h5_data
from src.data.event_representations import SignedVoxelGridGenerator
from src.models.cnn_lstm_voxel import VoxelCNNLSTM


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference on test set')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test-dir', type=str, default='h5',
                        help='Directory containing test .h5 files')
    parser.add_argument('--output-dir', type=str, default='submission',
                        help='Output directory for submission CSVs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on')
    parser.add_argument('--sequence-length', type=int, default=10,
                        help='Sequence length for LSTM')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for sliding window (1=overlap for averaging)')
    parser.add_argument('--test-timestamp-scale', type=float, default=1.0,
                        help='Timestamp scale for test data (1.0 for real 1µs data)')
    
    return parser.parse_args()


def predict_sequence(
    model: torch.nn.Module,
    events: dict,
    timestamps: np.ndarray,
    voxel_generator: SignedVoxelGridGenerator,
    sequence_length: int,
    stride: int,
    device: torch.device,
    timestamp_scale: float
) -> np.ndarray:
    """
    Predict poses for a full sequence with sliding window.
    
    Args:
        model: Trained VoxelCNNLSTM model
        events: Event dictionary from load_h5_data
        timestamps: Array of pose timestamps
        voxel_generator: SignedVoxelGridGenerator instance
        sequence_length: Number of frames per sequence
        stride: Stride for sliding window
        device: Device to run on
        timestamp_scale: Scale for timestamps (1.0 for 1µs, 100.0 for 100µs)
        
    Returns:
        predictions: Array of poses (num_timestamps, 7) [Tx, Ty, Tz, Qw, Qx, Qy, Qz]
    """
    model.eval()
    
    num_timestamps = len(timestamps)
    predictions = np.zeros((num_timestamps, 7), dtype=np.float32)
    counts = np.zeros(num_timestamps, dtype=np.float32)  # For averaging overlapping predictions
    
    # Generate sliding windows
    with torch.no_grad():
        for start_idx in range(0, num_timestamps - sequence_length + 1, stride):
            end_idx = start_idx + sequence_length
            
            # Generate voxel grids for this window
            window_timestamps = timestamps[start_idx:end_idx] * timestamp_scale
            voxel_list = []
            
            for ts in window_timestamps:
                t_start = ts
                t_end = t_start + voxel_generator.window_size_us
                voxel = voxel_generator.generate(events, t_start, t_end)
                voxel_list.append(voxel)
            
            # Stack and add batch dimension
            voxels = np.stack(voxel_list, axis=0)  # (seq_len, num_bins, H, W)
            voxels = torch.from_numpy(voxels).float().unsqueeze(0).to(device)  # (1, seq_len, num_bins, H, W)
            
            # Predict
            pred_translation, pred_rotation = model(voxels)
            
            # Concatenate
            pred_poses = torch.cat([pred_translation, pred_rotation], dim=-1)  # (1, seq_len, 7)
            pred_poses = pred_poses.squeeze(0).cpu().numpy()  # (seq_len, 7)
            
            # Accumulate predictions with averaging for overlap
            predictions[start_idx:end_idx] += pred_poses
            counts[start_idx:end_idx] += 1.0
    
    # Handle remaining frames at the end (if sequence_length doesn't divide evenly)
    if num_timestamps > sequence_length:
        last_start_idx = num_timestamps - sequence_length
        if counts[last_start_idx] == 0:
            # Predict for last window
            window_timestamps = timestamps[last_start_idx:] * timestamp_scale
            voxel_list = []
            
            with torch.no_grad():
                for ts in window_timestamps:
                    t_start = ts
                    t_end = t_start + voxel_generator.window_size_us
                    voxel = voxel_generator.generate(events, t_start, t_end)
                    voxel_list.append(voxel)
                
                voxels = np.stack(voxel_list, axis=0)
                voxels = torch.from_numpy(voxels).float().unsqueeze(0).to(device)
                
                pred_translation, pred_rotation = model(voxels)
                pred_poses = torch.cat([pred_translation, pred_rotation], dim=-1)
                pred_poses = pred_poses.squeeze(0).cpu().numpy()
                
                predictions[last_start_idx:] += pred_poses
                counts[last_start_idx:] += 1.0
    
    # Average overlapping predictions
    counts = np.maximum(counts, 1.0)  # Avoid division by zero
    predictions = predictions / counts[:, np.newaxis]
    
    # Renormalize quaternions
    quaternions = predictions[:, 3:]
    quaternions = quaternions / (np.linalg.norm(quaternions, axis=1, keepdims=True) + 1e-8)
    predictions[:, 3:] = quaternions
    
    return predictions


def create_submission_csv(
    predictions: np.ndarray,
    timestamps: np.ndarray,
    output_path: str
):
    """
    Create submission CSV file.
    
    Format: timestamp, Tx, Ty, Tz, Qx, Qy, Qz, Qw
    """
    df = pd.DataFrame({
        'timestamp': timestamps,
        'Tx': predictions[:, 0],
        'Ty': predictions[:, 1],
        'Tz': predictions[:, 2],
        'Qx': predictions[:, 4],  # Note: reordering from [Qw, Qx, Qy, Qz] to [Qx, Qy, Qz, Qw]
        'Qy': predictions[:, 5],
        'Qz': predictions[:, 6],
        'Qw': predictions[:, 3]
    })
    
    df.to_csv(output_path, index=False)
    print(f"Saved submission CSV: {output_path}")


def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    config = checkpoint.get('config', {})
    
    # Create model
    model = VoxelCNNLSTM(
        num_input_channels=config.get('model', {}).get('num_input_channels', 5),
        lstm_hidden_size=config.get('model', {}).get('lstm_hidden_size', 256),
        lstm_num_layers=config.get('model', {}).get('lstm_num_layers', 2),
        dropout=config.get('model', {}).get('dropout', 0.2),
        pretrained_backbone=False  # Not needed for inference
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    
    # Create voxel generator
    voxel_generator = SignedVoxelGridGenerator(
        height=config.get('data', {}).get('height', 720),
        width=config.get('data', {}).get('width', 1280),
        num_bins=config.get('data', {}).get('num_bins', 5),
        window_size_us=config.get('data', {}).get('window_size_us', 100000.0)
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find test files (T000.h5, T001.h5, etc.)
    test_pattern = os.path.join(args.test_dir, "T*.h5")
    test_files = sorted(glob(test_pattern))
    
    if len(test_files) == 0:
        print(f"Warning: No test files found matching {test_pattern}")
        print("Trying alternative pattern RT*.h5 (for synthetic test if needed)...")
        test_pattern = os.path.join(args.test_dir, "RT*.h5")
        test_files = sorted(glob(test_pattern))
    
    print(f"\nFound {len(test_files)} test sequences")
    
    # Process each test file
    for test_file in tqdm(test_files, desc="Processing sequences"):
        seq_id = os.path.basename(test_file).replace('.h5', '')
        
        # Load data
        data = load_h5_data(test_file)
        events = data['events']
        labels = data['labels']
        
        timestamps = labels['timestamp'].values
        
        print(f"\n{seq_id}: {len(timestamps)} poses, {len(events['t'])} events")
        
        # Run inference
        predictions = predict_sequence(
            model, events, timestamps,
            voxel_generator, args.sequence_length, args.stride,
            device, args.test_timestamp_scale
        )
        
        # Create submission CSV
        csv_path = os.path.join(args.output_dir, f"{seq_id}.csv")
        create_submission_csv(predictions, timestamps, csv_path)
    
    print(f"\n✓ Inference complete! Submission files saved to: {args.output_dir}")
    print(f"\nTo create submission.zip:")
    print(f"  cd {args.output_dir}")
    print(f"  zip -r ../submission.zip *.csv")


if __name__ == "__main__":
    main()
