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

import cv2
from scipy.spatial.transform import Rotation as ScipyRotation

from load_h5 import load_h5_data
from src.data.event_representations import SignedVoxelGridGenerator, ThreeChannelEventFrame
from src.models.cnn_lstm_voxel import DirectPoseCNN
from src.models.domain_adaptive_pose_net import DomainAdaptivePoseNet
from src.models.keypoint_net import KeypointPoseNet


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
    parser.add_argument('--num-frames', type=int, default=599,
                        help='Number of frames to predict for test data (when no labels/template available)')
    parser.add_argument('--template', type=str, default=None,
                        help='Path to template.csv for exact frame counts per sequence')
    
    return parser.parse_args()


def predict_frame_by_frame(
    model: torch.nn.Module,
    events: dict,
    timestamps: np.ndarray,
    event_generator,
    device: torch.device,
    timestamp_scale: float,
    is_dann: bool = False,
) -> np.ndarray:
    """
    Predict poses frame-by-frame (Direct CNN, no temporal model).
    
    Args:
        model: Trained DirectPoseCNN model
        events: Event dictionary from load_h5_data
        timestamps: Array of pose timestamps
        event_generator: ThreeChannelEventFrame or SignedVoxelGridGenerator instance
        device: Device to run on
        timestamp_scale: Scale for timestamps (1.0 for 1µs, 100.0 for 100µs)
        
    Returns:
        predictions: Array of poses (num_timestamps, 7) [Tx, Ty, Tz, Qw, Qx, Qy, Qz]
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for ts in tqdm(timestamps * timestamp_scale, desc="Predicting"):
            # Generate event frame for this timestamp
            t_start = ts
            t_end = t_start + event_generator.window_size_us
            event_frame = event_generator.generate(events, t_start, t_end)
            
            # Add batch dimension: (C, H, W) → (1, C, H, W)
            frame_tensor = torch.from_numpy(event_frame).float().unsqueeze(0).to(device)
            
            # Predict single pose (DomainAdaptivePoseNet needs seq dim; returns 3-tuple)
            out = model(frame_tensor.unsqueeze(1)) if is_dann else model(frame_tensor)
            pred_trans, pred_rot = out[0], out[1]
            
            # Concatenate: (1, 3) + (1, 4) → (1, 7)
            pose = torch.cat([pred_trans, pred_rot], dim=-1)
            predictions.append(pose.cpu().numpy())
    
    # Stack all predictions
    predictions = np.vstack(predictions)  # (num_timestamps, 7)
    
    # Renormalize quaternions to ensure unit norm
    quaternions = predictions[:, 3:]
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True) + 1e-8
    predictions[:, 3:] = quaternions
    
    return predictions


def predict_pnp(
    model: torch.nn.Module,
    events: dict,
    timestamps: np.ndarray,
    event_generator,
    device: torch.device,
    timestamp_scale: float,
    keypoints_3d: np.ndarray,
) -> np.ndarray:
    """
    Predict poses using CNN keypoint regression + PnP solver.

    The model predicts 8 × (u,v) normalized [0,1] for each event frame.
    cv2.solvePnP uses the known 3D body-frame keypoints and camera intrinsics
    to solve for 6-DoF pose geometrically.

    Args:
        model:          KeypointPoseNet
        events:         Event dictionary from load_h5_data
        timestamps:     Array of pose timestamps (in raw units before scale)
        event_generator: ThreeChannelEventFrame instance
        device:         Inference device
        timestamp_scale: Multiply timestamps by this before generating event frames
        keypoints_3d:   (8, 3) float64 — 3D keypoints in satellite body frame (meters)

    Returns:
        predictions: (num_timestamps, 7) [Tx, Ty, Tz, Qw, Qx, Qy, Qz]
    """
    # Camera intrinsics (from SPADES/camera.json)
    K = np.array([
        [1258.6057531097028, 0.0, 640.0],
        [0.0, 1258.6057531097028, 360.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    kp3d = keypoints_3d.reshape(-1, 1, 3).astype(np.float64)

    model.eval()
    num_timestamps = len(timestamps)
    predictions = np.zeros((num_timestamps, 7), dtype=np.float32)
    # Default fallback pose (identity rotation, 5m depth)
    predictions[:, 2] = 5.0
    predictions[:, 3] = 1.0   # Qw = 1

    last_valid_pose = None

    with torch.no_grad():
        for i, ts in enumerate(tqdm(timestamps * timestamp_scale, desc="Predicting (PnP)")):
            t_start = ts
            t_end   = t_start + event_generator.window_size_us
            frame   = event_generator.generate(events, t_start, t_end)

            frame_t = torch.from_numpy(frame).float().unsqueeze(0).to(device)
            kp_norm = model(frame_t).cpu().numpy().reshape(8, 2)   # normalized [0,1]

            # Back to pixel coordinates
            kp_px = kp_norm * np.array([[1280.0, 720.0]])   # (8, 2)
            kp_px_cv = kp_px.reshape(-1, 1, 2).astype(np.float64)

            # PnP solve
            ret, rvec, tvec, inliers = cv2.solvePnPRansac(
                kp3d, kp_px_cv, K, dist_coeffs,
                iterationsCount=100,
                reprojectionError=8.0,
                confidence=0.99,
                flags=cv2.SOLVEPNP_EPNP,
            )

            if not ret or inliers is None or len(inliers) < 4:
                # Fallback: use last valid prediction
                if last_valid_pose is not None:
                    predictions[i] = last_valid_pose
                continue

            # rvec → rotation matrix → quaternion [Qw, Qx, Qy, Qz]
            R_mat, _ = cv2.Rodrigues(rvec)
            quat_xyzw = ScipyRotation.from_matrix(R_mat).as_quat()  # [x, y, z, w]
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0],
                                  quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)

            pose = np.concatenate([tvec.flatten().astype(np.float32), quat_wxyz])
            predictions[i] = pose
            last_valid_pose = pose

    # Forward-fill any remaining fallback frames (where PnP failed)
    # Find first valid frame and backward-fill from it
    valid = ~np.all(predictions[:, 3:] == np.array([1.0, 0.0, 0.0, 0.0]), axis=1)
    valid &= ~(predictions[:, 2] == 5.0)   # not the default placeholder
    if valid.any():
        first = int(np.argmax(valid))
        if first > 0:
            predictions[:first] = predictions[first]
        for j in range(1, num_timestamps):
            if not valid[j]:
                predictions[j] = predictions[j - 1]

    # Renormalize quaternions
    quats = predictions[:, 3:]
    quats /= np.linalg.norm(quats, axis=1, keepdims=True) + 1e-8
    predictions[:, 3:] = quats

    return predictions


# Keep old function for compatibility with 5-channel voxel models (can be removed later)
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
    counts = np.zeros(num_timestamps, dtype=np.float32)

    # Sliding window — model outputs one pose per window (last GRU timestep only).
    # Store prediction only at the last frame index of each window so every
    # frame gets its own unique GRU-informed prediction (use stride=1).
    with torch.no_grad():
        for start_idx in range(0, num_timestamps - sequence_length + 1, stride):
            end_idx = start_idx + sequence_length
            last_idx = end_idx - 1  # prediction is for this frame only

            window_timestamps = timestamps[start_idx:end_idx] * timestamp_scale
            voxel_list = []
            for ts in window_timestamps:
                t_start = ts
                t_end = t_start + voxel_generator.window_size_us
                voxel = voxel_generator.generate(events, t_start, t_end)
                voxel_list.append(voxel)

            voxels = np.stack(voxel_list, axis=0)  # (seq_len, C, H, W)
            voxels = torch.from_numpy(voxels).float().unsqueeze(0).to(device)  # (1, seq_len, C, H, W)

            out = model(voxels)
            pred_translation, pred_rotation = out[0], out[1]
            pred_poses = torch.cat([pred_translation, pred_rotation], dim=-1)
            pred_poses = pred_poses.squeeze(0).cpu().numpy()  # (7,)

            # Store only at the last frame; average if multiple windows land here
            predictions[last_idx] += pred_poses
            counts[last_idx] += 1.0

    # Ensure the very last frame is covered when stride doesn't divide evenly
    if num_timestamps > sequence_length and counts[num_timestamps - 1] == 0:
        last_start_idx = num_timestamps - sequence_length
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
            out = model(voxels)
            pred_translation, pred_rotation = out[0], out[1]
            pred_poses = torch.cat([pred_translation, pred_rotation], dim=-1)
            pred_poses = pred_poses.squeeze(0).cpu().numpy()
            predictions[num_timestamps - 1] += pred_poses
            counts[num_timestamps - 1] += 1.0

    # Average frames that received multiple predictions
    valid = counts > 0
    predictions[valid] = predictions[valid] / counts[valid, np.newaxis]

    # Backward-fill frames before the first valid prediction (frames 0..seq_len-2)
    first_valid = int(np.argmax(valid)) if valid.any() else 0
    if first_valid > 0:
        predictions[:first_valid] = predictions[first_valid]

    # Forward-fill any remaining gaps (when stride > 1 leaves holes)
    for i in range(1, num_timestamps):
        if not valid[i]:
            predictions[i] = predictions[i - 1]

    # Renormalize quaternions
    quaternions = predictions[:, 3:]
    quaternions = quaternions / (np.linalg.norm(quaternions, axis=1, keepdims=True) + 1e-8)
    predictions[:, 3:] = quaternions

    return predictions


def create_submission_csv(
    predictions: np.ndarray,
    timestamps: np.ndarray,
    output_path: str,
    seq_id: str
):
    """
    Create submission CSV file.
    
    Format: timestamp, Tx, Ty, Tz, Qx, Qy, Qz, Qw
    Timestamp format: SEQID_NNN (e.g., RT901_001, RT901_002, ...)
    """
    # Format timestamps as SEQID_NNN (1-indexed, 3-digit padded)
    formatted_timestamps = [f"{seq_id}_{i+1:03d}" for i in range(len(predictions))]
    
    df = pd.DataFrame({
        'timestamp': formatted_timestamps,
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
    
    num_input_channels = config.get('model', {}).get('num_input_channels', 3)
    backbone = config.get('model', {}).get('backbone', 'resnet18')

    # Auto-detect model type from checkpoint state dict keys
    state_keys  = checkpoint['model_state_dict'].keys()
    is_keypoint = any(k.startswith('kp_head.') for k in state_keys)
    is_dann     = (not is_keypoint) and any(k.startswith('gru.') for k in state_keys)

    if is_keypoint:
        model = KeypointPoseNet(
            num_input_channels=num_input_channels,
            dropout=config.get('model', {}).get('dropout', 0.2),
            pretrained_backbone=False,
        )
        model_type = "KeypointPoseNet"
    elif is_dann:
        model = DomainAdaptivePoseNet(
            num_input_channels=num_input_channels,
            backbone=backbone,
            dropout=config.get('model', {}).get('dropout', 0.2),
            pretrained_backbone=False,
            grl_alpha=0.0,
        )
        model_type = "DomainAdaptivePoseNet"
    else:
        model = DirectPoseCNN(
            num_input_channels=num_input_channels,
            dropout=config.get('model', {}).get('dropout', 0.2),
            pretrained_backbone=False,
            backbone=backbone,
        )
        model_type = "DirectPoseCNN"

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully ({model_type}, {backbone} backbone)")
    
    # Create event frame generator based on num_bins
    num_bins = config.get('data', {}).get('num_bins', 3)
    
    if num_bins == 3:
        # 3-channel exponential decay representation
        event_generator = ThreeChannelEventFrame(
            height=config.get('data', {}).get('height', 720),
            width=config.get('data', {}).get('width', 1280),
            window_size_us=config.get('data', {}).get('window_size_us', 100000.0)
        )
    else:
        # Standard voxel grid representation
        event_generator = SignedVoxelGridGenerator(
            height=config.get('data', {}).get('height', 720),
            width=config.get('data', {}).get('width', 1280),
            num_bins=num_bins,
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
    
    # Get window size for timestamp generation
    window_size_us = config.get('data', {}).get('window_size_us', 100000.0)
    
    # Load template if provided (for exact frame counts per sequence)
    template_frames = {}
    if args.template and os.path.exists(args.template):
        template_df = pd.read_csv(args.template)
        # Parse sequence IDs and count frames per sequence
        for ts in template_df['timestamp']:
            seq_id = ts.rsplit('_', 1)[0]  # RT901_001 -> RT901
            template_frames[seq_id] = template_frames.get(seq_id, 0) + 1
        print(f"Loaded template with {len(template_df)} total frames across {len(template_frames)} sequences")
    
    # Process each test file
    for test_file in tqdm(test_files, desc="Processing sequences"):
        seq_id = os.path.basename(test_file).replace('.h5', '')
        
        # Load data
        data = load_h5_data(test_file)
        events = data['events']
        labels = data['labels']
        
        # Get timestamps: from labels if available, otherwise generate from events
        if labels is not None:
            timestamps = labels['timestamp'].values
        else:
            # For test files without labels: generate evenly-spaced timestamps
            # covering the event time range
            event_times = events['t']
            t_min, t_max = event_times.min(), event_times.max()
            # Use template frame count if available, otherwise default
            num_frames = template_frames.get(seq_id, args.num_frames)
            timestamps = np.linspace(t_min, t_max - window_size_us, num_frames)
            print(f"  Generated {num_frames} timestamps from event range [{t_min:.0f}, {t_max:.0f}] µs")
        
        print(f"\n{seq_id}: {len(timestamps)} poses, {len(events['t'])} events")
        
        # Route to correct inference function based on model type
        seq_len_ckpt = config.get('model', {}).get('sequence_length', 1)
        if is_keypoint:
            keypoints_3d = checkpoint.get('keypoints_3d', None)
            if keypoints_3d is None:
                raise RuntimeError(
                    "KeypointPoseNet checkpoint missing 'keypoints_3d' — "
                    "retrain with train_keypoints.py (it embeds them automatically)."
                )
            predictions = predict_pnp(
                model, events, timestamps,
                event_generator, device, args.test_timestamp_scale,
                keypoints_3d=keypoints_3d,
            )
        elif seq_len_ckpt == 1:
            predictions = predict_frame_by_frame(
                model, events, timestamps,
                event_generator, device, args.test_timestamp_scale,
                is_dann=is_dann,
            )
        else:
            predictions = predict_sequence(
                model, events, timestamps,
                event_generator, seq_len_ckpt, args.stride,
                device, args.test_timestamp_scale
            )
        
        # Create submission CSV
        csv_path = os.path.join(args.output_dir, f"{seq_id}.csv")
        create_submission_csv(predictions, timestamps, csv_path, seq_id)
    
    print(f"\n✓ Inference complete! Submission files saved to: {args.output_dir}")
    print(f"\nTo create submission.zip:")
    print(f"  cd {args.output_dir}")
    print(f"  zip -r ../submission.zip *.csv")


if __name__ == "__main__":
    main()
