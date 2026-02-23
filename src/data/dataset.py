"""
SPADES Dataset class for loading event sequences and generating voxel grids.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import random

# Add parent directory to path to import load_h5
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from load_h5 import load_h5_data
from src.data.event_representations import SignedVoxelGridGenerator, ThreeChannelEventFrame, filter_events_by_count


class SPADESVoxelDataset(Dataset):
    """
    PyTorch Dataset for SPADES event sequences.
    
    Generates sequences of voxel grids with corresponding pose labels.
    Supports both 25% and 100% data subsets with stratified sampling.
    
    Args:
        data_dir: Directory containing .h5 files
        sequence_ids: List of sequence identifiers (e.g., ['RT000', 'RT001', ...])
        sequence_length: Number of frames per sequence (default: 1 for frame-by-frame)
        sequence_stride: Stride between sequences (default: 5, ignored if sequence_length=1)
        voxel_generator: Event representation generator instance
        min_events: Minimum events per frame filter (default: 10000)
        transform: Optional augmentation transform
        synthetic_timestamp_scale: Scale for synthetic data (100µs units) (default: 100.0)
    """
    
    def __init__(
        self,
        data_dir: str,
        sequence_ids: List[str],
        sequence_length: int = 1,
        sequence_stride: int = 5,
        voxel_generator: Optional = None,
        min_events: int = 10000,
        transform: Optional[callable] = None,
        synthetic_timestamp_scale: float = 100.0,
        preprocessed_dir: Optional[str] = None
    ):
        self.data_dir = data_dir
        self.sequence_ids = sequence_ids
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.min_events = min_events
        self.transform = transform
        self.timestamp_scale = synthetic_timestamp_scale
        # Set preprocessed_dir based on split
        if preprocessed_dir is not None:
            self.preprocessed_dir = preprocessed_dir
        else:
            # Default logic: if sequence_ids matches 25pct split, use 25pct dir
            if len(sequence_ids) <= 75:
                self.preprocessed_dir = data_dir.replace('h5', 'preprocessed_voxels_25pct')
            else:
                self.preprocessed_dir = data_dir.replace('h5', 'preprocessed_voxels_100pct')
        # Initialize voxel generator
        if voxel_generator is None:
            self.voxel_generator = SignedVoxelGridGenerator()
        else:
            self.voxel_generator = voxel_generator
        # Build sequence index
        self.sequences = []
        self._build_sequence_index()
        print(f"Dataset initialized with {len(self.sequences)} sequences from {len(sequence_ids)} trajectories")
        
    def _build_sequence_index(self):
        """Build index of all valid sequences across trajectories."""
        for seq_id in self.sequence_ids:
            h5_path = os.path.join(self.data_dir, f"{seq_id}.h5")
            
            if not os.path.exists(h5_path):
                print(f"Warning: {h5_path} not found, skipping...")
                continue
            
            try:
                # Load data to get pose count
                data = load_h5_data(h5_path)
                events = data['events']
                labels = data['labels']
                
                num_poses = len(labels)
                
                # Generate sequence start indices with stride
                for start_idx in range(0, num_poses - self.sequence_length + 1, self.sequence_stride):
                    self.sequences.append({
                        'seq_id': seq_id,
                        'h5_path': h5_path,
                        'start_idx': start_idx,
                        'num_poses': num_poses
                    })
                    
            except Exception as e:
                print(f"Error loading {h5_path}: {e}")
                continue
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence of precomputed voxel grids and poses.
        """
        seq_info = self.sequences[idx]
        preprocessed_path = os.path.join(self.preprocessed_dir, f"{seq_info['seq_id']}_voxels.h5")
        start_idx = seq_info['start_idx']
        end_idx = start_idx + self.sequence_length
        import h5py
        with h5py.File(preprocessed_path, 'r') as f:
            voxels = f['voxels'][start_idx:end_idx]
            poses = f['poses'][start_idx:end_idx]
        if self.transform:
            voxels, poses = self.transform(voxels, poses)
        voxels_tensor = torch.from_numpy(voxels).float()
        poses_tensor = torch.from_numpy(poses).float()
        return voxels_tensor, poses_tensor
        voxels_tensor = torch.from_numpy(voxels).float()
        poses_tensor = torch.from_numpy(poses).float()
        
        return voxels_tensor, poses_tensor


def select_stratified_subset(
    total_sequences: int = 300,
    subset_pct: float = 0.25,
    seed: int = 42
) -> List[str]:
    """
    Select stratified subset of sequences covering range distribution.
    
    Args:
        total_sequences: Total number of sequences (default: 300)
        subset_pct: Percentage to select (default: 0.25 for 25%)
        seed: Random seed for reproducibility
        
    Returns:
        sequence_ids: List of sequence IDs (e.g., ['RT000', 'RT003', ...])
    """
    np.random.seed(seed)
    random.seed(seed)
    
    n_subset = int(total_sequences * subset_pct)
    
    # Stratify by range (based on sequence numbering assumption)
    # RT000-RT099: close range
    # RT100-RT199: mid range  
    # RT200-RT299: far range
    
    close_range = list(range(0, 100))
    mid_range = list(range(100, 200))
    far_range = list(range(200, 300))
    
    # Calculate proportional sampling
    # For 25%: 20 close, 35 mid, 20 far = 75 total
    n_close = int(n_subset * 0.27)  # ~27% close
    n_mid = int(n_subset * 0.47)    # ~47% mid
    n_far = n_subset - n_close - n_mid  # Remainder for far
    
    # Sample from each range
    selected_close = np.random.choice(close_range, n_close, replace=False)
    selected_mid = np.random.choice(mid_range, n_mid, replace=False)
    selected_far = np.random.choice(far_range, n_far, replace=False)
    
    # Combine and sort
    selected_indices = np.concatenate([selected_close, selected_mid, selected_far])
    selected_indices = np.sort(selected_indices)
    
    # Convert to sequence IDs
    sequence_ids = [f"RT{i:03d}" for i in selected_indices]
    
    print(f"Selected {len(sequence_ids)} sequences:")
    print(f"  Close range (0-99): {n_close} sequences")
    print(f"  Mid range (100-199): {n_mid} sequences")
    print(f"  Far range (200-299): {n_far} sequences")
    
    return sequence_ids


def train_val_split(
    sequence_ids: List[str],
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Split sequences into train and validation sets.
    
    Args:
        sequence_ids: List of sequence IDs
        val_ratio: Ratio of validation data (default: 0.1 for 10%)
        seed: Random seed
        
    Returns:
        train_ids: Training sequence IDs
        val_ids: Validation sequence IDs
    """
    np.random.seed(seed)
    random.seed(seed)
    
    n_val = int(len(sequence_ids) * val_ratio)
    n_train = len(sequence_ids) - n_val
    
    # Shuffle and split
    shuffled_ids = sequence_ids.copy()
    random.shuffle(shuffled_ids)
    
    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train:]
    
    print(f"Train/Val split: {n_train} training, {n_val} validation sequences")
    
    return sorted(train_ids), sorted(val_ids)


# Simple augmentation transforms
class RandomIntensityScale:
    """Randomly scale voxel intensities."""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2)):
        self.scale_range = scale_range
    
    def __call__(self, voxels: np.ndarray, poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scale = np.random.uniform(*self.scale_range)
        voxels = voxels * scale
        return voxels, poses


class RandomFrameDropout:
    """Randomly drop frames (set to zero) with given probability."""
    
    def __init__(self, drop_prob: float = 0.1):
        self.drop_prob = drop_prob
    
    def __call__(self, voxels: np.ndarray, poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        seq_len = voxels.shape[0]
        for i in range(seq_len):
            if np.random.rand() < self.drop_prob:
                voxels[i] = 0.0
        return voxels, poses


class SaltPepperNoise:
    """Add salt and pepper noise to simulate hot/dead pixels in event cameras."""
    
    def __init__(self, amount: float = 0.02, prob: float = 0.7):
        self.amount = amount
        self.prob = prob
    
    def __call__(self, voxels: np.ndarray, poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() > self.prob:
            return voxels, poses
        
        # Random amount within range
        amount = np.random.uniform(0.01, self.amount)
        
        for i in range(voxels.shape[0]):  # For each frame in sequence
            frame = voxels[i]  # (C, H, W)
            total_pixels = frame[0].size
            num_noise = int(amount * total_pixels)
            
            # Get frame dimensions
            h, w = frame.shape[1], frame.shape[2]
            
            # Salt (hot pixels - high positive values)
            salt_y = np.random.randint(0, h, num_noise // 2)
            salt_x = np.random.randint(0, w, num_noise // 2)
            max_val = np.abs(frame).max() + 0.5 if frame.size > 0 else 1.0
            for c in range(frame.shape[0]):
                frame[c, salt_y, salt_x] = max_val
            
            # Pepper (dead pixels - zero values)
            pepper_y = np.random.randint(0, h, num_noise // 2)
            pepper_x = np.random.randint(0, w, num_noise // 2)
            for c in range(frame.shape[0]):
                frame[c, pepper_y, pepper_x] = 0.0
            
            voxels[i] = frame
        
        return voxels, poses


class RandomErasing:
    """Randomly erase rectangular regions to simulate dropped event packets."""
    
    def __init__(self, prob: float = 0.5, scale_range: Tuple[float, float] = (0.1, 0.3)):
        self.prob = prob
        self.scale_range = scale_range
    
    def __call__(self, voxels: np.ndarray, poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() > self.prob:
            return voxels, poses
        
        for i in range(voxels.shape[0]):  # For each frame in sequence
            frame = voxels[i]  # (C, H, W)
            h, w = frame.shape[1], frame.shape[2]
            
            # Random erase size
            erase_h = int(np.random.uniform(*self.scale_range) * h)
            erase_w = int(np.random.uniform(*self.scale_range) * w)
            
            # Random position
            y = np.random.randint(0, max(1, h - erase_h))
            x = np.random.randint(0, max(1, w - erase_w))
            
            # Erase (set to zero)
            frame[:, y:y+erase_h, x:x+erase_w] = 0.0
            voxels[i] = frame
        
        return voxels, poses


class ComposeTransforms:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: List[callable]):
        self.transforms = transforms
    
    def __call__(self, voxels: np.ndarray, poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for transform in self.transforms:
            voxels, poses = transform(voxels, poses)
        return voxels, poses
