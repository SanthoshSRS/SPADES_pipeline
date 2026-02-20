"""
Event representation modules for converting event streams to dense voxel grids.
Implements signed 5-channel voxel grid representation optimized for LSTM processing.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


class SignedVoxelGridGenerator:
    """
    Generate signed 5-channel voxel grid representation from event stream.
    
    Key features:
    - 5 temporal bins (20ms per bin for 100ms window)
    - Signed accumulation: ON events = +1, OFF events = -1
    - Exploits 38/62% ON/OFF polarity bias from SPADES dataset
    - Bilinear temporal interpolation for smooth event placement
    - Log normalization for stability
    
    Args:
        height (int): Sensor height in pixels (default: 720)
        width (int): Sensor width in pixels (default: 1280)
        num_bins (int): Number of temporal bins (default: 5)
        window_size_us (float): Time window in microseconds (default: 100000 for 100ms)
    """
    
    def __init__(
        self,
        height: int = 720,
        width: int = 1280,
        num_bins: int = 5,
        window_size_us: float = 100000.0
    ):
        self.height = height
        self.width = width
        self.num_bins = num_bins
        self.window_size_us = window_size_us
        self.bin_size_us = window_size_us / num_bins
        
    def generate(
        self,
        events: Dict[str, np.ndarray],
        t_start: float,
        t_end: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate signed voxel grid from event stream.
        
        Args:
            events: Dictionary with keys 'x', 'y', 'p', 't' (numpy arrays)
                   x, y: pixel coordinates (int)
                   p: polarity (0=OFF, 1=ON)
                   t: timestamp in microseconds (float)
            t_start: Start timestamp in microseconds
            t_end: End timestamp in microseconds (if None, uses t_start + window_size_us)
            
        Returns:
            voxel_grid: Signed voxel grid of shape (num_bins, height, width)
                       dtype: float32
        """
        if t_end is None:
            t_end = t_start + self.window_size_us
            
        # Initialize voxel grid
        voxel_grid = np.zeros((self.num_bins, self.height, self.width), dtype=np.float32)
        
        # Filter events in time window
        mask = (events['t'] >= t_start) & (events['t'] < t_end)
        if not np.any(mask):
            return voxel_grid
        
        x = events['x'][mask]
        y = events['y'][mask]
        p = events['p'][mask]
        t = events['t'][mask]
        
        # Validate coordinates
        valid_coords = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        x = x[valid_coords]
        y = y[valid_coords]
        p = p[valid_coords]
        t = t[valid_coords]
        
        if len(x) == 0:
            return voxel_grid
        
        # Normalize time to [0, num_bins)
        t_normalized = (t - t_start) / self.window_size_us * self.num_bins
        t_normalized = np.clip(t_normalized, 0, self.num_bins - 1e-6)
        
        # Bilinear temporal interpolation
        bin_lower = np.floor(t_normalized).astype(np.int32)
        bin_upper = np.ceil(t_normalized).astype(np.int32)
        weight_upper = t_normalized - bin_lower
        weight_lower = 1.0 - weight_upper
        
        # Convert polarity to signed values: OFF(0) -> -1, ON(1) -> +1
        signed_polarity = 2.0 * p - 1.0
        
        # Accumulate events with bilinear interpolation
        for i in range(len(x)):
            xi, yi = int(x[i]), int(y[i])
            pol = signed_polarity[i]
            
            # Lower temporal bin
            if weight_lower[i] > 0:
                voxel_grid[bin_lower[i], yi, xi] += pol * weight_lower[i]
            
            # Upper temporal bin (if different)
            if weight_upper[i] > 0 and bin_upper[i] != bin_lower[i]:
                voxel_grid[bin_upper[i], yi, xi] += pol * weight_upper[i]
        
        # Apply log normalization: sign(x) * log(1 + |x|)
        voxel_grid = np.sign(voxel_grid) * np.log1p(np.abs(voxel_grid))
        
        return voxel_grid
    
    def generate_batch(
        self,
        events: Dict[str, np.ndarray],
        timestamps: List[float]
    ) -> np.ndarray:
        """
        Generate batch of voxel grids for multiple timestamps.
        
        Args:
            events: Event dictionary
            timestamps: List of start timestamps
            
        Returns:
            voxel_batch: Array of shape (len(timestamps), num_bins, height, width)
        """
        voxel_batch = np.stack([
            self.generate(events, t) for t in timestamps
        ], axis=0)
        
        return voxel_batch
    
    def __call__(self, *args, **kwargs):
        """Allow calling instance as function."""
        return self.generate(*args, **kwargs)


class StandardVoxelGridGenerator:
    """
    Standard 10-channel voxel grid (5 bins × 2 polarities separated).
    Provided as alternative if signed representation underperforms.
    """
    
    def __init__(
        self,
        height: int = 720,
        width: int = 1280,
        num_bins: int = 5,
        window_size_us: float = 100000.0
    ):
        self.height = height
        self.width = width
        self.num_bins = num_bins
        self.window_size_us = window_size_us
        self.num_channels = num_bins * 2  # 2 channels per temporal bin (ON/OFF)
        
    def generate(
        self,
        events: Dict[str, np.ndarray],
        t_start: float,
        t_end: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate 10-channel voxel grid (5 temporal bins, 2 polarity channels).
        
        Returns:
            voxel_grid: Shape (10, height, width) - [bin0_ON, bin0_OFF, bin1_ON, bin1_OFF, ...]
        """
        if t_end is None:
            t_end = t_start + self.window_size_us
            
        # Initialize: shape (num_bins, 2, height, width)
        voxel_grid = np.zeros((self.num_bins, 2, self.height, self.width), dtype=np.float32)
        
        # Filter events
        mask = (events['t'] >= t_start) & (events['t'] < t_end)
        if not np.any(mask):
            return voxel_grid.reshape(self.num_channels, self.height, self.width)
        
        x = events['x'][mask]
        y = events['y'][mask]
        p = events['p'][mask].astype(np.int32)
        t = events['t'][mask]
        
        # Validate coordinates
        valid_coords = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        x = x[valid_coords]
        y = y[valid_coords]
        p = p[valid_coords]
        t = t[valid_coords]
        
        if len(x) == 0:
            return voxel_grid.reshape(self.num_channels, self.height, self.width)
        
        # Normalize time
        t_normalized = (t - t_start) / self.window_size_us * self.num_bins
        t_normalized = np.clip(t_normalized, 0, self.num_bins - 1e-6)
        
        # Bilinear temporal interpolation
        bin_lower = np.floor(t_normalized).astype(np.int32)
        bin_upper = np.ceil(t_normalized).astype(np.int32)
        weight_upper = t_normalized - bin_lower
        weight_lower = 1.0 - weight_upper
        
        # Accumulate events by polarity
        for i in range(len(x)):
            xi, yi, pi = int(x[i]), int(y[i]), p[i]
            
            # Lower bin
            if weight_lower[i] > 0:
                voxel_grid[bin_lower[i], pi, yi, xi] += weight_lower[i]
            
            # Upper bin
            if weight_upper[i] > 0 and bin_upper[i] != bin_lower[i]:
                voxel_grid[bin_upper[i], pi, yi, xi] += weight_upper[i]
        
        # Reshape to (num_bins*2, height, width)
        voxel_grid = voxel_grid.reshape(self.num_channels, self.height, self.width)
        
        # Apply log normalization
        voxel_grid = np.log1p(voxel_grid)
        
        return voxel_grid


def filter_events_by_count(
    events: Dict[str, np.ndarray],
    t_start: float,
    t_end: float,
    min_events: int = 10000
) -> bool:
    """
    Filter events by minimum count threshold.
    
    Args:
        events: Event dictionary
        t_start: Start timestamp
        t_end: End timestamp
        min_events: Minimum number of events required
        
    Returns:
        valid: True if event count >= min_events
    """
    mask = (events['t'] >= t_start) & (events['t'] < t_end)
    event_count = np.sum(mask)
    return event_count >= min_events
