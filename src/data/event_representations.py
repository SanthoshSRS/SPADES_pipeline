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


class ThreeChannelEventFrame:
    """
    3-Channel event representation with exponential decay (SPADES paper method).
    
    Splits the time window into 3 sub-windows and applies exponential temporal decay
    within each sub-window. This creates a 3-channel representation compatible with
    RGB pretrained models (ResNet, EfficientNet, etc.).
    
    Paper: "SPADES: A Realistic Spacecraft Pose Estimation Dataset using Event Sensing"
    Method: 3C (3-Channel) representation with exponential decay
    
    Args:
        width: Image width (default: 1280)
        height: Image height (default: 720)
        window_size_us: Time window in microseconds (default: 100000 = 100ms)
        tau_us: Exponential decay time constant in microseconds (default: 30000 = 30ms)
    """
    
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        window_size_us: float = 100000.0,
        tau_us: float = 30000.0
    ):
        self.width = width
        self.height = height
        self.window_size_us = window_size_us
        self.tau_us = tau_us
        self.num_bins = 3  # For compatibility with dataset code
        
        # Sub-window boundaries (split into 3 equal parts)
        self.sub_window_size = window_size_us / 3.0
        self.sub_window_edges = [
            0.0,
            self.sub_window_size,
            2 * self.sub_window_size,
            window_size_us
        ]
    
    def generate(
        self,
        events: Dict[str, np.ndarray],
        t_start: float,
        t_end: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate 3-channel event frame with exponential decay.
        
        Args:
            events: Dictionary with keys 'x', 'y', 't' (timestamps in µs), 'p' (polarity)
            t_start: Start timestamp in microseconds
            t_end: End timestamp (if None, uses t_start + window_size_us)
            
        Returns:
            event_frame: Array of shape (3, height, width) with exponentially decayed events
        """
        if t_end is None:
            t_end = t_start + self.window_size_us
        
        # Filter events in time window
        t = events['t']
        mask = (t >= t_start) & (t < t_end)
        
        if not np.any(mask):
            # No events in window, return zeros
            return np.zeros((3, self.height, self.width), dtype=np.float32)
        
        x = events['x'][mask]
        y = events['y'][mask]
        t_filtered = t[mask]
        p = events['p'][mask]
        
        # Normalize timestamps to [0, window_size_us]
        t_normalized = t_filtered - t_start
        
        # Initialize 3-channel frame
        event_frame = np.zeros((3, self.height, self.width), dtype=np.float32)
        
        # Process each sub-window
        for channel_idx in range(3):
            sub_start = self.sub_window_edges[channel_idx]
            sub_end = self.sub_window_edges[channel_idx + 1]
            
            # Filter events in this sub-window
            sub_mask = (t_normalized >= sub_start) & (t_normalized < sub_end)
            
            if not np.any(sub_mask):
                continue
            
            x_sub = x[sub_mask]
            y_sub = y[sub_mask]
            t_sub = t_normalized[sub_mask]
            p_sub = p[sub_mask]
            
            # Apply exponential decay: exp(-(t_window_end - t_event) / tau)
            # Decay from end of sub-window backward
            time_to_end = sub_end - t_sub
            decay_weights = np.exp(-time_to_end / self.tau_us)
            
            # Signed accumulation: ON(+1), OFF(-1)
            signed_polarity = 2.0 * p_sub - 1.0  # Convert 0/1 to -1/+1
            weighted_events = signed_polarity * decay_weights
            
            # Accumulate into frame (clip coordinates to valid range)
            x_clipped = np.clip(x_sub, 0, self.width - 1)
            y_clipped = np.clip(y_sub, 0, self.height - 1)
            
            for i in range(len(x_clipped)):
                event_frame[channel_idx, y_clipped[i], x_clipped[i]] += weighted_events[i]
        
        # Apply log normalization per channel: sign(x) * log(1 + |x|)
        event_frame = np.sign(event_frame) * np.log1p(np.abs(event_frame))
        
        return event_frame
    
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
