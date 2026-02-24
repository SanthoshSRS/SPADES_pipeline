"""
Preprocess events into voxel grids for faster training.
Batch processes all sequences and saves to disk.
"""

import os
import sys
import argparse
import h5py
import numpy as np
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from load_h5 import load_h5_data
from src.data.event_representations import SignedVoxelGridGenerator, ThreeChannelEventFrame, filter_events_by_count


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess events into voxel grids')
    
    parser.add_argument('--input-dir', type=str, default='h5',
                        help='Directory containing .h5 files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for preprocessed voxels')
    parser.add_argument('--sequence-ids-file', type=str, default=None,
                        help='File containing sequence IDs to process (one per line)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--window-size-us', type=float, default=100000.0,
                        help='Voxel window size in microseconds')
    parser.add_argument('--num-bins', type=int, default=5,
                        help='Number of temporal bins')
    parser.add_argument('--min-events', type=int, default=10000,
                        help='Minimum events per frame (filters frames with too few events)')
    parser.add_argument('--timestamp-scale', type=float, default=100.0,
                        help='Timestamp scale (100 for synthetic 100µs data)')
    
    return parser.parse_args()


def process_sequence(args_tuple):
    """
    Process a single sequence and save voxel grids.
    
    Args:
        args_tuple: Tuple of (h5_path, output_dir, config_dict)
    """
    h5_path, output_dir, config = args_tuple
    
    seq_id = os.path.basename(h5_path).replace('.h5', '')
    output_path = os.path.join(output_dir, f"{seq_id}_voxels.h5")
    
    # Skip if already processed
    if os.path.exists(output_path):
        return f"{seq_id}: Already processed"
    
    try:
        # Load data
        data = load_h5_data(h5_path)
        events = data['events']
        labels = data['labels']
        
        # Create voxel generator based on num_bins
        if config['num_bins'] == 3:
            # 3-channel exponential decay representation (SPADES paper method)
            voxel_generator = ThreeChannelEventFrame(
                width=1280,
                height=720,
                window_size_us=config['window_size_us'],
                tau_us=30000.0  # 30ms decay constant
            )
        else:
            # Signed voxel grid representation (5-channel or custom)
            voxel_generator = SignedVoxelGridGenerator(
                height=720,
                width=1280,
                num_bins=config['num_bins'],
                window_size_us=config['window_size_us']
            )
        
        # Get timestamps
        if labels is not None:
            timestamps = labels['timestamp'].values * config['timestamp_scale']
        else:
            # Real test data: no pose labels — step through event stream in fixed windows
            t_events = events['t'].astype(float) * config['timestamp_scale']
            t_min = float(t_events[0])
            t_max = float(t_events[-1])
            timestamps = np.arange(t_min, t_max - config['window_size_us'], config['window_size_us'])
        num_poses = len(timestamps)
        
        # Generate voxel grids
        voxel_list = []
        valid_indices = []
        
        for i, ts in enumerate(timestamps):
            t_start = ts
            t_end = t_start + config['window_size_us']
            
            # Check event count filter
            if not filter_events_by_count(events, t_start, t_end, config['min_events'],
                                          timestamp_scale=config['timestamp_scale']):
                # Skip frames with insufficient events
                continue

            # Generate voxel grid
            voxel = voxel_generator.generate(events, t_start, t_end,
                                             timestamp_scale=config['timestamp_scale'])
            voxel_list.append(voxel)
            valid_indices.append(i)
        
        if len(voxel_list) == 0:
            return f"{seq_id}: No valid frames (all filtered)"
        
        # Stack voxels
        voxels = np.stack(voxel_list, axis=0)  # (num_valid, num_bins, H, W)

        pose_timestamps = timestamps[valid_indices]

        # Save to HDF5 with compression
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('voxels', data=voxels, compression='gzip', compression_opts=4)
            f.create_dataset('timestamps', data=pose_timestamps, compression='gzip', compression_opts=4)
            if labels is not None:
                pose_data = labels.iloc[valid_indices][['Tx', 'Ty', 'Tz', 'Qw', 'Qx', 'Qy', 'Qz']].values
                f.create_dataset('poses', data=pose_data, compression='gzip', compression_opts=4)
                f.create_dataset('valid_indices', data=np.array(valid_indices), compression='gzip', compression_opts=4)
            
            # Store metadata
            f.attrs['seq_id'] = seq_id
            f.attrs['num_frames'] = len(voxel_list)
            f.attrs['num_total_poses'] = num_poses
            f.attrs['num_bins'] = config['num_bins']
            f.attrs['window_size_us'] = config['window_size_us']
            f.attrs['min_events'] = config['min_events']
            f.attrs['representation'] = '3channel_decay' if config['num_bins'] == 3 else 'signed_voxel'
            if config['num_bins'] == 3:
                f.attrs['tau_us'] = 30000.0
        
        return f"{seq_id}: {len(voxel_list)}/{num_poses} frames ({len(voxel_list)/num_poses*100:.1f}%)"
        
    except Exception as e:
        return f"{seq_id}: ERROR - {str(e)}"


def main():
    args = parse_args()
    
    print(f"\n{'='*70}")
    print(f"SPADES Event Preprocessing - Voxel Grid Generation")
    print(f"{'='*70}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get sequence IDs
    if args.sequence_ids_file:
        # Load from file
        with open(args.sequence_ids_file, 'r') as f:
            sequence_ids = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(sequence_ids)} sequence IDs from {args.sequence_ids_file}")
    else:
        # Process all .h5 files in input directory
        h5_files = glob(os.path.join(args.input_dir, "RT*.h5"))
        sequence_ids = [os.path.basename(f).replace('.h5', '') for f in h5_files]
        print(f"Found {len(sequence_ids)} sequences in {args.input_dir}")
    
    # Build file paths
    h5_paths = [os.path.join(args.input_dir, f"{seq_id}.h5") for seq_id in sequence_ids]
    
    # Filter existing files
    h5_paths = [p for p in h5_paths if os.path.exists(p)]
    print(f"Processing {len(h5_paths)} sequences\n")
    
    if len(h5_paths) == 0:
        print("No sequences to process!")
        return
    
    # Configuration
    config = {
        'num_bins': args.num_bins,
        'window_size_us': args.window_size_us,
        'min_events': args.min_events,
        'timestamp_scale': args.timestamp_scale
    }
    
    print(f"Configuration:")
    print(f"  Temporal bins: {config['num_bins']}")
    print(f"  Window size: {config['window_size_us']/1000:.1f} ms")
    print(f"  Min events/frame: {config['min_events']}")
    print(f"  Timestamp scale: {config['timestamp_scale']}x")
    print(f"  Workers: {args.num_workers}")
    print(f"  Output: {args.output_dir}\n")
    
    # Prepare arguments for multiprocessing
    process_args = [(path, args.output_dir, config) for path in h5_paths]
    
    # Process in parallel
    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_sequence, process_args),
            total=len(process_args),
            desc="Processing sequences"
        ))
    
    # Print results
    print(f"\n{'='*70}")
    print("Processing Results:")
    print(f"{'='*70}\n")
    
    success_count = 0
    error_count = 0
    
    for result in results:
        print(f"  {result}")
        if "ERROR" in result:
            error_count += 1
        else:
            success_count += 1
    
    print(f"\n{'='*70}")
    print(f"Summary: {success_count} successful, {error_count} errors")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
