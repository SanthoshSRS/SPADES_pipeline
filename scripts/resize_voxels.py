#!/usr/bin/env python3
"""
Resize preprocessed voxel h5 files from 720×1280 → 256×448 (or any target size).

One-time preprocessing that makes training ~5× faster by reducing NFS read
size from 11MB/frame → 1.4MB/frame. Workers then spend time waiting for I/O
rather than CPU resize, and the GPU stays much busier.

Each output h5 file has the same layout as the input (voxels dataset, shape
(N, C, out_h, out_w)), stored with per-frame gzip chunks for fast random access.

Usage:
    HDF5_USE_FILE_LOCKING=FALSE python scripts/resize_voxels.py \\
        --input-dir preprocessed_voxels_100pct \\
        --output-dir preprocessed_voxels_256x448 \\
        --num-workers 20

Disk usage: ~25 GB for 300 sequences at 256×448 (vs ~200 GB at 720×1280).
Runtime:    ~20–30 min with 20 workers.
"""

import os
import argparse
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm


def resize_one(task: tuple):
    """Worker function: resize one sequence h5 file and write to output."""
    in_path, out_path, out_h, out_w = task

    try:
        with h5py.File(in_path, 'r', locking=False) as f_in:
            N, C, H, W = f_in['voxels'].shape

            with h5py.File(out_path, 'w', locking=False) as f_out:
                ds = f_out.create_dataset(
                    'voxels',
                    shape=(N, C, out_h, out_w),
                    dtype=np.float32,
                    compression='gzip',
                    compression_opts=4,
                    chunks=(1, C, out_h, out_w),   # one chunk per frame — fast random access
                )

                # Process 64 frames at a time: ~850 MB RAM per worker
                CHUNK = 64
                for i in range(0, N, CHUNK):
                    raw = f_in['voxels'][i:i + CHUNK]            # (k, C, H, W) float32
                    t   = torch.from_numpy(raw)
                    t   = F.interpolate(t, size=(out_h, out_w),
                                        mode='bilinear', align_corners=False)
                    ds[i:i + len(t)] = t.numpy()

        return Path(in_path).name, True, N

    except Exception as e:
        # Remove partial output on failure so it gets retried next run
        if os.path.exists(out_path):
            os.remove(out_path)
        return Path(in_path).name, False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description='Resize voxel h5 files to a smaller spatial resolution')
    parser.add_argument('--input-dir',   default='preprocessed_voxels_100pct',
                        help='Directory containing *_voxels.h5 at full resolution')
    parser.add_argument('--output-dir',  default='preprocessed_voxels_256x448',
                        help='Directory to write resized h5 files')
    parser.add_argument('--num-workers', type=int, default=20,
                        help='Parallel worker processes (default: 20)')
    parser.add_argument('--out-h',       type=int, default=256,
                        help='Output height (default: 256)')
    parser.add_argument('--out-w',       type=int, default=448,
                        help='Output width  (default: 448)')
    args = parser.parse_args()

    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    os.makedirs(args.output_dir, exist_ok=True)

    input_files = sorted(Path(args.input_dir).glob('*_voxels.h5'))
    if not input_files:
        print(f"No *_voxels.h5 files found in {args.input_dir}")
        return

    # Build task list — skip already-completed files
    tasks, skipped = [], 0
    for in_p in input_files:
        out_p = Path(args.output_dir) / in_p.name
        if out_p.exists():
            skipped += 1
        else:
            tasks.append((str(in_p), str(out_p), args.out_h, args.out_w))

    print(f"Input:       {args.input_dir}  ({len(input_files)} sequences found)")
    print(f"Output:      {args.output_dir}  ({args.out_h}×{args.out_w})")
    print(f"To process:  {len(tasks)}  |  Already done: {skipped}")
    print(f"Workers:     {args.num_workers}")
    print()

    if not tasks:
        print("All files already processed — nothing to do.")
        return

    done, failed, total_frames = 0, 0, 0
    with Pool(processes=args.num_workers) as pool:
        for name, ok, info in tqdm(
            pool.imap_unordered(resize_one, tasks),
            total=len(tasks),
            desc="Resizing",
        ):
            if ok:
                done += 1
                total_frames += info
            else:
                failed += 1
                print(f"\n  FAILED {name}: {info}")

    print(f"\n{'─'*50}")
    print(f"Done: {done} sequences, {total_frames:,} frames")
    if failed:
        print(f"Failed: {failed} sequences — re-run to retry")
    print()
    print("Train with the resized data:")
    print(f"  HDF5_USE_FILE_LOCKING=FALSE python train_keypoints.py \\")
    print(f"      --preprocessed-dir {args.output_dir} \\")
    print(f"      --keypoint-label-dir keypoint_labels \\")
    print(f"      --amp --num-workers 16 --batch-size 256")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)
    main()
