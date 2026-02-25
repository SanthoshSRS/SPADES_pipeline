#!/usr/bin/env python3
"""
Visualize projected keypoints on two backgrounds:
  1. Preprocessed voxel (what the model actually sees during training)
  2. Dense accumulated event frame (all events in sequence → clear satellite silhouette)

Usage:
    python scripts/visualize_keypoints_on_voxel.py \
        --seq-ids RT000 RT001 RT002 \
        --frame-idx 100 \
        --preprocessed-dir preprocessed_voxels_100pct \
        --keypoint-label-dir keypoint_labels \
        --h5-dir h5
"""

import os
import sys
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COLORS  = ['red', 'red', 'orange', 'orange', 'cyan', 'cyan', 'lime', 'lime']
MARKERS = ['o',   's',   'o',      's',       'o',    's',    'o',    's']
# circles = +X solar panel side, squares = -X solar panel side


def load_dense_frame(h5_path: str, frame_idx: int, timestamp_scale: float = 100.0):
    """Accumulate ALL events (ON polarity) into a 2D density map for a clear silhouette."""
    with h5py.File(h5_path, 'r') as f:
        xs = f['events']['xs'][()]
        ys = f['events']['ys'][()]
        ps = f['events']['ps'][()]
        labels = f['labels']['data'][()]

    # Use 10-second window centered on the requested frame for a clear satellite image
    timestamps = labels['timestamp'].astype(np.float64) * timestamp_scale  # µs
    t_center = timestamps[frame_idx]
    t_half   = 5_000_000   # ±5 seconds

    with h5py.File(h5_path, 'r') as f:
        ts_events = f['events']['ts'][()].astype(np.float64) * timestamp_scale  # → µs

    mask = (ts_events >= t_center - t_half) & (ts_events <= t_center + t_half)

    dense = np.zeros((720, 1280), dtype=np.float32)
    if mask.any():
        x_sel = xs[mask].astype(int)
        y_sel = ys[mask].astype(int)
        p_sel = ps[mask].astype(int)
        # ON events = positive, OFF = negative
        polarity = np.where(p_sel == 1, 1.0, -1.0)
        np.add.at(dense, (y_sel, x_sel), polarity)

    return dense


def visualize_sequence(seq_id, frame_idx, preprocessed_dir, keypoint_label_dir, h5_dir):
    # ── Load voxel (model input) ──────────────────────────────────────────────
    voxel_path = os.path.join(preprocessed_dir, f"{seq_id}_voxels.h5")
    if not os.path.exists(voxel_path):
        print(f"  {voxel_path} not found, skipping")
        return

    with h5py.File(voxel_path, 'r') as f:
        n_frames = f['voxels'].shape[0]
        idx = min(frame_idx, n_frames - 1)
        voxel = f['voxels'][idx]   # (C, H, W)

    voxel_display = np.abs(voxel).max(axis=0)   # (H, W) magnitude

    # ── Load dense accumulated frame ─────────────────────────────────────────
    h5_path = os.path.join(h5_dir, f"{seq_id}.h5")
    if os.path.exists(h5_path):
        dense = load_dense_frame(h5_path, idx)
    else:
        dense = None
        print(f"  Raw H5 not found: {h5_path} — will only show voxel")

    # ── Load keypoints ────────────────────────────────────────────────────────
    kp_path = os.path.join(keypoint_label_dir, f"{seq_id}_keypoints.npz")
    if not os.path.exists(kp_path):
        print(f"  {kp_path} not found, skipping")
        return

    kp_data = np.load(kp_path)
    kp_2d   = kp_data['keypoints_2d'][idx]   # (8, 2) pixels
    vis     = kp_data['visibility'][idx]      # (8,) bool

    # ── Plot ─────────────────────────────────────────────────────────────────
    n_cols = 2 if dense is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(13 * n_cols, 7))
    if n_cols == 1:
        axes = [axes]

    panels = [('Preprocessed voxel (model input)', voxel_display, 'gray')]
    if dense is not None:
        panels.append(('Dense accumulation ±5s (satellite silhouette)', dense, 'RdBu_r'))

    for ax, (title, img, cmap) in zip(axes, panels):
        vmin = np.percentile(img, 1)
        vmax = np.percentile(img, 99)
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper',
                  aspect='auto')

        for j in range(8):
            c, m = COLORS[j], MARKERS[j]
            if vis[j]:
                ax.scatter(kp_2d[j, 0], kp_2d[j, 1], c=c, s=150,
                           marker=m, edgecolors='white', linewidths=1.0, zorder=5)
                ax.annotate(str(j), (kp_2d[j, 0] + 8, kp_2d[j, 1] - 8),
                            fontsize=10, color='white', fontweight='bold')
            else:
                ax.scatter(kp_2d[j, 0], kp_2d[j, 1], c=c, s=80,
                           marker='x', alpha=0.5, zorder=3)

        ax.set_title(f"{seq_id} frame {idx}  |  {title}\n"
                     f"visible={vis.sum()}/8  "
                     f"[circles=+X panel, squares=-X panel]",
                     fontsize=9)
        ax.set_xlim(0, 1280)
        ax.set_ylim(720, 0)

    out = f"keypoint_voxel_{seq_id}_f{idx}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq-ids',           nargs='+', default=['RT000', 'RT001', 'RT002'])
    parser.add_argument('--frame-idx',         type=int, default=100)
    parser.add_argument('--preprocessed-dir',  default='preprocessed_voxels_100pct')
    parser.add_argument('--keypoint-label-dir',default='keypoint_labels')
    parser.add_argument('--h5-dir',            default='h5')
    args = parser.parse_args()

    for seq_id in args.seq_ids:
        print(f"Processing {seq_id}...")
        visualize_sequence(seq_id, args.frame_idx,
                           args.preprocessed_dir,
                           args.keypoint_label_dir,
                           args.h5_dir)
    print("Done.")


if __name__ == '__main__':
    main()
