#!/usr/bin/env python3
"""
Generate 2D keypoint labels for the keypoint PnP pipeline.

Projects 8 3D body-frame keypoints to 2D image coordinates using
ground-truth poses from each training H5 file.

Output: keypoint_labels/{seq_id}_keypoints.npz
  - keypoints_2d: (N, 8, 2) float32 — pixel coordinates (u, v)
  - visibility:   (N, 8)    bool    — True if in front of camera AND within image bounds

Usage:
    # Generate labels only
    python scripts/generate_keypoint_labels.py --h5-dir h5 --output-dir keypoint_labels

    # Generate + visualize first 3 sequences (saves keypoint_validation_RTXXX.png)
    python scripts/generate_keypoint_labels.py --h5-dir h5 --output-dir keypoint_labels --visualize

IMPORTANT: If the saved validation images show keypoints far from the satellite edges,
adjust KEYPOINTS_3D half-extents below before training.
"""

import os
import sys
import glob
import argparse
import numpy as np
import h5py
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Proba-2 body-frame keypoints (meters) ────────────────────────────────────
# Based on SPADES paper: physical mockup is 0.64 × 0.24 × 0.416m at 1:2.5 scale.
# Full-scale simulation model: ~1.6 × 0.6 × 1.04m.
# Axes:  X (±0.80): solar panel span, Y (±0.30): body height, Z (±0.52): depth
# GT pose: P_cam = R @ P_body + t  (R from [Qx,Qy,Qz,Qw], t = [Tx,Ty,Tz])
# Validate projections with --visualize before training!
#
# 14 keypoints = 8 bounding-box corners + 6 face centres
#   Corners: PnP-optimal (span full 3D volume, geometrically non-degenerate)
#   Face centres: at least 1 face always visible regardless of viewing angle,
#                 improves RANSAC inlier count and rotation coverage
KEYPOINTS_3D = np.array([
    # ── 8 bounding-box corners ──────────────────────────────────────────────
    [+0.80, +0.30, +0.52],   #  0: +X+Y+Z  (solar panel tip, top, front)
    [+0.80, +0.30, -0.52],   #  1: +X+Y-Z  (solar panel tip, top, back)
    [+0.80, -0.30, +0.52],   #  2: +X-Y+Z  (solar panel tip, bottom, front)
    [+0.80, -0.30, -0.52],   #  3: +X-Y-Z  (solar panel tip, bottom, back)
    [-0.80, +0.30, +0.52],   #  4: -X+Y+Z  (other solar panel tip, top, front)
    [-0.80, +0.30, -0.52],   #  5: -X+Y-Z
    [-0.80, -0.30, +0.52],   #  6: -X-Y+Z
    [-0.80, -0.30, -0.52],   #  7: -X-Y-Z  (other solar panel tip, bottom, back)
    # ── 6 face centres ──────────────────────────────────────────────────────
    [+0.80,   0.0,   0.0],   #  8: +X face centre  (solar panel face)
    [-0.80,   0.0,   0.0],   #  9: -X face centre  (other solar panel face)
    [  0.0, +0.30,   0.0],   # 10: +Y face centre  (top face)
    [  0.0, -0.30,   0.0],   # 11: -Y face centre  (bottom face)
    [  0.0,   0.0, +0.52],   # 12: +Z face centre  (front face)
    [  0.0,   0.0, -0.52],   # 13: -Z face centre  (back face)
], dtype=np.float64)  # (14, 3)

NUM_KEYPOINTS = len(KEYPOINTS_3D)   # 14

# Camera intrinsics from SPADES/camera.json
FX = FY = 1258.6057531097028
CX, CY   = 640.0, 360.0
IMG_W, IMG_H = 1280, 720


def project_keypoints(R: np.ndarray, t: np.ndarray, keypoints_3d: np.ndarray):
    """
    Project 3D body-frame keypoints to 2D image coordinates.

    Args:
        R:             (3, 3) rotation matrix (body → camera)
        t:             (3,)   translation vector (body origin in camera frame)
        keypoints_3d:  (N, 3) 3D keypoints in satellite body frame

    Returns:
        uv:      (N, 2) float32 — pixel coordinates (u, v)
        visible: (N,)   bool   — True if in front of camera AND within image
    """
    # Transform to camera frame
    P_cam = (R @ keypoints_3d.T).T + t          # (N, 3)

    z = P_cam[:, 2]
    in_front = z > 0.1                           # must be in front of camera

    z_safe = np.where(in_front, z, 1.0)
    u = FX * P_cam[:, 0] / z_safe + CX
    v = FY * P_cam[:, 1] / z_safe + CY

    in_bounds = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    visible = in_front & in_bounds

    return np.stack([u, v], axis=1).astype(np.float32), visible


def process_sequence(h5_path: str, output_dir: str, do_visualize: bool = False) -> float:
    """
    Process one training H5 file and save keypoint labels.

    Returns average keypoint visibility percentage (0-100).
    """
    seq_id = os.path.basename(h5_path).replace('.h5', '')
    out_path = os.path.join(output_dir, f"{seq_id}_keypoints.npz")

    with h5py.File(h5_path, 'r') as f:
        if 'labels' not in f:
            print(f"  Skipping {seq_id}: no labels group")
            return 0.0

        labels = f['labels']['data'][()]

    # Extract pose fields
    Tx = labels['Tx'].astype(np.float64)
    Ty = labels['Ty'].astype(np.float64)
    Tz = labels['Tz'].astype(np.float64)
    Qx = labels['Qx'].astype(np.float64)
    Qy = labels['Qy'].astype(np.float64)
    Qz = labels['Qz'].astype(np.float64)
    Qw = labels['Qw'].astype(np.float64)

    N = len(Tx)
    keypoints_2d = np.zeros((N, NUM_KEYPOINTS, 2), dtype=np.float32)
    visibility   = np.zeros((N, NUM_KEYPOINTS),    dtype=bool)

    for i in range(N):
        # scipy uses [x, y, z, w] quaternion convention
        R = Rotation.from_quat([Qx[i], Qy[i], Qz[i], Qw[i]]).as_matrix()
        t = np.array([Tx[i], Ty[i], Tz[i]])
        uv, vis = project_keypoints(R, t, KEYPOINTS_3D)
        keypoints_2d[i] = uv
        visibility[i]   = vis

    np.savez_compressed(out_path, keypoints_2d=keypoints_2d, visibility=visibility)

    vis_pct = float(visibility.mean() * 100)

    if do_visualize:
        _visualize(seq_id, h5_path, keypoints_2d, visibility, labels)

    return vis_pct


def _visualize(seq_id: str, h5_path: str, keypoints_2d: np.ndarray,
               visibility: np.ndarray, labels, frame_idx: int = 100):
    """Save a PNG showing projected keypoints overlaid on an event frame."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from src.data.event_representations import ThreeChannelEventFrame

        gen = ThreeChannelEventFrame(height=IMG_H, width=IMG_W, window_size_us=100000)

        with h5py.File(h5_path, 'r') as f:
            xs = f['events']['xs'][()]
            ys = f['events']['ys'][()]
            ts = f['events']['ts'][()]
            ps = f['events']['ps'][()]

        events = {'x': xs, 'y': ys, 't': ts, 'p': ps}

        # timestamps stored in 100µs units in training H5
        t_start = float(labels['timestamp'][frame_idx]) * 100.0
        frame = gen.generate(events, t_start, t_start + 100000)

        display = np.abs(frame).max(axis=0)  # magnitude across channels

        fig, ax = plt.subplots(1, 1, figsize=(13, 7))
        ax.imshow(display, cmap='gray', origin='upper')

        # corners (0-7): two shades per axis pair; face centres (8-13): white
        K_total = len(keypoints_2d[frame_idx])
        corner_colors  = ['red', 'red', 'orange', 'orange',
                          'cyan', 'cyan', 'lime', 'lime']
        face_colors    = ['magenta', 'magenta', 'yellow', 'yellow',
                          'white', 'white']
        all_colors  = corner_colors + face_colors[:max(0, K_total - 8)]
        corner_markers = ['o', 's', 'o', 's', 'o', 's', 'o', 's']
        face_markers   = ['D'] * 6
        all_markers = corner_markers + face_markers[:max(0, K_total - 8)]

        kp  = keypoints_2d[frame_idx]
        vis = visibility[frame_idx]

        for j in range(K_total):
            c = all_colors[j] if j < len(all_colors) else 'white'
            m = all_markers[j] if j < len(all_markers) else 'D'
            if vis[j]:
                ax.scatter(kp[j, 0], kp[j, 1], c=c, s=120, marker=m,
                           edgecolors='white', linewidths=0.5, zorder=5)
                ax.annotate(str(j), (kp[j, 0] + 5, kp[j, 1] - 5),
                            fontsize=9, color='white', fontweight='bold')
            else:
                ax.scatter(kp[j, 0], kp[j, 1], c=c, s=80, marker='x',
                           alpha=0.4, zorder=3)

        ax.set_title(
            f"{seq_id} frame {frame_idx}  |  "
            f"solid=visible ({vis.sum()}/{K_total}), x=out-of-bounds\n"
            f"Keypoints should appear near satellite edges. "
            f"If not, adjust KEYPOINTS_3D in this script.",
            fontsize=10
        )
        ax.set_xlim(0, IMG_W)
        ax.set_ylim(IMG_H, 0)

        out_png = f"keypoint_validation_{seq_id}.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=100)
        plt.close()
        print(f"    Saved {out_png}")

    except Exception as e:
        print(f"    Visualization failed for {seq_id}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 2D keypoint labels from ground-truth poses")
    parser.add_argument('--h5-dir',      default='h5',
                        help='Directory containing training H5 files (default: h5)')
    parser.add_argument('--output-dir',  default='keypoint_labels',
                        help='Output directory for .npz label files')
    parser.add_argument('--visualize',   action='store_true',
                        help='Save validation PNGs for first N sequences')
    parser.add_argument('--num-visualize', type=int, default=3,
                        help='Number of sequences to visualize (default: 3)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    h5_files = sorted(glob.glob(os.path.join(args.h5_dir, 'RT*.h5')))
    if not h5_files:
        print(f"ERROR: No RT*.h5 files found in {args.h5_dir}")
        sys.exit(1)

    print(f"Found {len(h5_files)} sequences in '{args.h5_dir}'")
    print(f"Output directory: '{args.output_dir}'")
    print(f"\nKeypoints 3D (body frame, meters):")
    for i, kp in enumerate(KEYPOINTS_3D):
        print(f"  [{i}] ({kp[0]:+.2f}, {kp[1]:+.2f}, {kp[2]:+.2f})")

    vis_pcts = []
    for i, h5_path in enumerate(tqdm(h5_files, desc="Generating labels")):
        do_viz = args.visualize and i < args.num_visualize
        vis_pct = process_sequence(h5_path, args.output_dir, do_visualize=do_viz)
        vis_pcts.append(vis_pct)

    print(f"\n{'─'*50}")
    print(f"Done. Processed {len(h5_files)} sequences.")
    print(f"Average keypoint visibility: {np.mean(vis_pcts):.1f}%")
    print(f"  ≥ 70%  → dimensions look correct, proceed to training")
    print(f"  40-70% → marginal, check validation images")
    print(f"  < 40%  → dimensions likely wrong, adjust KEYPOINTS_3D")

    if args.visualize:
        print(f"\nValidation images saved as keypoint_validation_RT*.png")
        print("Check that keypoints appear near the satellite's edges/corners.")


if __name__ == '__main__':
    main()
