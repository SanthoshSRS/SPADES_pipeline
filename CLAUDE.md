# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SPADES** (SPAcecraft Pose Estimation Dataset using Event Sensing) — a deep learning pipeline for 6-DOF spacecraft pose estimation from event camera data. Goal: beat the paper baseline (79° rotation error) with targets of Translation < 5%, Rotation < 10°.

**Dataset:** 300 synthetic training sequences (~179K poses) + 31 real test sequences. Events from a 1280×720 event camera at 10Hz pose sampling. Raw H5 files in `h5/`, preprocessed voxel grids in `preprocessed_voxels_25pct/` and `preprocessed_voxels_100pct/`.

## Commands

```bash
# System check (verifies imports, CUDA, etc.)
python test_pipeline.py

# Install dependencies
pip install -r requirements.txt

# Phase 1: 25% subset validation (~20 epochs)
python train.py --config config_25pct.yaml --data-subset 25pct --device cuda

# Phase 2: 100% production training (~100 epochs)
python train.py --config config_100pct.yaml --data-subset 100pct --device cuda

# Resume from checkpoint
python train.py --config config_100pct.yaml --data-subset 100pct --resume checkpoints/100pct_epoch_50.pth

# Mixed precision training
python train.py --config config_25pct.yaml --data-subset 25pct --amp

# Inference on test set
python inference.py --checkpoint checkpoints/100pct_best.pth --test-dir h5 --output-dir submission --test-timestamp-scale 1.0

# Preprocess events to voxel grids (large disk usage: 50GB for 25%, 200GB for 100%)
python scripts/preprocess_events.py --input-dir h5 --output-dir preprocessed_voxels_25pct --num-workers 4

# Monitor training
tensorboard --logdir logs

# Validate submission format
python scripts/validate_submission.py --submission-dir submission --verbose
```

## Architecture

### Model: DirectPoseCNN (`src/models/cnn_lstm_voxel.py`)
- **Backbone:** ResNet-18 or ResNet-50 with modified `conv1` accepting variable input channels (3 or 5)
- **Output:** Dual regression heads — Translation head `FC(512→128→3)` and Rotation head `FC(512→128→4)` with quaternion normalization
- `sequence_length=1` means frame-by-frame (no LSTM); batch shape `(B, 1, C, H, W)` is squeezed to `(B, C, H, W)` before the CNN forward pass

### Event Representations (`src/data/event_representations.py`)
Two representations of the event stream as spatial grids:
- **5-channel signed voxel:** Temporal bins with ON/OFF polarity accumulation
- **3-channel exponential decay:** Recent events weighted more heavily (active on branch `feature/3channel-direct-cnn`)

Both use a 100ms event window by default and return `(C, H, W)` tensors.

### Loss Function (`src/losses/pose_loss.py`)
`CompositePoseLoss = λ_t × MSE(t_pred, t_gt) + λ_r × 2·arccos(|q_pred·q_gt|)`. Both λ weights are 1.0 by default. The rotation term is geodesic distance on SO(3).

### Dataset (`src/data/dataset.py`)
`SPADESVoxelDataset` loads preprocessed `.h5` voxel grids. Training augmentations are event-camera specific: `RandomIntensityScale`, `RandomFrameDropout`, `SaltPepperNoise`, `RandomErasing`. No augmentations at validation/inference time.

### Training Pipeline
Two-phase approach documented in `README_PIPELINE.md`:
- **Phase 1 (config_25pct.yaml):** 75 sequences, 20 epochs — rapid validation. Pass criteria: Trans < 10%, Rot < 40°, train/val gap < 2×
- **Phase 2 (config_100pct.yaml):** 270 sequences, 100 epochs — production run

GPU optimizations applied in `train.py`: TF32 (`allow_tf32=True`), `torch.compile()`, AMP with `GradScaler`, gradient clipping (`max_norm=1.0`), `ReduceLROnPlateau` scheduler, and early stopping.

## Critical Implementation Notes

- **Timestamp scaling:** Synthetic H5 files use 100µs timestamp units; real test files use 1µs units. Use `--test-timestamp-scale 1.0` carefully — a 100× scaling may be needed depending on the checkpoint's training data.
- **Preprocessed data directory:** The dataset loader uses the preprocessed directory directly (no `sequence_ids_file` logic after recent refactor). The `--data-subset` argument (`25pct` or `100pct`) selects the corresponding `preprocessed_voxels_*` directory.
- **Quaternion convention:** Model outputs `[Qw, Qx, Qy, Qz]`; ground truth labels follow the same order.
- **Metrics:** Relative translation error (%) and rotation error in degrees are the primary eval metrics in `src/utils/metrics.py`.
