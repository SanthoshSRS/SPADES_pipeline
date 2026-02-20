# SPADES Pose Estimation Pipeline

Complete implementation for event-based spacecraft pose estimation using CNN+LSTM on signed voxel grids.

## Project Structure

```
SPADES/
├── src/
│   ├── data/
│   │   ├── event_representations.py  # Signed voxel grid generation (5-channel)
│   │   └── dataset.py                # PyTorch Dataset with sequence generation
│   ├── models/
│   │   └── cnn_lstm_voxel.py        # VoxelCNNLSTM architecture (ResNet-18 + LSTM)
│   ├── losses/
│   │   └── pose_loss.py             # Composite pose loss (MSE + geodesic)
│   └── utils/
│       ├── metrics.py               # Translation/rotation error metrics
│       └── visualization.py         # Training and result visualization
├── scripts/
│   ├── select_subset.py             # Stratified 25% sampling
│   ├── preprocess_events.py         # Batch voxel preprocessing
│   ├── validate_submission.py       # Submission format validation
│   └── generate_100pct_list.py      # Generate full dataset sequence list
├── train.py                          # Main training script
├── inference.py                      # Test set prediction
├── config_25pct.yaml                # 25% validation hyperparameters
├── config_100pct.yaml               # 100% production hyperparameters
└── h5/                              # SPADES dataset (.h5 files)
```

## Quick Start

### 1. Generate Sequence Lists

```bash
# Generate 25% stratified subset (75 sequences)
python scripts/select_subset.py --output sequence_ids_25pct.txt

# Generate 100% training list (270 sequences)
python scripts/generate_100pct_list.py
```

### 2. Preprocess Events (Optional but Recommended)

```bash
# Preprocess 25% subset (~50GB, ~2-3 hours with 4 workers)
python scripts/preprocess_events.py \
    --input-dir h5 \
    --output-dir preprocessed_voxels_25pct \
    --sequence-ids-file sequence_ids_25pct.txt \
    --num-workers 4

# Preprocess 100% dataset (~200GB, ~8-10 hours with 8 workers)
python scripts/preprocess_events.py \
    --input-dir h5 \
    --output-dir preprocessed_voxels_100pct \
    --sequence-ids-file sequence_ids_100pct.txt \
    --num-workers 8
```

### 3. Train Model

**Phase 1: 25% Validation (3-4 hours)**

```bash
python train.py --config config_25pct.yaml --data-subset 25pct --device cuda
```

**Decision Point:** Check metrics after 20 epochs:
- Translation error < 10%? ✓
- Rotation error < 40°? ✓
- Train/val gap < 2×? ✓
- Converged before epoch 20? ✓

If all criteria met, proceed to Phase 2.

**Phase 2: 100% Production (50 hours)**

```bash
python train.py --config config_100pct.yaml --data-subset 100pct --device cuda
```

Target performance:
- Translation error < 5%
- Rotation error < 10° (beat SPADES paper 79° baseline)

### 4. Run Inference

```bash
python inference.py \
    --checkpoint checkpoints/100pct_best.pth \
    --test-dir h5 \
    --output-dir submission \
    --test-timestamp-scale 1.0  # Use 1.0 for real test data (1µs timestamps)
```

### 5. Validate Submission

```bash
python scripts/validate_submission.py --submission-dir submission --verbose
```

### 6. Create Submission ZIP

```bash
cd submission
zip -r ../submission.zip *.csv
```

## Architecture

**VoxelCNNLSTM Model (~15M parameters)**

1. **Event Representation:** 5-channel signed voxel grid
   - 5 temporal bins (20ms each, 100ms window)
   - Signed accumulation: ON events = +1, OFF events = -1
   - Exploits 38/62% ON/OFF polarity bias from dataset

2. **CNN Backbone:** ResNet-18 (modified conv1 for 5 channels)
   - Extracts spatial features from each voxel grid frame
   - Output: 512-dimensional feature vector per frame

3. **LSTM:** 2 layers, 256 hidden units
   - Models temporal dependencies across 10-frame sequences
   - Unidirectional (causal) for real-time applicability

4. **Dual Regression Heads:**
   - Translation: FC(256→128→3) → [Tx, Ty, Tz]
   - Rotation: FC(256→128→4) → [Qw, Qx, Qy, Qz] (normalized quaternion)

## Loss Function

**Composite Pose Loss:**
- Translation: MSE ||t_pred - t_gt||²
- Rotation: Geodesic distance 2·arccos(|q_pred · q_gt|) on SO(3)
- Total: λ_t · L_trans + λ_r · L_rot (λ_t = λ_r = 1.0)

## Training Strategy

**Two-Phase Validation Approach:**

**Phase 1 (25% subset):**
- 75 sequences, ~24K poses, 20 epochs, ~3-4 hours
- Purpose: Rapid architecture validation
- Success criteria: Translation <10%, Rotation <40°

**Phase 2 (100% production):**
- 270 sequences, ~107K poses, 100 epochs, ~50 hours
- Purpose: Competition-ready model
- Target: Translation <5%, Rotation <10°

## Dataset

**SPADES (SPAcecraft Pose Estimation Dataset using Event Sensing)**
- Synthetic training: 300 sequences (RT000-RT299), 179,400 poses
- Real test: 31 sequences (T000+), 15,500 poses
- Event camera: 1280×720, 10Hz pose sampling, 60s duration
- Range distribution: 3.5-12m spacecraft distance

**Critical timestamp difference:**
- Synthetic: 100µs timestamp units
- Real test: 1µs timestamp units (100× scaling required in inference)

## Performance Targets

**SPADES Paper Baseline (arXiv:2311.05310):**
- 3-channel representation + Hybrid pipeline
- Rotation error: **79° on real test data**

**Our Target (CNN+LSTM + 5-channel signed voxels):**
- Translation: < 5% relative error
- Rotation: < 10° (8× improvement over baseline)
- Full temporal modeling vs frame-by-frame

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
h5py>=3.8.0
matplotlib>=3.7.0
pyyaml>=6.0
tqdm>=4.65.0
tensorboard>=2.13.0
```

## Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir logs
```

Monitor:
- Loss curves (train/val)
- Translation error (%)
- Rotation error (degrees)
- Learning rate schedule

## Notes

- **GPU Memory:** ~8GB required for batch_size=8
- **Disk Space:** 50GB (25%) or 200GB (100%) for preprocessed voxels
- **Training Time:** 3-4 hours (25%) or 50 hours (100%) on single GPU
- **Timestamp Units:** Critical to handle 100× difference between synthetic/real data

## Citation

```
@article{rathinam2023spades,
  title={SPADES: A Realistic Spacecraft Pose Estimation Dataset using Event Sensing},
  author={Rathinam, Arunkumar and Saripalli, Srikanth},
  journal={arXiv preprint arXiv:2311.05310},
  year={2023}
}
```
