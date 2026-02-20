"""
Quick test script to verify all imports and basic functionality.
Run this before starting training to catch any issues early.
"""

import sys
import torch
import numpy as np

print("=" * 70)
print("SPADES Pipeline - System Check")
print("=" * 70)

# Check Python version
print(f"\nPython version: {sys.version}")
assert sys.version_info >= (3, 8), "Python 3.8+ required"
print("  ✓ Python version OK")

# Check PyTorch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
print("  ✓ PyTorch OK")

# Test imports
print("\nTesting imports...")

try:
    from src.data.event_representations import SignedVoxelGridGenerator, filter_events_by_count
    print("  ✓ event_representations")
except Exception as e:
    print(f"  ✗ event_representations: {e}")

try:
    from src.data.dataset import SPADESVoxelDataset, select_stratified_subset, train_val_split
    print("  ✓ dataset")
except Exception as e:
    print(f"  ✗ dataset: {e}")

try:
    from src.models.cnn_lstm_voxel import VoxelCNNLSTM, count_parameters
    print("  ✓ cnn_lstm_voxel")
except Exception as e:
    print(f"  ✗ cnn_lstm_voxel: {e}")

try:
    from src.losses.pose_loss import CompositePoseLoss
    print("  ✓ pose_loss")
except Exception as e:
    print(f"  ✗ pose_loss: {e}")

try:
    from src.utils.metrics import compute_metrics_dict, MetricsTracker
    print("  ✓ metrics")
except Exception as e:
    print(f"  ✗ metrics: {e}")

try:
    from src.utils.visualization import plot_training_curves
    print("  ✓ visualization")
except Exception as e:
    print(f"  ✗ visualization: {e}")

# Test voxel generator
print("\nTesting voxel generator...")
try:
    voxel_gen = SignedVoxelGridGenerator(height=720, width=1280, num_bins=5)
    
    # Create dummy events
    dummy_events = {
        'x': np.random.randint(0, 1280, 10000),
        'y': np.random.randint(0, 720, 10000),
        'p': np.random.randint(0, 2, 10000),
        't': np.random.uniform(0, 100000, 10000)
    }
    
    voxel = voxel_gen.generate(dummy_events, 0, 100000)
    assert voxel.shape == (5, 720, 1280), f"Expected (5, 720, 1280), got {voxel.shape}"
    print(f"  ✓ Generated voxel grid: {voxel.shape}")
except Exception as e:
    print(f"  ✗ Voxel generator test failed: {e}")

# Test model creation
print("\nTesting model creation...")
try:
    model = VoxelCNNLSTM(num_input_channels=5)
    num_params = count_parameters(model)
    print(f"  ✓ Model created: {num_params:,} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(2, 10, 5, 720, 1280)
    model.eval()
    with torch.no_grad():
        trans, rot = model(dummy_input)
    
    assert trans.shape == (2, 10, 3), f"Translation shape incorrect: {trans.shape}"
    assert rot.shape == (2, 10, 4), f"Rotation shape incorrect: {rot.shape}"
    print(f"  ✓ Forward pass successful")
    print(f"    Translation: {trans.shape}")
    print(f"    Rotation: {rot.shape}")
except Exception as e:
    print(f"  ✗ Model test failed: {e}")

# Test loss function
print("\nTesting loss function...")
try:
    criterion = CompositePoseLoss()
    
    pred_trans = torch.randn(2, 10, 3)
    pred_rot = torch.nn.functional.normalize(torch.randn(2, 10, 4), p=2, dim=-1)
    gt_trans = torch.randn(2, 10, 3)
    gt_rot = torch.nn.functional.normalize(torch.randn(2, 10, 4), p=2, dim=-1)
    
    total_loss, trans_loss, rot_loss = criterion(pred_trans, pred_rot, gt_trans, gt_rot)
    
    print(f"  ✓ Loss computation successful")
    print(f"    Total loss: {total_loss.item():.4f}")
    print(f"    Translation loss: {trans_loss.item():.4f}")
    print(f"    Rotation loss: {rot_loss.item():.4f} rad ({torch.rad2deg(rot_loss).item():.2f}°)")
except Exception as e:
    print(f"  ✗ Loss test failed: {e}")

# Test metrics
print("\nTesting metrics...")
try:
    from src.utils.metrics import rotation_error_degrees, relative_translation_error
    
    # Test with identical predictions (should be ~0 error)
    perfect_trans = torch.randn(2, 10, 3)
    perfect_rot = torch.nn.functional.normalize(torch.randn(2, 10, 4), p=2, dim=-1)
    
    trans_err = relative_translation_error(perfect_trans, perfect_trans)
    rot_err = rotation_error_degrees(perfect_rot, perfect_rot)
    
    print(f"  ✓ Metrics computation successful")
    print(f"    Perfect prediction trans error: {trans_err.mean().item():.6f} (should be ~0)")
    print(f"    Perfect prediction rot error: {rot_err.mean().item():.6f}° (should be ~0)")
except Exception as e:
    print(f"  ✗ Metrics test failed: {e}")

# GPU memory check
if torch.cuda.is_available():
    print("\nGPU memory check...")
    try:
        # Allocate memory for typical batch
        dummy_batch = torch.randn(8, 10, 5, 720, 1280).cuda()
        model_gpu = VoxelCNNLSTM(num_input_channels=5).cuda()
        
        with torch.no_grad():
            output = model_gpu(dummy_batch)
        
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        
        print(f"  ✓ GPU memory test passed")
        print(f"    Allocated: {memory_allocated:.2f} GB")
        print(f"    Reserved: {memory_reserved:.2f} GB")
        
        # Clean up
        del dummy_batch, model_gpu, output
        torch.cuda.empty_cache()
        
        if memory_allocated > 12:
            print(f"  ⚠ Warning: Memory usage high ({memory_allocated:.2f} GB)")
            print(f"    Consider reducing batch size if you have <16GB GPU")
    except Exception as e:
        print(f"  ✗ GPU memory test failed: {e}")

print("\n" + "=" * 70)
print("System Check Complete!")
print("=" * 70)
print("\nNext steps:")
print("  1. Generate sequence lists: python scripts/select_subset.py")
print("  2. Preprocess events: python scripts/preprocess_events.py")
print("  3. Train model: python train.py --config config_25pct.yaml --data-subset 25pct")
print("=" * 70 + "\n")
