"""
Validate submission CSV files for competition format.
"""

import os
import argparse
import pandas as pd
import numpy as np
from glob import glob


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Validate submission CSV files')
    
    parser.add_argument('--submission-dir', type=str, required=True,
                        help='Directory containing submission CSV files')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed validation info')
    
    return parser.parse_args()


def validate_csv_format(csv_path: str, verbose: bool = False) -> dict:
    """
    Validate single CSV file.
    
    Expected format:
    - Columns: timestamp, Tx, Ty, Tz, Qx, Qy, Qz, Qw
    - No missing values
    - Quaternion unit norm (within tolerance)
    - Reasonable value ranges
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'num_poses': 0
    }
    
    seq_id = os.path.basename(csv_path).replace('.csv', '')
    
    try:
        # Load CSV
        df = pd.read_csv(csv_path)
        results['num_poses'] = len(df)
        
        # Check columns
        required_cols = ['timestamp', 'Tx', 'Ty', 'Tz', 'Qx', 'Qy', 'Qz', 'Qw']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            results['valid'] = False
            results['errors'].append(f"Missing columns: {missing_cols}")
            return results
        
        # Check for NaN values
        if df.isnull().any().any():
            results['valid'] = False
            nan_cols = df.columns[df.isnull().any()].tolist()
            results['errors'].append(f"NaN values found in columns: {nan_cols}")
        
        # Check for infinite values
        numeric_cols = ['Tx', 'Ty', 'Tz', 'Qx', 'Qy', 'Qz', 'Qw']
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                results['valid'] = False
                results['errors'].append(f"Infinite values in column: {col}")
        
        # Validate quaternion unit norm
        quaternions = df[['Qx', 'Qy', 'Qz', 'Qw']].values
        norms = np.linalg.norm(quaternions, axis=1)
        
        norm_tolerance = 0.01  # Allow 1% deviation
        invalid_norms = np.abs(norms - 1.0) > norm_tolerance
        
        if np.any(invalid_norms):
            num_invalid = np.sum(invalid_norms)
            max_deviation = np.max(np.abs(norms - 1.0))
            results['warnings'].append(
                f"Quaternion norm deviation: {num_invalid}/{len(norms)} poses "
                f"(max deviation: {max_deviation:.4f})"
            )
            
            if max_deviation > 0.1:  # Critical if >10% deviation
                results['valid'] = False
                results['errors'].append("Critical quaternion norm deviation (>0.1)")
        
        # Check translation range (should be reasonable for SPADES)
        trans = df[['Tx', 'Ty', 'Tz']].values
        trans_norms = np.linalg.norm(trans, axis=1)
        
        if np.any(trans_norms < 1.0) or np.any(trans_norms > 20.0):
            results['warnings'].append(
                f"Translation range unusual: {trans_norms.min():.2f} to {trans_norms.max():.2f} m "
                f"(expected 3.5-12m for SPADES)"
            )
        
        # Check timestamp ordering
        timestamps = df['timestamp'].values
        if not np.all(timestamps[1:] >= timestamps[:-1]):
            results['valid'] = False
            results['errors'].append("Timestamps not monotonically increasing")
        
        # Check for duplicate timestamps
        if len(np.unique(timestamps)) != len(timestamps):
            results['warnings'].append("Duplicate timestamps found")
        
        # Trajectory smoothness check (large jumps might indicate issues)
        if len(trans) > 1:
            diffs = np.linalg.norm(trans[1:] - trans[:-1], axis=1)
            max_jump = np.max(diffs)
            
            if max_jump > 1.0:  # Large jump (>1m between consecutive poses)
                results['warnings'].append(
                    f"Large trajectory jump detected: {max_jump:.2f} m "
                    f"(might indicate prediction issues)"
                )
        
    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"Failed to load/parse CSV: {str(e)}")
    
    return results


def main():
    args = parse_args()
    
    print(f"\n{'='*70}")
    print(f"SPADES Submission Validation")
    print(f"{'='*70}\n")
    
    # Find all CSV files
    csv_pattern = os.path.join(args.submission_dir, "*.csv")
    csv_files = sorted(glob(csv_pattern))
    
    if len(csv_files) == 0:
        print(f"ERROR: No CSV files found in {args.submission_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files\n")
    
    # Validate each file
    all_valid = True
    total_poses = 0
    
    for csv_path in csv_files:
        seq_id = os.path.basename(csv_path).replace('.csv', '')
        results = validate_csv_format(csv_path, verbose=args.verbose)
        
        total_poses += results['num_poses']
        
        # Print results
        status = "✓ VALID" if results['valid'] else "✗ INVALID"
        print(f"  {seq_id}: {status} ({results['num_poses']} poses)")
        
        if results['errors']:
            all_valid = False
            for error in results['errors']:
                print(f"    ERROR: {error}")
        
        if results['warnings'] and args.verbose:
            for warning in results['warnings']:
                print(f"    WARNING: {warning}")
        
        if not results['valid']:
            all_valid = False
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Validation Summary:")
    print(f"{'='*70}")
    print(f"  Total sequences: {len(csv_files)}")
    print(f"  Total poses: {total_poses}")
    print(f"  Status: {'✓ ALL VALID' if all_valid else '✗ VALIDATION FAILED'}")
    print(f"{'='*70}\n")
    
    if all_valid:
        print("✓ Submission ready for competition upload!")
        print(f"\nTo create submission.zip:")
        print(f"  cd {args.submission_dir}")
        print(f"  zip -r ../submission.zip *.csv")
    else:
        print("✗ Please fix validation errors before submitting")
    
    return 0 if all_valid else 1


if __name__ == "__main__":
    exit(main())
