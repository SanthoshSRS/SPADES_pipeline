"""
Select stratified subset of sequences for 25% validation training.
"""

import os
import argparse
import numpy as np
import random


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Select stratified sequence subset')
    
    parser.add_argument('--total-sequences', type=int, default=300,
                        help='Total number of sequences available')
    parser.add_argument('--subset-pct', type=float, default=0.25,
                        help='Percentage of sequences to select')
    parser.add_argument('--output', type=str, default='sequence_ids_25pct.txt',
                        help='Output file for sequence IDs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def select_stratified_subset(
    total_sequences: int = 300,
    subset_pct: float = 0.25,
    seed: int = 42
) -> list:
    """
    Select stratified subset covering range distribution.
    
    Assumes sequence numbering implies range:
    - RT000-RT099: Close range (3.5-6m)
    - RT100-RT199: Mid range (6-9m)
    - RT200-RT299: Far range (9-12m)
    
    Args:
        total_sequences: Total sequences (300)
        subset_pct: Percentage to select (0.25 = 25%)
        seed: Random seed
        
    Returns:
        sequence_ids: List of sequence IDs (e.g., ['RT000', 'RT005', ...])
    """
    np.random.seed(seed)
    random.seed(seed)
    
    n_subset = int(total_sequences * subset_pct)
    
    # Define range bins
    close_range = list(range(0, 100))      # RT000-RT099
    mid_range = list(range(100, 200))      # RT100-RT199
    far_range = list(range(200, 300))      # RT200-RT299
    
    # Proportional sampling for balance
    # For 25% (75 sequences): ~27% close, 47% mid, 27% far
    n_close = int(n_subset * 0.27)
    n_mid = int(n_subset * 0.47)
    n_far = n_subset - n_close - n_mid  # Remainder
    
    print(f"Selecting {n_subset} sequences ({subset_pct*100:.0f}%):")
    print(f"  Close range (0-99):   {n_close} sequences")
    print(f"  Mid range (100-199):  {n_mid} sequences")
    print(f"  Far range (200-299):  {n_far} sequences")
    
    # Sample from each range
    selected_close = np.random.choice(close_range, n_close, replace=False)
    selected_mid = np.random.choice(mid_range, n_mid, replace=False)
    selected_far = np.random.choice(far_range, n_far, replace=False)
    
    # Combine and sort
    selected_indices = np.concatenate([selected_close, selected_mid, selected_far])
    selected_indices = np.sort(selected_indices)
    
    # Convert to sequence IDs
    sequence_ids = [f"RT{i:03d}" for i in selected_indices]
    
    return sequence_ids


def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print(f"SPADES Stratified Sequence Selection")
    print(f"{'='*60}\n")
    
    # Select sequences
    sequence_ids = select_stratified_subset(
        total_sequences=args.total_sequences,
        subset_pct=args.subset_pct,
        seed=args.seed
    )
    
    # Save to file
    with open(args.output, 'w') as f:
        for seq_id in sequence_ids:
            f.write(f"{seq_id}\n")
    
    print(f"\n✓ Saved {len(sequence_ids)} sequence IDs to: {args.output}")
    
    # Display sample
    print(f"\nSample sequences (first 10):")
    for seq_id in sequence_ids[:10]:
        print(f"  {seq_id}")
    
    if len(sequence_ids) > 10:
        print(f"  ...")
        print(f"  (and {len(sequence_ids) - 10} more)")


if __name__ == "__main__":
    main()
