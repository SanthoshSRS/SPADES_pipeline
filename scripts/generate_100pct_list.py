"""
Generate sequence_ids_100pct.txt - all 270 training sequences (90% of 300 total).

Run this script to create the full training sequence list:
    python scripts/generate_100pct_list.py
"""

import os


def generate_100pct_list(output_file='sequence_ids_100pct.txt'):
    """
    Generate list of all training sequences (90% of 300 = 270 sequences).
    
    Following convention:
    - Total sequences: RT000-RT299 (300 sequences)
    - Training: 90% = 270 sequences (RT000-RT269)
    - Validation: 10% = 30 sequences (RT270-RT299)
    """
    # Use first 270 sequences for training (RT000-RT269)
    training_sequences = [f"RT{i:03d}" for i in range(270)]
    
    # Save to file
    with open(output_file, 'w') as f:
        for seq_id in training_sequences:
            f.write(f"{seq_id}\n")
    
    print(f"✓ Created {output_file} with {len(training_sequences)} sequences")
    print(f"  Training: RT000-RT269 (270 sequences)")
    print(f"  Reserved for test: RT270-RT299 (30 sequences)")
    
    return training_sequences


if __name__ == "__main__":
    sequences = generate_100pct_list()
