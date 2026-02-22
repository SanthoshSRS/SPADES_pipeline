"""
Load HDF5 event camera data for SPADES dataset.
Returns format expected by inference.py and data loaders.
"""

import h5py
import pandas as pd


def load_h5_data(h5_path: str) -> dict:
    """
    Load events and labels from an HDF5 file.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file.

    Returns
    -------
    dict with keys:
        'events': dict with keys 'x', 'y', 't', 'p' (singular, as expected by event generators)
        'labels': pd.DataFrame with pose labels and timestamps
    """
    with h5py.File(h5_path, "r") as f:
        # ---- Load events ----
        event_grp = f["events"]
        # Convert from plural keys (xs, ys, ts, ps) to singular (x, y, t, p)
        # as expected by event representation generators
        events = {
            "x": event_grp["xs"][()],
            "y": event_grp["ys"][()],
            "t": event_grp["ts"][()],
            "p": event_grp["ps"][()],
        }

        # ---- Load labels (if available - test files may not have them) ----
        labels_df = None
        if "labels" in f:
            labels_ds = f["labels"]["data"]
            labels_df = pd.DataFrame({
                "filename": labels_ds["filename"].astype(str),
                "Tx": labels_ds["Tx"],
                "Ty": labels_ds["Ty"],
                "Tz": labels_ds["Tz"],
                "Qx": labels_ds["Qx"],
                "Qy": labels_ds["Qy"],
                "Qz": labels_ds["Qz"],
                "Qw": labels_ds["Qw"],
                "timestamp": labels_ds["timestamp"],
            })

    return {
        'events': events,
        'labels': labels_df
    }


def main():
    """Test loading an h5 file."""
    h5_path = "./h5/RT000.h5"

    # Inspect label dtype
    with h5py.File(h5_path, "r") as f:
        print("Label dtype:")
        print(f["labels"]["data"].dtype)

    data = load_h5_data(h5_path)
    events = data['events']
    labels = data['labels']

    # Preview events
    print(f"\nEvent keys: {list(events.keys())}")
    print(f"Num events: {len(events['t'])}")
    print(f"\nLast 10 events:")
    print(pd.DataFrame(events).tail(10))

    # Preview labels
    print(f"\nNum labels: {len(labels)}")
    print("\nLast 10 labels:")
    print(labels.tail(10))


if __name__ == "__main__":
    main()
