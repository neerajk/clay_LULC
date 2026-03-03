import numpy as np
from pathlib import Path
import os

# Adjust this path to where your ground truth masks are stored
mask_dir = Path("../data/processed/lulc_masks")
mask_files = list(mask_dir.glob("*.npy"))

if not mask_files:
    print(f"❌ No .npy files found in {mask_dir.resolve()}")
    # Try looking for .tif if .npy isn't there
    mask_files = list(mask_dir.glob("*.tif"))

print(f"🔍 Analyzing {len(mask_files[:200])} masks for class distribution...")

# Use a fixed size for LULC (usually 20 classes for your setup)
total_counts = np.zeros(20, dtype=np.int64)

for f in mask_files[:200]: # Check first 200 for a solid statistical sample
    try:
        # If they are .npy files
        if f.suffix == '.npy':
            data = np.load(f)
        else:
            # If they are .tif files (requires rasterio)
            import rasterio
            with rasterio.open(f) as src:
                data = src.read(1)
        
        # Count pixels for classes 0-19
        counts = np.bincount(data.flatten(), minlength=20)
        # Ensure we only take the first 20 (in case of weird outliers)
        total_counts += counts[:20]
    except Exception as e:
        continue

print("\n📊 CLASS DISTRIBUTION IN YOUR DATASET:")
print("-" * 40)
grand_total = total_counts.sum()

for i, count in enumerate(total_counts):
    if count > 0:
        percentage = (count / grand_total) * 100
        print(f"Class {i:02d}: {count:10d} pixels | {percentage:6.2f}%")
    else:
        print(f"Class {i:02d}:          0 pixels |   0.00% (EXTINCT)")