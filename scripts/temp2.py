import numpy as np
from pathlib import Path
from tqdm import tqdm

def analyze_cube_nodata(dataset_root: str | Path):
    dataset_root = Path(dataset_root)
    # Find all .npz files in all subdirectories (years)
    cube_files = list(dataset_root.rglob("*.npz"))
    
    if not cube_files:
        print(f"❌ No .npz files found in {dataset_root.resolve()}")
        return

    print(f"🔍 Analyzing {len(cube_files)} cubes for any Class 0 (no-data) pixels...")

    total_cubes = len(cube_files)
    cubes_with_nodata = 0
    clean_cubes = 0
    
    # Track pixel-level stats across all cubes for the "similar output" requirement
    # Assuming 20 classes based on your previous snippet
    total_pixel_counts = np.zeros(20, dtype=np.int64)

    for f in tqdm(cube_files, desc="Checking Cubes"):
        try:
            with np.load(f) as data:
                # 'mask' is the key used in 02_generate_dataset.py and dataset.py
                mask = data['mask']
                
                # Check for ANY presence of Class 0
                has_nodata = (mask == 0).any()
                
                if has_nodata:
                    cubes_with_nodata += 1
                else:
                    clean_cubes += 1
                
                # Accumulate class counts for the summary
                counts = np.bincount(mask.flatten(), minlength=20)
                total_pixel_counts += counts[:20]
                
        except Exception as e:
            print(f"Error loading {f.name}: {e}")
            continue

    # --- REPORTING ---
    print("\n" + "="*50)
    print("📊 CUBE-LEVEL NO-DATA ANALYSIS")
    print("="*50)
    print(f"Total Cubes Scanned:          {total_cubes}")
    print(f"Clean Cubes (0% No-Data):     {clean_cubes} ({(clean_cubes/total_cubes)*100:.2f}%)")
    print(f"Cubes with ANY No-Data:       {cubes_with_nodata} ({(cubes_with_nodata/total_cubes)*100:.2f}%)")
    print(f"Action: These {cubes_with_nodata} cubes will be filtered out.")
    
    print("\n📊 PIXEL DISTRIBUTION (CURRENT STATE):")
    print("-" * 40)
    grand_total = total_pixel_counts.sum()
    for i, count in enumerate(total_pixel_counts):
        if count > 0:
            percentage = (count / grand_total) * 100
            print(f"Class {i:02d}: {count:12d} pixels | {percentage:6.2f}%")
        else:
            print(f"Class {i:02d}: {0:12d} pixels |   0.00% (EXTINCT)")
    print("="*50)

if __name__ == "__main__":
    # Point this to your generated dataset folder
    analyze_cube_nodata("../data/dataset")