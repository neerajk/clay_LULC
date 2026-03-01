import os
# Set environment to handle OpenMP duplicates on Mac
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
import traceback
from claymodel.module import ClayMAEModule 

# Set environment to handle OpenMP duplicates on Mac
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- CONFIGURATION ---
# Adjusted to match your screenshot: data/dataset/{year}/cube_*.npz
DATA_DIR = Path("../data/dataset") 
CKPT_PATH = Path("../models/clay-v1.5.ckpt")

_model = None
_device = None

def get_model():
    global _model, _device
    if _model is None:
        _device = torch.device("cpu")
        _model = ClayMAEModule.load_from_checkpoint(str(CKPT_PATH))
        _model.eval()
        _model.to(_device)
    return _model, _device

def process_cube(path):
    try:
        # 1. Load and check if embedding already exists
        with np.load(path, allow_pickle=True) as data:
            if 'embedding' in data.files:
                return "skipped"
            
            # Extract existing data using your keys
            image = data['image']
            mask = data['mask']
            latlon = data['latlon']
            year = data['year']

        # 2. Get Model & Device
        model, device = get_model()

        # 3. Pre-process (Normalization for CLAY)
        img_tensor = torch.from_numpy(image).float().to(device)
        img_tensor = (img_tensor / 10000.0).unsqueeze(0) 

        # 4. Inference: Extract Spatial Patches
        with torch.no_grad():
            # Get patch embeddings (excluding class token at index 0)
            unmsk_patch, _, _, _ = model.model.encoder(img_tensor)
            patch_embeddings = unmsk_patch[:, 1:, :] 
            
        embedding_np = patch_embeddings.cpu().squeeze().numpy()

        # 5. Save back to the SAME file with the new embedding key
        np.savez_compressed(
            path,
            image=image,
            mask=mask,
            latlon=latlon,
            year=year,
            embedding=embedding_np
        )
        return "success"

    except Exception:
        return f"error: {traceback.format_exc()}"

def run_parallel_embeddings():
    print("="*60)
    print("🚀 TASK 3: YEAR-BY-YEAR CLAY EMBEDDINGS (Parallel CPU)")
    print("="*60)
    
    if not CKPT_PATH.exists():
        print(f"❌ [ERROR] Checkpoint not found: {CKPT_PATH.resolve()}")
        return

    # --- YEAR-WISE DISCOVERY ---
    # Get all subdirectories in dataset (e.g., 1985, 1995, 2005)
    year_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    
    if not year_dirs:
        print(f"⚠️ [WARN] No year directories found in {DATA_DIR.resolve()}")
        return

    for y_dir in year_dirs:
        year_name = y_dir.name
        cube_paths = list(y_dir.glob("cube_*.npz"))
        
        if not cube_paths:
            continue

        print(f"\n📅 [STARTING] Processing Year: {year_name} | Found {len(cube_paths)} cubes")
        
        count_success = 0
        count_skipped = 0
        count_failed = 0

        # Process this specific year's cubes using 3 workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(process_cube, p): p for p in cube_paths}
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                              total=len(futures), 
                              desc=f"Year {year_name}"):
                result = future.result()
                
                if result == "success":
                    count_success += 1
                elif result == "skipped":
                    count_skipped += 1
                else:
                    count_failed += 1
                    tqdm.write(f"❌ Failed: {futures[future]}\n{result}")

        print(f"🏁 [COMPLETED] Year: {year_name}")
        print(f"   ✅ Success: {count_success} | ⏭️ Skipped: {count_skipped} | ❌ Failed: {count_failed}")

    print("\n" + "="*60)
    print("🎯 ALL DIRECTORIES PROCESSED")
    print("="*60)

if __name__ == "__main__":
    # Prevent internal torch parallelism from conflicting with our 3 workers
    torch.set_num_threads(1) 
    run_parallel_embeddings()