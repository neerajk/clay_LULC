import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

# --- IMPORT CLAY ---
from claymodel.module import ClayMAEModule

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATASET_BASE_DIR = Path('../data/dataset/')
EMBEDDING_BASE_DIR = Path('../data/embeddings/')

PLATFORM = "landsat-c2-l2"
BAND_NAMES = ["red", "green", "blue", "nir08", "swir16", "swir22"]

CHECKPOINT_PATH = "../models/clay-v1.5.ckpt"
METADATA_PATH = "../configs/metadata.yaml" 

# Set your batch size here! 
# (If 128 crashes your RAM, lower it to 64 or 32)
BATCH_SIZE = 128 
NUM_WORKERS = 4 

DEVICE = torch.device("cpu")

print("\n" + "="*50)
print("🚀 INITIALIZING PYTORCH BATCHED INFERENCE PIPELINE")
print(f"🖥️  Device:       {DEVICE}")
print(f"📦 Batch Size:   {BATCH_SIZE}")
print("="*50 + "\n")

# ==========================================
# 2. EXTRACT METADATA
# ==========================================
with open(METADATA_PATH, 'r') as f:
    clay_meta = yaml.safe_load(f)

GSD = clay_meta[PLATFORM]['gsd']
WAVELENGTHS = [clay_meta[PLATFORM]['bands']['wavelength'][b] for b in BAND_NAMES]
MEANS = [clay_meta[PLATFORM]['bands']['mean'][b] for b in BAND_NAMES]
STDS = [clay_meta[PLATFORM]['bands']['std'][b] for b in BAND_NAMES]

# ==========================================
# 3. CUSTOM PYTORCH DATASET
# ==========================================
class CubeDataset(Dataset):
    """Mimics ClayDataModule but reads our custom .npz files directly."""
    def __init__(self, npz_files, means, stds):
        self.files = npz_files
        # FORCE numpy to use float32 to prevent automatic upcasting to float64 (double)
        self.means = np.array(means, dtype=np.float32).reshape(-1, 1, 1)
        self.stds = np.array(stds, dtype=np.float32).reshape(-1, 1, 1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with np.load(path) as data:
            pixels = data['pixels'].astype(np.float32)
            mask = data['mask']
            latlon = np.concatenate([data['lat_norm'], data['lon_norm']]).astype(np.float32)
            time = np.concatenate([data['week_norm'], data['hour_norm']]).astype(np.float32)

        # Normalize on the fly (This will now stay as float32)
        pixels_norm = (pixels - self.means) / self.stds

        return {
            "pixels": pixels_norm,
            "time": time,
            "latlon": latlon,
            "mask": mask,
            "filename": path.name
        }

# ==========================================
# 4. BATCH CREATION HELPER
# ==========================================
def create_batch(batch_dict, wavelengths, gsd, device):
    """Formats the batched tensors exactly as CLAY expects."""
    batch = {}
    # Explicitly cast to torch.float32 just as a secondary safety net
    batch["pixels"] = batch_dict["pixels"].to(device, dtype=torch.float32)
    batch["time"] = batch_dict["time"].to(device, dtype=torch.float32)
    batch["latlon"] = batch_dict["latlon"].to(device, dtype=torch.float32)
    
    # Metadata scalars 
    batch["waves"] = torch.tensor(wavelengths, dtype=torch.float32).to(device)
    batch["gsd"] = torch.tensor(gsd, dtype=torch.float32).to(device)
    return batch
# ==========================================
# 5. INFERENCE LOGIC
# ==========================================
def run_batched_inference():
    print(f"📦 Loading CLAY Weights from: {CHECKPOINT_PATH}")
    module = ClayMAEModule.load_from_checkpoint(
        checkpoint_path=CHECKPOINT_PATH,
        model_size="large",
        metadata_path=METADATA_PATH,
        dolls=[16, 32, 64, 128, 256, 768, 1024],
        doll_weights=[1, 1, 1, 1, 1, 1, 1],
        mask_ration=0.0, 
        shuffle=False,
    )
    module.eval()
    module.to(DEVICE)
    print("✅ Model loaded!\n")

    year_dirs = [d for d in DATASET_BASE_DIR.iterdir() if d.is_dir() and "stripped" not in d.name]

    for year_dir in sorted(year_dirs):
        year = year_dir.name
        cube_files = list(year_dir.glob('cube_*.npz'))
        
        if not cube_files: continue
            
        out_dir = EMBEDDING_BASE_DIR / year
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*40}")
        print(f"📅 STARTING YEAR: {year} | {len(cube_files)} cubes")
        print(f"{'='*40}")

        # Initialize PyTorch Dataset & DataLoader
        dataset = CubeDataset(cube_files, MEANS, STDS)
        dataloader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False
        )

        cubes_processed = 0
        print_shapes_next = True # Flag to print shapes for the 1st batch of the year

        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Batches ({year})")):
            
            # Format batch for CLAY
            clay_batch = create_batch(batch_data, WAVELENGTHS, GSD, DEVICE)
            
            # 1. PRINT SHAPES (1st batch of the year)
            if print_shapes_next:
                print("\n   [SHAPE CHECK - FIRST BATCH]")
                print(f"   {PLATFORM:<15} px: {clay_batch['pixels'].shape} | time: {clay_batch['time'].shape} | latlon: {clay_batch['latlon'].shape}")
                print_shapes_next = False

            # Run Encoder
            with torch.no_grad():
                unmsk_patch, *_ = module.model.encoder(clay_batch)
            
            # Convert to numpy for saving
            unmsk_patch_np = unmsk_patch.cpu().numpy()
            masks_np = batch_data["mask"].numpy()
            filenames = batch_data["filename"]

            # 2. PRINT SHAPES (Every ~100 cubes)
            cubes_processed += len(filenames)
            if (cubes_processed % 100) < BATCH_SIZE and cubes_processed > BATCH_SIZE:
                # This triggers roughly every 100 cubes passed
                tqdm.write(f"   ➤ [Cube {cubes_processed}] unmsk_patch shape: {unmsk_patch.shape}")

            # Save individual .npz files
            for i in range(len(filenames)):
                out_filename = out_dir / f"emb_{filenames[i]}"
                np.savez_compressed(
                    out_filename,
                    embeddings=unmsk_patch_np[i],
                    mask=masks_np[i]
                )

    print(f"\n🎉 PIPELINE COMPLETE! All embeddings stored in {EMBEDDING_BASE_DIR.resolve()}")

if __name__ == "__main__":
    run_batched_inference()