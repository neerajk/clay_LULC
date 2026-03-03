import torch
import numpy as np
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    """Loads pre-computed CLAY embeddings and formats them for the CNN Decoder."""
    def __init__(self, npz_files):
        print(f"🗂️  Initializing EmbeddingDataset with {len(npz_files)} files...")
        self.files = npz_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx]) as data:
            embeds = data['embeddings'] 
            mask = data['mask'].astype(np.int64)
            
        # Reshape logic: (257, 1024) -> (16, 16, 1024) -> (1024, 16, 16)
        spatial_tensor = torch.tensor(
            embeds[1:, :].reshape(16, 16, 1024), 
            dtype=torch.float32
        ).permute(2, 0, 1)
        
        mask_tensor = torch.tensor(mask, dtype=torch.long)
        return spatial_tensor, mask_tensor