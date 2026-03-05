import torch
import numpy as np
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    """Loads pre-computed CLAY embeddings and formats them for the CNN Decoder."""
    def __init__(self, npz_files):
        self.files = sorted(npz_files)
        self._shape_logged = False
        print(f"🗂️  Initializing EmbeddingDataset with {len(self.files)} files...")

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _format_embeddings(embeds: np.ndarray) -> torch.Tensor:
        """
        Supported shapes:
        - (257, 1024): CLS + 16x16 patches (CLAY default)
        - (256, 1024): 16x16 patches without CLS
        - (16, 16, 1024): HWC patch-grid
        - (1024, 16, 16): already CHW
        """
        if embeds.ndim == 2:
            if embeds.shape == (257, 1024):
                patch_tokens = embeds[1:, :]
                return torch.tensor(
                    patch_tokens.reshape(16, 16, 1024),
                    dtype=torch.float32
                ).permute(2, 0, 1).contiguous()
            if embeds.shape == (256, 1024):
                return torch.tensor(
                    embeds.reshape(16, 16, 1024),
                    dtype=torch.float32
                ).permute(2, 0, 1).contiguous()
            raise ValueError(f"Unsupported 2D embedding shape: {embeds.shape}")

        if embeds.ndim == 3:
            if embeds.shape == (16, 16, 1024):
                return torch.tensor(embeds, dtype=torch.float32).permute(2, 0, 1).contiguous()
            if embeds.shape == (1024, 16, 16):
                return torch.tensor(embeds, dtype=torch.float32).contiguous()
            raise ValueError(f"Unsupported 3D embedding shape: {embeds.shape}")

        raise ValueError(f"Unsupported embedding ndim: {embeds.ndim}")

    def __getitem__(self, idx):
        path = self.files[idx]
        with np.load(path) as data:
            if "embeddings" not in data or "mask" not in data:
                raise KeyError(f"{path} is missing required keys. Found: {list(data.files)}")

            embeds = data["embeddings"]
            mask = data["mask"].astype(np.int64)

        spatial_tensor = self._format_embeddings(embeds)
        if mask.shape != (256, 256):
            raise ValueError(f"{path} has invalid mask shape {mask.shape}; expected (256, 256)")
        mask_tensor = torch.tensor(mask, dtype=torch.long)

        if not self._shape_logged:
            print(
                f"📏 Sample check: file={path.name} | embeddings={tuple(spatial_tensor.shape)} | "
                f"mask={tuple(mask_tensor.shape)} | classes={int(mask_tensor.min())}-{int(mask_tensor.max())}"
            )
            self._shape_logged = True

        return spatial_tensor, mask_tensor
