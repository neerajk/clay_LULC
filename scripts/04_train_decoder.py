import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dataset import EmbeddingDataset
from src.models.decoder import LULCSegmentationModule, EpochSummaryCallback
from src.visualization.plots import plot_training_results
from src.visualization.lulc_classes import LULC_CLASS_MAP

def run_training():
    # Only the main thread gets inside here, so it only prints once!
    print("Libraries imported successfully.")
    print("Ready to run training pipeline.")
    
    print("\n" + "="*60)
    print("🚀 INITIALIZING DECODER TRAINING PIPELINE")
    print("="*60)

    config_file = "../configs/train_config.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    EMBED_DIR = Path(config["paths"]["embed_dir"])
    CHECKPOINT_DIR = Path(config["paths"]["checkpoint_dir"])
    SEED = config["training"]["seed"]
    BATCH_SIZE = config["training"]["batch_size"]

    all_files = list(EMBED_DIR.rglob('emb_cube_*.npz'))
    train_size = int(0.8 * len(all_files))
    
    train_files, val_files = random_split(
        all_files, [train_size, len(all_files) - train_size], 
        generator=torch.Generator().manual_seed(SEED)
    )

    # Added persistent_workers=True to silence the Lightning warning
    train_loader = DataLoader(
        EmbeddingDataset(train_files), batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=config["dataloader"]["num_workers"], 
        pin_memory=config["dataloader"]["pin_memory"],
        persistent_workers=True 
    )
    val_loader = DataLoader(
        EmbeddingDataset(val_files), batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=config["dataloader"]["num_workers"], 
        pin_memory=config["dataloader"]["pin_memory"],
        persistent_workers=True
    )

    model = LULCSegmentationModule(
        num_classes=config["training"]["num_classes"], 
        lr=config["training"]["learning_rate"],
        ignore_index=config["training"]["ignore_index"]
    )
    
    csv_logger = CSVLogger(save_dir="../models/logs/", name="lulc_decoder")
    
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        logger=csv_logger,
        callbacks=[
            ModelCheckpoint(dirpath=CHECKPOINT_DIR, monitor='val_mIoU', mode='max'), 
            EarlyStopping(monitor='val_loss', patience=7),
            EpochSummaryCallback() 
        ],
        accelerator='auto', 
        devices=1,
        log_every_n_steps=10
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    plot_training_results(
        log_dir=csv_logger.experiment.metrics_file_path, 
        final_cm=model.last_val_cm,
        num_classes=config["training"]["num_classes"],
        ignore_index=config["training"]["ignore_index"],
        class_map=LULC_CLASS_MAP
    )

if __name__ == "__main__":
    run_training()