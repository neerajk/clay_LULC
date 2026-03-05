import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import yaml
import torch
import numpy as np
import json
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

import sys
# Ensure the root src folder is in the path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dataset import EmbeddingDataset
from src.models.decoder import LULCSegmentationModule, EpochSummaryCallback
from src.visualization.lulc_classes import LULC_CLASS_MAP

def _print_class_weight_table(pixel_counts, weights_np, class_map, ignore_index=0):
    present_ids = [i for i, c in enumerate(pixel_counts) if c > 0 and i != ignore_index]
    if not present_ids:
        print("⚠️  No non-ignore classes present in train split.")
        return

    present_ids = sorted(present_ids, key=lambda i: pixel_counts[i], reverse=True)
    total_pixels = float(sum(pixel_counts[i] for i in present_ids))

    print("\n📊 Class pixel distribution and computed weights")
    print("-" * 130)
    print(
        f"{'Class':>5} | {'Level-I':<20} | {'Level-II':<34} | "
        f"{'Pixels':>12} | {'Share %':>8} | {'Weight':>8}"
    )
    print("-" * 130)
    for cls_id in present_ids:
        cnt = int(pixel_counts[cls_id])
        share = (100.0 * cnt / total_pixels) if total_pixels > 0 else 0.0
        w = float(weights_np[cls_id])
        lvl1, lvl2 = class_map.get(cls_id, ("Unknown", "Unknown"))
        print(
            f"{cls_id:>5} | {lvl1:<20} | {lvl2:<34} | "
            f"{cnt:>12,} | {share:>7.2f}% | {w:>8.3f}"
        )
    print("-" * 130)

def _dominant_non_ignore_class(mask, num_classes, ignore_index):
    valid = mask[mask != ignore_index]
    if valid.size == 0:
        return None
    counts = np.bincount(valid.ravel(), minlength=num_classes)[:num_classes]
    if counts.sum() == 0:
        return None
    return int(np.argmax(counts))

def prepare_stratified_dataset(
    embed_dir,
    num_classes=20,
    split_ratio=0.8,
    ignore_index=0,
    min_valid_fraction=0.60,
    class_weight_power=0.5,
    class_weight_clip=6.0,
    seed=42
):
    """
    Builds a stratified split by dominant class while preserving each embedding<->mask pair.
    Also computes damped class weights from train pixels.
    """
    rng = np.random.default_rng(seed)
    all_files = sorted(Path(embed_dir).rglob('emb_cube_*.npz'))
    per_class_files = {i: [] for i in range(num_classes)}
    reject_stats = Counter()

    print(f"🚀 PHASE 1: Validating and indexing {len(all_files)} embedding cubes...")
    for f in tqdm(all_files, desc="Scanning Cubes"):
        try:
            with np.load(f) as data:
                if "embeddings" not in data or "mask" not in data:
                    reject_stats["missing_keys"] += 1
                    continue
                mask = data["mask"]
        except Exception:
            reject_stats["load_error"] += 1
            continue

        if mask.shape != (256, 256):
            reject_stats["bad_shape"] += 1
            continue
        if mask.min() < 0 or mask.max() >= num_classes:
            reject_stats["bad_label_range"] += 1
            continue

        valid_fraction = float((mask != ignore_index).mean())
        if valid_fraction < float(min_valid_fraction):
            reject_stats["low_valid_fraction"] += 1
            continue

        dom_cls = _dominant_non_ignore_class(mask, num_classes, ignore_index)
        if dom_cls is None:
            reject_stats["all_ignore"] += 1
            continue
        per_class_files[dom_cls].append(f)

    usable_files = sum(len(v) for v in per_class_files.values())
    print(
        f"✅ Usable cubes: {usable_files}/{len(all_files)} | "
        f"rejects={dict(reject_stats)}"
    )

    print("🚀 PHASE 2: Stratified split by dominant class...")
    train_list, val_list = [], []
    for cls_id in range(num_classes):
        files = per_class_files[cls_id]
        if not files:
            continue
        files = list(files)
        rng.shuffle(files)

        if len(files) == 1:
            n_val = 0
        else:
            n_val = int(round(len(files) * (1.0 - split_ratio)))
            n_val = max(1, n_val)
            n_val = min(n_val, len(files) - 1)

        val_list.extend(files[:n_val])
        train_list.extend(files[n_val:])

    if len(train_list) == 0:
        raise RuntimeError("No training cubes available after filtering.")
    if len(val_list) == 0:
        # Fallback to keep training robust if classes are too sparse.
        n_val = max(1, int(len(train_list) * 0.1))
        rng.shuffle(train_list)
        val_list = train_list[:n_val]
        train_list = train_list[n_val:]
        print(f"⚠️  Validation split was empty; fallback random split applied (val={len(val_list)}).")

    print(f"📦 Split sizes | train={len(train_list)} | val={len(val_list)}")

    # Log split for reproducibility
    log_path = Path("../models/logs/dataset_split.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({
            "train_files": [f.name for f in train_list],
            "val_files": [f.name for f in val_list],
            "stats": {
                "raw_files": len(all_files),
                "usable_files": usable_files,
                "rejected": dict(reject_stats),
                "num_classes": num_classes,
                "ignore_index": ignore_index,
                "min_valid_fraction": min_valid_fraction,
            }
        }, f, indent=4)
    print(f"✅ Split history logged to {log_path.resolve()}")

    print("🚀 PHASE 3: Calculating class weights from train pixels...")
    pixel_counts = np.zeros(num_classes)
    for f in tqdm(train_list, desc="Summing pixels"):
        with np.load(f) as data:
            pixel_counts += np.bincount(data['mask'].flatten(), minlength=num_classes)[:num_classes]

    pixel_counts[ignore_index] = 0
    weights_np = 1.0 / (np.power(pixel_counts + 1e-6, class_weight_power))
    weights_np[pixel_counts == 0] = 0.0
    positive = weights_np > 0
    if np.any(positive):
        weights_np[positive] = weights_np[positive] / weights_np[positive].mean()
        weights_np[positive] = np.clip(weights_np[positive], 0.0, class_weight_clip)
    weights_np[ignore_index] = 0.0
    weights = torch.tensor(weights_np, dtype=torch.float32)

    _print_class_weight_table(
        pixel_counts,
        weights_np,
        class_map=LULC_CLASS_MAP,
        ignore_index=ignore_index
    )

    return train_list, val_list, weights

def run_training():
    print("\n" + "="*60)
    print("🚀 INITIALIZING DECODER TRAINING PIPELINE")
    print("="*60)

    config_file = "../configs/train_config.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    pl.seed_everything(int(config["training"].get("seed", 42)), workers=True)

    # Data preparation
    train_files, val_files, sync_weights = prepare_stratified_dataset(
        config["paths"]["embed_dir"],
        num_classes=config["training"]["num_classes"],
        split_ratio=float(config["training"].get("train_split_ratio", 0.8)),
        ignore_index=int(config["training"]["ignore_index"]),
        min_valid_fraction=float(config["training"].get("min_valid_fraction", 0.60)),
        class_weight_power=float(config["training"].get("class_weight_power", 0.5)),
        class_weight_clip=float(config["training"].get("class_weight_clip", 6.0)),
        seed=int(config["training"].get("seed", 42))
    )

    # Model setup
    model = LULCSegmentationModule(
        num_classes=config["training"]["num_classes"],
        lr=config["training"]["learning_rate"],
        ignore_index=config["training"]["ignore_index"],
        class_weights=sync_weights,
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
        lr_factor=float(config["training"].get("lr_factor", 0.5)),
        lr_patience=int(config["training"].get("lr_patience", 4)),
        min_lr=float(config["training"].get("min_lr", 1e-6)),
        dice_weight=float(config["training"].get("dice_weight", 0.25)),
        patch_size=int(config["training"].get("patch_size", 16)),
        full_loss_weight=float(config["training"].get("full_loss_weight", 0.6)),
        coarse_loss_weight=float(config["training"].get("coarse_loss_weight", 0.4)),
        monitor_metric=str(config["training"].get("monitor_metric", "val_mIoU_patch")),
    )

    # Logger and callbacks
    csv_logger = CSVLogger(save_dir="../models/logs/", name="lulc_decoder")
    ckpt_dir = Path(config["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    monitor_metric = str(config["training"].get("monitor_metric", "val_mIoU_patch"))
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="decoder-epoch{epoch:02d}-patch{val_mIoU_patch:.4f}-full{val_mIoU:.4f}-vloss{val_loss:.4f}",
        monitor=monitor_metric,
        mode="max",
        save_top_k=int(config["training"].get("save_top_k", 3)),
        save_last=True,
        auto_insert_metric_name=False,
    )
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        mode="max",
        patience=int(config["training"].get("early_stopping_patience", 20)),
        min_delta=float(config["training"].get("early_stopping_min_delta", 0.0005)),
    )

    num_workers = int(config["dataloader"].get("num_workers", 0))
    loader_kwargs = dict(
        batch_size=int(config["training"]["batch_size"]),
        num_workers=num_workers,
        pin_memory=bool(config["dataloader"].get("pin_memory", False)),
        persistent_workers=bool(num_workers > 0),
    )

    print(
        f"🧪 Run config | epochs={config['training']['max_epochs']} | batch={config['training']['batch_size']} | "
        f"lr={config['training']['learning_rate']} | num_workers={num_workers} | monitor={monitor_metric}"
    )
    print(f"💾 Checkpoints => {ckpt_dir.resolve()}")

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        logger=csv_logger,
        log_every_n_steps=int(config["training"].get("log_every_n_steps", 10)),
        num_sanity_val_steps=0,
        callbacks=[
            checkpoint_callback,
            early_stopping,
            EpochSummaryCallback()
        ],
        accelerator='auto',
        devices=1,
        enable_progress_bar=True,
        gradient_clip_val=float(config["training"].get("gradient_clip_val", 1.0)),
    )

    # Fit
    trainer.fit(
        model,
        train_dataloaders=DataLoader(
            EmbeddingDataset(train_files),
            shuffle=True,
            **loader_kwargs
        ),
        val_dataloaders=DataLoader(
            EmbeddingDataset(val_files),
            shuffle=False,
            **loader_kwargs
        )
    )

if __name__ == "__main__":
    run_training()
