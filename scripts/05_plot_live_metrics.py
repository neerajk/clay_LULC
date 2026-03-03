import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

def plot_live_metrics():
    # Force the base directory to be the parent of the 'scripts' folder
    base_dir = Path(__file__).resolve().parents[1]
    log_base_dir = base_dir / "models" / "logs" / "lulc_decoder"
    
    print(f"🚀 Script started. Looking for logs in: {log_base_dir}")
    
    if not log_base_dir.exists():
        print(f"❌ Error: The directory {log_base_dir} does not exist!")
        return

    # 1. Find the most recent log directory
    versions = [d for d in log_base_dir.iterdir() if d.is_dir() and d.name.startswith("version_")]
    
    if not versions:
        print(f"❌ Error: No 'version_x' folders found inside {log_base_dir}")
        return
        
    latest_version = max(versions, key=lambda d: int(d.name.split("_")[1]))
    metrics_file = latest_version / "metrics.csv"
    
    print(f"📂 Checking latest version: {latest_version.name}")
    
    if not metrics_file.exists():
        print(f"❌ Error: metrics.csv not found in {latest_version.name}. The model might not have saved its first log yet (needs ~10-20 steps).")
        return
        
    # 2. Load and clean the CSV data
    print(f"📈 Loading data from {metrics_file}...")
    df = pd.read_csv(metrics_file)
    
    if df.empty:
        print("Empty CSV. Waiting for more data...")
        return

    # Grouping by epoch to align train and val metrics
    epoch_df = df.groupby('epoch').mean().reset_index()

    # 3. Set up the plotting
    plt.figure(figsize=(14, 6))
    
    # --- Plot 1: Loss ---
    plt.subplot(1, 2, 1)
    if 'train_loss' in epoch_df.columns:
        plt.plot(epoch_df['epoch'], epoch_df['train_loss'], label='Train Loss', color='green', marker='.')
    if 'val_loss' in epoch_df.columns:
        plt.plot(epoch_df['epoch'], epoch_df['val_loss'], label='Val Loss', color='red', marker='.')
    plt.title("Loss Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # --- Plot 2: mIoU ---
    plt.subplot(1, 2, 2)
    if 'train_mIoU' in epoch_df.columns:
        plt.plot(epoch_df['epoch'], epoch_df['train_mIoU'], label='Train mIoU', color='blue', marker='.')
    if 'val_mIoU' in epoch_df.columns:
        plt.plot(epoch_df['epoch'], epoch_df['val_mIoU'], label='Val mIoU', color='purple', marker='.')
    plt.title("mIoU Performance")
    plt.xlabel("Epoch")
    plt.ylabel("mIoU Score")
    plt.legend()

    # 4. Save the plot
    out_dir = base_dir / "models" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "live_learning_curves.png"
    
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"✅ Success! Plot saved to: {out_path}")

if __name__ == "__main__":
    plot_live_metrics()