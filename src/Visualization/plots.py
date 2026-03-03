import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_results(log_dir, final_cm, num_classes, ignore_index, class_map, out_dir="../models/metrics/"):
    print(f"\n📊 Initializing Metric Visualization Engine...")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    metrics_file = Path(log_dir) / "metrics.csv"
    if metrics_file.exists():
        print(f"   ➤ Found metrics log at: {metrics_file.resolve()}")
        df = pd.read_csv(metrics_file)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        epoch_df = df.groupby('epoch').mean().reset_index()
        
        ax1.plot(epoch_df['epoch'], epoch_df['train_loss'], label='Train Loss')
        ax1.plot(epoch_df['epoch'], epoch_df['val_loss'], label='Val Loss')
        ax1.set_title("Cross-Entropy Loss"); ax1.legend()
        
        ax2.plot(epoch_df['epoch'], epoch_df['train_mIoU'], label='Train mIoU')
        ax2.plot(epoch_df['epoch'], epoch_df['val_mIoU'], label='Val mIoU')
        ax2.set_title("Mean Intersection over Union (mIoU)"); ax2.legend()
        
        plt.tight_layout()
        lc_path = out_path / "learning_curves.png"
        plt.savefig(lc_path, dpi=300)
        print(f"   ✅ Saved Learning Curves to: {lc_path}")
        plt.close()
    else:
        print(f"   ⚠️ Could not find metrics.csv at {log_dir}")

    print("   ➤ Generating Validation Confusion Matrix...")
    valid_classes = [i for i in range(num_classes) if i != ignore_index and np.sum(final_cm[i]) > 0]
    cm_filtered = final_cm[np.ix_(valid_classes, valid_classes)]
    class_names = [class_map[i][1] for i in valid_classes]
    
    cm_normalized = cm_filtered.astype('float') / cm_filtered.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Validation Set Confusion Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    cm_path = out_path / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    print(f"   ✅ Saved Confusion Matrix to: {cm_path}")
    plt.close()
    print("🎉 All visualizations generated successfully!")