import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.visualization.lulc_classes import LULC_CLASS_MAP, class_colors


def downsample_for_plot(arr: np.ndarray, max_side: int) -> tuple[np.ndarray, int]:
    h, w = arr.shape
    scale = max(h, w) / float(max_side)
    if scale <= 1.0:
        return arr, 1
    step = int(np.ceil(scale))
    return arr[::step, ::step], step


def run(args):
    pred_tif = Path(args.pred_tif)
    if not pred_tif.exists():
        raise FileNotFoundError(f"Prediction tif not found: {pred_tif}")

    out_png = Path(args.out_png) if args.out_png else pred_tif.with_suffix("").with_name(pred_tif.stem + "_viz.png")
    out_csv = Path(args.out_csv) if args.out_csv else pred_tif.with_suffix("").with_name(pred_tif.stem + "_class_stats.csv")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(pred_tif) as ds:
        pred = ds.read(1)
        crs = ds.crs
        transform = ds.transform

    classes, counts = np.unique(pred, return_counts=True)
    total = int(pred.size)

    rows = []
    for cid, cnt in zip(classes, counts):
        cid_i = int(cid)
        lvl1, lvl2 = LULC_CLASS_MAP.get(cid_i, ("Unknown", "Unknown"))
        rows.append(
            {
                "ID": cid_i,
                "Level-I": lvl1,
                "Level-II": lvl2,
                "Pixels": int(cnt),
                "Percent": round(100.0 * int(cnt) / total, 4),
            }
        )
    stats_df = pd.DataFrame(rows).sort_values("Pixels", ascending=False).reset_index(drop=True)
    stats_df.to_csv(out_csv, index=False)

    plot_arr, step = downsample_for_plot(pred, max_side=args.max_side)
    cmap = ListedColormap(class_colors)
    norm = BoundaryNorm(np.arange(-0.5, len(class_colors) + 0.5, 1), cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=130, gridspec_kw={"width_ratios": [2.2, 1.2]})
    ax_map, ax_bar = axes

    im = ax_map.imshow(plot_arr, cmap=cmap, norm=norm, interpolation="nearest")
    title = args.title if args.title else f"Predicted LULC: {pred_tif.name}"
    ax_map.set_title(title, fontsize=13, fontweight="bold")
    ax_map.set_axis_off()

    present_ids = [int(c) for c in classes if int(c) in LULC_CLASS_MAP]
    present_ids = sorted(present_ids)
    cbar = plt.colorbar(im, ax=ax_map, fraction=0.035, pad=0.02, ticks=present_ids)
    cbar.ax.set_yticklabels([f"{cid}: {LULC_CLASS_MAP[cid][1]}" for cid in present_ids], fontsize=8)
    cbar.ax.set_title("Class", fontsize=9)

    top_n = min(args.top_n, len(stats_df))
    top_df = stats_df.head(top_n).iloc[::-1]  # reverse for horizontal bars
    bar_labels = [f"{int(r['ID'])}: {r['Level-II']}" for _, r in top_df.iterrows()]
    bar_vals = top_df["Percent"].values
    bar_colors = [class_colors[int(r["ID"])] if int(r["ID"]) < len(class_colors) else "#999999" for _, r in top_df.iterrows()]
    ax_bar.barh(bar_labels, bar_vals, color=bar_colors)
    ax_bar.set_xlabel("Coverage (%)")
    ax_bar.set_title(f"Top {top_n} classes by area")
    ax_bar.grid(axis="x", alpha=0.25)

    meta_text = (
        f"Raster shape: {pred.shape[0]} x {pred.shape[1]}\n"
        f"CRS: {crs}\n"
        f"Downsample step for plot: {step}\n"
        f"Unique classes: {len(classes)}\n"
        f"Stats CSV: {out_csv.name}"
    )
    ax_bar.text(0.01, -0.16, meta_text, transform=ax_bar.transAxes, fontsize=8, va="top")

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Plot saved: {out_png.resolve()}")
    print(f"✅ Class stats saved: {out_csv.resolve()}")
    print("\nTop classes:")
    print(stats_df.head(min(15, len(stats_df))).to_string(index=False))


def build_argparser():
    p = argparse.ArgumentParser(description="Plot predicted LULC GeoTIFF using lulc_classes.py legend.")
    p.add_argument("--pred-tif", required=True, help="Path to predicted LULC GeoTIFF.")
    p.add_argument("--out-png", default=None, help="Output PNG path (default: <pred>_viz.png).")
    p.add_argument("--out-csv", default=None, help="Output class stats CSV path (default: <pred>_class_stats.csv).")
    p.add_argument("--max-side", type=int, default=2000, help="Max side length for visualization downsample.")
    p.add_argument("--top-n", type=int, default=12, help="Top N classes to show in bar chart.")
    p.add_argument("--title", default=None, help="Optional custom plot title.")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
