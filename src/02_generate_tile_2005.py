import os
import gc
import time
import numpy as np
import pystac_client
import planetary_computer
import odc.stac
import rioxarray
import warnings
from tqdm import tqdm
from pyproj import Transformer
from dask.distributed import Client, LocalCluster

warnings.filterwarnings("ignore")

# --- ENCODING UTILITIES ---
def encode_scalar(x):
    return np.array([np.sin(2 * np.pi * x), np.cos(2 * np.pi * x)], dtype=np.float32)

def normalize_latlon(lat, lon):
    lat_norm = (lat + 90) / 180.0
    lon_norm = (lon + 180) / 360.0
    return lat_norm, lon_norm

def _item_month(item):
    if item.datetime is None:
        return None
    return int(item.datetime.month)

def _item_cloud(item):
    return float(item.properties.get("eo:cloud_cover", 100.0))

def _item_sensor_group(item):
    platform = str(item.properties.get("platform", "")).lower()
    instruments = item.properties.get("instruments", [])
    if not isinstance(instruments, (list, tuple)):
        instruments = [instruments]
    instruments_text = " ".join(str(v).lower() for v in instruments)
    
    if "landsat-5" in platform or ("tm" in instruments_text and "etm" not in instruments_text):
        return "TM"
    if ("landsat-7" in platform) or ("etm" in instruments_text):
        return "ETM+"
    return "OTHER"

def _bbox_overlaps(a, b):
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])

def _nodata_fraction(pixels, fill_dn_high, fill_dn_tol):
    nan_mask = np.any(np.isnan(pixels), axis=0)
    zero_dn_mask = np.all(pixels == 0, axis=0)
    high_fill_mask = np.all(pixels >= (fill_dn_high - fill_dn_tol), axis=0)
    return float(np.mean(nan_mask | zero_dn_mask | high_fill_mask))

BAND_MAP = {"red": "red", "green": "green", "blue": "blue", "nir08": "nir08"}

# --- FIXED CONFIGURATION FOR 2005 ---
CONFIG = {
    2005: {
        "col": "landsat-c2-l2",
        "date": "2004-01-01/2005-12-31", # Extended window to guarantee enough scenes for fusion
        "cloud_max": 30.0,
        "max_items": 5000,
        # CRITICAL FIX: Allow both TM and ETM+ to enable cross-sensor gap filling 
        "sensor_groups": ["TM", "ETM+"], 
        "apply_month_filter": False,
        "allowed_months": (1, 2, 3, 4, 5, 6, 10, 11, 12),
        "max_candidates_per_tile": 5, # We need multiple scenes to calculate a clean median
        "max_nodata_fraction": 0.15,  # Stricter threshold since we are gap-filling
        "fill_dn_high": 65535.0,
        "fill_dn_high_tol": 1.0,
    }
}

def generate_tiles():
    # Hardware Limits: Safe for 16GB RAM
    cluster = LocalCluster(n_workers=3, threads_per_worker=1, memory_limit="4GB")
    client = Client(cluster)
    print(f"Dask Dashboard: {client.dashboard_link}")

    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    total_saved = 0
    for year, meta in CONFIG.items():
        print(f"\n[INFO] Starting Year {year} | sensors={meta['sensor_groups']}")

        mask_path = f"../data/processed/lulc_masks/uk_{year}_30m.tif"
        if not os.path.exists(mask_path):
            print(f"[WARN] Mask not found: {mask_path}")
            continue

        mask = rioxarray.open_rasterio(mask_path)
        transformer = Transformer.from_crs(mask.rio.crs, "EPSG:4326", always_xy=True)
        out_dir = f"../data/dataset/{year}"
        os.makedirs(out_dir, exist_ok=True)

        # Pre-fetch candidate pool for the whole state
        mask_bbox_wgs84 = mask.rio.transform_bounds("EPSG:4326")
        search = catalog.search(
            collections=[meta["col"]],
            bbox=mask_bbox_wgs84,
            datetime=meta["date"],
            max_items=int(meta["max_items"]),
        )
        items_raw = list(search.items())
        
        # Filter Pipeline
        sensor_groups_set = set(meta["sensor_groups"])
        valid_items = [
            item for item in items_raw
            if item.bbox and len(item.bbox) >= 4
            and _item_sensor_group(item) in sensor_groups_set
            and _item_cloud(item) <= float(meta["cloud_max"])
        ]
        
        print(f"[INFO] Candidates ready for Tiling: {len(valid_items)} (Filtered from {len(items_raw)})")

        ny, nx = mask.sizes["y"], mask.sizes["x"]
        count_saved = 0
        sign_cache = {}

        for i, y in enumerate(tqdm(range(0, ny - 256, 256), desc=f"Year {year}")):
            for j, x in enumerate(range(0, nx - 256, 256)):
                lbl_tile = mask.isel(y=slice(y, y + 256), x=slice(x, x + 256))
                
                # Skip tiles outside the state boundary
                if not np.any(lbl_tile.values > 0):
                    continue

                tile_bbox_wgs84 = lbl_tile.rio.transform_bounds("EPSG:4326")
                overlap = [item for item in valid_items if _bbox_overlaps(tile_bbox_wgs84, item.bbox)]
                
                if not overlap:
                    continue

                # Get top N clearest scenes for this specific tile
                candidates = sorted(overlap, key=_item_cloud)[: int(meta["max_candidates_per_tile"])]
                
                signed_items = []
                for item in candidates:
                    if item.id not in sign_cache:
                        sign_cache[item.id] = planetary_computer.sign(item)
                    signed_items.append(sign_cache[item.id])

                try:
                    tile_geobox = lbl_tile.odc.geobox
                    
                    # 1. Load multiple scenes into a Time-Series array
                    ds_stack = odc.stac.load(
                        signed_items,
                        geobox=tile_geobox,
                        bands=list(BAND_MAP.values()),
                        resampling="bilinear",
                        fail_on_error=False,
                    )

                    # 2. Mask out 65535 (Fill) and 0 (SLC-off Gaps) so they are ignored by median
                    valid_data = ds_stack.where((ds_stack > 0) & (ds_stack < float(meta["fill_dn_high"])))
                    
                    # 3. Temporal Composite: Median across the time axis (Fills 95% of stripes)
                    ds_median = valid_data.median(dim="time")
                    
                    # 4. Spatial Interpolation: Fill any remaining micro-gaps smoothly.
                    # xarray requires interpolation index to be monotonic increasing.
                    x_desc = bool(ds_median.x.values[0] > ds_median.x.values[-1])
                    y_desc = bool(ds_median.y.values[0] > ds_median.y.values[-1])

                    ds_interp = ds_median
                    if x_desc:
                        ds_interp = ds_interp.sortby("x")
                    if y_desc:
                        ds_interp = ds_interp.sortby("y")

                    ds_interp = ds_interp.interpolate_na(dim="x", method="linear", limit=10)
                    ds_interp = ds_interp.interpolate_na(dim="y", method="linear", limit=10)

                    if x_desc:
                        ds_interp = ds_interp.sortby("x", ascending=False)
                    if y_desc:
                        ds_interp = ds_interp.sortby("y", ascending=False)

                    ds_filled = ds_interp
                    
                    # Compute the final dense array and replace any lingering NaNs with 0
                    pixels = ds_filled.fillna(0).to_array().compute().values

                    nodata_frac = _nodata_fraction(
                        pixels,
                        fill_dn_high=float(meta["fill_dn_high"]),
                        fill_dn_tol=float(meta["fill_dn_high_tol"]),
                    )
                    
                    if nodata_frac > float(meta["max_nodata_fraction"]):
                        continue
                        
                    if np.nanmax(pixels) <= 0:
                        continue

                    y_center = float(lbl_tile.y.values[128])
                    x_center = float(lbl_tile.x.values[128])
                    lon, lat = transformer.transform(x_center, y_center)
                    lat_norm, lon_norm = normalize_latlon(lat, lon)

                    np.savez_compressed(
                        f"{out_dir}/cube_{count_saved}.npz",
                        image=pixels.astype(np.float32),
                        mask=lbl_tile.values.squeeze().astype(np.uint8),
                        latlon=np.concatenate([encode_scalar(lat_norm), encode_scalar(lon_norm)]),
                        year=year,
                    )

                    count_saved += 1
                    total_saved += 1

                except Exception as e:
                    print(f"[ERROR] tile={i}_{j} | {e}")
                    continue
                finally:
                    gc.collect()

        print(f"[INFO] Year {year} complete. Saved {count_saved} gap-free cubes.")

    print(f"\nFinished. Total cubes saved: {total_saved}")
    client.close()
    cluster.close()

if __name__ == "__main__":
    generate_tiles()
