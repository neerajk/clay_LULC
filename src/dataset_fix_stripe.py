import os
import gc
import shutil
import numpy as np
import pystac_client
import planetary_computer
import odc.stac
import rioxarray
import warnings
import pandas as pd
from tqdm.auto import tqdm
from pyproj import Transformer
from dask.distributed import Client, LocalCluster
from pathlib import Path

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION & UTILITIES
# ==========================================
YEAR = 2005
DATA_DIR = Path(f'../data/dataset/{YEAR}/')
MASK_PATH = f"../data/processed/lulc_masks/uk_{YEAR}_30m.tif"
DATE_RANGE = "2004-01-01/2009-12-31"

# --- NEW DIRECTORY FOR EXTREME STRIPES ---
STRIPPED_DIR = Path(f'../data/dataset/{YEAR}_stripped_l7/')
os.makedirs(STRIPPED_DIR, exist_ok=True)

# --- THRESHOLDS ---
DEAD_PIXEL_THRESHOLD = 0.02       # 2% loss triggers the search for a replacement
EXTREME_STRIPE_THRESHOLD = 22.0   # >10% loss means the cube gets moved if no replacement is found
MIN_LULC_COVERAGE = 50.0

BAND_MAP = {"red": "red", "green": "green", "blue": "blue", "nir08": "nir08", "swir16": "swir16", "swir22": "swir22"}
ALLOWED_MONTHS = {1, 2, 3, 4, 5, 6, 10, 11, 12} 
MAX_NODATA_PERCENT = 35.0  
MAX_BLACK_PERCENT = 50.0   
MAX_CLOUD_BRIGHTNESS = 15000 
MAX_ALTERNATES = 5

def encode_scalar(x):
    return np.array([np.sin(2*np.pi*x), np.cos(2*np.pi*x)], dtype=np.float32)

def normalize_latlon(lat, lon):
    lat_norm = (lat + 90) / 180.0
    lon_norm = (lon + 180) / 360.0
    return lat_norm, lon_norm

def decode_latlon(lat_encoded, lon_encoded):
    def decode_scalar_pair(pair):
        angle = np.arctan2(pair[0], pair[1])
        return (angle / (2 * np.pi)) % 1.0
    lat_norm_val = decode_scalar_pair(lat_encoded)
    lon_norm_val = decode_scalar_pair(lon_encoded)
    return lat_norm_val * 180.0 - 90.0, lon_norm_val * 360.0 - 180.0

# ==========================================
# 2. PHASE 1: IDENTIFY CORRUPTED CUBES
# ==========================================
def get_corrupted_targets():
    cube_files = list(DATA_DIR.glob('cube_*.npz'))
    print(f"🔍 PHASE 1: Scanning {len(cube_files)} cubes for Landsat 7 Striping...")
    
    targets = []
    for cube_path in tqdm(cube_files, desc="Detecting Stripes"):
        try:
            with np.load(cube_path) as data:
                img = data['pixels']
                lat_encoded, lon_encoded = data['lat_norm'], data['lon_norm']
                
            dead_pixels = np.all(img == 0, axis=0)
            dead_ratio = float(np.mean(dead_pixels))
            
            if dead_ratio > DEAD_PIXEL_THRESHOLD:
                lat, lon = decode_latlon(lat_encoded, lon_encoded)
                targets.append({
                    'filename': cube_path.name,
                    'path': cube_path,
                    'lat': lat,
                    'lon': lon,
                    'loss': dead_ratio * 100
                })
        except Exception:
            continue
            
    print(f"🚨 Found {len(targets)} striped cubes needing replacement.\n")
    return targets

# ==========================================
# 3. PHASE 2: SURGICAL REPLACEMENT
# ==========================================
def patch_striped_cubes():
    targets = get_corrupted_targets()
    if not targets:
        print("✅ No corrupted cubes found! Your dataset is perfectly clean.")
        return

    cluster = LocalCluster(n_workers=3, threads_per_worker=1, memory_limit='4GB')
    client = Client(cluster)
    print(f"🚀 Dask Dashboard: {client.dashboard_link}")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    print(f"🗺️ Loading Ground Truth Mask: {MASK_PATH}")
    mask = rioxarray.open_rasterio(MASK_PATH)
    transformer = Transformer.from_crs(mask.rio.crs, "EPSG:4326", always_xy=True)
    ny, nx = mask.sizes['y'], mask.sizes['x']

    patched_count = 0
    moved_count = 0
    retained_count = 0
    total_targets = len(targets)

    print(f"\n{'='*70}\n🛠️ PHASE 2: FETCHING LANDSAT 5 REPLACEMENTS\n{'='*70}")
    
    for i in tqdm(range(0, ny-256, 256), desc="Scanning Grid"):
        if (patched_count + moved_count + retained_count) >= total_targets:
            break 

        for j in range(0, nx-256, 256):
            lbl_tile = mask.isel(y=slice(i, i+256), x=slice(j, j+256))
            valid_mask_pct = (np.sum(lbl_tile.values > 0) / lbl_tile.values.size) * 100
            
            if valid_mask_pct < MIN_LULC_COVERAGE:
                continue

            y_mid, x_mid = float(lbl_tile.y.values[128]), float(lbl_tile.x.values[128])
            lon, lat = transformer.transform(x_mid, y_mid)

            matched_target = None
            for t in targets:
                if abs(t['lat'] - lat) < 0.001 and abs(t['lon'] - lon) < 0.001:
                    matched_target = t
                    break

            if matched_target:
                loss_val = matched_target['loss']
                tqdm.write(f"\n🎯 TARGET MATCHED: {matched_target['filename']} (Original Loss: {loss_val:.1f}%)")
                tqdm.write(f"   Searching STAC for Landsat 5 data...")

                tile_bbox = lbl_tile.rio.transform_bounds("EPSG:4326")
                
                search = catalog.search(
                    collections=['landsat-c2-l2'],
                    bbox=tile_bbox,
                    datetime=DATE_RANGE,
                    query={
                        'eo:cloud_cover': {'lt': 10}, 
                        'platform': {'in': ['landsat-5']} 
                    }
                )
                
                items = [it for it in search.items() if it.datetime.month in ALLOWED_MONTHS]
                
                valid_pixels, best_item_used = None, None

                if not items:
                    tqdm.write(f"   ⚠️ No valid scenes returned from catalog.")
                else:
                    sorted_items = sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))
                    
                    for idx, item in enumerate(sorted_items[:MAX_ALTERNATES]):
                        props = item.properties
                        tqdm.write(f"   🔄 Test {idx+1}/{MAX_ALTERNATES}: {props.get('platform')} | Date: {item.datetime.date()}")

                        try:
                            ds_tile = odc.stac.load(
                                [item],
                                geobox=lbl_tile.odc.geobox,
                                bands=list(BAND_MAP.values()),
                                resampling="bilinear"
                            ).squeeze().compute()
                            
                            pixels = ds_tile.to_array().values

                            nan_pct = (np.sum(np.isnan(pixels).any(axis=0)) / (256*256)) * 100
                            black_pct = (np.sum((pixels == 0).all(axis=0)) / (256*256)) * 100
                            blue_mean = np.nanmean(pixels[2])

                            if nan_pct <= MAX_NODATA_PERCENT and black_pct <= MAX_BLACK_PERCENT and blue_mean < MAX_CLOUD_BRIGHTNESS:
                                tqdm.write(f"      ✅ PASS: Clean Image Acquired!")
                                valid_pixels = pixels
                                best_item_used = item
                                break 
                            else:
                                # --- NEW DETAILED FAIL LOGGING ---
                                tqdm.write(f"      ❌ FAIL: NaN={nan_pct:.1f}% | Black={black_pct:.1f}% | BlueMean={blue_mean:.1f}")
                                
                        except Exception as e:
                            tqdm.write(f"      ⚠️ Fetch Error: {str(e)[:50]}")
                            continue
                        finally:
                            gc.collect()

                # ==========================================
                # THE FALLBACK & MOVE LOGIC
                # ==========================================
                if valid_pixels is not None:
                    # WE FOUND A REPLACEMENT! Overwrite it.
                    try:
                        dt = best_item_used.datetime
                        week_n, hour_n = dt.isocalendar()[1]/52.0, dt.hour/24.0
                        lat_n, lon_n = normalize_latlon(lat, lon)

                        np.savez_compressed(
                            matched_target['path'],
                            pixels=valid_pixels.astype(np.float32),
                            mask=lbl_tile.values.squeeze().astype(np.uint8),
                            lat_norm=encode_scalar(lat_n),
                            lon_norm=encode_scalar(lon_n),
                            week_norm=encode_scalar(week_n),
                            hour_norm=encode_scalar(hour_n)
                        )
                        tqdm.write(f"   💾 SUCCESS -> Replaced {matched_target['filename']} with clean L5 data!\n")
                        patched_count += 1
                        targets.remove(matched_target)
                    except Exception as e:
                        tqdm.write(f"   ❌ ERROR saving file: {e}")
                else:
                    # WE FAILED TO FIND A REPLACEMENT. Do we keep it or move it?
                    if loss_val > EXTREME_STRIPE_THRESHOLD:
                        dest_path = STRIPPED_DIR / matched_target['filename']
                        try:
                            shutil.move(str(matched_target['path']), str(dest_path))
                            tqdm.write(f"   🚚 MOVED: {matched_target['filename']} -> {STRIPPED_DIR.name}/ (Extreme Loss: {loss_val:.1f}% > {EXTREME_STRIPE_THRESHOLD}%)")
                            moved_count += 1
                        except Exception as e:
                            tqdm.write(f"   ❌ ERROR moving file: {e}")
                    else:
                        tqdm.write(f"   🛡️ RETAINED: {matched_target['filename']} kept in main dataset (Slight Loss: {loss_val:.1f}% <= {EXTREME_STRIPE_THRESHOLD}%)")
                        retained_count += 1
                    
                    targets.remove(matched_target) # Remove from active list either way

    print(f"\n🎉 PIPELINE COMPLETE!")
    print(f"   ✅ Successfully Patched: {patched_count}")
    print(f"   🚚 Moved to {STRIPPED_DIR.name}: {moved_count}")
    print(f"   🛡️ Retained in dataset: {retained_count}")
    
    client.close(); cluster.close()

if __name__ == "__main__":
    patch_striped_cubes()