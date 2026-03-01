import os
import gc
import numpy as np
import pystac_client
import planetary_computer
import odc.stac
import rioxarray
import warnings
from tqdm.auto import tqdm
from pyproj import Transformer
from dask.distributed import Client, LocalCluster

warnings.filterwarnings("ignore")

# ==========================================
# 1. ENCODING UTILITIES
# ==========================================
def encode_scalar(x):
    return np.array([np.sin(2*np.pi*x), np.cos(2*np.pi*x)], dtype=np.float32)

def normalize_latlon(lat, lon):
    lat_norm = (lat + 90) / 180.0
    lon_norm = (lon + 180) / 360.0
    return lat_norm, lon_norm

# ==========================================
# 2. CONFIGURATION & THRESHOLDS
# ==========================================
CONFIG = {
    # 1985: {"col": "landsat-c2-l2", "date": "1987-01-01/1989-12-31"},
    1995: {"col": "landsat-c2-l2", "date": "1994-01-01/1996-12-31"},
    2005: {"col": "landsat-c2-l2", "date": "2004-01-01/2006-12-31"}
}

BAND_MAP = {
    "red": "red", "green": "green", "blue": "blue", 
    "nir08": "nir08", "swir16": "swir16", "swir22": "swir22"
}

ALLOWED_MONTHS = {1, 2, 3, 4, 5, 6, 10, 11, 12} 

# --- THRESHOLDS ---
MIN_LULC_COVERAGE = 50.0   # At least 50% of the tile must contain valid LULC classes
MAX_NODATA_PERCENT = 35.0  # Max NaN/Null pixels (edge of swath)
MAX_BLACK_PERCENT = 50.0   # Max true black pixels (0,0,0,0,0,0)
MAX_CLOUD_BRIGHTNESS = 15000 # Average Blue band value indicating local cloud/haze
MAX_ALTERNATES = 5         # Scenes to test before giving up on a tile

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def verify_and_generate():
    cluster = LocalCluster(n_workers=3, threads_per_worker=1, memory_limit='4GB')
    client = Client(cluster)
    print(f"🚀 Dask Dashboard: {client.dashboard_link}")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    for year, meta in CONFIG.items():
        print(f"\n{'='*80}\n📅 STARTING TARGET YEAR: {year}\n{'='*80}")
        
        mask_path = f"../data/processed/lulc_masks/uk_{year}_30m.tif"
        if not os.path.exists(mask_path):
            print(f"⚠️ [SKIP] Mask not found: {mask_path}")
            continue

        mask = rioxarray.open_rasterio(mask_path)
        transformer = Transformer.from_crs(mask.rio.crs, "EPSG:4326", always_xy=True)
        out_dir = f"../data/dataset/{year}"
        os.makedirs(out_dir, exist_ok=True)

        ny, nx = mask.sizes['y'], mask.sizes['x']
        count_saved = 0

        for i in tqdm(range(0, ny-256, 256), desc=f"Decade {year} Processing"):
            for j in range(0, nx-256, 256):
                
                lbl_tile = mask.isel(y=slice(i, i+256), x=slice(j, j+256))
                valid_mask_pixels = np.sum(lbl_tile.values > 0)
                valid_mask_pct = (valid_mask_pixels / lbl_tile.values.size) * 100
                
                # Silently skip entirely empty bounding box corners to save console space
                if valid_mask_pct == 0:
                    continue

                # If the tile touches the state but doesn't meet the 50% threshold, log it and skip
                if valid_mask_pct < MIN_LULC_COVERAGE:
                    tqdm.write(f"⏭️ SKIP TILE [{i},{j}]: Low LULC coverage ({valid_mask_pct:.1f}%)")
                    continue

                tqdm.write(f"\n📍 TILE [{i},{j}] | Valid LULC: {valid_mask_pct:.1f}% | Searching Catalog...")

                tile_bbox = lbl_tile.rio.transform_bounds("EPSG:4326")
                search = catalog.search(
                    collections=[meta["col"]], 
                    bbox=tile_bbox, 
                    datetime=meta["date"],
                    query={"eo:cloud_cover": {"lte": 20.0}}
                )
                
                items = [it for it in search.items() if it.datetime.month in ALLOWED_MONTHS]
                if not items: 
                    tqdm.write(f"   ⚠️ SKIP: No satellite scenes found in allowed months.")
                    continue

                sorted_items = sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))
                
                valid_pixels = None
                best_item_used = None

                # Test alternate scenes and print detailed logs for each attempt
                for idx, item in enumerate(sorted_items[:MAX_ALTERNATES]):
                    props = item.properties
                    sat_name = props.get('platform', 'Unknown')
                    # Join sensors in case there are multiple (e.g., ['oli', 'tirs'])
                    sensor_name = ", ".join(props.get('instruments', ['Unknown']))
                    acq_date = item.datetime.date()
                    catalog_cloud = props.get("eo:cloud_cover", 100)

                    tqdm.write(f"   🔄 Test {idx+1}/{MAX_ALTERNATES}: Sat: {sat_name} ({sensor_name}) | Date: {acq_date} | Catalog Cloud: {catalog_cloud:.1f}%")

                    try:
                        ds_tile = odc.stac.load(
                            [item],
                            geobox=lbl_tile.odc.geobox,
                            bands=list(BAND_MAP.values()),
                            resampling="bilinear"
                        ).squeeze().compute()
                        
                        pixels = ds_tile.to_array().values

                        # Calculate Quality Metrics
                        nan_mask = np.isnan(pixels).any(axis=0)
                        nan_pct = (np.sum(nan_mask) / nan_mask.size) * 100
                        
                        black_mask = (pixels == 0).all(axis=0)
                        black_pct = (np.sum(black_mask) / black_mask.size) * 100

                        blue_mean = np.nanmean(pixels[2])

                        # Log the specific metrics that caused a pass/fail
                        if nan_pct <= MAX_NODATA_PERCENT and black_pct <= MAX_BLACK_PERCENT and blue_mean < MAX_CLOUD_BRIGHTNESS:
                            tqdm.write(f"      ✅ PASS: NaN={nan_pct:.1f}% | Black={black_pct:.1f}% | BlueMean={blue_mean:.1f}")
                            valid_pixels = pixels
                            best_item_used = item
                            break 
                        else:
                            tqdm.write(f"      ❌ FAIL: NaN={nan_pct:.1f}% | Black={black_pct:.1f}% | BlueMean={blue_mean:.1f}")
                            
                    except Exception as e:
                        tqdm.write(f"      ⚠️ ERROR fetching scene: {str(e)[:50]}...")
                        continue
                    finally:
                        gc.collect()

                if valid_pixels is None:
                    tqdm.write(f"   ⏩ SKIP: Exhausted all {MAX_ALTERNATES} alternate scenes. None passed filters.")
                    continue

                # Save Sequence
                try:
                    dt = best_item_used.datetime
                    week_n, hour_n = dt.isocalendar()[1]/52.0, dt.hour/24.0
                    y_mid, x_mid = float(lbl_tile.y.values[128]), float(lbl_tile.x.values[128])
                    lon, lat = transformer.transform(x_mid, y_mid)
                    lat_n, lon_n = normalize_latlon(lat, lon)

                    np.savez_compressed(
                        f"{out_dir}/cube_{count_saved}.npz",
                        pixels=valid_pixels.astype(np.float32),
                        mask=lbl_tile.values.squeeze().astype(np.uint8),
                        lat_norm=encode_scalar(lat_n),
                        lon_norm=encode_scalar(lon_n),
                        week_norm=encode_scalar(week_n),
                        hour_norm=encode_scalar(hour_n)
                    )
                    
                    tqdm.write(f"   💾 SUCCESS -> Saved to cube_{count_saved}.npz\n")
                    count_saved += 1
                        
                except Exception as e:
                    tqdm.write(f"   ❌ FATAL ERROR Saving Cube: {e}\n")
                finally:
                    gc.collect()

        print(f"\n🎉 DECADE {year} COMPLETED! Successfully generated {count_saved} cubes.\n")

    client.close()
    cluster.close()

if __name__ == "__main__":
    verify_and_generate()