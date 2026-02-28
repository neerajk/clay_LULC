import os
import gc
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
    return np.array([np.sin(2*np.pi*x), np.cos(2*np.pi*x)], dtype=np.float32)

def normalize_latlon(lat, lon):
    lat_norm = (lat + 90) / 180.0
    lon_norm = (lon + 180) / 360.0
    return lat_norm, lon_norm

# Landsat C2 L2 Asset Mapping
BAND_MAP = {"red": "red", "green": "green", "blue": "blue", "nir08": "nir08"}

# Keep scenes in the low-cloud range requested by user.
CLOUD_MAX_PERCENT = 20.0

# Seasonal windows used in the source workflow:
# Jan-Mar (winter), Apr-Jun (summer), Oct-Dec (post-monsoon/winter).
ALLOWED_MONTHS_1995 = (1, 2, 3, 4, 5, 6, 10, 11, 12)

# Skip any tile that contains imagery no-data (black DN=0 across all bands).
SKIP_IF_ANY_ZERO_DN_PIXEL = True

CONFIG = {
    1995: {
        "col": "landsat-c2-l2",
        "date": "1994-01-01/1995-12-31",
        "allowed_months": ALLOWED_MONTHS_1995,
        "cloud_max": CLOUD_MAX_PERCENT
    }
}

def _item_month(item):
    if item.datetime is None:
        return None
    return int(item.datetime.month)


def _item_cloud(item):
    return float(item.properties.get("eo:cloud_cover", 100.0))


def verify_and_generate():
    # 16GB RAM Safety: 2 workers, 4GB each
    cluster = LocalCluster(n_workers=2, threads_per_worker=1, memory_limit='4GB')
    client = Client(cluster)
    print(f"Dask Dashboard: {client.dashboard_link}")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )

    total_saved = 0
    for year, meta in CONFIG.items():
        allowed_months = set(meta.get("allowed_months", ()))
        cloud_max = float(meta.get("cloud_max", CLOUD_MAX_PERCENT))
        print(
            f"[INFO] Year {year}: cloud <= {cloud_max:.1f}% | "
            f"months={sorted(allowed_months)}"
        )
        last_logged_scene_id = None

        mask_path = f"../data/processed/lulc_masks/uk_{year}_30m.tif"
        if not os.path.exists(mask_path):
            print(f"Mask not found: {mask_path}")
            continue

        # STEP 1: Load LULC mask as a spatial anchor
        mask = rioxarray.open_rasterio(mask_path)
        print(f"\n[DEBUG] Global Mask Bounds: {mask.rio.bounds()}")
        
        transformer = Transformer.from_crs(mask.rio.crs, "EPSG:4326", always_xy=True)
        out_dir = f"../data/dataset/{year}"
        os.makedirs(out_dir, exist_ok=True)

        ny, nx = mask.sizes['y'], mask.sizes['x']
        count_saved = 0

        # Tiling Loop: Break down LULC into 256x256 tiles
        for i, y in enumerate(tqdm(range(0, ny-256, 256), desc=f"Year {year}")):
            for j, x in enumerate(range(0, nx-256, 256)):
                
                # STEP 2: Extract 256x256 LULC tile
                lbl_tile = mask.isel(y=slice(y, y+256), x=slice(x, x+256))
                
                # Only proceed if the tile contains data (>0)
                if np.any(lbl_tile.values > 0):
                    
                    # STEP 3: Find matching scene for ONLY this tile's extent
                    tile_bbox_wgs84 = lbl_tile.rio.transform_bounds("EPSG:4326")
                    search = catalog.search(
                        collections=[meta["col"]], 
                        bbox=tile_bbox_wgs84, 
                        datetime=meta["date"],
                        max_items=50,
                        query={"eo:cloud_cover": {"lte": cloud_max}}
                    )
                    items = list(search.items())
                    
                    if not items:
                        continue # Skip if no satellite scenes cover this specific chip

                    items = [
                        item for item in items
                        if _item_cloud(item) <= cloud_max
                        and (_item_month(item) in allowed_months)
                    ]

                    if not items:
                        continue

                    # Pick the clearest candidate and sign only that scene.
                    best_item = min(
                        items,
                        key=_item_cloud
                    )
                    best_cloud = _item_cloud(best_item)
                    best_dt = (
                        best_item.datetime.isoformat()
                        if best_item.datetime is not None
                        else str(best_item.properties.get("datetime", ""))
                    )
                    scene_id = str(best_item.id)
                    sensor = str(best_item.properties.get("platform", "unknown"))
                    instruments = best_item.properties.get("instruments", [])
                    if isinstance(instruments, (list, tuple)):
                        instruments = ",".join(str(v) for v in instruments) if instruments else "unknown"
                    else:
                        instruments = str(instruments)
                    wrs_path = best_item.properties.get("landsat:wrs_path", "unknown")
                    wrs_row = best_item.properties.get("landsat:wrs_row", "unknown")
                    collection_cat = best_item.properties.get("landsat:collection_category", "unknown")
                    processing_level = best_item.properties.get("landsat:processing_level", "unknown")

                    # Print once when switching scenes to keep logs informative but concise.
                    if scene_id != last_logged_scene_id:
                        print(
                            "[SCENE] "
                            f"id={scene_id} | sensor={sensor} | instruments={instruments} | "
                            f"date={best_dt} | cloud={best_cloud:.2f}% | "
                            f"WRS={wrs_path}/{wrs_row} | collection={collection_cat} | "
                            f"level={processing_level}"
                        )
                        last_logged_scene_id = scene_id
                    signed_item = planetary_computer.sign(best_item)
                    
                    try:
                        # STEP 4: Load matching satellite tile using GeoBox
                        # This forces the satellite pixels to align perfectly with the mask tile
                        tile_geobox = lbl_tile.odc.geobox
                        
                        ds_tile = odc.stac.load(
                            [signed_item],
                            geobox=tile_geobox,
                            bands=list(BAND_MAP.values()),
                            resampling="bilinear", # Smooths re-projection
                            fail_on_error=False
                        ).squeeze().compute()
                        
                        pixels = ds_tile.to_array().values # Shape: [Bands, 256, 256]

                        # Reject tiles that contain no-data imagery pixels.
                        # A no-data pixel is treated as all-band DN=0 (black) or NaN.
                        nan_mask = np.any(np.isnan(pixels), axis=0)
                        zero_dn_mask = np.all(pixels == 0, axis=0)
                        if SKIP_IF_ANY_ZERO_DN_PIXEL and np.any(nan_mask | zero_dn_mask):
                            continue

                        # VERIFICATION PRINT (Every 50 tiles)
                        if count_saved % 50 == 0:
                            print(f"\n[VERIFY] Cube {count_saved}")
                            print(f"Mask Tile Shape: {lbl_tile.shape}")
                            print(f"Sat Tile Shape:  {pixels.shape}")
                            print(f"Sat Max Pixel:   {np.nanmax(pixels)}")
                            print(f"Scene Date:      {best_dt}")
                            print(f"Cloud Cover:     {best_cloud:.2f}%")
                            
                        # Final Save: Generate .npz cube
                        if np.nanmax(pixels) > 0:
                            # Coordinate Metadata
                            y_center = float(lbl_tile.y.values[128])
                            x_center = float(lbl_tile.x.values[128])
                            lon, lat = transformer.transform(x_center, y_center)
                            lat_norm, lon_norm = normalize_latlon(lat, lon)

                            np.savez_compressed(
                                f"{out_dir}/cube_{count_saved}.npz",
                                image=pixels.astype(np.float32),
                                mask=lbl_tile.values.squeeze().astype(np.uint8),
                                latlon=np.concatenate([encode_scalar(lat_norm), encode_scalar(lon_norm)]),
                                year=year
                            )
                            count_saved += 1
                            total_saved += 1
                            
                    except Exception as e:
                        print(f"[ERROR] Tile {i}_{j} failed: {e}")
                        continue
                    finally:
                        gc.collect() # Immediate memory cleanup

    print(f"\n🎉 Finished. Total valid cubes: {total_saved}")
    client.close()
    cluster.close()

if __name__ == "__main__":
    verify_and_generate()
