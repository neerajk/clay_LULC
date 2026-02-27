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


def encode_scalar(x):
    return np.array([np.sin(2 * np.pi * x), np.cos(2 * np.pi * x)], dtype=np.float32)


def normalize_latlon(lat, lon):
    lat_norm = (lat + 90) / 180.0
    lon_norm = (lon + 180) / 360.0
    return lat_norm, lon_norm


BAND_MAP = {"red": "red", "green": "green", "blue": "blue", "nir08": "nir08"}

CLOUD_MAX_PERCENT = 20.0
ALLOWED_MONTHS_2005 = (1, 2, 3, 4, 5, 6, 10, 11, 12)
TARGET_PLATFORM = "landsat-7"
TARGET_INSTRUMENT_KEYWORDS = ("etm", "etm+")
SKIP_IF_ANY_ZERO_DN_PIXEL = True

CONFIG = {
    2005: {
        "col": "landsat-c2-l2",
        "date": "2004-01-01/2005-12-31",
        "allowed_months": ALLOWED_MONTHS_2005,
        "cloud_max": CLOUD_MAX_PERCENT,
    }
}



def _item_month(item):
    if item.datetime is None:
        return None
    return int(item.datetime.month)


def _item_cloud(item):
    return float(item.properties.get("eo:cloud_cover", 100.0))


def _item_sensor_ok(item):
    platform = str(item.properties.get("platform", "")).lower()
    instruments = item.properties.get("instruments", [])
    if not isinstance(instruments, (list, tuple)):
        instruments = [instruments]
    instruments_text = " ".join(str(v).lower() for v in instruments)
    return (TARGET_PLATFORM in platform) and any(
        key in instruments_text for key in TARGET_INSTRUMENT_KEYWORDS
    )


def _bbox_overlaps(a, b):
    # a/b: (minx, miny, maxx, maxy)
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def generate_tiles():
    cluster = LocalCluster(n_workers=2, threads_per_worker=1, memory_limit="4GB")
    client = Client(cluster)
    print(f"Dask Dashboard: {client.dashboard_link}")

    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    total_saved = 0
    for year, meta in CONFIG.items():
        allowed_months = set(meta.get("allowed_months", ()))
        cloud_max = float(meta.get("cloud_max", CLOUD_MAX_PERCENT))
        print(
            f"[INFO] Year {year}: sensor={TARGET_PLATFORM}/ETM+ | "
            f"cloud <= {cloud_max:.1f}% | months={sorted(allowed_months)}"
        )

        last_logged_scene_id = None
        mask_path = f"../data/processed/lulc_masks/uk_{year}_30m.tif"
        if not os.path.exists(mask_path):
            print(f"Mask not found: {mask_path}")
            continue

        mask = rioxarray.open_rasterio(mask_path)
        transformer = Transformer.from_crs(mask.rio.crs, "EPSG:4326", always_xy=True)
        out_dir = f"../data/dataset/{year}"
        os.makedirs(out_dir, exist_ok=True)

        # Fetch year candidates once, then select per tile locally.
        mask_bbox_wgs84 = mask.rio.transform_bounds("EPSG:4326")
        print("[INFO] Querying yearly candidate scenes once for full mask extent...")
        year_search = catalog.search(
            collections=[meta["col"]],
            bbox=mask_bbox_wgs84,
            datetime=meta["date"],
            max_items=500,
            query={"eo:cloud_cover": {"lte": cloud_max}},
        )
        year_items_raw = list(year_search.items())
        print(f"[INFO] STAC candidates fetched: {len(year_items_raw)}")

        year_items = [
            item
            for item in year_items_raw
            if _item_cloud(item) <= cloud_max
            and (_item_month(item) in allowed_months)
            and _item_sensor_ok(item)
            and item.bbox is not None
            and len(item.bbox) >= 4
        ]
        print(f"[INFO] Candidates after month/sensor/cloud filters: {len(year_items)}")
        if not year_items:
            print("[WARN] No valid yearly candidates found after filtering; skipping year.")
            continue

        sign_cache = {}

        ny, nx = mask.sizes["y"], mask.sizes["x"]
        count_saved = 0
        tiles_total = 0
        tiles_with_label = 0
        tiles_no_scene = 0
        tiles_filtered_out = 0
        tiles_skipped_nodata = 0
        tiles_skipped_nonpositive = 0
        tiles_failed = 0
        n_rows = len(range(0, ny - 256, 256))
        year_start = time.time()

        for i, y in enumerate(tqdm(range(0, ny - 256, 256), desc=f"Year {year}")):
            for j, x in enumerate(range(0, nx - 256, 256)):
                tiles_total += 1
                lbl_tile = mask.isel(y=slice(y, y + 256), x=slice(x, x + 256))
                if not np.any(lbl_tile.values > 0):
                    continue
                tiles_with_label += 1

                tile_bbox_wgs84 = lbl_tile.rio.transform_bounds("EPSG:4326")
                items = [
                    item for item in year_items
                    if _bbox_overlaps(tile_bbox_wgs84, item.bbox)
                ]
                if not items:
                    tiles_no_scene += 1
                    continue

                best_item = min(items, key=_item_cloud)
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

                if scene_id != last_logged_scene_id:
                    print(
                        "[SCENE] "
                        f"id={scene_id} | sensor={sensor} | instruments={instruments} | "
                        f"date={best_dt} | cloud={best_cloud:.2f}% | "
                        f"WRS={wrs_path}/{wrs_row} | collection={collection_cat} | "
                        f"level={processing_level}"
                    )
                    last_logged_scene_id = scene_id

                if scene_id not in sign_cache:
                    sign_cache[scene_id] = planetary_computer.sign(best_item)
                signed_item = sign_cache[scene_id]

                try:
                    tile_geobox = lbl_tile.odc.geobox
                    ds_tile = odc.stac.load(
                        [signed_item],
                        geobox=tile_geobox,
                        bands=list(BAND_MAP.values()),
                        resampling="bilinear",
                        fail_on_error=False,
                    ).squeeze().compute()

                    pixels = ds_tile.to_array().values

                    nan_mask = np.any(np.isnan(pixels), axis=0)
                    zero_dn_mask = np.all(pixels == 0, axis=0)
                    if SKIP_IF_ANY_ZERO_DN_PIXEL and np.any(nan_mask | zero_dn_mask):
                        tiles_skipped_nodata += 1
                        continue

                    if np.nanmax(pixels) <= 0:
                        tiles_skipped_nonpositive += 1
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

                    if count_saved % 25 == 0:
                        print(f"\n[VERIFY] Cube {count_saved}")
                        print(f"Mask Tile Shape: {lbl_tile.shape}")
                        print(f"Sat Tile Shape:  {pixels.shape}")
                        print(f"Sat Max Pixel:   {np.nanmax(pixels)}")

                    count_saved += 1
                    total_saved += 1

                except Exception as e:
                    tiles_failed += 1
                    print(f"[ERROR] Tile {i}_{j} failed: {e}")
                    continue
                finally:
                    gc.collect()

            if (i + 1) % 2 == 0:
                elapsed = time.time() - year_start
                print(
                    "[STATUS] "
                    f"row_block={i + 1}/{n_rows} | tiles_seen={tiles_total} | "
                    f"label_tiles={tiles_with_label} | cubes_saved={count_saved} | "
                    f"no_overlap_scene={tiles_no_scene} | nodata_skip={tiles_skipped_nodata} | "
                    f"failed={tiles_failed} | signed_scenes={len(sign_cache)} | "
                    f"elapsed_min={elapsed/60.0:.1f}"
                )

        print(f"[INFO] Year {year} complete. Saved {count_saved} cubes.")
        print(
            "[SUMMARY] "
            f"tiles_total={tiles_total} | with_label={tiles_with_label} | "
            f"no_scene={tiles_no_scene} | filtered_out={tiles_filtered_out} | "
            f"skipped_nodata={tiles_skipped_nodata} | skipped_nonpositive={tiles_skipped_nonpositive} | "
            f"failed={tiles_failed}"
        )

    print(f"\nFinished. Total cubes saved: {total_saved}")
    client.close()
    cluster.close()


if __name__ == "__main__":
    generate_tiles()
