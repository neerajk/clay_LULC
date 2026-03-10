import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import argparse
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import yaml
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from tqdm.auto import tqdm
from pyproj import Transformer

from claymodel.module import ClayMAEModule

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models.decoder import LULCSegmentationModule


def encode_scalar(x: float) -> np.ndarray:
    return np.array([np.sin(2 * np.pi * x), np.cos(2 * np.pi * x)], dtype=np.float32)


def normalize_latlon(lat: float, lon: float) -> tuple[float, float]:
    lat_norm = (lat + 90.0) / 180.0
    lon_norm = (lon + 180.0) / 360.0
    return lat_norm, lon_norm


def parse_band_indices(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 6:
        raise ValueError(f"Expected 6 band indices, got {len(vals)} from '{text}'")
    if any(v <= 0 for v in vals):
        raise ValueError("Band indices are 1-based and must be > 0")
    return vals


def parse_band_files(text: str) -> list[Path]:
    vals = [Path(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 6:
        raise ValueError(
            "Expected 6 files for --band-files in order red,green,blue,nir08,swir16,swir22."
        )
    missing = [str(v) for v in vals if not v.exists()]
    if missing:
        raise FileNotFoundError(f"Band files not found: {missing}")
    return vals


def _try_mem_mb() -> float | None:
    # Optional memory telemetry without hard dependency on psutil.
    try:
        import psutil  # type: ignore
        return float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    except Exception:
        return None


def _gpu_mem_mb(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda" or (not torch.cuda.is_available()):
        return None, None
    alloc = torch.cuda.memory_allocated(device) / (1024 * 1024)
    reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
    return float(alloc), float(reserved)


def _fmt_sec(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    rem = seconds - mins * 60
    return f"{mins}m {rem:.1f}s"


def _print_file_table(paths: list[Path], header: str):
    print(f"\n{header}")
    print("-" * 108)
    for p in paths:
        exists = p.exists()
        readable = os.access(p, os.R_OK) if exists else False
        size_mb = (p.stat().st_size / (1024 * 1024)) if exists and p.is_file() else 0.0
        print(f"[FILE] {p} | exists={exists} | readable={readable} | size_mb={size_mb:.2f}")
    print("-" * 108)


def _scene_sort_key(scene_prefix: str) -> tuple[str, str]:
    dates = re.findall(r"(\d{8})", scene_prefix)
    newest = max(dates) if dates else "00000000"
    return newest, scene_prefix


def discover_band_files_from_dir(band_dir: Path, scene_id: str | None = None) -> tuple[list[Path], str, str]:
    if not band_dir.exists() or not band_dir.is_dir():
        raise FileNotFoundError(f"--band-dir not found or not a directory: {band_dir}")

    tif_files = sorted(band_dir.glob("*_SR_B*.TIF"))
    if not tif_files:
        raise FileNotFoundError(f"No '*_SR_B*.TIF' files found in {band_dir}")

    pat = re.compile(r"^(?P<prefix>.+)_SR_B(?P<band>\d+)\.TIF$")
    grouped = defaultdict(dict)  # prefix -> band -> path
    for f in tif_files:
        m = pat.match(f.name)
        if not m:
            continue
        grouped[m.group("prefix")][int(m.group("band"))] = f

    candidates = []
    print(f"🔎 Scanning band directory: {band_dir}")
    print(f"   Found SR band files: {len(tif_files)}")
    for prefix, band_map in grouped.items():
        if scene_id and scene_id not in prefix:
            continue
        print(f"   Candidate scene: {prefix} | bands={sorted(list(band_map.keys()))}")

        # L5/L7 expected order: B3,B2,B1,B4,B5,B7
        need_l57 = [3, 2, 1, 4, 5, 7]
        if all(b in band_map for b in need_l57):
            candidates.append((prefix, "landsat57", [band_map[b] for b in need_l57]))
            continue

        # L8/L9 expected order: B4,B3,B2,B5,B6,B7
        need_l89 = [4, 3, 2, 5, 6, 7]
        if all(b in band_map for b in need_l89):
            candidates.append((prefix, "landsat89", [band_map[b] for b in need_l89]))

    if not candidates:
        if scene_id:
            raise RuntimeError(f"No complete 6-band scene in {band_dir} matching scene-id '{scene_id}'")
        raise RuntimeError(f"No complete 6-band scene in {band_dir}")

    candidates.sort(key=lambda x: _scene_sort_key(x[0]), reverse=True)
    prefix, sensor_group, files = candidates[0]
    return files, prefix, sensor_group


def parse_datetime_from_scene(scene_path: Path, override_iso: str | None) -> datetime:
    if override_iso:
        dt = datetime.fromisoformat(override_iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    name = scene_path.name
    candidates = re.findall(r"(\d{8})", name)
    for c in candidates:
        try:
            dt = datetime.strptime(c, "%Y%m%d").replace(tzinfo=timezone.utc)
            if 1980 <= dt.year <= 2035:
                return dt
        except ValueError:
            continue

    print("⚠️ Could not parse acquisition date from filename. Using default 2005-07-01T12:00:00Z.")
    return datetime(2005, 7, 1, 12, 0, 0, tzinfo=timezone.utc)


def make_starts(size: int, tile: int, stride: int) -> list[int]:
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def center_write_crop(
    y: int,
    x: int,
    valid_h: int,
    valid_w: int,
    scene_h: int,
    scene_w: int,
    tile: int,
    stride: int,
) -> tuple[int, int, int, int]:
    if stride >= tile:
        return 0, valid_h, 0, valid_w
    pad = (tile - stride) // 2
    top = 0 if y == 0 else pad
    left = 0 if x == 0 else pad
    bottom = valid_h if (y + tile) >= scene_h else tile - pad
    right = valid_w if (x + tile) >= scene_w else tile - pad
    bottom = min(bottom, valid_h)
    right = min(right, valid_w)
    return top, bottom, left, right


def tokens_to_spatial_embeddings(tokens: torch.Tensor) -> torch.Tensor:
    # Supported outputs:
    # [B, 257, 1024] (CLS + patches), [B, 256, 1024], [B, 1024, 16, 16]
    if tokens.ndim == 3 and tokens.shape[1] == 257 and tokens.shape[2] == 1024:
        return tokens[:, 1:, :].reshape(tokens.shape[0], 16, 16, 1024).permute(0, 3, 1, 2).contiguous()
    if tokens.ndim == 3 and tokens.shape[1] == 256 and tokens.shape[2] == 1024:
        return tokens.reshape(tokens.shape[0], 16, 16, 1024).permute(0, 3, 1, 2).contiguous()
    if tokens.ndim == 4 and tokens.shape[1:] == (1024, 16, 16):
        return tokens.contiguous()
    raise ValueError(f"Unexpected CLAY encoder output shape: {tuple(tokens.shape)}")


def create_clay_batch(
    pixels: np.ndarray,
    times: np.ndarray,
    latlons: np.ndarray,
    waves: list[float],
    gsd: float,
    device: torch.device,
) -> dict:
    batch = {
        "pixels": torch.from_numpy(pixels).to(device=device, dtype=torch.float32),
        "time": torch.from_numpy(times).to(device=device, dtype=torch.float32),
        "latlon": torch.from_numpy(latlons).to(device=device, dtype=torch.float32),
        "waves": torch.tensor(waves, dtype=torch.float32, device=device),
        "gsd": torch.tensor(gsd, dtype=torch.float32, device=device),
    }
    return batch


def load_models(args, device):
    print(f"📦 Loading CLAY encoder: {args.clay_ckpt}")
    clay = ClayMAEModule.load_from_checkpoint(
        checkpoint_path=args.clay_ckpt,
        model_size="large",
        metadata_path=args.metadata_path,
        dolls=[16, 32, 64, 128, 256, 768, 1024],
        doll_weights=[1, 1, 1, 1, 1, 1, 1],
        mask_ration=0.0,
        shuffle=False,
    )
    clay.eval().to(device)

    print(f"📦 Loading decoder checkpoint: {args.decoder_ckpt}")
    decoder = LULCSegmentationModule.load_from_checkpoint(args.decoder_ckpt, map_location=device)
    decoder.eval().to(device)
    return clay, decoder


def infer_batch(clay, decoder, clay_batch):
    with torch.no_grad():
        tokens, *_ = clay.model.encoder(clay_batch)
        emb = tokens_to_spatial_embeddings(tokens)
        dec_out = decoder(emb)
        full_logits = dec_out[0] if isinstance(dec_out, (tuple, list)) else dec_out
        probs = torch.softmax(full_logits, dim=1)
        pred = torch.argmax(probs, dim=1).to(torch.uint8)
        conf = torch.max(probs, dim=1).values.to(torch.float32)
    return pred.cpu().numpy(), conf.cpu().numpy()


def run(args):
    t_run_start = time.perf_counter()
    print("\n" + "=" * 108)
    print("🚀 LULC INFERENCE START")
    print("=" * 108)
    print(f"⚙️ Args: {args}")

    with open(args.metadata_path, "r") as f:
        clay_meta = yaml.safe_load(f)
    if args.platform not in clay_meta:
        raise KeyError(f"Platform '{args.platform}' not found in {args.metadata_path}")

    band_names = ["red", "green", "blue", "nir08", "swir16", "swir22"]
    means = np.array([clay_meta[args.platform]["bands"]["mean"][b] for b in band_names], dtype=np.float32).reshape(6, 1, 1)
    stds = np.array([clay_meta[args.platform]["bands"]["std"][b] for b in band_names], dtype=np.float32).reshape(6, 1, 1)
    waves = [float(clay_meta[args.platform]["bands"]["wavelength"][b]) for b in band_names]
    gsd = float(clay_meta[args.platform]["gsd"])

    mode_count = int(bool(args.scene_tif)) + int(bool(args.band_files)) + int(bool(args.band_dir))
    if mode_count != 1:
        raise ValueError("Provide exactly one input mode: --scene-tif OR --band-files OR --band-dir")

    band_idx = parse_band_indices(args.band_indices) if args.scene_tif else None
    band_files = None

    device = torch.device("cuda" if torch.cuda.is_available() and (not args.cpu_only) else "cpu")
    print(f"🖥️ Device: {device}")
    if args.scene_tif:
        print(f"🧭 Input mode: stacked scene | band mapping (R,G,B,NIR,SWIR1,SWIR2): {band_idx}")
    elif args.band_files:
        band_files = parse_band_files(args.band_files)
        print("🧭 Input mode: separate band TIFFs (R,G,B,NIR,SWIR1,SWIR2).")
        for i, p in enumerate(band_files, start=1):
            print(f"   band{i}: {p}")
    else:
        band_files, scene_prefix, sensor_group = discover_band_files_from_dir(
            band_dir=Path(args.band_dir),
            scene_id=args.scene_id,
        )
        print(
            f"🧭 Input mode: auto-discovered from directory | dir={args.band_dir} | "
            f"scene={scene_prefix} | sensor={sensor_group}"
        )
        for i, p in enumerate(band_files, start=1):
            print(f"   band{i}: {p}")

    if args.scene_tif:
        _print_file_table([Path(args.scene_tif)], "📂 Input preflight")
    else:
        _print_file_table(band_files, "📂 Input preflight")

    t_model_load = time.perf_counter()
    clay, decoder = load_models(args, device)
    model_load_sec = time.perf_counter() - t_model_load
    print(f"✅ Models loaded in {_fmt_sec(model_load_sec)}")

    scene_path = Path(args.scene_tif) if args.scene_tif else band_files[0]
    out_path = Path(args.out_tif)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conf_path = Path(args.confidence_tif) if args.confidence_tif else None
    if conf_path:
        conf_path.parent.mkdir(parents=True, exist_ok=True)

    acq_dt = parse_datetime_from_scene(scene_path, args.acq_datetime)
    week_norm = encode_scalar(acq_dt.isocalendar().week / 52.0)
    hour_norm = encode_scalar(acq_dt.hour / 24.0)
    print(f"🗓️ Acquisition datetime used: {acq_dt.isoformat()}")

    tile = int(args.tile_size)
    stride = int(args.stride)
    if stride <= 0 or tile <= 0:
        raise ValueError("tile_size and stride must be positive integers.")
    if stride > tile:
        raise ValueError("stride cannot be greater than tile_size.")

    with ExitStack() as stack:
        t_open = time.perf_counter()
        if args.scene_tif:
            src = stack.enter_context(rasterio.open(scene_path))
            if src.count < max(band_idx):
                raise ValueError(
                    f"Input has {src.count} bands, but requested band index {max(band_idx)}. "
                    "Pass correct --band-indices."
                )
            src_read = lambda win: src.read(  # noqa: E731
                indexes=band_idx,
                window=win,
                boundless=True,
                fill_value=0,
                out_dtype="float32",
            )
            ref_ds = src
        else:
            src_list = [stack.enter_context(rasterio.open(str(p))) for p in band_files]
            ref_ds = src_list[0]
            for i, ds in enumerate(src_list[1:], start=2):
                if ds.width != ref_ds.width or ds.height != ref_ds.height:
                    raise ValueError(f"Band {i} has different shape: {(ds.height, ds.width)} vs {(ref_ds.height, ref_ds.width)}")
                if ds.crs != ref_ds.crs or ds.transform != ref_ds.transform:
                    raise ValueError(f"Band {i} has different CRS/transform from band 1.")
            src_read = lambda win: np.stack([  # noqa: E731
                ds.read(1, window=win, boundless=True, fill_value=0, out_dtype="float32")
                for ds in src_list
            ], axis=0)
        print(f"✅ Rasters opened in {_fmt_sec(time.perf_counter() - t_open)}")

        height, width = ref_ds.height, ref_ds.width
        ys = make_starts(height, tile, stride)
        xs = make_starts(width, tile, stride)
        total_tiles = len(ys) * len(xs)
        print(f"🧩 Scene shape: {height} x {width} | tiles: {total_tiles} | tile={tile} stride={stride}")
        print(
            f"🗺️ Scene georef | crs={ref_ds.crs} | transform={ref_ds.transform} | "
            f"dtype={ref_ds.dtypes[0] if ref_ds.dtypes else 'unknown'} | nodata={ref_ds.nodata}"
        )
        mem = _try_mem_mb()
        if mem is not None:
            print(f"🧠 Host memory (RSS): {mem:.1f} MB")
        g_alloc, g_res = _gpu_mem_mb(device)
        if g_alloc is not None:
            print(f"🎮 GPU memory | allocated={g_alloc:.1f} MB | reserved={g_res:.1f} MB")

        to_wgs84 = Transformer.from_crs(ref_ds.crs, "EPSG:4326", always_xy=True) if ref_ds.crs else None

        out_profile = ref_ds.profile.copy()
        out_profile.update(
            count=1,
            dtype="uint8",
            nodata=0,
            compress="lzw",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )

        conf_profile = out_profile.copy()
        conf_profile.update(dtype="float32", nodata=0.0)

        with rasterio.open(out_path, "w", **out_profile) as dst:
            conf_dst = rasterio.open(conf_path, "w", **conf_profile) if conf_path else None
            try:
                t_infer = time.perf_counter()
                batch_pixels = []
                batch_times = []
                batch_latlons = []
                batch_meta = []
                batch_count = 0
                batch_infer_sec = 0.0
                write_ops = 0
                nonempty_tiles = 0
                skipped_empty_tiles = 0

                pbar = tqdm(total=total_tiles, desc="Inference Tiles", unit="tile")
                for y in ys:
                    for x in xs:
                        win = Window(col_off=x, row_off=y, width=tile, height=tile)
                        arr = src_read(win)  # [6, tile, tile]
                        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

                        valid_h = min(tile, max(0, height - y))
                        valid_w = min(tile, max(0, width - x))
                        if valid_h <= 0 or valid_w <= 0:
                            pbar.update(1)
                            continue

                        valid_mask = np.any(arr[:, :valid_h, :valid_w] != 0.0, axis=0)
                        if not np.any(valid_mask):
                            skipped_empty_tiles += 1
                            pbar.update(1)
                            if (pbar.n % max(1, int(args.status_every))) == 0:
                                elapsed = max(1e-6, time.perf_counter() - t_infer)
                                pbar.set_postfix(
                                    batches=batch_count,
                                    write_ops=write_ops,
                                    nonempty=nonempty_tiles,
                                    skipped=skipped_empty_tiles,
                                    tiles_s=f"{(pbar.n / elapsed):.2f}",
                                )
                            continue
                        nonempty_tiles += 1

                        row_c = min(height - 1, y + (valid_h // 2))
                        col_c = min(width - 1, x + (valid_w // 2))
                        x_geo, y_geo = rasterio.transform.xy(ref_ds.transform, row_c, col_c, offset="center")
                        if to_wgs84:
                            lon, lat = to_wgs84.transform(x_geo, y_geo)
                        else:
                            lon, lat = x_geo, y_geo

                        lat_n, lon_n = normalize_latlon(lat, lon)
                        latlon = np.concatenate([encode_scalar(lat_n), encode_scalar(lon_n)]).astype(np.float32)
                        time_vec = np.concatenate([week_norm, hour_norm]).astype(np.float32)

                        arr_norm = (arr - means) / stds

                        batch_pixels.append(arr_norm.astype(np.float32))
                        batch_times.append(time_vec)
                        batch_latlons.append(latlon)
                        batch_meta.append((x, y, valid_h, valid_w, valid_mask))

                        if len(batch_pixels) == args.batch_size:
                            batch_count += 1
                            tb = time.perf_counter()
                            clay_batch = create_clay_batch(
                                pixels=np.stack(batch_pixels, axis=0),
                                times=np.stack(batch_times, axis=0),
                                latlons=np.stack(batch_latlons, axis=0),
                                waves=waves,
                                gsd=gsd,
                                device=device,
                            )
                            pred_batch, conf_batch = infer_batch(clay, decoder, clay_batch)
                            batch_infer_sec += (time.perf_counter() - tb)

                            for i in range(len(batch_meta)):
                                x0, y0, vh, vw, valid_mask_i = batch_meta[i]
                                pred_tile = pred_batch[i].astype(np.uint8)
                                conf_tile = conf_batch[i].astype(np.float32)

                                pred_tile[:vh, :vw][~valid_mask_i] = 0
                                conf_tile[:vh, :vw][~valid_mask_i] = 0.0

                                top, bottom, left, right = center_write_crop(
                                    y=y0,
                                    x=x0,
                                    valid_h=vh,
                                    valid_w=vw,
                                    scene_h=height,
                                    scene_w=width,
                                    tile=tile,
                                    stride=stride,
                                )
                                h_out = max(0, bottom - top)
                                w_out = max(0, right - left)
                                if h_out == 0 or w_out == 0:
                                    continue

                                out_win = Window(col_off=x0 + left, row_off=y0 + top, width=w_out, height=h_out)
                                dst.write(pred_tile[top:bottom, left:right], 1, window=out_win)
                                write_ops += 1
                                if conf_dst:
                                    conf_dst.write(conf_tile[top:bottom, left:right], 1, window=out_win)

                            batch_pixels.clear()
                            batch_times.clear()
                            batch_latlons.clear()
                            batch_meta.clear()

                        pbar.update(1)
                        if (pbar.n % max(1, int(args.status_every))) == 0:
                            elapsed = max(1e-6, time.perf_counter() - t_infer)
                            avg_b = (batch_infer_sec / max(1, batch_count))
                            g_alloc, g_res = _gpu_mem_mb(device)
                            postfix = {
                                "batches": batch_count,
                                "write_ops": write_ops,
                                "nonempty": nonempty_tiles,
                                "skipped": skipped_empty_tiles,
                                "tiles_s": f"{(pbar.n / elapsed):.2f}",
                                "avg_b_s": f"{avg_b:.3f}",
                            }
                            if g_alloc is not None:
                                postfix["gpu_mb"] = f"{g_alloc:.0f}/{g_res:.0f}"
                            pbar.set_postfix(postfix)

                if batch_pixels:
                    batch_count += 1
                    tb = time.perf_counter()
                    clay_batch = create_clay_batch(
                        pixels=np.stack(batch_pixels, axis=0),
                        times=np.stack(batch_times, axis=0),
                        latlons=np.stack(batch_latlons, axis=0),
                        waves=waves,
                        gsd=gsd,
                        device=device,
                    )
                    pred_batch, conf_batch = infer_batch(clay, decoder, clay_batch)
                    batch_infer_sec += (time.perf_counter() - tb)
                    for i in range(len(batch_meta)):
                        x0, y0, vh, vw, valid_mask_i = batch_meta[i]
                        pred_tile = pred_batch[i].astype(np.uint8)
                        conf_tile = conf_batch[i].astype(np.float32)

                        pred_tile[:vh, :vw][~valid_mask_i] = 0
                        conf_tile[:vh, :vw][~valid_mask_i] = 0.0

                        top, bottom, left, right = center_write_crop(
                            y=y0,
                            x=x0,
                            valid_h=vh,
                            valid_w=vw,
                            scene_h=height,
                            scene_w=width,
                            tile=tile,
                            stride=stride,
                        )
                        h_out = max(0, bottom - top)
                        w_out = max(0, right - left)
                        if h_out == 0 or w_out == 0:
                            continue

                        out_win = Window(col_off=x0 + left, row_off=y0 + top, width=w_out, height=h_out)
                        dst.write(pred_tile[top:bottom, left:right], 1, window=out_win)
                        write_ops += 1
                        if conf_dst:
                            conf_dst.write(conf_tile[top:bottom, left:right], 1, window=out_win)

                pbar.close()
                infer_elapsed = time.perf_counter() - t_infer
                print("\n📈 Inference Summary")
                print("-" * 108)
                print(f"total_tiles={total_tiles} | nonempty_tiles={nonempty_tiles} | skipped_empty_tiles={skipped_empty_tiles}")
                print(f"batches={batch_count} | write_ops={write_ops}")
                print(
                    f"infer_elapsed={_fmt_sec(infer_elapsed)} | "
                    f"tile_rate={total_tiles / max(1e-6, infer_elapsed):.2f} tiles/s | "
                    f"avg_batch_infer={batch_infer_sec / max(1, batch_count):.3f}s"
                )
                mem = _try_mem_mb()
                if mem is not None:
                    print(f"host_rss_mb={mem:.1f}")
                g_alloc, g_res = _gpu_mem_mb(device)
                if g_alloc is not None:
                    print(f"gpu_alloc_mb={g_alloc:.1f} | gpu_reserved_mb={g_res:.1f}")
                print("-" * 108)
            finally:
                if conf_dst:
                    conf_dst.close()

    print(f"✅ LULC prediction saved: {out_path.resolve()}")
    if conf_path:
        print(f"✅ Confidence raster saved: {conf_path.resolve()}")
    print(f"🏁 Total runtime: {_fmt_sec(time.perf_counter() - t_run_start)}")


def build_argparser():
    p = argparse.ArgumentParser(
        description="Predict LULC mask for a large Landsat GeoTIFF using CLAY encoder + trained decoder checkpoint."
    )
    p.add_argument("--scene-tif", default=None, help="Input stacked Landsat GeoTIFF path.")
    p.add_argument(
        "--band-files",
        default=None,
        help="Comma-separated 6 single-band TIFF paths in order red,green,blue,nir08,swir16,swir22.",
    )
    p.add_argument(
        "--band-dir",
        default=None,
        help="Directory with Landsat '*_SR_B*.TIF' files. Script auto-selects a complete scene.",
    )
    p.add_argument(
        "--scene-id",
        default=None,
        help="Optional substring for selecting scene from --band-dir (e.g., 20101215).",
    )
    p.add_argument("--out-tif", required=True, help="Output predicted LULC GeoTIFF path.")
    p.add_argument("--decoder-ckpt", required=True, help="Trained decoder checkpoint path (.ckpt).")
    p.add_argument("--clay-ckpt", default="../models/clay-v1.5.ckpt", help="CLAY checkpoint path.")
    p.add_argument("--metadata-path", default="../configs/metadata.yaml", help="CLAY metadata yaml path.")
    p.add_argument("--platform", default="landsat-c2-l2", help="Platform key in metadata.yaml.")
    p.add_argument(
        "--band-indices",
        default="1,2,3,4,5,6",
        help="For --scene-tif only: 1-based band indices in order red,green,blue,nir08,swir16,swir22.",
    )
    p.add_argument("--tile-size", type=int, default=256, help="Tile size in pixels.")
    p.add_argument("--stride", type=int, default=128, help="Stride in pixels. Use 256 for non-overlap.")
    p.add_argument("--batch-size", type=int, default=8, help="Inference batch size.")
    p.add_argument("--acq-datetime", default=None, help="Acquisition datetime (ISO), e.g. 2005-12-09T05:08:03Z.")
    p.add_argument("--confidence-tif", default=None, help="Optional path to write confidence GeoTIFF.")
    p.add_argument("--cpu-only", action="store_true", help="Force CPU inference even if CUDA is available.")
    p.add_argument(
        "--status-every",
        type=int,
        default=200,
        help="Update progress-bar detailed postfix every N processed tiles.",
    )
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
