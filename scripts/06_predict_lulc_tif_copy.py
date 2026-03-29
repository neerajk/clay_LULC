import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import argparse
import tempfile
import shutil
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def encode_scalar(x: float) -> np.ndarray:
    return np.array([np.sin(2 * np.pi * x), np.cos(2 * np.pi * x)], dtype=np.float32)

def normalize_latlon(lat: float, lon: float) -> tuple[float, float]:
    return (lat + 90.0) / 180.0, (lon + 180.0) / 360.0

def parse_band_indices(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 6: raise ValueError("Expected 6 band indices")
    return vals

def parse_band_files(text: str) -> list[Path]:
    vals = [Path(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 6: raise ValueError("Expected 6 files")
    return vals


def resolve_path(path_text: str | None) -> Path | None:
    if path_text is None:
        return None
    p = Path(path_text).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def find_default_decoder_ckpt() -> str:
    ckpt_dir = DEFAULT_MODELS_DIR / "checkpoints"
    if not ckpt_dir.exists():
        return ""
    candidates = sorted(ckpt_dir.glob("decoder-*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return str(candidates[0])
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        return str(last_ckpt)
    return ""


def discover_band_files_from_dir(band_dir: Path, scene_id: str | None = None) -> list[Path]:
    files = sorted(list(band_dir.glob("*_SR_B*.TIF")) + list(band_dir.glob("*_SR_B*.tif")))
    if not files:
        raise FileNotFoundError(f"No *_SR_B*.TIF files found in {band_dir}")

    grouped: dict[str, dict[int, Path]] = {}
    for f in files:
        m = re.match(r"(.+)_SR_B(\d+)\.TI[Ff]$", f.name)
        if not m:
            continue
        sid = m.group(1)
        b = int(m.group(2))
        grouped.setdefault(sid, {})[b] = f

    if scene_id:
        matched = {sid: bmap for sid, bmap in grouped.items() if scene_id in sid}
        if not matched:
            raise ValueError(f"No scene matched scene_id={scene_id} in {band_dir}")
        grouped = matched

    # Pick best complete scene (most bands, newest lexical scene id).
    selected_scene = sorted(grouped.keys(), key=lambda s: (len(grouped[s]), s), reverse=True)[0]
    bmap = grouped[selected_scene]

    # Order expected by Clay metadata band_names = [red, green, blue, nir08, swir16, swir22]
    if selected_scene.startswith(("LC08", "LC09")):
        order = [4, 3, 2, 5, 6, 7]
    else:
        order = [3, 2, 1, 4, 5, 7]  # Landsat 5/7

    missing = [b for b in order if b not in bmap]
    if missing:
        raise FileNotFoundError(
            f"Scene {selected_scene} in {band_dir} missing required bands {missing}. "
            f"Available={sorted(bmap.keys())}"
        )

    out = [bmap[b] for b in order]
    print(f"🔎 Auto-selected scene: {selected_scene}")
    for i, p in enumerate(out, 1):
        print(f"   band{i}: {p}")
    return out

def parse_datetime_from_scene(scene_path: Path, override_iso: str | None) -> datetime:
    if override_iso:
        dt = datetime.fromisoformat(override_iso.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    candidates = re.findall(r"(\d{8})", scene_path.name)
    for c in candidates:
        try:
            dt = datetime.strptime(c, "%Y%m%d").replace(tzinfo=timezone.utc)
            if 1980 <= dt.year <= 2035: return dt
        except ValueError: continue
    return datetime(2005, 7, 1, 12, 0, 0, tzinfo=timezone.utc)

def make_starts(size: int, tile: int, stride: int) -> list[int]:
    if size <= tile: return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile: starts.append(size - tile)
    return starts

def tokens_to_spatial_embeddings(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim == 3 and tokens.shape[1] == 257 and tokens.shape[2] == 1024:
        return tokens[:, 1:, :].reshape(tokens.shape[0], 16, 16, 1024).permute(0, 3, 1, 2).contiguous()
    if tokens.ndim == 3 and tokens.shape[1] == 256 and tokens.shape[2] == 1024:
        return tokens.reshape(tokens.shape[0], 16, 16, 1024).permute(0, 3, 1, 2).contiguous()
    if tokens.ndim == 4 and tokens.shape[1:] == (1024, 16, 16):
        return tokens.contiguous()
    raise ValueError(f"Unexpected CLAY shape: {tuple(tokens.shape)}")

def create_clay_batch(pixels, times, latlons, waves, gsd, device) -> dict:
    return {
        "pixels": torch.from_numpy(pixels).to(device=device, dtype=torch.float32),
        "time": torch.from_numpy(times).to(device=device, dtype=torch.float32),
        "latlon": torch.from_numpy(latlons).to(device=device, dtype=torch.float32),
        "waves": torch.tensor(waves, dtype=torch.float32, device=device),
        "gsd": torch.tensor(gsd, dtype=torch.float32, device=device),
    }

def load_models(args, device):
    print(f"📦 Loading CLAY encoder...")
    # Always deserialize checkpoints on CPU first. Direct checkpoint loading on
    # MPS can fail if any tensor in the checkpoint is float64.
    map_loc = "cpu"
    clay = ClayMAEModule.load_from_checkpoint(
        args.clay_ckpt, model_size="large", metadata_path=args.metadata_path,
        dolls=[16, 32, 64, 128, 256, 768, 1024], doll_weights=[1]*7, mask_ration=0.0, shuffle=False,
        map_location=map_loc
    )
    clay = clay.to(dtype=torch.float32, device=device).eval()

    print(f"📦 Loading decoder checkpoint...")
    decoder = LULCSegmentationModule.load_from_checkpoint(args.decoder_ckpt, map_location=map_loc)
    decoder = decoder.to(dtype=torch.float32, device=device).eval()
    return clay, decoder

def infer_batch(clay, decoder, clay_batch):
    """Modified to return Softmax Probabilities instead of Argmax classes for blending"""
    with torch.no_grad():
        tokens, *_ = clay.model.encoder(clay_batch)
        emb = tokens_to_spatial_embeddings(tokens)
        dec_out = decoder(emb)
        full_logits = dec_out[0] if isinstance(dec_out, (tuple, list)) else dec_out
        probs = torch.softmax(full_logits, dim=1) # Get raw probabilities
    return probs.cpu().numpy() # Shape: [B, 20, 256, 256]


def run(args):
    args.metadata_path = str(resolve_path(args.metadata_path))
    args.clay_ckpt = str(resolve_path(args.clay_ckpt))
    args.decoder_ckpt = str(resolve_path(args.decoder_ckpt))
    args.out_tif = str(resolve_path(args.out_tif))
    if args.confidence_tif:
        args.confidence_tif = str(resolve_path(args.confidence_tif))
    args.scene_tif = str(resolve_path(args.scene_tif)) if args.scene_tif else None
    args.band_dir = str(resolve_path(args.band_dir))

    with open(args.metadata_path, "r") as f:
        clay_meta = yaml.safe_load(f)

    band_names = ["red", "green", "blue", "nir08", "swir16", "swir22"]
    means = np.array([clay_meta[args.platform]["bands"]["mean"][b] for b in band_names], dtype=np.float32).reshape(6, 1, 1)
    stds = np.array([clay_meta[args.platform]["bands"]["std"][b] for b in band_names], dtype=np.float32).reshape(6, 1, 1)
    waves = [float(clay_meta[args.platform]["bands"]["wavelength"][b]) for b in band_names]
    gsd = float(clay_meta[args.platform]["gsd"])

    band_idx = parse_band_indices(args.band_indices) if args.scene_tif else None
    band_files = parse_band_files(args.band_files) if args.band_files else None
    if not args.scene_tif and band_files is None:
        band_files = discover_band_files_from_dir(Path(args.band_dir), args.scene_id)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu_only else "cpu")
    if torch.backends.mps.is_available() and not args.cpu_only:
        device = torch.device("mps")
    print(f"🖥️ Device: {device}")

    clay, decoder = load_models(args, device)
    num_classes = 20  # Total LULC classes

    if args.scene_tif is None and band_files is None:
        raise ValueError("Provide --scene-tif OR --band-files OR --band-dir with available bands.")

    scene_path = Path(args.scene_tif) if args.scene_tif else band_files[0]
    out_path = Path(args.out_tif)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conf_path = Path(args.confidence_tif) if args.confidence_tif else None

    acq_dt = parse_datetime_from_scene(scene_path, args.acq_datetime)
    week_norm = encode_scalar(acq_dt.isocalendar().week / 52.0)
    hour_norm = encode_scalar(acq_dt.hour / 24.0)

    tile = int(args.tile_size)
    stride = int(args.stride)

    # ==========================================
    # 🌟 BLENDING SETUP: Create 2D Hann Window
    # ==========================================
    window_1d = np.hanning(tile).astype(np.float32)
    window_2d = np.outer(window_1d, window_1d) # Creates a smooth Gaussian-like curve

    with ExitStack() as stack:
        if args.scene_tif:
            src = stack.enter_context(rasterio.open(scene_path))
            src_read = lambda win: src.read(indexes=band_idx, window=win, boundless=True, fill_value=0, out_dtype="float32")
            ref_ds = src
        else:
            src_list = [stack.enter_context(rasterio.open(str(p))) for p in band_files]
            ref_ds = src_list[0]
            src_read = lambda win: np.stack([ds.read(1, window=win, boundless=True, fill_value=0, out_dtype="float32") for ds in src_list], axis=0)

        height, width = ref_ds.height, ref_ds.width
        ys = make_starts(height, tile, stride)
        xs = make_starts(width, tile, stride)
        to_wgs84 = Transformer.from_crs(ref_ds.crs, "EPSG:4326", always_xy=True) if ref_ds.crs else None

        # ==========================================
        # 🌟 MEMORY MAP SETUP: Save RAM on Huge TIFFs
        # ==========================================
        temp_dir = tempfile.mkdtemp()
        prob_mem_path = Path(temp_dir) / "probs.dat"
        weight_mem_path = Path(temp_dir) / "weights.dat"
        
        # These act like arrays but stream from disk, preventing MemoryErrors
        full_probs = np.memmap(prob_mem_path, dtype='float32', mode='w+', shape=(num_classes, height, width))
        full_weights = np.memmap(weight_mem_path, dtype='float32', mode='w+', shape=(height, width))
        
        batch_pixels, batch_times, batch_latlons, batch_meta = [], [], [], []

        pbar = tqdm(total=len(ys) * len(xs), desc="🧩 Forward Pass (Extracting & Blending)")
        for y in ys:
            for x in xs:
                win = Window(col_off=x, row_off=y, width=tile, height=tile)
                arr = src_read(win)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

                valid_h = min(tile, max(0, height - y))
                valid_w = min(tile, max(0, width - x))
                if valid_h <= 0 or valid_w <= 0:
                    pbar.update(1); continue

                valid_mask = np.any(arr[:, :valid_h, :valid_w] != 0.0, axis=0)

                row_c, col_c = min(height - 1, y + (valid_h // 2)), min(width - 1, x + (valid_w // 2))
                x_geo, y_geo = rasterio.transform.xy(ref_ds.transform, row_c, col_c, offset="center")
                lon, lat = to_wgs84.transform(x_geo, y_geo) if to_wgs84 else (x_geo, y_geo)
                lat_n, lon_n = normalize_latlon(lat, lon)
                latlon = np.concatenate([encode_scalar(lat_n), encode_scalar(lon_n)]).astype(np.float32)

                batch_pixels.append(((arr - means) / stds).astype(np.float32))
                batch_times.append(np.concatenate([week_norm, hour_norm]).astype(np.float32))
                batch_latlons.append(latlon)
                batch_meta.append((x, y, valid_h, valid_w, valid_mask))

                if len(batch_pixels) == args.batch_size or (y == ys[-1] and x == xs[-1]):
                    clay_batch = create_clay_batch(
                        np.stack(batch_pixels, axis=0), np.stack(batch_times, axis=0),
                        np.stack(batch_latlons, axis=0), waves, gsd, device
                    )
                    prob_batch = infer_batch(clay, decoder, clay_batch)

                    for i in range(len(batch_meta)):
                        x0, y0, vh, vw, valid_mask_i = batch_meta[i]
                        prob_tile = prob_batch[i] # Shape [20, 256, 256]
                        
                        # Apply invalid mask to probabilities
                        prob_tile[:, ~valid_mask_i] = 0.0
                        
                        # 🌟 APPLY HANN WINDOW WEIGHTS
                        win_crop = window_2d[:vh, :vw]
                        weighted_probs = prob_tile[:, :vh, :vw] * win_crop
                        
                        # Add to the global memmap arrays
                        full_probs[:, y0:y0+vh, x0:x0+vw] += weighted_probs
                        full_weights[y0:y0+vh, x0:x0+vw] += win_crop

                    batch_pixels.clear(); batch_times.clear(); batch_latlons.clear(); batch_meta.clear()
                pbar.update(1)
        pbar.close()

        # ==========================================
        # 🌟 FINAL STEP: Stitch and Write to TIFF
        # ==========================================
        print("🗺️ Collapsing probabilities and writing smooth GeoTIFF...")
        out_profile = ref_ds.profile.copy()
        out_profile.update(count=1, dtype="uint8", nodata=0, compress="lzw", predictor=2, tiled=True, blockxsize=256, blockysize=256)
        
        conf_profile = out_profile.copy()
        conf_profile.update(dtype="float32", nodata=0.0)

        with rasterio.open(out_path, "w", **out_profile) as dst:
            conf_dst = rasterio.open(conf_path, "w", **conf_profile) if conf_path else None
            
            # Write in 1024-pixel vertical bands to keep RAM usage incredibly low
            for y in tqdm(range(0, height, 1024), desc="Writing Rows"):
                y_end = min(y + 1024, height)
                
                # Load chunk into memory
                weight_chunk = full_weights[y:y_end, :]
                prob_chunk = full_probs[:, y:y_end, :]
                
                # Normalize probabilities by the window weights
                weight_chunk[weight_chunk == 0] = 1.0 # Avoid division by zero
                prob_chunk /= weight_chunk[np.newaxis, :, :]
                
                # Calculate final prediction
                pred_chunk = np.argmax(prob_chunk, axis=0).astype(np.uint8)
                conf_chunk = np.max(prob_chunk, axis=0).astype(np.float32)
                
                # Restore pure No-Data boundaries
                nodata_mask = (full_weights[y:y_end, :] == 0)
                pred_chunk[nodata_mask] = 0
                conf_chunk[nodata_mask] = 0.0
                
                win = Window(col_off=0, row_off=y, width=width, height=y_end - y)
                dst.write(pred_chunk, 1, window=win)
                if conf_dst: conf_dst.write(conf_chunk, 1, window=win)

    # Clean up temp files
    shutil.rmtree(temp_dir)
    print(f"✅ Smooth LULC prediction saved: {out_path.resolve()}")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--scene-tif", default=None)
    p.add_argument("--band-files", default=None)
    p.add_argument("--band-dir", default=str(DEFAULT_DATA_DIR / "raw"))
    p.add_argument("--scene-id", default=None, help="Optional token to select scene from --band-dir.")
    p.add_argument("--out-tif", default=str(DEFAULT_DATA_DIR / "predictions" / "lulc_prediction_copy.tif"))
    p.add_argument("--decoder-ckpt", default=find_default_decoder_ckpt())
    p.add_argument("--clay-ckpt", default=str(DEFAULT_MODELS_DIR / "clay-v1.5.ckpt"))
    p.add_argument("--metadata-path", default=str(DEFAULT_CONFIGS_DIR / "metadata.yaml"))
    p.add_argument("--platform", default="landsat-c2-l2")
    p.add_argument("--band-indices", default="1,2,3,4,5,6")
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--acq-datetime", default=None)
    p.add_argument("--confidence-tif", default=None)
    p.add_argument("--cpu-only", action="store_true")
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    if not args.decoder_ckpt:
        raise FileNotFoundError(
            f"No decoder checkpoint found in {(DEFAULT_MODELS_DIR / 'checkpoints')}.\n"
            "Pass --decoder-ckpt explicitly."
        )
    if args.confidence_tif is None:
        out = Path(args.out_tif)
        args.confidence_tif = str(out.with_name(out.stem + "_conf.tif"))
    run(args)
