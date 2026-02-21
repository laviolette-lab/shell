# SPDX-FileCopyrightText: 2024-present barrettMCW <mjbarrett@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""
OMERO-backed whole-slide image inference.

Requires the ``omero`` optional dependency::

    pip install shell[omero]

This module connects to an OMERO server, fetches an image at a desired
physical resolution, runs the SHELL preprocessing + inference pipeline,
and saves the label map.
"""

from __future__ import annotations

import gc
import os
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import pyvips
import torch
import torch.nn.functional as F

from shell.inference import run_inference
from shell.model import build_model
from shell.preprocessing import apply_eho_chunked, estimate_stain_params

if TYPE_CHECKING:
    from omero.gateway import BlitzGateway

# Re-export the heavy OMERO helpers only when the extra is installed.
try:
    from omero.gateway import BlitzGateway as _BlitzGateway  # noqa: F811
    from omero.model import enums as omero_enums

    PIXEL_TYPES = {
        omero_enums.PixelsTypeint8: np.int8,
        omero_enums.PixelsTypeuint8: np.uint8,
        omero_enums.PixelsTypeint16: np.int16,
        omero_enums.PixelsTypeuint16: np.uint16,
        omero_enums.PixelsTypeint32: np.int32,
        omero_enums.PixelsTypeuint32: np.uint32,
        omero_enums.PixelsTypefloat: np.float32,
        omero_enums.PixelsTypedouble: np.float64,
    }
    _HAS_OMERO = True
except ImportError:
    _HAS_OMERO = False
    PIXEL_TYPES = {}

from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_DIM = 8192 * 1000


def _require_omero() -> None:
    if not _HAS_OMERO:
        msg = (
            "The 'omero' optional dependency is required for OMERO support. "
            "Install it with: pip install shell[omero]"
        )
        raise ImportError(msg)


# ---------------------------------------------------------------------------
# OMERO helpers
# ---------------------------------------------------------------------------


def _safe_float(v, default=None):  # noqa: ANN001, ANN202
    try:
        if v is None:
            return default
        if hasattr(v, "getValue"):
            return float(v.getValue())
        return float(v)
    except Exception:
        return default


def _get_physical_sizes_um(image) -> Tuple[float, float]:  # noqa: ANN001
    """Return (mpp_x, mpp_y) in microns/pixel, with robust fallbacks."""
    mpp_x = None
    mpp_y = None

    try:
        mpp_x = _safe_float(image.getPixelSizeX(), None)
    except Exception:
        pass
    try:
        mpp_y = _safe_float(image.getPixelSizeY(), None)
    except Exception:
        pass

    if mpp_x is None or mpp_y is None:
        try:
            pix = image.getPrimaryPixels()._obj
            if mpp_x is None:
                mpp_x = _safe_float(pix.getPhysicalSizeX(), None)
            if mpp_y is None:
                mpp_y = _safe_float(pix.getPhysicalSizeY(), None)
        except Exception:
            pass

    if mpp_x is None:
        mpp_x = 1.0
    if mpp_y is None:
        mpp_y = 1.0

    return mpp_x, mpp_y


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    lo = np.percentile(arr, 1)
    hi = np.percentile(arr, 99)
    arr = (arr - lo) / (hi - lo + 1e-8)
    np.clip(arr, 0, 1, out=arr)
    return (arr * 255).astype(np.uint8)


def _resize_tile(tile: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if tile.shape[0] == out_h and tile.shape[1] == out_w:
        return tile
    t = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).float()
    r = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return r[0, 0].byte().numpy()


def _fetch_channel_tiles(
    conn: BlitzGateway,
    pixels_id: int,
    level: int,
    c_idx: int,
    band_idx: int,
    level_sz_x: int,
    level_sz_y: int,
    tile_w: int,
    tile_h: int,
    out_w: int,
    out_h: int,
    sx: float,
    sy: float,
    dtype,  # noqa: ANN001
    crop_lx0: int = 0,
    crop_ly0: int = 0,
    crop_lx1: Optional[int] = None,
    crop_ly1: Optional[int] = None,
) -> Tuple[int, np.ndarray]:
    """Fetch tiles for one OMERO channel at a given pyramid level."""
    if crop_lx1 is None:
        crop_lx1 = level_sz_x
    if crop_ly1 is None:
        crop_ly1 = level_sz_y

    store = conn.c.sf.createRawPixelsStore()
    canvas = np.zeros((out_h, out_w), dtype=np.uint8)
    try:
        store.setPixelsId(pixels_id, False)
        store.setResolutionLevel(level)

        start_ty = (crop_ly0 // tile_h) * tile_h
        start_tx = (crop_lx0 // tile_w) * tile_w

        for ty in range(start_ty, crop_ly1, tile_h):
            ry0 = max(ty, crop_ly0)
            ry1 = min(ty + tile_h, crop_ly1, level_sz_y)
            rh = ry1 - ry0
            if rh <= 0:
                continue
            oy0 = int(round((ry0 - crop_ly0) * sy))
            oy1 = int(round((ry1 - crop_ly0) * sy))
            oh = max(1, oy1 - oy0)

            for tx in range(start_tx, crop_lx1, tile_w):
                rx0 = max(tx, crop_lx0)
                rx1 = min(tx + tile_w, crop_lx1, level_sz_x)
                rw = rx1 - rx0
                if rw <= 0:
                    continue
                ox0 = int(round((rx0 - crop_lx0) * sx))
                ox1 = int(round((rx1 - crop_lx0) * sx))
                ow = max(1, ox1 - ox0)

                raw = store.getTile(0, c_idx, 0, rx0, ry0, rw, rh)
                arr = np.frombuffer(raw, dtype=dtype).reshape(rh, rw).copy()
                arr_u8 = _to_uint8(arr)
                arr_rs = _resize_tile(arr_u8, oh, ow)
                oy1c = min(out_h, oy0 + arr_rs.shape[0])
                ox1c = min(out_w, ox0 + arr_rs.shape[1])
                canvas[oy0:oy1c, ox0:ox1c] = arr_rs[: oy1c - oy0, : ox1c - ox0]
    finally:
        try:
            store.close()
        except Exception:
            pass
    return band_idx, canvas


def _detect_tissue_bbox(
    rgb: np.ndarray,
    margin_frac: float = 0.03,
) -> Tuple[float, float, float, float]:
    gray = np.mean(rgb, axis=2)
    tissue = gray < 220
    del gray
    ys, xs = np.nonzero(tissue)
    del tissue
    if len(ys) == 0:
        return (0.0, 0.0, 1.0, 1.0)
    h, w = rgb.shape[:2]
    my = int(h * margin_frac)
    mx = int(w * margin_frac)
    fy0 = max(0, int(ys.min()) - my) / h
    fx0 = max(0, int(xs.min()) - mx) / w
    fy1 = min(h, int(ys.max()) + my) / h
    fx1 = min(w, int(xs.max()) + mx) / w
    return (fy0, fx0, fy1, fx1)


# ---------------------------------------------------------------------------
# Main OMERO fetch
# ---------------------------------------------------------------------------
def fetch_omero_rgb_at_mpp(
    conn: BlitzGateway,
    image_id: int,
    target_mpp: float,
    max_dim: Optional[int] = None,
    detect_tissue: bool = True,
    tissue_thumb_target: int = 3000,
    tissue_margin_frac: float = 0.03,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int, int, int, int, int]]:
    """Fetch an OMERO image as RGB uint8 at *target_mpp*.

    :return: (rgb, native_mpp, embed_info)
    """
    _require_omero()

    image = conn.getObject("Image", image_id)
    if image is None:
        msg = f"Image {image_id} not found in OMERO"
        raise ValueError(msg)

    size_x = image.getSizeX()
    size_y = image.getSizeY()
    size_c = image.getSizeC()
    pixels_id = image.getPixelsId()
    mpp_x, mpp_y = _get_physical_sizes_um(image)

    pix_obj = image.getPrimaryPixels()
    dtype = PIXEL_TYPES.get(pix_obj.getPixelsType().value, np.uint8)

    if size_c >= 3:
        channel_ids = [0, 1, 2]
    elif size_c == 1:
        channel_ids = [0, 0, 0]
    else:
        msg = f"Unsupported OMERO channel count: {size_c}"
        raise ValueError(msg)

    # Pyramid metadata
    probe = conn.c.sf.createRawPixelsStore()
    try:
        probe.setPixelsId(pixels_id, False)
        res_descs = probe.getResolutionDescriptions()
        num_levels = probe.getResolutionLevels()
    finally:
        try:
            probe.close()
        except Exception:
            pass

    # Best level for target MPP
    best_desc_idx = 0
    best_lsz_x, best_lsz_y = size_x, size_y
    best_lmpp_x, best_lmpp_y = mpp_x, mpp_y

    for di, desc in enumerate(res_descs):
        lx, ly = desc.sizeX, desc.sizeY
        lmx = mpp_x * size_x / lx
        lmy = mpp_y * size_y / ly
        if lmx <= target_mpp and lmx > best_lmpp_x:
            best_desc_idx = di
            best_lsz_x, best_lsz_y = lx, ly
            best_lmpp_x, best_lmpp_y = lmx, lmy

    best_level = num_levels - 1 - best_desc_idx

    sx = best_lmpp_x / target_mpp
    sy = best_lmpp_y / target_mpp
    full_out_w = max(1, int(round(best_lsz_x * sx)))
    full_out_h = max(1, int(round(best_lsz_y * sy)))

    # Native tile size
    probe2 = conn.c.sf.createRawPixelsStore()
    try:
        probe2.setPixelsId(pixels_id, False)
        probe2.setResolutionLevel(best_level)
        tile_w, tile_h = probe2.getTileSize()
    finally:
        try:
            probe2.close()
        except Exception:
            pass

    # Tissue bounding-box detection
    crop_lx0, crop_ly0 = 0, 0
    crop_lx1, crop_ly1 = best_lsz_x, best_lsz_y
    ox0_embed, oy0_embed = 0, 0
    out_w, out_h = full_out_w, full_out_h

    if detect_tissue:
        thumb_di = len(res_descs) - 1
        for di in range(len(res_descs) - 1, -1, -1):
            if max(res_descs[di].sizeX, res_descs[di].sizeY) >= tissue_thumb_target:
                thumb_di = di
                break
        thumb_lsz_x = res_descs[thumb_di].sizeX
        thumb_lsz_y = res_descs[thumb_di].sizeY
        thumb_level = num_levels - 1 - thumb_di

        probe3 = conn.c.sf.createRawPixelsStore()
        try:
            probe3.setPixelsId(pixels_id, False)
            probe3.setResolutionLevel(thumb_level)
            ttw, tth = probe3.getTileSize()
        finally:
            try:
                probe3.close()
            except Exception:
                pass

        thumb = np.zeros((thumb_lsz_y, thumb_lsz_x, 3), dtype=np.uint8)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futs = [
                pool.submit(
                    _fetch_channel_tiles,
                    conn,
                    pixels_id,
                    thumb_level,
                    channel_ids[b],
                    b,
                    thumb_lsz_x,
                    thumb_lsz_y,
                    ttw,
                    tth,
                    thumb_lsz_x,
                    thumb_lsz_y,
                    1.0,
                    1.0,
                    dtype,
                )
                for b in range(3)
            ]
            for f in as_completed(futs):
                bi, cv = f.result()
                thumb[..., bi] = cv

        fy0, fx0, fy1, fx1 = _detect_tissue_bbox(thumb, tissue_margin_frac)
        del thumb

        crop_lx0 = int(fx0 * best_lsz_x)
        crop_ly0 = int(fy0 * best_lsz_y)
        crop_lx1 = int(fx1 * best_lsz_x)
        crop_ly1 = int(fy1 * best_lsz_y)

        ox0_embed = int(round(crop_lx0 * sx))
        oy0_embed = int(round(crop_ly0 * sy))
        out_w = max(1, int(round((crop_lx1 - crop_lx0) * sx)))
        out_h = max(1, int(round((crop_ly1 - crop_ly0) * sy)))

    if max_dim is not None and (out_w > max_dim or out_h > max_dim):
        clamp = min(max_dim / out_w, max_dim / out_h)
        out_w = max(1, int(round(out_w * clamp)))
        out_h = max(1, int(round(out_h * clamp)))
        sx *= clamp
        sy *= clamp

    # Parallel fetch (cropped)
    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(
                _fetch_channel_tiles,
                conn,
                pixels_id,
                best_level,
                channel_ids[b],
                b,
                best_lsz_x,
                best_lsz_y,
                tile_w,
                tile_h,
                out_w,
                out_h,
                sx,
                sy,
                dtype,
                crop_lx0,
                crop_ly0,
                crop_lx1,
                crop_ly1,
            )
            for b in range(3)
        ]
        for future in as_completed(futures):
            band_idx, canvas = future.result()
            out[..., band_idx] = canvas

    embed_info = (oy0_embed, ox0_embed, full_out_h, full_out_w, out_h, out_w)
    return out, (mpp_x, mpp_y), embed_info


# ---------------------------------------------------------------------------
# High-level OMERO inference entry point
# ---------------------------------------------------------------------------
def infer_omero_wsi(
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    image_id: int,
    model_path: Optional[str] = None,
    output_path: str,
    target_mpp: float = 1.0,
    max_dim: Optional[int] = None,
    group_id: Optional[int] = None,
    save_eho: Optional[str] = None,
    stain_downsample: int = 4,
    no_tissue_crop: bool = False,
    device: str = "auto",
    model_version: Optional[str] = None,
) -> np.ndarray:
    """Full OMERO pipeline: fetch → preprocess → infer → save.

    :param model_path: path to trained ``.pth`` weights, or ``None`` to
        use bundled weights.
    :param model_version: version tag for bundled weights (ignored when
        *model_path* is set).
    :return: (H, W) uint8 label map.
    """
    _require_omero()
    from omero.gateway import BlitzGateway  # noqa: F811

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    conn = BlitzGateway(username, password, host=host, port=port, secure=True)
    if not conn.connect():
        msg = "Failed to connect to OMERO."
        raise RuntimeError(msg)

    try:
        if group_id is not None:
            conn.SERVICE_OPTS.setOmeroGroup(group_id)

        rgb, _native_mpp, embed_info = fetch_omero_rgb_at_mpp(
            conn=conn,
            image_id=image_id,
            target_mpp=target_mpp,
            max_dim=max_dim,
            detect_tissue=not no_tissue_crop,
        )
        oy0, ox0, full_h, full_w, crop_h, crop_w = embed_info

        ds = stain_downsample
        rgb_small = rgb[::ds, ::ds]
        sp = estimate_stain_params(rgb_small)
        del rgb_small
        gc.collect()

        eho = apply_eho_chunked(rgb, **sp)
        del rgb, sp
        gc.collect()

        if save_eho:
            os.makedirs(os.path.dirname(save_eho) or ".", exist_ok=True)
            pyvips.Image.new_from_array(eho).tiffsave(save_eho)

        model = build_model(model_path, device, model_version=model_version)
        pred_crop = run_inference(eho, model, device)
        del eho, model
        gc.collect()

        # Embed cropped prediction back into full-size canvas
        if oy0 != 0 or ox0 != 0 or crop_h != full_h or crop_w != full_w:
            pred_full = np.zeros((full_h, full_w), dtype=np.uint8)
            pred_full[
                oy0 : oy0 + pred_crop.shape[0],
                ox0 : ox0 + pred_crop.shape[1],
            ] = pred_crop
            del pred_crop
        else:
            pred_full = pred_crop

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pyvips.Image.new_from_array(pred_full).tiffsave(output_path)

        return pred_full

    finally:
        try:
            conn.close()
        except Exception:
            pass
