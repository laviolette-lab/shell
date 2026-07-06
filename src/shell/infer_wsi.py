# SPDX-FileCopyrightText: 2024-present barrettMCW <mjbarrett@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""
End-to-end inference on a local RGB whole-slide image.

Pipeline:
  1. Read the raw RGB image (TIFF, PNG, JPEG, etc.)
  2. Scale to the target microns-per-pixel (MPP)
  3. Macenko PCA colour deconvolution → 3-channel EHO image
  4. Sliding-window SegResNetVAE inference
  5. Return / save the resulting label image (uint8)

Label map:
  0 = Background / White
  1 = Epithelium
  2 = Stroma
"""

from __future__ import annotations

import gc
import logging
import os
import warnings
from types import ModuleType
from typing import cast

import numpy as np

# shell.inference and shell.model load torch at their module level.
# pyvips must come *after* them so PyTorch initialises its thread-pool
# runtime before libvips creates its own.  On macOS the reverse order
# (pyvips before torch) causes a segfault because both runtimes race to
# own the same OpenMP/GCD thread infrastructure.
from shell.inference import run_inference
from shell.model import build_model
from shell.post_process import PROFILES, post_process
from shell.transforms import (
    EHOd,
    TissueMaskd,
    apply_eho_chunked,
    detect_background,
    estimate_stain_params,
)

# pyvips intentionally after torch-loading shell imports above (macOS safety)
import pyvips  # isort: skip

log = logging.getLogger(__name__)

_OPENSLIDE_MODULE: ModuleType | None = None
_OPENSLIDE_IMPORT_FAILED: bool = False

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
TARGET_MPP: float = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_openslide() -> ModuleType | None:
    """Import and cache the ``openslide`` module lazily.

    Returns ``None`` when openslide is unavailable.
    """
    global _OPENSLIDE_MODULE, _OPENSLIDE_IMPORT_FAILED
    if _OPENSLIDE_IMPORT_FAILED:
        return None
    if _OPENSLIDE_MODULE is None:
        try:
            import openslide
        except Exception:
            _OPENSLIDE_IMPORT_FAILED = True
            return None
        _OPENSLIDE_MODULE = openslide
    return _OPENSLIDE_MODULE


def _read_mpp_from_openslide(image_path: str) -> tuple[float, float] | None:
    """Try to extract um/px from OpenSlide metadata.

    Returns ``(mpp_x, mpp_y)`` or ``None`` if the format is unsupported
    or the metadata is missing.
    """
    openslide = _get_openslide()
    if openslide is None:
        return None

    try:
        slide = openslide.OpenSlide(image_path)
    except (
        openslide.OpenSlideUnsupportedFormatError,
        openslide.OpenSlideError,
    ):
        return None

    try:
        raw_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        raw_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
        if raw_x is None or raw_y is None:
            return None
        mpp_x, mpp_y = float(raw_x), float(raw_y)
        if mpp_x <= 0 or mpp_y <= 0:
            return None
        return mpp_x, mpp_y
    finally:
        slide.close()


def _load_image(
    image_path: str,
) -> tuple[pyvips.Image | np.ndarray, str]:
    """Load an image, returning ``(data, source)`` where *source* is
    ``"vips"`` or ``"openslide"``.

    Tries pyvips first.  If pyvips cannot open the file (e.g. an exotic
    whole-slide format it does not support), falls back to OpenSlide.
    """
    # --- attempt 1: pyvips ---
    try:
        vips_img = pyvips.Image.new_from_file(image_path, access="sequential")
        return vips_img, "vips"
    except pyvips.Error:
        log.info("pyvips could not open %s; falling back to OpenSlide.", image_path)

    # --- attempt 2: openslide ---
    openslide = _get_openslide()
    if openslide is None:
        msg = (
            f"pyvips could not open '{image_path}', and OpenSlide is not available. "
            "Install openslide-python (and OpenSlide runtime) or use a format "
            "supported by pyvips."
        )
        raise ValueError(msg)

    try:
        slide = openslide.OpenSlide(image_path)
        dims = slide.dimensions  # (width, height)
        rgba = slide.read_region((0, 0), 0, dims)
        slide.close()
        arr: np.ndarray = np.array(rgba)[..., :3].copy()
        del rgba
        return arr, "openslide"
    except (
        openslide.OpenSlideUnsupportedFormatError,
        openslide.OpenSlideError,
    ) as exc:
        msg = (
            f"Neither pyvips nor OpenSlide could open '{image_path}'. "
            "Please check the file format."
        )
        raise ValueError(msg) from exc


def _vips_to_rgb_numpy(vips_img: pyvips.Image | np.ndarray) -> np.ndarray:
    """Convert a pyvips image or numpy array to (H, W, 3) uint8 RGB numpy array."""
    # If the caller accidentally passes a numpy array (e.g. from an openslide
    # fallback), accept it and normalise to (H, W, 3) uint8.
    if isinstance(vips_img, np.ndarray):
        arr = vips_img
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]
        return arr.astype(np.uint8)

    bands = vips_img.bands
    if bands == 1:
        vips_img = vips_img.bandjoin([vips_img, vips_img])
    elif bands == 4:
        vips_img = vips_img.extract_band(0, n=3)
    elif bands != 3:
        vips_img = vips_img.extract_band(0, n=3)
    return vips_img.numpy()


def _read_image_size(image_path: str) -> tuple[int, int]:
    """Return ``(height, width)`` for *image_path*."""
    try:
        vips_img = pyvips.Image.new_from_file(image_path, access="sequential")
        return int(vips_img.height), int(vips_img.width)
    except pyvips.Error:
        pass

    openslide = _get_openslide()
    if openslide is None:
        msg = f"Could not determine image size for '{image_path}'."
        raise ValueError(msg)

    try:
        slide = openslide.OpenSlide(image_path)
        width, height = slide.dimensions
        slide.close()
        return int(height), int(width)
    except (
        openslide.OpenSlideUnsupportedFormatError,
        openslide.OpenSlideError,
    ) as exc:
        msg = f"Could not determine image size for '{image_path}'."
        raise ValueError(msg) from exc


def _resize_label_map_nearest(
    label_map: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Resize a label map to ``(out_h, out_w)`` using nearest-neighbour."""
    in_h, in_w = label_map.shape[:2]
    if in_h == out_h and in_w == out_w:
        return label_map

    y_idx = np.clip((np.arange(out_h) * in_h / out_h).astype(np.int64), 0, in_h - 1)
    x_idx = np.clip((np.arange(out_w) * in_w / out_w).astype(np.int64), 0, in_w - 1)
    return label_map[y_idx[:, None], x_idx[None, :]]


# ---------------------------------------------------------------------------
# Tiled-pipeline helpers
# ---------------------------------------------------------------------------

def _tile_positions(length: int, tile_size: int, margin: int) -> list[int]:
    """Return tile start positions covering *length* with overlap margins.

    Each tile is ``tile_size`` pixels wide; adjacent tiles overlap by
    ``2 * margin`` so the center-crop (minus margins on each side)
    seamlessly covers *length*.
    """
    if length <= tile_size:
        return [0]
    step = tile_size - 2 * margin
    pos = list(range(0, length - tile_size + 1, step))
    if pos[-1] + tile_size < length:
        pos.append(length - tile_size)
    return pos


def _compute_norm_stats(eho_hwc: np.ndarray) -> dict:
    """Pre-compute per-channel min/max from a representative EHO thumbnail.

    Matches the training-time transform::

        ScaleIntensityd(minv=0.0, maxv=1.0, channel_wise=True)

    Statistics are derived from the thumbnail so that per-tile normalisation
    during tiled inference is consistent across the whole slide.

    Parameters
    ----------
    eho_hwc : (H, W, 3) uint8 array
        EHO image (typically computed on a thumbnail).
    """
    img = eho_hwc.astype(np.float32) / 255.0

    ch_mins: list[float] = []
    ch_maxs: list[float] = []
    for c in range(3):
        ch = img[..., c]
        ch_mins.append(float(ch.min()))
        ch_maxs.append(float(ch.max()))

    return {"ch_mins": ch_mins, "ch_maxs": ch_maxs}


def _normalize_tile(eho_chw) -> "torch.Tensor":
    """Normalise a (C, H, W) EHO uint8 tile to float32 [0, 1] per channel.

    Matches the training-time transform applied per crop::

        ScaleIntensityd(minv=0.0, maxv=1.0, channel_wise=True)

    Each channel is independently stretched to [0, 1] using the tile's own
    min/max.  Using per-tile statistics (rather than global thumbnail stats)
    ensures the model receives inputs in its expected [0, 1] distribution
    regardless of local staining variation.

    Accepts either a ``torch.Tensor`` or a ``numpy.ndarray`` (converted
    internally).  Returns a ``torch.Tensor``.
    """
    import torch

    if not isinstance(eho_chw, torch.Tensor):
        eho_chw = torch.from_numpy(eho_chw).float()
    else:
        eho_chw = eho_chw.float()

    out = eho_chw / 255.0  # uint8 → float [0, 1]
    for c in range(out.shape[0]):
        ch = out[c]
        ch_min = ch.min()
        ch_max = ch.max()
        rng = ch_max - ch_min
        if rng > 1e-8:
            out[c] = (ch - ch_min) / rng
        # else: constant channel (e.g. pure background) — leave as zeros
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def preprocess_wsi(
    image_path: str,
    *,
    target_mpp: float = TARGET_MPP,
    mpp: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a raw RGB image, scale to *target_mpp*, and produce an EHO image.

    Uses the MONAI transform pipeline (``TissueMaskd`` → ``EHOd``) from
    :mod:`shell.transforms`.

    :param image_path: path to an RGB image (TIFF, PNG, JPEG, etc.).
    :param target_mpp: desired microns-per-pixel.
    :param mpp: manual override for the source image um/px.  When
        ``None`` the value is read from slide metadata; if metadata is
        unavailable (e.g. plain PNG) a warning is emitted and scaling is
        skipped (the image is assumed to already be at *target_mpp*).
    :return: tuple of (H, W, 3) uint8 EHO image and (H, W) bool tissue mask.
    """
    from monai.data import MetaTensor
    from monai.transforms import Compose

    # 1. Determine MPP
    if mpp is not None:
        mpp_x = mpp_y = float(mpp)
    else:
        mpp_result = _read_mpp_from_openslide(image_path)
        if mpp_result is not None:
            mpp_x, mpp_y = mpp_result
        else:
            warnings.warn(
                f"Could not read um/px metadata from '{image_path}'. "
                "No resolution scaling will be applied. Use the --mpp "
                "flag (or the mpp= parameter) to specify the source "
                "resolution manually.",
                stacklevel=2,
            )
            mpp_x = mpp_y = target_mpp  # scale factor becomes 1.0

    # 2. Load image
    img_or_vips, source = _load_image(image_path)

    # 3. Scale to target MPP
    scale_x = target_mpp / mpp_x
    scale_y = target_mpp / mpp_y
    needs_scaling = not (abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6)

    if source == "vips":
        vips_img = cast(pyvips.Image, img_or_vips)
        if needs_scaling:
            # Use positional vscale/kernel to satisfy the pyvips stubs and
            # avoid type-checker complaints about keyword-only overloads.
            vips_img = vips_img.resize(1.0 / scale_x, 1.0 / scale_y, "lanczos3")
        image_np = _vips_to_rgb_numpy(vips_img)
        del vips_img
    else:
        # numpy array from openslide fallback
        image_np = img_or_vips
        if needs_scaling:
            vips_tmp = pyvips.Image.new_from_array(image_np)
            # cast to Image for the type-checker and use positional args for
            # the same reason as above.
            vips_tmp = cast(pyvips.Image, vips_tmp).resize(
                1.0 / scale_x, 1.0 / scale_y, "lanczos3"
            )
            image_np = _vips_to_rgb_numpy(vips_tmp)
            del vips_tmp

    # 4. Run MONAI transform pipeline: TissueMask → EHO
    pipeline = Compose(
        [
            TissueMaskd(keys=["image"]),
            EHOd(
                keys=["image"],
                tissue_mask_keys=["image_tissue_mask"],
            ),
        ]
    )
    data = pipeline({"image": MetaTensor(image_np)})
    del image_np
    gc.collect()

    # EHOd outputs (3, H, W) MetaTensor — convert back to (H, W, 3) uint8
    eho = data["image"].numpy().transpose(1, 2, 0).astype(np.uint8)
    tissue_mask = data["image_tissue_mask"].numpy().squeeze() > 0
    del data

    return eho, tissue_mask


def infer_wsi(
    input_path: str,
    output_path: str,
    model_path: str | None = None,
    *,
    model_version: str | None = None,
    target_mpp: float = TARGET_MPP,
    mpp: float | None = None,
    save_eho: str | None = None,
    save_raw: str | None = None,
    profile: str = "best_effort",
    mode: str = "wsi",
    tile_pad: int | None = None,
    device: str = "auto",
    _model=None,
) -> np.ndarray:
    """Tiled inference pipeline with pyvips streaming.

    Major design changes vs. the v1 (full-image) pipeline:

    * **pyvips thumbnail** — tissue mask and stain-parameter estimation
      run on a ~2 000 px thumbnail instead of the full image, cutting
      preprocessing from ~2 min to a few seconds.
    * **Tile-based EHO + inference** — RGB tiles are fetched lazily from
      pyvips, converted to EHO with pre-computed stain vectors, and fed
      to the model one at a time.  Peak memory drops from ~1 GB to
      ~150 MB.
    * **Tissue-aware tile skipping** — tiles where the tissue mask shows
      < 1 % tissue are never fetched or processed, saving ~30-40 % of
      model forward passes on a typical prostate WSI.
    * **Global normalisation** — per-channel mean/std and scale min/max
      are computed once on the thumbnail EHO and applied identically to
      every tile so that tiled normalisation matches full-image behaviour.

    Per-phase wall-clock timings are printed to stdout for benchmarking.

    :param input_path: raw RGB image (TIFF, PNG, JPEG, etc.).
    :param output_path: where to save the uint8 label TIFF.
    :param model_path: path to trained ``.pth`` weights, or ``None`` to
        use bundled weights.
    :param model_version: version tag for bundled weights (ignored when
        *model_path* is set).
    :param target_mpp: desired resolution.
    :param mpp: manual source um/px override.  See
        :func:`preprocess_wsi` for details.
    :param save_eho: optional path to save the intermediate EHO image.
    :param save_raw: optional path to save the raw model predictions as a
        three-band uint8 image (band 0 = inner/lumen, band 1 = outer/epithelium,
        band 2 = background), scaled to 0/255.  Saved at the original input
        resolution (same spatial extent as the returned label map).
    :param _model: optional pre-loaded model (skips Phase 3 disk load). The
        model must already be on the correct device and in eval mode.
    :param profile: post-processing filter profile.  One of
        ``"best_effort"``, ``"precise"``, or ``"sensitive"``.
        Defaults to ``"best_effort"``.
    :param mode: post-processing mode — ``"wsi"`` (default, full pipeline
        with tissue restriction and urethra detection), ``"biopsy"`` (tissue
        restriction but no urethra detection), or ``"tile"`` (no tissue mask,
        no urethra; reflect-pads the predictions before morphological ops).
    :param tile_pad: padding in pixels for ``mode='tile'``.  ``None``
        (default) auto-computes 50 % of the shorter output dimension.
    :param device: ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.
    :return: (H, W) uint8 label map at the original input resolution.
    """
    from time import perf_counter

    import torch
    import torch.nn.functional as F

    timings: dict[str, float] = {}

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if profile not in PROFILES:
        available = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown profile {profile!r}. Available: {available}")

    device_obj = torch.device(device)

    # ── Phase 1: Open image (lazy via pyvips) ────────────────────────
    t0 = perf_counter()

    if mpp is not None:
        mpp_x = mpp_y = float(mpp)
    else:
        mpp_result = _read_mpp_from_openslide(input_path)
        if mpp_result is not None:
            mpp_x, mpp_y = mpp_result
        else:
            warnings.warn(
                f"Could not read um/px metadata from '{input_path}'. "
                "No resolution scaling will be applied. Use the --mpp "
                "flag (or the mpp= parameter) to specify the source "
                "resolution manually.",
                stacklevel=2,
            )
            mpp_x = mpp_y = target_mpp

    vips_full = pyvips.Image.new_from_file(input_path, access="random")
    if vips_full.bands > 3:
        vips_full = vips_full.extract_band(0, n=3)

    scale_x = target_mpp / mpp_x
    scale_y = target_mpp / mpp_y
    needs_scaling = not (abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6)
    if needs_scaling:
        vips_full = vips_full.resize(1.0 / scale_x, vscale=1.0 / scale_y)

    H, W = vips_full.height, vips_full.width
    timings["open"] = perf_counter() - t0
    log.info("Image: %d x %d px (scaled=%s)", W, H, needs_scaling)

    # ── Phase 2: Thumbnail-based preprocessing ───────────────────────

    # 2a. Create thumbnails
    #     - Small (~2000px) for fast tissue masking
    #     - 4x (~5000px) for stain parameter estimation (matching EHOd)
    t0 = perf_counter()
    small_thumb_scale = min(1.0, 2000 / max(H, W))
    small_thumb_np = np.ascontiguousarray(
        vips_full.resize(small_thumb_scale).numpy()[:, :, :3]
    )
    timings["thumbnail"] = perf_counter() - t0

    # 2b. Tissue mask on small thumbnail → upscale to full resolution
    t0 = perf_counter()
    bg_mask_small = detect_background(small_thumb_np)
    tissue_small = ~bg_mask_small
    th_h, th_w = tissue_small.shape
    y_idx = np.clip((np.arange(H) * th_h / H).astype(np.int64), 0, th_h - 1)
    x_idx = np.clip((np.arange(W) * th_w / W).astype(np.int64), 0, th_w - 1)
    tissue_mask_full = tissue_small[y_idx[:, None], x_idx[None, :]]
    del bg_mask_small, tissue_small, small_thumb_np
    timings["tissue_mask"] = perf_counter() - t0
    tissue_pct = 100 * tissue_mask_full.mean()
    log.info("Tissue: %.1f%%", tissue_pct)

    # 2c. Stain parameters on 4x thumbnail (matches EHOd behavior)
    #     Reuse the small tissue mask (upscaled) instead of re-running
    #     detect_background on the larger thumbnail.
    t0 = perf_counter()
    stain_scale = min(1.0, max(512, max(H, W) // 4) / max(H, W))
    stain_thumb_np = np.ascontiguousarray(
        vips_full.resize(stain_scale).numpy()[:, :, :3]
    )
    # Nearest-neighbour upscale of tissue mask to stain thumbnail size
    st_h, st_w = stain_thumb_np.shape[:2]
    st_y = np.clip((np.arange(st_h) * H / st_h).astype(np.int64), 0, H - 1)
    st_x = np.clip((np.arange(st_w) * W / st_w).astype(np.int64), 0, W - 1)
    stain_bg = ~tissue_mask_full[st_y[:, None], st_x[None, :]]
    stain_params = estimate_stain_params(
        stain_thumb_np.astype(np.uint8),
        bg_mask=stain_bg,
    )
    timings["stain_params"] = perf_counter() - t0

    del stain_thumb_np, stain_bg

    # ── Phase 3: Load model ──────────────────────────────────────────
    t0 = perf_counter()
    if _model is not None:
        model = _model
        timings["model_load"] = 0.0
    else:
        model = build_model(model_path, device, model_version=model_version)
        timings["model_load"] = perf_counter() - t0

    # ── Phase 4: Tiled EHO + inference ───────────────────────────────
    t0 = perf_counter()

    tile_size = 2048
    margin = 128
    min_tissue_frac = 0.01

    y_positions = _tile_positions(H, tile_size, margin)
    x_positions = _tile_positions(W, tile_size, margin)

    inner_pred = np.zeros((H, W), dtype=bool)
    outer_pred = np.zeros((H, W), dtype=bool)
    hematoxylin_full = np.zeros((H, W), dtype=np.uint8)

    # Only allocate full EHO when the user wants it saved
    eho_full = np.zeros((H, W, 3), dtype=np.uint8) if save_eho else None

    n_total = len(y_positions) * len(x_positions)
    n_tissue = 0
    n_skipped = 0
    t_fetch = 0.0
    t_eho = 0.0
    t_model = 0.0
    t_stitch = 0.0

    for y0 in y_positions:
        for x0 in x_positions:
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)
            th, tw = y1 - y0, x1 - x0

            # ── skip non-tissue tiles entirely ──
            # Zeros in hematoxylin_full are handled by the masked
            # equalize_hist in _equalise_hematoxylin.
            if tissue_mask_full[y0:y1, x0:x1].mean() < min_tissue_frac:
                n_skipped += 1
                continue

            # ── fetch RGB tile from pyvips ──
            _tf = perf_counter()
            rgb_tile = np.ascontiguousarray(
                vips_full.crop(x0, y0, tw, th).numpy()[:, :, :3]
            )

            # ── EHO with pre-computed stain vectors ──
            eho_tile = apply_eho_chunked(
                rgb_tile.astype(np.uint8),
                chunk_rows=th,  # whole tile at once
                **stain_params,
            )
            del rgb_tile
            t_fetch += perf_counter() - _tf

            # Store hematoxylin channel for post-processing
            hematoxylin_full[y0:y1, x0:x1] = eho_tile[:, :, 1]
            if eho_full is not None:
                eho_full[y0:y1, x0:x1] = eho_tile

            # ── normalise ──
            _te = perf_counter()
            n_tissue += 1
            tile_t = _normalize_tile(
                torch.from_numpy(eho_tile).permute(2, 0, 1),
            )
            del eho_tile

            # ── pad to multiple of 64 ──
            pad_h = (64 - th % 64) % 64
            pad_w = (64 - tw % 64) % 64
            if pad_h or pad_w:
                padding = (
                    pad_w // 2,
                    pad_w - pad_w // 2,
                    pad_h // 2,
                    pad_h - pad_h // 2,
                )
                tile_t = F.pad(tile_t.unsqueeze(0), padding, "reflect")
            else:
                tile_t = tile_t.unsqueeze(0)
                padding = (0, 0, 0, 0)

            # ── model forward pass ──
            with torch.inference_mode():
                logits = model(tile_t.to(device_obj))
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            logits = logits.cpu()
            del tile_t

            # ── unpad ──
            if pad_h or pad_w:
                _, _, ph, pw = logits.shape
                logits = logits[
                    :, :,
                    padding[2] : ph - padding[3],
                    padding[0] : pw - padding[1],
                ]

            # ── sigmoid + threshold ──
            probs = torch.sigmoid(logits)
            inner_p = probs[0, 0]
            outer_p = probs[0, 1]
            bg_p = probs[0, 2]
            inner_tile = ((inner_p > 0.5) & (inner_p > bg_p)).numpy()
            outer_tile = ((outer_p > 0.5) & (outer_p > bg_p)).numpy()
            del logits, probs
            t_model += perf_counter() - _te

            # ── center-crop stitch ──
            _ts = perf_counter()
            vy0 = margin if y0 > 0 else 0
            vx0 = margin if x0 > 0 else 0
            vy1 = th - (margin if y1 < H else 0)
            vx1 = tw - (margin if x1 < W else 0)

            oy0, oy1 = y0 + vy0, y0 + vy1
            ox0, ox1 = x0 + vx0, x0 + vx1

            inner_pred[oy0:oy1, ox0:ox1] = inner_tile[vy0:vy1, vx0:vx1]
            outer_pred[oy0:oy1, ox0:ox1] = outer_tile[vy0:vy1, vx0:vx1]
            t_stitch += perf_counter() - _ts

    timings["tiled_inference"] = perf_counter() - t0
    timings["  fetch+eho"] = t_fetch
    timings["  model+norm"] = t_model
    timings["  stitch"] = t_stitch
    del model, vips_full
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info(
        "Tiles: %d total, %d processed, %d skipped (%.0f%% skipped)",
        n_total, n_tissue, n_skipped, 100 * n_skipped / max(n_total, 1),
    )

    # ── Phase 5: Save EHO intermediate ───────────────────────────────
    if save_eho and eho_full is not None:
        t0 = perf_counter()
        os.makedirs(os.path.dirname(save_eho) or ".", exist_ok=True)
        eho_full = np.ascontiguousarray(eho_full)
        log.info("Saving EHO: shape=%s dtype=%s C_contig=%s", eho_full.shape, eho_full.dtype, eho_full.flags["C_CONTIGUOUS"])
        eho_params = {"lossless": True} if save_eho.lower().endswith(".jp2") else {}
        pyvips.Image.new_from_array(eho_full.astype(np.uint8)).write_to_file(save_eho, **eho_params)
        del eho_full
        timings["save_eho"] = perf_counter() - t0

    # ── Phase 6: Post-processing ─────────────────────────────────────
    t0 = perf_counter()
    label_map = post_process(
        inner_pred,
        outer_pred,
        tissue_mask_full if mode == "wsi" or mode == "biopsy" else None,
        hematoxylin_full,
        mode=mode,
        profile_name=profile,
        tile_pad=tile_pad,
        verbose=True,
    )
    del tissue_mask_full, hematoxylin_full
    if not save_raw:
        del inner_pred, outer_pred
    gc.collect()
    timings["post_process"] = perf_counter() - t0

    # ── Phase 7: Resize to original resolution + save ────────────────
    t0 = perf_counter()
    input_h, input_w = _read_image_size(input_path)
    label_map = _resize_label_map_nearest(label_map, input_h, input_w)
    # Also resize raw predictions to original res before saving / returning
    if save_raw:
        inner_pred = _resize_label_map_nearest(
            inner_pred.astype(np.uint8), input_h, input_w
        ).astype(bool)
        outer_pred = _resize_label_map_nearest(
            outer_pred.astype(np.uint8), input_h, input_w
        ).astype(bool)
    timings["resize"] = perf_counter() - t0

    t0 = perf_counter()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    label_map = np.ascontiguousarray(label_map)
    log.info("Saving label_map: shape=%s dtype=%s C_contig=%s", label_map.shape, label_map.dtype, label_map.flags["C_CONTIGUOUS"]) 
    params = {"lossless": True} if output_path.lower().endswith(".jp2") else {}
    pyvips.Image.new_from_array(label_map.astype(np.uint8)).write_to_file(output_path, **params)
    timings["save"] = perf_counter() - t0

    if save_raw:
        t0 = perf_counter()
        os.makedirs(os.path.dirname(save_raw) or ".", exist_ok=True)
        # 3-band uint8 (0/255): inner, outer, background
        raw_bands = np.stack(
            [
                (inner_pred.astype(np.uint8) * 255),
                (outer_pred.astype(np.uint8) * 255),
                ((~(inner_pred | outer_pred)).astype(np.uint8) * 255),
            ],
            axis=-1,
        )
        raw_bands = np.ascontiguousarray(raw_bands)
        log.info("Saving raw_bands: shape=%s dtype=%s C_contig=%s", raw_bands.shape, raw_bands.dtype, raw_bands.flags["C_CONTIGUOUS"]) 
        raw_params = {"lossless": True} if save_raw.lower().endswith(".jp2") else {}
        pyvips.Image.new_from_array(raw_bands.astype(np.uint8)).write_to_file(save_raw, **raw_params)
        del raw_bands
        timings["save_raw"] = perf_counter() - t0

    # ── Timing summary ───────────────────────────────────────────────
    # Exclude indented sub-timings from total (they're breakdowns of parent phases)
    top_timings = {k: v for k, v in timings.items() if not k.startswith("  ")}
    total = sum(top_timings.values())
    print(f"\n{'=' * 60}")
    print("Pipeline timings (tiled)")
    print(f"{'=' * 60}")
    print(
        f"  Image: {W} x {H} px  |  Tissue: {tissue_pct:.1f}%  |  "
        f"Tiles: {n_total} total, {n_tissue} processed, {n_skipped} skipped"
    )
    bar_w = 30
    for stage, dt in timings.items():
        pct = 100 * dt / total
        bar = "\u2588" * int(bar_w * pct / 100)
        print(f"  {stage:<25} {dt:>7.2f}s  {pct:>5.1f}%  {bar}")
    print(f"  {'TOTAL':<25} {total:>7.2f}s")

    return label_map
