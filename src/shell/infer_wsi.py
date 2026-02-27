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

import numpy as np

# shell.inference and shell.model load torch at their module level.
# pyvips must come *after* them so PyTorch initialises its thread-pool
# runtime before libvips creates its own.  On macOS the reverse order
# (pyvips before torch) causes a segfault because both runtimes race to
# own the same OpenMP/GCD thread infrastructure.
from shell.inference import run_inference
from shell.model import build_model
from shell.preprocessing import (
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


def _vips_to_rgb_numpy(vips_img: pyvips.Image) -> np.ndarray:
    """Convert a pyvips image to (H, W, 3) uint8 RGB numpy array."""
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
# Public API
# ---------------------------------------------------------------------------
def preprocess_wsi(
    image_path: str,
    *,
    target_mpp: float = TARGET_MPP,
    mpp: float | None = None,
    stain_downsample: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a raw RGB image, scale to *target_mpp*, and produce an EHO image.

    :param image_path: path to an RGB image (TIFF, PNG, JPEG, etc.).
    :param target_mpp: desired microns-per-pixel.
    :param mpp: manual override for the source image um/px.  When
        ``None`` the value is read from slide metadata; if metadata is
        unavailable (e.g. plain PNG) a warning is emitted and scaling is
        skipped (the image is assumed to already be at *target_mpp*).
    :param stain_downsample: downsample factor for stain parameter estimation.
    :return: tuple of (H, W, 3) uint8 EHO image and (H, W) bool tissue mask.
    """
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
        vips_img = img_or_vips
        if needs_scaling:
            vips_img = vips_img.resize(
                1.0 / scale_x, vscale=1.0 / scale_y, kernel="lanczos3"
            )
        image_np = _vips_to_rgb_numpy(vips_img)
        del vips_img
    else:
        # numpy array from openslide fallback
        image_np = img_or_vips
        if needs_scaling:
            vips_tmp = pyvips.Image.new_from_array(image_np)
            vips_tmp = vips_tmp.resize(
                1.0 / scale_x, vscale=1.0 / scale_y, kernel="lanczos3"
            )
            image_np = _vips_to_rgb_numpy(vips_tmp)
            del vips_tmp

    # 4. Estimate stain parameters on a down-sampled copy
    ds = stain_downsample
    rgb_small = image_np[::ds, ::ds]
    bg_small = detect_background(rgb_small)
    sp = estimate_stain_params(rgb_small, bg_mask=bg_small)
    del rgb_small, bg_small
    gc.collect()

    # 4b. Build tissue mask from the background mask at full resolution
    bg_full = detect_background(image_np)
    tissue_mask = ~bg_full
    del bg_full

    # 5. Apply EHO colour transform (chunked)
    eho = apply_eho_chunked(image_np, **sp)
    del image_np, sp
    gc.collect()

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
    stain_downsample: int = 4,
    device: str = "auto",
) -> np.ndarray:
    """Full pipeline: preprocess -> model -> label image.

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
    :param stain_downsample: downsample factor for stain estimation.
    :param device: ``"auto"``, ``"cpu"``, or ``"cuda"``.
    :return: (H, W) uint8 label map at the original input resolution.
    """
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # 1. Preprocess
    eho, tissue_mask = preprocess_wsi(
        input_path,
        target_mpp=target_mpp,
        mpp=mpp,
        stain_downsample=stain_downsample,
    )

    if save_eho:
        os.makedirs(os.path.dirname(save_eho) or ".", exist_ok=True)
        pyvips.Image.new_from_array(eho).write_to_file(save_eho)

    # 2. Load model + inference
    model = build_model(model_path, device, model_version=model_version)
    label_map = run_inference(
        eho,
        model,
        device,
        tissue_mask=tissue_mask,
    )
    del eho, model, tissue_mask
    gc.collect()

    # 3. Always return/save at original input resolution.
    input_h, input_w = _read_image_size(input_path)
    label_map = _resize_label_map_nearest(label_map, input_h, input_w)

    # 4. Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pyvips.Image.new_from_array(label_map).write_to_file(output_path)

    return label_map
