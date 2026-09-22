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
import math
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
from shell.model import TILE_SIZE, _detect_cpu_budget, build_model
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

# Bounded to the real (cgroup-aware) CPU budget: a fixed guess like 12 can
# already oversubscribe a quota-limited container, and stacking it on top of
# ONNX Runtime's own thread pool during real inference makes it much worse.
pyvips.concurrency_set(_detect_cpu_budget())

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


def _onnx_cuda_available() -> bool:
    """Return whether ONNX Runtime advertises its CUDA execution provider."""
    try:
        import onnxruntime
    except ImportError:
        return False
    return "CUDAExecutionProvider" in onnxruntime.get_available_providers()


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


def _passes_tissue_threshold(fraction: float, min_tissue_frac: float) -> bool:
    """Apply the tissue-fraction gate, treating 0 as "any tissue at all"."""
    return fraction > 0 if min_tissue_frac <= 0 else fraction >= min_tissue_frac


def _tile_positions(
    length: int,
    tile_size: int,
    min_overlap: float = 0.25,
    max_overlap: float = 0.5,
) -> tuple[list[int], int]:
    """Return evenly spaced tile starts covering *length*.

    Uses the fewest tiles whose overlap fraction stays within
    ``[min_overlap, max_overlap]`` of *tile_size*. All adjacent tiles share
    the same pixel overlap, so the center-crop stitching seam is identical
    everywhere instead of one odd-sized tile snapped onto the end.

    :return: ``(starts, overlap_px)``.
    """
    if length <= tile_size:
        return [0], 0
    if not 0 <= min_overlap <= max_overlap < 1:
        raise ValueError("require 0 <= min_overlap <= max_overlap < 1")

    step_min = max(1, round(tile_size * (1 - max_overlap)))
    step_max = max(step_min, round(tile_size * (1 - min_overlap)))
    span = length - tile_size
    n_steps = max(1, math.ceil(span / step_max))
    step = span / n_steps
    last_start = length - tile_size
    starts = sorted({min(round(i * step), last_start) for i in range(n_steps + 1)})
    overlap_px = tile_size - round(step)
    return starts, overlap_px

import math
import numpy as np

def _mask_aware_tile_positions(
    tissue_mask: np.ndarray,
    y0: int,
    y1: int,
    tile_size: int,
    min_overlap: float,
    max_overlap: float,
    min_tissue_frac: float,
) -> list[tuple[list[int], int, int]]:
    """Return per-tissue-run tile schedules for this row band.

    Each element is ``(x0_list, overlap_px, optimal_y0)`` for one contiguous 
    tissue run in ``tissue_mask[y0:y1]``. The Y-coordinate is dynamically 
    shifted to perfectly center the tile over the tissue's true vertical bounds 
    within the run.
    """
    height, width = tissue_mask.shape
    if y0 >= y1 or width == 0:
        return []
        
    if width <= tile_size:
        fraction = float(tissue_mask[y0:y1].mean())
        # For a single tile spanning the whole width, we can also vertically center it
        row_has_tissue = tissue_mask[y0:y1].any(axis=1)
        if row_has_tissue.any():
            y_local_min = int(np.argmax(row_has_tissue))
            y_local_max = int(len(row_has_tissue) - 1 - np.argmax(row_has_tissue[::-1]))
            tissue_height = (y_local_max + 1) - y_local_min
            optimal_y0 = (y0 + y_local_min) - max(0, (tile_size - tissue_height) // 2)
            optimal_y0 = max(0, min(optimal_y0, height - tile_size))
        else:
            optimal_y0 = y0
            
        if _passes_tissue_threshold(fraction, min_tissue_frac):
            return [([0], 0, optimal_y0)]
        return []

    column_has_tissue = tissue_mask[y0:y1].any(axis=0)
    transitions = np.diff(np.r_[False, column_has_tissue, False].astype(np.int8))
    run_starts = np.flatnonzero(transitions == 1)
    run_ends = np.flatnonzero(transitions == -1)

    step_min = max(1, round(tile_size * (1 - max_overlap)))
    pad = step_min // 2
    last_start = width - tile_size

    runs: list[tuple[list[int], int, int]] = []
    for run_start, run_end in zip(run_starts, run_ends, strict=True):
        coverage_start = min(max(0, int(run_start) - pad), last_start)
        coverage_end = min(width, int(run_end) + pad)
        span = coverage_end - coverage_start - tile_size

        # 1. Find the tight Y-bounds of the tissue inside this specific horizontal run
        run_mask = tissue_mask[y0:y1, coverage_start:coverage_end]
        row_has_tissue = run_mask.any(axis=1)
        
        if not row_has_tissue.any():
            continue  # Edge case: padding missed tissue entirely
            
        y_local_min = int(np.argmax(row_has_tissue))
        y_local_max = int(len(row_has_tissue) - 1 - np.argmax(row_has_tissue[::-1]))
        
        true_y0 = y0 + y_local_min
        tissue_height = (y_local_max + 1) - y_local_min
        
        # 2. Center the tile vertically over the tissue block and clamp to image bounds
        optimal_y0 = true_y0 - max(0, (tile_size - tissue_height) // 2)
        optimal_y0 = max(0, min(optimal_y0, height - tile_size))

        # 3. Calculate horizontal scheduling
        if span <= 0:
            starts = [coverage_start]
            overlap_px = 0
        else:
            step_max = max(step_min, round(tile_size * (1 - min_overlap)))
            n_steps = max(1, math.ceil(span / step_max))
            step = span / n_steps
            starts = sorted(
                {
                    min(round(coverage_start + i * step), last_start)
                    for i in range(n_steps + 1)
                }
            )
            overlap_px = max(0, tile_size - round(step))

        # 4. Filter tiles based on the NEW vertically-shifted coordinate
        kept = [
            x0
            for x0 in starts
            if _passes_tissue_threshold(
                float(tissue_mask[
                    optimal_y0 : min(optimal_y0 + tile_size, height), 
                    x0 : min(x0 + tile_size, width)
                ].mean()),
                min_tissue_frac,
            )
        ]
        
        if kept:
            runs.append((kept, overlap_px, optimal_y0))

    return runs
import math
import numpy as np
from itertools import product

def _get_global_1d_schedule(
    start: int, 
    end: int, 
    tile_size: int, 
    min_overlap: float, 
    max_overlap: float, 
    max_bound: int
) -> list[int]:
    """
    Computes a mathematically consistent 1D grid spanning exactly from `start` to `end`.
    By locking this to the global bounds, we guarantee perfect phase alignment
    across the entire image, preventing colliding center-crops.
    """
    span = end - start
    last_start = max(0, max_bound - tile_size)

    if span <= tile_size:
        # A single tile is sufficient. Center it over the tissue span.
        opt_start = start + (span - tile_size) // 2
        return [max(0, min(opt_start, last_start))]

    step_min = max(1, round(tile_size * (1 - max_overlap)))
    step_max = max(step_min, round(tile_size * (1 - min_overlap)))

    # Determine the distance from the first tile's left edge to the last tile's left edge
    dist = span - tile_size
    
    # If the span is awkwardly sized (just over a tile width), force a minimum step
    # to prevent a severe maximum overlap violation.
    if dist < step_min:
        dist = step_min
        start = start + (span - tile_size - dist) // 2

    # Calculate exact number of steps required across the valid distance
    n_steps = max(1, math.ceil(dist / step_max))
    step = dist / n_steps
    
    # Generate the strict lattice sequence
    return [
        max(0, min(round(start + i * step), last_start))
        for i in range(n_steps + 1)
    ]

def get_mask_aware_tile_positions(
    tissue_mask: np.ndarray,
    tile_size: int,
    min_overlap: float,
    max_overlap: float,
    min_tissue_frac: float,
) -> list[tuple[int, int]]:
    """Return globally phase-locked (x0, y0) tile coordinates for the entire image."""
    height, width = tissue_mask.shape
    if height == 0 or width == 0:
        return []

    # 1. Find the strict global bounding box of ALL tissue.
    row_any = tissue_mask.any(axis=1)
    col_any = tissue_mask.any(axis=0)
    
    if not row_any.any():
        return []
        
    y_start, y_end = int(np.argmax(row_any)), int(height - np.argmax(row_any[::-1]))
    x_start, x_end = int(np.argmax(col_any)), int(width - np.argmax(col_any[::-1]))

    # 2. Generate a single, unified grid lattice for the entire tissue bounds.
    y_coords = _get_global_1d_schedule(y_start, y_end, tile_size, min_overlap, max_overlap, height)
    x_coords = _get_global_1d_schedule(x_start, x_end, tile_size, min_overlap, max_overlap, width)

    # 3. Filter the Cartesian grid, dropping tiles that fall strictly on background space.
    kept_positions = []
    for y0, x0 in product(y_coords, x_coords):
        tile_crop = tissue_mask[y0 : y0 + tile_size, x0 : x0 + tile_size]
        if float(tile_crop.mean()) >= min_tissue_frac:
            kept_positions.append((x0, y0))

    return kept_positions

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
    min_tissue_frac: float = 0.05,
    min_overlap: float = 0.25,
    max_overlap: float = 0.5,
    edge_thickness_px: int = 32,
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
    :param min_tissue_frac: minimum tissue fraction required to infer a tile.
        The default ``0.05`` skips tiles with only sparse tissue. Use ``0.0``
        to infer every tile containing any tissue pixel.
    :param min_overlap: minimum accepted overlap between adjacent tiles, as a
        fraction of the tile size. Default ``0.25``.
    :param max_overlap: maximum accepted overlap between adjacent tiles, as a
        fraction of the tile size. Within this ``[min_overlap, max_overlap]``
        band, the scheduler picks the fewest tiles needed for full coverage.
        Default ``0.5``.
    :param edge_thickness_px: WSI-only tissue boundary thickness to relabel
        as edge epithelium. Set to ``0`` to disable.
    :param device: ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.
    :return: (H, W) uint8 label map at the original input resolution.
    """
    from time import perf_counter

    import torch

    from shell.inference import GaussianMaskAwareInference

    timings: dict[str, float] = {}

    if device == "auto":
        onnx_model = model_path is None or model_path.lower().endswith(".onnx")
        if torch.cuda.is_available() or (onnx_model and _onnx_cuda_available()):
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

    # 2a. Create one bounded analysis thumbnail for both preprocessing steps.
    t0 = perf_counter()
    analysis_scale = min(1.0, 2048 / max(H, W))
    small_thumb_np = np.ascontiguousarray(
        vips_full.resize(analysis_scale).numpy()[:, :, :3]
    )
    timings["thumbnail"] = perf_counter() - t0

    # 2b. Tissue mask on small thumbnail -> upscale to full resolution
    t0 = perf_counter()
    bg_mask_small = detect_background(small_thumb_np)
    tissue_small = ~bg_mask_small
    th_h, th_w = tissue_small.shape
    mask_image = pyvips.Image.new_from_memory(
        np.ascontiguousarray(tissue_small, dtype=np.uint8).tobytes(),
        th_w,
        th_h,
        1,
        "uchar",
    )
    tissue_mask_full = (
        mask_image.resize(W / th_w, vscale=H / th_h, kernel="nearest")
        .numpy()
        .astype(bool)
    )
    del bg_mask_small, tissue_small
    timings["tissue_mask"] = perf_counter() - t0
    tissue_pct = 100 * tissue_mask_full.mean()
    log.info("Tissue: %.1f%%", tissue_pct)

    # 2c. Stain parameters on a bounded thumbnail.
    #     Calibration quality does not improve enough on giant thumbnails to
    #     justify the quadratic memory and CPU cost.
    t0 = perf_counter()
    stain_thumb_np = small_thumb_np
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

    del stain_thumb_np, stain_bg, small_thumb_np

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

    tile_size = TILE_SIZE[0]
    if not 0 <= min_overlap <= max_overlap < 1:
        raise ValueError("require 0 <= min_overlap <= max_overlap < 1")
    inference_engine = GaussianMaskAwareInference(
        model,
        device_obj,
        roi_size=(tile_size, tile_size),
        overlap=0.5,
    )
    if min_tissue_frac < 0 or min_tissue_frac > 1:
        raise ValueError("min_tissue_frac must be between 0 and 1")

    # Get the flat list of optimized 2D coordinates
    tile_positions = get_mask_aware_tile_positions(
        tissue_mask_full,
        tile_size,
        min_overlap,
        max_overlap,
        min_tissue_frac,
    )

    inner_pred = np.zeros((H, W), dtype=bool)
    outer_pred = np.zeros((H, W), dtype=bool)
    hematoxylin_full = np.zeros((H, W), dtype=np.uint8)

    # Only allocate full EHO when the user wants it saved
    eho_full = np.zeros((H, W, 3), dtype=np.uint8) if save_eho else None

    # Estimate roughly what the old grid size would have been for skipped stat reporting
    step_est = max(1, round(tile_size * (1 - max_overlap)))
    n_total = max(1, math.ceil(H / step_est)) * max(1, math.ceil(W / step_est))
    n_tissue = 0
    
    t_fetch = 0.0
    t_model = 0.0
    t_stitch = 0.0
    t_loop_start = perf_counter()
    progress_every = 10

    # Fetch and process independently using the flattened coordinate list
    for x0, y0 in tile_positions:
        y1 = min(y0 + tile_size, H)
        x1 = min(x0 + tile_size, W)
        tw = x1 - x0
        th = y1 - y0

        # ── fetch the exact RGB tile via pyvips random access ──
        _tf = perf_counter()
        rgb_tile = np.ascontiguousarray(
            vips_full.crop(x0, y0, tw, th).numpy()[:, :, :3]
        )
        eho_tile = apply_eho_chunked(
            rgb_tile.astype(np.uint8),
            chunk_rows=th,  # whole tile at once
            **stain_params,
        )
        del rgb_tile
        t_fetch += perf_counter() - _tf

        # ── Gaussian sliding-window inference ──
        _te = perf_counter()
        n_tissue += 1
        inner_tile, outer_tile = inference_engine.predict_tile(eho_tile)
        t_model += perf_counter() - _te

        if n_tissue % progress_every == 0:
            elapsed = perf_counter() - t_loop_start
            rate = n_tissue / elapsed
            log.info(
                "Progress: %d tiles processed (%.1f tiles/s, %.0fs elapsed)",
                n_tissue,
                rate,
                elapsed,
            )

        # ── stitch via max-pooling / OR ──
        _ts = perf_counter()
        
        # Store continuous hematoxylin / eho channels
        hematoxylin_full[y0:y1, x0:x1] = eho_tile[:, :, 1]
        if eho_full is not None:
            eho_full[y0:y1, x0:x1] = eho_tile
            
        del eho_tile
        
        # Logical OR the boolean outputs. Since GaussianMaskAwareInference 
        # suppresses the probabilities at tile edges before thresholding,
        # max-pooling (logical OR) overlapping tiles is perfectly safe.
        inner_pred[y0:y1, x0:x1] |= inner_tile
        outer_pred[y0:y1, x0:x1] |= outer_tile
        
        t_stitch += perf_counter() - _ts

    n_skipped = max(0, n_total - n_tissue)
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
        n_total,
        n_tissue,
        n_skipped,
        100 * n_skipped / max(n_total, 1),
    )

    # ── Phase 5: Save EHO intermediate ───────────────────────────────
    if save_eho and eho_full is not None:
        t0 = perf_counter()
        os.makedirs(os.path.dirname(save_eho) or ".", exist_ok=True)
        eho_full = np.ascontiguousarray(eho_full)
        log.info(
            "Saving EHO: shape=%s dtype=%s C_contig=%s",
            eho_full.shape,
            eho_full.dtype,
            eho_full.flags["C_CONTIGUOUS"],
        )
        eho_params = {"lossless": True} if save_eho.lower().endswith(".jp2") else {}
        pyvips.Image.new_from_array(eho_full.astype(np.uint8)).write_to_file(
            save_eho, **eho_params
        )
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
        edge_thickness_px=edge_thickness_px,
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
    log.info(
        "Saving label_map: shape=%s dtype=%s C_contig=%s",
        label_map.shape,
        label_map.dtype,
        label_map.flags["C_CONTIGUOUS"],
    )
    params = {"lossless": True} if output_path.lower().endswith(".jp2") else {}
    pyvips.Image.new_from_array(label_map.astype(np.uint8)).write_to_file(
        output_path, **params
    )
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
        log.info(
            "Saving raw_bands: shape=%s dtype=%s C_contig=%s",
            raw_bands.shape,
            raw_bands.dtype,
            raw_bands.flags["C_CONTIGUOUS"],
        )
        raw_params = {"lossless": True} if save_raw.lower().endswith(".jp2") else {}
        pyvips.Image.new_from_array(raw_bands.astype(np.uint8)).write_to_file(
            save_raw, **raw_params
        )
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
