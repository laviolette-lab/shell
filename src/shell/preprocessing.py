# SPDX-FileCopyrightText: 2024-present barrettMCW <mjbarrett@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""
Colour preprocessing for H&E whole-slide images.

This module contains functions for:

* Background / white-region detection (brightness + saturation + local entropy)
* Macenko PCA stain-parameter estimation
* Chunked
 EHO (Eosin, Hematoxylin, Optical-Density) image construction
"""

from __future__ import annotations

from typing import Any

import numpy as np
from macenko_pca import (
    find_stain_index,
    rgb_separate_stains_macenko_pca,
    stain_color_map,
)
from scipy.ndimage import label, uniform_filter
from scipy.ndimage import sum as ndimage_sum

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
WHITE_BRIGHTNESS_THRESHOLD: float = 0.82
WHITE_SATURATION_MAX: float = 0.25
ENTROPY_WINDOW: int = 9
ENTROPY_THRESHOLD: float = 2.4
MIN_LUMEN_AREA: int = 128


# ---------------------------------------------------------------------------
# Background-mask detection
# ---------------------------------------------------------------------------
def detect_background(
    image_np: np.ndarray,
    *,
    brightness_threshold: float = WHITE_BRIGHTNESS_THRESHOLD,
    saturation_max: float = WHITE_SATURATION_MAX,
    entropy_window: int = ENTROPY_WINDOW,
    entropy_threshold: float = ENTROPY_THRESHOLD,
    min_lumen_area: int = MIN_LUMEN_AREA,
) -> np.ndarray:
    """Build a boolean background mask from an RGB uint8 image.

    Uses a hybrid rule: bright + low saturation + low local Shannon entropy,
    with a conservative fallback to avoid over-masking tissue.

    :param image_np: (H, W, C) uint8 RGB image (C >= 3).
    :param brightness_threshold: minimum HSV V for background.
    :param saturation_max: maximum HSV S for background (entropy branch).
    :param entropy_window: side length of the uniform-filter window for entropy.
    :param entropy_threshold: maximum local entropy for background.
    :param min_lumen_area: minimum connected-component area to retain.
    :return: boolean mask, ``True`` where background/lumen detected.
    """
    # HSV V and S from RGB directly (avoids full float64 HSV conversion)
    cmax = np.max(image_np[..., :3], axis=2)
    cmin = np.min(image_np[..., :3], axis=2)
    val = cmax.astype(np.float32) / 255.0
    diff = cmax.astype(np.float32) - cmin.astype(np.float32)
    del cmin
    sat = np.zeros_like(val)
    nz = cmax > 0
    sat[nz] = diff[nz] / cmax[nz].astype(np.float32)
    del diff, nz, cmax

    # Local Shannon entropy on quantized grayscale (16 bins)
    gray_q = (
        0.299 * image_np[..., 0] + 0.587 * image_np[..., 1] + 0.114 * image_np[..., 2]
    ).astype(np.uint8) >> 4
    entropy_map = np.zeros(gray_q.shape, dtype=np.float32)
    for b in range(16):
        bin_mask = (gray_q == b).astype(np.float32)
        p = uniform_filter(bin_mask, size=entropy_window, mode="nearest")
        entropy_map -= p * np.log2(np.clip(p, 1e-6, 1.0))
        del bin_mask, p
    del gray_q

    # Hybrid white/lumen rule
    white_mask_entropy = (
        (val > brightness_threshold)
        & (sat < saturation_max)
        & (entropy_map < entropy_threshold)
    )
    # Conservative baseline (old behaviour) as safety fallback
    white_mask_base = (val > 0.83) & (sat < 0.12)

    frac_entropy = float(np.mean(white_mask_entropy))
    frac_base = float(np.mean(white_mask_base))
    if (
        frac_entropy < 0.01
        or frac_entropy > 0.90
        or frac_entropy > (frac_base * 1.8 + 1e-6)
    ):
        white_mask = white_mask_base
    else:
        white_mask = white_mask_entropy | (white_mask_base & (val > 0.94))
    del sat, val, entropy_map, white_mask_entropy, white_mask_base

    # Connected-component cleanup
    label_out = label(white_mask)
    if isinstance(label_out, tuple):
        labeled, num_features = label_out
    else:
        labeled = label_out
        num_features = int(np.max(labeled))

    if num_features > 0:
        comp_sizes = np.array(
            ndimage_sum(white_mask, labeled, range(1, num_features + 1))
        )
        keep = np.zeros(num_features + 1, dtype=bool)
        keep[(np.where(comp_sizes >= min_lumen_area)[0] + 1)] = True
        bg_mask: np.ndarray = keep[labeled]
        del keep, comp_sizes
    else:
        bg_mask = np.zeros(white_mask.shape, dtype=bool)
    del white_mask, labeled

    return bg_mask


# ---------------------------------------------------------------------------
# Background intensity (Io)
# ---------------------------------------------------------------------------
def compute_background_intensity(
    image_np: np.ndarray,
    bg_mask: np.ndarray,
) -> float:
    """Estimate the illumination intensity *Io* from background pixels.

    Pure-white (255, 255, 255) pixels are excluded to avoid scanner-artifact
    bias.

    :param image_np: (H, W, C) uint8 RGB image.
    :param bg_mask: boolean background mask.
    :return: estimated Io, clipped to [200, 255].
    """
    pure_white = (
        (image_np[..., 0] == 255)
        & (image_np[..., 1] == 255)
        & (image_np[..., 2] == 255)
    )
    bg_for_io = bg_mask & ~pure_white
    del pure_white

    if np.sum(bg_for_io) >= 100:
        io_val = float(
            np.clip(np.mean(image_np[bg_for_io], dtype=np.float64), 200, 255)
        )
    elif np.sum(bg_mask) >= 100:
        io_val = float(np.clip(np.mean(image_np[bg_mask], dtype=np.float64), 200, 255))
    else:
        io_val = 240.0
    del bg_for_io
    return io_val


# ---------------------------------------------------------------------------
# Stain-parameter estimation
# ---------------------------------------------------------------------------
def estimate_stain_params(
    rgb: np.ndarray, bg_mask: np.ndarray | None = None
) -> dict[str, Any]:
    """Estimate Macenko colour-deconvolution parameters from an RGB image.

    If *bg_mask* is not supplied it is computed automatically via
    :func:`detect_background`.

    :param rgb: (H, W, 3) uint8 RGB image.
    :param bg_mask: optional pre-computed background mask.
    :return: dict with keys ``Io``, ``w_est``, ``e_idx``, ``h_idx``,
        ``e_lo``, ``e_hi``, ``h_lo``, ``h_hi``, ``od_lo``, ``od_hi``.
    """
    if bg_mask is None:
        bg_mask = detect_background(rgb)

    Io = compute_background_intensity(rgb, bg_mask)

    w_est = rgb_separate_stains_macenko_pca(
        rgb,
        bg_int=[Io],
        mask_out=bg_mask,
    )

    h_idx = find_stain_index(
        stain_color_map["hematoxylin"],
        w_est,
    )
    e_idx = find_stain_index(
        stain_color_map["eosin"],
        w_est,
    )
    if h_idx == e_idx or h_idx == 2 or e_idx == 2:
        h_idx, e_idx = 0, 1

    # Robust stain-channel percentiles for normalisation
    Io_f = np.float32(Io)
    W_inv = np.linalg.pinv(w_est).astype(np.float32)
    e_vec = W_inv[e_idx]
    h_vec = W_inv[h_idx]

    od = np.clip(rgb.astype(np.float32), 1.0, Io_f)
    od /= Io_f
    np.log10(od, out=od)
    od *= -1.0

    e_conc = np.einsum("ijk,k->ij", od, e_vec)
    h_conc = np.einsum("ijk,k->ij", od, h_vec)
    e_lo = float(np.percentile(e_conc, 1))
    e_hi = float(np.percentile(e_conc, 99))
    h_lo = float(np.percentile(h_conc, 1))
    h_hi = float(np.percentile(h_conc, 99))
    del e_conc, h_conc

    if e_hi <= e_lo + 1e-8:
        e_lo, e_hi = 0.0, 1.0
    if h_hi <= h_lo + 1e-8:
        h_lo, h_hi = 0.0, 1.0

    # OD channel percentiles
    od_acc = np.zeros(rgb.shape[:2], dtype=np.float32)
    for c in range(3):
        ch = np.clip(rgb[..., c].astype(np.float32), 1.0, Io_f)
        ch /= Io_f
        np.log10(ch, out=ch)
        od_acc -= ch
    od_acc /= 3.0
    od_lo = float(np.percentile(od_acc, 1))
    od_hi = float(np.percentile(od_acc, 99))
    del od_acc

    return {
        "Io": Io,
        "w_est": w_est,
        "e_idx": e_idx,
        "h_idx": h_idx,
        "e_lo": e_lo,
        "e_hi": e_hi,
        "h_lo": h_lo,
        "h_hi": h_hi,
        "od_lo": od_lo,
        "od_hi": od_hi,
    }


# ---------------------------------------------------------------------------
# Chunked EHO construction
# ---------------------------------------------------------------------------
def apply_eho_chunked(
    image_np: np.ndarray,
    Io: float,
    w_est: np.ndarray,
    e_idx: int,
    h_idx: int,
    e_lo: float,
    e_hi: float,
    h_lo: float,
    h_hi: float,
    od_lo: float,
    od_hi: float,
    chunk_rows: int = 512,
) -> np.ndarray:
    """Build a 3-channel EHO image memory-efficiently in row chunks.

    Instead of calling a full colour-deconvolution routine (which allocates
    a full float64 stain array), we compute the two stain concentrations via
    a float32 matrix multiply and write each chunk directly into the output
    uint8 buffer.

    :param image_np: (H, W, 3) uint8 RGB image.
    :param Io: background illumination intensity.
    :param w_est: (3, 3) stain matrix from Macenko.
    :param e_idx: column index for eosin in *w_est*.
    :param h_idx: column index for hematoxylin in *w_est*.
    :param e_lo: 1st percentile of eosin concentration.
    :param e_hi: 99th percentile of eosin concentration.
    :param h_lo: 1st percentile of hematoxylin concentration.
    :param h_hi: 99th percentile of hematoxylin concentration.
    :param od_lo: 1st percentile of mean optical density.
    :param od_hi: 99th percentile of mean optical density.
    :param chunk_rows: number of image rows per processing chunk.
    :return: (H, W, 3) uint8 EHO image.
    """
    H, W = image_np.shape[:2]
    eho = np.zeros((H, W, 3), dtype=np.uint8)

    Io_f = np.float32(Io)
    W_inv = np.linalg.pinv(w_est).astype(np.float32)
    e_vec = W_inv[e_idx]
    h_vec = W_inv[h_idx]
    e_range = np.float32(e_hi - e_lo + 1e-8)
    h_range = np.float32(h_hi - h_lo + 1e-8)
    e_lo_f = np.float32(e_lo)
    h_lo_f = np.float32(h_lo)
    od_range = np.float32(od_hi - od_lo + 1e-8)
    od_lo_f = np.float32(od_lo)

    for r0 in range(0, H, chunk_rows):
        r1 = min(H, r0 + chunk_rows)
        chunk = np.clip(image_np[r0:r1].astype(np.float32), 1.0, Io_f)
        chunk /= Io_f
        np.log10(chunk, out=chunk)
        chunk *= -1.0  # OD = −log10(I / Io)

        # Stain concentrations
        e_conc = np.einsum("ijk,k->ij", chunk, e_vec)
        h_conc = np.einsum("ijk,k->ij", chunk, h_vec)

        e_conc -= e_lo_f
        e_conc /= e_range
        np.clip(e_conc, 0, 1, out=e_conc)
        eho[r0:r1, :, 0] = (255 - (e_conc * 255)).astype(np.uint8)

        h_conc -= h_lo_f
        h_conc /= h_range
        np.clip(h_conc, 0, 1, out=h_conc)
        eho[r0:r1, :, 1] = (255 - (h_conc * 255)).astype(np.uint8)
        del e_conc, h_conc

        # OD channel
        od = chunk.mean(axis=2)
        del chunk
        od -= od_lo_f
        od /= od_range
        np.clip(od, 0, 1, out=od)
        eho[r0:r1, :, 2] = (255 - (od * 255)).astype(np.uint8)
        del od

    return eho
