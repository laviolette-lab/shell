# SPDX-FileCopyrightText: 2024-present barrettMCW <mjbarrett@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""
End-to-end inference on a local RGB whole-slide TIFF.

Pipeline:
  1. Read the raw RGB .tiff WSI
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
import os

import numpy as np
import openslide
import pyvips

from shell.inference import run_inference
from shell.model import CLASS_NAMES, build_model
from shell.preprocessing import (
    apply_eho_chunked,
    compute_background_intensity,
    detect_background,
    estimate_stain_params,
)

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
TARGET_MPP: float = 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def preprocess_wsi(
    image_path: str,
    *,
    target_mpp: float = TARGET_MPP,
    stain_downsample: int = 4,
) -> np.ndarray:
    """Read a raw RGB WSI, scale to *target_mpp*, and produce an EHO image.

    :param image_path: path to an RGB ``.tiff`` whole-slide image.
    :param target_mpp: desired microns-per-pixel.
    :param stain_downsample: downsample factor for stain parameter estimation.
    :return: (H, W, 3) uint8 EHO image.
    """
    # 1. Get MPP metadata
    slide_os = openslide.OpenSlide(image_path)
    mpp_x = float(slide_os.properties.get(openslide.PROPERTY_NAME_MPP_X, 1.0))
    mpp_y = float(slide_os.properties.get(openslide.PROPERTY_NAME_MPP_Y, 1.0))
    slide_os.close()

    # 2. Scale to target MPP
    vips_image = pyvips.Image.new_from_file(image_path)
    scale_x = target_mpp / mpp_x
    scale_y = target_mpp / mpp_y
    scaled = vips_image.resize(1.0 / scale_x, vscale=1.0 / scale_y, kernel="lanczos3")
    image_np: np.ndarray = scaled.numpy()
    del vips_image, scaled

    # 3. Estimate stain parameters on a down-sampled copy
    ds = stain_downsample
    rgb_small = image_np[::ds, ::ds]
    bg_small = detect_background(rgb_small)
    sp = estimate_stain_params(rgb_small, bg_mask=bg_small)
    del rgb_small, bg_small
    gc.collect()

    # 4. Apply EHO colour transform (chunked)
    eho = apply_eho_chunked(image_np, **sp)
    del image_np, sp
    gc.collect()

    return eho


def infer_wsi(
    input_path: str,
    output_path: str,
    model_path: str | None = None,
    *,
    model_version: str | None = None,
    target_mpp: float = TARGET_MPP,
    save_eho: str | None = None,
    stain_downsample: int = 4,
    device: str = "auto",
) -> np.ndarray:
    """Full pipeline: preprocess → model → label image.

    :param input_path: raw RGB .tiff WSI.
    :param output_path: where to save the uint8 label TIFF.
    :param model_path: path to trained ``.pth`` weights, or ``None`` to
        use bundled weights.
    :param model_version: version tag for bundled weights (ignored when
        *model_path* is set).
    :param target_mpp: desired resolution.
    :param save_eho: optional path to save the intermediate EHO image.
    :param stain_downsample: downsample factor for stain estimation.
    :param device: ``"auto"``, ``"cpu"``, or ``"cuda"``.
    :return: (H, W) uint8 label map.
    """
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Preprocess
    eho = preprocess_wsi(
        input_path, target_mpp=target_mpp, stain_downsample=stain_downsample
    )

    if save_eho:
        os.makedirs(os.path.dirname(save_eho) or ".", exist_ok=True)
        pyvips.Image.new_from_array(eho).tiffsave(save_eho)

    # 2. Load model + inference
    model = build_model(model_path, device, model_version=model_version)
    label_map = run_inference(eho, model, device)
    del eho, model
    gc.collect()

    # 3. Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pyvips.Image.new_from_array(label_map).tiffsave(output_path)

    return label_map
