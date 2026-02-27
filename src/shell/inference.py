# SPDX-FileCopyrightText: 2024-present barrettMCW <mjbarrett@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""
Sliding-window inference for whole-slide EHO images.

The normalisation pipeline uses MONAI transforms (``Compose``) to match
the training-time preprocessing exactly::

    ScaleIntensityRanged(a_min=0, a_max=255, b_min=0, b_max=1)
    NormalizeIntensityd(nonzero=True, channel_wise=True)
    ScaleIntensityd()

Post-processing uses sigmoid activation with per-channel thresholding
against the background probability, producing a clean label map.

An optional *tissue_mask* can be supplied to zero out non-tissue regions
in the output, saving downstream processing.
"""

from __future__ import annotations

import gc
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureTyped,
    NormalizeIntensityd,
    ScaleIntensityd,
    ScaleIntensityRanged,
)
from torch.amp import autocast

# ---------------------------------------------------------------------------
# Default inference parameters
# ---------------------------------------------------------------------------
VAL_ROI_SIZE: tuple[int, int] = (512, 512)
VAL_SW_BATCH: int = 16
VAL_SW_OVERLAP: float = 0.25


# ---------------------------------------------------------------------------
# MONAI normalisation pipeline (matches training transforms)
# ---------------------------------------------------------------------------
def _build_normalize_pipeline() -> Compose:
    """Return a MONAI ``Compose`` that normalises an EHO ``image`` tensor.

    Expects ``data["image"]`` to be a (C, H, W) uint8-range tensor.
    """
    return Compose(
        [
            EnsureTyped(keys=["image"], track_meta=False),
            ScaleIntensityRanged(
                keys="image",
                a_min=0.0,
                a_max=255.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
            ScaleIntensityd(keys=["image"]),
        ]
    )


# Module-level singleton so the pipeline is built once.
_NORMALIZE: Compose | None = None


def _get_normalize() -> Compose:
    global _NORMALIZE
    if _NORMALIZE is None:
        _NORMALIZE = _build_normalize_pipeline()
    return _NORMALIZE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_inference(
    eho_image: np.ndarray,
    model: torch.nn.Module,
    device: torch.device | str = "cpu",
    *,
    roi_size: tuple[int, int] = VAL_ROI_SIZE,
    sw_batch_size: int = VAL_SW_BATCH,
    overlap: float = VAL_SW_OVERLAP,
    tissue_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Run sliding-window inference on an EHO (H, W, 3) uint8 image.

    :param eho_image: (H, W, 3) uint8 EHO image.
    :param model: trained SegResNetVAE in eval mode.
    :param device: computation device.
    :param roi_size: sliding-window patch size.
    :param sw_batch_size: number of patches per forward pass.
    :param overlap: fraction of overlap between sliding-window patches.
        Values > 0 enable Gaussian importance weighting so that
        overlapping patch centres contribute more than edges, which
        eliminates the grid artefacts visible with ``mode="constant"``.
    :param tissue_mask: optional boolean mask (H, W) where ``True``
        indicates tissue.  When supplied, non-tissue regions are forced
        to background (label 0) in the output.
    :return: (H, W) uint8 label map.
    """
    device_obj = torch.device(device) if isinstance(device, str) else device

    # HWC uint8 → CHW float tensor, then normalise via MONAI pipeline
    img_chw = torch.from_numpy(eho_image).permute(2, 0, 1).float()
    del eho_image

    normalise = _get_normalize()
    data = normalise({"image": img_chw})
    img_t = data["image"].unsqueeze(0)  # → NCHW
    del img_chw, data

    # Pad to multiple of 64 (required by SegResNet encoder)
    h, w = img_t.shape[-2:]
    pad_h = (64 - h % 64) % 64
    pad_w = (64 - w % 64) % 64
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    if pad_h or pad_w:
        img_t = F.pad(img_t, padding, "reflect")

    amp_device = "cuda" if device_obj.type == "cuda" else "cpu"
    with (
        torch.inference_mode(),
        autocast(amp_device),
        warnings.catch_warnings(),
    ):
        warnings.filterwarnings(
            "ignore",
            message="Using a non-tuple sequence for multidimensional indexing",
            category=UserWarning,
        )
        blend_mode = "gaussian" if overlap > 0 else "constant"
        logits = sliding_window_inference(
            img_t,
            roi_size,
            sw_batch_size,
            model,
            overlap=overlap,
            sw_device=device_obj,
            device=torch.device("cpu"),
            mode=blend_mode,
        )
    del img_t

    # Remove padding
    if pad_h or pad_w:
        _, _, ph, pw = logits.shape
        logits = logits[
            :, :, padding[2] : ph - padding[3], padding[0] : pw - padding[1]
        ]

    # ------------------------------------------------------------------
    # Post-processing: sigmoid + threshold + background comparison
    # ------------------------------------------------------------------
    probs = torch.sigmoid(logits)
    del logits

    inner_prob = probs[0, 0]  # channel 0 = epithelium / inner
    outer_prob = probs[0, 1]  # channel 1 = stroma / outer
    bg_prob = probs[0, 2]     # channel 2 = background
    del probs

    inner_mask = (inner_prob > 0.5) & (inner_prob > bg_prob)
    outer_mask = (outer_prob > 0.5) & (outer_prob > bg_prob)
    del inner_prob, outer_prob, bg_prob

    pred = np.zeros((inner_mask.shape[0], inner_mask.shape[1]), dtype=np.uint8)
    # Stroma first so epithelium takes priority at overlaps
    pred[outer_mask.numpy()] = 2
    pred[inner_mask.numpy()] = 1
    del inner_mask, outer_mask

    # Apply tissue mask: force non-tissue regions to background
    if tissue_mask is not None:
        pred[~tissue_mask[: pred.shape[0], : pred.shape[1]]] = 0

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pred
