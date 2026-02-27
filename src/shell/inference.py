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
import os
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
VAL_ROI_SIZE: tuple[int, int] = (2048, 2048)
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
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
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
    sw_batch_size: int | None = None,
    overlap: float = VAL_SW_OVERLAP,
    tissue_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Run sliding-window inference on an EHO (H, W, 3) uint8 image.

    :param eho_image: (H, W, 3) uint8 EHO image.
    :param model: trained SegResNetVAE in eval mode.
    :param device: computation device.
    :param roi_size: sliding-window patch size.
    :param sw_batch_size: number of patches per forward pass (None => auto).
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

    # Heuristic: choose a sliding-window batch size automatically when not provided.
    def _choose_sw_batch_size(roi: tuple[int, int], device_obj: torch.device) -> int:
        """Heuristic selection based on ROI area and device capabilities.

        Baseline: VAL_SW_BATCH for 512x512 on a mid-range GPU. Scale batch size
        inversely with ROI area and up-weight for devices with more memory
        (e.g. macOS unified memory / MPS).
        """
        base_area = 512 * 512
        roi_area = max(1, roi[0] * roi[1])
        area_ratio = base_area / roi_area

        # Try to estimate system RAM (in GB) on POSIX systems; conservative fallback.
        total_ram_gb: float | None = None
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            total_ram_gb = (pages * page_size) / (1024**3)
        except Exception:
            total_ram_gb = None

        # Device factor: raise batch size for GPUs, raise more for large unified RAM (MPS).
        if device_obj.type == "cuda":
            device_factor = 2.0
        elif device_obj.type == "mps":
            # macOS unified memory benefits from larger batches when system RAM is large.
            if total_ram_gb is not None:
                device_factor = max(1.5, min(8.0, total_ram_gb / 8.0))
            else:
                device_factor = 3.0
        else:
            # CPU: be conservative and scale with available RAM if known.
            if total_ram_gb is not None:
                device_factor = max(0.5, min(2.0, total_ram_gb / 32.0))
            else:
                device_factor = 0.5

        batch = max(1, int(VAL_SW_BATCH * area_ratio * device_factor))
        # Clamp to avoid enormous batches on pathological inputs.
        return min(max(batch, 1), 1024)

    local_sw_batch = (
        sw_batch_size
        if sw_batch_size is not None
        else _choose_sw_batch_size(roi_size, device_obj)
    )

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
            local_sw_batch,
            model,
            overlap=overlap,
            sw_device=device_obj,
            device=torch.device("cpu"),
            mode=blend_mode,
        )
        # MONAI's sliding_window_inference may return a Tensor, a tuple/list
        # (e.g. when the model returns multiple outputs), or a dict. Normalise
        # to a single Tensor here so downstream code and the type-checker see
        # a consistent type.
        if not isinstance(logits, torch.Tensor):
            # Tuple / list -> first element is expected to be the logits Tensor.
            if isinstance(logits, (tuple, list)):
                if len(logits) > 0 and isinstance(logits[0], torch.Tensor):
                    logits = logits[0]
                else:
                    # Fallback: try to coerce the first element to a tensor.
                    logits = torch.as_tensor(logits[0])
            elif isinstance(logits, dict):
                # Take the first tensor-like value from the dict.
                found = False
                for v in logits.values():
                    if isinstance(v, torch.Tensor):
                        logits = v
                        found = True
                        break
                if not found:
                    # Coerce the first value to a tensor as a last resort.
                    first_val = next(iter(logits.values()))
                    logits = torch.as_tensor(first_val)
            else:
                # If it's some other type, attempt a coercion to Tensor.
                logits = torch.as_tensor(logits)
    del img_t

    # Remove padding
    if pad_h or pad_w:
        # Ensure we have a Tensor before accessing shape (some MONAI
        # wrappers may return a tuple/dict). Normalise common container
        # types to a single Tensor.
        if not isinstance(logits, torch.Tensor):
            if (
                isinstance(logits, (tuple, list))
                and len(logits) > 0
                and isinstance(logits[0], torch.Tensor)
            ):
                logits = logits[0]
            elif isinstance(logits, dict):
                # take first tensor-like value
                found = False
                for v in logits.values():
                    if isinstance(v, torch.Tensor):
                        logits = v
                        found = True
                        break
                if not found:
                    # fall back to coercing first value
                    first_val = next(iter(logits.values()))
                    logits = torch.as_tensor(first_val)
            else:
                logits = torch.as_tensor(logits)

        _, _, ph, pw = logits.shape
        logits = logits[
            :, :, padding[2] : ph - padding[3], padding[0] : pw - padding[1]
        ]

    # ------------------------------------------------------------------
    # Post-processing: sigmoid + threshold + background comparison
    # ------------------------------------------------------------------
    # Ensure logits is a Tensor before applying sigmoid.
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)
    probs = torch.sigmoid(logits)
    del logits

    inner_prob = probs[0, 0]  # channel 0 = epithelium / inner
    outer_prob = probs[0, 1]  # channel 1 = stroma / outer
    bg_prob = probs[0, 2]  # channel 2 = background
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
