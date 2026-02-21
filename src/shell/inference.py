# SPDX-FileCopyrightText: 2024-present barrettMCW <mjbarrett@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""
Sliding-window inference for whole-slide EHO images.
"""

from __future__ import annotations

import gc

import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from torch.amp import autocast

# ---------------------------------------------------------------------------
# Default inference parameters
# ---------------------------------------------------------------------------
VAL_ROI_SIZE: tuple[int, int] = (512, 512)
VAL_SW_BATCH: int = 16
MAX_DIM: int = 8192


def run_inference(
    eho_image: np.ndarray,
    model: torch.nn.Module,
    device: torch.device | str = "cpu",
    *,
    roi_size: tuple[int, int] = VAL_ROI_SIZE,
    sw_batch_size: int = VAL_SW_BATCH,
    max_dim: int = MAX_DIM,
    overlap: float = 0.0,
) -> np.ndarray:
    """Run sliding-window inference on an EHO (H, W, 3) uint8 image.

    :param eho_image: (H, W, 3) uint8 EHO image.
    :param model: trained SegResNetVAE in eval mode.
    :param device: computation device.
    :param roi_size: sliding-window patch size.
    :param sw_batch_size: number of patches per forward pass.
    :param max_dim: maximum spatial dimension (centre-crop if exceeded).
    :param overlap: fraction of overlap between sliding-window patches.
    :return: (H, W) uint8 label map.
    """
    if isinstance(device, str):
        device = torch.device(device)

    h_orig, w_orig = eho_image.shape[:2]
    y0, x0 = 0, 0

    h_use, w_use = min(h_orig, max_dim), min(w_orig, max_dim)
    if h_use < h_orig or w_use < w_orig:
        y0 = (h_orig - h_use) // 2
        x0 = (w_orig - w_use) // 2
        eho_image = eho_image[y0 : y0 + h_use, x0 : x0 + w_use]

    # HWC uint8 → NCHW float32 [0, 1]
    img_t = (
        torch.from_numpy(eho_image)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div_(255.0)
    )
    del eho_image

    # Pad to multiple of 64
    h, w = img_t.shape[-2:]
    pad_h = (64 - h % 64) % 64
    pad_w = (64 - w % 64) % 64
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    if pad_h or pad_w:
        img_t = F.pad(img_t, padding, "constant", 0)

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.inference_mode(), autocast(amp_device):
        logits = sliding_window_inference(
            img_t,
            roi_size,
            sw_batch_size,
            model,
            overlap=overlap,
            sw_device=device,
            device=torch.device("cpu"),
            mode="constant",
        )
    del img_t

    # Remove padding
    if pad_h or pad_w:
        _, _, ph, pw = logits.shape
        logits = logits[
            :, :, padding[2] : ph - padding[3], padding[0] : pw - padding[1]
        ]

    pred = torch.argmax(logits[0], dim=0).numpy().astype(np.uint8)
    del logits
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Embed back into full canvas if we centre-cropped
    if h_use < h_orig or w_use < w_orig:
        full_pred = np.zeros((h_orig, w_orig), dtype=np.uint8)
        full_pred[y0 : y0 + h_use, x0 : x0 + w_use] = pred
        return full_pred

    return pred
