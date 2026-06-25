# SPDX-FileCopyrightText: 2024-present barrettMCW <mjbarrett@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""
MONAI dictionary transforms for histology image preprocessing.

Provides composable ``MapTransform`` classes for:

* Resolution scaling  (``Rescaled``, ``LoadImageAtScaled``)
* Tissue mask generation  (``TissueMaskd``)
* Macenko colour deconvolution  (``MarcenkoDeconvolutiond``,
  ``MarcenkoDeconvolutionAndGrayd``)
* EHO (Eosin, Hematoxylin, Optical-Density) conversion  (``EHOd``)

Lower-level helper functions (``detect_background``,
``compute_background_intensity``, ``estimate_stain_params``,
``apply_eho_chunked``) are also exported so they can be used directly
or by other modules (e.g. the OMERO tile-by-tile pipeline).
"""

from __future__ import annotations

import logging

import macenko_pca
import numpy as np
from monai.data import MetaTensor
from monai.transforms import MapTransform, Resize
from scipy.ndimage import label, uniform_filter
from scipy.ndimage import sum as ndimage_sum


# Lightweight local replacement for skimage.color.rgb2gray to avoid an
# optional dependency and to keep typing simple. This implements the
# standard luminance weights and expects input in the [0, 1] float range
# (matching skimage behavior). It returns a 2D array of floats in [0, 1].
def rgb2gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    # Expect HWC or CHW: prefer HWC (common in this module)
    if image.ndim == 3 and image.shape[-1] == 3:
        # Use common luminance weights
        weights = np.array([0.2989, 0.5870, 0.1140], dtype=image.dtype)
        return np.dot(image[..., :3], weights)
    # Fallback: reduce any trailing channel dimension
    if image.ndim == 3:
        return np.mean(image, axis=-1)
    # Unexpected shape: coerce to 2D via mean
    return np.mean(image, axis=-1)


def _get_pyvips():
    try:
        import pyvips as pv
    except Exception as e:
        raise RuntimeError("pyvips is required for this operation") from e
    return pv


# ---------------------------------------------------------------------------
# Resolution-scaling transforms
# ---------------------------------------------------------------------------


class Rescaled(MapTransform):
    """Rescale images from *in_res* to *out_res* microns-per-pixel."""

    def __init__(self, keys, in_res=0.46, out_res=2):
        super().__init__(keys)
        self.in_res = in_res
        self.out_res = out_res

    def __call__(self, data, in_res=None, resize_kwargs=None):
        if in_res is None:
            in_res = self.in_res
        if resize_kwargs is None:
            resize_kwargs = {}
        for key in self.keys:
            image = data[key]
            height, width = image.shape[1:]
            new_height = height * in_res // self.out_res
            new_width = width * in_res // self.out_res

            interpolation = (
                "linear" if len(image.shape) == 3 and image.shape[0] == 3 else "nearest"
            )

            if "mode" not in resize_kwargs:
                resize_kwargs["mode"] = interpolation

            resize_transform = Resize(
                spatial_size=(new_height, new_width), **resize_kwargs
            )
            image_rescaled = resize_transform(image)
            data[key] = image_rescaled
        return data


class LoadImageAtScaled(MapTransform):
    """Load an image using pyvips and rescale it to a target resolution.

    The image is converted to a NumPy array and then to a
    :class:`~monai.data.MetaTensor` in **HWC** format (consistent with
    pyvips output).
    """

    def __init__(
        self,
        keys,
        in_res=None,
        target_res=2.0,
        reader_kwargs=None,
        resize_kernel="linear",
        allow_missing_keys=False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.in_res = in_res
        self.target_res = target_res
        self.reader_kwargs = reader_kwargs if reader_kwargs is not None else {}
        self.resize_kernel = resize_kernel

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            filepath = d[key]
            try:
                pv = _get_pyvips()
                image_pv = pv.Image.new_from_file(str(filepath), **self.reader_kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load image {filepath} with pyvips: {e}"
                ) from e

            original_width = image_pv.width
            original_height = image_pv.height
            applied_scale_factor = 1.0

            if (
                self.in_res is not None
                and self.target_res is not None
                and self.in_res != self.target_res
            ):
                scale_factor = self.in_res / self.target_res
                if abs(scale_factor - 1.0) > 1e-6:
                    image_pv = image_pv.resize(scale_factor, kernel=self.resize_kernel)
                    applied_scale_factor = scale_factor

            img_array = image_pv.numpy()

            original_channel_dim = "no_channel"
            if image_pv.bands == 3:
                original_channel_dim = 2

            meta_info = {
                "filename_or_obj": str(filepath),
                "original_spatial_shape": (original_height, original_width),
                "spatial_shape": img_array.shape[1:],
            }

            meta_info["original_channel_dim"] = original_channel_dim
            if self.in_res is not None:
                meta_info["original_resolution"] = self.in_res
            if self.target_res is not None:
                meta_info["target_resolution"] = self.target_res
            if applied_scale_factor != 1.0:
                meta_info["applied_geometric_scale_factor"] = applied_scale_factor

            d[key] = MetaTensor(img_array, meta=meta_info)
        return d


# ---------------------------------------------------------------------------
# Tissue-mask transform
# ---------------------------------------------------------------------------


class TissueMaskd(MapTransform):
    """Generate a boolean tissue mask and store it under ``{key}_tissue_mask``.

    Uses :func:`detect_background` (brightness + saturation + entropy)
    and inverts the result so ``True`` = tissue.
    """

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        for key in self.keys:
            image = data[key].numpy()
            if image.ndim == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            bg_mask = detect_background(image)
            tissue_mask = ~bg_mask
            meta_info = {"original_channel_dim": "no_channel"}
            data[f"{key}_tissue_mask"] = MetaTensor(tissue_mask, meta=meta_info)
        return data


# ---------------------------------------------------------------------------
# Macenko colour-deconvolution transforms
# ---------------------------------------------------------------------------


class MarcenkoDeconvolutiond(MapTransform):
    """Perform Macenko colour deconvolution on H&E histology images."""

    stain_color_map = macenko_pca.stain_color_map

    def __init__(self, keys, keep_residual=False, tissue_mask_keys=None):
        super().__init__(keys)
        self.keep_residual = keep_residual
        self.tissue_mask_keys = tissue_mask_keys

    def get_w_est(self, image, tissue_mask=None):
        """Estimate stain vectors using Macenko's method."""
        if image.size > 16777216:  # 4096^2
            logging.info("Resizing image to half size for faster processing")
            resizer = Resize(
                spatial_size=(image.shape[0] // 4, image.shape[1] // 4),
                mode="linear",
            )
            image = image.transpose((2, 0, 1))
            image = resizer(image)
            image = image.numpy().transpose((1, 2, 0))
            if tissue_mask is not None:
                tissue_mask = tissue_mask[None, ...]
                tissue_mask = resizer(tissue_mask)
                tissue_mask = tissue_mask[0]

        # If a tissue_mask was provided as a tensor/MetaTensor, coerce to numpy
        if tissue_mask is not None:
            tissue_mask = np.asarray(tissue_mask)

        if tissue_mask is None:
            bg_mask = detect_background(image)
            tissue_mask = np.ascontiguousarray(~bg_mask).astype(bool)

        # Ensure tissue_mask is a NumPy boolean array for downstream processing
        tissue_mask = np.asarray(tissue_mask).astype(bool)

        return macenko_pca.rgb_separate_stains_macenko_pca(
            np.ascontiguousarray(image),
            None,
            mask_out=~tissue_mask,
        )

    def deconv(self, image, tissue_mask=None):
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))

        w_est = self.get_w_est(image, tissue_mask=tissue_mask)

        image_her = macenko_pca.color_deconvolution(np.ascontiguousarray(image), w_est)

        hematox_index = macenko_pca.find_stain_index(
            self.stain_color_map["hematoxylin"], w_est
        )
        eosin_index = macenko_pca.find_stain_index(self.stain_color_map["eosin"], w_est)

        image_her = image_her[..., [hematox_index, eosin_index, 2]]

        if not self.keep_residual:
            image_her = image_her[..., :2]

        image_her = np.transpose(image_her, (2, 0, 1))
        image_her = MetaTensor(image_her)
        return image_her

    def __call__(self, data):
        tissue_mask_keys = self.tissue_mask_keys
        if tissue_mask_keys is not None:
            if isinstance(tissue_mask_keys, (str, int)):
                tissue_mask_keys = [tissue_mask_keys]

        for i, key in enumerate(self.keys):
            tissue_mask = None
            if tissue_mask_keys is not None:
                if len(tissue_mask_keys) == 1:
                    mask_key = tissue_mask_keys[0]
                elif i < len(tissue_mask_keys):
                    mask_key = tissue_mask_keys[i]
                else:
                    mask_key = None

                if mask_key is not None and mask_key in data:
                    tissue_mask = data[mask_key].squeeze().numpy() > 0

            image = data[key].numpy()
            data[key] = self.deconv(image, tissue_mask=tissue_mask)
        return data


class MarcenkoDeconvolutionAndGrayd(MarcenkoDeconvolutiond):
    """Macenko deconvolution with an additional grayscale channel."""

    def __init__(self, keys, tissue_mask_keys=None):
        super().__init__(
            keys=keys, keep_residual=True, tissue_mask_keys=tissue_mask_keys
        )

    def __call__(self, data):
        tissue_mask_keys = self.tissue_mask_keys
        if tissue_mask_keys is not None:
            if isinstance(tissue_mask_keys, (str, int)):
                tissue_mask_keys = [tissue_mask_keys]

        for i, key in enumerate(self.keys):
            tissue_mask = None
            if tissue_mask_keys is not None:
                if len(tissue_mask_keys) == 1:
                    mask_key = tissue_mask_keys[0]
                elif i < len(tissue_mask_keys):
                    mask_key = tissue_mask_keys[i]
                else:
                    mask_key = None

                if mask_key is not None and mask_key in data:
                    tissue_mask = data[mask_key].squeeze().numpy() > 0

            image = data[key].numpy()
            if image.ndim == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            gray_image = rgb2gray(image / 255) * 255
            image_her = self.deconv(image, tissue_mask=tissue_mask).numpy()
            image_her[2] = gray_image
            data[key] = MetaTensor(image_her)
        return data


# ---------------------------------------------------------------------------
# EHO (Eosin, Hematoxylin, Optical-Density) helpers
# ---------------------------------------------------------------------------
WHITE_BRIGHTNESS_THRESHOLD: float = 0.82
WHITE_SATURATION_MAX: float = 0.25
ENTROPY_WINDOW: int = 9
ENTROPY_THRESHOLD: float = 2.4
MIN_LUMEN_AREA: int = 128


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

    Uses a hybrid rule: bright + low saturation + low local Shannon
    entropy, with a conservative fallback to avoid over-masking tissue.
    """
    cmax = np.max(image_np[..., :3], axis=2)
    cmin = np.min(image_np[..., :3], axis=2)
    val = cmax.astype(np.float32) / 255.0
    diff = cmax.astype(np.float32) - cmin.astype(np.float32)
    del cmin
    sat = np.zeros_like(val)
    nz = cmax > 0
    sat[nz] = diff[nz] / cmax[nz].astype(np.float32)
    del diff, nz, cmax

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

    white_mask_entropy = (
        (val > brightness_threshold)
        & (sat < saturation_max)
        & (entropy_map < entropy_threshold)
    )
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


def compute_background_intensity(
    image_np: np.ndarray,
    bg_mask: np.ndarray,
) -> float:
    """Estimate the illumination intensity *Io* from background pixels."""
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


def estimate_stain_params(rgb: np.ndarray, bg_mask: np.ndarray | None = None) -> dict:
    """Estimate Macenko colour-deconvolution parameters from an RGB image."""
    if bg_mask is None:
        bg_mask = detect_background(rgb)

    Io = compute_background_intensity(rgb, bg_mask)

    w_est = macenko_pca.rgb_separate_stains_macenko_pca(
        rgb,
        bg_int=[Io],
        mask_out=bg_mask,
    )

    h_idx = macenko_pca.find_stain_index(
        macenko_pca.stain_color_map["hematoxylin"],
        w_est,
    )
    e_idx = macenko_pca.find_stain_index(
        macenko_pca.stain_color_map["eosin"],
        w_est,
    )
    if h_idx == e_idx or h_idx == 2 or e_idx == 2:
        h_idx, e_idx = 0, 1

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

    Returns (H, W, 3) uint8 with channels [Eosin, Hematoxylin, OD].

    Channel convention (matches the training-time notebook EHOd):
        high value = more stain / more optical density
    Concentrations are percentile-normalised (1st–99th) and clipped to [0, 1].
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
        chunk *= -1.0

        e_conc = np.einsum("ijk,k->ij", chunk, e_vec)
        h_conc = np.einsum("ijk,k->ij", chunk, h_vec)

        e_conc -= e_lo_f
        e_conc /= e_range
        np.clip(e_conc, 0, 1, out=e_conc)
        eho[r0:r1, :, 0] = (e_conc * 255).astype(np.uint8)

        h_conc -= h_lo_f
        h_conc /= h_range
        np.clip(h_conc, 0, 1, out=h_conc)
        eho[r0:r1, :, 1] = (h_conc * 255).astype(np.uint8)
        del e_conc, h_conc

        od = chunk.mean(axis=2)
        del chunk
        od -= od_lo_f
        od /= od_range
        np.clip(od, 0, 1, out=od)
        eho[r0:r1, :, 2] = (od * 255).astype(np.uint8)
        del od

    return eho


# ---------------------------------------------------------------------------
# EHO MONAI dictionary transform
# ---------------------------------------------------------------------------


class EHOd(MapTransform):
    """Replace an RGB image with its EHO (Eosin, Hematoxylin, OD) representation.

    Output is a (3, H, W) uint8 MetaTensor with channels [E, H, OD].
    Uses chunked processing to keep memory low on large WSIs.
    """

    stain_color_map = macenko_pca.stain_color_map

    def __init__(self, keys, tissue_mask_keys=None, chunk_rows: int = 512):
        super().__init__(keys)
        self.tissue_mask_keys = tissue_mask_keys
        self.chunk_rows = chunk_rows

    def __call__(self, data):
        tissue_mask_keys = self.tissue_mask_keys
        if tissue_mask_keys is not None:
            if isinstance(tissue_mask_keys, (str, int)):
                tissue_mask_keys = [tissue_mask_keys]

        for i, key in enumerate(self.keys):
            image = data[key].numpy()
            # Ensure HWC
            if image.ndim == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))

            # Resolve tissue mask → invert for bg_mask
            bg_mask = None
            if tissue_mask_keys is not None:
                if len(tissue_mask_keys) == 1:
                    mask_key = tissue_mask_keys[0]
                elif i < len(tissue_mask_keys):
                    mask_key = tissue_mask_keys[i]
                else:
                    mask_key = None

                if mask_key is not None and mask_key in data:
                    tissue = data[mask_key].squeeze().numpy() > 0
                    bg_mask = ~tissue

            # Down-sample for stain-parameter estimation on large images
            est_image = image
            est_bg = bg_mask
            if image.size > 16777216:  # > 4096^2 pixels
                logging.info("Down-sampling image for stain-parameter estimation")
                scale = 4
                h, w = image.shape[:2]
                small_h, small_w = h // scale, w // scale
                resizer = Resize(spatial_size=(small_h, small_w), mode="linear")
                est_image = (
                    resizer(image.transpose((2, 0, 1))).numpy().transpose((1, 2, 0))
                )
                if est_bg is not None:
                    est_bg = resizer(est_bg[None].astype(np.float32)).numpy()[0] > 0.5

            params = estimate_stain_params(
                np.ascontiguousarray(est_image).astype(np.uint8),
                bg_mask=est_bg,
            )

            eho = apply_eho_chunked(
                np.ascontiguousarray(image).astype(np.uint8),
                chunk_rows=self.chunk_rows,
                **params,
            )

            # CHW MetaTensor
            data[key] = MetaTensor(eho.transpose((2, 0, 1)))
        return data
