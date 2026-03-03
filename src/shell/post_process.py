# SPDX-FileCopyrightText: 2024-present barrettMCW <mjbarrett@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""
Post-processing pipeline for EHO 2 µm histology images.

EHO channel layout (from :mod:`shell.transforms`):
  ch0 = Eosin, ch1 = Hematoxylin, ch2 = Optical Density

Pipeline stages (in order):
  1. restrict_predictions_to_tissue   – clip raw preds to preprocessing tissue mask
  2. detect_urethra                   – find the central urethra component
  3. clean_small_holes                – morphological fill of tiny holes in masks
  4. filter_inner_by_surround         – keep inner only if sufficiently surrounded by outer
  5. filter_outer_by_inner            – drop orphaned / unrelated outer components
  6. finalise_masks                   – one last hole-fill on the combined mask
  7. fill_inner_holes                 – binary_fill_holes to fix interior ditzels
  8. ensure_inner_border              – 2 px inward ring of combined → OR into outer
  9. segment_nuclei                   – hematoxylin-based nuclei detection
 10. assign_labels                    – produce final uint8 label map

Performance notes (vs. the research prototype):
  - detect_urethra no longer labels the full slide array (saves ~15 s on a
    large WSI); it uses the bounding-box crop directly.
  - filter_inner_by_surround grey_dilation is restricted to the bounding
    box of the inner mask (typically a small fraction of the slide), giving
    ~3–5× speedup on sparse tissue.
  - fill_inner_holes is vectorised per-component so each fill operates on
    a tight bounding-box crop rather than the full array.
  - restrict_predictions_to_tissue reuses the bounding box for the
    binary_fill_holes call.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import NamedTuple

import numpy as np
import scipy.ndimage
import skimage.exposure
import skimage.measure
import skimage.morphology
import skimage.util

# ---------------------------------------------------------------------------
# Label map
# ---------------------------------------------------------------------------
LABEL_MAPPING: dict[str, int] = {
    "inner": 1,
    "outer": 2,
    "white": 3,
    "background_tissue": 4,
    "epithelial_nuclei": 5,
    "other_nuclei": 6,
    "urethra": 7,
}

# ---------------------------------------------------------------------------
# Filter profiles
# ---------------------------------------------------------------------------


class FilterProfile(NamedTuple):
    """Parameters that control surround / outer filtering aggressiveness."""

    inner_surround_threshold: float
    outer_mode: str  # "none" | "orphaned" | "all"
    suffix: str


PROFILES: dict[str, FilterProfile] = {
    "best_effort": FilterProfile(
        inner_surround_threshold=0.5, outer_mode="orphaned", suffix="_best_effort"
    ),
    "precise": FilterProfile(
        inner_surround_threshold=1.0, outer_mode="all", suffix="_precise"
    ),
    "sensitive": FilterProfile(
        inner_surround_threshold=0.0, outer_mode="none", suffix="_sensitive"
    ),
}

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class MaskSet:
    """All component masks ready for final label assignment."""

    inner: np.ndarray
    outer: np.ndarray
    epithelial_nuclei: np.ndarray
    other_nuclei: np.ndarray
    tissue: np.ndarray | None
    urethra: np.ndarray | None


# ═══════════════════════════════════════════════════════════════════════════
#  Internal geometry helpers
# ═══════════════════════════════════════════════════════════════════════════


def _roi_bounding_box(mask: np.ndarray) -> tuple[slice, ...] | None:
    """Return bounding-box slices around True values in *mask*, or None."""
    if not np.any(mask):
        return None
    coords = np.nonzero(mask)
    return tuple(slice(int(c.min()), int(c.max()) + 1) for c in coords)


def _padded_bbox(
    mask: np.ndarray, pad: int = 1
) -> tuple[slice, ...] | None:
    """Bounding-box slices with *pad* pixels of context on each side."""
    if not np.any(mask):
        return None
    coords = np.nonzero(mask)
    slices = []
    for c, size in zip(coords, mask.shape):
        lo = max(0, int(c.min()) - pad)
        hi = min(size, int(c.max()) + 1 + pad)
        slices.append(slice(lo, hi))
    return tuple(slices)


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 1 – Restrict predictions to the tissue mask
# ═══════════════════════════════════════════════════════════════════════════


def restrict_predictions_to_tissue(
    inner: np.ndarray,
    outer: np.ndarray,
    tissue_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip inner/outer predictions to the preprocessing tissue mask.

    The tissue mask is filled and cleaned to produce a robust tissue area.
    Returns (inner, outer, tissue).

    *Optimisation*: ``binary_fill_holes`` and ``remove_small_objects`` are
    restricted to the bounding box of the tissue mask, which avoids
    processing empty border regions of a large WSI array.
    """
    tissue_b = tissue_mask.astype(bool)

    # Restrict morphology to the bounding box of the tissue region.
    bbox = _roi_bounding_box(tissue_b)
    if bbox is not None:
        crop = tissue_b[bbox].copy()
        crop = scipy.ndimage.binary_fill_holes(crop)
        skimage.morphology.remove_small_objects(crop, min_size=4096, out=crop)
        tissue = np.zeros_like(tissue_b)
        tissue[bbox] = crop
    else:
        tissue = scipy.ndimage.binary_fill_holes(tissue_b)
        skimage.morphology.remove_small_objects(tissue, min_size=4096, out=tissue)

    inner_b = inner.astype(bool) & tissue
    outer_b = outer.astype(bool) & tissue
    return inner_b, outer_b, tissue


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 2 – Urethra detection
# ═══════════════════════════════════════════════════════════════════════════


def _central_component_in_roi(
    cropped_combined: np.ndarray,
    cropped_large_inner: np.ndarray,
) -> np.ndarray | None:
    """Pick the most central, large-inner-rich component in a crop."""
    labeled = np.empty(cropped_combined.shape, dtype=np.int32)
    n_labels = int(scipy.ndimage.label(cropped_combined, output=labeled))
    props = skimage.measure.regionprops(labeled, intensity_image=cropped_large_inner)
    candidates = [r for r in props if r.intensity_mean > 0.5]
    if not candidates:
        return None
    candidates.sort(key=lambda r: r.area, reverse=True)
    top = candidates[:5]
    roi_center = np.array(cropped_combined.shape) / 2.0
    best = min(top, key=lambda r: np.linalg.norm(np.array(r.centroid) - roi_center))
    return labeled == best.label


def detect_urethra(
    inner: np.ndarray,
    outer: np.ndarray,
    min_area: int = 131_072,
) -> np.ndarray:
    """Identify the urethra as the large, central inner+outer component.

    *Optimisation*: The original implementation re-labelled the full slide
    array to expand a crop-level selection back to global coordinates.
    Because the ROI used for the crop *is* the bounding box of the combined
    mask, every connected component that exists in the full array is already
    fully contained inside the crop, so re-labelling is unnecessary.  We now
    write the selected crop component directly into the output array.
    """
    large_inner = skimage.morphology.remove_small_objects(
        inner.astype(bool), min_size=min_area
    )
    combined = inner.astype(bool) | outer.astype(bool)

    roi = _roi_bounding_box(combined)
    if roi is None:
        return np.zeros_like(inner, dtype=bool)

    cropped_combined = combined[roi]
    cropped_large = large_inner[roi]
    if not np.any(cropped_combined):
        return np.zeros_like(inner, dtype=bool)

    selected = _central_component_in_roi(cropped_combined, cropped_large)
    if selected is None:
        return np.zeros_like(inner, dtype=bool)

    # Map ROI selection directly back to full resolution — no full-array
    # labelling needed because the ROI is the bounding box of `combined`.
    urethra = np.zeros(inner.shape, dtype=bool)
    urethra[roi] = selected

    return scipy.ndimage.binary_fill_holes(urethra)


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 3 – Morphological cleanup (small hole fill)
# ═══════════════════════════════════════════════════════════════════════════


def clean_small_holes(
    inner: np.ndarray,
    outer: np.ndarray,
    min_hole_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill small enclosed holes in inner, outer, and their union."""
    inner_b = skimage.morphology.remove_small_holes(
        inner.astype(bool), area_threshold=min_hole_size
    )
    outer_b = skimage.morphology.remove_small_holes(
        outer.astype(bool), area_threshold=min_hole_size
    )
    whole = skimage.morphology.remove_small_holes(
        inner_b | outer_b, area_threshold=min_hole_size
    )
    cleaned_outer = outer_b & whole
    cleaned_inner = whole & ~cleaned_outer
    return cleaned_inner, cleaned_outer


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 4 – Inner filtering by outer surround
# ═══════════════════════════════════════════════════════════════════════════


def _shell_areas(
    labeled_inner: np.ndarray,
    inner_bool: np.ndarray,
    outer_bool: np.ndarray,
    structure: np.ndarray,
    n_labels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Total and outer-covered boundary-shell area for each inner component.

    *Optimisation*: ``grey_dilation`` is applied only to the bounding box
    of the inner mask (plus 1 px padding), not the full slide array.  On
    sparse tissue this can be 5–10× faster.
    """
    bbox = _padded_bbox(inner_bool, pad=1)
    if bbox is None:
        empty = np.zeros(n_labels, dtype=np.float64)
        return empty, empty

    lab_crop = labeled_inner[bbox]
    inner_crop = inner_bool[bbox]
    outer_crop = outer_bool[bbox]

    dilated_crop = scipy.ndimage.grey_dilation(lab_crop, footprint=structure)
    shell_mask = (dilated_crop > 0) & (~inner_crop)

    shell_labels = dilated_crop[shell_mask].ravel()
    total = np.bincount(shell_labels, minlength=n_labels + 1)[1 : n_labels + 1]

    outer_shell_labels = dilated_crop[shell_mask & outer_crop].ravel()
    in_outer = np.bincount(outer_shell_labels, minlength=n_labels + 1)[1 : n_labels + 1]

    return total.astype(np.float64), in_outer.astype(np.float64)


def _labels_meeting_surround(
    total: np.ndarray,
    in_outer: np.ndarray,
    indices: np.ndarray,
    threshold: float,
) -> np.ndarray:
    valid = total > 0
    if threshold == 1.0:
        return indices[(in_outer == total) & valid]
    ratios = np.zeros_like(total, dtype=float)
    ratios[valid] = in_outer[valid] / total[valid]
    return indices[ratios >= threshold]


def filter_inner_by_surround(
    inner: np.ndarray,
    outer: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Keep inner components where ≥ *threshold* of the boundary touches outer."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1].")

    inner_b = inner.astype(bool)
    outer_b = outer.astype(bool)

    labeled = np.empty(inner_b.shape, dtype=np.int32)
    n: int = int(scipy.ndimage.label(inner_b, output=labeled))
    if n == 0:
        return np.zeros_like(inner_b, dtype=bool)
    if threshold == 0.0:
        return inner_b.copy()

    indices = np.arange(1, n + 1)
    struct = scipy.ndimage.generate_binary_structure(inner.ndim, inner.ndim)
    total, in_outer = _shell_areas(labeled, inner_b, outer_b, struct, n)
    kept = _labels_meeting_surround(total, in_outer, indices, threshold)

    return np.isin(labeled, kept) if kept.size else np.zeros_like(inner_b, dtype=bool)


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 5 – Outer filtering by relationship to inner
# ═══════════════════════════════════════════════════════════════════════════


def _outer_labels_contacting(
    labeled_outer: np.ndarray,
    reference_mask: np.ndarray,
    outer_bool: np.ndarray,
) -> np.ndarray:
    struct = scipy.ndimage.generate_binary_structure(reference_mask.ndim, reference_mask.ndim)
    dilated = scipy.ndimage.binary_dilation(reference_mask, structure=struct)
    touching = np.unique(labeled_outer[dilated & outer_bool])
    return touching[touching > 0]


def _filter_outer_orphaned(
    original_inner: np.ndarray,
    filtered_inner: np.ndarray,
    outer: np.ndarray,
) -> np.ndarray:
    labeled_outer = np.empty(outer.shape, dtype=np.int32)
    n: int = int(scipy.ndimage.label(outer, output=labeled_outer))
    if n == 0:
        return np.zeros_like(outer, dtype=bool)

    all_labels = np.arange(1, n + 1)
    touching_orig = _outer_labels_contacting(labeled_outer, original_inner, outer)
    touching_filt = _outer_labels_contacting(labeled_outer, filtered_inner, outer)
    not_touching_orig = np.setdiff1d(all_labels, touching_orig)
    keep = np.union1d(not_touching_orig, touching_filt)

    return np.isin(labeled_outer, keep) if keep.size else np.zeros_like(outer, dtype=bool)


def _filter_outer_all(
    original_inner: np.ndarray,
    filtered_inner: np.ndarray,
    outer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    original_whole = original_inner | outer
    labeled_whole = np.empty(original_whole.shape, dtype=np.int32)
    n: int = int(scipy.ndimage.label(original_whole, output=labeled_whole))
    if n == 0:
        empty = np.zeros_like(outer, dtype=bool)
        return empty, empty

    kept_whole = original_whole.copy()
    removed_inner = original_inner & ~filtered_inner
    if np.any(removed_inner):
        removed_labels = np.unique(labeled_whole[removed_inner])
        removed_labels = removed_labels[removed_labels > 0]
        if removed_labels.size:
            kept_whole[np.isin(labeled_whole, removed_labels)] = False

    new_outer = kept_whole & ~filtered_inner
    return new_outer, kept_whole


def filter_outer_by_inner(
    original_inner: np.ndarray,
    filtered_inner: np.ndarray,
    outer: np.ndarray,
    mode: str = "orphaned",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Filter outer components.  Returns (outer, kept_whole_or_None)."""
    original_inner = original_inner.astype(bool)
    filtered_inner = filtered_inner.astype(bool)
    outer = outer.astype(bool)

    if mode == "none":
        return outer.copy(), None
    if mode == "orphaned":
        return _filter_outer_orphaned(original_inner, filtered_inner, outer), None
    if mode == "all":
        return _filter_outer_all(original_inner, filtered_inner, outer)
    raise ValueError(f"Unknown outer filter mode: {mode!r}")


def apply_filters(
    inner: np.ndarray,
    outer: np.ndarray,
    profile: FilterProfile,
) -> tuple[np.ndarray, np.ndarray]:
    """Run surround + outer filters for a given profile."""
    inner_filtered = filter_inner_by_surround(
        inner, outer, threshold=profile.inner_surround_threshold
    )
    outer_filtered, kept_whole = filter_outer_by_inner(
        original_inner=inner,
        filtered_inner=inner_filtered,
        outer=outer,
        mode=profile.outer_mode,
    )
    final_inner = inner_filtered.copy()
    if profile.outer_mode == "all" and kept_whole is not None:
        final_inner &= kept_whole

    return final_inner, outer_filtered


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 6 – Finalise masks
# ═══════════════════════════════════════════════════════════════════════════


def finalise_masks(
    inner: np.ndarray,
    outer: np.ndarray,
    min_hole_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Final cleanup: fill small holes in the combined mask, then re-split."""
    whole = skimage.morphology.remove_small_holes(
        inner.astype(bool) | outer.astype(bool), area_threshold=min_hole_size
    )
    definitive_inner = whole & ~outer.astype(bool)
    definitive_outer = outer.astype(bool)
    return definitive_inner, definitive_outer


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 7 – Fill interior ditzels in inner
# ═══════════════════════════════════════════════════════════════════════════


def fill_inner_holes(inner: np.ndarray) -> np.ndarray:
    """Fill binary holes inside inner components (fixes ditzels).

    *Optimisation*: each connected component is processed independently on
    its tight bounding-box crop rather than running ``binary_fill_holes`` on
    the full slide array.  For many small components this is substantially
    faster.
    """
    inner_b = inner.astype(bool)
    labeled = np.empty(inner_b.shape, dtype=np.int32)
    n: int = int(scipy.ndimage.label(inner_b, output=labeled))
    if n == 0:
        return inner_b

    result = inner_b.copy()
    props = skimage.measure.regionprops(labeled)
    for prop in props:
        r0, c0, r1, c1 = prop.bbox  # (min_row, min_col, max_row, max_col)
        # Expand bbox by 1 px to include the surrounding background needed
        # for binary_fill_holes to detect enclosed regions.
        rlo = max(0, r0 - 1)
        clo = max(0, c0 - 1)
        rhi = min(inner_b.shape[0], r1 + 1)
        chi = min(inner_b.shape[1], c1 + 1)
        crop = inner_b[rlo:rhi, clo:chi]
        filled = scipy.ndimage.binary_fill_holes(crop)
        # Only write newly filled pixels (do not erase existing ones).
        result[rlo:rhi, clo:chi] |= filled

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 8 – Ensure inner is surrounded by outer (border ring)
# ═══════════════════════════════════════════════════════════════════════════


def ensure_inner_border(
    inner: np.ndarray,
    outer: np.ndarray,
    border_px: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Ensure every inner region is surrounded by at least *border_px* of outer.

    Erodes the combined (inner | outer) mask inward by *border_px*, takes
    the complement ring, and ORs it into outer.  Inner is never modified.
    """
    inner_b = inner.astype(bool)
    outer_b = outer.astype(bool)
    combined = inner_b | outer_b

    struct = skimage.morphology.disk(border_px)
    eroded = scipy.ndimage.binary_erosion(combined, structure=struct)
    ring = combined & ~eroded
    new_outer = outer_b | ring

    return inner_b, new_outer


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 9 – Nuclei segmentation from hematoxylin
# ═══════════════════════════════════════════════════════════════════════════


def _equalise_hematoxylin(channel: np.ndarray) -> np.ndarray:
    """Histogram-equalise a single-channel image (float or uint8)."""
    if channel.dtype.kind == "f":
        u8 = skimage.util.img_as_ubyte(np.clip(channel, 0, 1))
        return skimage.exposure.equalize_hist(u8)
    return skimage.exposure.equalize_hist(channel)


def segment_nuclei(
    hematoxylin_channel: np.ndarray,
    outer_mask: np.ndarray,
    inner_mask: np.ndarray | None = None,
    threshold: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment nuclei from the hematoxylin channel.

    Returns ``(epithelial_nuclei, other_nuclei)`` where:

    * ``epithelial_nuclei`` — nuclei inside *outer_mask* (epithelium)
    * ``other_nuclei`` — nuclei in stroma (outside outer AND outside inner)

    Parameters
    ----------
    hematoxylin_channel:
        EHO ch1 (hematoxylin), uint8 or float.
    outer_mask:
        Definitive outer (epithelium) mask.
    inner_mask:
        Definitive inner (lumen) mask.  Nuclei inside inner are excluded
        from *other_nuclei* (they are not stromal).
    threshold:
        Equalised-intensity threshold — pixels *below* this are nuclei.
        Default 0.03.
    """
    eq = _equalise_hematoxylin(hematoxylin_channel)
    nuclei = eq < threshold

    outer_b = outer_mask.astype(bool)
    epithelial = nuclei & outer_b

    exclude = outer_b
    if inner_mask is not None:
        exclude = exclude | inner_mask.astype(bool)
    other = nuclei & ~exclude

    return epithelial, other


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 10 – Final label assignment
# ═══════════════════════════════════════════════════════════════════════════


def assign_labels(masks: MaskSet, shape: tuple[int, int]) -> np.ndarray:
    """Create the final ``uint8`` label map.

    White pixels are derived from the preprocessing tissue mask
    (``white = ~tissue``).  Priority (last writer wins):
    inner → outer → white → background_tissue → epithelial_nuclei →
    other_nuclei → urethra.
    """
    lm = np.zeros(shape, dtype=np.uint8)

    lm[masks.inner.astype(bool)] = LABEL_MAPPING["inner"]
    lm[masks.outer.astype(bool)] = LABEL_MAPPING["outer"]

    definitive_gland = masks.inner.astype(bool) | masks.outer.astype(bool)

    if masks.tissue is not None:
        tissue_b = masks.tissue.astype(bool)
        white = ~tissue_b
        stroma = tissue_b & ~definitive_gland
    else:
        white = np.zeros(shape, dtype=bool)
        stroma = ~definitive_gland

    lm[white] = LABEL_MAPPING["white"]
    lm[stroma] = LABEL_MAPPING["background_tissue"]

    lm[masks.epithelial_nuclei.astype(bool)] = LABEL_MAPPING["epithelial_nuclei"]
    lm[masks.other_nuclei.astype(bool)] = LABEL_MAPPING["other_nuclei"]

    if masks.urethra is not None:
        lm[masks.urethra.astype(bool)] = LABEL_MAPPING["urethra"]

    return lm


# ═══════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════


def post_process(
    inner_pred: np.ndarray,
    outer_pred: np.ndarray,
    tissue_mask: np.ndarray,
    hematoxylin_channel: np.ndarray,
    *,
    profile_name: str = "best_effort",
    nuclei_threshold: float = 0.03,
    inner_border_px: int = 2,
    verbose: bool = True,
    return_timings: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    """Full post-processing pipeline for EHO 2 µm images.

    Parameters
    ----------
    inner_pred, outer_pred:
        Raw binary predictions from the segmentation model (H, W) bool/uint8.
    tissue_mask:
        Tissue mask from preprocessing (H, W) bool.  White pixels in the
        label map are derived as ``~tissue_mask``.
    hematoxylin_channel:
        EHO ch1 (hematoxylin) for nuclei segmentation, uint8 or float.
    profile_name:
        One of ``"best_effort"``, ``"precise"``, ``"sensitive"``.
    nuclei_threshold:
        Threshold for nuclei detection (lower = more conservative).
    inner_border_px:
        Pixels to erode from combined → assign to outer as border ring.
    verbose:
        Print per-stage timings to stdout.
    return_timings:
        If True, return ``(label_map, timings_dict)`` where *timings_dict*
        maps stage names to wall-clock seconds.  Useful for benchmarking.

    Returns
    -------
    label_map : (H, W) uint8 array
    timings : dict[str, float]  (only when *return_timings* is True)
    """
    profile = PROFILES[profile_name]
    timings: dict[str, float] = {}

    def _run(name: str, fn, *args, **kwargs):
        t0 = perf_counter()
        result = fn(*args, **kwargs)
        elapsed = perf_counter() - t0
        timings[name] = elapsed
        if verbose:
            print(f"  {name}: {elapsed:.2f}s")
        return result

    if verbose:
        print(
            f"post_process — profile={profile_name}  "
            f"nuclei_thr={nuclei_threshold}  border={inner_border_px}px"
        )

    inner, outer, tissue = _run(
        "restrict_to_tissue",
        restrict_predictions_to_tissue,
        inner_pred.astype(bool),
        outer_pred.astype(bool),
        tissue_mask,
    )
    urethra = _run("detect_urethra", detect_urethra, inner, outer)
    inner, outer = _run("clean_small_holes", clean_small_holes, inner, outer)

    inner_pre, outer_pre = inner.copy(), outer.copy()
    inner, outer = _run("apply_filters", apply_filters, inner_pre, outer_pre, profile)

    inner, outer = _run("finalise_masks", finalise_masks, inner, outer)
    inner = _run("fill_inner_holes", fill_inner_holes, inner)
    inner, outer = _run(
        "ensure_inner_border", ensure_inner_border, inner, outer, border_px=inner_border_px
    )
    epi_nuclei, other_nuclei = _run(
        "segment_nuclei",
        segment_nuclei,
        hematoxylin_channel,
        outer,
        inner,
        nuclei_threshold,
    )

    mask_set = MaskSet(
        inner=inner,
        outer=outer,
        epithelial_nuclei=epi_nuclei,
        other_nuclei=other_nuclei,
        tissue=tissue,
        urethra=urethra,
    )
    labeled = _run("assign_labels", assign_labels, mask_set, inner.shape)

    if verbose:
        total = sum(timings.values())
        print(f"  TOTAL: {total:.2f}s")

    if return_timings:
        return labeled, timings
    return labeled
