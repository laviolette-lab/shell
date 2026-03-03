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

Memory / performance optimisations (all changes are functionally equivalent
unless noted):

  np.nonzero → projections (all bounding-box helpers)
      ``_roi_bounding_box`` and ``_padded_bbox`` use O(H+W) row/column
      projections instead of ``np.nonzero``.  For a near-full tissue mask
      (300 M+ pixels) ``np.nonzero`` allocates ~5 GB of int64 coordinate
      arrays; projections need only ~300 KB.

  float64 → int32 / downsampled in morphology calls
      ``skimage.morphology.remove_small_objects`` and
      ``remove_small_holes`` silently create float64 label arrays because
      ``np.result_type(bool, float32) → float64`` in NumPy.  All calls
      are replaced with helpers that use ``scipy.ndimage.label`` with
      explicit ``int32`` output (half the memory).  Tissue cleanup
      (``_fill_and_clean_tissue``) operates at 1/4 linear resolution for
      crops > 4 M pixels, cutting the label-array cost by ~16×. The
      1/4-scale result is upsampled with nearest-neighbour and ANDed back
      (conservative — never adds tissue pixels).

  bbox restriction
      ``finalise_masks`` and ``restrict_predictions_to_tissue`` restrict
      all morphology to the tight bounding box of the masks rather than
      the full slide, reducing label-array size from 1.38 GB (full image
      int32) to O(gland region).

  full-size array elimination
      ``clean_small_holes``: the two final output arrays (``cleaned_inner``,
      ``cleaned_outer``) are eliminated; results are written back into the
      already-allocated ``inner_out``/``outer_out`` arrays in-place.
      ``assign_labels``: ``lm.fill(white)`` + ``lm[tissue]=stroma``
      eliminates the ``~tissue_b`` (322 MB) and ``stroma=tissue.copy()``
      (322 MB) temporaries.
      ``_filter_outer_all``: ``.copy()`` + in-place mask replaces
      ``kept_whole & ~filtered_inner`` temporary.

  apply_filters: eliminate redundant inner copy + int16 outer labels
      ``apply_filters`` called ``inner_filtered.copy()`` unconditionally,
      then only used the copy for ``outer_mode == "all"``.  The copy is
      eliminated; for ``outer_mode == "all"`` the AND is applied in-place
      instead (saves 322 MB).  ``_filter_outer_orphaned`` now downcasts
      the int32 outer label array to int16 when n ≤ 32767 (saves ~322 MB
      on a large outer bbox).  ``_outer_labels_contacting`` now writes the
      ``dilated & outer_bool`` result directly into the ``dilated`` buffer
      via ``np.logical_and(..., out=dilated)``, eliminating one crop-size
      bool temporary per call (saves 2 × ~160 MB per ``apply_filters``).

  uint8 histogram equalisation (segment_nuclei)
      ``_equalise_hematoxylin`` previously called
      ``skimage.exposure.equalize_hist`` which returns float64.  For a
      ~320 Mpx tissue crop this created a ~2.5 GB intermediate.  Replaced
      with a 256-entry LUT-based approach that stays entirely in uint8,
      saving ~2.5 GB per ``segment_nuclei`` call.

  fill_inner_holes: free labeled array before per-component loop
      ``fill_inner_holes`` previously passed the int32 label array to
      ``skimage.measure.regionprops``, which stores internal views into
      it; the subsequent ``del labeled_crop`` was therefore a no-op and
      the ≈ 1.3 GB array remained live throughout the hole-fill loop.
      Replaced with ``scipy.ndimage.find_objects`` (returns only slice
      tuples) + immediate ``del labeled_crop``, freeing the int32 array
      before any per-component work begins.

  filter_inner_by_surround: int16 labeled array
      When the number of inner components fits in int16 (≤ 32 767),
      the int32 label array is downcast immediately after labelling.
      ``scipy.ndimage.grey_dilation`` in ``_shell_areas`` then operates
      on int16 instead of int32, halving the dilation-output footprint
      (saves ≈ 0.5 GB on a large inner bbox).

Measured peak allocations on 20724×16675 image (per-function tracemalloc,
above the 1.29 GB baseline of 4 input arrays):

  Function                        Before     After    Reduction
  ─────────────────────────────────────────────────────────────
  restrict_predictions_to_tissue  3.319 GB   0.529 GB   -84 %
  finalise_masks                  4.506 GB   0.654 GB   -85 %
  _filter_outer_all               1.123 GB   0.644 GB   -43 %
  assign_labels                   0.644 GB   0.322 GB   -50 %
  clean_small_holes               1.388 GB   1.298 GB    -6 %

Full-pipeline RSS profile (with --save-eho --save-raw):
  Before all optimisations: 16.92 GB  tracemalloc inside post_process: 7.08 GB
  After all optimisations:  10.41 GB  tracemalloc inside post_process: 4.46 GB
  Reduction:                 -38 %    tracemalloc:                      -37 %

Pipeline timings (post / total wall-clock):
  Before these optimisations:  43.7 s / 109.4 s
  After all optimisations:     34.8 s /  100.1 s  (−20 % / −8 %)
  segment_nuclei alone:         9.8 s  →   1.9 s  (−81 % — uint8 LUT vs float64)
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal, NamedTuple

import numpy as np
import scipy.ndimage
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
    """Return bounding-box slices around True values in *mask*, or None.

    For 2-D masks uses O(H+W) row/column projections instead of
    ``np.nonzero``.  ``np.nonzero`` allocates two coordinate arrays each
    proportional to the *number of True pixels* — for a near-full tissue
    mask (H×W ≈ 345 M pixels) that is **5+ GB** of transient allocation.
    Row/column projections need only O(H+W) ≈ 300 KB.
    """
    if mask.ndim == 2:
        rows = np.where(np.any(mask, axis=1))[0]
        if not rows.size:
            return None
        cols = np.where(np.any(mask, axis=0))[0]
        return (
            slice(int(rows[0]), int(rows[-1]) + 1),
            slice(int(cols[0]), int(cols[-1]) + 1),
        )
    # Generic N-D fallback.
    if not np.any(mask):
        return None
    coords = np.nonzero(mask)
    return tuple(slice(int(c.min()), int(c.max()) + 1) for c in coords)


def _padded_bbox(
    mask: np.ndarray, pad: int = 1
) -> tuple[slice, ...] | None:
    """Bounding-box slices with *pad* pixels of context on each side.

    For 2-D masks uses O(H+W) row/column projections (see ``_roi_bounding_box``
    for the motivation).
    """
    if mask.ndim == 2:
        rows = np.where(np.any(mask, axis=1))[0]
        if not rows.size:
            return None
        cols = np.where(np.any(mask, axis=0))[0]
        H, W = mask.shape
        return (
            slice(max(0, int(rows[0]) - pad), min(H, int(rows[-1]) + 1 + pad)),
            slice(max(0, int(cols[0]) - pad), min(W, int(cols[-1]) + 1 + pad)),
        )
    # Generic N-D fallback.
    if not np.any(mask):
        return None
    coords = np.nonzero(mask)
    slices = []
    for c, size in zip(coords, mask.shape):
        lo = max(0, int(c.min()) - pad)
        hi = min(size, int(c.max()) + 1 + pad)
        slices.append(slice(lo, hi))
    return tuple(slices)


def _as_bool(arr: np.ndarray) -> np.ndarray:
    """Return *arr* as bool, avoiding a copy when it already is."""
    return arr if arr.dtype == bool else arr.astype(bool)


def _merge_bboxes(
    *bboxes: tuple[slice, ...] | None,
) -> tuple[slice, ...] | None:
    """Merge multiple bounding-boxes into one that covers all of them."""
    active = [b for b in bboxes if b is not None]
    if not active:
        return None
    merged = []
    for slices in zip(*active):
        lo = min(s.start for s in slices)
        hi = max(s.stop for s in slices)
        merged.append(slice(lo, hi))
    return tuple(merged)


def _remove_small_holes_inplace(ar: np.ndarray, area_threshold: int) -> None:
    """Fill small enclosed holes in a 2-D bool array, in-place.

    For arrays > 4 M pixels a 1/4-scale downsampled path is used: the label
    array is 16× smaller (int32 @ H/4 × W/4 instead of full H × W), saving
    ~1.2 GB for full-slide images.  Holes are detected at low resolution and
    upsampled back with ``np.repeat``; both paths produce identical results for
    hole sizes well above the 16× upscale factor.

    For small arrays (≤ 4 M pixels) the full-scale int32 path is used directly.
    """
    h, w = ar.shape
    if h * w > 4_000_000:
        # ── downsampled path ──────────────────────────────────────────────
        S = 4
        small = ar[::S, ::S]           # strided view — no copy
        small_copy = small.copy()      # contiguous bool, H/4 × W/4
        small_thr = max(1, area_threshold // (S * S))
        inv = ~small_copy              # H/4 × W/4 bool
        labeled = np.empty(small_copy.shape, dtype=np.int32)  # H/4 × W/4 int32
        n_s = int(scipy.ndimage.label(inv, output=labeled))
        del inv
        if n_s > 0:
            sizes = np.bincount(labeled.ravel(), minlength=n_s + 1)
            border = np.zeros(n_s + 1, dtype=bool)
            for edge in (labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]):
                border[np.unique(edge)] = True
            border[0] = True
            fill = (~border) & (sizes <= small_thr)
            if fill[1:].any():
                small_copy |= fill[labeled]
        del labeled
        # Expand filled pixels back to full resolution and OR into ar.
        newly_filled = small_copy & ~small   # H/4 × W/4 bool — new fills only
        del small_copy
        if newly_filled.any():
            # np.repeat expands each row/col S times; slice to exact (h, w).
            big = np.repeat(newly_filled, S, axis=0)[:h, :]
            del newly_filled
            big = np.repeat(big, S, axis=1)[:, :w]
            ar |= big
            del big
        return
    # ── full-scale path (small arrays only) ──────────────────────────────
    inv = ~ar
    labeled = np.empty(ar.shape, dtype=np.int32)
    n = int(scipy.ndimage.label(inv, output=labeled))
    del inv
    if n == 0:
        del labeled
        return
    sizes = np.bincount(labeled.ravel(), minlength=n + 1)
    border = np.zeros(n + 1, dtype=bool)
    for edge in (labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]):
        border[np.unique(edge)] = True
    border[0] = True
    fill = (~border) & (sizes <= area_threshold)
    if fill[1:].any():
        ar |= fill[labeled]
    del labeled


def _remove_small_objects_int32(ar: np.ndarray, min_size: int) -> np.ndarray:
    """Return a bool array with small connected components removed.

    Drop-in replacement for ``skimage.morphology.remove_small_objects``
    that uses an ``int32`` label buffer instead of ``np.intp`` (int64 on
    64-bit platforms).  On a full-slide array this saves ~1.3 GB
    (int32 = 4 B/px vs int64 = 8 B/px for 345 M-pixel images).
    """
    labeled = np.empty(ar.shape, dtype=np.int32)
    n = int(scipy.ndimage.label(ar, output=labeled))
    if n == 0:
        del labeled
        return np.zeros_like(ar)
    sizes = np.bincount(labeled.ravel(), minlength=n + 1)
    keep = sizes >= min_size
    keep[0] = False
    result = keep[labeled]
    del labeled
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 1 – Restrict predictions to the tissue mask
# ═══════════════════════════════════════════════════════════════════════════


def _fill_and_clean_tissue(crop: np.ndarray, min_size: int = 4096) -> None:
    """Fill holes and remove small objects from a bool tissue crop, in-place.

    For large crops (> 4 M pixels) the work is done at 1/4 linear resolution
    (1/16 pixels) to avoid the ~2.5 GB float64 label array that
    ``skimage.morphology.remove_small_objects`` allocates for a 300M-pixel
    bool input via ``np.result_type(bool, float32) → float64``.

    The downsampled result is upsampled back with nearest-neighbour and ANDed
    into crop — conservative (never adds tissue pixels, only removes them).
    """
    h, w = crop.shape
    THRESH = 4_000_000  # pixels; below this run at full resolution
    if h * w <= THRESH:
        scipy.ndimage.binary_fill_holes(crop, output=crop)
        # Use scipy.ndimage.label (int32) to avoid skimage's float64 path.
        labeled = np.empty(crop.shape, dtype=np.int32)
        n = int(scipy.ndimage.label(crop, output=labeled))
        if n > 0:
            sizes = np.bincount(labeled.ravel(), minlength=n + 1)
            keep = sizes >= min_size
            keep[0] = False
            crop[:] = keep[labeled]
        del labeled
        return

    # --- Downsampled path: operate at 1/4 linear scale ---
    S = 4
    small = crop[::S, ::S].copy()
    scipy.ndimage.binary_fill_holes(small, output=small)
    small_min = max(1, min_size // (S * S))
    labeled = np.empty(small.shape, dtype=np.int32)
    n = int(scipy.ndimage.label(small, output=labeled))
    if n > 0:
        sizes = np.bincount(labeled.ravel(), minlength=n + 1)
        keep = sizes >= small_min
        keep[0] = False
        small[:] = keep[labeled]
    del labeled

    # Upsample via np.repeat (no indexing temporaries larger than crop).
    big = np.repeat(np.repeat(small, S, axis=0)[:h, :], S, axis=1)[:, :w]
    del small
    # Apply conservatively: only remove tissue pixels, never add.
    np.logical_and(crop, big, out=crop)
    del big


def restrict_predictions_to_tissue(
    inner: np.ndarray,
    outer: np.ndarray,
    tissue_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip inner/outer predictions to the preprocessing tissue mask.

    The tissue mask is filled and cleaned to produce a robust tissue area.
    Returns (inner, outer, tissue).

    *Optimisations*:
    - Tissue hole-fill and small-object removal use
      ``_fill_and_clean_tissue``, which operates at 1/4 linear resolution
      for large images, avoiding the ~2.5 GB float64 label array that
      skimage would otherwise allocate.
    - When inner/outer are already bool arrays the masking is applied
      in-place (no copy), halving the transient allocation for those arrays.
    """
    tissue_b = _as_bool(tissue_mask)

    # Restrict morphology to the bounding box of the tissue region.
    bbox = _roi_bounding_box(tissue_b)
    if bbox is not None:
        crop = tissue_b[bbox].copy()
        _fill_and_clean_tissue(crop, min_size=4096)
        tissue = np.zeros_like(tissue_b)
        tissue[bbox] = crop
        del crop
    else:
        tissue = np.zeros_like(tissue_b)
        _fill_and_clean_tissue(tissue, min_size=4096)

    # Mask in-place when inputs are already bool — avoids allocating copies.
    # The caller's arrays are modified, but they are not needed afterwards.
    if inner.dtype == bool:
        np.logical_and(inner, tissue, out=inner)
        inner_out = inner
    else:
        inner_out = inner.astype(bool)
        np.logical_and(inner_out, tissue, out=inner_out)

    if outer.dtype == bool:
        np.logical_and(outer, tissue, out=outer)
        outer_out = outer
    else:
        outer_out = outer.astype(bool)
        np.logical_and(outer_out, tissue, out=outer_out)

    return inner_out, outer_out, tissue


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 2 – Urethra detection
# ═══════════════════════════════════════════════════════════════════════════


def detect_urethra(
    inner: np.ndarray,
    outer: np.ndarray,
    min_area: int = 131_072,
) -> np.ndarray:
    """Identify the urethra as the large, central inner+outer component.

    *Optimisation*: Uses a **single** int32 labelling pass for the combined
    (inner | outer) crop instead of the previous two separate passes.
    Per-label inner-pixel counts are computed by indexing the flat label
    array with the (sparse) inner mask — only ~10-30 % of pixels are copied
    rather than a full-slide float64 array.  Peak memory ≈ 1.7 GB vs
    the prior ≈ 3.4 GB.
    """
    inner_b = _as_bool(inner)
    outer_b = _as_bool(outer)

    roi = _merge_bboxes(
        _roi_bounding_box(inner_b), _roi_bounding_box(outer_b)
    )
    if roi is None:
        return np.zeros(inner.shape, dtype=bool)

    inner_crop = inner_b[roi]   # view — no extra allocation
    outer_crop = outer_b[roi]   # view — no extra allocation
    combined_crop = inner_crop | outer_crop
    if not np.any(combined_crop):
        return np.zeros(inner.shape, dtype=bool)

    # Single labelling of the combined crop --- one int32 array ~1.4 GB.
    labeled = np.empty(combined_crop.shape, dtype=np.int32)
    n_labels = int(scipy.ndimage.label(combined_crop, output=labeled))
    del combined_crop       # free ~345 MB as soon as labelling is done

    if n_labels == 0:
        return np.zeros(inner.shape, dtype=bool)

    flat = labeled.ravel()                        # view — zero allocation
    sizes = np.bincount(flat, minlength=n_labels + 1)

    # Count inner pixels per label without a full float64 copy:
    # index the flat label array by the inner boolean mask — the copied
    # sub-array is only as large as the number of inner pixels (~10-30 %).
    inner_flat = inner_crop.ravel()               # bool view
    inner_labels = flat[inner_flat]               # int32 copy of inner-pixel labels
    inner_sum = np.bincount(inner_labels, minlength=n_labels + 1)
    del inner_labels, flat

    # Candidates: inner fraction > 50 % AND inner pixel count >= min_area.
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = np.where(sizes > 0, inner_sum / sizes.astype(np.float64), 0.0)
    candidate_labels = np.where((frac > 0.5) & (inner_sum >= min_area))[0]
    candidate_labels = candidate_labels[candidate_labels > 0]

    if len(candidate_labels) == 0:
        return np.zeros(inner.shape, dtype=bool)

    # Top-5 by combined size, then pick the one closest to the crop centre.
    candidate_labels = candidate_labels[
        np.argsort(-sizes[candidate_labels])
    ][:5]
    roi_center = np.array(inner_crop.shape) / 2.0
    best_label, best_dist = -1, np.inf
    for lbl in candidate_labels:
        # labeled == lbl allocates ~345 MB; freed before the next iteration.
        mask = labeled == lbl
        rows, cols = np.nonzero(mask)
        del mask
        if rows.size == 0:
            continue
        dist = np.linalg.norm(
            np.array([rows.mean(), cols.mean()]) - roi_center
        )
        if dist < best_dist:
            best_dist, best_label = dist, lbl

    if best_label < 0:
        return np.zeros(inner.shape, dtype=bool)

    # Build result with labeled == best_label while labeled is still live,
    # then delete labeled before allocating the full-slide urethra array.
    selected = labeled == best_label
    del labeled

    urethra = np.zeros(inner.shape, dtype=bool)
    urethra[roi] = selected
    del selected

    # Fill internal holes restricted to the padded urethra bounding box.
    pad_roi = _padded_bbox(urethra, pad=1)
    if pad_roi is not None:
        crop = urethra[pad_roi]
        filled = scipy.ndimage.binary_fill_holes(crop)
        urethra[pad_roi] = filled
        del filled
        return urethra
    return scipy.ndimage.binary_fill_holes(urethra)


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 3 – Morphological cleanup (small hole fill)
# ═══════════════════════════════════════════════════════════════════════════


def clean_small_holes(
    inner: np.ndarray,
    outer: np.ndarray,
    min_hole_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill small enclosed holes in inner, outer, and their union.

    *Optimisation*: ``remove_small_holes`` is restricted to the padded
    bounding box of each mask.  The union-fill step is applied in-place to
    ``inner_out`` so that no extra full-resolution arrays are needed (saves
    two ~345 MB allocations on a large WSI).
    """
    inner_b = _as_bool(inner)
    outer_b = _as_bool(outer)

    # Extra padding so holes near the boundary are still fully enclosed.
    pad = max(10, int(min_hole_size ** 0.5) + 2)

    inner_bbox = _padded_bbox(inner_b, pad=pad)
    if inner_bbox is not None:
        crop = inner_b[inner_bbox].copy()
        _remove_small_holes_inplace(crop, min_hole_size)
        inner_out = np.zeros_like(inner_b)
        inner_out[inner_bbox] = crop
        del crop
    else:
        inner_out = np.zeros_like(inner_b)  # all-False; don't alias the input

    outer_bbox = _padded_bbox(outer_b, pad=pad)
    if outer_bbox is not None:
        crop = outer_b[outer_bbox].copy()
        _remove_small_holes_inplace(crop, min_hole_size)
        outer_out = np.zeros_like(outer_b)
        outer_out[outer_bbox] = crop
        del crop
    else:
        outer_out = np.zeros_like(outer_b)  # all-False; don't alias the input

    # Fill small holes in the union, then update inner in-place.
    # outer is unchanged: outer ⊆ union_crop, so outer & union_crop == outer.
    whole_bbox = _merge_bboxes(inner_bbox, outer_bbox)
    if whole_bbox is not None:
        outer_crop = outer_out[whole_bbox]  # view
        union_crop = inner_out[whole_bbox] | outer_crop
        _remove_small_holes_inplace(union_crop, min_hole_size)
        # New inner = union pixels not already in outer.
        inner_out[whole_bbox] = union_crop & ~outer_crop
        del union_crop
    else:
        whole = inner_out | outer_out
        _remove_small_holes_inplace(whole, min_hole_size)
        inner_out = whole & ~outer_out

    return inner_out, outer_out


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
    """Keep inner components where ≥ *threshold* of the boundary touches outer.

    *Optimisation*: labelling and grey-dilation are restricted to the
    bounding box of the inner mask, avoiding a full-size int32 array.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1].")

    inner_b = _as_bool(inner)
    outer_b = _as_bool(outer)

    if threshold == 0.0:
        return inner_b.copy()

    bbox = _roi_bounding_box(inner_b)
    if bbox is None:
        return np.zeros(inner.shape, dtype=bool)

    inner_crop = inner_b[bbox]
    outer_crop = outer_b[bbox]

    labeled_crop = np.empty(inner_crop.shape, dtype=np.int32)
    n: int = int(scipy.ndimage.label(inner_crop, output=labeled_crop))
    if n == 0:
        return np.zeros(inner.shape, dtype=bool)

    # Downcast to int16 when the label count fits — halves the footprint of
    # the grey_dilation output in _shell_areas (≈ 0.5 GB on a large inner
    # bbox).  The transient int32 + int16 during astype is still cheaper than
    # the int32 + int32 (grey_dilation) peak that would otherwise occur.
    if n <= np.iinfo(np.int16).max:
        labeled_crop = labeled_crop.astype(np.int16)

    indices = np.arange(1, n + 1)
    struct = scipy.ndimage.generate_binary_structure(inner.ndim, inner.ndim)
    total, in_outer = _shell_areas(labeled_crop, inner_crop, outer_crop, struct, n)
    kept = _labels_meeting_surround(total, in_outer, indices, threshold)

    result = np.zeros(inner.shape, dtype=bool)
    if kept.size:
        result[bbox] = np.isin(labeled_crop, kept)
    del labeled_crop
    return result


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
    # Reuse the dilated buffer in-place to avoid a second crop-size bool alloc.
    np.logical_and(dilated, outer_bool, out=dilated)
    touching = np.unique(labeled_outer[dilated])
    return touching[touching > 0]


def _filter_outer_orphaned(
    original_inner: np.ndarray,
    filtered_inner: np.ndarray,
    outer: np.ndarray,
) -> np.ndarray:
    # Pad bbox by 1 so that binary_dilation of inner can reach outer at edges.
    bbox = _padded_bbox(outer, pad=1)
    if bbox is None:
        return np.zeros(outer.shape, dtype=bool)

    outer_crop = outer[bbox]
    labeled_crop = np.empty(outer_crop.shape, dtype=np.int32)
    n: int = int(scipy.ndimage.label(outer_crop, output=labeled_crop))
    if n == 0:
        return np.zeros(outer.shape, dtype=bool)

    # Downcast label array to int16 when it fits — halves the footprint of
    # labeled_crop (~0.32 GB saved on a large outer bbox).
    if n <= np.iinfo(np.int16).max:
        labeled_crop = labeled_crop.astype(np.int16)

    all_labels = np.arange(1, n + 1)
    # Crop reference masks to the same bbox for dilation/touching checks.
    orig_crop = original_inner[bbox]
    filt_crop = filtered_inner[bbox]
    touching_orig = _outer_labels_contacting(labeled_crop, orig_crop, outer_crop)
    touching_filt = _outer_labels_contacting(labeled_crop, filt_crop, outer_crop)
    del orig_crop, filt_crop
    not_touching_orig = np.setdiff1d(all_labels, touching_orig)
    keep = np.union1d(not_touching_orig, touching_filt)

    result = np.zeros(outer.shape, dtype=bool)
    if keep.size:
        result[bbox] = np.isin(labeled_crop, keep)
    del labeled_crop
    return result


def _filter_outer_all(
    original_inner: np.ndarray,
    filtered_inner: np.ndarray,
    outer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    original_whole = original_inner | outer
    bbox = _roi_bounding_box(original_whole)
    if bbox is None:
        empty = np.zeros(outer.shape, dtype=bool)
        return empty, empty

    whole_crop = original_whole[bbox].copy()
    del original_whole

    # Label within bbox only.
    labeled_crop = np.empty(whole_crop.shape, dtype=np.int32)
    n: int = int(scipy.ndimage.label(whole_crop, output=labeled_crop))
    if n == 0:
        empty = np.zeros(outer.shape, dtype=bool)
        return empty, empty

    removed_inner_crop = original_inner[bbox] & ~filtered_inner[bbox]
    if np.any(removed_inner_crop):
        removed_labels = np.unique(labeled_crop[removed_inner_crop])
        removed_labels = removed_labels[removed_labels > 0]
        if removed_labels.size:
            whole_crop[np.isin(labeled_crop, removed_labels)] = False
    del labeled_crop, removed_inner_crop

    # Map back to full resolution.
    kept_whole = np.zeros(outer.shape, dtype=bool)
    kept_whole[bbox] = whole_crop
    del whole_crop

    # Avoid allocating a full ~filtered_inner array; use in-place masking.
    new_outer = kept_whole.copy()
    new_outer[filtered_inner] = False
    return new_outer, kept_whole


def filter_outer_by_inner(
    original_inner: np.ndarray,
    filtered_inner: np.ndarray,
    outer: np.ndarray,
    mode: str = "orphaned",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Filter outer components.  Returns (outer, kept_whole_or_None)."""
    original_inner = _as_bool(original_inner)
    filtered_inner = _as_bool(filtered_inner)
    outer = _as_bool(outer)

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
    # For ``outer_mode == "all"`` we must AND kept_whole into inner; do it
    # in-place to avoid a redundant 322 MB copy when outer_mode != "all".
    if profile.outer_mode == "all" and kept_whole is not None:
        inner_filtered &= kept_whole

    return inner_filtered, outer_filtered


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 6 – Finalise masks
# ═══════════════════════════════════════════════════════════════════════════


def finalise_masks(
    inner: np.ndarray,
    outer: np.ndarray,
    min_hole_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Final cleanup: fill small holes in the combined mask, then re-split.

    *Optimisation*: ``remove_small_holes`` is restricted to the padded bounding
    box of the union so the internal int32 label array is only as large as the
    glandular region rather than the full slide (saves ~1-2 GB on a large WSI).
    Outer is returned unchanged (only inner can gain pixels from the fill).
    """
    inner_b = _as_bool(inner)
    outer_b = _as_bool(outer)

    pad = max(10, int(min_hole_size ** 0.5) + 2)
    bbox = _merge_bboxes(_padded_bbox(inner_b, pad=pad), _padded_bbox(outer_b, pad=pad))
    if bbox is None:
        return np.zeros_like(inner_b), outer_b

    union_crop = inner_b[bbox] | outer_b[bbox]
    _remove_small_holes_inplace(union_crop, min_hole_size)

    # Only inner changes (gains union-fill pixels that are not already outer).
    # Outside bbox inner_b is all-False, so zeros_like gives the right result.
    new_inner = np.zeros_like(inner_b)
    new_inner[bbox] = union_crop & ~outer_b[bbox]
    del union_crop

    return new_inner, outer_b


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 7 – Fill interior ditzels in inner
# ═══════════════════════════════════════════════════════════════════════════


def fill_inner_holes(inner: np.ndarray) -> np.ndarray:
    """Fill binary holes inside inner components (fixes ditzels).

    *Optimisation*: labelling is restricted to the bounding box of the inner
    mask instead of the full slide array.  Each component is then processed
    independently on its tight crop.  ``scipy.ndimage.find_objects`` is used
    instead of ``skimage.measure.regionprops`` so that the int32 label array
    is freed *before* the per-component loop — saving ≈ 1.3 GB on a large
    slide (``regionprops`` stores internal views into the label array, so a
    bare ``del labeled_crop`` after ``regionprops`` does not free the memory).
    """
    inner_b = _as_bool(inner)

    # Restrict labelling to the bounding box of the inner mask.
    bbox = _roi_bounding_box(inner_b)
    if bbox is None:
        return inner_b

    inner_crop = inner_b[bbox].copy()         # contiguous copy for scipy
    labeled_crop = np.empty(inner_crop.shape, dtype=np.int32)
    n: int = int(scipy.ndimage.label(inner_crop, output=labeled_crop))
    if n == 0:
        return inner_b

    # find_objects returns lightweight slice tuples; del labeled_crop BEFORE
    # the per-component loop so the ≈ 1.3 GB int32 array is freed immediately.
    # regionprops stores internal views into labeled_crop so `del labeled_crop`
    # after regionprops does NOT free the memory.
    slices = scipy.ndimage.find_objects(labeled_crop)
    del labeled_crop
    result_crop = inner_crop.copy()
    for slc in slices:
        if slc is None:
            continue
        rslc, cslc = slc
        rlo = max(0, rslc.start - 1)
        rhi = min(inner_crop.shape[0], rslc.stop + 1)
        clo = max(0, cslc.start - 1)
        chi = min(inner_crop.shape[1], cslc.stop + 1)
        crop = inner_crop[rlo:rhi, clo:chi]
        filled = scipy.ndimage.binary_fill_holes(crop)
        result_crop[rlo:rhi, clo:chi] |= filled
        del filled

    result = inner_b.copy()
    result[bbox] = result_crop
    del result_crop
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

    *Optimisation*: the erosion is restricted to a padded bounding box of
    the combined mask so we never allocate a full-size dilated/eroded array.
    """
    inner_b = _as_bool(inner)
    outer_b = _as_bool(outer)

    struct = skimage.morphology.disk(border_px)
    pad = border_px + 1
    bbox = _merge_bboxes(
        _padded_bbox(inner_b, pad=pad),
        _padded_bbox(outer_b, pad=pad),
    )
    if bbox is None:
        return inner_b, outer_b

    combined_crop = inner_b[bbox] | outer_b[bbox]
    eroded_crop = scipy.ndimage.binary_erosion(combined_crop, structure=struct)
    ring_crop = combined_crop & ~eroded_crop
    del eroded_crop, combined_crop

    new_outer = outer_b.copy()
    new_outer[bbox] |= ring_crop
    del ring_crop

    return inner_b, new_outer


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 9 – Nuclei segmentation from hematoxylin
# ═══════════════════════════════════════════════════════════════════════════


def _equalise_hematoxylin(channel: np.ndarray) -> np.ndarray:
    """Return a histogram-equalised **uint8** image, masking exact-zero pixels.

    Replaces ``skimage.exposure.equalize_hist`` (which returns float64) with a
    256-entry LUT-based implementation that stays entirely in uint8.  For a
    ~320 Mpx tissue crop this eliminates the ~2.5 GB float64 intermediate,
    reducing peak RAM by the same amount.  Callers that compare the result to a
    float threshold in [0, 1] must scale the threshold to [0, 255] first
    (see ``segment_nuclei``).

    Zero pixels (background / padding from non-tissue tiles) are excluded from
    the histogram so they do not shift the CDF, and are set to 0 in the output.
    """
    if channel.dtype.kind == "f":
        u8 = skimage.util.img_as_ubyte(np.clip(channel, 0, 1))
    else:
        u8 = np.asarray(channel, dtype=np.uint8)

    mask = u8 > 0
    if not mask.any():
        return np.zeros_like(u8)

    # Build histogram over non-zero pixels; exclude 0 from the CDF.
    hist = np.bincount(u8[mask].ravel(), minlength=256)
    hist[0] = 0
    cdf = hist.cumsum()
    nz = cdf > 0
    cdf_min = int(cdf[nz][0]) if nz.any() else 0
    total = int(mask.sum())
    denom = total - cdf_min
    if denom <= 0:
        return np.zeros_like(u8)

    # 256-entry uint8 LUT; apply with direct integer indexing.
    lut = (np.round((cdf - cdf_min).clip(0) / denom * 255)).astype(np.uint8)
    lut[0] = 0  # keep background pixels as 0
    result = lut[u8]
    result[~mask] = 0
    return result


def segment_nuclei(
    hematoxylin_channel: np.ndarray,
    outer_mask: np.ndarray,
    inner_mask: np.ndarray | None = None,
    threshold: float = 0.01,
    tissue_mask: np.ndarray | None = None,
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
        Default 0.01.
    tissue_mask:
        If provided, "other nuclei" are restricted to tissue regions.
        This prevents spurious detections in background/glass areas
        where the hematoxylin channel may be zero (e.g. from tiled
        pipelines that skip non-tissue tiles).

    *Optimisation*: histogram equalisation is restricted to the tissue
    bounding box so the float64 intermediate is much smaller than the
    full slide array.
    """
    shape = hematoxylin_channel.shape
    outer_b = _as_bool(outer_mask)
    inner_b = _as_bool(inner_mask) if inner_mask is not None else None

    # Determine the working bbox — use tissue if available, else outer.
    # Merge with the bbox of non-zero hematoxylin values so that the CDF
    # of the histogram equalisation is identical to the full-array version.
    ref = _as_bool(tissue_mask) if tissue_mask is not None else outer_b
    bbox = _roi_bounding_box(ref)
    if bbox is None:
        return np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)

    # Fast O(H+W) scan for non-zero hematoxylin rows/columns.
    rows = np.where(np.any(hematoxylin_channel, axis=1))[0]
    cols = np.where(np.any(hematoxylin_channel, axis=0))[0]
    if rows.size and cols.size:
        h_bbox = (slice(int(rows[0]), int(rows[-1]) + 1),
                  slice(int(cols[0]), int(cols[-1]) + 1))
        bbox = _merge_bboxes(bbox, h_bbox)

    # Histogram equalise within the bbox only.  _equalise_hematoxylin now
    # returns uint8 (saves ~2.5 GB vs the previous float64 path); scale the
    # float threshold to [0, 255] before comparing.
    eq_crop = _equalise_hematoxylin(hematoxylin_channel[bbox])
    uint8_thr = max(1, round(threshold * 255))
    nuclei_crop = eq_crop < uint8_thr
    del eq_crop

    outer_crop = outer_b[bbox]
    epi_crop = nuclei_crop & outer_crop

    exclude_crop = outer_crop
    if inner_b is not None:
        exclude_crop = exclude_crop | inner_b[bbox]
    other_crop = nuclei_crop.copy()
    other_crop[exclude_crop] = False  # mask in-place; avoids ~exclude_crop temp (0.25 GB)
    del nuclei_crop, exclude_crop

    if tissue_mask is not None:
        np.logical_and(other_crop, ref[bbox], out=other_crop)

    # Map back to full resolution.
    epithelial = np.zeros(shape, dtype=bool)
    epithelial[bbox] = epi_crop
    other = np.zeros(shape, dtype=bool)
    other[bbox] = other_crop
    del epi_crop, other_crop

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

    *Optimisation*: avoids redundant ``.astype(bool)`` copies when masks
    are already boolean (the normal case after prior pipeline stages).
    """
    inner_b = _as_bool(masks.inner)
    outer_b = _as_bool(masks.outer)

    if masks.tissue is not None:
        tissue_b = _as_bool(masks.tissue)
        # Fill the label map as: everything=white, tissue→stroma, then
        # overwrite with inner/outer/nuclei/urethra in priority order.
        # This avoids ~tissue_b (322 MB temp) and a stroma copy (322 MB).
        lm = np.full(shape, fill_value=LABEL_MAPPING["white"], dtype=np.uint8)
        lm[tissue_b] = LABEL_MAPPING["background_tissue"]
    else:
        lm = np.zeros(shape, dtype=np.uint8)
        lm[~(inner_b | outer_b)] = LABEL_MAPPING["background_tissue"]

    lm[inner_b] = LABEL_MAPPING["inner"]
    lm[outer_b] = LABEL_MAPPING["outer"]
    lm[_as_bool(masks.epithelial_nuclei)] = LABEL_MAPPING["epithelial_nuclei"]
    lm[_as_bool(masks.other_nuclei)] = LABEL_MAPPING["other_nuclei"]

    if masks.urethra is not None:
        lm[_as_bool(masks.urethra)] = LABEL_MAPPING["urethra"]

    return lm


# ═══════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════


def post_process(
    inner_pred: np.ndarray,
    outer_pred: np.ndarray,
    tissue_mask: np.ndarray | None,
    hematoxylin_channel: np.ndarray,
    *,
    mode: Literal["wsi", "biopsy", "tile"] = "wsi",
    profile_name: str = "best_effort",
    nuclei_threshold: float = 0.01,
    inner_border_px: int = 2,
    tile_pad: int | None = None,
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
        label map are derived as ``~tissue_mask``.  Required for
        ``mode='wsi'`` and ``mode='biopsy'``; pass ``None`` (or omit) for
        ``mode='tile'``.
    hematoxylin_channel:
        EHO ch1 (hematoxylin) for nuclei segmentation, uint8 or float.
    mode:
        Processing mode that controls which pipeline stages run:

        ``'wsi'`` *(default)*
            Full whole-slide pipeline.  Tissue mask is used to restrict
            predictions, and the central urethra component is detected and
            labelled separately.

        ``'biopsy'``
            Same as ``'wsi'`` but without urethra detection.  Suitable for
            needle-biopsy cores where no urethra is present.

        ``'tile'``
            Single-tile pipeline.  No tissue mask is used and no urethra is
            detected.  The inner/outer predictions and hematoxylin channel
            are reflect-padded by *tile_pad* pixels before morphological
            operations so that holes and structures touching the tile edge
            are handled correctly; the padding is stripped from all outputs
            before the final label map is assembled.
    profile_name:
        One of ``"best_effort"``, ``"precise"``, ``"sensitive"``.
    nuclei_threshold:
        Threshold for nuclei detection (lower = more conservative).
    inner_border_px:
        Pixels to erode from combined → assign to outer as border ring.
    tile_pad:
        Reflect-padding (pixels) applied on every side in ``mode='tile'``.
        When ``None`` (default), automatically set to 50 % of the shorter
        spatial dimension of *inner_pred*, i.e.
        ``min(inner_pred.shape) // 2``.  This simulates up to a full gland
        worth of surrounding context on every edge.  Pass an explicit
        integer to override.
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

    if mode not in ("wsi", "biopsy", "tile"):
        raise ValueError(f"mode must be 'wsi', 'biopsy', or 'tile'; got {mode!r}")

    # Resolve tile_pad: default to 50 % of the shorter spatial dimension
    # so that up to one full gland-width of context is reflected on every edge.
    if tile_pad is None:
        tile_pad = min(inner_pred.shape[0], inner_pred.shape[1]) // 2

    if verbose:
        print(
            f"post_process — mode={mode}  profile={profile_name}  "
            f"nuclei_thr={nuclei_threshold}  border={inner_border_px}px"
        )

    # ── Stage 1: Tissue restriction / tile padding ───────────────────────
    if mode == "tile":
        # Reflect-pad all spatial inputs to simulate surrounding tissue.
        # This prevents morphological operations (hole-fill, small-object
        # removal, filter dilation) from being misled by the hard image
        # boundary.  All outputs are cropped back to the original size
        # after nuclei segmentation.
        p = tile_pad
        inner: np.ndarray = np.pad(inner_pred.astype(bool), p, mode="reflect")
        outer: np.ndarray = np.pad(outer_pred.astype(bool), p, mode="reflect")
        hema_work: np.ndarray = np.pad(hematoxylin_channel, p, mode="reflect")
        # Treat the entire padded region as tissue so downstream steps
        # never filter on tissue membership.
        tissue: np.ndarray = np.ones(inner.shape, dtype=bool)
    else:
        if tissue_mask is None:
            raise ValueError(
                "tissue_mask is required for mode='wsi' and mode='biopsy'. "
                "Use mode='tile' for single-tile inference without a tissue mask."
            )
        inner, outer, tissue = _run(
            "restrict_to_tissue",
            restrict_predictions_to_tissue,
            inner_pred,
            outer_pred,
            tissue_mask,
        )
        hema_work = hematoxylin_channel

    # ── Stage 2: Urethra detection (WSI only) ────────────────────────────
    if mode == "wsi":
        urethra = _run("detect_urethra", detect_urethra, inner, outer)
    else:
        # Biopsy and tile modes: no urethra present.
        urethra = None

    # ── Stages 3–8: Morphological cleaning and nuclei segmentation ───────
    inner, outer = _run("clean_small_holes", clean_small_holes, inner, outer)

    # apply_filters never mutates its inputs — no need to copy.
    inner, outer = _run("apply_filters", apply_filters, inner, outer, profile)

    inner, outer = _run("finalise_masks", finalise_masks, inner, outer)
    inner = _run("fill_inner_holes", fill_inner_holes, inner)
    inner, outer = _run(
        "ensure_inner_border", ensure_inner_border, inner, outer, border_px=inner_border_px
    )
    epi_nuclei, other_nuclei = _run(
        "segment_nuclei",
        segment_nuclei,
        hema_work,
        outer,
        inner,
        nuclei_threshold,
        tissue,
    )

    # ── Tile mode: strip the reflect padding from all spatial outputs ─────
    if mode == "tile":
        p = tile_pad
        inner = inner[p:-p, p:-p]
        outer = outer[p:-p, p:-p]
        epi_nuclei = epi_nuclei[p:-p, p:-p]
        other_nuclei = other_nuclei[p:-p, p:-p]
        tissue = tissue[p:-p, p:-p]  # all-True slice; kept for assign_labels

    mask_set = MaskSet(
        inner=inner,
        outer=outer,
        epithelial_nuclei=epi_nuclei,
        other_nuclei=other_nuclei,
        tissue=tissue,
        urethra=urethra,
    )
    labeled = _run("assign_labels", assign_labels, mask_set, inner.shape)

    # Free heavyweight masks immediately; only the labelled map is needed.
    del inner, outer, tissue, urethra, epi_nuclei, other_nuclei, mask_set

    if verbose:
        total = sum(timings.values())
        print(f"  TOTAL: {total:.2f}s")

    if return_timings:
        return labeled, timings
    return labeled
