# SPDX-FileCopyrightText: 2024-present barrettMCW <mjbarrett@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""
OMERO-backed whole-slide image inference.

Requires the ``omero`` optional dependency::

    pip install shell[omero]

Architecture
------------
1. Fetches a low-resolution thumbnail to build a tissue mask.
2. Identifies background tiles from the mask and fetches a handful to
   estimate background illumination intensity (*Io*).
3. Estimates stain parameters from the thumbnail (already downsampled).
4. Uses a **producer/consumer pipeline** — a dedicated fetch thread
   pulls tiles from OMERO (sequentially, to keep Ice WSS happy) and
   applies EHO normalisation, while the main thread runs model inference
   on tiles already in a bounded queue.

The fetch thread never imports or touches ``torch``.  ``torch`` is loaded
lazily in the main thread only once the fetch thread is running.  Since
numpy releases the GIL, EHO work in the fetch thread runs concurrently
with torch inference in the main thread.

Peak memory is proportional to ``prefetch_depth`` tiles rather than the
full slide.
"""

from __future__ import annotations

import gc
import logging
import os
import queue
import threading
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import numpy as np

if TYPE_CHECKING:
    from omero.gateway import BlitzGateway

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional OMERO dependency gate
# ---------------------------------------------------------------------------
# Lazy OMERO loader: defer importing the heavy omero modules until runtime.
# This avoids importing OMERO (and its C extensions) at module import time
# which can race with other libraries (e.g. torch / pyvips) and cause
# platform-specific crashes.
_OMERO_BG_CLASS = None
_OMERO_LOADED = False
PIXEL_TYPES: dict = {}


def _ensure_omero() -> None:
    """Attempt to import OMERO & populate `PIXEL_TYPES` / `_OMERO_BG_CLASS`.

    This is idempotent and safe to call from any runtime location. If the
    import fails the function leaves `_OMERO_LOADED` as False.
    """
    global _OMERO_BG_CLASS, PIXEL_TYPES, _OMERO_LOADED
    if _OMERO_LOADED:
        return
    try:
        import omero  # noqa: F401  (import to ensure runtime available)
        from omero.gateway import BlitzGateway as _BG
        from omero.model import enums as omero_enums

        # Map likely enum attribute names to numpy dtypes, but use getattr
        # so code is robust to differences across OMERO versions.
        _mapping = {
            "PixelsTypeint8": np.int8,
            "PixelsTypeuint8": np.uint8,
            "PixelsTypeint16": np.int16,
            "PixelsTypeuint16": np.uint16,
            "PixelsTypeint32": np.int32,
            "PixelsTypeuint32": np.uint32,
            "PixelsTypefloat": np.float32,
            "PixelsTypedouble": np.float64,
        }

        pix = {}
        for name, dtype in _mapping.items():
            enum_val = getattr(omero_enums, name, None)
            if enum_val is not None:
                pix[enum_val] = dtype

        PIXEL_TYPES.clear()
        PIXEL_TYPES.update(pix)

        _OMERO_BG_CLASS = _BG
        _OMERO_LOADED = True
    except Exception:
        # Leave PIXEL_TYPES empty and _OMERO_LOADED False on failure.
        PIXEL_TYPES.clear()
        _OMERO_BG_CLASS = None
        _OMERO_LOADED = False


def _require_omero() -> None:
    # Ensure the OMERO runtime is available at call time. This calls the
    # lazy loader which attempts to import OMERO and populate the runtime
    # objects. If import fails we raise the usual ImportError.
    _ensure_omero()
    if not _OMERO_LOADED:
        msg = (
            "The 'omero' optional dependency is required for OMERO support. "
            "Install it with: pip install shell[omero]"
        )
        raise ImportError(msg)


# ---------------------------------------------------------------------------
# Sentinel used to signal the consumer that the producer is done.
# ---------------------------------------------------------------------------
_SENTINEL = object()

# ---------------------------------------------------------------------------
# Lazy Macenko imports
# ---------------------------------------------------------------------------
_MACENKO_IMPORTS: dict[str, Any] | None = None


def _macenko() -> dict[str, Any]:
    """Lazy-import macenko_pca symbols."""
    global _MACENKO_IMPORTS
    if _MACENKO_IMPORTS is None:
        from macenko_pca import (
            find_stain_index,
            rgb_separate_stains_macenko_pca,
            stain_color_map,
        )

        _MACENKO_IMPORTS = {
            "find_stain_index": find_stain_index,
            "rgb_separate_stains_macenko_pca": rgb_separate_stains_macenko_pca,
            "stain_color_map": stain_color_map,
        }
    return _MACENKO_IMPORTS


# ---------------------------------------------------------------------------
# OMERO connection helpers
# ---------------------------------------------------------------------------


def create_omero_connection(
    host: str,
    port: int,
    username: str,
    password: str,
    *,
    secure: bool = True,
) -> BlitzGateway:
    """Create and return a connected :class:`BlitzGateway`.

    Handles both standard Ice connections (plain hostname) and WebSocket
    connections (``wss://`` or ``ws://`` URLs, e.g. through an nginx
    reverse-proxy).

    Parameters
    ----------
    host : str
        OMERO server hostname **or** a ``wss://``/``ws://`` URL.
    port : int
        Server port (used as fallback when the URL doesn't include one).
    username, password : str
        OMERO credentials.
    secure : bool
        Enable SSL for standard Ice connections (ignored for ``wss``/``ws``).

    Returns
    -------
    BlitzGateway
        A connected gateway instance.
    """
    _require_omero()
    # Use the lazily-loaded BlitzGateway class populated by _ensure_omero().
    _BG = _OMERO_BG_CLASS
    if _BG is None:
        # Defensive: if the class is not available treat as missing dependency.
        raise ImportError(
            "OMERO BlitzGateway class not available; ensure OMERO is installed"
        )

    parsed = urlparse(host)

    # Diagnostic print to help trace macOS crashes during connection setup.
    try:
        logger.debug(
            f"DIAG: create_omero_connection start host={host!r} port={port} scheme={parsed.scheme!r}"
        )
    except Exception:
        # Best-effort diagnostic; do not stop the connection logic if logging fails.
        pass

    if parsed.scheme in ("wss", "ws"):
        import omero  # type: ignore[import-untyped]

        ws_host = parsed.hostname
        if not ws_host:
            msg = f"Could not parse hostname from URL: {host}"
            raise ValueError(msg)
        ws_port = parsed.port or port or (443 if parsed.scheme == "wss" else 80)
        ws_path = parsed.path or "/"
        proto = parsed.scheme

        router = f"OMERO.Glacier2/router:{proto} -p {ws_port} -h {ws_host} -r {ws_path}"
        logger.info(
            "Creating OMERO client with %s router: %s:%s%s",
            proto.upper(),
            ws_host,
            ws_port,
            ws_path,
        )

        try:
            logger.debug(
                f"DIAG: creating omero.client with router={router!r} (proto={proto})"
            )
        except Exception:
            pass

        client = omero.client(args=["--Ice.Default.Router=" + router])
        try:
            client.createSession(username, password)
            try:
                logger.debug("DIAG: omero.client.createSession succeeded")
            except Exception:
                pass
        except Exception as exc:
            msg = (
                f"Failed to create OMERO session via "
                f"{proto}://{ws_host}:{ws_port}{ws_path}: {exc}"
            )
            raise RuntimeError(msg) from exc

        conn = _BG(client_obj=client)
        # If we created a raw omero.client earlier (websocket path) attach it
        # to the BlitzGateway instance so it can be closed explicitly later.
        try:
            setattr(conn, "_omero_ws_client", client)
        except Exception:
            pass
        try:
            connected = conn.connect()
            try:
                logger.debug(f"DIAG: BlitzGateway.connect() returned {connected!r}")
            except Exception:
                pass
        except Exception as exc:
            # Surface the connect() error with a diagnostic print before raising.
            try:
                logger.debug(f"DIAG: BlitzGateway.connect() raised: {exc}")
            except Exception:
                pass
            raise

        if not connected:
            msg = (
                f"BlitzGateway.connect() failed after session creation "
                f"({proto}://{ws_host}:{ws_port}{ws_path})"
            )
            raise RuntimeError(msg)
        try:
            logger.debug(
                "DIAG: create_omero_connection returning connected WS BlitzGateway"
            )
        except Exception:
            pass
        return conn

    # Standard Ice connection
    clean_host = parsed.hostname or host
    actual_port = parsed.port or port

    try:
        logger.debug(
            f"DIAG: creating standard BlitzGateway to {clean_host}:{actual_port} secure={secure}"
        )
    except Exception:
        pass

    conn = _BG(
        username,
        password,
        host=clean_host,
        port=actual_port,
        secure=secure,
    )
    try:
        connected = conn.connect()
        try:
            logger.debug(f"DIAG: BlitzGateway.connect() returned {connected!r}")
        except Exception:
            pass
    except Exception as exc:
        try:
            logger.debug(f"DIAG: BlitzGateway.connect() raised: {exc}")
        except Exception:
            pass
        raise

    if not connected:
        msg = f"Failed to connect to OMERO at {clean_host}:{actual_port}"
        raise RuntimeError(msg)

    try:
        logger.debug("DIAG: create_omero_connection returning connected BlitzGateway")
    except Exception:
        pass
    return conn


def _safe_close_blitzgateway(conn: Any) -> None:
    """Best-effort close of a BlitzGateway and any attached raw omero.client.

    Different OMERO transports / versions expose slightly different cleanup
    APIs.  Try several likely methods without raising to ensure Ice
    communicators are destroyed where possible.
    """
    try:
        # Preferred: BlitzGateway.close()
        if hasattr(conn, "close"):
            try:
                conn.close()
            except Exception:
                pass
    except Exception:
        pass

    try:
        # Some versions expose disconnect()
        if hasattr(conn, "disconnect"):
            try:
                conn.disconnect()
            except Exception:
                pass
    except Exception:
        pass

    try:
        # If we attached a raw omero.client during creation, try to close/destroy it.
        client_obj = getattr(conn, "_omero_ws_client", None)
        if client_obj is not None:
            if hasattr(client_obj, "close"):
                try:
                    client_obj.close()
                except Exception:
                    pass
            if hasattr(client_obj, "destroy"):
                try:
                    client_obj.destroy()
                except Exception:
                    pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Pixel helpers
# ---------------------------------------------------------------------------


def _safe_float(v: Any, default: float | None = None) -> float | None:
    try:
        if v is None:
            return default
        if hasattr(v, "getValue"):
            return float(v.getValue())
        return float(v)
    except Exception:
        return default


def _get_physical_sizes_um(image: Any) -> tuple[float, float]:
    """Return ``(mpp_x, mpp_y)`` in microns/pixel, with robust fallbacks."""
    mpp_x: float | None = None
    mpp_y: float | None = None

    try:
        mpp_x = _safe_float(image.getPixelSizeX(), None)
    except Exception:
        pass
    try:
        mpp_y = _safe_float(image.getPixelSizeY(), None)
    except Exception:
        pass

    if mpp_x is None or mpp_y is None:
        try:
            pix = image.getPrimaryPixels()._obj
            if mpp_x is None:
                mpp_x = _safe_float(pix.getPhysicalSizeX(), None)
            if mpp_y is None:
                mpp_y = _safe_float(pix.getPhysicalSizeY(), None)
        except Exception:
            pass

    if mpp_x is None:
        mpp_x = 1.0
    if mpp_y is None:
        mpp_y = 1.0

    return mpp_x, mpp_y


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    lo = np.percentile(arr, 1)
    hi = np.percentile(arr, 99)
    arr = (arr - lo) / (hi - lo + 1e-8)
    np.clip(arr, 0, 1, out=arr)
    return (arr * 255).astype(np.uint8)


def _resize_tile(tile: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if tile.shape[0] == out_h and tile.shape[1] == out_w:
        return tile
    from scipy.ndimage import zoom as _scipy_zoom

    zy = out_h / tile.shape[0]
    zx = out_w / tile.shape[1]
    return np.clip(
        _scipy_zoom(tile.astype(np.float32), (zy, zx), order=1), 0, 255
    ).astype(np.uint8)


# ---------------------------------------------------------------------------
# OMERO tile fetching
# ---------------------------------------------------------------------------


def _fetch_channel_tiles(
    conn: BlitzGateway,
    pixels_id: int,
    level: int,
    c_idx: int,
    band_idx: int,
    level_sz_x: int,
    level_sz_y: int,
    tile_w: int,
    tile_h: int,
    out_w: int,
    out_h: int,
    sx: float,
    sy: float,
    dtype: Any,
    crop_lx0: int = 0,
    crop_ly0: int = 0,
    crop_lx1: int | None = None,
    crop_ly1: int | None = None,
) -> tuple[int, np.ndarray]:
    """Fetch tiles for one OMERO channel at a given pyramid level."""
    if crop_lx1 is None:
        crop_lx1 = level_sz_x
    if crop_ly1 is None:
        crop_ly1 = level_sz_y

    store = conn.c.sf.createRawPixelsStore()
    canvas = np.zeros((out_h, out_w), dtype=np.uint8)
    try:
        store.setPixelsId(pixels_id, False)
        store.setResolutionLevel(level)

        start_ty = (crop_ly0 // tile_h) * tile_h
        start_tx = (crop_lx0 // tile_w) * tile_w

        for ty in range(start_ty, crop_ly1, tile_h):
            ry0 = max(ty, crop_ly0)
            ry1 = min(ty + tile_h, crop_ly1, level_sz_y)
            rh = ry1 - ry0
            if rh <= 0:
                continue
            oy0 = round((ry0 - crop_ly0) * sy)
            oy1 = round((ry1 - crop_ly0) * sy)
            oh = max(1, oy1 - oy0)

            for tx in range(start_tx, crop_lx1, tile_w):
                rx0 = max(tx, crop_lx0)
                rx1 = min(tx + tile_w, crop_lx1, level_sz_x)
                rw = rx1 - rx0
                if rw <= 0:
                    continue
                ox0 = round((rx0 - crop_lx0) * sx)
                ox1 = round((rx1 - crop_lx0) * sx)
                ow = max(1, ox1 - ox0)

                raw = store.getTile(0, c_idx, 0, rx0, ry0, rw, rh)
                arr = np.frombuffer(raw, dtype=dtype).reshape(rh, rw).copy()
                arr_u8 = _to_uint8(arr)
                arr_rs = _resize_tile(arr_u8, oh, ow)
                oy1c = min(out_h, oy0 + arr_rs.shape[0])
                ox1c = min(out_w, ox0 + arr_rs.shape[1])
                canvas[oy0:oy1c, ox0:ox1c] = arr_rs[: oy1c - oy0, : ox1c - ox0]
    finally:
        try:
            store.close()
        except Exception:
            pass
    return band_idx, canvas


def _fetch_region_rgb_single_store(
    conn: BlitzGateway,
    pixels_id: int,
    level: int,
    channel_ids: list[int],
    level_sz_x: int,
    level_sz_y: int,
    tile_w: int,
    tile_h: int,
    dtype: Any,
    *,
    lx0: int,
    ly0: int,
    lx1: int,
    ly1: int,
    out_w: int,
    out_h: int,
    store: Any | None = None,
) -> np.ndarray:
    """Fetch a rectangular RGB region using a **single** RawPixelsStore.

    Attempts a **bulk** fetch first — one ``getTile`` call per channel
    for the entire region — which dramatically reduces network
    round-trips (3 calls instead of ``ceil(W/tile_w) * ceil(H/tile_h)
    * 3``).  Falls back to native-tile-sized iteration when the server
    rejects the large request.

    Parameters
    ----------
    store : optional
        A pre-opened ``RawPixelsStore`` already configured with the
        correct *pixels_id* and *resolution level*.  When provided the
        caller is responsible for closing the store; this function will
        **not** close it.  When ``None`` (the default) a fresh store is
        created and closed automatically as before.

    Returns
    -------
    np.ndarray
        ``(out_h, out_w, 3)`` uint8 RGB array.
    """
    region_lw = lx1 - lx0
    region_lh = ly1 - ly0
    if region_lw <= 0 or region_lh <= 0:
        return np.full((out_h, out_w, 3), 255, dtype=np.uint8)

    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    owns_store = store is None
    if owns_store:
        store = conn.c.sf.createRawPixelsStore()
        store.setPixelsId(pixels_id, False)
        store.setResolutionLevel(level)

    try:
        # ── Fast path: bulk fetch (3 calls instead of N*3) ────────────
        try:
            for b, c_idx in enumerate(channel_ids):
                raw = store.getTile(
                    0,
                    c_idx,
                    0,
                    lx0,
                    ly0,
                    region_lw,
                    region_lh,
                )
                arr = (
                    np.frombuffer(raw, dtype=dtype).reshape(region_lh, region_lw).copy()
                )
                arr_u8 = _to_uint8(arr)
                del arr
                if arr_u8.shape[0] != out_h or arr_u8.shape[1] != out_w:
                    arr_u8 = _resize_tile(arr_u8, out_h, out_w)
                rgb[: arr_u8.shape[0], : arr_u8.shape[1], b] = arr_u8[:out_h, :out_w]
                del arr_u8
            return rgb
        except Exception:
            logger.debug(
                "Bulk fetch failed for region %dx%d; falling back to tiled",
                region_lw,
                region_lh,
            )
            rgb[:] = 0

        # ── Slow path: iterate over native tiles ─────────────────────
        sx = out_w / region_lw
        sy = out_h / region_lh

        start_ty = (ly0 // tile_h) * tile_h
        start_tx = (lx0 // tile_w) * tile_w

        for ty in range(start_ty, ly1, tile_h):
            ry0 = max(ty, ly0)
            ry1 = min(ty + tile_h, ly1, level_sz_y)
            rh = ry1 - ry0
            if rh <= 0:
                continue
            oy0 = round((ry0 - ly0) * sy)
            oy1 = round((ry1 - ly0) * sy)
            oh = max(1, oy1 - oy0)

            for tx in range(start_tx, lx1, tile_w):
                rx0 = max(tx, lx0)
                rx1 = min(tx + tile_w, lx1, level_sz_x)
                rw = rx1 - rx0
                if rw <= 0:
                    continue
                ox0 = round((rx0 - lx0) * sx)
                ox1 = round((rx1 - lx0) * sx)
                ow = max(1, ox1 - ox0)

                for b, c_idx in enumerate(channel_ids):
                    raw = store.getTile(0, c_idx, 0, rx0, ry0, rw, rh)
                    arr = np.frombuffer(raw, dtype=dtype).reshape(rh, rw).copy()
                    arr_u8 = _to_uint8(arr)
                    arr_rs = _resize_tile(arr_u8, oh, ow)
                    oy1c = min(out_h, oy0 + arr_rs.shape[0])
                    ox1c = min(out_w, ox0 + arr_rs.shape[1])
                    rgb[oy0:oy1c, ox0:ox1c, b] = arr_rs[: oy1c - oy0, : ox1c - ox0]
    finally:
        if owns_store:
            try:
                store.close()
            except Exception:
                pass

    return rgb


# ---------------------------------------------------------------------------
# Stain parameter estimation helpers
# ---------------------------------------------------------------------------


def _estimate_stain_params_with_io(
    rgb: np.ndarray,
    bg_mask: np.ndarray,
    io_val: float,
) -> dict[str, Any]:
    """Estimate Macenko stain parameters using a pre-computed *Io*.

    Accepts *Io* from an external source (e.g. full-resolution background
    tiles) rather than computing it from the supplied *rgb* array.
    """
    m = _macenko()
    find_stain_index = m["find_stain_index"]
    rgb_separate_stains_macenko_pca = m["rgb_separate_stains_macenko_pca"]
    stain_color_map = m["stain_color_map"]

    w_est = rgb_separate_stains_macenko_pca(
        rgb,
        bg_int=[io_val],
        mask_out=bg_mask,
    )

    h_idx = find_stain_index(stain_color_map["hematoxylin"], w_est)
    e_idx = find_stain_index(stain_color_map["eosin"], w_est)
    if h_idx == e_idx or h_idx == 2 or e_idx == 2:
        h_idx, e_idx = 0, 1

    io_f = np.float32(io_val)
    w_inv = np.linalg.pinv(w_est).astype(np.float32)
    e_vec = w_inv[e_idx]
    h_vec = w_inv[h_idx]

    od = np.clip(rgb.astype(np.float32), 1.0, io_f)
    od /= io_f
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
        ch = np.clip(rgb[..., c].astype(np.float32), 1.0, io_f)
        ch /= io_f
        np.log10(ch, out=ch)
        od_acc -= ch
    od_acc /= 3.0
    od_lo = float(np.percentile(od_acc, 1))
    od_hi = float(np.percentile(od_acc, 99))
    del od_acc

    return {
        "Io": io_val,
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
# Thumbnail / tissue-mask helpers
# ---------------------------------------------------------------------------


def _fetch_thumbnail_rgb(
    conn: BlitzGateway,
    pixels_id: int,
    channel_ids: list[int],
    res_descs: list[Any],
    num_levels: int,
    dtype: Any,
    *,
    min_thumb_size: int = 2000,
) -> tuple[np.ndarray, int, int, int]:
    """Fetch a low-resolution RGB thumbnail from the coarsest suitable level.

    Returns ``(rgb, thumb_level, thumb_lsz_x, thumb_lsz_y)``.
    """
    thumb_di = len(res_descs) - 1
    for di in range(len(res_descs) - 1, -1, -1):
        if max(res_descs[di].sizeX, res_descs[di].sizeY) >= min_thumb_size:
            thumb_di = di
            break

    thumb_lsz_x = res_descs[thumb_di].sizeX
    thumb_lsz_y = res_descs[thumb_di].sizeY
    thumb_level = num_levels - 1 - thumb_di

    probe = conn.c.sf.createRawPixelsStore()  # type: ignore[union-attr]
    try:
        probe.setPixelsId(pixels_id, False)
        probe.setResolutionLevel(thumb_level)
        ttw, tth = probe.getTileSize()
    finally:
        try:
            probe.close()
        except Exception:
            pass

    thumb = np.zeros((thumb_lsz_y, thumb_lsz_x, 3), dtype=np.uint8)
    for b in range(3):
        bi, cv = _fetch_channel_tiles(
            conn,
            pixels_id,
            thumb_level,
            channel_ids[b],
            b,
            thumb_lsz_x,
            thumb_lsz_y,
            ttw,
            tth,
            thumb_lsz_x,
            thumb_lsz_y,
            1.0,
            1.0,
            dtype,
        )
        thumb[..., bi] = cv

    return thumb, thumb_level, thumb_lsz_x, thumb_lsz_y


def _build_tissue_mask(thumb_rgb: np.ndarray) -> np.ndarray:
    """Return a boolean tissue mask (``True`` = tissue) from a thumbnail."""
    from shell.transforms import detect_background

    bg_mask = detect_background(thumb_rgb)
    return ~bg_mask


def _tile_has_tissue(
    tissue_mask: np.ndarray,
    thumb_h: int,
    thumb_w: int,
    out_h: int,
    out_w: int,
    oy0: int,
    ox0: int,
    oh: int,
    ow: int,
    min_tissue_frac: float = 0.01,
) -> bool:
    """Check whether an output-space tile overlaps with tissue."""
    sy = thumb_h / out_h
    sx = thumb_w / out_w
    my0 = max(0, int(oy0 * sy))
    mx0 = max(0, int(ox0 * sx))
    my1 = min(thumb_h, int((oy0 + oh) * sy) + 1)
    mx1 = min(thumb_w, int((ox0 + ow) * sx) + 1)
    if my1 <= my0 or mx1 <= mx0:
        return False
    region = tissue_mask[my0:my1, mx0:mx1]
    return float(np.mean(region)) >= min_tissue_frac


def _pick_background_tile_coords(
    tissue_mask: np.ndarray,
    thumb_h: int,
    thumb_w: int,
    out_h: int,
    out_w: int,
    tile_size: int,
    num_tiles: int,
) -> list[tuple[int, int, int, int]]:
    """Return output-space ``(oy0, ox0, oh, ow)`` for pure background tiles."""
    all_bg: list[tuple[int, int, int, int]] = []

    for oy0 in range(0, out_h, tile_size):
        oh = min(tile_size, out_h - oy0)
        for ox0 in range(0, out_w, tile_size):
            ow = min(tile_size, out_w - ox0)
            if not _tile_has_tissue(
                tissue_mask,
                thumb_h,
                thumb_w,
                out_h,
                out_w,
                oy0,
                ox0,
                oh,
                ow,
                min_tissue_frac=0.001,
            ):
                all_bg.append((oy0, ox0, oh, ow))

    if not all_bg:
        return []

    bg_tiles: list[tuple[int, int, int, int]] = []
    step = max(1, len(all_bg) // num_tiles)
    for i in range(0, len(all_bg), step):
        bg_tiles.append(all_bg[i])
        if len(bg_tiles) >= num_tiles:
            break

    return bg_tiles


def _fetch_region_rgb(
    conn: BlitzGateway,
    pixels_id: int,
    level: int,
    channel_ids: list[int],
    level_sz_x: int,
    level_sz_y: int,
    tile_w: int,
    tile_h: int,
    dtype: Any,
    *,
    lx0: int,
    ly0: int,
    lx1: int,
    ly1: int,
    out_w: int,
    out_h: int,
    store: Any | None = None,
) -> np.ndarray:
    """Fetch a rectangular region from OMERO as RGB uint8.

    Parameters
    ----------
    store : optional
        Pre-opened ``RawPixelsStore`` (see
        :func:`_fetch_region_rgb_single_store`).
    """
    return _fetch_region_rgb_single_store(
        conn,
        pixels_id,
        level,
        channel_ids,
        level_sz_x,
        level_sz_y,
        tile_w,
        tile_h,
        dtype,
        lx0=lx0,
        ly0=ly0,
        lx1=lx1,
        ly1=ly1,
        out_w=out_w,
        out_h=out_h,
        store=store,
    )


# ---------------------------------------------------------------------------
# Tile-grid builder (overlap-aware)
# ---------------------------------------------------------------------------


def _build_tile_grid(
    out_h: int,
    out_w: int,
    inference_tile_size: int,
    tile_overlap: int,
    tissue_mask: np.ndarray | None,
    thumb_h: int,
    thumb_w: int,
    min_tissue_frac: float,
    no_tissue_crop: bool,
) -> tuple[list[tuple[int, int, int, int, int, int, int, int]], int, int, int]:
    """Build a raster-order tile schedule with overlap and keep regions.

    Each entry is ``(row, col, oy0, ox0, oh, ow, keep_y0, keep_x0)``
    where *keep_y0* / *keep_x0* are the local offsets into the tile
    prediction that should be written to the output canvas.  The keep
    region extends from ``(keep_y0, keep_x0)`` to ``(keep_y0 + keep_h,
    keep_x0 + keep_w)`` with the heights / widths derived by the consumer
    from the tile position and its neighbours.

    Returns ``(schedule, num_rows, num_cols, tiles_total)``.
    """
    stride = max(1, inference_tile_size - tile_overlap)

    # Collect row / col origins
    row_origins = list(range(0, out_h, stride))
    col_origins = list(range(0, out_w, stride))
    num_rows = len(row_origins)
    num_cols = len(col_origins)

    schedule: list[tuple[int, int, int, int, int, int, int, int]] = []
    tiles_total = 0

    for ri, oy0 in enumerate(row_origins):
        oh = min(inference_tile_size, out_h - oy0)
        for ci, ox0 in enumerate(col_origins):
            ow = min(inference_tile_size, out_w - ox0)
            tiles_total += 1

            # Tissue gate
            if not no_tissue_crop and tissue_mask is not None:
                if not _tile_has_tissue(
                    tissue_mask,
                    thumb_h,
                    thumb_w,
                    out_h,
                    out_w,
                    oy0,
                    ox0,
                    oh,
                    ow,
                    min_tissue_frac=min_tissue_frac,
                ):
                    continue

            # Keep-region: split overlap evenly with neighbours.
            top_trim = tile_overlap // 2 if ri > 0 else 0
            left_trim = tile_overlap // 2 if ci > 0 else 0

            schedule.append((ri, ci, oy0, ox0, oh, ow, top_trim, left_trim))

    return schedule, num_rows, num_cols, tiles_total


def _precompute_tile_info(
    tile_schedule: list[tuple[int, int, int, int, int, int, int, int]],
    num_rows: int,
    num_cols: int,
    tile_overlap: int,
) -> list[tuple[int, int, int, int, int, int, int, int, int, int]]:
    """Precompute full keep regions (including keep_h / keep_w).

    Returns entries of
    ``(ri, ci, oy0, ox0, oh, ow, keep_y0, keep_x0, keep_h, keep_w)``.
    """
    result: list[tuple[int, int, int, int, int, int, int, int, int, int]] = []
    for ri, ci, oy0, ox0, oh, ow, keep_y0, keep_x0 in tile_schedule:
        bottom_trim = (
            (tile_overlap - tile_overlap // 2)
            if tile_overlap > 0 and ri < num_rows - 1
            else 0
        )
        right_trim = (
            (tile_overlap - tile_overlap // 2)
            if tile_overlap > 0 and ci < num_cols - 1
            else 0
        )
        if oh > 1:
            bottom_trim = min(bottom_trim, oh - keep_y0 - 1)
        else:
            bottom_trim = 0
        if ow > 1:
            right_trim = min(right_trim, ow - keep_x0 - 1)
        else:
            right_trim = 0
        keep_h = oh - keep_y0 - bottom_trim
        keep_w = ow - keep_x0 - right_trim
        result.append((ri, ci, oy0, ox0, oh, ow, keep_y0, keep_x0, keep_h, keep_w))
    return result


# ---------------------------------------------------------------------------
# Producer thread — fetch + EHO (no torch)
# ---------------------------------------------------------------------------


def _fetch_worker(
    worker_id: int,
    conn: "BlitzGateway",
    work_queue: queue.Queue[Any],
    result_queue: queue.Queue[Any],
    stop_event: threading.Event,
    pixels_id: int,
    best_level: int,
    channel_ids: list[int],
    best_lsz_x: int,
    best_lsz_y: int,
    tile_w: int,
    tile_h: int,
    dtype: Any,
    sx: float,
    sy: float,
    stain_params: dict[str, Any],
) -> None:
    """Single fetch worker — uses a BlitzGateway connection provided by the
    main thread.

    The main thread is responsible for creating the OMERO connections and
    passing a connection object to each worker. Workers must NOT close the
    shared connection; the coordinator will close connections after workers
    exit.

    Pulls tile descriptors from *work_queue*, fetches pixel data from
    OMERO (using the bulk-fetch fast-path when possible), applies EHO
    normalisation, and places the result on *result_queue*.

    The bounded *result_queue* provides back-pressure: if inference falls
    behind, workers block on ``put`` until space is available.
    """
    from shell.transforms import apply_eho_chunked

    # Connection is provided by the caller (created in the main/coordinator
    # thread). Worker owns a RawPixelsStore instance derived from it.
    store = None
    try:
        store = conn.c.sf.createRawPixelsStore()
        store.setPixelsId(pixels_id, False)
        store.setResolutionLevel(best_level)

        while not stop_event.is_set():
            try:
                item = work_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is _SENTINEL:
                # Put it back so other workers also see it and exit.
                work_queue.put(_SENTINEL)
                break

            (
                _ri,
                _ci,
                oy0,
                ox0,
                oh,
                ow,
                keep_y0,
                keep_x0,
                keep_h,
                keep_w,
            ) = item

            lx0 = round(ox0 / sx)
            ly0 = round(oy0 / sy)
            lx1 = min(best_lsz_x, round((ox0 + ow) / sx))
            ly1 = min(best_lsz_y, round((oy0 + oh) / sy))

            rgb = _fetch_region_rgb_single_store(
                conn,
                pixels_id,
                best_level,
                channel_ids,
                best_lsz_x,
                best_lsz_y,
                tile_w,
                tile_h,
                dtype,
                lx0=lx0,
                ly0=ly0,
                lx1=lx1,
                ly1=ly1,
                out_w=ow,
                out_h=oh,
                store=store,
            )

            tile_eho = apply_eho_chunked(rgb, **stain_params)
            del rgb

            result_queue.put(
                (oy0, ox0, oh, ow, keep_y0, keep_x0, keep_h, keep_w, tile_eho)
            )
    except Exception as exc:
        logger.exception("Fetch worker %d encountered an error", worker_id)
        result_queue.put(exc)
    finally:
        # Close only the RawPixelsStore opened by this worker.
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
        # Do NOT close the BlitzGateway connection here; the coordinator
        # (creator) is responsible for closing it.


def _parallel_fetch_coordinator(
    host: str,
    port: int,
    username: str,
    password: str,
    tile_infos: list[tuple[int, int, int, int, int, int, int, int, int, int]],
    result_queue: queue.Queue[Any],
    pixels_id: int,
    best_level: int,
    channel_ids: list[int],
    best_lsz_x: int,
    best_lsz_y: int,
    tile_w: int,
    tile_h: int,
    dtype: Any,
    sx: float,
    sy: float,
    stain_params: dict[str, Any],
    num_workers: int,
) -> None:
    """Start *num_workers* fetch threads, each with its own OMERO session.

    The coordinator now creates the BlitzGateway connections in the main
    coordinator thread and passes a dedicated connection to each worker.
    This avoids importing / initialising the OMERO native runtime inside
    worker threads (which can trigger macOS-level races and segfaults).
    """
    work_queue: queue.Queue[Any] = queue.Queue()
    stop_event = threading.Event()

    for info in tile_infos:
        work_queue.put(info)
    # Single poison pill — each worker that sees it puts it back.
    work_queue.put(_SENTINEL)

    n_tiles = len(tile_infos)
    logger.info(
        "Starting %d fetch worker(s) for %d tiles …",
        num_workers,
        n_tiles,
    )

    # Pre-create one BlitzGateway connection per worker in this (main)
    # coordinator thread to avoid initialising OMERO / Ice inside workers.
    conns: list[Any] = []
    for i in range(num_workers):
        try:
            conn = create_omero_connection(host, port, username, password)
            conns.append(conn)
            logger.debug("Created OMERO connection for worker %d", i)
        except Exception as exc:
            # Clean up any already-created connections and re-raise.
            for c in conns:
                try:
                    c.close()
                except Exception:
                    pass
            logger.exception("Failed to create OMERO connection for worker %d", i)
            raise

    workers: list[threading.Thread] = []
    for i in range(num_workers):
        t = threading.Thread(
            target=_fetch_worker,
            kwargs={
                "worker_id": i,
                "conn": conns[i],
                "work_queue": work_queue,
                "result_queue": result_queue,
                "stop_event": stop_event,
                "pixels_id": pixels_id,
                "best_level": best_level,
                "channel_ids": channel_ids,
                "best_lsz_x": best_lsz_x,
                "best_lsz_y": best_lsz_y,
                "tile_w": tile_w,
                "tile_h": tile_h,
                "dtype": dtype,
                "sx": sx,
                "sy": sy,
                "stain_params": stain_params,
            },
            name=f"omero-fetch-{i}",
            daemon=True,
        )
        t.start()
        workers.append(t)

    for t in workers:
        t.join()

    result_queue.put(_SENTINEL)
    logger.info("All fetch workers finished.")

    # Close all pre-created BlitzGateway connections used by the workers.
    # Use the best-effort helper to avoid leaving Ice communicators open.
    for c in conns:
        try:
            _safe_close_blitzgateway(c)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def _save_results(
    pred: np.ndarray,
    eho_canvas: np.ndarray | None,
    save_eho: str | None,
    output_path: str,
) -> np.ndarray:
    """Write prediction (and optional EHO) images to disk.

    Output format is inferred from the file extension (``.png``, ``.tiff``,
    etc.) via :meth:`pyvips.Image.write_to_file`.

    ``pyvips`` is imported lazily here (not at module level) so that
    libvips initialises its thread runtime *after* PyTorch has already
    initialised its own.  On macOS the reverse order causes a segfault
    because both libraries try to own the process's OpenMP / thread-pool
    runtime.
    """
    import pyvips  # must come after torch is loaded — see note above

    if save_eho and eho_canvas is not None:
        os.makedirs(os.path.dirname(save_eho) or ".", exist_ok=True)
        pyvips.Image.new_from_array(eho_canvas).write_to_file(save_eho)
        logger.info("Saved EHO image to %s", save_eho)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pyvips.Image.new_from_array(pred).write_to_file(output_path)
    logger.info("Saved prediction to %s", output_path)

    return pred


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def infer_omero_wsi(
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    image_id: int,
    model_path: str | None = None,
    output_path: str,
    target_mpp: float = 1.0,
    group_id: int | None = None,
    save_eho: str | None = None,
    no_tissue_crop: bool = False,
    device: str = "auto",
    model_version: str | None = None,
    inference_tile_size: int = 2048,
    tile_overlap: int = 128,
    sw_overlap: float = 0.25,
    num_bg_tiles: int = 4,
    min_thumb_size: int = 2000,
    min_tissue_frac: float = 0.01,
    prefetch_depth: int = 8,
    num_fetch_workers: int = 4,
) -> np.ndarray:
    """Tile-based OMERO inference pipeline.

    Overlaps network I/O with model inference using a producer/consumer
    architecture (like a PyTorch DataLoader):

    * A **fetch thread** sequentially pulls tiles from OMERO, applies EHO
      colour normalisation (pure numpy — releases the GIL), and places
      ready-to-infer tiles into a bounded queue.
    * The **main thread** pulls tiles from the queue and runs model
      inference.  While inference runs, the fetch thread is already
      downloading the next tiles.

    Adjacent tiles overlap by *tile_overlap* pixels.  The producer
    caches EHO border strips so that each pixel is fetched from OMERO
    **exactly once**; the overlap data is reused from the cache.  Each
    tile's prediction is centre-cropped so the blended boundary region
    receives context from both neighbours, eliminating stitching
    artefacts.

    The bounded queue (sized by *prefetch_depth*) limits peak memory to
    roughly ``prefetch_depth * tile_size**2`` pixels plus the model.

    Parameters
    ----------
    host, port, username, password : str / int
        OMERO connection details.
    image_id : int
        OMERO image identifier.
    model_path : str or None
        Path to trained ``.pth`` weights, or ``None`` for bundled weights.
    model_version : str or None
        Version tag for bundled weights (ignored when *model_path* is set).
    output_path : str
        Where to save the prediction image (format from extension).
    target_mpp : float
        Desired microns-per-pixel.
    group_id : int or None
        OMERO group to switch to.
    save_eho : str or None
        Optional path to save the full EHO image.  **Note**: this requires
        allocating the full EHO canvas in memory.
    no_tissue_crop : bool
        Process every tile regardless of the tissue mask.
    device : str
        ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.
    inference_tile_size : int
        Side length (in output pixels) of each processing tile.  Should be
        a multiple of 64.  Default 2048.
    tile_overlap : int
        Overlap in output pixels between adjacent OMERO tiles.  Each pixel
        in the overlap zone is fetched only once (the producer caches EHO
        border strips).  Centre-cropping removes the seam.  Set to 0 to
        disable.  Default 128.
    sw_overlap : float
        Fractional overlap for the MONAI sliding-window inference *within*
        each tile (0-1).  Higher values improve patch-boundary quality at
        the cost of more forward passes.  Default 0.25.
    num_bg_tiles : int
        Number of background tiles to fetch for *Io* estimation.
    min_thumb_size : int
        Minimum longest edge for the thumbnail level.
    min_tissue_frac : float
        Minimum fraction of tissue pixels for a tile to be processed.
    prefetch_depth : int
        How many tiles the fetch thread may queue ahead of inference.
        Higher values use more memory but tolerate more latency jitter.
    num_fetch_workers : int
        Number of parallel OMERO connections used to fetch tiles
        concurrently.  Each worker opens its own session.  Higher
        values overlap more network I/O but consume more server
        sessions.  Default 4.

    Returns
    -------
    np.ndarray
        ``(H, W)`` uint8 label map.
    """
    try:
        logger.debug(
            f"DIAG: infer_omero_wsi start host={host!r} port={port} image_id={image_id} "
            f"inference_tile_size={inference_tile_size} num_fetch_workers={num_fetch_workers}"
        )
    except Exception:
        pass

    _require_omero()
    try:
        logger.debug("DIAG: _require_omero() returned")
    except Exception:
        pass
    # Sanity-check: overlap must leave room for a meaningful core.
    if tile_overlap < 0:
        tile_overlap = 0
    if tile_overlap >= inference_tile_size // 2:
        logger.warning(
            "tile_overlap=%d is >= half the tile size (%d); clamping to %d",
            tile_overlap,
            inference_tile_size,
            inference_tile_size // 4,
        )
        tile_overlap = inference_tile_size // 4

    # Ensure PyTorch and the model are initialised in the main thread BEFORE
    # creating OMERO connections / starting any worker threads. This avoids
    # macOS-level races between native runtimes (torch / libvips / Ice).
    try:
        import torch  # type: ignore

        from shell.model import build_model
    except Exception:
        # If torch or model scaffolding is not available we'll let the
        # existing _require_omero() / create_omero_connection surface an
        # appropriate error later. Continue to attempt OMERO connection.
        torch = None  # type: ignore
        build_model = None  # type: ignore

    if build_model is not None:
        if device == "auto":
            try:
                if torch is not None and torch.cuda.is_available():
                    device = "cuda"
                elif torch is not None and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            except Exception:
                device = "cpu"
        logger.info("Loading model (device=%s) …", device)
        model = build_model(model_path, device, model_version=model_version)
    else:
        model = None  # type: ignore

    conn = create_omero_connection(host, port, username, password)

    try:
        if group_id is not None:
            conn.SERVICE_OPTS.setOmeroGroup(group_id)

        # ------------------------------------------------------------------
        # 1. Image metadata
        # ------------------------------------------------------------------
        image = conn.getObject("Image", image_id)
        if image is None:
            msg = f"Image {image_id} not found in OMERO"
            raise ValueError(msg)

        size_x = image.getSizeX()
        size_y = image.getSizeY()
        size_c = image.getSizeC()
        pixels_id = image.getPixelsId()
        mpp_x, mpp_y = _get_physical_sizes_um(image)

        pix_obj = image.getPrimaryPixels()
        dtype = PIXEL_TYPES.get(pix_obj.getPixelsType().value, np.uint8)

        if size_c >= 3:
            channel_ids = [0, 1, 2]
        elif size_c == 1:
            channel_ids = [0, 0, 0]
        else:
            msg = f"Unsupported OMERO channel count: {size_c}"
            raise ValueError(msg)

        # Pyramid metadata
        probe = conn.c.sf.createRawPixelsStore()  # type: ignore[union-attr]
        try:
            probe.setPixelsId(pixels_id, False)
            res_descs = probe.getResolutionDescriptions()
            num_levels = probe.getResolutionLevels()
        finally:
            try:
                probe.close()
            except Exception:
                pass

        # Best level for target MPP
        best_desc_idx = 0
        best_lsz_x, best_lsz_y = size_x, size_y
        best_lmpp_x, best_lmpp_y = mpp_x, mpp_y

        for di, desc in enumerate(res_descs):
            lx, ly = desc.sizeX, desc.sizeY
            lmx = mpp_x * size_x / lx
            lmy = mpp_y * size_y / ly
            if lmx <= target_mpp and lmx > best_lmpp_x:
                best_desc_idx = di
                best_lsz_x, best_lsz_y = lx, ly
                best_lmpp_x, best_lmpp_y = lmx, lmy

        best_level = num_levels - 1 - best_desc_idx

        sx = best_lmpp_x / target_mpp
        sy = best_lmpp_y / target_mpp
        out_w = max(1, round(best_lsz_x * sx))
        out_h = max(1, round(best_lsz_y * sy))

        # Native tile size at best level
        probe2 = conn.c.sf.createRawPixelsStore()  # type: ignore[union-attr]
        try:
            probe2.setPixelsId(pixels_id, False)
            probe2.setResolutionLevel(best_level)
            tile_w, tile_h = probe2.getTileSize()
        finally:
            try:
                probe2.close()
            except Exception:
                pass

        logger.info(
            "Image %d: level %d (%dx%d), output %dx%d, sx=%.3f sy=%.3f",
            image_id,
            best_level,
            best_lsz_x,
            best_lsz_y,
            out_w,
            out_h,
            sx,
            sy,
        )

        # ------------------------------------------------------------------
        # 2. Fetch thumbnail and build tissue mask
        # ------------------------------------------------------------------
        logger.info("Fetching thumbnail for tissue mask …")
        thumb_rgb, _thumb_level, thumb_lsz_x, thumb_lsz_y = _fetch_thumbnail_rgb(
            conn,
            pixels_id,
            channel_ids,
            res_descs,
            num_levels,
            dtype,
            min_thumb_size=min_thumb_size,
        )
        logger.info(
            "Thumbnail: %dx%d (level %d)",
            thumb_lsz_x,
            thumb_lsz_y,
            _thumb_level,
        )

        tissue_mask = _build_tissue_mask(thumb_rgb)
        bg_mask_thumb = ~tissue_mask
        tissue_frac = float(np.mean(tissue_mask))
        logger.info("Tissue fraction in thumbnail: %.1f%%", tissue_frac * 100)

        # ------------------------------------------------------------------
        # 3. Fetch a few background tiles at full resolution for Io
        # ------------------------------------------------------------------
        from shell.transforms import compute_background_intensity

        bg_tile_coords = _pick_background_tile_coords(
            tissue_mask,
            thumb_lsz_y,
            thumb_lsz_x,
            out_h,
            out_w,
            inference_tile_size,
            num_bg_tiles,
        )

        if bg_tile_coords:
            logger.info(
                "Fetching %d background tile(s) for Io estimation …",
                len(bg_tile_coords),
            )
            bg_pixels: list[np.ndarray] = []
            for oy0_bg, ox0_bg, oh_bg, ow_bg in bg_tile_coords:
                lx0_bg = round(ox0_bg / sx)
                ly0_bg = round(oy0_bg / sy)
                lx1_bg = min(best_lsz_x, round((ox0_bg + ow_bg) / sx))
                ly1_bg = min(best_lsz_y, round((oy0_bg + oh_bg) / sy))
                tile_rgb = _fetch_region_rgb(
                    conn,
                    pixels_id,
                    best_level,
                    channel_ids,
                    best_lsz_x,
                    best_lsz_y,
                    tile_w,
                    tile_h,
                    dtype,
                    lx0=lx0_bg,
                    ly0=ly0_bg,
                    lx1=lx1_bg,
                    ly1=ly1_bg,
                    out_w=ow_bg,
                    out_h=oh_bg,
                )
                bg_pixels.append(tile_rgb)
            bg_stack = np.concatenate(
                [t.reshape(-1, 3) for t in bg_pixels],
                axis=0,
            )
            bg_mask_for_io = np.ones(bg_stack.shape[0], dtype=bool)
            bg_2d = bg_stack.reshape(-1, 1, 3)
            bg_mask_2d = bg_mask_for_io.reshape(-1, 1)
            io_val = compute_background_intensity(bg_2d, bg_mask_2d)
            logger.info("Estimated Io from background tiles: %.1f", io_val)
            del bg_pixels, bg_stack, bg_mask_for_io, bg_2d, bg_mask_2d
        else:
            logger.warning(
                "No pure background tiles found; estimating Io from thumbnail.",
            )
            io_val = compute_background_intensity(thumb_rgb, bg_mask_thumb)

        # ------------------------------------------------------------------
        # 4. Estimate stain parameters from thumbnail + full-res Io
        # ------------------------------------------------------------------
        logger.info("Estimating stain parameters from thumbnail …")
        stain_params = _estimate_stain_params_with_io(
            thumb_rgb,
            bg_mask_thumb,
            io_val,
        )
        del thumb_rgb, bg_mask_thumb
        gc.collect()

        logger.info(
            "Stain params: e_idx=%d h_idx=%d Io=%.1f",
            stain_params["e_idx"],
            stain_params["h_idx"],
            stain_params["Io"],
        )

        # ------------------------------------------------------------------
        # 5. Build tile schedule (overlap-aware)
        # ------------------------------------------------------------------
        tile_schedule, num_rows, num_cols, tiles_total = _build_tile_grid(
            out_h=out_h,
            out_w=out_w,
            inference_tile_size=inference_tile_size,
            tile_overlap=tile_overlap,
            tissue_mask=tissue_mask,
            thumb_h=thumb_lsz_y,
            thumb_w=thumb_lsz_x,
            min_tissue_frac=min_tissue_frac,
            no_tissue_crop=no_tissue_crop,
        )
        tiles_tissue = len(tile_schedule)

        del tissue_mask
        gc.collect()

        logger.info(
            "Tile grid: %d total, %d tissue (%.1f%% skipped)",
            tiles_total,
            tiles_tissue,
            (1 - tiles_tissue / max(tiles_total, 1)) * 100,
        )

        if tiles_tissue == 0:
            logger.warning("No tissue tiles found — writing empty prediction.")
            try:
                conn.close()
            except Exception:
                pass
            pred = np.zeros((out_h, out_w), dtype=np.uint8)
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            import pyvips

            pyvips.Image.new_from_array(pred).write_to_file(output_path)
            logger.info("Saved prediction to %s", output_path)
            return pred

        # ------------------------------------------------------------------
        # 6. Pipelined inference — prefetch tiles while inferring
        # ------------------------------------------------------------------
        return _run_pipeline(
            conn=conn,
            host=host,
            port=port,
            username=username,
            password=password,
            tile_schedule=tile_schedule,
            num_rows=num_rows,
            num_cols=num_cols,
            tiles_tissue=tiles_tissue,
            pixels_id=pixels_id,
            best_level=best_level,
            channel_ids=channel_ids,
            best_lsz_x=best_lsz_x,
            best_lsz_y=best_lsz_y,
            tile_w=tile_w,
            tile_h=tile_h,
            dtype=dtype,
            sx=sx,
            sy=sy,
            stain_params=stain_params,
            out_h=out_h,
            out_w=out_w,
            prefetch_depth=prefetch_depth,
            model=model,
            device=device,
            save_eho=save_eho,
            output_path=output_path,
            tile_overlap=tile_overlap,
            roi_size=inference_tile_size,
            sw_overlap=sw_overlap,
            num_fetch_workers=num_fetch_workers,
        )

    except Exception:
        try:
            _safe_close_blitzgateway(conn)
        except Exception:
            pass
        raise


# ---------------------------------------------------------------------------
# Pipeline implementation
# ---------------------------------------------------------------------------


def _run_pipeline(
    *,
    conn: BlitzGateway,
    host: str,
    port: int,
    username: str,
    password: str,
    tile_schedule: list[tuple[int, int, int, int, int, int, int, int]],
    num_rows: int,
    num_cols: int,
    tiles_tissue: int,
    pixels_id: int,
    best_level: int,
    channel_ids: list[int],
    best_lsz_x: int,
    best_lsz_y: int,
    tile_w: int,
    tile_h: int,
    dtype: Any,
    sx: float,
    sy: float,
    stain_params: dict[str, Any],
    out_h: int,
    out_w: int,
    prefetch_depth: int,
    model: Any,
    device: str,
    save_eho: str | None,
    output_path: str,
    tile_overlap: int,
    roi_size: int,
    sw_overlap: float,
    num_fetch_workers: int = 4,
) -> np.ndarray:
    """Producer/consumer pipeline: parallel fetch+EHO → inference in main.

    *num_fetch_workers* threads each open their own OMERO connection and
    pull tiles from a shared work queue.  The main thread consumes
    ready-to-infer EHO tiles and runs model inference.

    Using multiple connections overlaps network round-trip latency across
    tiles, and the bulk-fetch optimisation in
    :func:`_fetch_region_rgb_single_store` reduces per-tile round-trips
    from ``O(N_subtiles * 3)`` to just 3 (one per channel).

    Each queued item contains the tile EHO **and** the keep region that
    the consumer should write to the output canvas, implementing the
    centre-crop blending strategy that eliminates stitching artefacts.
    """
    tile_queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, prefetch_depth))

    # Precompute full keep-region info so workers are self-contained.
    tile_infos = _precompute_tile_info(
        tile_schedule,
        num_rows,
        num_cols,
        tile_overlap,
    )

    fetch_thread = threading.Thread(
        target=_parallel_fetch_coordinator,
        kwargs={
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "tile_infos": tile_infos,
            "result_queue": tile_queue,
            "pixels_id": pixels_id,
            "best_level": best_level,
            "channel_ids": channel_ids,
            "best_lsz_x": best_lsz_x,
            "best_lsz_y": best_lsz_y,
            "tile_w": tile_w,
            "tile_h": tile_h,
            "dtype": dtype,
            "sx": sx,
            "sy": sy,
            "stain_params": stain_params,
            "num_workers": num_fetch_workers,
        },
        name="omero-fetch-coordinator",
        daemon=True,
    )
    try:
        # torch is imported here for device checks used by run_inference.
        # The model itself is provided by the caller (loaded earlier in the
        # main infer_omero_wsi function) and must be passed via the `model`
        # parameter. Starting the fetch thread now is safe because the model
        # has already been initialised on the main thread.
        import torch  # type: ignore

        from shell.inference import run_inference

        if device == "auto":
            try:
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            except Exception:
                device = "cpu"

        # Start the fetch thread immediately; the heavy model runtime was
        # already initialised by the caller before calling _run_pipeline.
        try:
            fetch_thread.start()
            logger.info("Started OMERO fetch thread.")
        except Exception:
            logger.exception("Failed to start OMERO fetch thread; continuing.")

        pred = np.zeros((out_h, out_w), dtype=np.uint8)

        eho_canvas: np.ndarray | None = None
        if save_eho:
            eho_canvas = np.full((out_h, out_w, 3), 255, dtype=np.uint8)

        processed = 0
        while True:
            item = tile_queue.get()

            if item is _SENTINEL:
                break
            if isinstance(item, BaseException):
                raise item

            oy0, ox0, oh, ow, keep_y0, keep_x0, keep_h, keep_w, tile_eho = item

            if eho_canvas is not None:
                eho_canvas[oy0 : oy0 + oh, ox0 : ox0 + ow] = tile_eho

            tile_pred = run_inference(
                tile_eho,
                model,
                device,
                roi_size=(roi_size, roi_size),
                overlap=sw_overlap,
            )
            del tile_eho

            # Write only the centre-cropped keep region to avoid seams.
            ph, pw = tile_pred.shape[:2]
            ky1 = min(keep_y0 + keep_h, ph)
            kx1 = min(keep_x0 + keep_w, pw)
            kept = tile_pred[keep_y0:ky1, keep_x0:kx1]
            out_y0 = oy0 + keep_y0
            out_x0 = ox0 + keep_x0
            ch, cw = kept.shape[:2]
            pred[out_y0 : out_y0 + ch, out_x0 : out_x0 + cw] = kept
            del tile_pred, kept

            processed += 1
            if processed % 25 == 0:
                gc.collect()
            if processed % 10 == 0 or processed == tiles_tissue:
                logger.info(
                    "Progress: %d / %d tiles (%.0f%%)",
                    processed,
                    tiles_tissue,
                    processed / tiles_tissue * 100,
                )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return _save_results(pred, eho_canvas, save_eho, output_path)

    finally:
        fetch_thread.join(timeout=30)
        if fetch_thread.is_alive():
            logger.warning("Fetch thread did not exit within 30 s; continuing cleanup.")
        try:
            _safe_close_blitzgateway(conn)
        except Exception:
            pass
