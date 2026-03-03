# SPDX-FileCopyrightText: 2024-present barrettMCW <mjbarrett@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""
Command-line interface for SHELL.

Provides two subcommands:

* ``shell infer`` — inference on a local RGB whole-slide image
* ``shell infer-omero`` — inference on an OMERO image (requires ``shell[omero]``)
"""

from __future__ import annotations

import os

# Ensure PyTorch initialises its runtime and threads before other native
# libraries (pyvips, Ice/OMERO) to avoid macOS-level races that can lead
# to segmentation faults when different C extensions compete for thread
# runtime ownership.  Set conservative thread counts before importing torch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

try:
    # Import torch early so it initialises its native runtime in the main
    # thread. This is intentional — we only probe __version__ to force
    # initialisation without performing heavy work here.
    import torch  # type: ignore

    _ = torch.__version__  # ensure attribute access triggers initialisation
    TORCH_AVAILABLE = True
except Exception:
    # If torch is unavailable we still continue — the program will emit
    # a clearer error later when trying to build/load the model.
    TORCH_AVAILABLE = False

import argparse
import getpass
import logging
import sys

from shell.__about__ import __version__

# Module logger — used for diagnostic messages instead of printing directly.
log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    # Global flags (before subcommands)
    parser = argparse.ArgumentParser(
        prog="shell",
        description="SHELL — SHELL Highlights Epithelium and Lumen Locations",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="store_true",
        dest="show_version",
        help="Print version and exit.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-v for INFO, -vv for DEBUG).",
    )

    subparsers = parser.add_subparsers(dest="command")

    # ── infer (local file) ────────────────────────────────────────────
    infer_p = subparsers.add_parser(
        "infer",
        help="Run inference on a local RGB whole-slide image.",
    )
    infer_p.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to the raw RGB image (TIFF, PNG, JPEG, etc.).",
    )
    infer_p.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to save the predicted label image.",
    )
    infer_p.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model weights (default: bundled latest).",
    )
    infer_p.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Bundled model version to use (e.g. v1). Ignored if --model-path is set.",
    )
    infer_p.add_argument(
        "--target-mpp",
        type=float,
        default=1.0,
        help="Desired output resolution (um/pixel).",
    )
    infer_p.add_argument(
        "--mpp",
        type=float,
        default=None,
        help=(
            "Source image resolution in um/pixel. Required for formats "
            "that lack embedded resolution metadata (e.g. PNG, JPEG). "
            "When omitted the value is read from slide metadata; if "
            "metadata is unavailable, no resolution scaling is applied."
        ),
    )
    infer_p.add_argument(
        "--save-eho",
        type=str,
        default=None,
        help="Optional: also save the intermediate EHO image.",
    )
    infer_p.add_argument(
        "--save-raw",
        type=str,
        default=None,
        help=(
            "Optional: save raw model predictions as a 2-band uint8 image "
            "(band 0 = inner / lumen, band 1 = outer / epithelium). "
            "Useful for debugging or re-running post-processing offline."
        ),
    )
    infer_p.add_argument(
        "--profile",
        type=str,
        default="best_effort",
        choices=["best_effort", "precise", "sensitive"],
        help=(
            "Post-processing filter profile. "
            "'best_effort' (default) balances sensitivity and precision; "
            "'precise' is more conservative; 'sensitive' keeps more predictions."
        ),
    )
    infer_p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, mps).",
    )

    # ── infer-omero ───────────────────────────────────────────────────
    omero_p = subparsers.add_parser(
        "infer-omero",
        help="Run inference on an OMERO image (requires shell[omero]).",
    )
    omero_p.add_argument("--host", type=str, required=True, help="OMERO host")
    omero_p.add_argument("--port", type=int, default=4064, help="OMERO port")
    omero_p.add_argument("--username", type=str, required=True, help="OMERO username")
    omero_p.add_argument(
        "--password",
        type=str,
        default=None,
        help="OMERO password (omit to prompt)",
    )
    omero_p.add_argument("--group-id", type=int, default=None, help="OMERO group id")
    omero_p.add_argument(
        "--image-id",
        type=int,
        required=True,
        help="OMERO image id",
    )
    omero_p.add_argument(
        "--target-mpp",
        type=float,
        default=1.0,
        help="Desired output resolution (um/pixel)",
    )
    omero_p.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model weights (default: bundled latest)",
    )
    omero_p.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Bundled model version to use (e.g. v1). Ignored if --model-path is set.",
    )
    omero_p.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output path for predicted label image",
    )
    omero_p.add_argument("--save-eho", type=str, default=None)
    omero_p.add_argument("--no-tissue-crop", action="store_true")
    omero_p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, mps).",
    )
    omero_p.add_argument(
        "--inference-tile-size",
        type=int,
        default=2048,
        help="Side length (output pixels) of each processing tile. Default 2048.",
    )
    omero_p.add_argument(
        "--tile-overlap",
        type=int,
        default=128,
        help=(
            "Overlap in output pixels between adjacent OMERO tiles. "
            "Each pixel in the overlap zone is fetched only once (the "
            "producer caches EHO border strips). Centre-cropping removes "
            "the seam. Set to 0 to disable. Default 128."
        ),
    )
    omero_p.add_argument(
        "--sw-overlap",
        type=float,
        default=0.25,
        help=(
            "Fractional overlap for the MONAI sliding-window inference "
            "within each tile (0-1). Higher values improve patch-boundary "
            "quality at the cost of more forward passes. Default 0.25."
        ),
    )
    omero_p.add_argument(
        "--num-bg-tiles",
        type=int,
        default=4,
        help="Number of background tiles to fetch for Io estimation. Default 4.",
    )
    omero_p.add_argument(
        "--min-thumb-size",
        type=int,
        default=2000,
        help="Minimum longest edge for the thumbnail level. Default 2000.",
    )
    omero_p.add_argument(
        "--min-tissue-frac",
        type=float,
        default=0.01,
        help="Minimum tissue fraction for a tile to be processed. Default 0.01.",
    )
    omero_p.add_argument(
        "--prefetch-depth",
        type=int,
        default=8,
        help=(
            "How many tiles the fetch thread may queue ahead of inference. "
            "Higher values use more memory but tolerate more latency jitter. "
            "Default 8."
        ),
    )
    omero_p.add_argument(
        "--fetch-workers",
        type=int,
        default=4,
        help=(
            "Number of parallel OMERO connections used to fetch tiles "
            "concurrently. Each worker opens its own session. Higher "
            "values overlap more network I/O but consume more server "
            "sessions. Default 4."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI.

    :param argv: argument list to parse (defaults to ``sys.argv[1:]``).
    :return: exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging based on --verbose / -v
    verbosity = getattr(args, "verbose", 0)
    if verbosity >= 2:
        log_level = logging.DEBUG
    elif verbosity >= 1:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Handle --version as a non-exiting flag so tests calling main(["--version"])
    # don't trigger argparse's SystemExit. Print the package version and exit 0.
    if getattr(args, "show_version", False):
        print(__version__)
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "infer":
        from shell.infer_wsi import infer_wsi

        print(f"Input:  {args.input}")
        print(f"Output: {args.output}")

        label_map = infer_wsi(
            input_path=args.input,
            output_path=args.output,
            model_path=args.model_path,
            model_version=args.model_version,
            target_mpp=args.target_mpp,
            mpp=args.mpp,
            save_eho=args.save_eho,
            save_raw=args.save_raw,
            profile=args.profile,
            device=args.device,
        )

        import numpy as np

        from shell.model import CLASS_NAMES

        print(f"Label values: {np.unique(label_map)} — {CLASS_NAMES}")
        return 0

    if args.command == "infer-omero":
        # Diagnostic prints added to help narrow where a macOS segfault occurs.
        # These prints are intentionally lightweight and flushed so they appear
        # even if the process crashes in a C extension later on.
        log.debug("DIAG: infer-omero selected")
        log.debug("DIAG: about to import infer_omero_wsi")
        from shell.infer_omero_wsi import infer_omero_wsi

        log.debug("DIAG: imported infer_omero_wsi")

        if args.password is None:
            args.password = getpass.getpass("OMERO password: ")

        print(f"Connecting to {args.host}:{args.port} …", flush=True)
        log.debug("DIAG: about to call infer_omero_wsi")
        try:
            infer_omero_wsi(
                host=args.host,
                port=args.port,
                username=args.username,
                password=args.password,
                image_id=args.image_id,
                model_path=args.model_path,
                model_version=args.model_version,
                output_path=args.output,
                target_mpp=args.target_mpp,
                group_id=args.group_id,
                save_eho=args.save_eho,
                no_tissue_crop=args.no_tissue_crop,
                device=args.device,
                inference_tile_size=args.inference_tile_size,
                tile_overlap=args.tile_overlap,
                sw_overlap=args.sw_overlap,
                num_bg_tiles=args.num_bg_tiles,
                min_thumb_size=args.min_thumb_size,
                min_tissue_frac=args.min_tissue_frac,
                prefetch_depth=args.prefetch_depth,
                num_fetch_workers=args.fetch_workers,
            )
            log.debug("DIAG: infer_omero_wsi returned normally")
        except BaseException as e:
            # Catch and print Python-level exceptions; note segfaults (native crashes)
            # will not be caught here, but these diagnostics will show progress up to
            # the crash point.
            import sys
            import traceback

            log.debug("DIAG: infer_omero_wsi raised an exception:")
            traceback.print_exc()
            # Re-raise so existing behaviour (non-zero exit) remains.
            raise
        log.debug("DIAG: after infer_omero_wsi (should have saved output)")
        print(f"Saved prediction to {args.output}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
