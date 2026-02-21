# SPDX-FileCopyrightText: 2024-present barrettMCW <mjbarrett@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""
Command-line interface for SHELL.

Provides two subcommands:

* ``shell infer`` — inference on a local RGB .tiff WSI
* ``shell infer-omero`` — inference on an OMERO image (requires ``shell[omero]``)
"""

from __future__ import annotations

import argparse
import getpass
import sys

from shell.__about__ import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
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

    subparsers = parser.add_subparsers(dest="command")

    # ── infer (local file) ────────────────────────────────────────────
    infer_p = subparsers.add_parser(
        "infer",
        help="Run inference on a local RGB .tiff whole-slide image.",
    )
    infer_p.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to the raw RGB .tiff whole-slide image.",
    )
    infer_p.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to save the predicted label TIFF.",
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
        "--save-eho",
        type=str,
        default=None,
        help="Optional: also save the intermediate EHO image.",
    )
    infer_p.add_argument(
        "--stain-downsample",
        type=int,
        default=4,
        help="Downsample factor for stain parameter estimation.",
    )
    infer_p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda).",
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
    omero_p.add_argument("--max-dim", type=int, default=None)
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
        help="Output path for predicted label TIFF",
    )
    omero_p.add_argument("--save-eho", type=str, default=None)
    omero_p.add_argument("--no-tissue-crop", action="store_true")
    omero_p.add_argument("--stain-downsample", type=int, default=4)
    omero_p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI.

    :param argv: argument list to parse (defaults to ``sys.argv[1:]``).
    :return: exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

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
            save_eho=args.save_eho,
            stain_downsample=args.stain_downsample,
            device=args.device,
        )

        import numpy as np

        from shell.model import CLASS_NAMES

        print(f"Label values: {np.unique(label_map)} — {CLASS_NAMES}")
        return 0

    if args.command == "infer-omero":
        from shell.infer_omero_wsi import infer_omero_wsi

        if args.password is None:
            args.password = getpass.getpass("OMERO password: ")

        print(f"Connecting to {args.host}:{args.port} …")
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
            max_dim=args.max_dim,
            group_id=args.group_id,
            save_eho=args.save_eho,
            stain_downsample=args.stain_downsample,
            no_tissue_crop=args.no_tissue_crop,
            device=args.device,
        )
        print(f"Saved prediction to {args.output}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
