#!/usr/bin/env python3
"""Profile the SHELL WSI pipeline with cProfile to find preprocessing hotspots.

Runs :func:`shell.infer_wsi.infer_wsi` under ``cProfile`` and writes a
``.prof`` stats file plus a printed top-N breakdown by cumulative and
total (self) time. The existing phase timings (open/thumbnail/tissue_mask/
stain_params/tiled_inference/...) are still printed by ``infer_wsi`` itself;
this script adds *function-level* detail to explain what inside each phase
is actually expensive.

Use ``--skip-model`` to replace the real model with a no-op that returns
zero logits of the correct shape. This isolates the cost of image I/O,
tissue masking, stain estimation, EHO conversion, and tiling/stitching
from the cost of running the segmentation network itself.

Examples
--------
Profile the full pipeline (real model, real device selection)::

    hatch run python scripts/profile_preprocessing.py LR10_N101_S08_HE.jp2

Profile preprocessing only (skip the model forward pass)::

    hatch run python scripts/profile_preprocessing.py LR10_N101_S08_HE.jp2 \\
        --skip-model

Then inspect the resulting ``.prof`` file with snakeviz::

    pip install snakeviz
    snakeviz profile_output/profile.prof
"""

from __future__ import annotations

import argparse
import cProfile
import logging
import pstats
import sys
from pathlib import Path


def _build_dummy_model(tile_size: tuple[int, int], num_classes: int, device: str):
    """Return a callable that mimics the model interface but skips compute."""
    import torch

    class _DummyModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, x: "torch.Tensor") -> "torch.Tensor":
            b = x.shape[0]
            return torch.zeros(
                (b, num_classes, *tile_size), dtype=torch.float32, device=x.device
            )

    return _DummyModel().to(device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="Path to the WSI to profile (e.g. LR10_N101_S08_HE.jp2)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="profile_output",
        help="Directory for the label map, .prof file, and logs. Default: profile_output/",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument("--mpp", type=float, default=None, help="Manual source um/px override.")
    parser.add_argument("--target-mpp", type=float, default=None, help="Target um/px. Defaults to shell's TARGET_MPP.")
    parser.add_argument(
        "--min-tissue-frac",
        type=float,
        default=None,
        help="Minimum tissue fraction for mask-aware inference tiles.",
    )
    parser.add_argument("--model-path", type=str, default=None, help="Explicit model weights path.")
    parser.add_argument("--model-version", type=str, default=None, help="Bundled model version tag.")
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Replace the model with a no-op to isolate preprocessing/tiling cost.",
    )
    parser.add_argument("--top", type=int, default=40, help="Number of rows to print per sort order.")
    parser.add_argument(
        "--sort",
        type=str,
        default="cumulative,tottime",
        help="Comma-separated pstats sort keys to report (each printed separately).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_out = out_dir / f"{input_path.stem}_labels.tif"
    prof_path = out_dir / "profile.prof"

    from shell.infer_wsi import infer_wsi
    from shell.model import NUM_CLASSES, TILE_SIZE

    kwargs: dict = {
        "device": args.device,
        "model_path": args.model_path,
        "model_version": args.model_version,
    }
    if args.mpp is not None:
        kwargs["mpp"] = args.mpp
    if args.target_mpp is not None:
        kwargs["target_mpp"] = args.target_mpp
    if args.min_tissue_frac is not None:
        kwargs["min_tissue_frac"] = args.min_tissue_frac

    if args.skip_model:
        # infer_wsi resolves "auto" internally, but the dummy model still
        # needs a concrete device string for tensor placement.
        device_for_dummy = args.device if args.device != "auto" else "cpu"
        kwargs["device"] = device_for_dummy
        kwargs["_model"] = _build_dummy_model(TILE_SIZE, NUM_CLASSES, device_for_dummy)
        print("Skipping real model forward pass (dummy model returns zeros).")

    print(f"Profiling infer_wsi on {input_path} ...")
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        infer_wsi(str(input_path), str(label_out), **kwargs)
    finally:
        profiler.disable()

    profiler.dump_stats(str(prof_path))
    print(f"\nRaw profile stats written to: {prof_path}")
    print("View interactively with: snakeviz " + str(prof_path))

    for sort_key in args.sort.split(","):
        sort_key = sort_key.strip()
        print(f"\n{'=' * 70}\nTop {args.top} by '{sort_key}'\n{'=' * 70}")
        pstats.Stats(profiler, stream=sys.stdout).sort_stats(sort_key).print_stats(args.top)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# run this command redirect stdout/err to a file:
# hatch run python scripts/profile_preprocessing.py LR10_N101_S08_HE.jp2 --skip-model 2>&1 | tee profile.log