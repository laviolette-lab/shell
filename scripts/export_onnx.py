#!/usr/bin/env python3
"""Export the bundled SegResNetVAE checkpoint to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from shell.model import LATEST_MODEL, MODEL_INPUT_SIZE, _resolve_bundled_weights


def _build_model_for_export(device: torch.device) -> torch.nn.Module:
    """Create the reference SegResNetVAE model in eval mode."""
    from monai.networks.nets import SegResNetVAE

    model = SegResNetVAE(
        spatial_dims=2,
        init_filters=16,
        in_channels=3,
        out_channels=3,
        dropout_prob=0.2,
        norm=("GROUP", {"num_groups": 8}),
        act=("MISH", {"inplace": True}),
        input_image_size=MODEL_INPUT_SIZE,
        vae_nz=256,
        vae_estimate_std=True,
    ).to(device)
    model.eval()
    return model


def export_onnx(
    checkpoint: str | None = None,
    *,
    version: str | None = None,
    output: str | None = None,
    device: str = "cpu",
) -> Path:
    """Export a bundled checkpoint to ONNX and return the output path."""
    if checkpoint is None:
        if version is None:
            version = LATEST_MODEL
        checkpoint = str(_resolve_bundled_weights(version))

    dev = torch.device(device)
    model = _build_model_for_export(dev)
    state = torch.load(checkpoint, map_location=dev, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    if output is None:
        target = Path(checkpoint).with_suffix(".onnx")
    else:
        target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, 320, 320, device=dev)
    torch.onnx.export(
        model,
        dummy,
        str(target),
        export_params=True,
        # PyTorch 2.14 emits Mish at opset 18; opset 17 cannot downgrade it.
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "logits": {0: "batch_size", 2: "height", 3: "width"},
        },
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the latest SHELL checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a .pth checkpoint to export.")
    parser.add_argument("--version", type=str, default=None, help="Bundled model version to export (e.g. v1).")
    parser.add_argument("--output", type=str, default=None, help="Destination .onnx path.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for export (cpu/cuda/mps).")
    args = parser.parse_args()

    out = export_onnx(args.checkpoint, version=args.version, output=args.output, device=args.device)
    print(f"Exported ONNX model: {out}")


if __name__ == "__main__":
    main()
