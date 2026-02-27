# SPDX-FileCopyrightText: 2024-present barrettMCW <mjbarrett@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""
Model construction and helpers for SegResNetVAE inference.

The VAE branch is only used during training; at eval time ``forward``
returns logits directly.

Model weights are bundled inside the package under ``weights/``.  The
:data:`MODEL_REGISTRY` maps human-readable version tags to filenames,
and :data:`LATEST_MODEL` always points at the current recommended
version.  To upgrade the default model:

1. Drop the new ``.pth`` file into ``src/shell/weights/``.
2. Add an entry to :data:`MODEL_REGISTRY`.
3. Update :data:`LATEST_MODEL`.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import PIL.Image
import torch
from monai.networks.nets import SegResNetVAE

PIL.Image.MAX_IMAGE_PIXELS = None  # disable DecompressionBombError for large WSIs

# ---------------------------------------------------------------------------
# Default hyper-parameters (keep in sync with train.ipynb)
# ---------------------------------------------------------------------------
NUM_CLASSES: int = 3
TILE_SIZE: tuple[int, int] = (320, 320)
CLASS_NAMES: dict[int, str] = {0: "Background", 1: "Epithelium", 2: "Stroma"}

# ---------------------------------------------------------------------------
# Versioned model registry
# ---------------------------------------------------------------------------
#: Maps version tags to weight filenames inside ``src/shell/weights/``.
#: To add a new model, place the ``.pth`` file in ``weights/`` and add an
#: entry here.
MODEL_REGISTRY: dict[str, str] = {
    "v1": "model_v1.pth",
}

#: The version tag used when no explicit model path is provided.
LATEST_MODEL: str = "v1"


def _resolve_bundled_weights(version: str | None = None) -> Path:
    """Return the filesystem path to a bundled model weight file.

    :param version: A key in :data:`MODEL_REGISTRY`.  ``None`` means
        :data:`LATEST_MODEL`.
    :return: resolved :class:`~pathlib.Path` to the ``.pth`` file.
    :raises KeyError: If *version* is not in :data:`MODEL_REGISTRY`.
    :raises FileNotFoundError: If the weight file is missing from the
        installed package.
    """
    if version is None:
        version = LATEST_MODEL

    if version not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        msg = f"Unknown model version {version!r}. Available versions: {available}"
        raise KeyError(msg)

    filename = MODEL_REGISTRY[version]

    # importlib.resources works with both editable installs and proper
    # wheels.  ``files()`` returns a Traversable; ``joinpath`` reaches
    # into the sub-package.
    weights_pkg = importlib.resources.files("shell") / "weights" / filename
    # Materialise to a real filesystem path (may extract from a zip).
    with importlib.resources.as_file(weights_pkg) as p:
        resolved = Path(p)
    if not resolved.exists():
        msg = f"Bundled weight file not found: {resolved}"
        raise FileNotFoundError(msg)
    return resolved


# ---------------------------------------------------------------------------
# Patch SegResNetVAE.forward so eval skips the VAE loss branch
# ---------------------------------------------------------------------------
def _patch_segresntvae_forward() -> None:
    """Monkey-patch ``SegResNetVAE.forward`` to skip VAE loss at eval time."""

    def _forward(self, x):
        x_enc, down_x = self.encode(x)
        down_x.reverse()
        x_dec = self.decode(x_enc, down_x)
        if self.training:
            vae_loss = self._get_vae_loss(x, x_enc)
            return x_dec, vae_loss
        return x_dec

    SegResNetVAE.forward = _forward


_patch_segresntvae_forward()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_model(
    model_path: str | None = None,
    device: torch.device | str = "cpu",
    *,
    model_version: str | None = None,
    num_classes: int = NUM_CLASSES,
    tile_size: tuple[int, int] = TILE_SIZE,
) -> SegResNetVAE:
    """Load a trained SegResNetVAE onto *device*.

    The model weights can be specified in three ways (highest priority
    first):

    1. **model_path** — an explicit filesystem path to a ``.pth`` file.
    2. **model_version** — a key in :data:`MODEL_REGISTRY` (e.g.
       ``"v1"``).  The corresponding bundled weight file is used.
    3. If neither is given, the bundled weights for
       :data:`LATEST_MODEL` are loaded automatically.

    :param model_path: filesystem path to a ``.pth`` state dict, or
        ``None`` to use bundled weights.
    :param device: target device (``"cpu"``, ``"cuda"``, …).
    :param model_version: version tag for bundled weights (ignored when
        *model_path* is set).
    :param num_classes: number of output classes.
    :param tile_size: spatial size used during training.
    :return: the model in eval mode.
    """
    if isinstance(device, str):
        device = torch.device(device)

    if model_path is None:
        resolved = _resolve_bundled_weights(model_version)
        model_path = str(resolved)

    model = SegResNetVAE(
        spatial_dims=2,
        init_filters=16,
        in_channels=3,
        out_channels=num_classes,
        dropout_prob=0.1,
        norm=("GROUP", {"num_groups": 4}),
        act=("RELU", {"inplace": True}),
        input_image_size=tile_size,
        vae_nz=256,
        vae_estimate_std=True,
    ).to(device)

    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
