# SPDX-FileCopyrightText: 2024-present barrettMCW <mjbarrett@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""
Model construction and helpers for SegResNetVAE inference.

The VAE branch is only used during training; at eval time ``forward``
returns logits directly.

ONNX models are bundled inside the package under ``weights/``. Source PTH
checkpoints remain repository-only export inputs.
"""

from __future__ import annotations

import importlib.resources
import os
from pathlib import Path

import torch


def _detect_cpu_budget() -> int:
    """Return a safe thread count, honoring cgroup CPU quotas.

    ``os.cpu_count()`` reports the host's total cores even inside a
    cgroup-limited container (e.g. 96 cores reported with an 8-core quota).
    Sizing thread pools (ONNX Runtime, PyTorch) from the unclamped count
    causes severe oversubscription that can look like a hang.
    """
    quota_cores: int | None = None
    try:
        cgroup_max = Path("/sys/fs/cgroup/cpu.max")
        if cgroup_max.exists():
            quota, period = cgroup_max.read_text().split()
            if quota != "max":
                quota_cores = max(1, int(quota) // int(period))
        else:
            quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
            period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
            if quota_path.exists() and period_path.exists():
                quota = int(quota_path.read_text())
                period = int(period_path.read_text())
                if quota > 0:
                    quota_cores = max(1, quota // period)
    except OSError:
        quota_cores = None

    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1

    if quota_cores is not None:
        available = min(available, quota_cores)

    return max(1, available)

# ---------------------------------------------------------------------------
# Default hyper-parameters (keep in sync with train.ipynb)
# ---------------------------------------------------------------------------
NUM_CLASSES: int = 3
TILE_SIZE: tuple[int, int] = (2048, 2048)
MODEL_INPUT_SIZE: tuple[int, int] = (320, 320)

#: Final label map produced by :mod:`shell.post_process`.
CLASS_NAMES: dict[int, str] = {
    0: "Background",
    1: "Inner (lumen)",
    2: "Outer (epithelium)",
    3: "White / glass",
    4: "Background tissue (stroma)",
    5: "Epithelial nuclei",
    6: "Other nuclei",
    7: "Urethra",
    8: "Edge epithelium (WSI only)",
}

# ---------------------------------------------------------------------------
# Versioned model registry
# ---------------------------------------------------------------------------
#: Maps version tags to ONNX filenames inside ``src/shell/weights/``.
MODEL_REGISTRY: dict[str, str] = {
    "v1": "model_v1.onnx",
}

#: The version tag used when no explicit model path is provided.
LATEST_MODEL: str = "v1"


def _resolve_bundled_weight_path(version: str | None = None) -> Path:
    """Resolve a bundled ONNX artifact."""
    if version is None:
        version = LATEST_MODEL

    if version not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        msg = f"Unknown model version {version!r}. Available versions: {available}"
        raise KeyError(msg)

    filename = MODEL_REGISTRY[version]
    weights_pkg = importlib.resources.files("shell") / "weights" / filename
    with importlib.resources.as_file(weights_pkg) as p:
        resolved = Path(p)
    if resolved.exists():
        return resolved

    msg = f"Bundled ONNX model not found for version {version!r}: {filename}"
    raise FileNotFoundError(msg)


def _resolve_bundled_weights(version: str | None = None) -> Path:
    """Return the filesystem path to a bundled model weight file.

    :param version: A key in :data:`MODEL_REGISTRY`.  ``None`` means
        :data:`LATEST_MODEL`.
    :return: resolved :class:`~pathlib.Path` to the ``.onnx`` file.
    :raises KeyError: If *version* is not in :data:`MODEL_REGISTRY`.
    :raises FileNotFoundError: If the weight file is missing from the
        installed package.
    """
    return _resolve_bundled_weight_path(version)


class ONNXModelWrapper:
    """Small wrapper that makes an ONNX Runtime session behave like a torch Module."""

    def __init__(self, session, device: str = "cpu") -> None:
        self.session = session
        self.device = torch.device(device)
        self._input_name = session.get_inputs()[0].name
        self._output_name = session.get_outputs()[0].name

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        return self

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        expected_shape = self.session.get_inputs()[0].shape
        actual_shape = tuple(x.shape)
        for axis, (actual, expected) in enumerate(
            zip(actual_shape, expected_shape, strict=False)
        ):
            if isinstance(expected, int) and actual != expected:
                raise ValueError(
                    f"ONNX model expects dimension {expected} at axis {axis}, "
                    f"but received {actual}."
                )
        if len(actual_shape) == 4 and (
            actual_shape[-2] % 64 != 0 or actual_shape[-1] % 64 != 0
        ):
            raise ValueError(
                "ONNX model requires spatial dimensions divisible by 64; "
                f"received {actual_shape[-2]}x{actual_shape[-1]}."
            )
        if x.device.type != "cpu":
            x = x.cpu()
        inputs = {self._input_name: x.detach().cpu().numpy()}
        outputs = self.session.run([self._output_name], inputs)[0]
        return torch.from_numpy(outputs).to(self.device)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_model(
    model_path: str | None = None,
    device: torch.device | str = "cpu",
    *,
    model_version: str | None = None,
    num_classes: int = NUM_CLASSES,
    tile_size: tuple[int, int] = MODEL_INPUT_SIZE,
) -> torch.nn.Module:
    """Load the bundled ONNX model onto *device*.

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
    :param tile_size: retained for API compatibility; ONNX controls its shape.
    :return: the model in eval mode.
    """
    if isinstance(device, str):
        device = torch.device(device)

    if model_path is None:
        resolved = _resolve_bundled_weights(model_version)
        model_path = str(resolved)

    if not model_path.lower().endswith(".onnx"):
        raise ValueError(
            "Inference requires an ONNX model. PTH checkpoints are supported "
            "only by scripts/export_onnx.py."
        )

    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        msg = (
            "onnxruntime is required for inference; install the "
            "platform-appropriate package."
        )
        raise RuntimeError(msg) from exc

    providers = ["CPUExecutionProvider"]
    if device.type == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif device.type == "mps":
        providers = ["CPUExecutionProvider"]

    # The CPU arena allocator never returns memory to the OS once grown, so a
    # long loop of many tile inferences (WSI tiling) steadily raises peak RSS.
    # Disabling it trades a little per-call allocation overhead for bounded
    # memory use, which matters far more for large slides than raw throughput.
    #
    # Thread counts are bounded to the real (cgroup-aware) CPU budget rather
    # than left at ONNX Runtime's default, which sizes off the host's total
    # core count and can wildly oversubscribe a quota-limited container.
    cpu_budget = _detect_cpu_budget()
    torch.set_num_threads(cpu_budget)
    sess_options = ort.SessionOptions()
    sess_options.enable_cpu_mem_arena = False
    sess_options.enable_mem_pattern = False
    sess_options.intra_op_num_threads = cpu_budget
    sess_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model_path, sess_options=sess_options, providers=providers
    )
    active_providers = session.get_providers()
    if device.type == "cuda" and "CUDAExecutionProvider" not in active_providers:
        raise RuntimeError(
            "CUDA was requested, but ONNX Runtime did not activate "
            f"CUDAExecutionProvider. Active providers: {active_providers}. "
            "Install onnxruntime-gpu and verify the CUDA libraries are available."
        ) from None
    return ONNXModelWrapper(session, device=device.type)

    raise AssertionError("unreachable ONNX inference path")
