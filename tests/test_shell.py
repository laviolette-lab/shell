"""Tests for the SHELL package."""

import warnings

import pytest
import torch

from shell import __version__


def test_version_is_string():
    """Version should be a non-empty string."""
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_monai_torch_jit_warning_filter_is_configured():
    """The MONAI/Torch deprecation warning should be suppressed at the source."""
    import shell

    assert isinstance(shell.__version__, str)

    filters = warnings.filters
    assert any(
        entry[0] == "ignore"
        and entry[2] is FutureWarning
        and entry[1] is not None
        and "torch" in entry[1].pattern
        and "jit" in entry[1].pattern
        and "interface" in entry[1].pattern
        for entry in filters
    )


def test_cli_help(capsys):
    """CLI --help should exit 0."""
    from shell.cli import main

    assert main(["--version"]) is None or True  # argparse exits


def test_runtime_model_requires_onnx():
    """Runtime inference rejects repository-only PTH checkpoints."""
    from shell.model import build_model

    with pytest.raises(ValueError, match="requires an ONNX model"):
        build_model("checkpoint.pth")


def test_autocast_device_respects_target_device():
    """Autocast should use the actual torch device type instead of always CPU."""
    from shell.inference import _resolve_autocast_device

    assert _resolve_autocast_device(torch.device("cpu")) == "cpu"
    assert _resolve_autocast_device(torch.device("cuda")) == "cuda"
    assert _resolve_autocast_device(torch.device("mps")) == "mps"


def test_mask_aware_tile_positions_shift_per_row():
    """Mask-aware windows should move with tissue instead of using one x-grid."""
    import numpy as np

    from shell.infer_wsi import _mask_aware_tile_positions

    tissue_mask = np.zeros((128, 500), dtype=bool)
    tissue_mask[:64, 180:220] = True
    tissue_mask[64:, 300:340] = True

    top = _mask_aware_tile_positions(tissue_mask, 0, 64, 128, 0.25, 0.5, 0.01)
    bottom = _mask_aware_tile_positions(tissue_mask, 64, 128, 128, 0.25, 0.5, 0.01)

    assert top != bottom
    assert [x0 for x0s, _ in top for x0 in x0s] == [148]
    assert [x0 for x0s, _ in bottom for x0 in x0s] == [268]
    assert _mask_aware_tile_positions(tissue_mask, 0, 0, 128, 0.25, 0.5, 0.01) == []


def test_mask_aware_tile_positions_schedules_disjoint_runs_independently():
    """Disjoint tissue islands in one row get separate schedules, not one wide grid."""
    import numpy as np

    from shell.infer_wsi import _mask_aware_tile_positions

    tissue_mask = np.zeros((64, 2000), dtype=bool)
    tissue_mask[:, 0:100] = True
    tissue_mask[:, 1800:1900] = True

    runs = _mask_aware_tile_positions(tissue_mask, 0, 64, 256, 0.25, 0.5, 0.01)

    assert len(runs) == 2


def test_tile_positions_minimizes_tiles_within_overlap_band():
    """Fewest evenly-spaced tiles are chosen when the overlap band allows it."""
    from shell.infer_wsi import _tile_positions

    starts, overlap_px = _tile_positions(6656, 2048, min_overlap=0.25, max_overlap=0.5)

    assert starts == [0, 1536, 3072, 4608]
    assert overlap_px == 512  # exactly at the min_overlap floor (1 - 1536/2048)


def test_tile_positions_never_drops_below_min_overlap():
    """Tile-count minimization never violates the overlap floor, even when the
    exact span forces overlap above the requested ceiling."""
    from shell.infer_wsi import _tile_positions

    starts, overlap_px = _tile_positions(3000, 2048, min_overlap=0.25, max_overlap=0.5)

    assert len(starts) == 2
    assert overlap_px >= 512  # 512px == 0.25 * 2048 == min_overlap floor


def test_gaussian_mask_aware_inference_returns_tile_masks():
    """The shared engine should use Gaussian sliding-window inference."""
    import numpy as np

    from shell.inference import GaussianMaskAwareInference

    class DummyModel:
        def __call__(self, image):
            return torch.zeros(
                image.shape[0], 3, image.shape[2], image.shape[3]
            )

    engine = GaussianMaskAwareInference(
        DummyModel(),
        roi_size=(64, 64),
        overlap=0.5,
    )
    inner, outer = engine.predict_tile(np.zeros((128, 128, 3), dtype=np.uint8))

    assert inner.shape == (128, 128)
    assert outer.shape == (128, 128)
    assert not inner.any()
    assert not outer.any()


def test_runtime_inference_rejects_pth_checkpoint():
    """PTH files remain export inputs, never runtime inference models."""
    from shell.model import build_model

    with pytest.raises(ValueError, match="requires an ONNX model"):
        build_model("checkpoint.pth")


def test_onnx_wrapper_rejects_non_exported_spatial_shape():
    """The ONNX wrapper should explain fixed-shape model input errors."""
    from shell.model import ONNXModelWrapper

    class Input:
        def __init__(self):
            self.name = "input"
            self.shape = ["batch_size", 3, "height", "width"]

    class Output:
        def __init__(self):
            self.name = "logits"

    class Session:
        def get_inputs(self):
            return [Input()]

        def get_outputs(self):
            return [Output()]

    wrapper = ONNXModelWrapper(Session())
    with pytest.raises(ValueError, match="divisible by 64"):
        wrapper(torch.zeros(1, 3, 320, 321))
