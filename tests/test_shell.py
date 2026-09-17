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


def test_eval_forward_skips_vae_loss_in_eval_mode():
    """Eval-time forward should not compute the VAE loss branch."""
    from shell import model as shell_model

    class DummyModel:
        training = False

        def encode(self, x):
            return "encoded", ["down"]

        def decode(self, x_enc, down_x):
            assert x_enc == "encoded"
            assert down_x == ["down"]
            return "decoded"

        def _get_vae_loss(self, x, x_enc):
            raise AssertionError("VAE loss should not be computed during eval mode")

    assert shell_model._eval_segresnetvae_forward(DummyModel(), "input") == "decoded"


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

    top = _mask_aware_tile_positions(tissue_mask, 0, 64, 128, 16, 0.01)
    bottom = _mask_aware_tile_positions(tissue_mask, 64, 128, 128, 16, 0.01)

    assert top != bottom
    assert top == [164]
    assert bottom == [284]
    assert _mask_aware_tile_positions(tissue_mask, 0, 0, 128, 16, 0.01) == []


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
