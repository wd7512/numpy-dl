"""End-to-end MLP comparison across numpy-dl, PyTorch, and TensorFlow.

Tests full forward + backward pass through a multi-layer network
with identical weights, inputs, and loss.
"""

import pytest

np_dl = pytest.importorskip("numpy_dl", reason="numpy-dl package required")
torch = pytest.importorskip("torch", reason="PyTorch required for validation tests")
tf = pytest.importorskip("tensorflow", reason="TensorFlow required for validation tests")


class TestMLPForward:
    def test_output_matches_pytorch(self) -> None:
        """2-layer MLP forward pass matches PyTorch with same weights."""
        # TODO: implement when Dense + activations exist
        pytest.skip("MLP components not yet implemented")

    def test_output_matches_tensorflow(self) -> None:
        """2-layer MLP forward pass matches TensorFlow with same weights."""
        # TODO: implement when Dense + activations exist
        pytest.skip("MLP components not yet implemented")


class TestMLPBackward:
    def test_gradients_match_pytorch(self) -> None:
        """2-layer MLP backward pass gradients match PyTorch."""
        # TODO: implement when Dense + activations exist
        pytest.skip("MLP components not yet implemented")
