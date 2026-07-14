"""Validate numpy-dl Dense layer against PyTorch nn.Linear and TF Dense.

Tests forward pass output and backward pass gradients match within tolerance.
"""

import pytest

np_dl = pytest.importorskip("numpy_dl", reason="numpy-dl package required")
torch = pytest.importorskip("torch", reason="PyTorch required for validation tests")
tf = pytest.importorskip("tensorflow", reason="TensorFlow required for validation tests")


class TestDenseForward:
    """Compare Dense layer forward pass across frameworks."""

    def test_matches_pytorch_linear(self) -> None:
        """numpy-dl Dense output matches PyTorch nn.Linear with same weights."""
        # TODO: implement when Dense layer exists
        pytest.skip("Dense layer not yet implemented")

    def test_matches_tensorflow_dense(self) -> None:
        """numpy-dl Dense output matches tf.keras.layers.Dense with same weights."""
        # TODO: implement when Dense layer exists
        pytest.skip("Dense layer not yet implemented")


class TestDenseBackward:
    """Compare Dense layer backward pass across frameworks."""

    def test_gradient_matches_pytorch(self) -> None:
        """numpy-dl Dense input gradients match PyTorch nn.Linear backward."""
        # TODO: implement when Dense layer exists
        pytest.skip("Dense layer not yet implemented")

    def test_weight_gradients_match_pytorch(self) -> None:
        """numpy-dl Dense weight gradients match PyTorch nn.Linear backward."""
        # TODO: implement when Dense layer exists
        pytest.skip("Dense layer not yet implemented")
