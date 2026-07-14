"""Validate numpy-dl activations against PyTorch and TensorFlow equivalents.

Tests ReLU, Sigmoid, Tanh forward and backward passes.
"""

import pytest

np_dl = pytest.importorskip("numpy_dl", reason="numpy-dl package required")
torch = pytest.importorskip("torch", reason="PyTorch required for validation tests")
tf = pytest.importorskip("tensorflow", reason="TensorFlow required for validation tests")


class TestReLU:
    def test_forward_matches_pytorch(self) -> None:
        # TODO: implement when ReLU exists
        pytest.skip("ReLU not yet implemented")

    def test_backward_matches_pytorch(self) -> None:
        # TODO: implement when ReLU exists
        pytest.skip("ReLU not yet implemented")


class TestSigmoid:
    def test_forward_matches_pytorch(self) -> None:
        # TODO: implement when Sigmoid exists
        pytest.skip("Sigmoid not yet implemented")

    def test_backward_matches_pytorch(self) -> None:
        # TODO: implement when Sigmoid exists
        pytest.skip("Sigmoid not yet implemented")


class TestTanh:
    def test_forward_matches_pytorch(self) -> None:
        # TODO: implement when Tanh exists
        pytest.skip("Tanh not yet implemented")

    def test_backward_matches_pytorch(self) -> None:
        # TODO: implement when Tanh exists
        pytest.skip("Tanh not yet implemented")
