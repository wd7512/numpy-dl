"""Validate numpy-dl optimizers against PyTorch equivalents.

Tests SGD and Adam weight updates match after identical gradient steps.
"""

import pytest

np_dl = pytest.importorskip("numpy_dl", reason="numpy-dl package required")
torch = pytest.importorskip("torch", reason="PyTorch required for validation tests")
tf = pytest.importorskip("tensorflow", reason="TensorFlow required for validation tests")


class TestSGD:
    def test_weight_update_matches_pytorch(self) -> None:
        # TODO: implement when SGD exists
        pytest.skip("SGD not yet implemented")


class TestAdam:
    def test_weight_update_matches_pytorch(self) -> None:
        # TODO: implement when Adam exists
        pytest.skip("Adam not yet implemented")
