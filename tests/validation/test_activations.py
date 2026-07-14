"""Validate numpy-dl activations against PyTorch equivalents.

Tests ReLU, Sigmoid, and Softmax forward and backward passes.
"""

import numpy as np
import pytest

from numpy_dl.nn.activations import ReLU, Sigmoid, softmax

torch = pytest.importorskip("torch", reason="PyTorch required for validation tests")


class TestReLU:
    def test_forward_matches_pytorch(self) -> None:
        """numpy-dl ReLU forward matches PyTorch."""
        np_relu = ReLU()
        torch_relu = torch.nn.ReLU()

        x_np = np.random.randn(5, 4).astype(np.float32)
        x_t = torch.tensor(x_np, dtype=torch.float32)

        np.testing.assert_allclose(
            np_relu.forward(x_np), torch_relu(x_t).numpy(), atol=1e-6
        )

    def test_backward_matches_pytorch(self) -> None:
        """numpy-dl ReLU backward matches PyTorch."""
        np_relu = ReLU()
        torch_relu = torch.nn.ReLU()

        x_np = np.random.randn(5, 4).astype(np.float32)
        x_t = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)

        grad_out_np = np.random.randn(*x_np.shape).astype(np.float32)
        np_relu.forward(x_np)
        np_grad = np_relu.backward(grad_out_np)

        torch_out = torch_relu(x_t)
        torch_out.backward(torch.tensor(grad_out_np))
        torch_grad = x_t.grad.numpy()

        np.testing.assert_allclose(np_grad, torch_grad, atol=1e-6)


class TestSigmoid:
    def test_forward_matches_pytorch(self) -> None:
        """numpy-dl Sigmoid forward matches PyTorch."""
        np_sig = Sigmoid()
        torch_sig = torch.nn.Sigmoid()

        x_np = np.random.randn(5, 4).astype(np.float32)
        x_t = torch.tensor(x_np, dtype=torch.float32)

        np.testing.assert_allclose(
            np_sig.forward(x_np), torch_sig(x_t).numpy(), atol=1e-6
        )

    def test_backward_matches_pytorch(self) -> None:
        """numpy-dl Sigmoid backward matches PyTorch."""
        np_sig = Sigmoid()
        torch_sig = torch.nn.Sigmoid()

        x_np = np.random.randn(5, 4).astype(np.float32)
        x_t = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)

        grad_out_np = np.random.randn(*x_np.shape).astype(np.float32)
        np_sig.forward(x_np)
        np_grad = np_sig.backward(grad_out_np)

        torch_out = torch_sig(x_t)
        torch_out.backward(torch.tensor(grad_out_np))
        torch_grad = x_t.grad.numpy()

        np.testing.assert_allclose(np_grad, torch_grad, atol=1e-6)


class TestSoftmax:
    def test_matches_pytorch(self) -> None:
        """numpy-dl softmax matches PyTorch."""
        x_np = np.random.randn(5, 4).astype(np.float32)
        x_t = torch.tensor(x_np, dtype=torch.float32)

        np.testing.assert_allclose(
            softmax(x_np), torch.nn.functional.softmax(x_t, dim=-1).numpy(), atol=1e-6
        )
