"""Validate numpy-dl Dense layer against PyTorch nn.Linear.

Tests forward pass output and backward pass gradients match within tolerance.
"""

import numpy as np
import pytest

from numpy_dl.nn.layers import Dense

torch = pytest.importorskip("torch", reason="PyTorch required for validation tests")


class TestDenseForward:
    """Compare Dense layer forward pass against PyTorch."""

    def test_matches_pytorch_linear(self) -> None:
        """numpy-dl Dense output matches PyTorch nn.Linear with same weights."""
        in_feat, out_feat = 4, 3
        batch = 5

        np_layer = Dense(in_feat, out_feat)
        torch_layer = torch.nn.Linear(in_feat, out_feat, bias=True)

        # Share weights: PyTorch Linear stores (out, in), numpy-dl stores (in, out)
        np_layer.W[:] = torch_layer.weight.detach().numpy().T
        np_layer.b[:] = torch_layer.bias.detach().numpy().reshape(1, -1)

        x_np = np.random.randn(batch, in_feat).astype(np.float32)
        x_t = torch.tensor(x_np, dtype=torch.float32)

        out_np = np_layer.forward(x_np)
        out_torch = torch_layer(x_t).detach().numpy()

        np.testing.assert_allclose(out_np, out_torch, atol=1e-6)


class TestDenseBackward:
    """Compare Dense layer backward pass against PyTorch autograd."""

    def test_gradient_matches_pytorch(self) -> None:
        """numpy-dl Dense input gradients match PyTorch nn.Linear backward."""
        in_feat, out_feat = 4, 3
        batch = 5

        np_layer = Dense(in_feat, out_feat)
        torch_layer = torch.nn.Linear(in_feat, out_feat, bias=True)

        np_layer.W[:] = torch_layer.weight.detach().numpy().T
        np_layer.b[:] = torch_layer.bias.detach().numpy().reshape(1, -1)

        x_np = np.random.randn(batch, in_feat).astype(np.float32)
        x_t = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)

        # numpy-dl forward + backward
        out_np = np_layer.forward(x_np)
        grad_out_np = np.random.randn(*out_np.shape).astype(np.float32)
        grad_input_np = np_layer.backward(grad_out_np)

        # PyTorch forward + backward
        out_t = torch_layer(x_t)
        grad_out_t = torch.tensor(grad_out_np, dtype=torch.float32)
        out_t.backward(grad_out_t)
        grad_input_torch = x_t.grad.numpy()

        np.testing.assert_allclose(grad_input_np, grad_input_torch, atol=1e-6)

    def test_weight_gradients_match_pytorch(self) -> None:
        """numpy-dl Dense weight gradients match PyTorch nn.Linear backward."""
        in_feat, out_feat = 4, 3
        batch = 5

        np_layer = Dense(in_feat, out_feat)
        torch_layer = torch.nn.Linear(in_feat, out_feat, bias=True)

        np_layer.W[:] = torch_layer.weight.detach().numpy().T
        np_layer.b[:] = torch_layer.bias.detach().numpy().reshape(1, -1)

        x_np = np.random.randn(batch, in_feat).astype(np.float32)
        x_t = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)

        # numpy-dl forward + backward
        out_np = np_layer.forward(x_np)
        grad_out_np = np.random.randn(*out_np.shape).astype(np.float32)
        np_layer.backward(grad_out_np)

        # PyTorch forward + backward
        out_t = torch_layer(x_t)
        grad_out_t = torch.tensor(grad_out_np, dtype=torch.float32)
        out_t.backward(grad_out_t)

        # Compare weight gradients (numpy-dl W is (in, out), PyTorch is (out, in))
        np.testing.assert_allclose(
            np_layer.dW, torch_layer.weight.grad.numpy().T, atol=1e-6
        )
        np.testing.assert_allclose(
            np_layer.db, torch_layer.bias.grad.numpy().reshape(1, -1), atol=1e-6
        )
