"""End-to-end MLP comparison between numpy-dl and PyTorch.

Tests full forward + backward pass through a multi-layer network
with identical weights, inputs, and loss.
"""

import numpy as np
import pytest

from numpy_dl.nn.activations import ReLU
from numpy_dl.nn.layers import Dense
from numpy_dl.nn.losses import mse_loss

torch = pytest.importorskip("torch", reason="PyTorch required for validation tests")


def _sync_layer(np_layer: Dense, torch_layer: torch.nn.Linear) -> None:
    """Copy weights from torch.nn.Linear to numpy-dl Dense."""
    np_layer.W[:] = torch_layer.weight.detach().numpy().T
    np_layer.b[:] = torch_layer.bias.detach().numpy().reshape(1, -1)


class TestMLPForward:
    def test_output_matches_pytorch(self) -> None:
        """2-layer MLP forward pass matches PyTorch with same weights."""
        np.random.seed(42)
        in_feat, hidden, out_feat = 4, 8, 2
        batch = 5

        # numpy-dl MLP
        np_dense1 = Dense(in_feat, hidden)
        np_relu = ReLU()
        np_dense2 = Dense(hidden, out_feat)

        # PyTorch MLP
        torch_dense1 = torch.nn.Linear(in_feat, hidden)
        torch_relu = torch.nn.ReLU()
        torch_dense2 = torch.nn.Linear(hidden, out_feat)

        # Share weights
        _sync_layer(np_dense1, torch_dense1)
        _sync_layer(np_dense2, torch_dense2)

        x_np = np.random.randn(batch, in_feat).astype(np.float32)
        x_t = torch.tensor(x_np, dtype=torch.float32)

        # numpy-dl forward
        h_np = np_dense1.forward(x_np)
        h_np = np_relu.forward(h_np)
        out_np = np_dense2.forward(h_np)

        # PyTorch forward
        h_t = torch_dense1(x_t)
        h_t = torch_relu(h_t)
        out_t = torch_dense2(h_t).detach().numpy()

        np.testing.assert_allclose(out_np, out_t, atol=1e-6)


class TestMLPBackward:
    def test_gradients_match_pytorch(self) -> None:
        """2-layer MLP backward pass gradients match PyTorch."""
        np.random.seed(42)
        in_feat, hidden, out_feat = 4, 8, 2
        batch = 5

        # numpy-dl MLP
        np_dense1 = Dense(in_feat, hidden)
        np_relu = ReLU()
        np_dense2 = Dense(hidden, out_feat)

        # PyTorch MLP
        torch_dense1 = torch.nn.Linear(in_feat, hidden)
        torch_relu = torch.nn.ReLU()
        torch_dense2 = torch.nn.Linear(hidden, out_feat)

        # Share weights
        _sync_layer(np_dense1, torch_dense1)
        _sync_layer(np_dense2, torch_dense2)

        x_np = np.random.randn(batch, in_feat).astype(np.float32)
        y_true = np.random.randn(batch, out_feat).astype(np.float32)

        # numpy-dl forward + loss + backward
        h = np_dense1.forward(x_np)
        h = np_relu.forward(h)
        out_np = np_dense2.forward(h)
        _, grad = mse_loss(out_np, y_true)
        grad = np_dense2.backward(grad)
        grad = np_relu.backward(grad)
        np_dense1.backward(grad)

        # PyTorch forward + loss + backward
        x_t = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
        h_t = torch_dense1(x_t)
        h_t = torch_relu(h_t)
        out_t = torch_dense2(h_t)
        y_true_t = torch.tensor(y_true, dtype=torch.float32)
        loss_t = torch.nn.functional.mse_loss(out_t, y_true_t, reduction="mean")
        loss_t.backward()

        # Compare weight gradients
        np.testing.assert_allclose(
            np_dense1.dW, torch_dense1.weight.grad.numpy().T, atol=1e-6
        )
        np.testing.assert_allclose(
            np_dense1.db, torch_dense1.bias.grad.numpy().reshape(1, -1), atol=1e-6
        )
        np.testing.assert_allclose(
            np_dense2.dW, torch_dense2.weight.grad.numpy().T, atol=1e-6
        )
        np.testing.assert_allclose(
            np_dense2.db, torch_dense2.bias.grad.numpy().reshape(1, -1), atol=1e-6
        )
