"""Validate numpy-dl optimizers against PyTorch equivalents.

Tests SGD and Adam weight updates match after identical gradient steps.
"""

import numpy as np
import pytest

from numpy_dl.optim.adam import Adam
from numpy_dl.optim.sgd import SGD

torch = pytest.importorskip("torch", reason="PyTorch required for validation tests")


class TestSGD:
    def test_weight_update_matches_pytorch(self) -> None:
        """numpy-dl SGD weight update matches PyTorch SGD."""
        shape = (3, 4)
        lr = 0.1
        np.random.seed(42)

        w_np = np.random.randn(*shape).astype(np.float32)
        w_t = w_np.copy()
        grad_np = np.random.randn(*shape).astype(np.float32)

        params = [(w_np, grad_np)]
        opt = SGD(params, lr=lr)
        opt.step()

        w_torch = torch.tensor(w_t, dtype=torch.float32, requires_grad=False)
        grad_t = torch.tensor(grad_np, dtype=torch.float32)
        w_torch = w_torch - lr * grad_t

        np.testing.assert_allclose(w_np, w_torch.numpy(), atol=1e-7)


class TestAdam:
    def test_weight_update_matches_pytorch(self) -> None:
        """numpy-dl Adam weight update matches PyTorch Adam."""
        shape = (3, 4)
        lr, beta1, beta2, eps = 0.001, 0.9, 0.999, 1e-8
        np.random.seed(42)

        w_np = np.random.randn(*shape).astype(np.float32)
        w_t = w_np.copy()
        grad_np = np.random.randn(*shape).astype(np.float32)

        # numpy-dl: pass gradient directly (not via zeros from init)
        params = [(w_np, grad_np)]
        opt = Adam(params, lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        opt._params = [(w_np, grad_np)]
        opt.step()

        # PyTorch
        w_torch = torch.tensor(w_t, dtype=torch.float32)
        torch_param = torch.nn.Parameter(w_torch)
        torch_opt = torch.optim.Adam(
            [torch_param], lr=lr, betas=(beta1, beta2), eps=eps
        )
        torch_param.grad = torch.tensor(grad_np)
        torch_opt.step()

        np.testing.assert_allclose(w_np, w_torch.detach().numpy(), atol=1e-6)

    def test_multiple_steps_match_pytorch(self) -> None:
        """numpy-dl Adam matches PyTorch Adam after multiple steps."""
        shape = (3, 4)
        lr, beta1, beta2, eps = 0.001, 0.9, 0.999, 1e-8
        np.random.seed(42)

        w_np = np.random.randn(*shape).astype(np.float32)
        w_t = w_np.copy()

        params = [(w_np, np.zeros_like(w_np))]
        opt = Adam(params, lr=lr, beta1=beta1, beta2=beta2, eps=eps)

        w_torch = torch.tensor(w_t, dtype=torch.float32)
        torch_param = torch.nn.Parameter(w_torch)
        torch_opt = torch.optim.Adam(
            [torch_param], lr=lr, betas=(beta1, beta2), eps=eps
        )

        for _ in range(5):
            grad_np = np.random.randn(*shape).astype(np.float32)
            # numpy-dl
            params[0] = (w_np, grad_np)
            opt._params = params
            opt.step()

            # PyTorch
            torch_param.grad = torch.tensor(grad_np)
            torch_opt.step()

        np.testing.assert_allclose(w_np, w_torch.detach().numpy(), atol=1e-5)
