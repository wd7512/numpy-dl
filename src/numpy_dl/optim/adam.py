"""Adam optimizer with bias-corrected first and second moment estimates.

Creator references:
    Kingma, D.P. & Ba, J. (2014). "Adam: A Method for Stochastic Optimization."
    — Adaptive moment estimation.

Mathematical equations:
    m_t  = β1 * m_{t-1} + (1 - β1) * g_t
    v_t  = β2 * v_{t-1} + (1 - β2) * g_t^2
    m̂_t  = m_t / (1 - β1^t)
    v̂_t  = v_t / (1 - β2^t)
    θ_t  = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class Adam:
    """Adam optimizer (Kingma & Ba, 2014).

    Args:
        params: List of (parameter, gradient) pairs.
        lr: Learning rate. Defaults to 0.001.
        beta1: Exponential decay for first moment. Defaults to 0.9.
        beta2: Exponential decay for second moment. Defaults to 0.999.
        eps: Small constant for numerical stability. Defaults to 1e-8.
    """

    def __init__(
        self,
        params: Sequence[tuple[np.ndarray, np.ndarray]],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self._params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t: int = 0
        self._m: list[np.ndarray] = [np.zeros_like(p) for p, _ in self._params]
        self._v: list[np.ndarray] = [np.zeros_like(p) for p, _ in self._params]

    def step(self) -> None:
        """Update parameters using bias-corrected Adam."""
        self.t += 1
        bc1 = 1.0 - self.beta1**self.t
        bc2 = 1.0 - self.beta2**self.t
        for i, (param, grad) in enumerate(self._params):
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * grad
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * grad**2
            m_hat = self._m[i] / bc1
            v_hat = self._v[i] / bc2
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
