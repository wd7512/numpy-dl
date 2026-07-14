"""Stochastic Gradient Descent optimizer.

Creator references:
    Robbins, H. & Monro, S. (1951). "A Stochastic Approximation Method."
    — Stochastic gradient descent.

Mathematical equations:
    θ = θ - lr * ∇L(θ)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class SGD:
    """Vanilla Stochastic Gradient Descent optimizer.

    Args:
        params: List of (parameter, gradient) pairs.
        lr: Learning rate. Defaults to 0.01.
    """

    def __init__(self, params: Sequence[tuple[np.ndarray, np.ndarray]], lr: float = 0.01) -> None:
        self._params = list(params)
        self.lr = lr

    def step(self) -> None:
        """Update each parameter: param -= lr * grad."""
        for param, grad in self._params:
            param -= self.lr * grad
