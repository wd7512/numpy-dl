"""Neural network layer base class and dense (fully-connected) layer.

Creator references:
    Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for
    Information Storage and Organization in the Brain." — Perceptron.
    Rumelhart, D.E., Hinton, G.E. & Williams, R.J. (1986). "Learning
    representations by back-propagating errors." — Backpropagation.

Mathematical equations:
    Forward:  y = x @ W + b
    Backward: dW = x^T @ dL/dy,  db = sum(dL/dy, axis=0),  dx = dL/dy @ W^T
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from numpy_dl.utils.init import he_init


class Layer:
    """Abstract base class for all neural network layers.

    Every layer implements ``forward(x)`` and ``backward(grad)``.
    ``forward`` caches any state needed for ``backward``.
    ``backward`` computes parameter gradients and returns the upstream gradient.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Dense(Layer):
    """Fully-connected (dense) layer: y = x @ W + b.

    Attributes:
        W: Weight matrix of shape (in_features, out_features).
        b: Bias vector of shape (1, out_features).
        dW: Gradient of the loss w.r.t. W (set during ``backward``).
        db: Gradient of the loss w.r.t. b (set during ``backward``).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_init: Callable[[tuple[int, int]], np.ndarray] = he_init,
    ) -> None:
        self.W: np.ndarray = weight_init((in_features, out_features))
        self.b: np.ndarray = np.zeros((1, out_features))
        self._input: np.ndarray | None = None
        self.dW: np.ndarray = np.zeros_like(self.W)
        self.db: np.ndarray = np.zeros_like(self.b)
        self._cached_params: list[tuple[np.ndarray, np.ndarray]] = [
            (self.W, self.dW), (self.b, self.db)
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute y = x @ W + b and cache input for backward.

        Args:
            x: Input of shape (batch, in_features).

        Returns:
            Output of shape (batch, out_features).

        Raises:
            ValueError: If input dimensions don't match layer specs.
        """
        if x.ndim != 2:
            raise ValueError(f"Dense expects 2D input (batch, in_features), got {x.ndim}D")
        if x.shape[1] != self.W.shape[0]:
            raise ValueError(
                f"Input features {x.shape[1]} != layer in_features {self.W.shape[0]}"
            )
        self._input = x
        return x @ self.W + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Compute gradients and return upstream gradient.

        Args:
            grad: Upstream gradient dL/dy of shape (batch, out_features).

        Returns:
            Gradient w.r.t. input: dL/dx = grad @ W^T.

        Raises:
            RuntimeError: If called before forward().
        """
        if self._input is None:
            raise RuntimeError("Dense.backward() called before forward()")
        self.dW[:] = self._input.T @ grad
        self.db[:] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.T

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return [(W, dW), (b, db)] for the optimizer."""
        return self._cached_params
