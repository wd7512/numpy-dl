"""Activation functions and layers for neural networks.

Creator references:
    Hahnloser, R.H. (1998). "Digital selection and analogue amplification
    coexist in a cortex-inspired silicon circuit." — ReLU activation.
    Bridle, J.S. (1990). "Probabilistic interpretation of feedforward
    classification network outputs." — Softmax function.

Mathematical equations:
    ReLU:    f(x) = max(0, x)
    Sigmoid: σ(x) = 1 / (1 + exp(-x))
    Softmax: softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
"""

from __future__ import annotations

import numpy as np


class ReLU:
    """Rectified Linear Unit activation layer.

    Forward: f(x) = max(0, x)
    Backward: f'(x) = 1 if x > 0, else 0
    """

    def __init__(self) -> None:
        self._input: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply ReLU: output = max(0, x).

        Args:
            x: Input array of any shape.

        Returns:
            Element-wise max(0, x).
        """
        self._input = x
        return np.maximum(0, x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backpropagate through ReLU.

        Args:
            grad: Upstream gradient (same shape as forward input).

        Returns:
            Gradient masked by the forward mask (x > 0).

        Raises:
            RuntimeError: If called before forward().
        """
        if self._input is None:
            raise RuntimeError("ReLU.backward() called before forward()")
        return grad * (self._input > 0).astype(grad.dtype)


class Sigmoid:
    """Sigmoid activation layer.

    Forward: σ(x) = 1 / (1 + exp(-x))
    Backward: σ'(x) = σ(x) * (1 - σ(x))
    """

    def __init__(self) -> None:
        self._output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply sigmoid: output = 1 / (1 + exp(-x)).

        Args:
            x: Input array of any shape.

        Returns:
            Element-wise sigmoid output in (0, 1).
        """
        self._output = 1.0 / (1.0 + np.exp(-x))
        return self._output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backpropagate through sigmoid.

        Args:
            grad: Upstream gradient.

        Returns:
            Gradient: grad * σ(x) * (1 - σ(x)).

        Raises:
            RuntimeError: If called before forward().
        """
        if self._output is None:
            raise RuntimeError("Sigmoid.backward() called before forward()")
        return grad * self._output * (1.0 - self._output)


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax along the last axis.

    softmax(x_i) = exp(x_i) / Σ_j exp(x_j)

    Uses the log-sum-exp trick for numerical stability.

    Args:
        x: Input array of shape (..., C) where C is the class dimension.

    Returns:
        Softmax probabilities summing to 1 along the last axis.
    """
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
