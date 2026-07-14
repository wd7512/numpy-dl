"""Loss functions for training neural networks.

Creator references:
    Legendre, A.M. (1805). "Nouvelles méthodes pour la détermination des orbites
    des comètes." — Mean Squared Error.
    Shannon, C.E. (1948). "A Mathematical Theory of Communication." — Cross-entropy.

Mathematical equations:
    MSE:  L = (1/n) * Σ (y_pred - y_true)^2
          dL/dy_pred = 2 * (y_pred - y_true) / n

    Softmax Cross-Entropy:
          L = -(1/n) * Σ y_true * log(softmax(y_pred))
          dL/dy_pred = (softmax(y_pred) - y_true) / n
"""

from __future__ import annotations

import numpy as np

from numpy_dl.nn.activations import softmax


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute Mean Squared Error loss and its gradient.

    L = (1/n) * Σ (y_pred - y_true)^2

    Args:
        y_pred: Predicted values of shape (n, *).
        y_true: Target values, same shape as ``y_pred``.

    Returns:
        Tuple of (scalar loss, gradient w.r.t. y_pred).
    """
    n = y_true.size
    loss = float(np.mean((y_pred - y_true) ** 2))
    grad = 2.0 * (y_pred - y_true) / n
    return loss, grad


def softmax_cross_entropy_loss(
    y_pred: np.ndarray, y_true: np.ndarray
) -> tuple[float, np.ndarray]:
    """Compute softmax cross-entropy loss and its gradient.

    L = -(1/n) * Σ y_true * log(softmax(y_pred))

    Args:
        y_pred: Raw logits of shape (n, num_classes).
        y_true: One-hot encoded targets, same shape as ``y_pred``.

    Returns:
        Tuple of (scalar loss, gradient w.r.t. y_pred logits).
    """
    n = y_true.size
    probs = softmax(y_pred)
    loss = float(-np.mean(np.sum(y_true * np.log(probs + 1e-12), axis=-1)))
    grad = (probs - y_true) / n
    return loss, grad
