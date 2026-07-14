"""Weight initialization strategies for neural network parameters.

Creator references:
    He, K. et al. (2015). "Delving Deep into Rectifiers: Surpassing Human-Level
    Performance on ImageNet Classification." — He initialization.
    Glorot, X. & Bengio, Y. (2010). "Understanding the difficulty of training
    deep feedforward neural networks." — Xavier/Glorot initialization.

Mathematical equations:
    He init:    W ~ N(0, 1) * sqrt(2 / fan_in)
    Xavier init: W ~ N(0, 1) * sqrt(1 / fan_in)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def he_init(shape: Sequence[int]) -> np.ndarray:
    """Initialize weights using He initialization (He et al., 2015).

    Designed for use with ReLU activations.

    W ~ N(0, 1) * sqrt(2 / fan_in)

    Args:
        shape: Shape of the weight tensor. The last axis is ``fan_in``.

    Returns:
        Weight array drawn from N(0, sqrt(2/fan_in)).
    """
    fan_in = shape[-1]
    return np.random.randn(*shape) * np.sqrt(2.0 / fan_in)


def xavier_init(shape: Sequence[int]) -> np.ndarray:
    """Initialize weights using Xavier/Glorot initialization (Glorot & Bengio, 2010).

    Designed for use with sigmoid or tanh activations.

    W ~ N(0, 1) * sqrt(1 / fan_in)

    Args:
        shape: Shape of the weight tensor. The last axis is ``fan_in``.

    Returns:
        Weight array drawn from N(0, sqrt(1/fan_in)).
    """
    fan_in = shape[-1]
    return np.random.randn(*shape) * np.sqrt(1.0 / fan_in)
