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


def he_init(
    shape: Sequence[int], rng: np.random.RandomState | None = None
) -> np.ndarray:
    """Initialize weights using He initialization (He et al., 2015).

    Designed for use with ReLU activations.

    W ~ N(0, 1) * sqrt(2 / fan_in)

    Args:
        shape: Shape of the weight tensor. The last axis is ``fan_in``.
        rng: Optional NumPy RandomState for reproducibility. If None, a
            module-level RandomState seeded at import is used, so behaviour is
            independent of the global ``np.random`` state.

    Returns:
        Weight array drawn from N(0, sqrt(2/fan_in)).
    """
    fan_in = shape[-1]
    r = rng if rng is not None else _default_rng
    return r.randn(*shape) * np.sqrt(2.0 / fan_in)


def xavier_init(
    shape: Sequence[int], rng: np.random.RandomState | None = None
) -> np.ndarray:
    """Initialize weights using Xavier/Glorot initialization (Glorot & Bengio, 2010).

    Designed for use with sigmoid or tanh activations.

    W ~ N(0, 1) * sqrt(1 / fan_in)

    Args:
        shape: Shape of the weight tensor. The last axis is ``fan_in``.
        rng: Optional NumPy RandomState for reproducibility. If None, a
            module-level RandomState seeded at import is used, so behaviour is
            independent of the global ``np.random`` state.

    Returns:
        Weight array drawn from N(0, sqrt(1/fan_in)).
    """
    fan_in = shape[-1]
    r = rng if rng is not None else _default_rng
    return r.randn(*shape) * np.sqrt(1.0 / fan_in)


# Module-private RNG used when no rng is injected. Decouples the default from
# global ``np.random`` state so two unrelated imports don't influence each other.
_default_rng = np.random.RandomState(seed=None)
