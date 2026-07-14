"""Sequential container for chaining neural network layers.

Creator references:
    Rumelhart, D.E., Hinton, G.E. & Williams, R.J. (1986).
    — Feedforward network architecture.

Mathematical equations:
    Forward:  x_{i+1} = layer_i(x_i)
    Backward: grad_{i} = layer_i.backward(grad_{i+1})
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from numpy_dl.nn.layers import Layer


class Sequential:
    """A linear stack of layers.

    Forward pass chains layers in order; backward pass reverses them.
    Collects all trainable parameters from child layers.
    """

    def __init__(self, layers: Sequence[Layer]) -> None:
        self._layers = list(layers)
        if not self._layers:
            logging.getLogger(__name__).warning(
                "Sequential created with no layers — forward/backward are no-ops"
            )

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self._layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self._layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        params: list[tuple[np.ndarray, np.ndarray]] = []
        for layer in self._layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params
