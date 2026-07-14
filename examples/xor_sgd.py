"""XOR training example using Dense layers, ReLU, MSE loss, and SGD.

Trains a small network: Dense(2,4) -> ReLU -> Dense(4,1) -> MSE
to learn the XOR function. Should converge to loss < 0.01 in ~1000 epochs.
"""

from __future__ import annotations

import logging

import numpy as np

from numpy_dl.nn.activations import ReLU
from numpy_dl.nn.layers import Dense
from numpy_dl.nn.losses import mse_loss
from numpy_dl.optim.sgd import SGD

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

np.random.seed(0)

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([[0], [1], [1], [0]], dtype=np.float64)

# Network
dense1 = Dense(2, 4)
relu = ReLU()
dense2 = Dense(4, 1)

# Optimizer
params = dense1.parameters() + dense2.parameters()
optimizer = SGD(params, lr=0.05)

# Training loop
for epoch in range(1001):
    # Forward
    out = dense1.forward(X)
    out = relu.forward(out)
    out = dense2.forward(out)

    # Loss
    loss, grad = mse_loss(out, y)

    # Backward
    grad = dense2.backward(grad)
    grad = relu.backward(grad)
    grad = dense1.backward(grad)

    # Update
    optimizer.step()

    if epoch % 100 == 0:
        logger.info("Epoch %d  loss=%.6f", epoch, loss)

logger.info("Final loss: %.6f", loss)
