"""XOR training example using Dense layers, ReLU, MSE loss, and Adam.

Trains a small network: Dense(2,4) -> ReLU -> Dense(4,1) -> MSE
to learn the XOR function. Converges faster than SGD due to adaptive
learning rates.

Creator references:
    Kingma, D.P. & Ba, J. (2014). "Adam: A Method for Stochastic Optimization."
"""

from __future__ import annotations

import logging

import numpy as np

from numpy_dl.nn.activations import ReLU
from numpy_dl.nn.layers import Dense
from numpy_dl.nn.losses import mse_loss
from numpy_dl.optim.adam import Adam

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

np.random.seed(0)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([[0], [1], [1], [0]], dtype=np.float64)

dense1 = Dense(2, 4)
relu = ReLU()
dense2 = Dense(4, 1)

params = dense1.parameters() + dense2.parameters()
optimizer = Adam(params, lr=0.01)

for epoch in range(1001):
    out = dense1.forward(X)
    out = relu.forward(out)
    out = dense2.forward(out)

    loss, grad = mse_loss(out, y)

    grad = dense2.backward(grad)
    grad = relu.backward(grad)
    grad = dense1.backward(grad)

    optimizer.step()

    if epoch % 100 == 0:
        logger.info("Epoch %d  loss=%.6f", epoch, loss)

logger.info("Final loss: %.6f", loss)
