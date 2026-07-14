"""Integration test: train XOR to convergence."""

import numpy as np

from numpy_dl.nn.activations import ReLU
from numpy_dl.nn.layers import Dense
from numpy_dl.nn.losses import mse_loss
from numpy_dl.optim.sgd import SGD


def test_xor_converges() -> None:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([[0], [1], [1], [0]], dtype=np.float64)

    np.random.seed(0)
    dense1 = Dense(2, 4)
    relu = ReLU()
    dense2 = Dense(4, 1)
    params = dense1.parameters() + dense2.parameters()
    optimizer = SGD(params, lr=0.05)

    for _ in range(2000):
        out = dense1.forward(X)
        out = relu.forward(out)
        out = dense2.forward(out)
        loss, grad = mse_loss(out, y)
        grad = dense2.backward(grad)
        grad = relu.backward(grad)
        grad = dense1.backward(grad)
        optimizer.step()

    assert loss < 0.01, f"XOR did not converge: loss={loss:.6f}"
