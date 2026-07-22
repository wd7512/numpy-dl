"""Tests for the SGD optimizer."""

import numpy as np

from numpy_dl.optim.sgd import SGD


class TestSGD:
    def test_step_updates_parameters(self) -> None:
        param = np.array([1.0, 2.0, 3.0])
        grad = np.array([0.1, 0.2, 0.3])
        opt = SGD([(param, grad)], lr=1.0)
        opt.step()
        np.testing.assert_array_almost_equal(param, [0.9, 1.8, 2.7])

    def test_learning_rate_scaling(self) -> None:
        param = np.array([1.0])
        grad = np.array([1.0])
        opt = SGD([(param, grad)], lr=0.01)
        opt.step()
        np.testing.assert_array_almost_equal(param, [0.99])
