"""Tests for loss functions."""

import numpy as np

from numpy_dl.nn.losses import mse_loss, softmax_cross_entropy_loss


class TestMSELoss:
    def test_perfect_prediction(self) -> None:
        y = np.array([[1.0, 2.0]])
        loss, grad = mse_loss(y, y)
        assert loss == 0.0
        np.testing.assert_array_equal(grad, np.zeros_like(y))

    def test_gradient_value(self) -> None:
        y_pred = np.array([[2.0]])
        y_true = np.array([[1.0]])
        loss, grad = mse_loss(y_pred, y_true)
        assert loss == 1.0
        np.testing.assert_array_almost_equal(grad, np.array([[2.0]]))


class TestSoftmaxCrossEntropyLoss:
    def test_with_one_hot(self) -> None:
        y_pred = np.array([[10.0, 1.0, 1.0]])
        y_true = np.array([[1.0, 0.0, 0.0]])
        loss, grad = softmax_cross_entropy_loss(y_pred, y_true)
        assert loss < 0.1
        np.testing.assert_array_almost_equal(grad.sum(), 0.0, decimal=5)

    def test_gradient_shape(self) -> None:
        y_pred = np.random.randn(4, 3)
        y_true = np.eye(3)[np.random.randint(0, 3, 4)]
        _, grad = softmax_cross_entropy_loss(y_pred, y_true)
        assert grad.shape == y_pred.shape
