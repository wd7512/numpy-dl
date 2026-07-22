"""Tests for ReLU, Sigmoid, and softmax."""

import numpy as np

from numpy_dl.nn.activations import ReLU, Sigmoid, softmax


class TestReLU:
    def test_negative_becomes_zero(self) -> None:
        relu = ReLU()
        x = np.array([-1.0, -2.0, 0.0])
        out = relu.forward(x)
        np.testing.assert_array_equal(out, [0.0, 0.0, 0.0])

    def test_positive_passthrough(self) -> None:
        relu = ReLU()
        x = np.array([1.0, 2.0, 3.0])
        out = relu.forward(x)
        np.testing.assert_array_equal(out, x)

    def test_backward(self) -> None:
        relu = ReLU()
        x = np.array([-1.0, 2.0, 0.0])
        relu.forward(x)
        grad = np.array([1.0, 1.0, 1.0])
        dx = relu.backward(grad)
        np.testing.assert_array_equal(dx, [0.0, 1.0, 0.0])


class TestSigmoid:
    def test_output_range(self) -> None:
        sig = Sigmoid()
        x = np.array([-10.0, 0.0, 10.0])
        out = sig.forward(x)
        assert np.all(out > 0) and np.all(out < 1)

    def test_backward(self) -> None:
        sig = Sigmoid()
        x = np.array([0.0])
        out = sig.forward(x)
        dx = sig.backward(np.array([1.0]))
        np.testing.assert_array_almost_equal(dx, out * (1 - out))


class TestSoftmax:
    def test_sums_to_one(self) -> None:
        x = np.array([[1.0, 2.0, 3.0]])
        probs = softmax(x)
        np.testing.assert_almost_equal(probs.sum(), 1.0)

    def test_batch(self) -> None:
        x = np.random.randn(4, 3)
        probs = softmax(x)
        np.testing.assert_array_almost_equal(probs.sum(axis=-1), np.ones(4))
