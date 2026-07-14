"""Tests for Dense layer forward and backward passes."""

import numpy as np

from numpy_dl.nn.layers import Dense, Layer


class TestDenseForward:
    def test_output_shape(self) -> None:
        layer = Dense(3, 5)
        x = np.random.randn(2, 3)
        out = layer.forward(x)
        assert out.shape == (2, 5)

    def test_single_sample(self) -> None:
        layer = Dense(4, 2)
        x = np.ones((1, 4))
        out = layer.forward(x)
        assert out.shape == (1, 2)


class TestDenseBackward:
    def test_gradient_shapes(self) -> None:
        layer = Dense(3, 5)
        x = np.random.randn(4, 3)
        layer.forward(x)
        grad = np.random.randn(4, 5)
        dx = layer.backward(grad)
        assert dx.shape == (4, 3)
        assert layer.dW is not None
        assert layer.dW.shape == (3, 5)
        assert layer.db is not None
        assert layer.db.shape == (1, 5)

    def test_gradient_flow(self) -> None:
        layer = Dense(2, 2, weight_init=np.ones)
        x = np.array([[1.0, 2.0]])
        layer.forward(x)
        grad = np.array([[1.0, 1.0]])
        dx = layer.backward(grad)
        expected_dW = np.array([[1.0, 1.0], [2.0, 2.0]])
        np.testing.assert_array_almost_equal(layer.dW, expected_dW)
        np.testing.assert_array_almost_equal(dx, np.array([[2.0, 2.0]]))


class TestLayerBase:
    def test_base_raises(self) -> None:
        layer = Layer()
        try:
            layer.forward(np.zeros(1))
        except NotImplementedError:
            pass
        else:
            raise AssertionError("Layer.forward should raise NotImplementedError")
        try:
            layer.backward(np.zeros(1))
        except NotImplementedError:
            pass
        else:
            raise AssertionError("Layer.backward should raise NotImplementedError")
