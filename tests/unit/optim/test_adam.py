"""Tests for the Adam optimizer."""

import numpy as np

from numpy_dl.optim.adam import Adam


class TestAdam:
    def test_step_updates_parameters(self) -> None:
        param = np.array([1.0, 2.0, 3.0])
        grad = np.array([0.1, 0.2, 0.3])
        opt = Adam([(param, grad)], lr=0.01)
        opt.step()
        assert param[0] != 1.0

    def test_bias_correction(self) -> None:
        param = np.array([1.0])
        grad = np.array([1.0])
        opt = Adam([(param, grad)], lr=0.001)
        opt.step()
        m = 0.9 * 0.0 + (1.0 - 0.9) * 1.0
        v = 0.999 * 0.0 + (1.0 - 0.999) * 1.0
        m_hat = m / (1.0 - 0.9**1)
        v_hat = v / (1.0 - 0.999**1)
        expected = 1.0 - 0.001 * m_hat / (np.sqrt(v_hat) + 1e-8)
        np.testing.assert_array_almost_equal(param, [expected])

    def test_state_accumulates(self) -> None:
        param = np.array([1.0])
        grad = np.array([1.0])
        opt = Adam([(param, grad)], lr=0.001)
        opt.step()
        assert opt.t == 1
        opt.step()
        assert opt.t == 2
        expected_m = 0.9 * (0.9 * 0.0 + 0.1 * 1.0) + 0.1 * 1.0
        np.testing.assert_array_almost_equal(opt._m[0], [expected_m])

    def test_multiple_steps(self) -> None:
        param = np.array([5.0])
        grad = np.array([0.5])
        opt = Adam([(param, grad)], lr=0.1)
        for _ in range(10):
            opt.step()
        assert opt.t == 10
        assert param[0] < 5.0
