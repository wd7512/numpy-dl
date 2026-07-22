"""Tests for numpy_dl.utils.init (he_init, xavier_init)."""

import functools

import numpy as np

from numpy_dl.utils.init import he_init, xavier_init


class TestHeInit:
    def test_shape_preserved(self) -> None:
        W = he_init((4, 8), rng=np.random.RandomState(0))
        assert W.shape == (4, 8)

    def test_same_rng_reproducible(self) -> None:
        w1 = he_init((3, 3), rng=np.random.RandomState(7))
        w2 = he_init((3, 3), rng=np.random.RandomState(7))
        np.testing.assert_array_equal(w1, w2)

    def test_different_seeds_diverge(self) -> None:
        w1 = he_init((3, 3), rng=np.random.RandomState(1))
        w2 = he_init((3, 3), rng=np.random.RandomState(2))
        assert not np.array_equal(w1, w2)

    def test_default_rng_not_tied_to_global(self) -> None:
        # Reset the module RNG by reimporting is brittle; instead verify the
        # default RNG draws from its own stream, not global np.random: setting
        # global seed(123) then sampling randn must not match what he_init
        # (no rng) returns next, even when re-seeded identically.
        np.random.seed(123)
        global_sample = np.random.randn(3, 3).copy()
        np.random.seed(123)
        init_sample = he_init((3, 3))
        # Both are sqrt(2/fan_in) * randn; if they shared global state the raw
        # randn parts would be equal. Compare after scaling out the sqrt factor.
        scale = np.sqrt(2.0 / 3)
        assert not np.array_equal(init_sample / scale, global_sample)

    def test_expected_scale_he(self) -> None:
        rng = np.random.RandomState(0)
        W = he_init((1000, 100), rng=rng)
        assert abs(W.std() - np.sqrt(2.0 / 100)) < 0.02


class TestXavierInit:
    def test_shape_preserved(self) -> None:
        W = xavier_init((4, 8), rng=np.random.RandomState(0))
        assert W.shape == (4, 8)

    def test_same_rng_reproducible(self) -> None:
        w1 = xavier_init((3, 3), rng=np.random.RandomState(7))
        w2 = xavier_init((3, 3), rng=np.random.RandomState(7))
        np.testing.assert_array_equal(w1, w2)

    def test_expected_scale_xavier(self) -> None:
        rng = np.random.RandomState(0)
        W = xavier_init((1000, 100), rng=rng)
        assert abs(W.std() - np.sqrt(1.0 / 100)) < 0.02


class TestInjectableViaDense:
    """Confirm the injectable RNG threads through Dense via functools.partial."""

    def test_dense_with_injected_rng_is_reproducible(self) -> None:
        from numpy_dl.nn.layers import Dense

        init = functools.partial(he_init, rng=np.random.RandomState(99))
        l1 = Dense(4, 8, weight_init=init)
        l2 = Dense(4, 8, weight_init=functools.partial(he_init, rng=np.random.RandomState(99)))
        np.testing.assert_array_equal(l1.W, l2.W)
