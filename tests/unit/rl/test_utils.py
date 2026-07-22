"""Tests for rl utility functions."""

import numpy as np
import pytest

from numpy_dl.rl.utils import (
    categorical_sample,
    compute_gae,
    epsilon_greedy,
    normalize_advantages,
)


class TestEpsilonGreedy:
    def test_greedy_when_epsilon_zero(self) -> None:
        rng = np.random.RandomState(0)
        q = np.array([1.0, 5.0, 3.0])
        for _ in range(20):
            assert epsilon_greedy(q, 0.0, rng) == 1

    def test_random_when_epsilon_one(self) -> None:
        rng = np.random.RandomState(42)
        q = np.array([1.0, 2.0])
        actions = {epsilon_greedy(q, 1.0, rng) for _ in range(100)}
        assert actions == {0, 1}

    def test_empty_q_values_raises_clear_error(self) -> None:
        rng = np.random.RandomState(0)
        with pytest.raises(ValueError, match="epsilon_greedy: q_values must not be empty"):
            epsilon_greedy(np.array([]), 0.5, rng)


class TestCategoricalSample:
    def test_returns_valid_action(self) -> None:
        rng = np.random.RandomState(0)
        probs = np.array([0.2, 0.3, 0.5])
        action = categorical_sample(probs, rng)
        assert 0 <= action < 3

    def test_respects_distribution(self) -> None:
        rng = np.random.RandomState(42)
        probs = np.array([0.0, 0.0, 1.0])
        for _ in range(20):
            assert categorical_sample(probs, rng) == 2

    def test_empty_probs_raises_clear_error(self) -> None:
        rng = np.random.RandomState(0)
        with pytest.raises(ValueError, match="categorical_sample: probs must not be empty"):
            categorical_sample(np.array([]), rng)


class TestNormalizeAdvantages:
    def test_zero_mean(self) -> None:
        advantages = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normed = normalize_advantages(advantages)
        assert abs(np.mean(normed)) < 1e-6

    def test_unit_variance(self) -> None:
        advantages = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normed = normalize_advantages(advantages)
        assert abs(np.std(normed) - 1.0) < 1e-6

    def test_constant_returns(self) -> None:
        advantages = np.array([3.0, 3.0, 3.0])
        normed = normalize_advantages(advantages)
        np.testing.assert_allclose(normed, 0.0)


class TestComputeGAE:
    def test_returns_correct_shapes(self) -> None:
        T = 10
        rewards = np.ones(T)
        values = np.zeros(T)
        dones = np.zeros(T, dtype=bool)
        dones[-1] = True
        advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        assert advantages.shape == (T,)
        assert returns.shape == (T,)

    def test_terminal_value_zero(self) -> None:
        rewards = np.array([1.0, 0.0])
        values = np.array([0.0, 0.0])
        dones = np.array([False, True])
        advantages, returns = compute_gae(
            rewards, values, dones, gamma=0.99, lam=0.95
        )
        np.testing.assert_allclose(returns, advantages + values)

    def test_no_discount(self) -> None:
        rewards = np.array([1.0, 2.0, 3.0])
        values = np.array([0.0, 0.0, 0.0])
        dones = np.array([False, False, True])
        advantages, returns = compute_gae(rewards, values, dones, gamma=1.0, lam=1.0)
        np.testing.assert_allclose(advantages, [6.0, 5.0, 3.0])
        np.testing.assert_allclose(returns, [6.0, 5.0, 3.0])

    def test_all_done(self) -> None:
        rewards = np.array([1.0, 2.0])
        values = np.array([0.5, 0.5])
        dones = np.array([True, True])
        advantages, returns = compute_gae(
            rewards, values, dones, gamma=0.99, lam=0.95
        )
        np.testing.assert_allclose(advantages, [0.5, 1.5])
        np.testing.assert_allclose(returns, [1.0, 2.0])

    def test_advantages_plus_values_equal_returns(self) -> None:
        T = 15
        rng = np.random.RandomState(42)
        rewards = rng.randn(T)
        values = rng.randn(T)
        dones = np.zeros(T, dtype=bool)
        dones[5] = True
        dones[10] = True
        dones[-1] = True
        advantages, returns = compute_gae(
            rewards, values, dones, gamma=0.99, lam=0.95
        )
        np.testing.assert_allclose(returns, advantages + values)
