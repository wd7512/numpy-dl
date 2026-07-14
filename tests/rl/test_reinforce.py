"""Tests for REINFORCE agent."""

import numpy as np

from numpy_dl.rl.reinforce import REINFORCEAgent
from numpy_dl.rl.utils import categorical_sample, normalize_advantages


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


class TestREINFORCEInit:
    def test_policy_net_has_parameters(self) -> None:
        agent = REINFORCEAgent(state_dim=4, action_dim=2, seed=0)
        assert len(agent.policy_net.parameters()) > 0

    def test_trajectory_empty(self) -> None:
        agent = REINFORCEAgent(state_dim=4, action_dim=2, seed=0)
        assert len(agent._rewards) == 0


class TestREINFORCESelectAction:
    def test_returns_valid_action(self) -> None:
        agent = REINFORCEAgent(state_dim=4, action_dim=2, seed=0)
        state = np.random.randn(4)
        action = agent.select_action(state)
        assert 0 <= action < 2

    def test_different_actions_possible(self) -> None:
        agent = REINFORCEAgent(state_dim=4, action_dim=3, seed=42)
        actions = set()
        for _ in range(200):
            state = np.random.randn(4)
            actions.add(agent.select_action(state))
        assert len(actions) > 1


class TestREINFORCETrainStep:
    def test_no_error_with_data(self) -> None:
        agent = REINFORCEAgent(state_dim=4, action_dim=2, seed=0)
        for _ in range(10):
            agent.store_transition(np.random.randn(4), 0, 0.0, 1.0)
        avg_return = agent.train_step()
        assert isinstance(avg_return, float)

    def test_no_error_with_empty_trajectory(self) -> None:
        agent = REINFORCEAgent(state_dim=4, action_dim=2, seed=0)
        avg_return = agent.train_step()
        assert avg_return == 0.0

    def test_policy_weights_change(self) -> None:
        agent = REINFORCEAgent(state_dim=4, action_dim=2, seed=0)
        weights_before = [p.copy() for p, _ in agent.policy_net.parameters()]
        for _ in range(10):
            agent.store_transition(np.random.randn(4), 0, 0.0, 1.0)
        agent.train_step()
        weights_after = [p.copy() for p, _ in agent.policy_net.parameters()]
        changed = any(
            not np.allclose(wb, wa) for wb, wa in zip(weights_before, weights_after)
        )
        assert changed

    def test_trajectory_cleared_after_train(self) -> None:
        agent = REINFORCEAgent(state_dim=4, action_dim=2, seed=0)
        for _ in range(5):
            agent.store_transition(np.random.randn(4), 1, 0.0, 1.0)
        agent.train_step()
        assert len(agent._rewards) == 0
        assert len(agent._states) == 0
        assert len(agent._actions) == 0


class TestREINFORCEComputeReturns:
    def test_single_reward(self) -> None:
        agent = REINFORCEAgent(state_dim=2, action_dim=2, gamma=0.99, seed=0)
        agent._rewards = [5.0]
        returns = agent._compute_returns()
        np.testing.assert_allclose(returns, [5.0])

    def test_discounted_chain(self) -> None:
        agent = REINFORCEAgent(state_dim=2, action_dim=2, gamma=0.5, seed=0)
        agent._rewards = [1.0, 2.0, 3.0]
        returns = agent._compute_returns()
        # G_2 = 3.0
        # G_1 = 2.0 + 0.5 * 3.0 = 3.5
        # G_0 = 1.0 + 0.5 * 3.5 = 2.75
        np.testing.assert_allclose(returns, [2.75, 3.5, 3.0])
