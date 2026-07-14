"""Tests for PPO agent."""

import numpy as np

from numpy_dl.rl.ppo import PPOAgent
from numpy_dl.rl.utils import compute_gae


class TestPPOInit:
    def test_actor_has_parameters(self) -> None:
        agent = PPOAgent(state_dim=4, action_dim=2, seed=0)
        assert len(agent.actor.parameters()) > 0

    def test_critic_has_parameters(self) -> None:
        agent = PPOAgent(state_dim=4, action_dim=2, seed=0)
        assert len(agent.critic.parameters()) > 0

    def test_actor_critic_separate(self) -> None:
        agent = PPOAgent(state_dim=4, action_dim=2, seed=0)
        actor_params = [id(p) for p, _ in agent.actor.parameters()]
        critic_params = [id(p) for p, _ in agent.critic.parameters()]
        assert len(set(actor_params) & set(critic_params)) == 0

    def test_trajectory_empty(self) -> None:
        agent = PPOAgent(state_dim=4, action_dim=2, seed=0)
        assert len(agent._rewards) == 0
        assert len(agent._states) == 0


class TestPPOActorCriticShapes:
    def test_actor_output_shape(self) -> None:
        agent = PPOAgent(state_dim=4, action_dim=3, seed=0)
        state = np.random.randn(4)
        probs = agent._get_action_probs(state)
        assert probs.shape == (3,)

    def test_critic_output_shape(self) -> None:
        agent = PPOAgent(state_dim=4, action_dim=2, seed=0)
        state = np.random.randn(4)
        value = agent._get_value(state)
        assert isinstance(value, float)

    def test_actor_batch_forward(self) -> None:
        agent = PPOAgent(state_dim=4, action_dim=2, seed=0)
        states = np.random.randn(5, 4)
        logits = agent.actor.forward(states)
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        assert probs.shape == (5, 2)


class TestPPOSelectAction:
    def test_returns_valid_action(self) -> None:
        agent = PPOAgent(state_dim=4, action_dim=2, seed=0)
        state = np.random.randn(4)
        action, log_prob, value = agent.select_action(state)
        assert 0 <= action < 2
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_returns_tuple_of_three(self) -> None:
        agent = PPOAgent(state_dim=4, action_dim=3, seed=42)
        state = np.random.randn(4)
        result = agent.select_action(state)
        assert len(result) == 3

    def test_different_actions_possible(self) -> None:
        agent = PPOAgent(state_dim=4, action_dim=3, seed=42)
        actions = set()
        for _ in range(200):
            state = np.random.randn(4)
            action, _, _ = agent.select_action(state)
            actions.add(action)
        assert len(actions) > 1


class TestPPOTrainStep:
    def test_no_error_with_data(self) -> None:
        agent = PPOAgent(state_dim=4, action_dim=2, seed=0)
        for _ in range(20):
            action, log_prob, value = agent.select_action(np.random.randn(4))
            agent.store_transition(
                np.random.randn(4), action, log_prob, 1.0, value, False
            )
        avg_reward = agent.train_step()
        assert isinstance(avg_reward, float)

    def test_no_error_with_empty_trajectory(self) -> None:
        agent = PPOAgent(state_dim=4, action_dim=2, seed=0)
        avg_reward = agent.train_step()
        assert avg_reward == 0.0

    def test_trajectory_cleared_after_train(self) -> None:
        agent = PPOAgent(state_dim=4, action_dim=2, seed=0)
        for _ in range(10):
            action, log_prob, value = agent.select_action(np.random.randn(4))
            agent.store_transition(
                np.random.randn(4), action, log_prob, 1.0, value, False
            )
        agent.train_step()
        assert len(agent._rewards) == 0
        assert len(agent._states) == 0
        assert len(agent._actions) == 0
        assert len(agent._log_probs) == 0
        assert len(agent._values) == 0
        assert len(agent._dones) == 0

    def test_actor_weights_change(self) -> None:
        agent = PPOAgent(state_dim=4, action_dim=2, seed=0)
        weights_before = [p.copy() for p, _ in agent.actor.parameters()]
        for _ in range(20):
            action, log_prob, value = agent.select_action(np.random.randn(4))
            agent.store_transition(
                np.random.randn(4), action, log_prob, 1.0, value, False
            )
        agent.train_step()
        weights_after = [p.copy() for p, _ in agent.actor.parameters()]
        changed = any(
            not np.allclose(wb, wa) for wb, wa in zip(weights_before, weights_after)
        )
        assert changed

    def test_critic_weights_change(self) -> None:
        agent = PPOAgent(state_dim=4, action_dim=2, seed=0)
        weights_before = [p.copy() for p, _ in agent.critic.parameters()]
        for _ in range(20):
            action, log_prob, value = agent.select_action(np.random.randn(4))
            agent.store_transition(
                np.random.randn(4), action, log_prob, 1.0, value, False
            )
        agent.train_step()
        weights_after = [p.copy() for p, _ in agent.critic.parameters()]
        changed = any(
            not np.allclose(wb, wa) for wb, wa in zip(weights_before, weights_after)
        )
        assert changed


class TestPPOClippedObjective:
    def test_clipping_reduces_large_update(self) -> None:
        ratio = 1.5
        advantage = 10.0
        clip_eps = 0.2
        unclipped = ratio * advantage
        clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
        assert np.minimum(unclipped, clipped) == clipped

    def test_small_ratio_not_clipped(self) -> None:
        ratio = 0.95
        advantage = 1.0
        clip_eps = 0.2
        unclipped = ratio * advantage
        clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
        assert np.minimum(unclipped, clipped) == unclipped

    def test_negative_advantage_clips_differently(self) -> None:
        ratio = 1.5
        advantage = -1.0
        clip_eps = 0.2
        unclipped = ratio * advantage
        clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
        assert np.minimum(unclipped, clipped) == unclipped


class TestGAEComputation:
    def test_gae_single_step(self) -> None:
        rewards = np.array([1.0])
        values = np.array([0.5])
        dones = np.array([True])
        advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        np.testing.assert_allclose(advantages, [0.5])
        np.testing.assert_allclose(returns, [1.0])

    def test_gae_two_steps(self) -> None:
        rewards = np.array([1.0, 2.0])
        values = np.array([0.5, 1.0])
        dones = np.array([False, True])
        advantages, returns = compute_gae(rewards, values, dones, gamma=1.0, lam=1.0)
        # t=1: delta=2.0 + 1.0*0.0 - 1.0 = 1.0, gae=1.0 (terminal)
        # t=0: delta=1.0 + 1.0*1.0 - 0.5 = 1.5, gae=1.5 + 1.0*1.0*1.0 = 2.5
        np.testing.assert_allclose(advantages, [2.5, 1.0])
        np.testing.assert_allclose(returns, advantages + values)

    def test_gae_with_done_in_middle(self) -> None:
        rewards = np.array([1.0, 2.0, 3.0])
        values = np.array([0.5, 1.0, 1.5])
        dones = np.array([False, True, False])
        advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        delta_1 = 2.0 + 0.99 * 0.0 - 1.0
        adv_1 = delta_1
        delta_2 = 3.0 + 0.99 * 0.0 - 1.5
        adv_2 = delta_2
        delta_0 = 1.0 + 0.99 * values[1] - 0.5
        adv_0 = delta_0 + 0.99 * 0.95 * adv_1
        np.testing.assert_allclose(advantages, [adv_0, adv_1, adv_2])
        np.testing.assert_allclose(returns, advantages + values)
