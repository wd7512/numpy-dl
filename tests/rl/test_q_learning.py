"""Tests for Q-Learning agent."""

import numpy as np

from numpy_dl.rl.q_learning import QLearningAgent


class TestQLearningInit:
    def test_q_table_shape(self) -> None:
        agent = QLearningAgent(n_states=16, n_actions=4, seed=0)
        assert agent.q_table.shape == (16, 4)
        assert np.all(agent.q_table == 0)


class TestEpsilonGreedy:
    def test_greedy_selects_best_action(self) -> None:
        agent = QLearningAgent(n_states=2, n_actions=3, seed=0)
        agent.q_table[0] = [0.1, 0.9, 0.3]
        for _ in range(100):
            assert agent.select_action(0, epsilon=0.0) == 1

    def test_random_action_with_high_epsilon(self) -> None:
        agent = QLearningAgent(n_states=2, n_actions=4, seed=0)
        actions = {agent.select_action(0, epsilon=1.0) for _ in range(200)}
        assert len(actions) > 1


class TestQLearningUpdate:
    def test_update_q_value(self) -> None:
        agent = QLearningAgent(n_states=2, n_actions=2, lr=0.1, gamma=0.99, seed=0)
        td = agent.update(state=0, action=0, reward=1.0, next_state=1, done=True)
        expected = 0 + 0.1 * (1.0 + 0 - 0)
        np.testing.assert_almost_equal(agent.q_table[0, 0], expected)
        np.testing.assert_almost_equal(td, 1.0)

    def test_update_with_discount(self) -> None:
        agent = QLearningAgent(n_states=2, n_actions=2, lr=0.1, gamma=0.9, seed=0)
        agent.q_table[1, 0] = 5.0
        agent.update(state=0, action=0, reward=1.0, next_state=1, done=False)
        expected = 0 + 0.1 * (1.0 + 0.9 * 5.0 - 0)
        np.testing.assert_almost_equal(agent.q_table[0, 0], expected)


class TestTrainingLoop:
    def test_train_runs(self) -> None:
        from numpy_dl.environments.grid_world import GridWorld

        env = GridWorld(n=4, seed=0)
        agent = QLearningAgent(n_states=16, n_actions=4, seed=42)
        rewards = agent.train(env, episodes=10)
        assert len(rewards) == 10
        assert all(isinstance(r, float) for r in rewards)
