"""Tests for DQN agent."""

import numpy as np

from numpy_dl.rl.dqn import DQNAgent


class TestDQNInit:
    def test_q_net_has_parameters(self) -> None:
        agent = DQNAgent(state_dim=4, action_dim=2, seed=0)
        assert len(agent.q_net.parameters()) > 0

    def test_target_net_synced(self) -> None:
        agent = DQNAgent(state_dim=4, action_dim=2, seed=0)
        q_params = agent.q_net.parameters()
        t_params = agent.target_net.parameters()
        for (qw, _), (tw, _) in zip(q_params, t_params):
            np.testing.assert_array_equal(qw, tw)

    def test_buffer_empty(self) -> None:
        agent = DQNAgent(state_dim=4, action_dim=2, seed=0)
        assert len(agent.buffer) == 0


class TestDQNSelectAction:
    def test_returns_valid_action(self) -> None:
        agent = DQNAgent(state_dim=4, action_dim=2, seed=0)
        state = np.random.randn(4)
        action = agent.select_action(state)
        assert 0 <= action < 2

    def test_greedy_when_epsilon_zero(self) -> None:
        agent = DQNAgent(state_dim=4, action_dim=3, seed=0)
        agent.epsilon = 0.0
        state = np.ones(4)
        q_vals = agent.q_net.forward(state.reshape(1, -1))[0]
        best = int(np.argmax(q_vals))
        for _ in range(50):
            assert agent.select_action(state) == best


class TestDQNTrainStep:
    def test_no_error_with_enough_data(self) -> None:
        agent = DQNAgent(state_dim=4, action_dim=2, buffer_size=100, seed=0)
        for _ in range(50):
            s = np.random.randn(4)
            agent.buffer.push(s, 0, 1.0, s, False)
        loss = agent.train_step(batch_size=16)
        assert isinstance(loss, float)

    def test_no_error_with_insufficient_data(self) -> None:
        agent = DQNAgent(state_dim=4, action_dim=2, seed=0)
        loss = agent.train_step(batch_size=32)
        assert loss == 0.0


class TestDQNTargetUpdate:
    def test_update_copies_weights(self) -> None:
        agent = DQNAgent(state_dim=4, action_dim=2, seed=0)
        for p, _ in agent.q_net.parameters():
            p[:] = 42.0
        agent.update_target()
        for (qw, _), (tw, _) in zip(
            agent.q_net.parameters(), agent.target_net.parameters()
        ):
            np.testing.assert_array_equal(qw, tw)
