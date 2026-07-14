"""Deep Q-Network (DQN) agent with experience replay and target network.

Creator references:
    Mnih, V. et al. (2015). "Human-level control through deep
    reinforcement learning." — DQN with replay buffer and target network.
    Lin, L.J. (1993). "Reinforcement learning with experience replay."

Mathematical equations:
    L = E[(r + γ max_a' Q_target(s',a') - Q(s,a))^2]
    Q-target update: θ_target ← θ_q  (periodic hard copy)
"""

from __future__ import annotations

import logging

import numpy as np

from numpy_dl.memory.replay_buffer import ReplayBuffer
from numpy_dl.nn.activations import ReLU
from numpy_dl.nn.layers import Dense
from numpy_dl.nn.sequential import Sequential
from numpy_dl.rl.utils import epsilon_greedy

logger = logging.getLogger(__name__)


class DQNAgent:
    """Deep Q-Network agent with replay buffer and target network.

    Args:
        state_dim: Dimensionality of the state space.
        action_dim: Number of discrete actions.
        hidden_dim: Hidden layer size. Defaults to 128.
        lr: Learning rate. Defaults to 1e-3.
        gamma: Discount factor. Defaults to 0.99.
        epsilon: Initial exploration rate. Defaults to 1.0.
        epsilon_min: Minimum exploration rate. Defaults to 0.01.
        epsilon_decay: Multiplicative decay per episode. Defaults to 0.995.
        buffer_size: Replay buffer capacity. Defaults to 10000.
        seed: Random seed. Defaults to None.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        seed: int | None = None,
    ) -> None:
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._rng = np.random.RandomState(seed)

        self.q_net = Sequential(
            [
                Dense(state_dim, hidden_dim),
                ReLU(),
                Dense(hidden_dim, hidden_dim),
                ReLU(),
                Dense(hidden_dim, action_dim),
            ]
        )
        self.target_net = Sequential(
            [
                Dense(state_dim, hidden_dim),
                ReLU(),
                Dense(hidden_dim, hidden_dim),
                ReLU(),
                Dense(hidden_dim, action_dim),
            ]
        )
        self.update_target()

        self._q_params = self.q_net.parameters()
        self._optimizer_m = [np.zeros_like(p) for p, _ in self._q_params]
        self._optimizer_v = [np.ones_like(p) for p, _ in self._q_params]
        self._optimizer_t = 0

        self.buffer = ReplayBuffer(buffer_size)

    def _sync_target(self) -> None:
        """Copy q_net weights into target_net."""
        q_layers = [layer for layer in self.q_net._layers if isinstance(layer, Dense)]
        t_layers = [layer for layer in self.target_net._layers if isinstance(layer, Dense)]
        for q_layer, t_layer in zip(q_layers, t_layers):
            t_layer.W[:] = q_layer.W
            t_layer.b[:] = q_layer.b

    def select_action(self, state: np.ndarray) -> int:
        """Select action via ε-greedy using q_net."""
        state = np.atleast_2d(state)
        q_values = self.q_net.forward(state)
        return epsilon_greedy(q_values[0], self.epsilon, self._rng)

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self, batch_size: int = 32) -> float:
        """Sample a batch and update q_net via MSE loss.

        Returns:
            The scalar training loss.
        """
        if len(self.buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        q_values = self.q_net.forward(states)
        q_a = q_values[np.arange(batch_size), actions]

        next_q = self.target_net.forward(next_states)
        max_next_q = np.max(next_q, axis=1)
        targets = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = float(np.mean((q_a - targets) ** 2))
        q_grad = 2.0 * (q_a - targets) / batch_size

        full_grad = np.zeros_like(q_values)
        full_grad[np.arange(batch_size), actions] = q_grad

        self.q_net.backward(full_grad)
        self._optimizer_t += 1
        bc1 = 1.0 - 0.9**self._optimizer_t
        bc2 = 1.0 - 0.999**self._optimizer_t
        for i, (param, _grad) in enumerate(self._q_params):
            self._optimizer_m[i] = 0.9 * self._optimizer_m[i] + 0.1 * _grad
            self._optimizer_v[i] = 0.999 * self._optimizer_v[i] + 0.001 * _grad**2
            m_hat = self._optimizer_m[i] / bc1
            v_hat = self._optimizer_v[i] / bc2
            param -= 1e-3 * m_hat / (np.sqrt(np.abs(v_hat)) + 1e-8)

        return float(loss)

    def update_target(self) -> None:
        """Copy q_net weights to target_net."""
        self._sync_target()

    def train(self, env: object, episodes: int = 1000, max_steps: int = 10000) -> list[float]:
        """Run the full DQN training loop.

        Args:
            env: Environment with reset() -> state and step(action) -> (state, reward, done, info).
            episodes: Number of episodes to train.
            max_steps: Maximum steps per episode to prevent infinite loops.

        Returns:
            List of total rewards per episode.
        """
        rewards: list[float] = []
        for ep in range(episodes):
            state = env.reset()
            total = 0.0
            done = False
            steps = 0
            while not done and steps < max_steps:
                steps += 1
                action = self.select_action(state)
                result = env.step(action)
                if len(result) == 4:
                    next_state, reward, done, _info = result
                else:
                    next_state, reward, done = result
                self.store_transition(state, action, reward, next_state, done)
                loss = self.train_step()
                if np.isnan(loss) or np.isinf(loss):
                    logger.warning("NaN/inf loss detected at episode %d, step; breaking", ep)
                    break
                state = next_state
                total += reward
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if ep % 10 == 0:
                self.update_target()
            rewards.append(total)
        return rewards
