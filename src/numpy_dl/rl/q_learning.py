"""Tabular Q-Learning agent.

Creator references:
    Watkins, C.J.C.H. & Dayan, P. (1992). "Q-Learning." Machine Learning, 8(3-4), 279-292.
    Sutton, R.S. & Barto, A.G. (2018). "Reinforcement Learning: An Introduction."
    Chapter 6 — Temporal-Difference Learning.

Mathematical equations:
    Q-Learning update rule:
        Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]

    ε-greedy action selection:
        a = argmax_a Q(s,a)   with probability 1 - ε
        a = uniform(0, |A|-1)  with probability ε
"""

from __future__ import annotations

import numpy as np


class QLearningAgent:
    """Tabular Q-Learning agent with ε-greedy exploration.

    Args:
        n_states: Number of discrete states.
        n_actions: Number of discrete actions.
        lr: Learning rate α.
        gamma: Discount factor γ.
        epsilon: Initial exploration rate ε.
        seed: Random seed.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self._rng = np.random.RandomState(seed)
        self.q_table = np.zeros((n_states, n_actions))

    def select_action(self, state: int, epsilon: float | None = None) -> int:
        """Select action via ε-greedy policy."""
        eps = epsilon if epsilon is not None else self.epsilon
        if self._rng.random() < eps:
            return int(self._rng.randint(0, self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def update(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ) -> float:
        """Apply Q-Learning update. Returns the TD error."""
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        return td_error

    def train(self, env: object, episodes: int = 500) -> list[float]:
        """Train agent on an environment with reset()/step() interface.

        Args:
            env: Environment with reset() -> int, step(a) -> (s, r, done).
            episodes: Number of episodes to train.

        Returns:
            List of total rewards per episode.
        """
        rewards: list[float] = []
        for _ in range(episodes):
            state = env.reset()  # type: ignore[union-attr]
            total = 0.0
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done = env.step(action)  # type: ignore[union-attr]
                self.update(state, action, reward, next_state, done)
                state = next_state
                total += reward
            rewards.append(total)
        return rewards
