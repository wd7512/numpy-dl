"""Experience replay buffer for off-policy reinforcement learning.

Creator references:
    Lin, L.J. (1993). "Reinforcement learning with experience replay."
    Mnih, V. et al. (2015). "Human-level control through deep
    reinforcement learning." — Experience replay in DQN.

Mathematical equations:
    Buffer stores tuples: (s, a, r, s', done)
    Sampled uniformly at random: batch ~ Uniform(0, |B|)
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Fixed-size circular replay buffer storing transition tuples.

    Transitions are stored as (state, action, reward, next_state, done).
    When the buffer is full, the oldest transition is overwritten.

    Args:
        capacity: Maximum number of transitions to store.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._buffer: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self._pos: int = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the buffer.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode ended.
        """
        transition = (
            np.array(state),
            int(action),
            float(reward),
            np.array(next_state),
            bool(done),
        )
        if len(self._buffer) < self._capacity:
            self._buffer.append(transition)
        else:
            self._buffer[self._pos] = transition
        self._pos = (self._pos + 1) % self._capacity

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of transitions uniformly at random.

        Returns:
            states:      (batch_size, state_dim)
            actions:     (batch_size,)
            rewards:     (batch_size,)
            next_states: (batch_size, state_dim)
            dones:       (batch_size,)
        """
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float64)
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch], dtype=np.float64)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Current number of transitions in the buffer."""
        return len(self._buffer)
