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

    Storage uses preallocated NumPy arrays (one per field) sized to
    ``capacity``. Per-field shapes are inferred lazily from the first
    ``push``: subsequent pushes write into the slot at ``_pos`` in place,
    avoiding the list-of-tuples Python overhead the previous implementation
    carried. All pushes after the first must broadcast into the same shapes
    or a ``ValueError`` is raised — heterogeneously-shaped transitions are
    a usage error, not silently truncated.

    Args:
        capacity: Maximum number of transitions to store.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._size = 0
        self._pos = 0
        self._states: np.ndarray | None = None
        self._actions: np.ndarray | None = None
        self._rewards: np.ndarray | None = None
        self._next_states: np.ndarray | None = None
        self._dones: np.ndarray | None = None

    def _allocate(self, state: np.ndarray, next_state: np.ndarray) -> None:
        """Allocate the per-field arrays from the first transition's shapes."""
        state_arr = np.asarray(state)
        next_state_arr = np.asarray(next_state)
        self._states = np.zeros((self._capacity, *state_arr.shape), dtype=state_arr.dtype)
        self._next_states = np.zeros(
            (self._capacity, *next_state_arr.shape), dtype=next_state_arr.dtype
        )
        self._actions = np.zeros(self._capacity, dtype=np.int64)
        self._rewards = np.zeros(self._capacity, dtype=np.float64)
        self._dones = np.zeros(self._capacity, dtype=np.float64)

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
        if self._states is None:
            self._allocate(state, next_state)
        assert self._next_states is not None
        assert self._actions is not None
        assert self._rewards is not None
        assert self._dones is not None
        self._states[self._pos] = state
        self._next_states[self._pos] = next_state
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._dones[self._pos] = done
        self._pos = (self._pos + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of transitions uniformly at random.

        Returns:
            states:      (batch_size, *state_shape)
            actions:     (batch_size,)
            rewards:     (batch_size,)
            next_states: (batch_size, *state_shape)
            dones:       (batch_size,)
        """
        if self._states is None or self._size == 0:
            raise ValueError("ReplayBuffer.sample: buffer is empty")
        indices = np.random.choice(self._size, size=batch_size, replace=False)
        return (
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._dones[indices],
        )

    def __len__(self) -> int:
        """Current number of transitions in the buffer."""
        return self._size
