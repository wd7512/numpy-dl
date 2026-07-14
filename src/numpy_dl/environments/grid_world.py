"""GridWorld environment for reinforcement learning.

Creator references:
    Sutton, R.S. & Barto, A.G. (2018). "Reinforcement Learning: An Introduction."
    Chapter 1 — Gridworld examples.

Mathematical equations:
    State:  s ∈ {0, 1, ..., n*n-1}  (row-major encoding of n×n grid)
    Actions: a ∈ {0, 1, 2, 3} → {up, right, down, left}
    Transition: s' = clamp(s + Δ(a), 0, n*n-1)  (walls block movement)
    Reward: R(s, a) = -1 if s' ≠ goal, +10 if s' = goal
"""

from __future__ import annotations

import numpy as np


class GridWorld:
    """n×n gridworld where the agent navigates to a goal cell.

    Args:
        n: Grid side length.
        goal: Goal cell index (row * n + col). Defaults to bottom-right.
        seed: Random seed for reproducibility.
    """

    ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

    def __init__(self, n: int = 4, goal: int | None = None, seed: int | None = None) -> None:
        self.n = n
        self.n_states = n * n
        self.n_actions = 4
        self.goal = goal if goal is not None else n * n - 1
        self._rng = np.random.RandomState(seed)
        self._state = 0

    def reset(self, state: int = 0) -> int:
        """Reset environment and return initial state."""
        self._state = state
        return self._state

    def step(self, action: int) -> tuple[int, float, bool]:
        """Take one action and return (next_state, reward, done)."""
        dr, dc = self.ACTIONS[action]
        row, col = divmod(self._state, self.n)
        row = max(0, min(self.n - 1, row + dr))
        col = max(0, min(self.n - 1, col + dc))
        self._state = row * self.n + col
        done = self._state == self.goal
        reward = 10.0 if done else -1.0
        return self._state, reward, done
