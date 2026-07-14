"""FrozenLake environment for reinforcement learning.

Creator references:
    OpenAI Gymnasium — FrozenLake-v1
    Sutton, R.S. & Barto, A.G. (2018). "Reinforcement Learning: An Introduction."

Mathematical equations:
    State:  s ∈ {0, 1, ..., 15}  (4×4 grid, row-major)
    Actions: a ∈ {0, 1, 2, 3} → {left, down, right, up}
    Transition: P(s'|s,a) — stochastic when is_slippery=True
    Reward: R(s, a) = +1 if s = goal, 0 otherwise
"""

from __future__ import annotations

import numpy as np


class FrozenLake:
    """4×4 FrozenLake with stochastic transitions.

    Grid layout (default):
        S F F F
        F H F H
        F F F H
        H F F G

    S=start, F=frozen, H=hole, G=goal.

    Args:
        is_slippery: If True, transitions are stochastic (8 directions).
        seed: Random seed.
    """

    MAP = [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    ]

    ACTIONS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

    def __init__(self, is_slippery: bool = True, seed: int | None = None) -> None:
        self.n_states = 16
        self.n_actions = 4
        self.is_slippery = is_slippery
        self._rng = np.random.RandomState(seed)
        self._state = 0
        self._desc = [list(row) for row in self.MAP]

    def _tile(self, r: int, c: int) -> str:
        return self._desc[r][c]

    def reset(self, state: int = 0) -> int:
        """Reset environment and return initial state."""
        self._state = state
        return self._state

    def step(self, action: int) -> tuple[int, float, bool]:
        """Take one action and return (next_state, reward, done)."""
        if self.is_slippery:
            effect = self._rng.choice([action, (action + 1) % 4, (action + 3) % 4])
        else:
            effect = action
        dr, dc = self.ACTIONS[effect]
        r, c = divmod(self._state, 4)
        nr = max(0, min(3, r + dr))
        nc = max(0, min(3, c + dc))
        self._state = nr * 4 + nc
        tile = self._tile(nr, nc)
        done = tile == "G" or tile == "H"
        reward = 1.0 if tile == "G" else 0.0
        return self._state, reward, done
