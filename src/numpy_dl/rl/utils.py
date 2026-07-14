"""Utility functions for reinforcement learning agents.

Creator references:
    Sutton, R.S. & Barto, A.G. (2018). "Reinforcement Learning:
    An Introduction." Chapter 2 — ε-greedy action selection.

Mathematical equations:
    ε-greedy: a = argmax_a Q(s,a) w.p. 1-ε, random w.p. ε
"""

from __future__ import annotations

import numpy as np


def epsilon_greedy(q_values: np.ndarray, epsilon: float, rng: np.random.RandomState) -> int:
    """Select an action using ε-greedy policy.

    Args:
        q_values: Array of Q-values for each action.
        epsilon: Probability of choosing a random action.
        rng: NumPy RandomState for reproducibility.

    Returns:
        Selected action index.
    """
    if rng.random() < epsilon:
        return int(rng.randint(0, len(q_values)))
    return int(np.argmax(q_values))
