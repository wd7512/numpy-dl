"""Utility functions for reinforcement learning agents.

Creator references:
    Sutton, R.S. & Barto, A.G. (2018). "Reinforcement Learning:
    An Introduction." Chapter 2 — ε-greedy action selection.
    Williams, R.J. (1992). "Simple statistical gradient-following
    algorithms for connectionist reinforcement learning." — REINFORCE.

Mathematical equations:
    ε-greedy:  a = argmax_a Q(s,a) w.p. 1-ε, random w.p. ε
    Categorical: a ~ Categorical(p_1, ..., p_n)
    Normalize:  â = (a - μ) / (σ + ε)
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


def categorical_sample(probs: np.ndarray, rng: np.random.RandomState) -> int:
    """Sample an action from a categorical distribution.

    Args:
        probs: Probability array of shape (n_actions,) summing to 1.
        rng: NumPy RandomState for reproducibility.

    Returns:
        Sampled action index.
    """
    return int(rng.choice(len(probs), p=probs))


def normalize_advantages(advantages: np.ndarray) -> np.ndarray:
    """Normalize advantages to zero mean and unit variance.

    Args:
        advantages: Array of advantage or return values.

    Returns:
        Normalized array with mean ~0 and std ~1.
    """
    mean = np.mean(advantages)
    std = np.std(advantages)
    if std < 1e-8:
        return advantages - mean
    return (advantages - mean) / std
