"""Utility functions for reinforcement learning agents.

Creator references:
    Sutton, R.S. & Barto, A.G. (2018). "Reinforcement Learning:
    An Introduction." Chapter 2 — ε-greedy action selection.
    Williams, R.J. (1992). "Simple statistical gradient-following
    algorithms for connectionist reinforcement learning." — REINFORCE.
    Schulman, J., Moritz, P., Levine, S., Jordan, M. & Abbeel, P. (2015).
    "High-Dimensional Continuous Control Using Generalized Advantage
    Estimation." — GAE.

Mathematical equations:
    ε-greedy:  a = argmax_a Q(s,a) w.p. 1-ε, random w.p. ε
    Categorical: a ~ Categorical(p_1, ..., p_n)
    Normalize:  â = (a - μ) / (σ + ε)
    GAE:  Â_t = Σ_{l=0}^{T-t-1} (γλ)^l δ_{t+l}
    TD:   δ_t = r_t + γV(s_{t+1}) - V(s_t)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


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
        logger.warning("Advantage std near zero (%.2e), skipping normalization", std)
        return advantages - mean
    return (advantages - mean) / std


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation (GAE).

    Creator references:
        Schulman, J., Moritz, P., Levine, S., Jordan, M. & Abbeel, P. (2015).
        "High-Dimensional Continuous Control Using Generalized Advantage
        Estimation." — arXiv:1506.02438.

    Mathematical equations:
        δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        Â_t = Σ_{l=0}^{T-t-1} (γλ)^l δ_{t+l}

    Args:
        rewards: Array of shape (T,) with rewards per timestep.
        values: Array of shape (T,) with critic value estimates.
            values[t] is V(s_t). A terminal value of 0 is assumed.
        dones: Boolean array of shape (T,). True if the episode ended
            at timestep t.
        gamma: Discount factor ∈ [0, 1).
        lam: GAE lambda parameter ∈ [0, 1).

    Returns:
        Tuple of (advantages, returns) each of shape (T,).
    """
    T = len(rewards)
    advantages = np.zeros(T)
    gae = 0.0

    for t in reversed(range(T)):
        next_value = 0.0 if dones[t] or t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * (0.0 if dones[t] else gae)
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns
