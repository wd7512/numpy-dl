"""Adapter wrapping Gymnasium environments to the numpy-dl minimal API.

Creator references:
    OpenAI Gymnasium — https://gymnasium.farama.org/

Converts the Gymnasium (v0.26+) API:
    reset() -> (state, info)
    step(action) -> (state, reward, terminated, truncated, info)

To the numpy-dl minimal API:
    reset() -> state
    step(action) -> (state, reward, done, info)
"""

from __future__ import annotations

import numpy as np


class GymAdapter:
    """Wraps a Gymnasium environment to match the numpy-dl interface.

    Args:
        env: A Gymnasium environment instance.
    """

    def __init__(self, env: object) -> None:
        self._env = env

    def reset(self) -> np.ndarray:
        """Reset the environment and return the observation as a numpy array."""
        result = self._env.reset()
        if isinstance(result, tuple):
            obs = result[0]
        else:
            obs = result
        return np.asarray(obs)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Step the environment.

        Returns:
            (state, reward, done, info) where done = terminated or truncated.
        """
        obs, reward, terminated, truncated, _ = self._env.step(action)
        return np.asarray(obs), float(reward), terminated or truncated, {}
