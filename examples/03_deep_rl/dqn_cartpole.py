"""Train a DQN agent on CartPole-v1.

Requires: pip install gymnasium
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    try:
        import gymnasium as gym
    except ImportError:
        logger.error("gymnasium is required for this example. Install with: pip install gymnasium")
        sys.exit(1)

    from numpy_dl.environments.gym_adapter import GymAdapter
    from numpy_dl.rl.dqn import DQNAgent

    env = GymAdapter(gym.make("CartPole-v1"))
    state_dim = 4
    action_dim = 2

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        seed=42,
    )

    episodes = 200
    rewards = agent.train(env, episodes=episodes)

    for i, r in enumerate(rewards):
        if i % 10 == 0:
            avg = sum(rewards[max(0, i - 9) : i + 1]) / min(10, i + 1)
            logger.info("Episode %d  reward=%.1f  avg_10=%.1f  eps=%.3f", i, r, avg, agent.epsilon)

    logger.info("Training complete. Final 10-episode avg: %.1f", sum(rewards[-10:]) / 10)


if __name__ == "__main__":
    main()
