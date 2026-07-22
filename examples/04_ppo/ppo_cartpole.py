"""Train PPO on CartPole-v1.

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
        logger.error(
            "gymnasium is required for this example. Install with: pip install gymnasium"
        )
        sys.exit(1)

    from numpy_dl.environments.gym_adapter import GymAdapter
    from numpy_dl.rl.ppo import PPOAgent

    env = GymAdapter(gym.make("CartPole-v1"))

    agent = PPOAgent(
        state_dim=4,
        action_dim=2,
        hidden_dim=64,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        epochs=10,
        batch_size=64,
        seed=0,
    )

    episodes = 500
    rewards = agent.train(env, episodes=episodes)

    for i in range(0, episodes, 50):
        avg = sum(rewards[i : i + 50]) / 50
        logger.info("Episode %d-%d  avg_reward=%.1f", i, i + 49, avg)

    logger.info(
        "Training complete. Final 50-episode avg: %.1f",
        sum(rewards[-50:]) / 50,
    )


if __name__ == "__main__":
    main()
