"""Train PPO on LunarLander-v3.

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

    env = GymAdapter(gym.make("LunarLander-v3"))

    agent = PPOAgent(
        state_dim=8,
        action_dim=4,
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

    episodes = 1000
    rewards = agent.train(env, episodes=episodes)

    for i in range(0, episodes, 100):
        avg = sum(rewards[i : i + 100]) / 100
        logger.info("Episode %d-%d  avg_reward=%.1f", i, i + 99, avg)

    logger.info(
        "Training complete. Final 100-episode avg: %.1f",
        sum(rewards[-100:]) / 100,
    )


if __name__ == "__main__":
    main()
