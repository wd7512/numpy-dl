"""Train a Q-Learning agent on FrozenLake.

Mathematical equations:
    Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]
"""

import logging

from numpy_dl.environments.frozen_lake import FrozenLake
from numpy_dl.rl.q_learning import QLearningAgent

logging.basicConfig(level=logging.INFO, format="Episode %(d)s | Reward: %(f).1f")


def main() -> None:
    env = FrozenLake(is_slippery=True, seed=0)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        lr=0.1,
        gamma=0.99,
        epsilon=0.1,
        seed=42,
    )
    rewards = agent.train(env, episodes=2000)
    for ep, r in enumerate(rewards):
        logging.info("Episode %d | Reward: %.1f", ep, r)
    logging.info("Average reward (last 100): %.2f", sum(rewards[-100:]) / 100)


if __name__ == "__main__":
    main()
