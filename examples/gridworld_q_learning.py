"""Train a Q-Learning agent on GridWorld.

Mathematical equations:
    Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]
"""

import logging

from numpy_dl.environments.grid_world import GridWorld
from numpy_dl.rl.q_learning import QLearningAgent

logging.basicConfig(level=logging.INFO, format="Episode %(d)s | Reward: %(f).1f")


def main() -> None:
    env = GridWorld(n=4, seed=0)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        lr=0.1,
        gamma=0.99,
        epsilon=0.1,
        seed=42,
    )
    rewards = agent.train(env, episodes=500)
    for ep, r in enumerate(rewards):
        logging.info("Episode %d | Reward: %.1f", ep, r)
    logging.info("Average reward (last 50): %.2f", sum(rewards[-50:]) / 50)


if __name__ == "__main__":
    main()
