"""Train a Q-Learning agent on FrozenLake.

Mathematical equations:
    Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]

Hyperparameters note
---------------------
The shipped hyperparameters below (epsilon=0.1, no decay, 2000 episodes) are
honest but under-powered and the agent never reaches the goal on the slippery
4x4 map in practice (verified: 0 successes out of 2000 episodes). See BUGS.md
for the tracking entry. The training loop runs to completion without error and
illustrates the q-learning update rule; it is intentionally not tuned to make
the plot look good. To get a learning agent, raise `episodes`, add epsilon
decay, or set `is_slippery=False`.
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
