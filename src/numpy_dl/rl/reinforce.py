"""REINFORCE (Monte Carlo Policy Gradient) agent.

Creator references:
    Williams, R.J. (1992). "Simple statistical gradient-following
    algorithms for connectionist reinforcement learning."

Mathematical equations:
    ∇J(θ) = E[∇log π(a|s;θ) * G_t]
    where G_t = Σ_{k=t}^{T} γ^{k-t} * r_k
"""

from __future__ import annotations

import logging

import numpy as np

from numpy_dl.nn.activations import ReLU, softmax
from numpy_dl.nn.layers import Dense
from numpy_dl.nn.sequential import Sequential
from numpy_dl.optim.adam import Adam
from numpy_dl.rl.utils import categorical_sample, normalize_advantages

logger = logging.getLogger(__name__)


class REINFORCEAgent:
    """REINFORCE (Monte Carlo Policy Gradient) agent.

    Maintains a stochastic policy network and collects full episodes
    before performing a gradient update. On-policy: requires complete
    episode trajectories before training.

    Args:
        state_dim: Dimensionality of the state space.
        action_dim: Number of discrete actions.
        hidden_dim: Hidden layer size. Defaults to 64.
        lr: Learning rate. Defaults to 1e-3.
        gamma: Discount factor. Defaults to 0.99.
        seed: Random seed. Defaults to None.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        seed: int | None = None,
    ) -> None:
        self.action_dim = action_dim
        self.gamma = gamma
        self._rng = np.random.RandomState(seed)

        self.policy_net = Sequential(
            [
                Dense(state_dim, hidden_dim),
                ReLU(),
                Dense(hidden_dim, action_dim),
            ]
        )

        self._optimizer = Adam(self.policy_net.parameters(), lr=lr)

        self._states: list[np.ndarray] = []
        self._actions: list[int] = []
        self._rewards: list[float] = []

    def select_action(self, state: np.ndarray) -> int:
        """Select action by sampling from the policy distribution.

        Args:
            state: Current state observation of shape (state_dim,).

        Returns:
            Sampled action index.
        """
        state = np.atleast_2d(state)
        logits = self.policy_net.forward(state)
        probs = softmax(logits)[0]
        return categorical_sample(probs, self._rng)

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
    ) -> None:
        """Store a transition from the current episode.

        Args:
            state: State at which the action was taken.
            action: Action taken.
            reward: Reward received after taking the action.
        """
        self._states.append(state.copy())
        self._actions.append(action)
        self._rewards.append(reward)

    def _compute_returns(self) -> np.ndarray:
        """Compute discounted returns G_t = Σ_{k=0}^{T-t-1} γ^k * r_{t+k}.

        Returns:
            Array of shape (T,) with discounted return per timestep.
        """
        rewards = np.asarray(self._rewards, dtype=np.float64)
        T = rewards.shape[0]
        # Discount factor between timestep t and every later timestep k>=t:
        # D[t, k] = gamma^(k-t), else 0. Returns = D @ rewards.
        powers = np.arange(T)
        discount = np.where(
            powers[None, :] >= powers[:, None],
            self.gamma ** (powers[None, :] - powers[:, None]),
            0.0,
        )
        return discount @ rewards

    def train_step(self) -> float:
        """Compute policy gradient and update network parameters.

        Collects the stored trajectory, computes discounted returns,
        normalizes them, and performs a single policy gradient update.

        Returns:
            The mean discounted return of the episode (for logging).
        """
        if len(self._rewards) == 0:
            return 0.0

        returns = self._compute_returns()
        returns = normalize_advantages(returns)

        states = np.array(self._states)
        logits = self.policy_net.forward(states)
        probs = softmax(logits)

        T = len(self._actions)
        actions_arr = np.array(self._actions)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(T), actions_arr] = 1.0

        d_logits = -returns[:, np.newaxis] * (one_hot - probs)

        self.policy_net.backward(d_logits)
        self._optimizer.step()

        avg_return = float(np.mean(returns))
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        return avg_return

    def train(
        self, env: object, episodes: int = 500, max_steps: int = 10000
    ) -> list[float]:
        """Run the full REINFORCE training loop.

        Args:
            env: Environment with reset() -> state and step(action) -> (state, reward, done, info).
            episodes: Number of episodes to train.
            max_steps: Maximum steps per episode to prevent infinite loops.

        Returns:
            List of total rewards per episode.
        """
        rewards: list[float] = []
        for ep in range(episodes):
            state = env.reset()
            done = False
            total = 0.0
            steps = 0
            while not done and steps < max_steps:
                steps += 1
                action = self.select_action(state)
                result = env.step(action)
                if len(result) == 4:
                    next_state, reward, done, _info = result
                else:
                    next_state, reward, done = result
                self.store_transition(state, action, reward)
                state = next_state
                total += reward
            avg_return = self.train_step()
            if np.isnan(avg_return) or np.isinf(avg_return):
                logger.warning("NaN/inf return detected at episode %d; breaking", ep)
                break
            rewards.append(total)
        return rewards
