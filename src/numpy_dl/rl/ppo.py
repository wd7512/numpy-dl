"""Proximal Policy Optimization (PPO) agent.

Creator references:
    Schulman, J., Wolski, F., Dhariwal, P., Radford, A. & Klimov, O. (2017).
    "Proximal Policy Optimization Algorithms." — arXiv:1707.06347.
    Schulman, J., Moritz, P., Levine, S., Jordan, M. & Abbeel, P. (2015).
    "High-Dimensional Continuous Control Using Generalized Advantage
    Estimation."
    Mnih, V., Badia, A.P., Mirza, M. et al. (2016). "Asynchronous Methods
    for Deep Reinforcement Learning." — A2C baseline.

Mathematical equations:
    Clipped surrogate: L_CLIP(θ) = E[min(r_t(θ)Â_t,
                                        clip(r_t(θ), 1-ε, 1+ε)Â_t)]
    Importance ratio:  r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)
    Value loss:        L_V = (V_θ(s) - V_target)^2
    Entropy bonus:     H = -Σ π(a|s) log π(a|s)
"""

from __future__ import annotations

import logging

import numpy as np

from numpy_dl.nn.activations import ReLU, softmax
from numpy_dl.nn.layers import Dense
from numpy_dl.nn.sequential import Sequential
from numpy_dl.optim.adam import Adam
from numpy_dl.rl.utils import categorical_sample, compute_gae, normalize_advantages

logger = logging.getLogger(__name__)


class PPOAgent:
    """Proximal Policy Optimization agent with clipped surrogate objective.

    Uses separate actor and critic networks. Collects full episode
    trajectories, computes GAE advantages, and performs multiple epochs
    of mini-batch PPO updates.

    Args:
        state_dim: Dimensionality of the state space.
        action_dim: Number of discrete actions.
        hidden_dim: Hidden layer size. Defaults to 64.
        lr_actor: Actor learning rate. Defaults to 3e-4.
        lr_critic: Critic learning rate. Defaults to 1e-3.
        gamma: Discount factor. Defaults to 0.99.
        lam: GAE lambda. Defaults to 0.95.
        clip_eps: PPO clip epsilon. Defaults to 0.2.
        epochs: Number of PPO epochs per update. Defaults to 10.
        batch_size: Mini-batch size. Defaults to 64.
        entropy_coeff: Entropy bonus coefficient. Defaults to 0.01.
        value_coeff: Value loss coefficient. Defaults to 0.5.
        seed: Random seed. Defaults to None.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 10,
        batch_size: int = 64,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self._rng = np.random.RandomState(seed)

        self.actor = Sequential(
            [
                Dense(state_dim, hidden_dim),
                ReLU(),
                Dense(hidden_dim, action_dim),
            ]
        )

        self.critic = Sequential(
            [
                Dense(state_dim, hidden_dim),
                ReLU(),
                Dense(hidden_dim, 1),
            ]
        )

        self._actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self._critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)

        self._states: list[np.ndarray] = []
        self._actions: list[int] = []
        self._log_probs: list[float] = []
        self._rewards: list[float] = []
        self._values: list[float] = []
        self._dones: list[bool] = []

    def _get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """Compute softmax action probabilities for a single state.

        Args:
            state: State of shape (state_dim,).

        Returns:
            Probability array of shape (action_dim,).
        """
        state = np.atleast_2d(state)
        logits = self.actor.forward(state)
        return softmax(logits)[0]

    def _get_value(self, state: np.ndarray) -> float:
        """Compute value estimate for a single state.

        Args:
            state: State of shape (state_dim,).

        Returns:
            Scalar value estimate.
        """
        state = np.atleast_2d(state)
        return float(self.critic.forward(state)[0, 0])

    def select_action(self, state: np.ndarray) -> tuple[int, float, float]:
        """Select an action and return action, log_prob, and value.

        Args:
            state: Current state observation of shape (state_dim,).

        Returns:
            Tuple of (action, log_probability, value_estimate).
        """
        probs = self._get_action_probs(state)
        action = categorical_sample(probs, self._rng)
        log_prob = float(np.log(probs[action] + 1e-8))
        value = self._get_value(state)
        return action, log_prob, value

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        """Store a transition from the current episode.

        Args:
            state: State at which the action was taken.
            action: Action taken.
            log_prob: Log probability of the action under the current policy.
            reward: Reward received after taking the action.
            value: Critic value estimate V(s).
            done: Whether the episode ended after this step.
        """
        self._states.append(state.copy())
        self._actions.append(action)
        self._log_probs.append(log_prob)
        self._rewards.append(reward)
        self._values.append(value)
        self._dones.append(done)

    def train_step(self) -> float:
        """Compute GAE and perform PPO clipped updates.

        Computes advantages and returns from the stored trajectory,
        then runs multiple epochs of mini-batch gradient updates on
        both actor and critic networks.

        Returns:
            The mean episode reward (for logging).
        """
        if len(self._rewards) == 0:
            return 0.0

        states = np.array(self._states)
        actions = np.array(self._actions)
        old_log_probs = np.array(self._log_probs)
        rewards_arr = np.array(self._rewards)
        values_arr = np.array(self._values)
        dones = np.array(self._dones)

        advantages, returns = compute_gae(
            rewards_arr, values_arr, dones, self.gamma, self.lam
        )
        advantages = normalize_advantages(advantages)

        T = len(rewards_arr)
        indices = np.arange(T)
        avg_reward = float(np.mean(rewards_arr))

        for _ in range(self.epochs):
            self._rng.shuffle(indices)

            for start in range(0, T, self.batch_size):
                end = min(start + self.batch_size, T)
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_size_actual = len(batch_idx)

                logits = self.actor.forward(batch_states)
                probs = softmax(logits)
                log_probs = np.log(probs + 1e-8)
                taken_log_probs = log_probs[np.arange(batch_size_actual), batch_actions]
                old_taken_log_probs = batch_old_log_probs

                ratio = np.exp(taken_log_probs - old_taken_log_probs)
                unclipped = ratio * batch_advantages
                clipped = np.clip(
                    ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                ) * batch_advantages

                if np.isnan(unclipped).any() or np.isinf(unclipped).any():
                    logger.warning(
                        "NaN/inf detected in PPO surrogate (epoch loop); "
                        "aborting remaining epochs to avoid corrupting weights."
                    )
                    self._states.clear()
                    self._actions.clear()
                    self._log_probs.clear()
                    self._rewards.clear()
                    self._values.clear()
                    self._dones.clear()
                    return avg_reward

                one_hot = np.zeros_like(probs)
                one_hot[np.arange(batch_size_actual), batch_actions] = 1.0
                is_unclipped = (unclipped <= clipped).astype(np.float64)

                entropy = -np.sum(probs * log_probs, axis=-1)
                # Per-row entropy H_t (shape (B,)) is required by the
                # softmax-CE-style derivative dH_t/dlogits[t,k] = -p[t,k] * (log p[t,k] + H_t).
                # Previously this used scalar mean_entropy, which FD showed does not match
                # the actual derivative (see tests/rl/test_gradients.py and BUGS.md).

                surrogate_grad = (
                    -batch_advantages[:, np.newaxis]
                    * ratio[:, np.newaxis]
                    * is_unclipped[:, np.newaxis]
                    * (one_hot - probs)
                )
                entropy_grad = (
                    self.entropy_coeff
                    * probs
                    * (log_probs + entropy[:, np.newaxis])
                )
                actor_grad = surrogate_grad + entropy_grad

                self.actor.backward(actor_grad)
                self._actor_optimizer.step()

                values = self.critic.forward(batch_states)
                value_grad = (
                    2.0
                    * self.value_coeff
                    * (values[:, 0] - batch_returns)
                    / batch_size_actual
                )
                self.critic.backward(value_grad[:, np.newaxis])
                self._critic_optimizer.step()

        self._states.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._rewards.clear()
        self._values.clear()
        self._dones.clear()

        return avg_reward

    def train(self, env: object, episodes: int = 500, max_steps: int = 10000) -> list[float]:
        """Full training loop on an environment.

        Args:
            env: Environment with reset() -> state and
                step(action) -> (state, reward, done, info).
            episodes: Number of episodes to train.
            max_steps: Maximum steps per episode to prevent infinite loops.

        Returns:
            List of total rewards per episode.
        """
        rewards: list[float] = []
        for _ in range(episodes):
            state = env.reset()
            total = 0.0
            done = False
            steps = 0
            while not done and steps < max_steps:
                steps += 1
                action, log_prob, value = self.select_action(state)
                result = env.step(action)
                if len(result) == 4:
                    next_state, reward, done, _info = result
                else:
                    next_state, reward, done = result
                self.store_transition(state, action, log_prob, reward, value, done)
                state = next_state
                total += reward
            self.train_step()
            rewards.append(total)
        return rewards
