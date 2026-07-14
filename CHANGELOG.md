# Changelog

All notable changes to NumPy-DL will be documented in this file.

## [0.6.0] - 2026-07-14

### Added
- **PPO (Proximal Policy Optimization)** with actor-critic architecture
  - Separate actor and critic networks
  - Clipped surrogate objective (ε=0.2)
  - Generalized Advantage Estimation (GAE, λ=0.95)
  - Entropy bonus for exploration
  - Multi-epoch mini-batch updates
- PPO training examples for CartPole-v1 and LunarLander-v3
- GAE computation utility (`compute_gae`)
- Advantage normalization utility (`normalize_advantages`)
- 21 PPO tests + 12 RL utility tests

## [0.5.0] - 2026-07-14

### Added
- **REINFORCE (Monte Carlo Policy Gradient)** algorithm
  - Policy network with softmax output
  - Discounted return computation
  - Advantage normalization
  - On-policy training loop
- Categorical sampling utility (`categorical_sample`)
- REINFORCE training example for CartPole-v1
- 15 REINFORCE tests

## [0.4.0] - 2026-07-14

### Added
- **Deep Q-Network (DQN)** with experience replay and target network
  - Q-network and target network architecture
  - ε-greedy action selection
  - Experience replay buffer
  - Periodic target network synchronization
- Replay buffer (`ReplayBuffer`) with circular overwrite
- Epsilon-greedy utility (`epsilon_greedy`)
- GymAdapter for Gymnasium environments
- Sequential model container
- DQN training example for CartPole-v1
- 14 DQN + replay buffer tests

## [0.3.0] - 2026-07-14

### Added
- **Tabular Q-Learning** algorithm
  - Q-table with ε-greedy exploration
  - Q-learning update rule
- **GridWorld** environment (n×n grid with walls and goals)
- **FrozenLake** environment (4×4 stochastic slippery grid)
- GridWorld and FrozenLake training examples
- 11 environment + Q-learning tests

## [0.2.0] - 2026-07-14

### Added
- **Adam optimizer** with bias-corrected moment estimates
- Weight save/load utilities (`save_weights`, `load_weights`)
- XOR training example with Adam
- 5 Adam + save/load tests

## [0.1.0] - 2026-07-14

### Added
- **Neural Network Core**
  - `Layer` base class with forward/backward API
  - `Dense` (fully-connected) layer
  - `ReLU` and `Sigmoid` activation layers
  - `softmax` forward-only function
  - `mse_loss` and `softmax_cross_entropy_loss` functions
  - `SGD` optimizer
  - He and Xavier weight initialization
  - `Sequential` model container
- XOR classification example
- 20 core NN tests
