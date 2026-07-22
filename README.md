# NumPy-DL

> A lightweight deep learning and reinforcement learning toolkit implemented from scratch using only NumPy.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](pyproject.toml)
[![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)](#development-roadmap)

---

## Vision

Modern deep learning frameworks such as PyTorch, TensorFlow, and JAX provide powerful APIs for implementing deep reinforcement learning algorithms. However, they also introduce significant dependency, deployment, and complexity overhead that is unnecessary for many small-scale projects.

NumPy-DL aims to bridge the gap between educational implementations and production-scale frameworks by providing a minimal, dependency-light library for building and training neural networks and reinforcement learning agents.

The project is designed for:

- Small-scale reinforcement learning projects
- Educational use
- Research prototypes
- Embedded or constrained environments
- Developers who want deep learning without a heavyweight framework

The philosophy is simple:

> **One dependency. Understand every line of code.**

---

## Prior Art

Before starting, it's worth being precise about what already exists, since parts of this idea are well-trodden.

- **Pure-NumPy neural network engines** are common — dozens of "MLP from scratch" repositories implement layers, activations, backprop, and optimizers in NumPy alone. This part of the problem is not novel; it's tutorial-tier and well understood.
- **Tabular Q-learning in NumPy** is ubiquitous in introductory RL material.
- The closest existing match to the *full scope* of this project is [`monoelh/deep-reinforcement-learning_DDQN_PPO_HER`](https://github.com/monoelh/deep-reinforcement-learning_DDQN_PPO_HER) — a pure-NumPy MLP + DDQN framework with experimental PPO and HER code. It proves the idea is achievable, but it is notebook-based, unmaintained, not packaged, and has no stable public API.
- Every actively maintained "lightweight" RL library found (RLkit, reinforce-lib, GenRL, drlkit, Tianshou, rllib, etc.) depends on PyTorch or TensorFlow.

**Conclusion:** no maintained, pip-installable library currently takes a NumPy-only NN engine through the full arc of Q-learning → DQN → REINFORCE → PPO behind a clean, framework-style API. The novelty of NumPy-DL is not any single component — it's the **integration**: one small autodiff-free engine, shared across algorithms of increasing difficulty, packaged and tested as a real library rather than a notebook. PPO in particular is the honest stress test, since it requires manually deriving and backpropagating the clipped surrogate objective and GAE through two networks without any autodiff to lean on.

---

## Problem Statement

There is currently no mature, actively maintained library that provides modern deep reinforcement learning algorithms implemented entirely with NumPy, packaged behind a stable, reusable API.

Existing solutions generally fall into two categories:

### Educational Examples

These repositories are excellent for learning but are often:

- incomplete
- unstructured
- not reusable
- designed for a single notebook

### Production Frameworks

Libraries such as PyTorch make implementing RL straightforward but introduce:

- GPU dependencies
- automatic differentiation frameworks
- large installation sizes
- significant abstraction
- complex deployment

For many projects this is unnecessary.

NumPy-DL fills the gap by providing a lightweight implementation that remains understandable while still supporting modern reinforcement learning algorithms.

---

## Research Question

Can a minimal NumPy-only deep learning framework provide sufficient functionality to implement modern reinforcement learning algorithms — up to and including PPO — for small-scale applications, while reducing the dependency and complexity overhead of existing deep learning frameworks?

A realistic MVP stops at **NumPy MLP + tabular Q-learning + DQN + REINFORCE**. PPO is a stretch goal: it tests whether the framework has matured enough to support more advanced optimization (actor-critic, GAE, clipped objectives) rather than remaining a purely educational neural network toy.

---

## Goals

### Primary Goals

- Build a minimal neural network library using only NumPy
- Support modern reinforcement learning algorithms
- Keep the codebase readable and educational
- Minimize dependencies
- Provide a clean, intuitive API

### Secondary Goals

- Maintain high code quality
- Provide comprehensive documentation
- Include examples for every algorithm
- Encourage community contributions

---

## Non-Goals

NumPy-DL is **not** intended to replace PyTorch.

The project does **not** aim to provide:

- GPU acceleration
- Automatic differentiation
- Distributed training
- Transformer architectures
- Production-scale training
- Highly optimized tensor operations
- Large-scale benchmarks or SOTA results

If users require maximum performance, they should use PyTorch or JAX.

---

## Design Philosophy

### Minimal Dependencies

Required packages:

```
numpy
```

Optional development dependencies:

```
pytest
ruff
```

Nothing else should be required.

### Readability First

The implementation should prioritize clarity over clever optimizations.

Good code should answer:

- Why does this algorithm work?
- What mathematical operation is happening?
- Can a student understand this implementation?

### Progressive Complexity

Algorithms should build upon one another.

```
Linear Algebra
      ↓
Neural Networks
      ↓
Q-Learning
      ↓
DQN
      ↓
REINFORCE
      ↓
PPO
```

### Explicit Mathematics

Whenever practical, equations should closely match their implementation.

For example:

```
Q ← Q + α(r + γ max(Q') − Q)
```

should map directly to code.

### Usability

The API should resemble common RL frameworks. A user should not need to understand the internal gradient calculations to train an agent.

```python
from numpy_dl.rl import DQN

agent = DQN(state_size=4, action_size=2)
agent.train(environment, episodes=500)
```

---

## Project Scope

### Phase 1 — Neural Network Core

- Dense (Linear) layer
- ReLU, Sigmoid, Tanh, Softmax
- Forward propagation
- Manual backpropagation
- Mean Squared Error, Cross Entropy losses
- SGD, Adam optimizers

### Phase 2 — Classical Reinforcement Learning

- Multi-Armed Bandits
- Tabular Q-Learning
- SARSA (optional)

### Phase 3 — Deep Reinforcement Learning

**DQN**
- Neural Q-function
- Replay Buffer
- Target Network
- ε-Greedy Exploration

**REINFORCE**
- Policy Network
- Softmax Sampling
- Return Estimation
- Policy Gradient

**PPO** *(stretch goal)*
- Actor Network
- Critic Network
- Generalized Advantage Estimation (GAE)
- Clipped Objective
- Entropy Bonus

### Out of Scope

- GPU acceleration
- Distributed training
- Large-scale benchmarks
- Transformer architectures
- Production-grade RL performance
- Automatic differentiation comparable to PyTorch

---

## Success Criteria

The project is successful if it can:

1. Train a DQN agent on a small environment such as CartPole.
2. Train a REINFORCE agent on a discrete-action environment.
3. Demonstrate PPO learning on a simple discrete or continuous task.
4. Require only `pip install numpy` at runtime.
5. Provide readable implementations where users can inspect and understand the mathematics behind each algorithm.

---

## Package Structure

```
numpy-dl/
│
├── pyproject.toml
├── README.md
├── LICENSE
├── CHANGELOG.md
│
├── src/
│   └── numpy_dl/
│       ├── __init__.py
│       │
│       ├── nn/
│       │   ├── layers.py
│       │   ├── activations.py
│       │   ├── losses.py
│       │   ├── initializers.py
│       │   └── models.py
│       │
│       ├── optim/
│       │   ├── sgd.py
│       │   └── adam.py
│       │
│       ├── rl/
│       │   ├── q_learning.py
│       │   ├── dqn.py
│       │   ├── reinforce.py
│       │   └── ppo.py
│       │
│       ├── memory/
│       │   └── replay_buffer.py
│       │
│       ├── utils/
│       │
│       └── environments/
│
├── examples/
│   ├── 01_nn_core/        # xor_sgd.py, xor_adam.py
│   ├── 02_tabular_rl/     # gridworld_q_learning.py, frozen_lake_q_learning.py
│   ├── 03_deep_rl/        # dqn_cartpole.py, reinforce_cartpole.py
│   └── 04_ppo/            # ppo_cartpole.py, ppo_lunar_lander.py
├── tests/
│   ├── unit/              # nn, optim, rl, memory, environments unit tests
│   ├── integration/       # end-to-end training (XOR with SGD / Adam)
│   └── validation/        # opt-in PyTorch comparison (pip install .[validation])
└── docs/
```

---

## Development Roadmap

| Version | Scope | Examples |
|---|---|---|
| **0.1.0** | Linear layer, ReLU, Softmax, SGD, MSE loss | XOR classification |
| **0.2.0** | Adam optimizer, improved model API, save/load weights | MNIST (optional), regression |
| **0.3.0** | Tabular Q-Learning | GridWorld, Frozen Lake |
| **0.4.0** | Replay buffer, DQN, target network | CartPole |
| **0.5.0** | REINFORCE | CartPole |
| **0.6.0** | PPO (actor-critic, GAE, clipped objective) | Lunar Lander, CartPole |
| **1.0.0** | Stable API, full documentation, complete testing, published on PyPI | — |

---

## Versioning

The project follows [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`):

- **Patch** — bug fixes only (`0.2.0 → 0.2.1`)
- **Minor** — new features without breaking the API (`0.3.0 → 0.4.0`)
- **Major** — breaking API changes (`1.0.0 → 2.0.0`)

---

## Installation (development)

The project is developed with [uv](https://docs.astral.sh/uv/) rather than calling `pip`/`python` directly. Install uv once ([instructions](https://docs.astral.sh/uv/getting-started/installation/)), then:

```bash
git clone https://github.com/wd7512/numpy-dl.git
cd numpy-dl
uv sync --extra dev
```

`uv sync` creates a `.venv`, installs the package in editable mode, and installs dev dependencies (`pytest`, `ruff`) — all pinned by `uv.lock` for reproducibility. There's no need to manually create or activate a virtual environment; prefix commands with `uv run` instead:

```bash
uv run pytest
uv run ruff check .
uv run python examples/01_nn_core/xor_sgd.py
```

Build backend is `hatchling` via `pyproject.toml`.

---

## Testing

```bash
uv run pytest
```

Test coverage should include:

- Layer correctness
- Gradient correctness (e.g. numerical gradient checking against analytic backprop)
- Optimizers
- Loss functions
- RL algorithms

Every mathematical component should have unit tests.

---

## Development Standards

**Code style**
- Type hints everywhere
- Docstrings for all public APIs
- Small modules
- Readability over optimization

**Commit messages**

Good:
```
Add Dense layer
Implement Adam optimizer
Add replay buffer
Implement PPO policy update
```

Bad:
```
fix
stuff
changes
```

---

## Contributing

Contributions are welcome once the v0.1.0 NN core lands. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines (coming soon). In the meantime, feel free to open issues for design discussion.

---

## License

MIT License — see [LICENSE](LICENSE). The goal is to maximize educational reuse while allowing commercial use.

---

## Long-Term Vision

NumPy-DL should become the reference implementation for understanding modern deep reinforcement learning without relying on heavyweight frameworks.

A new user should be able to clone the repository and understand the implementation of:

- Neural Networks
- Backpropagation
- Gradient Descent
- Q-Learning
- Deep Q Networks
- REINFORCE
- PPO

without needing to navigate hundreds of thousands of lines of framework code.

The library should remain intentionally small, dependency-light, and mathematically transparent throughout its lifetime.
