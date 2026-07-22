# Ethos & Planning

This is the project's manifesto: why it exists, what it deliberately excludes, and how
it got built. None of this is needed to *use* the library — see `README.md` for that.
It's here for anyone deciding whether to contribute, or curious about the reasoning
behind the design choices.

---

## Vision

Modern deep learning frameworks such as PyTorch, TensorFlow, and JAX provide powerful
APIs for implementing deep reinforcement learning algorithms. However, they also
introduce significant dependency, deployment, and complexity overhead that is
unnecessary for many small-scale projects.

NumPy-DL aims to bridge the gap between educational implementations and
production-scale frameworks by providing a minimal, dependency-light library for
building and training neural networks and reinforcement learning agents.

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

Before starting, it's worth being precise about what already exists, since parts of
this idea are well-trodden.

- **Pure-NumPy neural network engines** are common — dozens of "MLP from scratch"
  repositories implement layers, activations, backprop, and optimizers in NumPy alone.
  This part of the problem is not novel; it's tutorial-tier and well understood.
- **Tabular Q-learning in NumPy** is ubiquitous in introductory RL material.
- The closest existing match to the *full scope* of this project is
  [`monoelh/deep-reinforcement-learning_DDQN_PPO_HER`](https://github.com/monoelh/deep-reinforcement-learning_DDQN_PPO_HER)
  — a pure-NumPy MLP + DDQN framework with experimental PPO and HER code. It proves the
  idea is achievable, but it is notebook-based, unmaintained, not packaged, and has no
  stable public API.
- Every actively maintained "lightweight" RL library found (RLkit, reinforce-lib,
  GenRL, drlkit, Tianshou, rllib, etc.) depends on PyTorch or TensorFlow.

**Conclusion:** no maintained, pip-installable library currently takes a NumPy-only NN
engine through the full arc of Q-learning → DQN → REINFORCE → PPO behind a clean,
framework-style API. The novelty of NumPy-DL is not any single component — it's the
**integration**: one small autodiff-free engine, shared across algorithms of increasing
difficulty, packaged and tested as a real library rather than a notebook. PPO in
particular is the honest stress test, since it requires manually deriving and
backpropagating the clipped surrogate objective and GAE through two networks without
any autodiff to lean on.

---

## Problem Statement

There is currently no mature, actively maintained library that provides modern deep
reinforcement learning algorithms implemented entirely with NumPy, packaged behind a
stable, reusable API.

Existing solutions generally fall into two categories:

**Educational examples** — excellent for learning but often incomplete,
unstructured, not reusable, and designed for a single notebook.

**Production frameworks** — libraries such as PyTorch make implementing RL
straightforward but introduce GPU dependencies, automatic differentiation frameworks,
large installation sizes, significant abstraction, and complex deployment. For many
projects this is unnecessary.

NumPy-DL fills the gap by providing a lightweight implementation that remains
understandable while still supporting modern reinforcement learning algorithms.

---

## Research Question

Can a minimal NumPy-only deep learning framework provide sufficient functionality to
implement modern reinforcement learning algorithms — up to and including PPO — for
small-scale applications, while reducing the dependency and complexity overhead of
existing deep learning frameworks?

A realistic MVP stops at **NumPy MLP + tabular Q-learning + DQN + REINFORCE**. PPO is
a stretch goal: it tests whether the framework has matured enough to support more
advanced optimization (actor-critic, GAE, clipped objectives) rather than remaining a
purely educational neural network toy.

---

## Goals

**Primary**
- Build a minimal neural network library using only NumPy
- Support modern reinforcement learning algorithms
- Keep the codebase readable and educational
- Minimize dependencies
- Provide a clean, intuitive API

**Secondary**
- Maintain high code quality
- Provide comprehensive documentation
- Include examples for every algorithm
- Encourage community contributions

---

## Non-Goals

NumPy-DL is **not** intended to replace PyTorch. It does not aim to provide GPU
acceleration, automatic differentiation, distributed training, transformer
architectures, production-scale training, highly optimized tensor operations, or
large-scale benchmarks/SOTA results. If users require maximum performance, they
should use PyTorch or JAX.

---

## Design Philosophy

**Minimal dependencies** — `numpy` at runtime; `pytest`/`ruff` for development.
Nothing else should be required.

**Readability first** — clarity over clever optimizations. Good code should answer:
why does this algorithm work, what mathematical operation is happening, can a student
understand this implementation.

**Progressive complexity** — algorithms build on one another:
```
Linear Algebra → Neural Networks → Q-Learning → DQN → REINFORCE → PPO
```

**Explicit mathematics** — equations should closely match their implementation. For
example, `Q ← Q + α(r + γ max(Q') − Q)` should map directly to code.

**Usability** — the API should resemble common RL frameworks. A user shouldn't need
to understand the internal gradient calculations to train an agent:
```python
from numpy_dl.rl import DQN

agent = DQN(state_size=4, action_size=2)
agent.train(environment, episodes=500)
```

---

## Project Scope

**Phase 1 — Neural Network Core**: Dense layer, ReLU/Sigmoid/Tanh/Softmax, forward
propagation, manual backpropagation, MSE/Cross-Entropy losses, SGD/Adam optimizers.

**Phase 2 — Classical RL**: Multi-armed bandits, tabular Q-learning, SARSA (optional).

**Phase 3 — Deep RL**:
- *DQN*: neural Q-function, replay buffer, target network, ε-greedy exploration
- *REINFORCE*: policy network, softmax sampling, return estimation, policy gradient
- *PPO (stretch goal)*: actor/critic networks, GAE, clipped objective, entropy bonus

**Out of scope**: GPU acceleration, distributed training, large-scale benchmarks,
transformer architectures, production-grade RL performance, autodiff comparable to
PyTorch.

---

## Success Criteria

1. Train a DQN agent on a small environment such as CartPole.
2. Train a REINFORCE agent on a discrete-action environment.
3. Demonstrate PPO learning on a simple discrete or continuous task.
4. Require only `pip install numpy` at runtime.
5. Provide readable implementations where users can inspect and understand the
   mathematics behind each algorithm.

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

## Long-Term Vision

NumPy-DL should become the reference implementation for understanding modern deep
reinforcement learning without relying on heavyweight frameworks.

A new user should be able to clone the repository and understand the implementation
of neural networks, backpropagation, gradient descent, Q-learning, deep Q-networks,
REINFORCE, and PPO — without needing to navigate hundreds of thousands of lines of
framework code.

The library should remain intentionally small, dependency-light, and mathematically
transparent throughout its lifetime.
