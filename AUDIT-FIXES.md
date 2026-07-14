# Audit Fixes — numpy-dl

Fix every item below. Run `uv run ruff check . && uv run pytest -v` after each batch of fixes.

## CRITICAL

### 1. DQN optimizer broken — second moment `v` not persisted
File: src/numpy_dl/rl/dqn.py
The inline Adam-like optimizer recomputes `v` from scratch each step instead of maintaining an exponential moving average. Fix: persist `v` in `_optimizer_states` alongside `m`. On first call, initialize `v` to ones_like. On subsequent calls, update: `v = 0.999 * v + 0.001 * grad**2`.

### 2. No NaN/infinity detection in training loops
Files: src/numpy_dl/rl/dqn.py, src/numpy_rl/reinforce.py, src/numpy_rl/ppo.py
Add a check after each train_step: if loss is NaN or inf, log a warning and break the training loop. Use `import logging; logger = logging.getLogger(__name__)` at module top.

### 3. No input validation on Dense.forward()
File: src/numpy_dl/nn/layers.py
In `forward()`, validate that `x.ndim == 2` and `x.shape[1] == self.W.shape[0]`. Raise `ValueError` with a clear message including expected vs actual shapes.

### 4. Backward before forward produces opaque error
Files: src/numpy_dl/nn/layers.py, src/numpy_dl/nn/activations.py
In `backward()`, check if cached state is None. If so, raise `RuntimeError("backward() called before forward()")`.

## HIGH

### 5. pyproject.toml version mismatch
File: pyproject.toml
Change `version = "0.0.1"` to `version = "0.6.0"`.

### 6. numpy>=1.24 has no upper bound
File: pyproject.toml
Change `"numpy>=1.24"` to `"numpy>=1.24,<3"` in both `[project.dependencies]` and `[dependency-groups]`.

### 7. load_weights has zero error handling
File: src/numpy_dl/utils/save_load.py
Wrap the load in try/except: check file exists, handle KeyError for missing keys, validate shapes match layer expectations. Raise descriptive errors.

### 8. replay_buffer.py doesn't copy state arrays
File: src/numpy_dl/memory/replay_buffer.py
In `push()`, copy state and next_state: `state = np.array(state)` and `next_state = np.array(next_state)`.

### 9. No max_steps guard in RL train() loops
Files: src/numpy_dl/rl/dqn.py, src/numpy_rl/reinforce.py, src/numpy_rl/ppo.py
Add `max_steps: int = 10000` parameter to `train()`. Break the inner while loop if steps exceed max_steps.

### 10. replay_buffer.py O(capacity) sampling
File: src/numpy_dl/memory/replay_buffer.py
Replace list-of-tuples with struct-of-arrays: pre-allocate numpy arrays for states, actions, rewards, next_states, dones. Use `np.random.randint` for sampling. This is a significant refactor but the core API (push/sample/__len__) stays the same.

## MEDIUM

### 11. compute_gae and _compute_returns use Python for-loops
Files: src/numpy_rl/rl/utils.py, src/numpy_dl/rl/reinforce.py
Vectorize `_compute_returns()`: use `np.cumsum` with reversed gamma powers.
For `compute_gae()`: keep the loop (sequential dependency makes full vectorization complex) but add a comment explaining why.

### 12. parameters() creates new list on every call
Files: src/numpy_dl/nn/layers.py, src/numpy_dl/nn/sequential.py
In Dense, cache the parameters list in __init__. In Sequential, cache on first call.

### 13. gym_adapter.py forces dtype copy every timestep
File: src/numpy_dl/environments/gym_adapter.py
Remove the `dtype=np.float64` from `np.asarray` — let it keep the gym's native dtype.

### 14. reinforce.py store_transition accepts unused log_prob
File: src/numpy_dl/rl/reinforce.py
Remove the `log_prob` parameter from `store_transition()` since it's recomputed in `train_step()`.

### 15. Add logging to library
Files: all src/numpy_dl/ files
Add `import logging; logger = logging.getLogger(__name__)` to each module. Add `logger.debug()` for key operations (forward pass shapes, loss values in training loops). Do NOT add logging to hot paths (forward/backward inner loops).

### 16. normalize_advantages silently degenerates
File: src/numpy_rl/rl/utils.py
When std < 1e-8, log a warning: `logger.warning("Advantage std near zero (%.2e), skipping normalization", std)`.

### 17. Duplicate optional-dependencies and dependency-groups
File: pyproject.toml
Remove the `[dependency-groups]` section — keep only `[project.optional-dependencies]`.

## LOW

### 18. he_init/xavier_init use global RNG
File: src/numpy_dl/utils/init.py
Add optional `rng` parameter to both functions. Default to `np.random.randn` if not provided.

### 19. Loss diff computed twice in mse_loss
File: src/numpy_dl/nn/losses.py
Cache `diff = y_pred - y_true` and reuse.

### 20. Empty Sequential is silent no-op
File: src/numpy_dl/nn/sequential.py
In __init__, if layers is empty, log a warning.

### 21. epsilon_greedy crashes on empty q_values
File: src/numpy_rl/rl/utils.py
Add guard: `if len(q_values) == 0: raise ValueError("q_values is empty")`.

### 22. Python 3.9 EOL
File: pyproject.toml
Change `requires-python = ">=3.9"` to `">=3.10"`. Update CI matrix to remove 3.9.
