# Known Issues

Verified against `src/` on 2026-07-22. Supersedes `AUDIT-FIXES.md`, which was stale
(most of its items were already fixed and it didn't track that).

## Open

- **PPO has no NaN/inf guard.** ~~`dqn.py` and `reinforce.py` both break their training
  loop on a NaN/inf loss; `ppo.py` doesn't.~~ FIXED 2026-07-22: `train_step` now
  checks `unclipped` for NaN/inf and aborts remaining epochs/logging, matching the
  guard pattern in `dqn.py`/`reinforce.py`. Test added: `tests/rl/test_ppo.py::
  TestPPOTrainStep::test_nan_inf_guard_aborts_training`. *(rl/ppo.py)*
- **`epsilon_greedy` crashes on empty `q_values`** with an unhelpful numpy error
  instead of a clear one. Untested. *(rl/utils.py)* FIXED 2026-07-22: both
  `epsilon_greedy` and `categorical_sample` now raise `ValueError` with a clear
  message on empty input; tests added.
- **No numerical gradient checking anywhere.** Every backward pass is hand-derived and
  none are checked against finite differences, including PPO's clipped surrogate — the
  hardest one to get right by hand. `tests/rl/test_ppo.py::TestPPOClippedObjective`
  only checks the scalar min-of-surrogates formula, not the actual gradient computed in
  `train_step`.
- **`ReplayBuffer` is list-of-tuples**, not preallocated arrays. Works, but sampling
  and storage both do more Python-level work than necessary.
- **`_compute_returns()` in `reinforce.py`** is an unvectorized Python loop. (The GAE
  loop in `rl/utils.py` is intentionally sequential and fine as-is.) FIXED
  2026-07-22: replaced with a vectorized `discount @ rewards` matmul, where
  `discount[t,k] = gamma^(k-t)` for `k>=t` and 0 elsewhere. The GAE loop in
  `rl/utils.py` remains sequential (correct per BUGS.md note). Test added:
  `tests/rl/test_reinforce.py::TestREINFORCEComputeReturns::test_matches_naive_loop_on_random_rewards`.
- **`Sequential.parameters()` rebuilds its list on every call**; `Dense.parameters()`
  already caches, so this is just an inconsistency, not a functional bug. FIXED
  2026-07-22: now builds `self._cached_params` once in `__init__` and returns it, mirroring `Dense`'s pattern. Tests added in `tests/nn/test_sequential.py`.
- **`he_init`/`xavier_init` use the global `np.random.randn`**, not an injectable RNG,
  so they aren't reproducible independent of global numpy state. FIXED 2026-07-22:
  both accept an optional `rng: np.random.RandomState`; when None, they draw from
  a module-private RandomState instead of global `np.random`. Users inject an RNG
  via `functools.partial(he_init, rng=my_rng)`. The two XOR integration tests that
  relied on `np.random.seed(0)` were updated to inject a `RandomState(0)` directly.
  New `tests/nn/test_init.py` covers reproducibility, divergence, scale, and the
  Dense-with-injected-RNG path.
- **`requires-python` is still `>=3.9`**; 3.9 is EOL. FIXED 2026-07-22: bumped to
  `>=3.10` (pyproject.toml), dropped the 3.9 classifier, raised ruff `target-version`
  to `py310`, repinned `.python-version` to 3.11, fixed the resulting `typing.Callable`
  → `collections.abc.Callable` UP035 finding in `nn/layers.py`.
- **FrozenLake example never reaches the goal.** `examples/frozen_lake_q_learning.py`
  ships `epsilon=0.1`, no epsilon decay, 2000 episodes on the slippery 4x4 map.
  Verified 2026-07-22: 0 of 2000 episodes reach the goal. The example runs without
  error and illustrates the q-learning update, but would need higher `episodes`,
  epsilon decay, or `is_slippery=False` before it shows actual learning. The
  example file now carries an honest note rather than silently using different
  hyperparameters to make the plot look good. *(examples/frozen_lake_q_learning.py)*

## Verified fixed (no longer tracked)

Optimizer state persistence in DQN, Dense shape validation, backward-before-forward
errors, `pyproject.toml` version/numpy bound, `load_weights` error handling, replay
buffer copy-on-push, `max_steps` guards, gym adapter dtype, advantage-normalization
near-zero warning, empty-`Sequential` warning.

## Checked and NOT a bug

The PPO clip mask (`is_unclipped = unclipped <= clipped`) was suspected of being a
sign/masking error. Verified against the textbook clip-gradient rule across 100k random
`(ratio, advantage)` pairs (zero mismatches) and against finite differences on a real
trajectory (mean relative error 0.8%, max 4.7%, consistent with FD step size). The
gradient is correct — flagged above only because nothing in the test suite would catch
a regression here.

## Secondary / deferred (not bugs, tracked for v1.0.0)

- **Examples should be readable notebooks, not scripts.** Currently all 8 examples
  under `examples/` are `.py` entry points using `logging.basicConfig`. A secondary
  v1.0.0 goal is to migrate these into simple, readable notebook form so the
  progressive-complexity story (NN core → tabular RL → deep RL → PPO) is walking
  through one notebook at a time. This interacts with `RESTRUCTURE.md`'s proposal to
  group `examples/` into `0X_{nn_core,tabular_rl,deep_rl,ppo}/` phase directories —
  the directory reorg should land first, then the script-to-notebook conversion
  inside each phase dir.
