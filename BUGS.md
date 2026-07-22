# Known Issues

Verified against `src/` on 2026-07-22. Supersedes `AUDIT-FIXES.md`, which was stale
(most of its items were already fixed and it didn't track that).

## Open

- **PPO has no NaN/inf guard.** `dqn.py` and `reinforce.py` both break their training
  loop on a NaN/inf loss; `ppo.py` doesn't. PPO's `ratio = exp(new_log_prob -
  old_log_prob)` can blow up across epochs, and nothing currently stops training if it
  does. *(rl/ppo.py)*
- **`epsilon_greedy` crashes on empty `q_values`** with an unhelpful numpy error
  instead of a clear one. Untested. *(rl/utils.py)*
- **No numerical gradient checking anywhere.** Every backward pass is hand-derived and
  none are checked against finite differences, including PPO's clipped surrogate — the
  hardest one to get right by hand. `tests/rl/test_ppo.py::TestPPOClippedObjective`
  only checks the scalar min-of-surrogates formula, not the actual gradient computed in
  `train_step`.
- **`ReplayBuffer` is list-of-tuples**, not preallocated arrays. Works, but sampling
  and storage both do more Python-level work than necessary.
- **`_compute_returns()` in `reinforce.py`** is an unvectorized Python loop. (The GAE
  loop in `rl/utils.py` is intentionally sequential and fine as-is.)
- **`Sequential.parameters()` rebuilds its list on every call**; `Dense.parameters()`
  already caches, so this is just an inconsistency, not a functional bug.
- **`he_init`/`xavier_init` use the global `np.random.randn`**, not an injectable RNG,
  so they aren't reproducible independent of global numpy state.
- **`requires-python` is still `>=3.9`**; 3.9 is EOL.

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
