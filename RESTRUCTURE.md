# Proposed structure for v1.0.0

Current layout works but has accumulated dev-process scaffolding that shouldn't
ship in a 1.0 release. Proposal below is organized by what changes, not a full tree
dump.

## Root: trim to 5 files

Keep: `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `LICENSE`, `pyproject.toml`.

Remove or move:
- `AUDIT-FIXES.md` → delete once `BUGS.md` items are resolved or transferred to
  GitHub issues. Task-list docs shouldn't live in the repo long-term.
- `AGENTS.md` / `opencode.json` → move into `.github/` or a `.agent/` dir, or drop
  entirely if not actively used. These are tooling config, not project docs; a
  reader browsing the repo root shouldn't hit them before `README.md`.
- `docs/plans/v0.*.0-spec.md` (7 files, ~345 lines) → delete. These are pre-work
  specs for versions that have already shipped; `CHANGELOG.md` already records what
  landed. If historical design rationale is worth keeping, fold the *decisions*
  (not the task breakdowns) into a single `docs/design-notes.md`, not one file per
  version.
- `BUGS.md` → keep at root until v1.0.0, then convert remaining items to GitHub
  issues and delete the file. A shipped 1.0 shouldn't carry a bugs-in-progress
  file — it should carry a clean CHANGELOG and an issue tracker.

Net: root `.md` count goes from 6 (README, AUDIT-FIXES, AGENTS, CHANGELOG,
CONTRIBUTING, +BUGS now) down to 3 permanent ones (README, CHANGELOG,
CONTRIBUTING) by release time.

## `tests/`: collapse three overlapping suites into one

Currently: `tests/nn|optim|rl|memory|environments/` (unit tests, good) +
`tests/test_integration/` (2 files) + `tests/validation/` (4 files, PyTorch
comparison) + `tests/test_placeholder.py` (dead — asserts a version string, kept
"until real tests exist," which they now do).

Proposed:
```
tests/
├── unit/            # rename from flat nn/optim/rl/memory/environments split
│   ├── nn/
│   ├── optim/
│   ├── rl/
│   ├── memory/
│   └── environments/
├── integration/      # merge test_integration/ + the XOR-style end-to-end cases
└── validation/        # unchanged — PyTorch comparison, opt-in via `pip install .[validation]`
```
- Delete `test_placeholder.py`.
- Delete `__pycache__` directories from version control if they're currently
  tracked (check `.gitignore` covers `tests/**/__pycache__`).
- Add `tests/unit/rl/test_gradients.py` — the numerical-gradient-check utility
  the README promises but doesn't have yet (see `BUGS.md`). This is the single
  highest-value addition for a 1.0 release, not a structural change but worth
  naming here since it defines where it should live.

## `src/numpy_dl/`: no major reshuffling needed

The module split (`nn/`, `optim/`, `rl/`, `memory/`, `utils/`, `environments/`) is
already sound and matches the README's documented package structure — don't
change it for the sake of change. Two small additions:

- `nn/` — add a `gradcheck.py` with the finite-difference utility, importable as
  `numpy_dl.nn.gradcheck.check_gradients(layer, x)`, used by both the test suite
  and available to end users debugging a custom layer.
- `rl/utils.py` is getting crowded (epsilon_greedy, categorical_sample,
  normalize_advantages, compute_gae — four unrelated concerns in one file). Fine
  pre-1.0; worth splitting into `rl/exploration.py` (epsilon_greedy,
  categorical_sample) and `rl/advantages.py` (normalize_advantages, compute_gae)
  if the file keeps growing, but not urgent enough to block a release.

## `examples/`: group by phase, not left flat

Currently 8 files flat (`xor_sgd.py`, `xor_adam.py`,
`gridworld_q_learning.py`, `frozen_lake_q_learning.py`, `dqn_cartpole.py`,
`reinforce_cartpole.py`, `ppo_cartpole.py`, `ppo_lunar_lander.py`). Fine at 8
files, but the README's own "Progressive Complexity" pitch (Linear Algebra →
NN → Q-Learning → DQN → REINFORCE → PPO) isn't reflected in the file listing a
new user sees. Proposed:
```
examples/
├── 01_nn_core/          xor_sgd.py, xor_adam.py
├── 02_tabular_rl/        gridworld_q_learning.py, frozen_lake_q_learning.py
├── 03_deep_rl/           dqn_cartpole.py, reinforce_cartpole.py
└── 04_ppo/               ppo_cartpole.py, ppo_lunar_lander.py
```
Numbering makes the progressive-complexity story navigable directly from a file
browser, matching the pitch already made in the README.

## `docs/`: mostly empty currently, decide its purpose before 1.0

Right now `docs/` only contains `plans/` (deleted above) and a `.gitkeep`. Either:
(a) delete `docs/` entirely and let `README.md` + module docstrings carry all
documentation for 1.0, which is consistent with the "minimal" pitch, or
(b) commit to it being real user-facing docs (API reference, tutorial) — but
don't leave it as a graveyard for internal planning notes.
Given the project's own philosophy ("One dependency. Understand every line of
code.") — option (a) fits better until there's a real need for generated docs.

## Net effect

- Root `.md` files: 6 → 3 (permanent) + `BUGS.md` temporarily until resolved
- `docs/plans/`: 7 files deleted
- `tests/`: one dead file removed, three ambiguous top-level dirs (`test_integration`,
  `validation`, flat unit dirs) organized under a single `tests/{unit,integration,validation}/` split
- `examples/`: flat 8 files → 4 numbered phase directories matching the README's
  own progressive-complexity pitch
- No changes to `src/numpy_dl/` package layout — it's already the right shape
