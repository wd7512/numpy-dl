# Contributing to NumPy-DL

Thanks for your interest in NumPy-DL. The project is in early development (pre-v0.1.0),
so the most valuable contributions right now are design discussion via issues rather
than large PRs, until the NN core API stabilizes.

## Development setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and running
commands — not `pip`/`python` directly.

```bash
git clone https://github.com/wd7512/numpy-dl.git
cd numpy-dl
uv sync --group dev
```

This creates `.venv` and installs the package (editable) plus dev dependencies, pinned by
`uv.lock`. You do not need to activate the venv manually — prefix commands with `uv run`.

If you add or change a dependency in `pyproject.toml`, run `uv lock` to update `uv.lock`,
and commit both files together.

## Running tests

```bash
uv run pytest
```

## Code style

- Format/lint with `ruff` (`uv run ruff check .`)
- Type hints on all public functions and classes
- Docstrings on all public APIs (numpy-style or Google-style, be consistent within a file)
- Prefer readable code over clever/optimized code — this is an educational library first

## Commit messages

Use short, imperative commit messages:

```
Add Dense layer
Implement Adam optimizer
Fix gradient shape bug in Softmax backward
```

Avoid vague messages like `fix`, `stuff`, `changes`.

## Adding a new component (layer / loss / optimizer / algorithm)

1. Implement it in the relevant module under `src/numpy_dl/`.
2. Add unit tests under `tests/`, including a numerical gradient check if it involves
   backpropagation (compare analytic gradients to finite-difference approximations).
3. Add or update an example under `examples/` if it's a new RL algorithm.
4. Update `CHANGELOG.md` under `[Unreleased]`.

## Reporting bugs / proposing features

Open a GitHub issue. For algorithm-level design questions (e.g. "how should PPO's
GAE computation be structured without autodiff?"), please open a discussion issue
rather than jumping straight to a PR — the project intentionally favors clarity and
consistency over ad-hoc implementations.
