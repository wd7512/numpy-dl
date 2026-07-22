# Agent Instructions

This is the NumPy-DL project — a lightweight deep learning and reinforcement learning
toolkit implemented from scratch using only NumPy.

## Commands

- Install dependencies: `uv sync --extra dev`
- Lint: `uv run ruff check`
- Test: `uv run pytest`
- Build: `uv build`

## Rules

- Use `uv` for dependency management and command execution.
- Keep changes small and focused.
- Add or update tests for behavior changes.
- Run `uv run ruff check` and `uv run pytest` before finishing code changes.
- Do not add runtime dependencies unless they are required by package behavior.
- Prefer clear, boring Python over clever abstractions.
- Use `logging` (stdlib) for all output — never `print()`. Configure via `logging.basicConfig()` in entry points.
- This is an educational library — readability over optimization.
