"""Shared fixtures and skip logic for SOTA comparison tests.

These tests validate numpy-dl implementations against PyTorch.
They are skipped by default when PyTorch is not installed.

Run with: uv sync --group dev --group validation && uv run pytest tests/validation/
"""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch required for validation tests")
