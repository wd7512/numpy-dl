"""Shared fixtures and skip logic for SOTA comparison tests.

These tests validate numpy-dl implementations against PyTorch and TensorFlow.
They are skipped by default when the frameworks are not installed.

Run with: uv sync --group validation && uv run pytest tests/validation/
"""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch required for validation tests")
tf = pytest.importorskip("tensorflow", reason="TensorFlow required for validation tests")
