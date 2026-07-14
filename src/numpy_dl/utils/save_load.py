"""Save and load neural network weights using NumPy's .npz format.

Creator references:
    NumPy Developers. "NumPy Reference: numpy.savez."
    — Serialized array storage.

Mathematical equations:
    save:  arrays[k_i] = W_i  for each parameter W_i in model
    load:  W_i = arrays[k_i]  for each indexed key k_i
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from numpy_dl.nn.layers import Dense, Layer


def _collect_layers(model: Layer | list[Layer]) -> list[Dense]:
    """Collect Dense layers from a model or list of layers."""
    if isinstance(model, Dense):
        return [model]
    if isinstance(model, list):
        layers: list[Dense] = []
        for layer in model:
            if isinstance(layer, Dense):
                layers.append(layer)
        return layers
    raise TypeError(f"Expected Dense or list of Layers, got {type(model)}")


def save_weights(model: Dense | list[Layer], path: str | Path) -> None:
    """Save model weights to a .npz file with indexed keys.

    Args:
        model: A Dense layer or list of layers.
        path: File path to save to (will add .npz if missing).
    """
    p = Path(path)
    if p.suffix != ".npz":
        p = p.with_suffix(".npz")
    layers = _collect_layers(model)
    arrays: dict[str, np.ndarray] = {}
    idx = 0
    for layer in layers:
        arrays[f"layer_{idx}_W"] = layer.W
        arrays[f"layer_{idx}_b"] = layer.b
        idx += 1
    np.savez(p, **arrays)


def load_weights(model: Dense | list[Layer], path: str | Path) -> None:
    """Load model weights from a .npz file.

    Args:
        model: A Dense layer or list of layers.
        path: File path to load from (will add .npz if missing).

    Raises:
        FileNotFoundError: If the weights file does not exist.
        KeyError: If required weight keys are missing from the file.
        ValueError: If weight shapes don't match the model.
    """
    p = Path(path)
    if p.suffix != ".npz":
        p = p.with_suffix(".npz")
    if not p.exists():
        raise FileNotFoundError(f"Weights file not found: {p}")
    try:
        data = np.load(p)
    except Exception as e:
        raise ValueError(f"Failed to load weights from {p}: {e}") from e
    layers = _collect_layers(model)
    for idx, layer in enumerate(layers):
        w_key = f"layer_{idx}_W"
        b_key = f"layer_{idx}_b"
        if w_key not in data:
            raise KeyError(f"Missing key {w_key!r} in weights file. Available: {list(data.keys())}")
        if data[w_key].shape != layer.W.shape:
            raise ValueError(
                f"Shape mismatch for {w_key}: file {data[w_key].shape} != model {layer.W.shape}"
            )
        layer.W[:] = data[w_key]
        layer.b[:] = data[b_key]
