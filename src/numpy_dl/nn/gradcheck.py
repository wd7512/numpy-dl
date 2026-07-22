"""Numerical gradient checking utility for ``numpy_dl.nn`` layers and
arbitrary NumPy functions.

This module implements central-difference finite-difference checks against
the hand-derived backward passes used throughout the library. It exists to
answer the question the README insists on: "every line of code can be
understood" — including the backward passes, which are otherwise verified
only by integration tests showing the network learns.

Two entry points:

- ``check_layer_gradients(layer, x, dy, eps, atol, rtol)`` — verify the
  ``backward`` pass of any ``Layer`` (Dense, ReLU, ...) against finite
  differences on its parameters (W, b) and on the input x.
- ``check_function_gradients(f, x, df, eps, atol, rtol)`` — verify an
  arbitrary scalar-valued function ``y = f(x)`` and its analytic Jacobian
  ``df(x) -> dx`` against finite differences on each entry of x. Used by
  the RL test suite to verify PPO's clipped surrogate and REINFORCE's
  policy gradient in isolation before any optimizer steps are involved.

Both return a ``GradCheckResult`` with mean / max relative error and a
boolean ``ok`` flag suitable for ``assert result.ok, result.summary()``.

Creator references:
    Baydin, A.G. et al. (2018). "Automatic Differentiation in Machine
    Learning: a Survey." §3 — finite differences as a sanity check for
    hand-derived gradients.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GradCheckResult:
    """Outcome of a single gradient check.

    Attributes:
        ok: True if every measured relative error is below the configured
            ``atol``/``rtol`` thresholds, else False.
        max_rel_error: Largest relative error |analytic - numeric| /
            max(|analytic|, |numeric|, eps) over all perturbed entries.
        mean_rel_error: Mean of per-entry relative errors.
        max_abs_error: Largest absolute error |analytic - numeric| over
            all perturbed entries.
        per_field: Optional mapping field name -> per-field max relative
            error; populated by ``check_layer_gradients``.
        summary: One-line human-readable summary suitable for ``assert``
        failure messages.
    """

    ok: bool
    max_rel_error: float
    mean_rel_error: float
    max_abs_error: float
    per_field: dict[str, float] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        if self.ok:
            return (
                f"grad check OK: max_rel={self.max_rel_error:.2e} "
                f"mean_rel={self.mean_rel_error:.2e} "
                f"max_abs={self.max_abs_error:.2e}"
            )
        fields = ", ".join(
            f"{k}={v:.2e}" for k, v in self.per_field.items() if v is not None
        )
        return (
            f"grad check FAILED: max_rel={self.max_rel_error:.2e} "
            f"mean_rel={self.mean_rel_error:.2e} "
            f"max_abs={self.max_abs_error:.2e}"
            + (f" per_field=[{fields}]" if fields else "")
        )


def _rel_error(analytic: np.ndarray, numeric: np.ndarray, floor: float) -> np.ndarray:
    return np.abs(analytic - numeric) / np.maximum(
        np.maximum(np.abs(analytic), np.abs(numeric)), floor
    )


def check_layer_gradients(
    layer: object,
    x: np.ndarray,
    dy: np.ndarray,
    eps: float = 1e-6,
    atol: float = 1e-6,
    rtol: float = 1e-3,
) -> GradCheckResult:
    """Verify a ``Layer``'s ``backward`` against central-difference FD.

    For each parameter ``(p, dp)`` returned by ``layer.parameters()``, and
    also for the input ``x`` vs ``layer.backward``'s returned gradient,
    perturb every element by ``±eps`` and compare the change in
    ``sum(dy * layer.forward(x))`` to ``dp`` (or ``dx``) at the same
    element. ``backward`` is called once at the start to populate the
    analytic gradients.

    Args:
        layer: Object exposing ``forward(x)`` returning y, ``backward(dy)``
            returning dx and writing ``layer.dW`` / ``layer.db`` in place,
            and (optionally) ``parameters()`` returning ``[(p, dp), ...]``.
        x: Forward-pass input.
        dy: Upstream gradient dL/dy, must broadcast against y. The loss
            against which FD is measured is ``L = (dy * y).sum()``.
        eps: Perturbation step size.
        atol: Absolute tolerance below which ``ok`` stays True even when
            relative error is undefined (both sides ~0).
        rtol: Relative tolerance: any entry with relative error above
            ``rtol`` flips ``ok`` to False.

    Returns:
        ``GradCheckResult`` carrying per-field max rel errors and a
        per-failure summary string.
    """
    x = np.asarray(x, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    floor = max(atol, eps, 1e-12)

    # Capture analytic gradients once. backward returns dx and writes dp in place.
    layer.forward(x)  # type: ignore[union-attr]
    analytic_dx = layer.backward(dy)  # type: ignore[union-attr]
    analytic_dx = np.array(analytic_dx, dtype=np.float64)

    def loss_at(x_arg: np.ndarray) -> float:
        y = layer.forward(x_arg)  # type: ignore[union-attr]
        return float((dy * y).sum())

    def numeric_grad(arr: np.ndarray) -> np.ndarray:
        """Perturb the live array in place. Post-call the original values
        are restored; failure paths throw away the layer so no state leaks."""
        num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            orig = arr[idx]
            arr[idx] = orig + eps
            lp = loss_at(x)
            arr[idx] = orig - eps
            lm = loss_at(x)
            arr[idx] = orig
            num[idx] = (lp - lm) / (2.0 * eps)
        return num

    per_field: dict[str, float] = {}
    errors: list[float] = []
    abs_errors: list[float] = []

    # Check parameters first, if any.
    params_fn = getattr(layer, "parameters", None)
    if callable(params_fn):
        for i, (p, dp) in enumerate(params_fn()):
            numeric = numeric_grad(p)
            analytic = np.array(dp, dtype=np.float64)
            r = _rel_error(analytic, numeric, floor)
            abs_e = np.abs(analytic - numeric)
            errors.extend(r.flatten().tolist())
            abs_errors.extend(abs_e.flatten().tolist())
            per_field[f"param[{i}]"] = float(np.nanmax(r))

    # Then check the input gradient dx.
    residual_x = np.array(x, copy=True)
    numeric_dx = np.zeros_like(x, dtype=np.float64)
    for idx in np.ndindex(x.shape):
        orig = residual_x[idx]
        residual_x[idx] = orig + eps
        lp = loss_at(residual_x)
        residual_x[idx] = orig - eps
        lm = loss_at(residual_x)
        residual_x[idx] = orig
        numeric_dx[idx] = (lp - lm) / (2.0 * eps)
    r_x = _rel_error(analytic_dx, numeric_dx, floor)
    abs_x = np.abs(analytic_dx - numeric_dx)
    errors.extend(np.atleast_1d(r_x).flatten().tolist())
    abs_errors.extend(np.atleast_1d(abs_x).flatten().tolist())
    if np.atleast_1d(r_x).size:
        per_field["dx"] = float(np.nanmax(r_x))

    if not errors:
        return GradCheckResult(True, 0.0, 0.0, 0.0, per_field)

    errors_arr = np.array(errors)
    abs_arr = np.array(abs_errors)
    max_rel = float(np.nanmax(errors_arr))
    mean_rel = float(np.nanmean(errors_arr))
    max_abs = float(np.nanmax(abs_arr))

    ok = bool(max_rel <= rtol or max_abs <= atol)
    return GradCheckResult(ok, max_rel, mean_rel, max_abs, per_field)


def check_function_gradients(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    df: Callable[[np.ndarray], np.ndarray],
    eps: float = 1e-6,
    atol: float = 1e-6,
    rtol: float = 1e-3,
) -> GradCheckResult:
    """Verify an analytic Jacobian ``df`` against central-difference FD.

    ``f`` and ``df`` operate on flat NumPy arrays of the same shape. The
    convention is ``y = f(x)`` (any shape) with summary loss
    ``L = y.dot(w)`` for a randomly chosen weight vector ``w`` (independent
    of `x`), so ``dL/dx = J^T w`` for ``J = df(x)``. Perturbing each entry
    of ``x`` by ``±eps`` measures the numeric Jacobian-vector product and
    compares elementwise to ``df(x) * w``. Used to verify composite
    gradients (PPO clipped surrogate, REINFORCE policy gradient) without
    having to build a fake autodiff engine.

    Args:
        f: Function ``x -> y`` (same-shape arrays). The checked quantities
            are the entries of the analytic gradient multiplied by the
            same random weight vector ``w`` used to weight the loss.
        df: Analytic Jacobian ``x -> J`` where ``J`` has the same shape as
            `x` * `y`. For elementwise scalar `y` (one-sample loss) this
            is just ``dL/dx``.
        eps: Perturbation step size.
        atol: Absolute tolerance below which ``ok`` stays True even when
            relative error is undefined.
        rtol: Relative tolerance.

    Returns:
        ``GradCheckResult`` giving max / mean rel error and ``ok`` flag.
    """
    x = np.asarray(x, dtype=np.float64)
    floor = max(atol, eps, 1e-12)

    analytic = np.array(df(x), dtype=np.float64)
    numeric = np.zeros_like(x, dtype=np.float64)

    for idx in np.ndindex(x.shape):
        orig = x[idx]
        x[idx] = orig + eps
        lp = float(np.sum(f(x)))
        x[idx] = orig - eps
        lm = float(np.sum(f(x)))
        x[idx] = orig
        numeric[idx] = (lp - lm) / (2.0 * eps)

    r = _rel_error(analytic, numeric, floor)
    abs_e = np.abs(analytic - numeric)
    max_rel = float(np.nanmax(r))
    mean_rel = float(np.nanmean(r))
    max_abs = float(np.nanmax(abs_e))
    ok = bool(max_rel <= rtol or max_abs <= atol)
    return GradCheckResult(ok, max_rel, mean_rel, max_abs)
