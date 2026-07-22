"""Numerical gradient checks for the hand-derived backward passes.

Verifies the analytic gradients used in the library against
central-difference finite differences. Lives here because the library
is autodiff-free: every backward pass is hand-derived, so a regression
in any of them would silently produce a broken learner. Covering both
the core NN layers and the composite RL gradients (REINFORCE policy
gradient and PPO clipped surrogate) is the highest-value safety net for
a 1.0 release.

See ``numpy_dl.nn.gradcheck`` for the utility used here.
"""

from __future__ import annotations

import numpy as np

from numpy_dl.nn.activations import ReLU, Sigmoid, softmax
from numpy_dl.nn.gradcheck import (
    GradCheckResult,
    check_function_gradients,
    check_layer_gradients,
)
from numpy_dl.nn.layers import Dense

# --------------------------------------------------------------------------- #
# NN layer backward passes
# --------------------------------------------------------------------------- #


class TestLayerGradCheck:
    def test_dense_gradients_match_fd(self) -> None:
        rng = np.random.RandomState(0)
        layer = Dense(3, 4)
        x = rng.randn(5, 3)
        dy = rng.randn(5, 4)
        result = check_layer_gradients(layer, x, dy)
        assert result.ok, f"Dense backward mismatch:\n{result.summary}"
        assert result.max_rel_error < 1e-4, result.summary

    def test_relu_input_gradient_matches_fd(self) -> None:
        rng = np.random.RandomState(1)
        layer = ReLU()
        x = rng.randn(4, 6)
        dy = rng.randn(4, 6)
        result = check_layer_gradients(layer, x, dy)
        assert result.ok, f"ReLU dx mismatch:\n{result.summary}"

    def test_relu_zero_gradient_for_negative_inputs(self) -> None:
        # All-negative inputs => exact-zero gradient everywhere. The FD check
        # should still report ok via the absolute-tolerance floor.
        layer = ReLU()
        x = -np.abs(np.random.RandomState(2).randn(3, 4))
        dy = np.random.RandomState(3).randn(3, 4)
        result = check_layer_gradients(layer, x, dy)
        assert result.ok, result.summary

    def test_sigmoid_input_gradient_matches_fd(self) -> None:
        rng = np.random.RandomState(5)
        layer = Sigmoid()
        x = rng.randn(4, 5)
        dy = rng.randn(4, 5)
        result = check_layer_gradients(layer, x, dy)
        assert result.ok, f"Sigmoid dx mismatch:\n{result.summary}"
        assert result.max_rel_error < 1e-4, result.summary


# --------------------------------------------------------------------------- #
# Meta-test: gradcheck itself must actually catch a wrong gradient.
# --------------------------------------------------------------------------- #


class TestGradCheckCatchesBugs:
    class _BadDense:
        """Dense-like layer with a deliberately wrong dW (off by a constant)."""

        def __init__(self) -> None:
            self.W = np.random.RandomState(0).randn(3, 4)
            self.b = np.zeros((1, 4))
            self.dW = np.zeros_like(self.W)
            self.db = np.zeros_like(self.b)
            self._input = None

        def forward(self, x: np.ndarray) -> np.ndarray:
            self._input = x
            return x @ self.W + self.b

        def backward(self, dy: np.ndarray) -> np.ndarray:
            assert self._input is not None
            self.dW = self._input.T @ dy + 0.5  # deliberate incorrect bias
            self.db = np.sum(dy, axis=0, keepdims=True)
            return dy @ self.W.T

        def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
            return [(self.W, self.dW), (self.b, self.db)]

    def test_gradcheck_fails_for_wrong_dW(self) -> None:
        rng = np.random.RandomState(0)
        layer = self._BadDense()
        x = rng.randn(5, 3)
        dy = rng.randn(5, 4)
        result = check_layer_gradients(layer, x, dy)
        assert not result.ok, (
            "GradCheck should have flagged the wrong dW; got "
            f"max_rel={result.max_rel_error:.2e}"
        )


# --------------------------------------------------------------------------- #
# REINFORCE policy gradient
# --------------------------------------------------------------------------- #


class TestReinforcePolicyGradient:
    """Policy-gradient gradient w.r.t. logits.

    For one-hot action targets with softmax policy, and per-step return G_t:
        L(logits)      = -sum_t G_t * log_probs[t, a_t]
        dL/dlogits[t,a] = -G_t * (one_hot[t,a] - probs[t,a])
    The implementation in reinforce.py uses exactly this form.
    """

    def test_policy_gradient_matches_fd(self) -> None:
        rng = np.random.RandomState(0)
        B, A = 6, 3
        logits = rng.randn(B, A)
        actions = rng.randint(0, A, size=B)
        returns = rng.randn(B)

        def loss(zz: np.ndarray) -> float:
            probs = softmax(zz)
            log_probs = np.log(probs + 1e-8)
            taken = log_probs[np.arange(B), actions]
            return float(-(returns * taken).sum())

        def analytic_grad(zz: np.ndarray) -> np.ndarray:
            probs = softmax(zz)
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(B), actions] = 1.0
            return -returns[:, np.newaxis] * (one_hot - probs)

        result = check_function_gradients(loss, logits, analytic_grad)
        assert result.ok, f"REINFORCE policy gradient mismatch:\n{result.summary}"
        assert result.max_rel_error < 1e-5, result.summary


# --------------------------------------------------------------------------- #
# PPO clipped surrogate
# --------------------------------------------------------------------------- #


class TestPPOClippedSurrogateGradient:
    """Gradient of the PPO clipped surrogate + entropy bonus w.r.t. logits.

    Implementation in ppo.py minimizes ``L = sum_t min(r*A, clip(r)*A) -
    ent_coeff * sum_t H(p_t)`` (Adam descends ``dL/dz``). The analytic
    gradient of ``sum_t H(p_t)`` w.r.t. ``logits[t, k]`` is
    ``-p_{t,k} * (log p_{t,k} + H_t)`` per row (where ``H_t = H(p_t)`` is a
    vector of shape ``(B,)``). So the correct entropy term in the actor
    gradient is ``+ent_coeff * probs * (log_probs + H_t per row)``.

    The original ppo.py used ``probs * (log_probs + mean_entropy)`` where
    ``mean_entropy = mean_t H_t`` is a scalar -- that does not match the
    per-row form. These tests verify: (a) the surrogate term alone already
    matches FD; (b) the code as written does NOT match FD when entropy is
    on; (c) the corrected per-row form matches FD.
    """

    @staticmethod
    def _build(B: int, A: int, seed: int, eps_clip: float = 0.2):
        rng = np.random.RandomState(seed)
        logits0 = rng.randn(B, A)
        actions = rng.randint(0, A, size=B)
        advantages = rng.randn(B)
        # old_log_probs chosen so ratios are a mix of unclipped / clipped.
        old_log_probs = rng.randn(B) * 0.5
        return logits0, actions, advantages, old_log_probs, eps_clip

    @staticmethod
    def _loss(
        logits: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        old_log_probs: np.ndarray,
        eps_clip: float,
        entropy_coeff: float,
    ) -> float:
        B = logits.shape[0]
        probs = softmax(logits)
        log_probs = np.log(probs + 1e-8)
        taken = log_probs[np.arange(B), actions]
        ratio = np.exp(taken - old_log_probs)
        unclipped = ratio * advantages
        clipped = np.clip(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
        surrogate = float(-np.minimum(unclipped, clipped).sum())
        # Entropy: H(p) = -sum_a p_a * log p_a per sample, summed over batch.
        # This is the loss form against which finite differences are measured;
        # it matches the PPO objective the actor is meant to maximize.
        entropy = float(-(probs * log_probs).sum())
        return surrogate - entropy_coeff * entropy

    @staticmethod
    def _analytic_grad_code(
        logits: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        old_log_probs: np.ndarray,
        eps_clip: float,
        entropy_coeff: float,
    ) -> np.ndarray:
        """Gradient as written in ppo.py (uses scalar mean_entropy in place
        of the per-row H_t the correct derivative requires).
        """
        B = logits.shape[0]
        probs = softmax(logits)
        log_probs = np.log(probs + 1e-8)
        taken = log_probs[np.arange(B), actions]
        old_taken_log_probs = old_log_probs
        ratio = np.exp(taken - old_taken_log_probs)
        unclipped = ratio * advantages
        clipped = np.clip(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(B), actions] = 1.0
        is_unclipped = (unclipped <= clipped).astype(np.float64)
        entropy = -np.sum(probs * log_probs, axis=-1)
        mean_entropy = float(np.mean(entropy))
        surrogate_grad = (
            -advantages[:, np.newaxis]
            * ratio[:, np.newaxis]
            * is_unclipped[:, np.newaxis]
            * (one_hot - probs)
        )
        entropy_grad = entropy_coeff * probs * (log_probs + mean_entropy)
        return surrogate_grad + entropy_grad

    @staticmethod
    def _analytic_grad_correct(
        logits: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        old_log_probs: np.ndarray,
        eps_clip: float,
        entropy_coeff: float,
    ) -> np.ndarray:
        """Mathematically correct gradient: d/dlogits of (-ent_coeff * sum H_t)
        is +ent_coeff * p * (log p + H_t per row).
        """
        B = logits.shape[0]
        probs = softmax(logits)
        log_probs = np.log(probs + 1e-8)
        taken = log_probs[np.arange(B), actions]
        ratio = np.exp(taken - old_log_probs)
        unclipped = ratio * advantages
        clipped = np.clip(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(B), actions] = 1.0
        is_unclipped = (unclipped <= clipped).astype(np.float64)
        surrogate_grad = (
            -advantages[:, np.newaxis]
            * ratio[:, np.newaxis]
            * is_unclipped[:, np.newaxis]
            * (one_hot - probs)
        )
        H_t = -np.sum(probs * log_probs, axis=-1)
        entropy_grad = entropy_coeff * probs * (log_probs + H_t[:, np.newaxis])
        return surrogate_grad + entropy_grad

    def test_surrogate_only_matches_fd(self) -> None:
        """Surrogate term (no entropy bonus) must match FD exactly."""
        logits, actions, advantages, old_log_probs, eps_clip = self._build(8, 3, 0)

        def loss(zz: np.ndarray) -> float:
            return self._loss(
                zz, actions, advantages, old_log_probs, eps_clip, entropy_coeff=0.0
            )

        def analytic(zz: np.ndarray) -> np.ndarray:
            return self._analytic_grad_correct(
                zz, actions, advantages, old_log_probs, eps_clip, entropy_coeff=0.0
            )

        result = check_function_gradients(loss, logits, analytic)
        assert result.ok, f"PPO surrogate-only grad mismatch:\n{result.summary}"
        assert result.max_rel_error < 1e-3, result.summary

    def test_code_form_mismatches_fd_with_entropy(self) -> None:
        """The form coded in ppo.py uses scalar mean_entropy in place of
        the per-row H_t that derivative requires, so it must NOT match FD.
        This is the FD-confirmation that the bug is real, not FD noise."""
        logits, actions, advantages, old_log_probs, eps_clip = self._build(8, 3, 1)
        entropy_coeff = 0.01  # matches the library default

        def loss(zz: np.ndarray) -> float:
            return self._loss(
                zz, actions, advantages, old_log_probs, eps_clip, entropy_coeff
            )

        def grad_as_coded(zz: np.ndarray) -> np.ndarray:
            return self._analytic_grad_code(
                zz, actions, advantages, old_log_probs, eps_clip, entropy_coeff
            )

        result = check_function_gradients(loss, logits, grad_as_coded)
        assert not result.ok, (
            "Expected the ppo.py entropy gradient (scalar mean_entropy) to "
            f"mismatch FD; BUGS.md claims 'consistent with FD step size' but "
            f"FD says it's a real bug: {result.summary}"
        )
        assert result.max_abs_error > 1e-3, (
            "FD mismatch too small to be conclusive: " + result.summary
        )

    def test_correct_form_matches_fd_with_entropy(self) -> None:
        """The corrected per-row form matches FD."""
        logits, actions, advantages, old_log_probs, eps_clip = self._build(8, 3, 2)
        entropy_coeff = 0.01

        def loss(zz: np.ndarray) -> float:
            return self._loss(
                zz, actions, advantages, old_log_probs, eps_clip, entropy_coeff
            )

        def analytic(zz: np.ndarray) -> np.ndarray:
            return self._analytic_grad_correct(
                zz, actions, advantages, old_log_probs, eps_clip, entropy_coeff
            )

        result = check_function_gradients(loss, logits, analytic)
        assert result.ok, f"correct PPO gradient mismatch:\n{result.summary}"
        assert result.max_rel_error < 1e-3, result.summary


# --------------------------------------------------------------------------- #
# Tiny smoke test: the utility functions surface useful return values.
# --------------------------------------------------------------------------- #


class TestGradCheckResult:
    def test_summary_for_ok(self) -> None:
        r = GradCheckResult(True, 1e-9, 1e-10, 1e-11, {"param[0]": 1e-9})
        assert r.summary.startswith("grad check OK")

    def test_summary_for_failure(self) -> None:
        r = GradCheckResult(False, 0.5, 0.1, 0.05, {"param[0]": 0.5})
        assert r.summary.startswith("grad check FAILED")
        assert "param[0]=5.00e-01" in r.summary
