# SOTA Validation Tests — Implement Against PyTorch

## Goal
Implement the validation tests that compare numpy-dl implementations against PyTorch equivalents with identical weights and inputs.

## Scope

### tests/validation/test_layers.py
- Compare Dense forward pass against `torch.nn.Linear` with same weights
- Compare Dense backward gradients against PyTorch autograd
- Share weights via numpy arrays, compare outputs to atol=1e-6

### tests/validation/test_activations.py
- Compare ReLU forward/backward against PyTorch
- Compare Sigmoid forward/backward against PyTorch
- Compare Softmax output against PyTorch softmax

### tests/validation/test_optimizers.py
- Compare SGD weight update against torch.optim.SGD
- Compare Adam weight update against torch.optim.Adam (including bias correction)

### tests/validation/test_mlp.py
- End-to-end 2-layer MLP: forward + backward
- Same weights, same input, same loss → compare gradients

## Approach
- Use `np.testing.assert_allclose(atol=1e-6)` for numerical comparison
- Share weights: convert numpy ↔ torch tensors
- Skip if PyTorch not installed (already handled by importorskip)
- Remove TensorFlow dependency — PyTorch only (cleaner)
