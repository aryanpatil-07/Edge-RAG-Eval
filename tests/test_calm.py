"""Unit tests for CALM modular cross-attention architecture and diagnostic pipeline."""

from __future__ import annotations

import pytest
import torch

from src.models.calm_mini import (
    CALMCompositionalWrapper,
    CALMCrossAttentionLayer,
    CALMMiniCrossAttention,
)
from src.pipeline import main, run_calm_test


def test_calm_forward_dimension_preservation():
    """Verify that CALMCrossAttentionLayer preserves anchor hidden dimension (2048)."""
    batch_size = 2
    seq_len = 16
    ctx_len = 32
    anchor_dim = 2048
    augment_dim = 384

    layer = CALMCrossAttentionLayer(
        anchor_dim=anchor_dim,
        augment_dim=augment_dim,
        attn_dim=512,
        num_heads=8,
    )

    anchor = torch.randn(batch_size, seq_len, anchor_dim)
    augment = torch.randn(batch_size, ctx_len, augment_dim)

    out = layer(anchor, augment)

    assert out.shape == (batch_size, seq_len, anchor_dim)
    assert out.dtype == anchor.dtype


def test_calm_variable_sequence_lengths():
    """Verify layer handles varied batch sizes and context lengths correctly."""
    layer = CALMCrossAttentionLayer(anchor_dim=2048, augment_dim=384, attn_dim=512, num_heads=8)

    test_shapes = [
        ((1, 8, 2048), (1, 64, 384)),
        ((4, 32, 2048), (4, 16, 384)),
        ((1, 1, 2048), (1, 1, 384)),
    ]

    for (b_a, s_a, d_a), (b_g, s_g, d_g) in test_shapes:
        anchor = torch.randn(b_a, s_a, d_a)
        augment = torch.randn(b_g, s_g, d_g)
        out = layer(anchor, augment)
        assert out.shape == (b_a, s_a, d_a)


def test_calm_zero_init_equivalence():
    """Verify zero degradation at init: output is exactly equal to anchor states when gamma=0."""
    layer = CALMCrossAttentionLayer(anchor_dim=2048, augment_dim=384, attn_dim=512, num_heads=8)

    assert layer.gamma.item() == 0.0

    anchor = torch.randn(3, 12, 2048)
    augment = torch.randn(3, 24, 384)

    out = layer(anchor, augment)

    # Must be exact bit-for-bit equivalence because gamma is 0.0
    assert torch.equal(out, anchor)


def test_calm_parameter_budget_under_5m():
    """Verify trainable parameter count is strictly under 5M and <1% of base model."""
    layer = CALMCrossAttentionLayer(anchor_dim=2048, augment_dim=384, attn_dim=512, num_heads=8)
    wrapper = CALMCompositionalWrapper(
        anchor_dim=2048,
        augment_dim=384,
        attn_dim=512,
        num_heads=8,
        base_model_params=2_000_000_000,
    )

    trainable_params = layer.trainable_parameters

    # Hard requirements: <5M parameters and <1.0% of Gemma-2B capacity
    assert trainable_params < 5_000_000
    assert trainable_params == 2_490_369  # Exact analytical count

    breakdown = wrapper.get_parameter_breakdown()
    assert breakdown["is_under_5m"] is True
    assert breakdown["is_under_1_percent"] is True
    assert breakdown["trainable_percentage"] < 1.0
    assert pytest.approx(breakdown["trainable_percentage"], rel=1e-2) == 0.124


def test_calm_gradient_isolation():
    """Verify base tensors remain frozen (grad is None) while gradients flow to adapter weights."""
    layer = CALMCrossAttentionLayer(anchor_dim=2048, augment_dim=384, attn_dim=512, num_heads=8)

    # Base tensors with requires_grad=False (representing frozen models)
    anchor = torch.randn(2, 16, 2048, requires_grad=False)
    augment = torch.randn(2, 32, 384, requires_grad=False)

    # Step 1: Backward with gamma = 0.0 verifies gamma receives gradient
    out = layer(anchor, augment)
    loss = out.sum()
    loss.backward()

    assert anchor.grad is None
    assert augment.grad is None
    assert layer.gamma.grad is not None

    # Step 2: Backward with active gamma (gamma=1.0) verifies projection matrices receive gradients
    with torch.no_grad():
        layer.gamma.data.fill_(1.0)
    layer.zero_grad()

    out_active = layer(anchor, augment)
    loss_active = out_active.sum()
    loss_active.backward()

    # Anchor and augment tensors must still have grad is None
    assert anchor.grad is None
    assert augment.grad is None

    # Adapter projection matrices must have active gradients
    assert layer.q_proj.weight.grad is not None
    assert layer.k_proj.weight.grad is not None
    assert layer.v_proj.weight.grad is not None
    assert layer.out_proj.weight.grad is not None
    assert layer.gamma.grad is not None


def test_calm_with_attention_mask():
    """Verify cross-attention layer supports 2D and 3D attention masks."""
    layer = CALMCrossAttentionLayer(anchor_dim=2048, augment_dim=384, attn_dim=512, num_heads=8)

    batch_size = 2
    seq_len = 8
    ctx_len = 16

    anchor = torch.randn(batch_size, seq_len, 2048)
    augment = torch.randn(batch_size, ctx_len, 384)

    # 2D mask: (batch_size, ctx_len)
    mask_2d = torch.ones(batch_size, ctx_len)
    mask_2d[:, 12:] = 0  # Mask out last 4 context tokens
    out_2d = layer(anchor, augment, attention_mask=mask_2d)
    assert out_2d.shape == (batch_size, seq_len, 2048)

    # 3D mask: (batch_size, seq_len, ctx_len)
    mask_3d = torch.ones(batch_size, seq_len, ctx_len)
    out_3d = layer(anchor, augment, attention_mask=mask_3d)
    assert out_3d.shape == (batch_size, seq_len, 2048)


def test_calm_wrapper_forward():
    """Verify CALMCompositionalWrapper executes cleanly and matches layer output."""
    wrapper = CALMCompositionalWrapper(anchor_dim=2048, augment_dim=384, attn_dim=512, num_heads=8)

    anchor = torch.randn(2, 16, 2048)
    augment = torch.randn(2, 32, 384)

    out = wrapper(anchor, augment)
    assert out.shape == (2, 16, 2048)
    assert torch.equal(out, anchor)  # Zero-init


def test_calm_invalid_head_dimension():
    """Verify ValueError is raised when attn_dim is not divisible by num_heads."""
    with pytest.raises(ValueError, match="must be divisible by num_heads"):
        CALMCrossAttentionLayer(anchor_dim=2048, augment_dim=384, attn_dim=500, num_heads=8)


def test_calm_alias():
    """Verify backward compatibility alias CALMMiniCrossAttention points to the same class."""
    assert CALMMiniCrossAttention is CALMCrossAttentionLayer


def test_calm_diagnostic_runner():
    """Verify run_calm_test() executes without error and returns exit code 0."""
    exit_code = run_calm_test(anchor_dim=2048, augment_dim=384, attn_dim=512, num_heads=8)
    assert exit_code == 0


def test_cli_test_calm_flag():
    """Verify CLI main entrypoint executes --test-calm successfully with exit code 0."""
    exit_code = main(["--test-calm"])
    assert exit_code == 0
