"""Unit tests for CALM modular cross-attention architecture and diagnostic pipeline."""

from __future__ import annotations

import pytest
import torch

from src.models.calm_mini import (
    CALMCompositionalWrapper,
    CALMCrossAttentionLayer,
    CALMMiniCrossAttention,
    compute_attention_allocation,
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


def test_attention_weights_sum_to_one():
    """Verify that CALM cross-attention weights sum strictly to 1.0 across keys."""
    layer = CALMCrossAttentionLayer(anchor_dim=2048, augment_dim=384, attn_dim=512, num_heads=8)
    anchor = torch.randn(2, 7, 2048)
    augment = torch.randn(2, 25, 384)

    out, attn_weights = layer(anchor, augment, return_attention_weights=True)

    assert attn_weights.shape == (2, 8, 7, 25)
    sums = attn_weights.sum(dim=-1)
    expected = torch.ones_like(sums)
    assert torch.allclose(sums, expected, atol=1e-5)


def test_compute_attention_allocation_synthetic_orthogonal():
    """Verify compute_attention_allocation calculates exact mass on orthogonal synthetic attention."""
    # Pure gold attention: first 5 tokens receive all attention mass
    attn_pure_gold = torch.zeros(1, 1, 1, 10)
    attn_pure_gold[0, 0, 0, :5] = 0.2

    alloc_gold = compute_attention_allocation(attn_pure_gold, (0, 5), (5, 10))
    assert alloc_gold["gold_attention_mass"] == pytest.approx(1.0, abs=1e-4)
    assert alloc_gold["distractor_attention_mass"] == pytest.approx(0.0, abs=1e-4)
    assert alloc_gold["leakage_ratio"] == pytest.approx(0.0, abs=1e-4)

    # Pure distractor attention: last 5 tokens receive all attention mass
    attn_pure_dist = torch.zeros(1, 1, 1, 10)
    attn_pure_dist[0, 0, 0, 5:] = 0.2

    alloc_dist = compute_attention_allocation(attn_pure_dist, (0, 5), (5, 10))
    assert alloc_dist["gold_attention_mass"] == pytest.approx(0.0, abs=1e-4)
    assert alloc_dist["distractor_attention_mass"] == pytest.approx(1.0, abs=1e-4)
    assert alloc_dist["leakage_ratio"] == pytest.approx(1.0, abs=1e-4)

    # Balanced 50/50 allocation
    attn_half = torch.full((1, 1, 1, 10), 0.1)
    alloc_half = compute_attention_allocation(attn_half, (0, 5), (5, 10))
    assert alloc_half["gold_attention_mass"] == pytest.approx(0.5, abs=1e-4)
    assert alloc_half["distractor_attention_mass"] == pytest.approx(0.5, abs=1e-4)
    assert alloc_half["leakage_ratio"] == pytest.approx(0.5, abs=1e-4)


def test_gradient_isolation_with_attention_weights():
    """Verify base anchor and augment tensors remain frozen when return_attention_weights=True."""
    wrapper = CALMCompositionalWrapper(anchor_dim=2048, augment_dim=384, attn_dim=512, num_heads=8)
    with torch.no_grad():
        wrapper.cross_attention.gamma.data.fill_(1.0)

    anchor = torch.randn(2, 6, 2048, requires_grad=False)
    augment = torch.randn(2, 14, 384, requires_grad=False)

    out, attn_weights = wrapper(anchor, augment, return_attention_weights=True)
    loss = out.sum() + attn_weights.sum()
    loss.backward()

    # Base tensors must remain completely frozen
    assert anchor.grad is None
    assert augment.grad is None

    # Trainable adapter projection matrices receive gradients
    assert wrapper.cross_attention.q_proj.weight.grad is not None
    assert wrapper.cross_attention.k_proj.weight.grad is not None
    assert wrapper.cross_attention.v_proj.weight.grad is not None
    assert wrapper.cross_attention.out_proj.weight.grad is not None


def test_cli_inspect_attention_flag():
    """Verify CLI main entrypoint executes --inspect-attention successfully with limit 2."""
    exit_code = main(["--inspect-attention", "--limit", "2"])
    assert exit_code == 0
