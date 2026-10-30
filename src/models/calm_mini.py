"""Modular CALM (Composition to Augment Language Models) cross-attention module.

Replicates the core architectural blueprint of CALM (Bansal et al., Google DeepMind, 2024,
arXiv:2401.02412). Provides lightweight, zero-initialized gated cross-attention adapters
to compose representations between frozen edge anchors (Gemma-2B) and augmenting
retrieval contexts (MiniLM).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CALMCrossAttentionLayer(nn.Module):
    """Modular CALM cross-attention layer with zero-initialized gating.

    Computes:
        Query = H_anchor * W_Q
        Key   = H_aug * W_K
        Value = H_aug * W_V
        AttentionOutput = softmax(Query * Key^T / sqrt(d_k)) * Value
        H_tilde = H_anchor + gamma * (AttentionOutput * W_O)

    Attributes:
        anchor_dim: Dimensionality of anchor model hidden states (e.g., 2048 for Gemma-2B).
        augment_dim: Dimensionality of augmenting retrieved representations (e.g., 384 for all-MiniLM-L6-v2).
        attn_dim: Dimensionality of cross-attention projection space (e.g., 512).
        num_heads: Number of attention heads (e.g., 8).
        head_dim: Dimensionality per head (attn_dim // num_heads = 64).
        gamma: Learned scalar gating parameter, initialized to 0.0.
    """

    def __init__(
        self,
        anchor_dim: int = 2048,
        augment_dim: int = 384,
        attn_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        """Initialize CALMCrossAttentionLayer.

        Args:
            anchor_dim: Hidden dimension of base anchor LLM (Gemma-2B = 2048).
            augment_dim: Hidden dimension of augmenting context encoder (MiniLM = 384).
            attn_dim: Cross-attention projection dimension (default: 512).
            num_heads: Number of cross-attention heads (default: 8).
            dropout: Dropout probability applied to attention weights.
        """
        super().__init__()
        if attn_dim % num_heads != 0:
            raise ValueError(f"attn_dim ({attn_dim}) must be divisible by num_heads ({num_heads})")

        self.anchor_dim = anchor_dim
        self.augment_dim = augment_dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads

        # Linear projections
        self.q_proj = nn.Linear(anchor_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(augment_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(augment_dim, attn_dim, bias=False)
        self.out_proj = nn.Linear(attn_dim, anchor_dim, bias=False)

        # Learned gating scalar initialized strictly to 0.0 (preserves base model behavior at init)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialize projection weights with scaled Gaussian
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projection matrices with Xavier normal initialization."""
        nn.init.xavier_normal_(self.q_proj.weight)
        nn.init.xavier_normal_(self.k_proj.weight)
        nn.init.xavier_normal_(self.v_proj.weight)
        nn.init.xavier_normal_(self.out_proj.weight)

    def forward(
        self,
        anchor_hidden_states: torch.Tensor,
        augment_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Execute forward cross-attention and gated residual connection.

        Args:
            anchor_hidden_states: Anchor representations of shape (batch, seq_len, anchor_dim).
            augment_hidden_states: Augmenting representations of shape (batch, ctx_len, augment_dim).
            attention_mask: Optional attention mask of shape (batch, 1, seq_len, ctx_len) or (batch, ctx_len).
            return_attention_weights: Whether to return the normalized cross-attention weights tensor.

        Returns:
            torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
                - Gated augmented representation of shape (batch, seq_len, anchor_dim) if return_attention_weights is False.
                - Tuple of (output, attention_weights) of shape (batch, num_heads, seq_len, ctx_len) if return_attention_weights is True.
        """
        batch_size, seq_len, _ = anchor_hidden_states.shape
        _, ctx_len, _ = augment_hidden_states.shape

        # Projections
        # Q: (B, seq_len, attn_dim) -> (B, num_heads, seq_len, head_dim)
        q = self.q_proj(anchor_hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # K, V: (B, ctx_len, attn_dim) -> (B, num_heads, ctx_len, head_dim)
        k = self.k_proj(augment_hidden_states).view(
            batch_size, ctx_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(augment_hidden_states).view(
            batch_size, ctx_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # (B, ctx_len) -> (B, 1, 1, ctx_len)
                mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                mask = attention_mask[:, None, :, :]
            else:
                mask = attention_mask
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        dropped_attn = self.dropout(attn_weights)

        # Attention context: (B, num_heads, seq_len, head_dim)
        context = torch.matmul(dropped_attn, v)

        # Reshape to (B, seq_len, attn_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.attn_dim)

        # Project back to anchor_dim: (B, seq_len, anchor_dim)
        attn_output = self.out_proj(context)

        # Gated residual: H_tilde = H_anchor + gamma * AttentionOutput
        output = anchor_hidden_states + self.gamma * attn_output

        if return_attention_weights:
            return output, attn_weights
        return output

    @property
    def trainable_parameters(self) -> int:
        """Return total number of trainable parameters in this layer."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CALMCompositionalWrapper(nn.Module):
    """Compositional wrapper integrating frozen base model with CALM cross-attention adapter.

    Maintains frozen representations and computes parameter breakdowns to verify
    that trainable weights remain strictly under 1% of base model capacity.
    """

    def __init__(
        self,
        anchor_dim: int = 2048,
        augment_dim: int = 384,
        attn_dim: int = 512,
        num_heads: int = 8,
        base_model_params: int = 2_000_000_000,
        dropout: float = 0.0,
    ) -> None:
        """Initialize CALMCompositionalWrapper.

        Args:
            anchor_dim: Anchor LLM hidden dimension (default: 2048).
            augment_dim: Augmenting context dimension (default: 384).
            attn_dim: Projection dimension (default: 512).
            num_heads: Number of attention heads (default: 8).
            base_model_params: Reference base model parameter count (default: 2B).
            dropout: Attention dropout rate.
        """
        super().__init__()
        self.anchor_dim = anchor_dim
        self.augment_dim = augment_dim
        self.base_model_params = base_model_params

        self.cross_attention = CALMCrossAttentionLayer(
            anchor_dim=anchor_dim,
            augment_dim=augment_dim,
            attn_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        anchor_hidden_states: torch.Tensor,
        augment_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the cross-attention adapter.

        Args:
            anchor_hidden_states: Anchor representations (B, L, anchor_dim).
            augment_hidden_states: Augment representations (B, L_aug, augment_dim).
            attention_mask: Optional attention mask.
            return_attention_weights: Whether to return attention weights.

        Returns:
            torch.Tensor | Tuple[torch.Tensor, torch.Tensor]: Composed output or (output, attention_weights).
        """
        return self.cross_attention(
            anchor_hidden_states=anchor_hidden_states,
            augment_hidden_states=augment_hidden_states,
            attention_mask=attention_mask,
            return_attention_weights=return_attention_weights,
        )

    def get_parameter_breakdown(self) -> Dict[str, Any]:
        """Compute parameter statistics and efficiency ratios.

        Returns:
            Dict[str, Any]: Dictionary containing parameter counts and compliance flags.
        """
        trainable = self.cross_attention.trainable_parameters
        total = self.base_model_params + trainable
        trainable_pct = (trainable / total) * 100.0

        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": self.base_model_params,
            "trainable_percentage": trainable_pct,
            "is_under_1_percent": trainable_pct < 1.0,
            "is_under_5m": trainable < 5_000_000,
            "gamma_value": float(self.cross_attention.gamma.item()),
            "anchor_dim": self.anchor_dim,
            "augment_dim": self.augment_dim,
            "attn_dim": self.cross_attention.attn_dim,
            "num_heads": self.cross_attention.num_heads,
        }


# Alias for backward compatibility
CALMMiniCrossAttention = CALMCrossAttentionLayer


# ---------------------------------------------------------------------------
# Attention Allocation Inspection Utility
# ---------------------------------------------------------------------------

def compute_attention_allocation(
    attention_weights: torch.Tensor,
    gold_token_span: tuple[int, int],
    distractor_token_span: tuple[int, int],
) -> dict[str, float]:
    """Computes the percentage of attention mass allocated to Gold vs Distractor.

    Args:
        attention_weights: Cross-attention weights tensor of shape (B, H, L_anchor, L_aug),
            (H, L_anchor, L_aug), or (L_anchor, L_aug).
        gold_token_span: (start_idx, end_idx) token span in augment keys for gold context.
        distractor_token_span: (start_idx, end_idx) token span in augment keys for distractor context.

    Returns:
        dict[str, float]:
            - 'gold_attention_mass': float in [0.0, 1.0]
            - 'distractor_attention_mass': float in [0.0, 1.0]
            - 'leakage_ratio': float in [0.0, 1.0] representing distractor_mass / (gold_mass + distractor_mass)
    """
    if not isinstance(attention_weights, torch.Tensor):
        attention_weights = torch.tensor(attention_weights, dtype=torch.float32)

    # Average attention across batch, head, and anchor sequence dimensions -> (L_aug,)
    if attention_weights.dim() == 4:
        mean_attn = attention_weights.mean(dim=(0, 1, 2))
    elif attention_weights.dim() == 3:
        mean_attn = attention_weights.mean(dim=(0, 1))
    elif attention_weights.dim() == 2:
        mean_attn = attention_weights.mean(dim=0)
    elif attention_weights.dim() == 1:
        mean_attn = attention_weights
    else:
        raise ValueError(f"Unsupported attention weights shape: {attention_weights.shape}")

    total_keys = len(mean_attn)
    g_start, g_end = gold_token_span
    d_start, d_end = distractor_token_span

    # Safe index bounds
    g_start = max(0, min(g_start, total_keys))
    g_end = max(0, min(g_end, total_keys))
    d_start = max(0, min(d_start, total_keys))
    d_end = max(0, min(d_end, total_keys))

    gold_mass = float(mean_attn[g_start:g_end].sum().item()) if g_end > g_start else 0.0
    distractor_mass = float(mean_attn[d_start:d_end].sum().item()) if d_end > d_start else 0.0

    total_mass = gold_mass + distractor_mass
    leakage = float(distractor_mass / total_mass) if total_mass > 0.0 else 0.0

    return {
        "gold_attention_mass": round(gold_mass, 4),
        "distractor_attention_mass": round(distractor_mass, 4),
        "leakage_ratio": round(leakage, 4),
    }
