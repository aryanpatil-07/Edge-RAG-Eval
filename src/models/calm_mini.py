"""Modular CALM (Composition to Augment Language Models) cross-attention module.

Provides lightweight cross-attention adapters to fuse representations
from edge retrieval anchors and foundation models (e.g., Gemma-2B).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CALMMiniCrossAttention(nn.Module):
    """Lightweight cross-attention adapter layer for modular knowledge composition.

    Projects anchor model representations into the target model's latent space,
    enabling edge RAG grounding without fine-tuning full base weights.
    """

    def __init__(
        self,
        anchor_dim: int = 384,
        target_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the cross-attention adapter.

        Args:
            anchor_dim: Hidden dimension of the retrieval anchor model (e.g., 384).
            target_dim: Hidden dimension of the target edge LLM (e.g., 2048 for Gemma-2B).
            num_heads: Number of cross-attention heads.
            dropout: Attention dropout probability.
        """
        super().__init__()
        self.anchor_dim = anchor_dim
        self.target_dim = target_dim
        self.num_heads = num_heads

        self.proj_anchor = nn.Linear(anchor_dim, target_dim, bias=False)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=target_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(target_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        target_hidden_states: torch.Tensor,
        anchor_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Perform cross-attention between target hidden states and anchor context.

        Args:
            target_hidden_states: Tensor of shape (batch_size, seq_len, target_dim).
            anchor_hidden_states: Tensor of shape (batch_size, anchor_seq_len, anchor_dim).
            attention_mask: Optional binary or additive attention mask.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Augmented hidden states and attention weights.
        """
        proj_context = self.proj_anchor(anchor_hidden_states)
        attn_out, attn_weights = self.cross_attn(
            query=target_hidden_states,
            key=proj_context,
            value=proj_context,
            need_weights=True,
        )
        output = self.norm(target_hidden_states + self.dropout(attn_out))
        return output, attn_weights
