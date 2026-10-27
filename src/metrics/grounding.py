"""Retrieval grounding evaluation metrics."""

from __future__ import annotations

import numpy as np


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D or single-row 2D vectors.

    Args:
        a: First vector of shape (D,) or (1, D).
        b: Second vector of shape (D,) or (1, D).

    Returns:
        float: Cosine similarity score in range [-1.0, 1.0].
    """
    v1 = np.asarray(a).flatten()
    v2 = np.asarray(b).flatten()

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))


def evaluate_grounding_precision(
    query_emb: np.ndarray,
    candidate_embs: np.ndarray,
    gold_index: int,
    top_k: int = 1,
) -> float:
    """Evaluate whether the gold ground-truth document is retrieved in top-k.

    Args:
        query_emb: Query embedding vector (1, D) or (D,).
        candidate_embs: Candidate document embeddings (N, D).
        gold_index: Index of the true relevant context document.
        top_k: Rank threshold for hit calculation.

    Returns:
        float: 1.0 if gold document is within top-k ranks, else 0.0.
    """
    q = np.asarray(query_emb).reshape(1, -1)
    q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

    docs = np.asarray(candidate_embs)
    docs_norm = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-12)

    similarities = np.dot(docs_norm, q_norm.T).flatten()
    ranked_indices = np.argsort(-similarities)[:top_k]

    return 1.0 if gold_index in ranked_indices else 0.0
