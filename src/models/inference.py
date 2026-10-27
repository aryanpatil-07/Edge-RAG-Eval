"""Inference module for dense embedding extraction and similarity computation.

Wraps sentence-transformers models with automatic device selection,
normalized embeddings, and latency profiling.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def detect_device(preferred_device: Optional[str] = None) -> str:
    """Detect the most performant compute device available.

    Args:
        preferred_device: Optional explicit device string ('cuda', 'mps', 'cpu', or 'auto').

    Returns:
        Device identifier string suitable for PyTorch and Transformers.
    """
    if preferred_device and preferred_device != "auto":
        return preferred_device

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EmbeddingClient:
    """Client wrapper around SentenceTransformer models for retrieval benchmarking.

    Attributes:
        model_name: Hugging Face model repository identifier or local path.
        device: Resolved compute device used for inference.
        model: Underlying SentenceTransformer instance.
        dimension: Dimensionality of output embeddings (e.g., 384 for all-MiniLM-L6-v2).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ) -> None:
        """Initialize the EmbeddingClient with target model and device.

        Args:
            model_name: Name or path of the embedding model.
            device: Compute device identifier or 'auto' for automatic detection.
        """
        self.model_name = model_name
        self.device = detect_device(device)
        logger.info(
            "Initializing EmbeddingClient(model='%s', device='%s')",
            self.model_name,
            self.device,
        )

        load_start = time.perf_counter()
        self.model = SentenceTransformer(self.model_name, device=self.device)
        load_time_ms = (time.perf_counter() - load_start) * 1000.0

        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(
            "Model loaded successfully in %.2f ms. Embedding dimension: %d",
            load_time_ms,
            self.dimension,
        )

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode input text or batch of texts into dense vector embeddings.

        Args:
            texts: Single string or list of text strings to embed.
            normalize: Whether to return L2-normalized embeddings. Defaults to True.
            batch_size: Batch size for inference execution. Defaults to 32.

        Returns:
            np.ndarray: Matrix of embeddings with shape (N, dimension).
        """
        if isinstance(texts, str):
            input_texts = [texts]
        else:
            input_texts = list(texts)

        start_time = time.perf_counter()
        embeddings = self.model.encode(
            input_texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)

        per_sample_ms = latency_ms / max(len(input_texts), 1)
        logger.info(
            "Encoded %d sample(s) -> shape %s in %.2f ms (%.2f ms/sample)",
            len(input_texts),
            embeddings.shape,
            latency_ms,
            per_sample_ms,
        )

        return embeddings
