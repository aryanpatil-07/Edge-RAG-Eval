"""Model architectures and inference clients for Edge-RAG-Eval."""

from src.models.calm_mini import CALMMiniCrossAttention
from src.models.inference import EmbeddingClient, detect_device

__all__ = [
    "EmbeddingClient",
    "detect_device",
    "CALMMiniCrossAttention",
]
