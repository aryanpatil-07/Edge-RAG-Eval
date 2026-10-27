"""Model architectures and inference clients for Edge-RAG-Eval."""

from src.models.calm_mini import CALMMiniCrossAttention
from src.models.inference import (
    EmbeddingClient,
    GenerationResult,
    GemmaInferenceClient,
    check_ollama,
    detect_device,
)

__all__ = [
    "EmbeddingClient",
    "GemmaInferenceClient",
    "GenerationResult",
    "check_ollama",
    "detect_device",
    "CALMMiniCrossAttention",
]
