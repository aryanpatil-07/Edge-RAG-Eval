"""Unit and integration tests for Edge-RAG-Eval pipeline and models."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.distractor_sampler import DistractorSampler
from src.metrics.grounding import compute_cosine_similarity, evaluate_grounding_precision
from src.models.inference import EmbeddingClient, detect_device
from src.pipeline import get_system_diagnostics, main, run_smoke_test


def test_detect_device():
    """Verify device selection returns a valid compute device string."""
    device = detect_device()
    assert device in {"cpu", "cuda", "mps"}

    explicit_cpu = detect_device("cpu")
    assert explicit_cpu == "cpu"


def test_system_diagnostics():
    """Verify diagnostic collector returns all expected platform keys."""
    diagnostics = get_system_diagnostics()
    assert "Operating System" in diagnostics
    assert "Python Version" in diagnostics
    assert "PyTorch Version" in diagnostics
    assert "CUDA Available" in diagnostics
    assert "GPU Device" in diagnostics
    assert "Total VRAM" in diagnostics


def test_embedding_client_dimension_and_encoding():
    """Verify EmbeddingClient initializes with all-MiniLM-L6-v2 and outputs (1, 384)."""
    client = EmbeddingClient(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    assert client.dimension == 384

    test_text = "Edge-RAG-Eval: Decoupled retrieval grounding testbed."
    embedding = client.encode(test_text)

    # Shape assertion
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1, 384)

    # Normalization check (L2 norm should be approximately 1.0)
    norm = np.linalg.norm(embedding[0])
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_embedding_client_batch_encoding():
    """Verify EmbeddingClient encodes batch inputs accurately."""
    client = EmbeddingClient(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    texts = [
        "First document about neural edge computing.",
        "Second document regarding context injection.",
    ]
    batch_embeddings = client.encode(texts)
    assert batch_embeddings.shape == (2, 384)


def test_smoke_test_end_to_end():
    """Verify end-to-end smoke test executes cleanly and returns exit code 0."""
    exit_code = run_smoke_test(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    assert exit_code == 0


def test_cli_test_flag():
    """Verify CLI main entrypoint executes --test successfully."""
    exit_code = main(["--test"])
    assert exit_code == 0


def test_distractor_sampler():
    """Verify distractor sampler mixes gold and distractor passages."""
    corpus = [f"Distractor document {i}" for i in range(10)]
    sampler = DistractorSampler(corpus=corpus, seed=123)

    gold = "True gold document for grounding."
    mixed = sampler.inject_noise(gold_context=gold, num_distractors=4, shuffle=True)

    assert len(mixed) == 5
    assert gold in mixed


def test_grounding_metrics():
    """Verify cosine similarity and top-k precision metrics."""
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])
    vec3 = np.array([0.0, 1.0, 0.0])

    assert pytest.approx(compute_cosine_similarity(vec1, vec2)) == 1.0
    assert pytest.approx(compute_cosine_similarity(vec1, vec3)) == 0.0

    candidates = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    query = np.array([1.0, 0.0])
    assert evaluate_grounding_precision(query, candidates, gold_index=0, top_k=1) == 1.0
    assert evaluate_grounding_precision(query, candidates, gold_index=1, top_k=1) == 0.0
