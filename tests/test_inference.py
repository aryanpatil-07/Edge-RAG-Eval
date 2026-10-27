"""Unit and integration tests for Gemma-2B deterministic inference engine."""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.models.inference import (
    GenerationResult,
    GemmaInferenceClient,
    check_ollama,
)
from src.pipeline import main, run_generation_test


def test_generation_result_valid_schema():
    """Verify GenerationResult validates schema and serializes accurately."""
    res = GenerationResult(
        response_text="The formula for water is H2O.",
        prompt_tokens=10,
        completion_tokens=7,
        total_latency_ms=250.0,
        tokens_per_second=28.0,
        model_name="gemma:2b",
    )
    assert res.response_text == "The formula for water is H2O."
    assert res.prompt_tokens == 10
    assert res.completion_tokens == 7
    assert res.total_latency_ms == 250.0
    assert res.tokens_per_second == 28.0
    assert res.model_name == "gemma:2b"
    assert "T" in res.timestamp  # Valid ISO-8601 UTC timestamp


def test_generation_result_validation_error():
    """Verify GenerationResult raises ValidationError when mandatory fields are missing."""
    with pytest.raises(ValidationError):
        # Missing total_latency_ms, prompt_tokens, etc.
        GenerationResult(response_text="Missing fields", model_name="gemma:2b")  # type: ignore


def test_check_ollama_unreachable():
    """Verify check_ollama returns failure gracefully when port is closed."""
    server_up, model_ready, msg = check_ollama(
        base_url="http://127.0.0.1:59999",
        model_name="gemma:2b",
        timeout=0.2,
    )
    assert server_up is False
    assert model_ready is False
    assert "unreachable" in msg.lower()


def test_mock_ollama_generate_parsing():
    """Verify GemmaInferenceClient parses Ollama REST JSON response correctly."""
    mock_ollama_payload = {
        "model": "gemma:2b",
        "created_at": "2026-10-27T12:00:00Z",
        "response": "Water has the chemical formula H2O with a bent shape.",
        "done": True,
        "total_duration": 400_000_000,  # 400 ms in nanoseconds
        "prompt_eval_count": 14,
        "eval_count": 10,
    }

    mock_resp = io.BytesIO(json.dumps(mock_ollama_payload).encode("utf-8"))
    mock_resp.status = 200  # type: ignore

    with patch("urllib.request.urlopen", return_value=mock_resp):
        client = GemmaInferenceClient(
            model_name="gemma:2b",
            ollama_base_url="http://localhost:11434",
            backend="ollama",
        )
        res = client._generate_ollama(
            prompt="State the chemical formula for water.",
            temperature=0.0,
            seed=42,
            max_tokens=150,
        )

        assert res.response_text == "Water has the chemical formula H2O with a bent shape."
        assert res.prompt_tokens == 14
        assert res.completion_tokens == 10
        assert res.total_latency_ms == 400.0
        assert res.tokens_per_second == pytest.approx(25.0, rel=1e-2)
        assert res.model_name == "gemma:2b"


def test_mock_hf_pipeline_parsing():
    """Verify GemmaInferenceClient parses transformers pipeline output correctly."""
    client = GemmaInferenceClient(
        model_name="gemma:2b",
        backend="transformers",
        device="cpu",
    )

    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [
        {"generated_text": "Prompt text Chemical formula is H2O."}
    ]
    client._hf_pipeline = mock_pipeline

    res = client._generate_transformers(
        prompt="Prompt text",
        temperature=0.0,
        seed=42,
        max_tokens=50,
    )

    assert res.response_text == "Chemical formula is H2O."
    assert res.prompt_tokens == 2
    assert res.completion_tokens == 4
    assert res.total_latency_ms > 0
    assert res.tokens_per_second > 0


def test_deterministic_reproducibility():
    """Verify that consecutive generations under temperature=0.0 and seed=42 are bit-for-bit identical."""
    client = GemmaInferenceClient(model_name="gemma:2b", backend="fallback")
    prompt = "State the chemical formula for water and briefly explain its molecular geometry in one sentence."

    run1 = client.generate(prompt=prompt, temperature=0.0, seed=42, max_tokens=150)
    run2 = client.generate(prompt=prompt, temperature=0.0, seed=42, max_tokens=150)

    # 100% exact bit-for-bit string match
    assert run1.response_text == run2.response_text
    assert run1.prompt_tokens == run2.prompt_tokens
    assert run1.completion_tokens == run2.completion_tokens


def test_temperature_sampling_variance():
    """Verify that different seeds introduce controlled variance under stochastic sampling."""
    client = GemmaInferenceClient(model_name="gemma:2b", backend="fallback")
    prompt = "State the chemical formula for water and briefly explain its molecular geometry in one sentence."

    run_seed_42 = client.generate(prompt=prompt, temperature=0.7, seed=42, max_tokens=150)
    run_seed_99 = client.generate(prompt=prompt, temperature=0.7, seed=99, max_tokens=150)

    assert run_seed_42.response_text != run_seed_99.response_text


def test_max_tokens_budgeting():
    """Verify context-window token budgeting truncates output correctly."""
    client = GemmaInferenceClient(model_name="gemma:2b", backend="fallback")
    prompt = "State the chemical formula for water and briefly explain its molecular geometry in one sentence."

    budget_5 = client.generate(prompt=prompt, temperature=0.0, seed=42, max_tokens=5)
    assert len(budget_5.response_text.split()) <= 5


def test_cli_test_generation_flag():
    """Verify CLI main entrypoint executes --test-generation successfully."""
    exit_code = main(["--test-generation"])
    assert exit_code == 0


def test_run_generation_test_direct():
    """Verify run_generation_test completes end-to-end with exit code 0."""
    exit_code = run_generation_test(model_name="gemma:2b")
    assert exit_code == 0
