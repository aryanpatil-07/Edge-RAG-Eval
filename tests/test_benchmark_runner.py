"""Unit tests for automated 500-sample benchmark runner (Conditions 1 & 2).

Validates prompt assembly, checkpoint save and resume mechanisms, result serialization,
paired statistical significance testing, and CLI execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.benchmark_runner import (
    BenchmarkRunner,
    BenchmarkSummary,
    compare_conditions,
    compute_paired_significance,
    format_rag_prompt,
)
from src.metrics.oss_adapter import ContextNoiseEvaluator, MetricResult
from src.models.inference import GenerationResult, GemmaInferenceClient
from src.pipeline import main, run_benchmark


# ---------------------------------------------------------------------------
# Prompt Assembly Formatter Tests
# ---------------------------------------------------------------------------

def test_prompt_assembly_condition_1():
    """Verify Condition 1 prompt format contains only gold context without distractor."""
    gold = "The Apollo 11 lunar module Eagle touched down on the Moon on July 20, 1969."
    query = "In what year did Apollo 11 land?"
    prompt = format_rag_prompt(
        query=query,
        gold_context=gold,
        distractor_context="Random unrelated context.",
        condition="gold_only",
    )

    assert "Answer the question based strictly on the provided context." in prompt
    assert "Context:\n" in prompt
    assert gold in prompt
    assert "Random unrelated context." not in prompt
    assert f"Question: {query}" in prompt
    assert prompt.endswith("Answer:")


def test_prompt_assembly_condition_2():
    """Verify Condition 2 prompt format contains gold context concatenated with distractor."""
    gold = "The Apollo 11 lunar module Eagle touched down on the Moon on July 20, 1969."
    distractor = "Photosynthesis converts solar photons into chemical energy in chloroplasts."
    query = "In what year did Apollo 11 land?"
    prompt = format_rag_prompt(
        query=query,
        gold_context=gold,
        distractor_context=distractor,
        condition="random_distractor",
    )

    assert gold in prompt
    assert distractor in prompt
    assert f"{gold}\n\n{distractor}" in prompt
    assert f"Question: {query}" in prompt


def test_prompt_assembly_condition_2_missing_distractor_fallback():
    """Verify Condition 2 falls back cleanly to gold context when distractor is None."""
    gold = "Gold reference passage text."
    query = "Sample question?"
    prompt = format_rag_prompt(
        query=query,
        gold_context=gold,
        distractor_context=None,
        condition="random_distractor",
    )
    assert gold in prompt
    assert "None" not in prompt


# ---------------------------------------------------------------------------
# Statistical Significance (Wilcoxon Signed-Rank Test) Tests
# ---------------------------------------------------------------------------

def test_paired_significance_identical_scores():
    """Verify Wilcoxon test returns statistic=0.0 and p_value=1.0 for identical vectors."""
    c1 = [0.8, 0.9, 0.5, 0.7, 1.0]
    c2 = [0.8, 0.9, 0.5, 0.7, 1.0]
    res = compute_paired_significance(c1, c2)
    assert res["statistic"] == 0.0
    assert res["p_value"] == 1.0


def test_paired_significance_differing_scores():
    """Verify Wilcoxon test returns valid statistic and p-value for differing vectors."""
    c1 = [1.0, 0.9, 0.8, 0.95, 1.0, 0.85]
    c2 = [0.6, 0.5, 0.7, 0.60, 0.7, 0.55]
    res = compute_paired_significance(c1, c2)
    assert "statistic" in res
    assert 0.0 <= res["p_value"] <= 1.0


def test_paired_significance_insufficient_samples():
    """Verify Wilcoxon test gracefully handles single-sample or mismatched lengths."""
    res = compute_paired_significance([1.0], [0.5])
    assert res["p_value"] == 1.0

    res_mismatch = compute_paired_significance([1.0, 0.5], [0.5])
    assert res_mismatch["p_value"] == 1.0


# ---------------------------------------------------------------------------
# Checkpoint Save & Resume Mechanism Tests
# ---------------------------------------------------------------------------

def test_checkpoint_save_and_resume_on_mock_batch(tmp_path: Path):
    """Verify BenchmarkRunner saves intermediate checkpoints and resumes seamlessly."""
    # 1. Create a mock 5-sample dataset
    mock_data = [
        {
            "sample_id": f"eval_test_{i:03d}",
            "query": f"Query question {i}?",
            "ground_truth_answer": f"Answer {i}",
            "gold_context": f"Gold context evidence for answer {i}.",
            "random_distractor": f"Distractor noise text {i}.",
            "adversarial_distractor": f"Adversarial text {i}.",
        }
        for i in range(5)
    ]
    data_file = tmp_path / "mock_eval_5.json"
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(mock_data, f)

    results_dir = tmp_path / "results"
    client = GemmaInferenceClient(model_name="gemma:2b", backend="fallback")
    evaluator = ContextNoiseEvaluator()

    runner = BenchmarkRunner(
        inference_client=client,
        evaluator=evaluator,
        dataset_path=str(data_file),
        results_dir=str(results_dir),
    )

    # 2. Run initial batch with limit=3 (saving checkpoint)
    summary_part = runner.run_condition("gold_only", limit=3, checkpoint_every=1)
    assert summary_part.sample_count == 3
    assert len(summary_part.results) == 3

    ckpt_path = runner.get_checkpoint_path("gold_only")
    assert ckpt_path.is_file()

    with open(ckpt_path, "r", encoding="utf-8") as f:
        ckpt_data = json.load(f)
    assert ckpt_data["completed_count"] == 3
    assert len(ckpt_data["records"]) == 3

    # 3. Resume with limit=5
    summary_full = runner.run_condition("gold_only", limit=5, checkpoint_every=1)
    assert summary_full.sample_count == 5
    assert len(summary_full.results) == 5

    # Check that sample IDs 0..4 are all present and ordered
    for idx, r in enumerate(summary_full.results):
        assert r["sample_id"] == f"eval_test_{idx:03d}"


# ---------------------------------------------------------------------------
# BenchmarkSummary Schema & Serialization Tests
# ---------------------------------------------------------------------------

def test_benchmark_summary_serialization():
    """Verify BenchmarkSummary validates types and serializes correctly to dict."""
    summary = BenchmarkSummary(
        condition="gold_only",
        model_name="gemma:2b",
        sample_count=10,
        exact_match_mean=0.80,
        exact_match_std=0.15,
        token_f1_mean=0.88,
        token_f1_std=0.12,
        grounding_score_mean=0.95,
        grounding_score_std=0.08,
        abstention_rate=0.0,
        avg_latency_ms=45.2,
        avg_tokens_per_second=30.5,
        results=[{"sample_id": "eval_001", "exact_match": 1.0}],
    )

    dumped = summary.model_dump()
    assert dumped["condition"] == "gold_only"
    assert dumped["sample_count"] == 10
    assert dumped["exact_match_mean"] == 0.80
    assert len(dumped["results"]) == 1


def test_compare_conditions_structure():
    """Verify compare_conditions generates required delta and significance metrics."""
    res_c1 = [
        {"sample_id": "eval_001", "exact_match": 1.0, "token_f1": 1.0, "grounding_score": 1.0, "latency_ms": 40.0},
        {"sample_id": "eval_002", "exact_match": 1.0, "token_f1": 0.9, "grounding_score": 0.9, "latency_ms": 42.0},
    ]
    res_c2 = [
        {"sample_id": "eval_001", "exact_match": 1.0, "token_f1": 0.8, "grounding_score": 0.8, "latency_ms": 50.0},
        {"sample_id": "eval_002", "exact_match": 0.0, "token_f1": 0.5, "grounding_score": 0.7, "latency_ms": 52.0},
    ]

    sum1 = BenchmarkSummary(
        condition="gold_only",
        model_name="gemma:2b",
        sample_count=2,
        exact_match_mean=1.0,
        exact_match_std=0.0,
        token_f1_mean=0.95,
        token_f1_std=0.05,
        grounding_score_mean=0.95,
        grounding_score_std=0.05,
        abstention_rate=0.0,
        avg_latency_ms=41.0,
        avg_tokens_per_second=25.0,
        results=res_c1,
    )
    sum2 = BenchmarkSummary(
        condition="random_distractor",
        model_name="gemma:2b",
        sample_count=2,
        exact_match_mean=0.5,
        exact_match_std=0.5,
        token_f1_mean=0.65,
        token_f1_std=0.15,
        grounding_score_mean=0.75,
        grounding_score_std=0.05,
        abstention_rate=0.0,
        avg_latency_ms=51.0,
        avg_tokens_per_second=20.0,
        results=res_c2,
    )

    comparison = compare_conditions(sum1, sum2)
    assert comparison["sample_count"] == 2
    assert comparison["exact_match"]["delta"] == -0.5
    assert comparison["token_f1"]["delta"] == pytest.approx(-0.30)
    assert comparison["latency_ms"]["delta"] == pytest.approx(10.0)
    assert "wilcoxon_p" in comparison["exact_match"]


# ---------------------------------------------------------------------------
# CLI Integration Tests
# ---------------------------------------------------------------------------

def test_cli_run_benchmark_smoke_limit(tmp_path: Path):
    """Verify CLI main entrypoint executes --run-benchmark with --limit 5 successfully."""
    custom_results = tmp_path / "custom_results"
    exit_code = main([
        "--run-benchmark",
        "--conditions", "1,2",
        "--limit", "5",
        "--results-dir", str(custom_results),
    ])
    assert exit_code == 0

    # Verify output raw result files exist
    c1_path = custom_results / "gemma2b_condition1_raw.json"
    c2_path = custom_results / "gemma2b_condition2_raw.json"
    assert c1_path.is_file()
    assert c2_path.is_file()

    with open(c1_path, "r", encoding="utf-8") as f:
        data_c1 = json.load(f)
    assert data_c1["condition"] == "gold_only"
    assert len(data_c1["records"]) == 5
