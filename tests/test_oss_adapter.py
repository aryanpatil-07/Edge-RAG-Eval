"""Unit tests for Google DeepMind and Hugging Face OSS adapter (ContextNoiseEvaluator)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.metrics.oss_adapter import (
    ContextNoiseEvaluator,
    EvaluationInput,
    MetricResult,
    compute_exact_match,
    compute_grounding_score,
    compute_token_f1,
    detect_abstention,
    detect_entity_misattribution,
    extract_distinctive_distractor_entities,
    normalize_text,
)
from src.pipeline import main, run_oss_adapter_test


# ---------------------------------------------------------------------------
# Schema & Validation Tests
# ---------------------------------------------------------------------------

def test_evaluation_input_valid():
    """Verify EvaluationInput accepts conforming schema and valid conditions."""
    sample = EvaluationInput(
        sample_id="eval_001",
        query="In what year was the Treaty of Utrecht signed?",
        context="The Treaty of Utrecht was signed in 1713.",
        condition="gold_only",
        response="1713",
        ground_truth="1713",
        distractor_context=None,
        metadata={"domain": "history"},
    )
    assert sample.sample_id == "eval_001"
    assert sample.condition == "gold_only"
    assert sample.response == "1713"
    assert sample.metadata["domain"] == "history"


def test_evaluation_input_invalid_condition():
    """Verify EvaluationInput raises ValidationError on unsupported condition literals."""
    with pytest.raises(ValidationError):
        EvaluationInput(
            sample_id="eval_001",
            query="Sample query?",
            context="Sample context.",
            condition="invalid_condition",  # type: ignore
            response="Answer",
            ground_truth="Answer",
        )


def test_metric_result_serialization():
    """Verify MetricResult validates and serializes required fields."""
    res = MetricResult(
        exact_match=1.0,
        token_f1=0.95,
        grounding_score=0.88,
        entity_misattribution=False,
        abstention=False,
        latency_ms=1.25,
        error_category="none",
    )
    dumped = res.model_dump()
    assert dumped["exact_match"] == 1.0
    assert dumped["token_f1"] == 0.95
    assert dumped["error_category"] == "none"


# ---------------------------------------------------------------------------
# Metric Formulation Tests
# ---------------------------------------------------------------------------

def test_normalization_and_exact_match():
    """Verify exact match handles punctuation, casing, and whitespace normalization."""
    assert compute_exact_match("William the Conqueror", "william the conqueror") == 1.0
    assert compute_exact_match("1969.", "1969") == 1.0
    assert compute_exact_match("The United States", "united states") == 1.0
    assert compute_exact_match("Harald Hardrada", "William the Conqueror") == 0.0


def test_token_f1_score():
    """Verify token F1 handles perfect, partial, and disjoint lexical overlaps."""
    assert compute_token_f1("Apollo 11 Moon landing", "Apollo 11 Moon landing") == 1.0
    assert compute_token_f1("Apollo 11 landing", "Apollo 11 Moon landing") == pytest.approx(6.0 / 7.0, rel=1e-3)
    assert compute_token_f1("Completely disjoint text", "Apollo 11 Moon landing") == 0.0
    assert compute_token_f1("", "") == 1.0


def test_grounding_score_bounded():
    """Verify grounding score measures content token overlap bounded in [0.0, 1.0]."""
    context = "Photosynthesis occurs inside chloroplasts using solar energy to synthesize glucose."
    grounded_resp = "Chloroplasts use solar energy to synthesize glucose."
    score = compute_grounding_score(grounded_resp, context)
    assert 0.0 <= score <= 1.0
    assert score > 0.8

    ungrounded_resp = "Ancient Rome constructed vast stone aqueducts across Europe."
    score_un = compute_grounding_score(ungrounded_resp, context)
    assert score_un == 0.0


# ---------------------------------------------------------------------------
# Error Diagnostics Tests (Misattribution & Abstention)
# ---------------------------------------------------------------------------

def test_detect_abstention_phrases():
    """Verify abstention classifier identifies standardized refusal phrases."""
    abstention_examples = [
        "Based on the text provided, there is insufficient context to answer.",
        "The question cannot be determined from the available passages.",
        "I do not know the answer as it is not mentioned in the text.",
        "The context provides insufficient information regarding this event.",
        "Information is not provided in the reference document.",
    ]
    for text in abstention_examples:
        assert detect_abstention(text) is True, f"Failed to detect abstention in: '{text}'"

    factual_examples = [
        "The Apollo 11 mission landed in 1969.",
        "Water has the chemical formula H2O with a bent molecular geometry.",
        "William defeated Harold at the Battle of Hastings.",
    ]
    for text in factual_examples:
        assert detect_abstention(text) is False, f"False positive abstention detected in: '{text}'"


def test_detect_entity_misattribution():
    """Verify entity misattribution flags when response adopts distractor entities."""
    gold_context = "The Apollo 11 mission landed in July 1969 carrying Neil Armstrong and Buzz Aldrin."
    distractor_context = "The Apollo 11 mission landed in October 1971 carrying Alan Shepard and Edgar Mitchell."
    ground_truth = "1969"

    # Distinctive distractor entities should include 1971, Shepard, Mitchell
    diff = extract_distinctive_distractor_entities(distractor_context, gold_context, ground_truth)
    assert "1971" in diff or "shepard" in diff

    # 1. Response adopting distractor entity -> Misattribution: True
    misattr_resp = "The Apollo 11 landed in 1971 carrying Alan Shepard."
    assert detect_entity_misattribution(misattr_resp, ground_truth, gold_context, distractor_context) is True

    # 2. Response with correct ground truth -> Misattribution: False
    correct_resp = "The Apollo 11 landed in 1969 carrying Neil Armstrong."
    assert detect_entity_misattribution(correct_resp, ground_truth, gold_context, distractor_context) is False

    # 3. Response with abstention -> Misattribution: False
    abstain_resp = "The provided context contains insufficient information to answer."
    assert detect_entity_misattribution(abstain_resp, ground_truth, gold_context, distractor_context) is False

    # 4. None distractor context -> Misattribution: False
    assert detect_entity_misattribution(misattr_resp, ground_truth, gold_context, None) is False


# ---------------------------------------------------------------------------
# Evaluator Engine & Batch Aggregation Tests
# ---------------------------------------------------------------------------

def test_evaluator_sample_execution():
    """Verify ContextNoiseEvaluator.evaluate_sample returns valid MetricResult."""
    evaluator = ContextNoiseEvaluator()

    inp = EvaluationInput(
        sample_id="eval_test",
        query="Who wrote Hamlet?",
        context="William Shakespeare wrote Hamlet around 1600.",
        condition="gold_only",
        response="William Shakespeare",
        ground_truth="William Shakespeare",
        distractor_context=None,
    )

    res = evaluator.evaluate_sample(inp)
    assert res.exact_match == 1.0
    assert res.token_f1 == 1.0
    assert res.grounding_score == 1.0
    assert res.entity_misattribution is False
    assert res.abstention is False
    assert res.error_category == "none"
    assert res.latency_ms >= 0.0


def test_evaluator_batch_aggregation_and_ci():
    """Verify evaluate_batch computes moments and 95% bootstrap confidence intervals."""
    evaluator = ContextNoiseEvaluator()

    gold_ctx = "Albert Einstein formulated General Relativity in 1915."
    adv_ctx = "Niels Bohr formulated General Relativity in 1923."

    inputs = [
        # Correct sample
        EvaluationInput(
            sample_id="s1",
            query="When was General Relativity formulated?",
            context=gold_ctx,
            condition="gold_only",
            response="1915",
            ground_truth="1915",
            distractor_context=None,
        ),
        # Misattributed sample
        EvaluationInput(
            sample_id="s2",
            query="When was General Relativity formulated?",
            context=f"{gold_ctx}\n\n{adv_ctx}",
            condition="adversarial_distractor",
            response="It was formulated in 1923 by Niels Bohr.",
            ground_truth="1915",
            distractor_context=adv_ctx,
        ),
        # Abstaining sample
        EvaluationInput(
            sample_id="s3",
            query="When was General Relativity formulated?",
            context=f"{gold_ctx}\n\n{adv_ctx}",
            condition="adversarial_distractor",
            response="The provided context contains conflicting reports and insufficient context to answer.",
            ground_truth="1915",
            distractor_context=adv_ctx,
        ),
    ]

    stats = evaluator.evaluate_batch(inputs, n_bootstrap=500, seed=42)

    assert stats["sample_count"] == 3
    assert 0.0 <= stats["exact_match_mean"] <= 1.0
    assert 0.0 <= stats["token_f1_mean"] <= 1.0
    assert stats["entity_misattribution_rate"] == pytest.approx(1.0 / 3.0)
    assert stats["abstention_rate"] == pytest.approx(1.0 / 3.0)

    # Confidence interval bounds
    assert stats["exact_match_ci_95"][0] <= stats["exact_match_ci_95"][1]
    assert stats["token_f1_ci_95"][0] <= stats["token_f1_ci_95"][1]

    # Condition breakdown
    assert "gold_only" in stats["by_condition"]
    assert "adversarial_distractor" in stats["by_condition"]


def test_cli_test_oss_adapter_runner():
    """Verify run_oss_adapter_test executes cleanly and returns exit code 0."""
    exit_code = run_oss_adapter_test()
    assert exit_code == 0


def test_cli_test_oss_adapter_flag():
    """Verify CLI main entrypoint --test-oss-adapter executes with exit code 0."""
    exit_code = main(["--test-oss-adapter"])
    assert exit_code == 0
