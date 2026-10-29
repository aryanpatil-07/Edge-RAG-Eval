"""Unit tests for Grounding Metrics Engine & Entity Misattribution Scorer.

Validates text normalization, Exact Match (EM), Token F1, entity misattribution detection,
standardized abstention classification, composite grounding evaluation, and CLI integration.
"""

from __future__ import annotations

import pytest

from src.metrics.grounding import (
    ABSTENTION_MARKERS,
    GroundingMetrics,
    compute_exact_match,
    compute_grounding_score,
    compute_token_f1,
    detect_abstention,
    detect_entity_misattribution,
    evaluate_grounding,
    extract_distinctive_distractor_entities,
    extract_key_entities,
    get_tokens,
    normalize_text,
)
from src.pipeline import main, run_metrics_test


# ---------------------------------------------------------------------------
# Text Normalization & Token Processing Tests
# ---------------------------------------------------------------------------

def test_normalize_text_edge_cases():
    """Verify text normalization handles punctuation, articles, apostrophes, and whitespace."""
    # 1. Punctuation stripping and casing
    assert normalize_text("!!...Hello, World??--") == "hello world"

    # 2. Contraction apostrophes
    assert normalize_text("It's a user's device's feature.") == "its users devices feature"

    # 3. Article stripping (a, an, the)
    assert normalize_text("A dog, an eagle, and the elephant") == "dog eagle and elephant"

    # 4. Whitespace collapsing (tabs, newlines, multiple spaces)
    assert normalize_text("   multiple \t\n spaces \r\n   between words   ") == "multiple spaces between words"

    # 5. Empty and whitespace-only strings
    assert normalize_text("") == ""
    assert normalize_text("    ") == ""

    # 6. Punctuation-only strings
    assert normalize_text("!@#$%^&*()_+-=[]{}|;':\",./<>?") == ""


def test_get_tokens_utility():
    """Verify get_tokens yields whitespace-separated normalized unigrams."""
    tokens = get_tokens("The quick brown fox jumps over the lazy dog.")
    expected = ["quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
    assert tokens == expected

    assert get_tokens("") == []
    assert get_tokens("the a an") == []


# ---------------------------------------------------------------------------
# Core Accuracy Metrics: Exact Match & Token F1
# ---------------------------------------------------------------------------

def test_compute_exact_match():
    """Verify binary exact match across identical, disjoint, and normalized strings."""
    # Exact match identical
    assert compute_exact_match("Neil Armstrong", "Neil Armstrong") == 1.0
    assert compute_exact_match("neil armstrong", "Neil Armstrong") == 1.0

    # Punctuation and article normalization
    assert compute_exact_match("The Apollo 11.", "apollo 11") == 1.0
    assert compute_exact_match("1969", "1969.") == 1.0

    # Disjoint strings
    assert compute_exact_match("Neil Armstrong", "Buzz Aldrin") == 0.0

    # Empty string edge cases
    assert compute_exact_match("", "") == 1.0
    assert compute_exact_match("Non-empty", "") == 0.0
    assert compute_exact_match("", "Non-empty") == 0.0


def test_compute_token_f1():
    """Verify token F1 for identical, disjoint, and overlapping token sets."""
    # Identical
    assert compute_token_f1("Apollo 11 Moon landing", "Apollo 11 Moon landing") == 1.0

    # Empty strings
    assert compute_token_f1("", "") == 1.0
    assert compute_token_f1("Apollo", "") == 0.0
    assert compute_token_f1("", "Apollo") == 0.0

    # Disjoint
    assert compute_token_f1("Mars exploration", "Venus atmosphere") == 0.0

    # Partial overlap
    # Pred: "Apollo 11 Moon landing" -> ['apollo', '11', 'moon', 'landing'] (len 4)
    # GT:   "Apollo 11 mission"      -> ['apollo', '11', 'mission'] (len 3)
    # Overlap: ['apollo', '11'] (2 tokens)
    # Precision = 2/4 = 0.5, Recall = 2/3, F1 = 2 * 0.5 * (2/3) / (0.5 + 2/3) = (2/3) / (7/6) = 4/7
    score = compute_token_f1("Apollo 11 Moon landing", "Apollo 11 mission")
    assert pytest.approx(score, rel=1e-3) == 4.0 / 7.0


def test_compute_grounding_score():
    """Verify content token grounding score against reference context."""
    context = "Photosynthesis occurs inside chloroplasts using solar energy to synthesize glucose."
    grounded_resp = "Chloroplasts use solar energy to synthesize glucose."
    score = compute_grounding_score(grounded_resp, context)
    assert 0.8 <= score <= 1.0

    ungrounded_resp = "Ancient Rome constructed vast stone aqueducts across Europe."
    score_un = compute_grounding_score(ungrounded_resp, context)
    assert score_un == 0.0

    # Empty response edge case
    assert compute_grounding_score("", context) == 1.0


# ---------------------------------------------------------------------------
# Model Abstention Classification Tests
# ---------------------------------------------------------------------------

def test_detect_abstention_positive_controls():
    """Verify detection across standardized refusal and uncertainty phrasing."""
    positive_phrases = [
        "Based on the provided text, there is insufficient context to answer.",
        "The true location cannot be determined from the passages.",
        "Information is not provided in the context.",
        "The context contains conflicting information regarding the event.",
        "The provided context contains conflicting facts.",
        "I do not know the answer to this inquiry.",
        "I don't know the exact date.",
        "The reference document does not provide enough information.",
        "This query is unanswerable with the supplied evidence.",
        "Details are not mentioned in the text.",
        "The system is unable to determine the target value.",
        "Unable to answer based on current passages.",
    ]
    for phrase in positive_phrases:
        assert detect_abstention(phrase) is True, f"Failed to detect abstention in: '{phrase}'"


def test_detect_abstention_negative_controls():
    """Verify factual and non-refusal outputs do not trigger false positive abstention."""
    negative_phrases = [
        "Neil Armstrong was the commander of the Apollo 11 mission.",
        "Water boils at 100 degrees Celsius under standard atmospheric pressure.",
        "Alexander Fleming discovered penicillin in 1928 at St. Mary's Hospital.",
        "The Treaty of Utrecht was signed in 1713.",
        "Superconductors exhibit zero electrical resistance at low temperatures.",
    ]
    for phrase in negative_phrases:
        assert detect_abstention(phrase) is False, f"False positive abstention detected in: '{phrase}'"

    # Empty string
    assert detect_abstention("") is False


# ---------------------------------------------------------------------------
# Entity Extraction & Distinctive Distractor Isolation Tests
# ---------------------------------------------------------------------------

def test_extract_key_entities():
    """Verify extract_key_entities isolates numbers, entity tokens, and filters stopwords."""
    text = "In 1971, Alan Shepard and Edgar Mitchell commanded Apollo 14."
    entities = extract_key_entities(text)
    assert "1971" in entities
    assert "alan" in entities
    assert "shepard" in entities
    assert "mitchell" in entities
    assert "14" in entities
    assert "in" not in entities  # stopword
    assert "and" not in entities  # stopword

    assert extract_key_entities("") == []


def test_extract_distinctive_distractor_entities():
    """Verify distinctive entities isolation between distractor and gold contexts."""
    gold = "The Apollo 11 mission was commanded by Neil Armstrong and landed in 1969."
    distractor = "The Apollo 11 mission was commanded by Alan Shepard and landed in 1971."
    ground_truth = "Neil Armstrong"

    distinct = extract_distinctive_distractor_entities(distractor, gold, ground_truth)
    assert "alan" in distinct
    assert "shepard" in distinct
    assert "1971" in distinct
    assert "armstrong" not in distinct
    assert "1969" not in distinct
    assert "apollo" not in distinct  # shared

    # When distractor is empty
    assert extract_distinctive_distractor_entities("", gold, ground_truth) == set()


# ---------------------------------------------------------------------------
# Entity Misattribution Detection Tests
# ---------------------------------------------------------------------------

def test_detect_entity_misattribution_positive_control():
    """Verify entity misattribution flags when response adopts distractor entities."""
    gold = "The Apollo 11 mission was commanded by Neil Armstrong and landed in 1969."
    distractor = "The Apollo 11 mission was commanded by Alan Shepard and landed in 1971."
    prediction = "The mission was commanded by Alan Shepard in 1971."

    misattr, leaked = detect_entity_misattribution(prediction, gold, distractor)
    assert misattr is True
    assert "alan" in leaked or "shepard" in leaked
    assert "1971" in leaked


def test_detect_entity_misattribution_negative_controls():
    """Verify negative controls do not falsely flag misattribution."""
    gold = "The Apollo 11 mission was commanded by Neil Armstrong and landed in 1969."
    distractor = "The Apollo 11 mission was commanded by Alan Shepard and landed in 1971."

    # 1. Prediction containing true gold context entities
    pred_gold = "The mission was commanded by Neil Armstrong in 1969."
    misattr, leaked = detect_entity_misattribution(pred_gold, gold, distractor)
    assert misattr is False
    assert leaked == []

    # 2. Prediction containing unrelated external hallucination (not in distractor)
    pred_unrelated = "John Glenn commanded the Friendship 7 flight."
    misattr, leaked = detect_entity_misattribution(pred_unrelated, gold, distractor)
    assert misattr is False
    assert leaked == []

    # 3. None distractor context
    misattr, leaked = detect_entity_misattribution(pred_gold, gold, None)
    assert misattr is False
    assert leaked == []

    # 4. Empty distractor context
    misattr, leaked = detect_entity_misattribution(pred_gold, gold, "")
    assert misattr is False
    assert leaked == []

    # 5. Empty prediction
    misattr, leaked = detect_entity_misattribution("", gold, distractor)
    assert misattr is False
    assert leaked == []


# ---------------------------------------------------------------------------
# Composite Grounding Evaluator Tests
# ---------------------------------------------------------------------------

def test_evaluate_grounding_case_1_perfect_answer():
    """Verify Case 1: Exact match and high grounding score."""
    gold = "The Apollo 11 mission was commanded by Neil Armstrong and landed in 1969."
    distractor = "The Apollo 11 mission was commanded by Alan Shepard and landed in 1971."
    ground_truth = "Neil Armstrong"
    prediction = "Neil Armstrong"

    res = evaluate_grounding(prediction, ground_truth, gold, distractor)
    assert res.exact_match == 1.0
    assert res.token_f1 == 1.0
    assert res.misattribution is False
    assert res.entity_misattribution is False
    assert res.leaked_entities == []
    assert res.abstention is False
    assert res.grounding_score == 1.0
    assert res.composite_grounding_score == 1.0


def test_evaluate_grounding_case_2_partial_match():
    """Verify Case 2: Partial overlap with F1 around 0.65 without misattribution."""
    gold = "Superconductors exhibit zero electrical resistance at low temperatures below the critical threshold."
    distractor = "Semiconductors display variable conductivity dependent on thermal dopant concentration."
    ground_truth = "Superconductors exhibit zero electrical resistance at low temperatures"
    prediction = "Superconductors exhibit zero electrical resistance under pressure"

    res = evaluate_grounding(prediction, ground_truth, gold, distractor)
    assert res.exact_match == 0.0
    assert 0.60 <= res.token_f1 <= 0.75
    assert res.misattribution is False
    assert res.leaked_entities == []
    assert res.abstention is False
    assert res.composite_grounding_score > 0.0


def test_evaluate_grounding_case_3_distractor_poisoned():
    """Verify Case 3: Poisoned prediction adopts false entity and sets composite to 0.0."""
    gold = "The Apollo 11 mission was commanded by Neil Armstrong and landed in 1969."
    distractor = "The Apollo 11 mission was commanded by Alan Shepard and landed in 1971."
    ground_truth = "Neil Armstrong"
    prediction = "The Apollo 11 mission was commanded by Alan Shepard in 1971."

    res = evaluate_grounding(prediction, ground_truth, gold, distractor)
    assert res.exact_match == 0.0
    assert res.misattribution is True
    assert len(res.leaked_entities) > 0
    assert "alan" in res.leaked_entities or "shepard" in res.leaked_entities
    assert res.abstention is False
    assert res.composite_grounding_score == 0.0


def test_evaluate_grounding_case_4_valid_abstention():
    """Verify Case 4: Principled abstention detected with misattribution suppressed."""
    gold = "Alexander Fleming discovered penicillin at St. Mary's Hospital in London."
    distractor = "Howard Florey discovered penicillin at Oxford University."
    ground_truth = "Alexander Fleming"
    prediction = "The provided context contains conflicting facts and cannot be determined."

    res = evaluate_grounding(prediction, ground_truth, gold, distractor)
    assert res.exact_match == 0.0
    assert res.token_f1 == 0.0
    assert res.abstention is True
    assert res.misattribution is False
    assert res.leaked_entities == []
    assert res.composite_grounding_score == 0.0


def test_grounding_metrics_serialization():
    """Verify GroundingMetrics model serializes correctly and supports alias property."""
    metrics = GroundingMetrics(
        exact_match=1.0,
        token_f1=0.92,
        misattribution=False,
        leaked_entities=[],
        abstention=False,
        grounding_score=0.85,
        composite_grounding_score=0.78,
    )
    dumped = metrics.model_dump()
    assert dumped["exact_match"] == 1.0
    assert dumped["token_f1"] == 0.92
    assert dumped["misattribution"] is False
    assert metrics.entity_misattribution is False


# ---------------------------------------------------------------------------
# CLI Integration Tests
# ---------------------------------------------------------------------------

def test_cli_test_metrics_runner():
    """Verify direct invocation of run_metrics_test exits with code 0."""
    exit_code = run_metrics_test()
    assert exit_code == 0


def test_cli_test_metrics_flag():
    """Verify CLI main entrypoint executes --test-metrics flag successfully."""
    exit_code = main(["--test-metrics"])
    assert exit_code == 0
