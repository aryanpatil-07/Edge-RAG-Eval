"""Evaluation metrics for edge retrieval grounding and noisy context resilience."""

from src.metrics.grounding import (
    ABSTENTION_MARKERS,
    STOPWORDS,
    GroundingMetrics,
    compute_cosine_similarity,
    compute_exact_match,
    compute_grounding_score,
    compute_token_f1,
    detect_abstention,
    detect_entity_misattribution,
    evaluate_grounding,
    evaluate_grounding_precision,
    extract_distinctive_distractor_entities,
    extract_key_entities,
    get_tokens,
    normalize_text,
)
from src.metrics.oss_adapter import (
    ContextNoiseEvaluator,
    EvaluationInput,
    MetricResult,
)

__all__ = [
    "compute_cosine_similarity",
    "evaluate_grounding_precision",
    "normalize_text",
    "get_tokens",
    "compute_exact_match",
    "compute_token_f1",
    "compute_grounding_score",
    "detect_abstention",
    "extract_key_entities",
    "extract_distinctive_distractor_entities",
    "detect_entity_misattribution",
    "GroundingMetrics",
    "evaluate_grounding",
    "EvaluationInput",
    "MetricResult",
    "ContextNoiseEvaluator",
    "STOPWORDS",
    "ABSTENTION_MARKERS",
]
