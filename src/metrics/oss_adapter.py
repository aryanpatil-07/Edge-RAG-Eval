"""Google DeepMind and Hugging Face upstream-compatible evaluator adapter.

Implements the ContextNoiseEvaluator interface conforming to google-deepmind/proeval
and huggingface/lighteval decoupled evaluation protocols.
"""

from __future__ import annotations

import collections
import logging
import re
import string
import time
from typing import Any, Dict, List, Literal, Optional, Sequence, Set

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from src.metrics.grounding import (
    ABSTENTION_MARKERS,
    STOPWORDS,
    compute_exact_match,
    compute_grounding_score,
    compute_token_f1,
    detect_abstention,
    detect_entity_misattribution as detect_entity_misattribution_grounding,
    extract_distinctive_distractor_entities,
    get_tokens,
    normalize_text,
)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class EvaluationInput(BaseModel):
    """Standardized input record for decoupled generative RAG evaluation.

    Conforms to Google DeepMind proeval and Hugging Face lighteval data schemas.
    """

    sample_id: str = Field(description="Unique benchmark sample identifier.")
    query: str = Field(description="User or benchmark prompt string.")
    context: str = Field(description="Provided context passage(s) seen by the model.")
    condition: Literal["gold_only", "random_distractor", "adversarial_distractor"] = Field(
        description="Controlled experimental context perturbation condition."
    )
    response: str = Field(description="Generated model response to evaluate.")
    ground_truth: str = Field(description="Authoritative reference ground-truth answer.")
    distractor_context: Optional[str] = Field(
        default=None,
        description="Optional distractor context string used to audit entity misattribution.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional evaluation metadata (e.g., token counts, model names, latency).",
    )


class MetricResult(BaseModel):
    """Stateless metric evaluation container for a single sample.

    Attributes:
        exact_match: Binary score (0.0 or 1.0) indicating exact normalized string equality.
        token_f1: Harmonic mean of token-level precision and recall in range [0.0, 1.0].
        grounding_score: Ratio of response content tokens grounded in the gold context.
        entity_misattribution: True if the response incorporated distinct entities from the distractor.
        abstention: True if the model explicitly refused or noted insufficient context.
        latency_ms: Evaluation computation latency in milliseconds.
        error_category: Discrete diagnostic categorization tag.
    """

    exact_match: float = Field(description="Exact match indicator (0.0 or 1.0).")
    token_f1: float = Field(description="Token-level F1 score in [0.0, 1.0].")
    grounding_score: float = Field(description="Proportion of content tokens grounded in context [0.0, 1.0].")
    entity_misattribution: bool = Field(
        description="Whether contradictory distractor entities were adopted into response."
    )
    abstention: bool = Field(description="Whether the model exhibited calibrated abstention behavior.")
    latency_ms: float = Field(default=0.0, description="Evaluation execution latency in milliseconds.")
    error_category: Literal["none", "misattribution", "abstention", "hallucination"] = Field(
        default="none",
        description="Discrete diagnostic error category for error budgeting.",
    )


# ---------------------------------------------------------------------------
# Metric Utilities (Delegated to src.metrics.grounding)
# ---------------------------------------------------------------------------

def detect_entity_misattribution(
    response: str,
    ground_truth: str,
    gold_context: str,
    distractor_context: Optional[str],
) -> bool:
    """Determine whether the response incorporated contradictory distractor entities.

    Args:
        response: Generated response string.
        ground_truth: Reference ground truth.
        gold_context: Gold context passage.
        distractor_context: Perturbed or distractor context passage.

    Returns:
        bool: True if response adopts entities unique to the distractor while failing exact match.
    """
    if not distractor_context or not distractor_context.strip():
        return False

    # Exact match with ground truth excludes misattribution
    if compute_exact_match(response, ground_truth) == 1.0:
        return False

    # Principled abstention is not entity misattribution
    if detect_abstention(response):
        return False

    misattr, _ = detect_entity_misattribution_grounding(
        prediction=response,
        gold_context=gold_context,
        distractor_context=distractor_context,
    )
    return misattr



# ---------------------------------------------------------------------------
# ContextNoiseEvaluator
# ---------------------------------------------------------------------------

class ContextNoiseEvaluator:
    """Pluggable evaluator for generative RAG resilience under retrieval context noise.

    Designed for upstream integration into google-deepmind/proeval and huggingface/lighteval.
    Stateless, fully typed, and produces both continuous metrics and discrete error flags.
    """

    def __init__(self) -> None:
        """Initialize the ContextNoiseEvaluator."""
        pass

    def evaluate_sample(self, input_item: EvaluationInput) -> MetricResult:
        """Evaluate a single generation input against ground-truth and distractor evidence.

        Args:
            input_item: Typed EvaluationInput record.

        Returns:
            MetricResult: Evaluated scores and discrete diagnostic error flags.
        """
        start_time = time.perf_counter()

        em = compute_exact_match(input_item.response, input_item.ground_truth)
        f1 = compute_token_f1(input_item.response, input_item.ground_truth)
        grounding = compute_grounding_score(input_item.response, input_item.context)
        abstains = detect_abstention(input_item.response)

        misattribution = False
        if input_item.distractor_context:
            gold_ctx = input_item.metadata.get("gold_context", input_item.context)
            misattribution = detect_entity_misattribution(
                response=input_item.response,
                ground_truth=input_item.ground_truth,
                gold_context=gold_ctx,
                distractor_context=input_item.distractor_context,
            )

        # Categorize discrete error
        if em == 1.0 or f1 >= 0.85:
            category: Literal["none", "misattribution", "abstention", "hallucination"] = "none"
        elif misattribution:
            category = "misattribution"
        elif abstains:
            category = "abstention"
        else:
            category = "hallucination"

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return MetricResult(
            exact_match=em,
            token_f1=round(f1, 4),
            grounding_score=round(grounding, 4),
            entity_misattribution=misattribution,
            abstention=abstains,
            latency_ms=round(latency_ms, 3),
            error_category=category,
        )

    def evaluate_batch(
        self,
        inputs: Sequence[EvaluationInput],
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Aggregate evaluation metrics over a batch of evaluation inputs.

        Computes mean, standard deviation, and 95% bootstrap confidence intervals for continuous
        metrics alongside rate statistics for discrete error categories.

        Args:
            inputs: Sequence of EvaluationInput records.
            n_bootstrap: Number of bootstrap resamples for confidence interval estimation.
            seed: Random seed for reproducible bootstrap sampling.

        Returns:
            Dict[str, Any]: Comprehensive aggregated summary dictionary.
        """
        if not inputs:
            return {
                "sample_count": 0,
                "exact_match_mean": 0.0,
                "token_f1_mean": 0.0,
                "grounding_score_mean": 0.0,
                "entity_misattribution_rate": 0.0,
                "abstention_rate": 0.0,
            }

        results: List[MetricResult] = [self.evaluate_sample(item) for item in inputs]

        em_scores = np.array([r.exact_match for r in results])
        f1_scores = np.array([r.token_f1 for r in results])
        grounding_scores = np.array([r.grounding_score for r in results])
        latencies = np.array([r.latency_ms for r in results])

        misattr_flags = np.array([1.0 if r.entity_misattribution else 0.0 for r in results])
        abstain_flags = np.array([1.0 if r.abstention else 0.0 for r in results])

        # Bootstrap confidence intervals (95% CI)
        rng = np.random.default_rng(seed)
        n_samples = len(inputs)

        def bootstrap_ci(values: np.ndarray) -> Tuple[float, float]:
            if n_samples <= 1:
                mean_val = float(np.mean(values))
                return mean_val, mean_val
            boot_means = [float(np.mean(rng.choice(values, size=n_samples, replace=True))) for _ in range(n_bootstrap)]
            return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

        em_ci = bootstrap_ci(em_scores)
        f1_ci = bootstrap_ci(f1_scores)
        grounding_ci = bootstrap_ci(grounding_scores)

        # Condition breakdown
        by_condition: Dict[str, Dict[str, float]] = {}
        conditions = {"gold_only", "random_distractor", "adversarial_distractor"}
        for cond in conditions:
            cond_indices = [i for i, inp in enumerate(inputs) if inp.condition == cond]
            if cond_indices:
                by_condition[cond] = {
                    "count": len(cond_indices),
                    "exact_match_mean": float(np.mean(em_scores[cond_indices])),
                    "token_f1_mean": float(np.mean(f1_scores[cond_indices])),
                    "grounding_score_mean": float(np.mean(grounding_scores[cond_indices])),
                    "entity_misattribution_rate": float(np.mean(misattr_flags[cond_indices])),
                    "abstention_rate": float(np.mean(abstain_flags[cond_indices])),
                }

        return {
            "sample_count": n_samples,
            "exact_match_mean": float(np.mean(em_scores)),
            "exact_match_std": float(np.std(em_scores)),
            "exact_match_ci_95": [round(em_ci[0], 4), round(em_ci[1], 4)],
            "token_f1_mean": float(np.mean(f1_scores)),
            "token_f1_std": float(np.std(f1_scores)),
            "token_f1_ci_95": [round(f1_ci[0], 4), round(f1_ci[1], 4)],
            "grounding_score_mean": float(np.mean(grounding_scores)),
            "grounding_score_std": float(np.std(grounding_scores)),
            "grounding_score_ci_95": [round(grounding_ci[0], 4), round(grounding_ci[1], 4)],
            "entity_misattribution_rate": float(np.mean(misattr_flags)),
            "abstention_rate": float(np.mean(abstain_flags)),
            "latency_ms_mean": float(np.mean(latencies)),
            "by_condition": by_condition,
        }
