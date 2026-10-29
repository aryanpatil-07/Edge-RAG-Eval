"""Evaluation metrics for edge retrieval grounding and noisy context resilience.

Implements pure functions for string normalization, core accuracy (Exact Match, Token F1),
entity misattribution detection, model abstention detection, and composite grounding scoring.
"""

from __future__ import annotations

import collections
import re
import string
from typing import Any, List, Optional, Sequence, Set, Tuple

import numpy as np
from pydantic import BaseModel, Field

# Standard stopword set for content-token isolation and grounding evaluation
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
}

# Standardized refusal and abstention phrases
ABSTENTION_MARKERS: Sequence[str] = (
    "insufficient context",
    "cannot be determined",
    "not enough information",
    "information is not provided",
    "not mentioned in the text",
    "unable to answer",
    "i do not know",
    "i don't know",
    "does not provide",
    "cannot answer",
    "no information",
    "context does not mention",
    "unanswerable",
    "conflicting facts",
    "conflicting information",
    "conflicting reports",
    "conflicting details",
    "not stated in the context",
    "insufficient information",
    "unable to determine",
    "cannot verify",
)


# ---------------------------------------------------------------------------
# Text Normalization & Token Processing
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, stripping punctuation, stripping articles, and collapsing whitespace.

    Args:
        text: Input string.

    Returns:
        str: Normalized text string.
    """
    text = text.lower()
    # Strip punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)
    # Strip English articles (a, an, the)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Collapse whitespace
    return " ".join(text.split())


def get_tokens(text: str) -> list[str]:
    """Tokenize normalized text into a list of whitespace-separated unigram tokens.

    Args:
        text: Input string.

    Returns:
        list[str]: Unigram token list.
    """
    return normalize_text(text).split()


# ---------------------------------------------------------------------------
# Core Accuracy Metrics
# ---------------------------------------------------------------------------

def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute binary exact match (EM) between prediction and ground-truth strings.

    Args:
        prediction: Model generated output text.
        ground_truth: True reference answer.

    Returns:
        float: 1.0 if normalized strings match exactly, else 0.0.
    """
    norm_pred = normalize_text(prediction)
    norm_gt = normalize_text(ground_truth)
    if not norm_gt:
        return 1.0 if not norm_pred else 0.0
    return 1.0 if norm_pred == norm_gt else 0.0


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Compute harmonic mean of token-level precision and recall.

    Args:
        prediction: Model generated output text.
        ground_truth: True reference answer.

    Returns:
        float: Token F1 score in range [0.0, 1.0].
    """
    pred_tokens = get_tokens(prediction)
    gt_tokens = get_tokens(ground_truth)

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    f1 = (2.0 * precision * recall) / (precision + recall)
    return float(f1)


def compute_grounding_score(response: str, context: str) -> float:
    """Compute content token overlap ratio between model response and context.

    Args:
        response: Model generated response.
        context: Reference context text.

    Returns:
        float: Fraction of content tokens in response present in context in [0.0, 1.0].
    """
    resp_tokens = [w for w in get_tokens(response) if w not in STOPWORDS and len(w) > 1]
    if not resp_tokens:
        return 1.0

    ctx_tokens = set(get_tokens(context))
    grounded_count = sum(1 for w in resp_tokens if w in ctx_tokens)
    return float(grounded_count / len(resp_tokens))


# ---------------------------------------------------------------------------
# Abstention & Entity Misattribution Detectors
# ---------------------------------------------------------------------------

def detect_abstention(prediction: str) -> bool:
    """Check for standardized refusal and uncertainty markers.

    Args:
        prediction: Model prediction or response string.

    Returns:
        bool: True if prediction exhibits refusal/uncertainty marker, else False.
    """
    if not prediction:
        return False
    norm = prediction.lower()
    return any(marker in norm for marker in ABSTENTION_MARKERS)


def extract_key_entities(text: str) -> list[str]:
    """Extract key entity and informative tokens from text in order of appearance.

    Args:
        text: Input passage or response string.

    Returns:
        list[str]: Deduplicated sequence of entity tokens.
    """
    if not text:
        return []
    raw_tokens = re.findall(r"\b[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*\b", text)
    entities: list[str] = []
    seen: set[str] = set()
    for tok in raw_tokens:
        norm = tok.lower()
        if norm in STOPWORDS:
            continue
        if tok.isdigit() or len(norm) >= 2:
            if norm not in seen:
                seen.add(norm)
                entities.append(norm)
    return entities


def extract_distinctive_distractor_entities(
    distractor_context: str,
    gold_context: str,
    ground_truth: str = "",
) -> set[str]:
    """Extract entity tokens present in the distractor but absent from gold context and ground truth.

    Args:
        distractor_context: Adversarial or noisy passage.
        gold_context: Ground-truth reference passage.
        ground_truth: Target answer text (optional).

    Returns:
        set[str]: Distinctive distractor entity tokens.
    """
    if not distractor_context:
        return set()

    clean_gold = gold_context
    if distractor_context and distractor_context in gold_context:
        isolated = gold_context.replace(distractor_context, "").strip()
        if isolated:
            clean_gold = isolated

    dist_entities = set(extract_key_entities(distractor_context))
    gold_entities = set(extract_key_entities(clean_gold))
    gt_entities = set(extract_key_entities(ground_truth)) if ground_truth else set()

    return dist_entities - (gold_entities | gt_entities)


def detect_entity_misattribution(
    prediction: str,
    gold_context: str,
    distractor_context: str | None,
) -> tuple[bool, list[str]]:
    """Determine whether the prediction contains informative entities from distractor absent in gold.

    Args:
        prediction: Generated model prediction string.
        gold_context: Gold reference context passage.
        distractor_context: Optional perturbed or distractor context passage.

    Returns:
        tuple[bool, list[str]]: (misattributed status, list of leaked entity tokens).
    """
    if not distractor_context or not distractor_context.strip() or not prediction:
        return False, []

    # Distinctive entities strictly present in distractor and absent from gold context
    distractor_only = extract_distinctive_distractor_entities(
        distractor_context=distractor_context,
        gold_context=gold_context,
    )
    if not distractor_only:
        return False, []

    pred_entities = extract_key_entities(prediction)
    leaked = [e for e in pred_entities if e in distractor_only]
    misattributed = len(leaked) > 0
    return misattributed, leaked


# ---------------------------------------------------------------------------
# Composite Grounding Evaluation Models
# ---------------------------------------------------------------------------

class GroundingMetrics(BaseModel):
    """Composite container for retrieval grounding and error diagnostics metrics."""

    exact_match: float = Field(description="Exact match indicator (0.0 or 1.0).")
    token_f1: float = Field(description="Token-level F1 score in [0.0, 1.0].")
    misattribution: bool = Field(description="Whether distractor entities were leaked into prediction.")
    leaked_entities: list[str] = Field(default_factory=list, description="List of leaked distractor entity tokens.")
    abstention: bool = Field(description="Whether the response exhibits calibrated refusal/uncertainty.")
    grounding_score: float = Field(description="Ratio of content tokens grounded in gold context [0.0, 1.0].")
    composite_grounding_score: float = Field(description="Composite score weighting accuracy, grounding, and resilience.")

    @property
    def entity_misattribution(self) -> bool:
        """Alias for misattribution property."""
        return self.misattribution


def evaluate_grounding(
    prediction: str,
    ground_truth: str,
    gold_context: str,
    distractor_context: str | None = None,
) -> GroundingMetrics:
    """Evaluate prediction against ground truth and contexts across all grounding dimensions.

    Args:
        prediction: Generated model prediction.
        ground_truth: True reference answer.
        gold_context: Gold reference context passage.
        distractor_context: Optional perturbed or distractor context passage.

    Returns:
        GroundingMetrics: Bundled metrics including EM, F1, misattribution, leaked entities,
            abstention, grounding score, and composite score.
    """
    em = compute_exact_match(prediction, ground_truth)
    f1 = compute_token_f1(prediction, ground_truth)
    abstains = detect_abstention(prediction)
    misattr, leaked = detect_entity_misattribution(
        prediction=prediction,
        gold_context=gold_context,
        distractor_context=distractor_context,
    )

    # Exact match with ground truth excludes misattribution
    if em == 1.0:
        misattr = False
        leaked = []

    # Calibrated abstention excludes misattribution
    if abstains:
        misattr = False
        leaked = []

    grounding = compute_grounding_score(prediction, gold_context)

    if misattr or abstains:
        composite = 0.0
    else:
        composite = round(f1 * grounding, 4) if (f1 > 0 and grounding > 0) else 0.0

    return GroundingMetrics(
        exact_match=em,
        token_f1=round(f1, 4),
        misattribution=misattr,
        leaked_entities=leaked,
        abstention=abstains,
        grounding_score=round(grounding, 4),
        composite_grounding_score=composite,
    )


# ---------------------------------------------------------------------------
# Retrieval Vector Grounding Metrics
# ---------------------------------------------------------------------------

def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D or single-row 2D vectors.

    Args:
        a: First vector of shape (D,) or (1, D).
        b: Second vector of shape (D,) or (1, D).

    Returns:
        float: Cosine similarity score in range [-1.0, 1.0].
    """
    v1 = np.asarray(a).flatten()
    v2 = np.asarray(b).flatten()

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))


def evaluate_grounding_precision(
    query_emb: np.ndarray,
    candidate_embs: np.ndarray,
    gold_index: int,
    top_k: int = 1,
) -> float:
    """Evaluate whether the gold ground-truth document is retrieved in top-k.

    Args:
        query_emb: Query embedding vector (1, D) or (D,).
        candidate_embs: Candidate document embeddings (N, D).
        gold_index: Index of the true relevant context document.
        top_k: Rank threshold for hit calculation.

    Returns:
        float: 1.0 if gold document is within top-k ranks, else 0.0.
    """
    q = np.asarray(query_emb).reshape(1, -1)
    q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

    docs = np.asarray(candidate_embs)
    docs_norm = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-12)

    similarities = np.dot(docs_norm, q_norm.T).flatten()
    ranked_indices = np.argsort(-similarities)[:top_k]

    return 1.0 if gold_index in ranked_indices else 0.0
