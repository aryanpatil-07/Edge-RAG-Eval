"""Automated benchmark runner and statistical evaluation engine for Edge-RAG-Eval.

Orchestrates multi-condition RAG evaluation across benchmark samples (Gold Only vs Random Distractor),
records raw generations and grounding metrics, performs checkpointing/resumption, and computes
paired Wilcoxon signed-rank statistical significance tests.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import wilcoxon

from src.metrics.oss_adapter import ContextNoiseEvaluator, EvaluationInput
from src.models.inference import GemmaInferenceClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class BenchmarkSummary(BaseModel):
    """Aggregated benchmark evaluation summary for an experimental condition."""

    condition: Literal["gold_only", "random_distractor", "adversarial_distractor"] = Field(
        description="Experimental condition name ('gold_only', 'random_distractor', or 'adversarial_distractor')."
    )
    model_name: str = Field(description="Model identifier evaluated.")
    sample_count: int = Field(description="Number of evaluation samples evaluated.")
    exact_match_mean: float = Field(description="Mean exact match score.")
    exact_match_std: float = Field(description="Standard deviation of exact match score.")
    token_f1_mean: float = Field(description="Mean token F1 score.")
    token_f1_std: float = Field(description="Standard deviation of token F1 score.")
    grounding_score_mean: float = Field(description="Mean content token grounding score.")
    grounding_score_std: float = Field(description="Standard deviation of grounding score.")
    abstention_rate: float = Field(description="Proportion of samples exhibiting calibrated abstention.")
    entity_misattribution_rate: float = Field(
        default=0.0,
        description="Proportion of samples exhibiting entity misattribution from distractor.",
    )
    avg_latency_ms: float = Field(description="Average total generation and evaluation latency in ms.")
    avg_tokens_per_second: float = Field(description="Average generation throughput in tokens/sec.")
    results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-sample evaluation records."
    )


class ComparativeAnalysis(BaseModel):
    """3-Way comparative analysis across Condition 1, Condition 2, and Condition 3."""

    model_name: str = Field(description="Model identifier evaluated.")
    sample_count: int = Field(description="Number of paired samples evaluated across all 3 conditions.")
    condition_1_summary: Optional[Dict[str, Any]] = Field(
        default=None, description="Aggregated metrics summary for Condition 1 (Gold Only)."
    )
    condition_2_summary: Optional[Dict[str, Any]] = Field(
        default=None, description="Aggregated metrics summary for Condition 2 (Random Distractor)."
    )
    condition_3_summary: Optional[Dict[str, Any]] = Field(
        default=None, description="Aggregated metrics summary for Condition 3 (Adversarial Distractor)."
    )
    exact_match: Dict[str, Any] = Field(description="Exact Match comparison statistics.")
    token_f1: Dict[str, Any] = Field(description="Token F1 comparison statistics.")
    grounding_score: Dict[str, Any] = Field(description="Grounding Score comparison statistics.")
    entity_misattribution: Dict[str, Any] = Field(description="Entity Misattribution comparison statistics.")
    abstention_rate: Dict[str, Any] = Field(description="Abstention Rate comparison statistics.")
    latency_ms: Dict[str, Any] = Field(description="Latency comparison statistics.")
    relative_drops: Dict[str, float] = Field(
        description="Relative percentage drops: delta_em_1_to_3_pct, delta_f1_1_to_3_pct."
    )
    significance_tests: Dict[str, Any] = Field(
        description="Paired Wilcoxon signed-rank tests for C1 vs C3 and C2 vs C3."
    )

    @property
    def delta_em_1_to_3_pct(self) -> float:
        """Relative percentage drop in Exact Match from Condition 1 to Condition 3."""
        return self.relative_drops.get("delta_em_1_to_3_pct", 0.0)

    @property
    def delta_f1_1_to_3_pct(self) -> float:
        """Relative percentage drop in Token F1 from Condition 1 to Condition 3."""
        return self.relative_drops.get("delta_f1_1_to_3_pct", 0.0)


# ---------------------------------------------------------------------------
# Prompt Assembly Formatter
# ---------------------------------------------------------------------------

def format_rag_prompt(
    query: str,
    gold_context: str,
    distractor_context: Optional[str] = None,
    condition: Literal["gold_only", "random_distractor", "adversarial_distractor"] = "gold_only",
) -> str:
    """Standardizes the in-context RAG prompt structure for edge evaluation.

    Args:
        query: Target question string.
        gold_context: True reference context passage.
        distractor_context: Optional distractor passage (random for C2, adversarial for C3).
        condition: 'gold_only', 'random_distractor', or 'adversarial_distractor'.

    Returns:
        str: Formatted in-context RAG prompt string.
    """
    if condition in ("random_distractor", "adversarial_distractor") and distractor_context:
        context_text = f"{gold_context}\n\n{distractor_context}"
    else:
        context_text = gold_context

    return (
        "Answer the question based strictly on the provided context. "
        "If the context does not contain sufficient facts to answer, state 'Insufficient context'.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


# ---------------------------------------------------------------------------
# Statistical Significance Testing
# ---------------------------------------------------------------------------

def compute_paired_significance(
    scores_c1: Sequence[float],
    scores_c2: Sequence[float],
) -> Dict[str, float]:
    """Compute paired Wilcoxon signed-rank test between Condition 1 and Condition 2.

    Args:
        scores_c1: Paired scores from Condition 1 (Gold Only).
        scores_c2: Paired scores from Condition 2 (Random Distractor).

    Returns:
        Dict[str, float]: Test statistic and two-sided p-value.
    """
    if len(scores_c1) != len(scores_c2) or len(scores_c1) < 2:
        return {"statistic": 0.0, "p_value": 1.0}

    arr_c1 = np.array(scores_c1, dtype=float)
    arr_c2 = np.array(scores_c2, dtype=float)

    # If all paired differences are zero, p-value is 1.0 (identical distribution)
    if np.all(arr_c1 == arr_c2):
        return {"statistic": 0.0, "p_value": 1.0}

    try:
        res = wilcoxon(arr_c1, arr_c2, zero_method="wilcox", alternative="two-sided")
        return {"statistic": float(res.statistic), "p_value": float(res.pvalue)}
    except Exception as e:
        logger.warning("Wilcoxon signed-rank test failed: %s", e)
        return {"statistic": 0.0, "p_value": 1.0}


def compare_conditions(
    summary_c1: BenchmarkSummary,
    summary_c2: BenchmarkSummary,
) -> Dict[str, Any]:
    """Aggregate paired comparative statistics and Wilcoxon test results across conditions.

    Args:
        summary_c1: BenchmarkSummary for Condition 1 (Gold Only).
        summary_c2: BenchmarkSummary for Condition 2 (Random Distractor).

    Returns:
        Dict[str, Any]: Comprehensive comparative dictionary including metric deltas and p-values.
    """
    c1_records = {r["sample_id"]: r for r in summary_c1.results}
    c2_records = {r["sample_id"]: r for r in summary_c2.results}
    common_ids = [sid for sid in c1_records if sid in c2_records]

    # Paired metrics vectors
    em_c1 = [c1_records[sid]["exact_match"] for sid in common_ids]
    em_c2 = [c2_records[sid]["exact_match"] for sid in common_ids]

    f1_c1 = [c1_records[sid]["token_f1"] for sid in common_ids]
    f1_c2 = [c2_records[sid]["token_f1"] for sid in common_ids]

    gr_c1 = [c1_records[sid]["grounding_score"] for sid in common_ids]
    gr_c2 = [c2_records[sid]["grounding_score"] for sid in common_ids]

    lat_c1 = [c1_records[sid]["latency_ms"] for sid in common_ids]
    lat_c2 = [c2_records[sid]["latency_ms"] for sid in common_ids]

    em_sig = compute_paired_significance(em_c1, em_c2)
    f1_sig = compute_paired_significance(f1_c1, f1_c2)
    gr_sig = compute_paired_significance(gr_c1, gr_c2)
    lat_sig = compute_paired_significance(lat_c1, lat_c2)

    return {
        "model_name": summary_c1.model_name,
        "sample_count": len(common_ids),
        "exact_match": {
            "condition_1_mean": summary_c1.exact_match_mean,
            "condition_1_std": summary_c1.exact_match_std,
            "condition_2_mean": summary_c2.exact_match_mean,
            "condition_2_std": summary_c2.exact_match_std,
            "delta": summary_c2.exact_match_mean - summary_c1.exact_match_mean,
            "wilcoxon_stat": em_sig["statistic"],
            "wilcoxon_p": em_sig["p_value"],
        },
        "token_f1": {
            "condition_1_mean": summary_c1.token_f1_mean,
            "condition_1_std": summary_c1.token_f1_std,
            "condition_2_mean": summary_c2.token_f1_mean,
            "condition_2_std": summary_c2.token_f1_std,
            "delta": summary_c2.token_f1_mean - summary_c1.token_f1_mean,
            "wilcoxon_stat": f1_sig["statistic"],
            "wilcoxon_p": f1_sig["p_value"],
        },
        "grounding_score": {
            "condition_1_mean": summary_c1.grounding_score_mean,
            "condition_1_std": summary_c1.grounding_score_std,
            "condition_2_mean": summary_c2.grounding_score_mean,
            "condition_2_std": summary_c2.grounding_score_std,
            "delta": summary_c2.grounding_score_mean - summary_c1.grounding_score_mean,
            "wilcoxon_stat": gr_sig["statistic"],
            "wilcoxon_p": gr_sig["p_value"],
        },
        "abstention_rate": {
            "condition_1": summary_c1.abstention_rate,
            "condition_2": summary_c2.abstention_rate,
            "delta": summary_c2.abstention_rate - summary_c1.abstention_rate,
        },
        "latency_ms": {
            "condition_1_mean": summary_c1.avg_latency_ms,
            "condition_2_mean": summary_c2.avg_latency_ms,
            "delta": summary_c2.avg_latency_ms - summary_c1.avg_latency_ms,
            "wilcoxon_stat": lat_sig["statistic"],
            "wilcoxon_p": lat_sig["p_value"],
        },
    }


# ---------------------------------------------------------------------------
# BenchmarkRunner Orchestrator
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Dedicated benchmark orchestration engine for automated multi-condition evaluation."""

    def __init__(
        self,
        inference_client: Optional[GemmaInferenceClient] = None,
        evaluator: Optional[ContextNoiseEvaluator] = None,
        dataset_path: str = "data/processed/eval_500.json",
        results_dir: str = "results",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the BenchmarkRunner with clients and dataset.

        Args:
            inference_client: Generation client (defaults to GemmaInferenceClient()).
            evaluator: OSS metric evaluator (defaults to ContextNoiseEvaluator()).
            dataset_path: Filepath to processed benchmark dataset json.
            results_dir: Directory path where results and checkpoints are stored.
            config: Optional evaluation configuration mapping.
        """
        self.inference_client = inference_client or GemmaInferenceClient()
        self.evaluator = evaluator or ContextNoiseEvaluator()
        self.dataset_path = Path(dataset_path)
        self.results_dir = Path(results_dir)
        self.checkpoints_dir = self.results_dir / "checkpoints"
        self.config = config or {}

        # Ensure directory structure exists
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        if not self.dataset_path.is_file():
            raise FileNotFoundError(
                f"Benchmark dataset not found at '{self.dataset_path}'. Run --build-dataset first."
            )

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            self.dataset: List[Dict[str, Any]] = json.load(f)

        logger.info(
            "BenchmarkRunner initialized with %d dataset records from '%s'.",
            len(self.dataset),
            self.dataset_path,
        )

    def get_clean_model_name(self) -> str:
        """Return sanitized model identifier string for filenames."""
        raw = self.inference_client.model_name
        return raw.replace(":", "").replace("/", "_").split()[0]

    def get_checkpoint_path(self, condition: str) -> Path:
        """Return checkpoint json filepath for a given condition."""
        model_tag = self.get_clean_model_name()
        if condition in ("adversarial_distractor", "condition3", "3"):
            return self.checkpoints_dir / f"{model_tag}_condition3_checkpoint.json"
        elif condition in ("random_distractor", "condition2", "2"):
            return self.checkpoints_dir / f"{model_tag}_random_distractor_checkpoint.json"
        elif condition in ("gold_only", "condition1", "1"):
            return self.checkpoints_dir / f"{model_tag}_gold_only_checkpoint.json"
        return self.checkpoints_dir / f"{model_tag}_{condition}_checkpoint.json"

    def load_checkpoint(self, condition: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint dictionary if file exists."""
        ckpt_path = self.get_checkpoint_path(condition)
        paths_to_try = [ckpt_path]
        model_tag = self.get_clean_model_name()
        if condition in ("adversarial_distractor", "condition3", "3"):
            paths_to_try.append(self.checkpoints_dir / f"{model_tag}_adversarial_distractor_checkpoint.json")
            paths_to_try.append(self.checkpoints_dir / "gemma2b_condition3_checkpoint.json")

        for p in paths_to_try:
            if p.is_file():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return data
                except Exception as e:
                    logger.warning("Failed to read checkpoint at '%s': %s", p, e)
        return None

    def save_checkpoint(
        self,
        condition: str,
        records: List[Dict[str, Any]],
        completed_count: int,
    ) -> None:
        """Persist intermediate evaluation records to checkpoint file."""
        ckpt_path = self.get_checkpoint_path(condition)
        payload = {
            "condition": condition,
            "model_name": self.inference_client.model_name,
            "completed_count": completed_count,
            "records": records,
        }
        temp_path = ckpt_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        temp_path.replace(ckpt_path)
        logger.debug("Saved checkpoint to '%s' (%d records).", ckpt_path, completed_count)

    def run_condition(
        self,
        condition: Literal["gold_only", "random_distractor", "adversarial_distractor"],
        limit: Optional[int] = None,
        checkpoint_every: int = 25,
    ) -> BenchmarkSummary:
        """Execute automated benchmark evaluation for a specific experimental condition.

        Args:
            condition: 'gold_only' (Condition 1), 'random_distractor' (Condition 2), or 'adversarial_distractor' (Condition 3).
            limit: Optional integer limit on number of samples to process.
            checkpoint_every: Checkpoint saving frequency (default: 25).

        Returns:
            BenchmarkSummary: Complete evaluated metrics summary and per-sample results.
        """
        total_available = len(self.dataset)
        target_count = min(limit, total_available) if limit is not None else total_available

        logger.info(
            "Executing Benchmark Condition '%s' on %d samples (checkpoint_every=%d)...",
            condition,
            target_count,
            checkpoint_every,
        )

        records: List[Dict[str, Any]] = []
        existing_records: List[Dict[str, Any]] = []
        start_idx = 0

        # Check for existing checkpoint to resume
        ckpt = self.load_checkpoint(condition)
        if ckpt and "records" in ckpt:
            existing_records = ckpt.get("records", [])
            if len(existing_records) >= target_count:
                logger.info(
                    "Checkpoint contains %d completed records (target: %d). Using existing checkpoint.",
                    len(existing_records),
                    target_count,
                )
                records = existing_records[:target_count]
                start_idx = target_count
            else:
                logger.info(
                    "Resuming Condition '%s' from sample %d of %d using checkpoint.",
                    condition,
                    len(existing_records),
                    target_count,
                )
                records = list(existing_records)
                start_idx = len(existing_records)

        # Main evaluation loop
        for i in range(start_idx, target_count):
            sample = self.dataset[i]
            sample_id = sample.get("sample_id", f"sample_{i:04d}")
            query = sample["query"]
            gold_context = sample["gold_context"]
            if condition == "random_distractor":
                distractor = sample.get("random_distractor", None)
            elif condition == "adversarial_distractor":
                distractor = sample.get("adversarial_distractor", None)
            else:
                distractor = None
            ground_truth = sample["ground_truth_answer"]

            prompt = format_rag_prompt(
                query=query,
                gold_context=gold_context,
                distractor_context=distractor,
                condition=condition,
            )

            # Generate output deterministically
            gen_res = self.inference_client.generate(
                prompt=prompt,
                temperature=0.0,
                seed=42,
                max_tokens=150,
            )

            # Evaluate sample using OSS adapter schema
            context_seen = (
                f"{gold_context}\n\n{distractor}"
                if (condition in ("random_distractor", "adversarial_distractor") and distractor)
                else gold_context
            )
            eval_input = EvaluationInput(
                sample_id=sample_id,
                query=query,
                context=context_seen,
                condition=condition,
                response=gen_res.response_text,
                ground_truth=ground_truth,
                distractor_context=distractor if condition in ("random_distractor", "adversarial_distractor") else None,
                metadata={
                    "gold_context": gold_context,
                    "prompt_tokens": gen_res.prompt_tokens,
                    "completion_tokens": gen_res.completion_tokens,
                    "tokens_per_second": gen_res.tokens_per_second,
                },
            )
            metric_res = self.evaluator.evaluate_sample(eval_input)

            record = {
                "sample_id": sample_id,
                "query": query,
                "condition": condition,
                "ground_truth": ground_truth,
                "prompt": prompt,
                "response": gen_res.response_text,
                "exact_match": metric_res.exact_match,
                "token_f1": metric_res.token_f1,
                "grounding_score": metric_res.grounding_score,
                "entity_misattribution": metric_res.entity_misattribution,
                "abstention": metric_res.abstention,
                "error_category": metric_res.error_category,
                "latency_ms": gen_res.total_latency_ms,
                "eval_latency_ms": metric_res.latency_ms,
                "tokens_per_second": gen_res.tokens_per_second,
                "prompt_tokens": gen_res.prompt_tokens,
                "completion_tokens": gen_res.completion_tokens,
            }
            records.append(record)

            if (i + 1) % checkpoint_every == 0 or (i + 1) == target_count:
                self.save_checkpoint(condition, records, completed_count=len(records))
                logger.info(
                    "[%s] Progress: %d / %d samples completed (%.1f%%).",
                    condition,
                    len(records),
                    target_count,
                    (len(records) / target_count) * 100.0,
                )

        # Aggregate summary statistics
        em_vals = [r["exact_match"] for r in records]
        f1_vals = [r["token_f1"] for r in records]
        gr_vals = [r["grounding_score"] for r in records]
        ab_vals = [1.0 if r["abstention"] else 0.0 for r in records]
        mis_vals = [1.0 if r.get("entity_misattribution", False) else 0.0 for r in records]
        lat_vals = [r["latency_ms"] for r in records]
        tps_vals = [r["tokens_per_second"] for r in records]

        summary = BenchmarkSummary(
            condition=condition,
            model_name=self.inference_client.model_name,
            sample_count=len(records),
            exact_match_mean=round(float(np.mean(em_vals)), 4) if em_vals else 0.0,
            exact_match_std=round(float(np.std(em_vals)), 4) if em_vals else 0.0,
            token_f1_mean=round(float(np.mean(f1_vals)), 4) if f1_vals else 0.0,
            token_f1_std=round(float(np.std(f1_vals)), 4) if f1_vals else 0.0,
            grounding_score_mean=round(float(np.mean(gr_vals)), 4) if gr_vals else 0.0,
            grounding_score_std=round(float(np.std(gr_vals)), 4) if gr_vals else 0.0,
            abstention_rate=round(float(np.mean(ab_vals)), 4) if ab_vals else 0.0,
            entity_misattribution_rate=round(float(np.mean(mis_vals)), 4) if mis_vals else 0.0,
            avg_latency_ms=round(float(np.mean(lat_vals)), 2) if lat_vals else 0.0,
            avg_tokens_per_second=round(float(np.mean(tps_vals)), 2) if tps_vals else 0.0,
            results=records,
        )

        # Save final state to checkpoint only if progress was extended or matched
        if len(records) >= len(existing_records):
            self.save_checkpoint(condition, records, completed_count=len(records))
        return summary

    def generate_comparative_report(
        self,
        c1_results: Any,
        c2_results: Any,
        c3_results: Any,
        output_path: Optional[str | Path] = None,
    ) -> ComparativeAnalysis:
        """Generate and save 3-way comparative report using this runner's defaults."""
        out = output_path or (self.results_dir / f"{self.get_clean_model_name()}_3way_summary.json")
        return generate_comparative_report(
            c1_results=c1_results,
            c2_results=c2_results,
            c3_results=c3_results,
            output_path=out,
        )


# ---------------------------------------------------------------------------
# 3-Way Comparative Report Generator
# ---------------------------------------------------------------------------

def _extract_summary_and_records(item: Any) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]], str]:
    """Helper to extract summary metadata and per-sample mapping from various input types."""
    if isinstance(item, BenchmarkSummary):
        records = {r["sample_id"]: r for r in item.results}
        summary = item.model_dump(exclude={"results"})
        model_name = item.model_name
        return summary, records, model_name

    if isinstance(item, (str, Path)):
        with open(item, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(item, dict):
        data = item
    else:
        raise TypeError(f"Unsupported benchmark result type: {type(item)}")

    records_list = data.get("records") or data.get("results", [])
    records = {r["sample_id"]: r for r in records_list}
    summary = data.get("summary") or data
    model_name = data.get("model_name") or summary.get("model_name", "gemma:2b")
    return summary, records, model_name


def generate_comparative_report(
    c1_results: Any,
    c2_results: Any,
    c3_results: Any,
    output_path: Optional[str | Path] = None,
) -> ComparativeAnalysis:
    """Generate 3-way comparative benchmark analysis across Conditions 1, 2, and 3.

    Compares mean and standard deviations for EM, F1, Grounding Score, Entity Misattribution Rate,
    and Abstention Rate. Computes relative drops:
        Delta EM (1 -> 3) = ((EM_C3 - EM_C1) / EM_C1) * 100%
        Delta F1 (1 -> 3) = ((F1_C3 - F1_C1) / F1_C1) * 100%
    Calculates paired Wilcoxon signed-rank test p-values between C1 vs C3 and C2 vs C3,
    and serializes the aggregated report to json.

    Args:
        c1_results: BenchmarkSummary, dict, or filepath for Condition 1 (Gold Only).
        c2_results: BenchmarkSummary, dict, or filepath for Condition 2 (Random Distractor).
        c3_results: BenchmarkSummary, dict, or filepath for Condition 3 (Adversarial Distractor).
        output_path: Optional filepath to serialize ComparativeAnalysis json.

    Returns:
        ComparativeAnalysis: Pydantic model with 3-way comparative metrics, deltas, and Wilcoxon p-values.
    """
    sum_c1, rec_c1, model_1 = _extract_summary_and_records(c1_results)
    sum_c2, rec_c2, model_2 = _extract_summary_and_records(c2_results)
    sum_c3, rec_c3, model_3 = _extract_summary_and_records(c3_results)

    model_name = model_1 or model_2 or model_3 or "gemma:2b"

    # Identify common sample IDs across all three conditions
    common_ids = [sid for sid in rec_c1 if sid in rec_c2 and sid in rec_c3]
    if not common_ids:
        common_ids = list(rec_c3.keys())

    # Extract paired metric vectors
    em_1 = [float(rec_c1[sid]["exact_match"]) for sid in common_ids if sid in rec_c1]
    em_2 = [float(rec_c2[sid]["exact_match"]) for sid in common_ids if sid in rec_c2]
    em_3 = [float(rec_c3[sid]["exact_match"]) for sid in common_ids if sid in rec_c3]

    f1_1 = [float(rec_c1[sid]["token_f1"]) for sid in common_ids if sid in rec_c1]
    f1_2 = [float(rec_c2[sid]["token_f1"]) for sid in common_ids if sid in rec_c2]
    f1_3 = [float(rec_c3[sid]["token_f1"]) for sid in common_ids if sid in rec_c3]

    gr_1 = [float(rec_c1[sid]["grounding_score"]) for sid in common_ids if sid in rec_c1]
    gr_2 = [float(rec_c2[sid]["grounding_score"]) for sid in common_ids if sid in rec_c2]
    gr_3 = [float(rec_c3[sid]["grounding_score"]) for sid in common_ids if sid in rec_c3]

    mis_1 = [1.0 if rec_c1[sid].get("entity_misattribution", False) else 0.0 for sid in common_ids if sid in rec_c1]
    mis_2 = [1.0 if rec_c2[sid].get("entity_misattribution", False) else 0.0 for sid in common_ids if sid in rec_c2]
    mis_3 = [1.0 if rec_c3[sid].get("entity_misattribution", False) else 0.0 for sid in common_ids if sid in rec_c3]

    ab_1 = [1.0 if rec_c1[sid].get("abstention", False) else 0.0 for sid in common_ids if sid in rec_c1]
    ab_2 = [1.0 if rec_c2[sid].get("abstention", False) else 0.0 for sid in common_ids if sid in rec_c2]
    ab_3 = [1.0 if rec_c3[sid].get("abstention", False) else 0.0 for sid in common_ids if sid in rec_c3]

    lat_1 = [float(rec_c1[sid].get("latency_ms", 0.0)) for sid in common_ids if sid in rec_c1]
    lat_2 = [float(rec_c2[sid].get("latency_ms", 0.0)) for sid in common_ids if sid in rec_c2]
    lat_3 = [float(rec_c3[sid].get("latency_ms", 0.0)) for sid in common_ids if sid in rec_c3]

    # Compute descriptive statistics
    em_mean_1, em_std_1 = float(np.mean(em_1)) if em_1 else 0.0, float(np.std(em_1)) if em_1 else 0.0
    em_mean_2, em_std_2 = float(np.mean(em_2)) if em_2 else 0.0, float(np.std(em_2)) if em_2 else 0.0
    em_mean_3, em_std_3 = float(np.mean(em_3)) if em_3 else 0.0, float(np.std(em_3)) if em_3 else 0.0

    f1_mean_1, f1_std_1 = float(np.mean(f1_1)) if f1_1 else 0.0, float(np.std(f1_1)) if f1_1 else 0.0
    f1_mean_2, f1_std_2 = float(np.mean(f1_2)) if f1_2 else 0.0, float(np.std(f1_2)) if f1_2 else 0.0
    f1_mean_3, f1_std_3 = float(np.mean(f1_3)) if f1_3 else 0.0, float(np.std(f1_3)) if f1_3 else 0.0

    gr_mean_1, gr_std_1 = float(np.mean(gr_1)) if gr_1 else 0.0, float(np.std(gr_1)) if gr_1 else 0.0
    gr_mean_2, gr_std_2 = float(np.mean(gr_2)) if gr_2 else 0.0, float(np.std(gr_2)) if gr_2 else 0.0
    gr_mean_3, gr_std_3 = float(np.mean(gr_3)) if gr_3 else 0.0, float(np.std(gr_3)) if gr_3 else 0.0

    mis_rate_1 = float(np.mean(mis_1)) if mis_1 else 0.0
    mis_rate_2 = float(np.mean(mis_2)) if mis_2 else 0.0
    mis_rate_3 = float(np.mean(mis_3)) if mis_3 else 0.0

    ab_rate_1 = float(np.mean(ab_1)) if ab_1 else 0.0
    ab_rate_2 = float(np.mean(ab_2)) if ab_2 else 0.0
    ab_rate_3 = float(np.mean(ab_3)) if ab_3 else 0.0

    lat_mean_1 = float(np.mean(lat_1)) if lat_1 else 0.0
    lat_mean_2 = float(np.mean(lat_2)) if lat_2 else 0.0
    lat_mean_3 = float(np.mean(lat_3)) if lat_3 else 0.0

    # Wilcoxon paired tests (C1 vs C3 and C2 vs C3)
    sig_em_1_3 = compute_paired_significance(em_1, em_3)
    sig_em_2_3 = compute_paired_significance(em_2, em_3)

    sig_f1_1_3 = compute_paired_significance(f1_1, f1_3)
    sig_f1_2_3 = compute_paired_significance(f1_2, f1_3)

    sig_gr_1_3 = compute_paired_significance(gr_1, gr_3)
    sig_gr_2_3 = compute_paired_significance(gr_2, gr_3)

    sig_mis_1_3 = compute_paired_significance(mis_1, mis_3)
    sig_mis_2_3 = compute_paired_significance(mis_2, mis_3)

    sig_lat_1_3 = compute_paired_significance(lat_1, lat_3)
    sig_lat_2_3 = compute_paired_significance(lat_2, lat_3)

    # Relative performance drops
    if em_mean_1 > 0:
        delta_em_1_to_3_pct = round(((em_mean_3 - em_mean_1) / em_mean_1) * 100.0, 2)
    else:
        delta_em_1_to_3_pct = 0.0

    if f1_mean_1 > 0:
        delta_f1_1_to_3_pct = round(((f1_mean_3 - f1_mean_1) / f1_mean_1) * 100.0, 2)
    else:
        delta_f1_1_to_3_pct = 0.0

    if em_mean_2 > 0:
        delta_em_2_to_3_pct = round(((em_mean_3 - em_mean_2) / em_mean_2) * 100.0, 2)
    else:
        delta_em_2_to_3_pct = 0.0

    if f1_mean_2 > 0:
        delta_f1_2_to_3_pct = round(((f1_mean_3 - f1_mean_2) / f1_mean_2) * 100.0, 2)
    else:
        delta_f1_2_to_3_pct = 0.0

    report = ComparativeAnalysis(
        model_name=model_name,
        sample_count=len(common_ids),
        condition_1_summary=sum_c1,
        condition_2_summary=sum_c2,
        condition_3_summary=sum_c3,
        exact_match={
            "c1_mean": round(em_mean_1, 4),
            "c1_std": round(em_std_1, 4),
            "c2_mean": round(em_mean_2, 4),
            "c2_std": round(em_std_2, 4),
            "c3_mean": round(em_mean_3, 4),
            "c3_std": round(em_std_3, 4),
            "delta_1_to_3": round(em_mean_3 - em_mean_1, 4),
            "delta_2_to_3": round(em_mean_3 - em_mean_2, 4),
            "wilcoxon_c1_c3_stat": sig_em_1_3["statistic"],
            "wilcoxon_c1_c3_p": sig_em_1_3["p_value"],
            "wilcoxon_c2_c3_stat": sig_em_2_3["statistic"],
            "wilcoxon_c2_c3_p": sig_em_2_3["p_value"],
        },
        token_f1={
            "c1_mean": round(f1_mean_1, 4),
            "c1_std": round(f1_std_1, 4),
            "c2_mean": round(f1_mean_2, 4),
            "c2_std": round(f1_std_2, 4),
            "c3_mean": round(f1_mean_3, 4),
            "c3_std": round(f1_std_3, 4),
            "delta_1_to_3": round(f1_mean_3 - f1_mean_1, 4),
            "delta_2_to_3": round(f1_mean_3 - f1_mean_2, 4),
            "wilcoxon_c1_c3_stat": sig_f1_1_3["statistic"],
            "wilcoxon_c1_c3_p": sig_f1_1_3["p_value"],
            "wilcoxon_c2_c3_stat": sig_f1_2_3["statistic"],
            "wilcoxon_c2_c3_p": sig_f1_2_3["p_value"],
        },
        grounding_score={
            "c1_mean": round(gr_mean_1, 4),
            "c1_std": round(gr_std_1, 4),
            "c2_mean": round(gr_mean_2, 4),
            "c2_std": round(gr_std_2, 4),
            "c3_mean": round(gr_mean_3, 4),
            "c3_std": round(gr_std_3, 4),
            "delta_1_to_3": round(gr_mean_3 - gr_mean_1, 4),
            "delta_2_to_3": round(gr_mean_3 - gr_mean_2, 4),
            "wilcoxon_c1_c3_stat": sig_gr_1_3["statistic"],
            "wilcoxon_c1_c3_p": sig_gr_1_3["p_value"],
            "wilcoxon_c2_c3_stat": sig_gr_2_3["statistic"],
            "wilcoxon_c2_c3_p": sig_gr_2_3["p_value"],
        },
        entity_misattribution={
            "c1_rate": round(mis_rate_1, 4),
            "c2_rate": round(mis_rate_2, 4),
            "c3_rate": round(mis_rate_3, 4),
            "delta_1_to_3": round(mis_rate_3 - mis_rate_1, 4),
            "delta_2_to_3": round(mis_rate_3 - mis_rate_2, 4),
            "wilcoxon_c1_c3_stat": sig_mis_1_3["statistic"],
            "wilcoxon_c1_c3_p": sig_mis_1_3["p_value"],
            "wilcoxon_c2_c3_stat": sig_mis_2_3["statistic"],
            "wilcoxon_c2_c3_p": sig_mis_2_3["p_value"],
        },
        abstention_rate={
            "c1_rate": round(ab_rate_1, 4),
            "c2_rate": round(ab_rate_2, 4),
            "c3_rate": round(ab_rate_3, 4),
            "delta_1_to_3": round(ab_rate_3 - ab_rate_1, 4),
            "delta_2_to_3": round(ab_rate_3 - ab_rate_2, 4),
        },
        latency_ms={
            "c1_mean": round(lat_mean_1, 2),
            "c2_mean": round(lat_mean_2, 2),
            "c3_mean": round(lat_mean_3, 2),
            "delta_1_to_3": round(lat_mean_3 - lat_mean_1, 2),
            "delta_2_to_3": round(lat_mean_3 - lat_mean_2, 2),
            "wilcoxon_c1_c3_stat": sig_lat_1_3["statistic"],
            "wilcoxon_c1_c3_p": sig_lat_1_3["p_value"],
            "wilcoxon_c2_c3_stat": sig_lat_2_3["statistic"],
            "wilcoxon_c2_c3_p": sig_lat_2_3["p_value"],
        },
        relative_drops={
            "delta_em_1_to_3_pct": delta_em_1_to_3_pct,
            "delta_f1_1_to_3_pct": delta_f1_1_to_3_pct,
            "delta_em_2_to_3_pct": delta_em_2_to_3_pct,
            "delta_f1_2_to_3_pct": delta_f1_2_to_3_pct,
        },
        significance_tests={
            "c1_vs_c3": {
                "exact_match_p": sig_em_1_3["p_value"],
                "token_f1_p": sig_f1_1_3["p_value"],
                "grounding_score_p": sig_gr_1_3["p_value"],
                "entity_misattribution_p": sig_mis_1_3["p_value"],
                "latency_p": sig_lat_1_3["p_value"],
            },
            "c2_vs_c3": {
                "exact_match_p": sig_em_2_3["p_value"],
                "token_f1_p": sig_f1_2_3["p_value"],
                "grounding_score_p": sig_gr_2_3["p_value"],
                "entity_misattribution_p": sig_mis_2_3["p_value"],
                "latency_p": sig_lat_2_3["p_value"],
            },
        },
    )

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)
        logger.info("Saved 3-way comparative benchmark report to '%s'.", out_p)

    return report


# ---------------------------------------------------------------------------
# Visual Reporting (Rich & Fallback ASCII)
# ---------------------------------------------------------------------------

def display_benchmark_rich(comparison: Dict[str, Any]) -> None:
    """Render comparative benchmark evaluation results table using rich formatting."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console(width=max(Console().width or 100, 110))

    table = Table(
        title=f"Edge-RAG-Eval | Benchmark Summary ({comparison['model_name']})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Evaluation Dimension", style="bold white")
    table.add_column("Condition 1 (Gold Only)", justify="center")
    table.add_column("Condition 2 (Random Distractor)", justify="center")
    table.add_column("Delta (C2 - C1)", justify="center")
    table.add_column("Wilcoxon p-value", justify="center")
    table.add_column("Significance", justify="center")

    # 1. Exact Match
    em = comparison["exact_match"]
    em_sig = "[bold green]p < 0.001[/bold green]" if em["wilcoxon_p"] < 0.001 else f"p = {em['wilcoxon_p']:.4f}"
    sig_label = "[bold red]Stat. Significant[/bold red]" if em["wilcoxon_p"] < 0.05 else "[dim]Not Significant[/dim]"
    delta_str = f"[bold red]{em['delta']:+.4f}[/bold red]" if em["delta"] < 0 else f"{em['delta']:+.4f}"
    table.add_row(
        "Exact Match (EM)",
        f"{em['condition_1_mean']:.4f} +/- {em['condition_1_std']:.3f}",
        f"{em['condition_2_mean']:.4f} +/- {em['condition_2_std']:.3f}",
        delta_str,
        em_sig,
        sig_label,
    )

    # 2. Token F1
    f1 = comparison["token_f1"]
    f1_sig = "[bold green]p < 0.001[/bold green]" if f1["wilcoxon_p"] < 0.001 else f"p = {f1['wilcoxon_p']:.4f}"
    sig_label = "[bold red]Stat. Significant[/bold red]" if f1["wilcoxon_p"] < 0.05 else "[dim]Not Significant[/dim]"
    delta_str = f"[bold red]{f1['delta']:+.4f}[/bold red]" if f1["delta"] < 0 else f"{f1['delta']:+.4f}"
    table.add_row(
        "Token F1 Score",
        f"{f1['condition_1_mean']:.4f} +/- {f1['condition_1_std']:.3f}",
        f"{f1['condition_2_mean']:.4f} +/- {f1['condition_2_std']:.3f}",
        delta_str,
        f1_sig,
        sig_label,
    )

    # 3. Grounding Score
    gr = comparison["grounding_score"]
    gr_sig = "[bold green]p < 0.001[/bold green]" if gr["wilcoxon_p"] < 0.001 else f"p = {gr['wilcoxon_p']:.4f}"
    sig_label = "[bold red]Stat. Significant[/bold red]" if gr["wilcoxon_p"] < 0.05 else "[dim]Not Significant[/dim]"
    delta_str = f"[bold red]{gr['delta']:+.4f}[/bold red]" if gr["delta"] < 0 else f"{gr['delta']:+.4f}"
    table.add_row(
        "Grounding Score",
        f"{gr['condition_1_mean']:.4f} +/- {gr['condition_1_std']:.3f}",
        f"{gr['condition_2_mean']:.4f} +/- {gr['condition_2_std']:.3f}",
        delta_str,
        gr_sig,
        sig_label,
    )

    # 4. Calibrated Abstention
    ab = comparison["abstention_rate"]
    table.add_row(
        "Calibrated Abstention",
        f"{ab['condition_1'] * 100:.2f}%",
        f"{ab['condition_2'] * 100:.2f}%",
        f"{ab['delta'] * 100:+.2f}%",
        "[dim]N/A (binary)[/dim]",
        "[dim]Baseline Refusal[/dim]",
    )

    # 5. Average Latency
    lat = comparison["latency_ms"]
    lat_sig = "[bold green]p < 0.001[/bold green]" if lat["wilcoxon_p"] < 0.001 else f"p = {lat['wilcoxon_p']:.4f}"
    delta_str = f"[bold yellow]{lat['delta']:+.1f} ms[/bold yellow]" if lat["delta"] > 0 else f"{lat['delta']:+.1f} ms"
    table.add_row(
        "Avg Latency (ms)",
        f"{lat['condition_1_mean']:.1f} ms",
        f"{lat['condition_2_mean']:.1f} ms",
        delta_str,
        lat_sig,
        "[cyan]Context Overhead[/cyan]",
    )

    console.print()
    console.print(Panel.fit(
        f"[bold green]Automated Benchmark Execution Complete[/bold green]\n"
        f"[bold white]Target Model:[/] [cyan]{comparison['model_name']}[/cyan]\n"
        f"[bold white]Sample Count:[/] {comparison['sample_count']} samples evaluated per condition\n"
        f"[bold white]Condition 1:[/] Gold Context Only (Control Baseline)\n"
        f"[bold white]Condition 2:[/] Gold Context + Random Distractor (Low Noise Overhead)\n"
        f"[bold white]Statistical Protocol:[/] Paired two-sided Wilcoxon signed-rank test",
        border_style="green",
    ))
    console.print(table)
    console.print()


def display_benchmark_ascii(comparison: Dict[str, Any]) -> None:
    """Fallback plain-text ASCII renderer for benchmark comparison."""
    print("\n" + "=" * 80)
    print(f"Edge-RAG-Eval | Benchmark Comparison ({comparison['model_name']})")
    print("=" * 80)
    print(f"Evaluated Samples: {comparison['sample_count']} per condition\n")

    em = comparison["exact_match"]
    print(f"Exact Match (EM)    : C1={em['condition_1_mean']:.4f} | C2={em['condition_2_mean']:.4f} | Delta={em['delta']:+.4f} (p={em['wilcoxon_p']:.4e})")

    f1 = comparison["token_f1"]
    print(f"Token F1 Score      : C1={f1['condition_1_mean']:.4f} | C2={f1['condition_2_mean']:.4f} | Delta={f1['delta']:+.4f} (p={f1['wilcoxon_p']:.4e})")

    gr = comparison["grounding_score"]
    print(f"Grounding Score     : C1={gr['condition_1_mean']:.4f} | C2={gr['condition_2_mean']:.4f} | Delta={gr['delta']:+.4f} (p={gr['wilcoxon_p']:.4e})")

    ab = comparison["abstention_rate"]
    print(f"Abstention Rate     : C1={ab['condition_1']*100:.1f}% | C2={ab['condition_2']*100:.1f}% | Delta={ab['delta']*100:+.1f}%")

    lat = comparison["latency_ms"]
    print(f"Avg Latency (ms)    : C1={lat['condition_1_mean']:.1f}ms | C2={lat['condition_2_mean']:.1f}ms | Delta={lat['delta']:+.1f}ms (p={lat['wilcoxon_p']:.4e})")
    print("=" * 80 + "\n")


def display_3way_benchmark_rich(report: ComparativeAnalysis | Dict[str, Any]) -> None:
    """Render unified 3-way comparative benchmark results table using rich formatting."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    data = report.model_dump() if isinstance(report, ComparativeAnalysis) else report
    console = Console(width=max(Console().width or 100, 115))

    table = Table(
        title=f"Edge-RAG-Eval | 3-Way Benchmark Summary ({data['model_name']})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Evaluation Dimension", style="bold white")
    table.add_column("Condition 1 (Gold)", justify="center")
    table.add_column("Condition 2 (Random)", justify="center")
    table.add_column("Condition 3 (Adversarial)", justify="center")
    table.add_column("Delta (C1 -> C3)", justify="center")
    table.add_column("p-value", justify="center")
    table.add_column("Significance", justify="center")

    # 1. Exact Match
    em = data["exact_match"]
    em_p = em["wilcoxon_c1_c3_p"]
    em_sig = "[bold green]p < 0.001[/bold green]" if em_p < 0.001 else f"p = {em_p:.4f}"
    em_label = "[bold red]Stat. Significant[/bold red]" if em_p < 0.05 else "[dim]Not Significant[/dim]"
    delta_em = f"{em['delta_1_to_3']:+.4f}"
    rel_em = data["relative_drops"].get("delta_em_1_to_3_pct", 0.0)
    if rel_em != 0.0:
        delta_em += f" ({rel_em:+.1f}%)"
    table.add_row(
        "Exact Match (EM)",
        f"{em['c1_mean']:.4f} +/- {em['c1_std']:.3f}",
        f"{em['c2_mean']:.4f} +/- {em['c2_std']:.3f}",
        f"{em['c3_mean']:.4f} +/- {em['c3_std']:.3f}",
        delta_em,
        em_sig,
        em_label,
    )

    # 2. Token F1
    f1 = data["token_f1"]
    f1_p = f1["wilcoxon_c1_c3_p"]
    f1_sig = "[bold green]p < 0.001[/bold green]" if f1_p < 0.001 else f"p = {f1_p:.4f}"
    f1_label = "[bold red]Stat. Significant[/bold red]" if f1_p < 0.05 else "[dim]Not Significant[/dim]"
    delta_f1 = f"[bold red]{f1['delta_1_to_3']:+.4f}[/bold red]" if f1['delta_1_to_3'] < 0 else f"{f1['delta_1_to_3']:+.4f}"
    rel_f1 = data["relative_drops"].get("delta_f1_1_to_3_pct", 0.0)
    if rel_f1 != 0.0:
        delta_f1 += f" ({rel_f1:+.1f}%)"
    table.add_row(
        "Token F1 Score",
        f"{f1['c1_mean']:.4f} +/- {f1['c1_std']:.3f}",
        f"{f1['c2_mean']:.4f} +/- {f1['c2_std']:.3f}",
        f"{f1['c3_mean']:.4f} +/- {f1['c3_std']:.3f}",
        delta_f1,
        f1_sig,
        f1_label,
    )

    # 3. Grounding Score
    gr = data["grounding_score"]
    gr_p = gr["wilcoxon_c1_c3_p"]
    gr_sig = "[bold green]p < 0.001[/bold green]" if gr_p < 0.001 else f"p = {gr_p:.4f}"
    gr_label = "[bold red]Stat. Significant[/bold red]" if gr_p < 0.05 else "[dim]Not Significant[/dim]"
    delta_gr = f"[bold red]{gr['delta_1_to_3']:+.4f}[/bold red]" if gr['delta_1_to_3'] < 0 else f"{gr['delta_1_to_3']:+.4f}"
    table.add_row(
        "Grounding Score",
        f"{gr['c1_mean']:.4f} +/- {gr['c1_std']:.3f}",
        f"{gr['c2_mean']:.4f} +/- {gr['c2_std']:.3f}",
        f"{gr['c3_mean']:.4f} +/- {gr['c3_std']:.3f}",
        delta_gr,
        gr_sig,
        gr_label,
    )

    # 4. Entity Misattribution (%)
    mis = data["entity_misattribution"]
    mis_p = mis["wilcoxon_c1_c3_p"]
    mis_sig = "[bold green]p < 0.001[/bold green]" if mis_p < 0.001 else f"p = {mis_p:.4f}"
    mis_label = "[bold red]Distractor Poisoned[/bold red]" if mis["c3_rate"] > 0 else "[dim]Zero Poisoning[/dim]"
    delta_mis = f"[bold red]{mis['delta_1_to_3'] * 100:+.1f}%[/bold red]"
    table.add_row(
        "Entity Misattribution (%)",
        f"{mis['c1_rate'] * 100:.1f}%",
        f"{mis['c2_rate'] * 100:.1f}%",
        f"{mis['c3_rate'] * 100:.1f}%",
        delta_mis,
        mis_sig,
        mis_label,
    )

    # 5. Abstention Rate (%)
    ab = data["abstention_rate"]
    table.add_row(
        "Abstention Rate (%)",
        f"{ab['c1_rate'] * 100:.1f}%",
        f"{ab['c2_rate'] * 100:.1f}%",
        f"{ab['c3_rate'] * 100:.1f}%",
        f"{ab['delta_1_to_3'] * 100:+.1f}%",
        "[dim]N/A (binary)[/dim]",
        "[cyan]Calibrated Uncertainty[/cyan]" if ab["c3_rate"] > 0 else "[dim]Zero Refusal[/dim]",
    )

    # 6. Avg Latency (ms)
    lat = data["latency_ms"]
    lat_p = lat["wilcoxon_c1_c3_p"]
    lat_sig = "[bold green]p < 0.001[/bold green]" if lat_p < 0.001 else f"p = {lat_p:.4f}"
    delta_lat = f"[bold yellow]{lat['delta_1_to_3']:+.1f} ms[/bold yellow]"
    table.add_row(
        "Avg Latency (ms)",
        f"{lat['c1_mean']:.1f} ms",
        f"{lat['c2_mean']:.1f} ms",
        f"{lat['c3_mean']:.1f} ms",
        delta_lat,
        lat_sig,
        "[cyan]Context Overhead[/cyan]",
    )

    console.print()
    console.print(Panel.fit(
        f"[bold green]3-Way Benchmark Comparative Matrix Complete[/bold green]\n"
        f"[bold white]Target Model:[/] [cyan]{data['model_name']}[/cyan]\n"
        f"[bold white]Sample Count:[/] {data['sample_count']} samples evaluated across all 3 conditions\n"
        f"[bold white]Condition 1:[/] Gold Context Only (Control Baseline)\n"
        f"[bold white]Condition 2:[/] Gold Context + Random Distractor (Low Noise)\n"
        f"[bold white]Condition 3:[/] Gold Context + Adversarial Distractor (High Semantic Noise)\n"
        f"[bold white]Statistical Protocol:[/] Paired two-sided Wilcoxon signed-rank test",
        border_style="green",
    ))
    console.print(table)
    console.print()


def display_3way_benchmark_ascii(report: ComparativeAnalysis | Dict[str, Any]) -> None:
    """Fallback plain-text ASCII renderer for 3-way benchmark comparison."""
    data = report.model_dump() if isinstance(report, ComparativeAnalysis) else report
    print("\n" + "=" * 88)
    print(f"Edge-RAG-Eval | 3-Way Benchmark Comparison ({data['model_name']})")
    print("=" * 88)
    print(f"Evaluated Samples: {data['sample_count']} per condition\n")

    em = data["exact_match"]
    print(f"Exact Match (EM)          : C1={em['c1_mean']:.4f} | C2={em['c2_mean']:.4f} | C3={em['c3_mean']:.4f} | Delta(1->3)={em['delta_1_to_3']:+.4f} (p={em['wilcoxon_c1_c3_p']:.4e})")

    f1 = data["token_f1"]
    print(f"Token F1 Score            : C1={f1['c1_mean']:.4f} | C2={f1['c2_mean']:.4f} | C3={f1['c3_mean']:.4f} | Delta(1->3)={f1['delta_1_to_3']:+.4f} (p={f1['wilcoxon_c1_c3_p']:.4e})")

    gr = data["grounding_score"]
    print(f"Grounding Score           : C1={gr['c1_mean']:.4f} | C2={gr['c2_mean']:.4f} | C3={gr['c3_mean']:.4f} | Delta(1->3)={gr['delta_1_to_3']:+.4f} (p={gr['wilcoxon_c1_c3_p']:.4e})")

    mis = data["entity_misattribution"]
    print(f"Entity Misattribution (%) : C1={mis['c1_rate']*100:.1f}% | C2={mis['c2_rate']*100:.1f}% | C3={mis['c3_rate']*100:.1f}% | Delta(1->3)={mis['delta_1_to_3']*100:+.1f}% (p={mis['wilcoxon_c1_c3_p']:.4e})")

    ab = data["abstention_rate"]
    print(f"Abstention Rate (%)       : C1={ab['c1_rate']*100:.1f}% | C2={ab['c2_rate']*100:.1f}% | C3={ab['c3_rate']*100:.1f}% | Delta(1->3)={ab['delta_1_to_3']*100:+.1f}%")

    lat = data["latency_ms"]
    print(f"Avg Latency (ms)          : C1={lat['c1_mean']:.1f}ms | C2={lat['c2_mean']:.1f}ms | C3={lat['c3_mean']:.1f}ms | Delta(1->3)={lat['delta_1_to_3']:+.1f}ms (p={lat['wilcoxon_c1_c3_p']:.4e})")
    print("=" * 88 + "\n")
