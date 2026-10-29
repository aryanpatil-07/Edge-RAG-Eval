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

    condition: Literal["gold_only", "random_distractor"] = Field(
        description="Experimental condition name ('gold_only' or 'random_distractor')."
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
    avg_latency_ms: float = Field(description="Average total generation and evaluation latency in ms.")
    avg_tokens_per_second: float = Field(description="Average generation throughput in tokens/sec.")
    results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-sample evaluation records."
    )


# ---------------------------------------------------------------------------
# Prompt Assembly Formatter
# ---------------------------------------------------------------------------

def format_rag_prompt(
    query: str,
    gold_context: str,
    distractor_context: Optional[str] = None,
    condition: Literal["gold_only", "random_distractor"] = "gold_only",
) -> str:
    """Standardizes the in-context RAG prompt structure for edge evaluation.

    Args:
        query: Target question string.
        gold_context: True reference context passage.
        distractor_context: Optional random distractor passage for Condition 2.
        condition: 'gold_only' (Condition 1) or 'random_distractor' (Condition 2).

    Returns:
        str: Formatted in-context RAG prompt string.
    """
    if condition == "random_distractor" and distractor_context:
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
        return self.checkpoints_dir / f"{model_tag}_{condition}_checkpoint.json"

    def load_checkpoint(self, condition: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint dictionary if file exists."""
        ckpt_path = self.get_checkpoint_path(condition)
        if ckpt_path.is_file():
            try:
                with open(ckpt_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data
            except Exception as e:
                logger.warning("Failed to read checkpoint at '%s': %s", ckpt_path, e)
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
        condition: Literal["gold_only", "random_distractor"],
        limit: Optional[int] = None,
        checkpoint_every: int = 25,
    ) -> BenchmarkSummary:
        """Execute automated benchmark evaluation for a specific experimental condition.

        Args:
            condition: 'gold_only' (Condition 1) or 'random_distractor' (Condition 2).
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
            distractor = sample.get("random_distractor", None)
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
            context_seen = f"{gold_context}\n\n{distractor}" if (condition == "random_distractor" and distractor) else gold_context
            eval_input = EvaluationInput(
                sample_id=sample_id,
                query=query,
                context=context_seen,
                condition=condition,
                response=gen_res.response_text,
                ground_truth=ground_truth,
                distractor_context=distractor if condition == "random_distractor" else None,
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
            avg_latency_ms=round(float(np.mean(lat_vals)), 2) if lat_vals else 0.0,
            avg_tokens_per_second=round(float(np.mean(tps_vals)), 2) if tps_vals else 0.0,
            results=records,
        )

        # Save final state to checkpoint only if progress was extended or matched
        if len(records) >= len(existing_records):
            self.save_checkpoint(condition, records, completed_count=len(records))
        return summary


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
