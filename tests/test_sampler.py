"""Unit and integration tests for adversarial distractor sampler and dataset builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from src.data.distractor_sampler import (
    CURATED_SEED_CORPUS,
    BenchmarkSample,
    DistractorSampler,
    compute_statistical_separation,
    perturb_entities,
)
from src.pipeline import main, run_sampler_test


@pytest.fixture(scope="module")
def sampler() -> DistractorSampler:
    """Shared DistractorSampler instance for test module."""
    return DistractorSampler(device="cpu", seed=42)


def test_random_distractor_upper_similarity_bound(sampler: DistractorSampler):
    """Verify random distractors strictly adhere to upper similarity bound (< 0.25)."""
    for item in CURATED_SEED_CORPUS:
        query = item["query"]
        gold = item["gold_context"]
        candidates = [
            item["random_context"],
            item["adversarial_context"],
            "Photosynthesis generates oxygen in chloroplasts using solar radiation.",
        ]

        rand_dist, _, sim_rand, _ = sampler.sample_distractors(
            query=query,
            gold_passage=gold,
            candidate_pool=candidates,
            random_threshold=0.25,
            adversarial_threshold=0.65,
        )

        assert isinstance(rand_dist, str)
        assert len(rand_dist) > 0
        assert sim_rand < 0.25, f"Random distractor similarity {sim_rand} exceeded upper threshold 0.25"


def test_adversarial_distractor_lower_similarity_bound(sampler: DistractorSampler):
    """Verify adversarial distractors strictly meet lower similarity bound (>= 0.65)."""
    for item in CURATED_SEED_CORPUS:
        query = item["query"]
        gold = item["gold_context"]
        candidates = [
            item["random_context"],
            item["adversarial_context"],
        ]

        _, adv_dist, _, sim_adv = sampler.sample_distractors(
            query=query,
            gold_passage=gold,
            candidate_pool=candidates,
            random_threshold=0.25,
            adversarial_threshold=0.65,
        )

        assert isinstance(adv_dist, str)
        assert len(adv_dist) > 0
        assert sim_adv >= 0.65, f"Adversarial distractor similarity {sim_adv} fell below threshold 0.65"


def test_dataset_schema_integrity_eval_500():
    """Verify dataset schema and bounds for generated 500-sample benchmark dataset."""
    dataset_path = Path("data/processed/eval_500.json")
    assert dataset_path.is_file(), "data/processed/eval_500.json must exist"

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 500, f"Expected 500 samples, observed {len(data)}"

    for i, item in enumerate(data):
        # 1. Pydantic schema validation
        sample = BenchmarkSample.model_validate(item)
        assert sample.sample_id == f"eval_{i+1:03d}"
        assert len(sample.query.strip()) > 0
        assert len(sample.ground_truth_answer.strip()) > 0
        assert len(sample.gold_context.strip()) > 0
        assert len(sample.random_distractor.strip()) > 0
        assert len(sample.adversarial_distractor.strip()) > 0

        # 2. Strict similarity constraints
        assert sample.sim_random_query < 0.25, (
            f"Sample {sample.sample_id} random similarity {sample.sim_random_query} >= 0.25"
        )
        assert sample.sim_adversarial_query >= 0.65, (
            f"Sample {sample.sample_id} adversarial similarity {sample.sim_adversarial_query} < 0.65"
        )


def test_statistical_separation_mann_whitney():
    """Verify Mann-Whitney U test confirms statistical separation with p < 0.001."""
    dataset_path = Path("data/processed/eval_500.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    stats = compute_statistical_separation(samples)

    assert stats["sample_count"] == 500
    assert stats["statistically_distinct"] is True
    assert stats["p_value"] < 0.001
    assert stats["separation_margin"] > 0.35, (
        f"Separation margin ({stats['separation_margin']}) between min adv and max rand should be > 0.35"
    )

    # Percentiles validation
    assert stats["random"]["p90"] < 0.25
    assert stats["adversarial"]["p25"] >= 0.65


def test_entity_perturbation_conflicting_output():
    """Verify entity perturbation engine modifies entities and factual claims."""
    source_text = "The Apollo 11 mission landed on the Moon in 1969 with Neil Armstrong."
    perturbed = perturb_entities(source_text, answer="1969", seed=42)

    assert perturbed != source_text
    assert "1969" not in perturbed or "Neil Armstrong" not in perturbed
    assert len(perturbed) > 10


def test_legacy_distractor_sampler_compatibility():
    """Verify DistractorSampler retains backward compatibility with legacy API signatures."""
    corpus = [f"Legacy candidate passage {i}" for i in range(10)]
    sampler = DistractorSampler(corpus=corpus, seed=123)

    # Legacy sample_distractors(k=3, exclude=...)
    legacy_samples = sampler.sample_distractors(3, exclude=["Legacy candidate passage 0"])
    assert len(legacy_samples) == 3
    assert "Legacy candidate passage 0" not in legacy_samples

    # Legacy inject_noise(gold_context, num_distractors)
    mixed = sampler.inject_noise("True gold document", num_distractors=2)
    assert len(mixed) == 3
    assert "True gold document" in mixed


def test_cli_test_sampler_runner():
    """Verify run_sampler_test executes cleanly and returns exit code 0."""
    exit_code = run_sampler_test(device="cpu")
    assert exit_code == 0


def test_cli_test_sampler_flag():
    """Verify CLI entrypoint --test-sampler returns exit code 0."""
    exit_code = main(["--test-sampler"])
    assert exit_code == 0
