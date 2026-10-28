"""Data utilities, distractor sampling, and dataset construction."""

from src.data.distractor_sampler import (
    BenchmarkSample,
    DistractorSampler,
    build_evaluation_dataset,
    compute_statistical_separation,
    fetch_raw_squad_or_seed,
    perturb_entities,
)

__all__ = [
    "DistractorSampler",
    "BenchmarkSample",
    "build_evaluation_dataset",
    "compute_statistical_separation",
    "fetch_raw_squad_or_seed",
    "perturb_entities",
]
