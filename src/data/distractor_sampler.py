"""Distractor sampling module for noisy context injection in edge RAG evaluation."""

from __future__ import annotations

import random
from typing import List, Optional, Sequence


class DistractorSampler:
    """Samples distractor contexts and mixes them with gold grounding documents.

    Used to benchmark edge foundation model resilience under increasing
    noise ratios (signal-to-noise ratio in context).
    """

    def __init__(
        self,
        corpus: Optional[Sequence[str]] = None,
        seed: int = 42,
    ) -> None:
        """Initialize the distractor sampler.

        Args:
            corpus: Pool of distractor passages. If None, an empty pool is initialized.
            seed: Random seed for deterministic context injection.
        """
        self.corpus = list(corpus) if corpus is not None else []
        self.rng = random.Random(seed)

    def add_passages(self, passages: Sequence[str]) -> None:
        """Add new distractor passages to the pool.

        Args:
            passages: Sequence of text passages.
        """
        self.corpus.extend(passages)

    def sample_distractors(self, k: int, exclude: Optional[Sequence[str]] = None) -> List[str]:
        """Sample k distractor passages from the corpus excluding specific passages.

        Args:
            k: Number of distractors to draw.
            exclude: Optional passages to avoid sampling (e.g., gold documents).

        Returns:
            List[str]: Sampled distractor texts.
        """
        excluded_set = set(exclude) if exclude is not None else set()
        candidates = [p for p in self.corpus if p not in excluded_set]

        if not candidates:
            return []

        if len(candidates) < k:
            return self.rng.choices(candidates, k=k)
        return self.rng.sample(candidates, k=k)

    def inject_noise(
        self,
        gold_context: str,
        num_distractors: int,
        shuffle: bool = True,
    ) -> List[str]:
        """Mix gold context with distractor passages.

        Args:
            gold_context: The true relevant document context.
            num_distractors: Number of noisy distractor passages to inject.
            shuffle: Whether to randomly position the gold passage among distractors.

        Returns:
            List[str]: Ordered context passages containing gold and distractors.
        """
        distractors = self.sample_distractors(num_distractors, exclude=[gold_context])
        mixed = [gold_context] + distractors
        if shuffle:
            self.rng.shuffle(mixed)
        return mixed
