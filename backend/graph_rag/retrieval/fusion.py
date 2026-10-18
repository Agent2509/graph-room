"""Reciprocal rank fusion. Fifteen lines instead of a reranker dependency."""

from collections import defaultdict


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Fuse ranked id lists into one ranking. Higher score is better."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] += 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
