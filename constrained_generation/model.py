"""A deterministic toy "local model" producing logits over the vocabulary.

It stands in for a real local model so the constrained-generation engine can
be exercised end to end without shipping weights.  Logits are a pure function
of (seed, prefix, token), so generation is reproducible and tests are stable.
"""
from __future__ import annotations

import hashlib


class ToyModel:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def logits(self, prefix: str, candidates: list[str]) -> list[float]:
        """One logit per candidate token, deterministic in (seed, prefix)."""
        scores: list[float] = []
        for token in candidates:
            digest = hashlib.sha256(
                f"{self.seed}\t{prefix}\t{token}".encode("utf-8")
            ).digest()
            # Map the first 8 digest bytes into [0, 1).
            value = int.from_bytes(digest[:8], "big") / 2**64
            scores.append(value)
        return scores
