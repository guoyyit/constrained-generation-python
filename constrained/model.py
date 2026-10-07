"""Mock local model producing deterministic logits.

The interface mirrors a real local-inference model so it can be swapped out:
given the token context, return a logit per vocabulary token.  The mock is
deterministic per (seed, context) so tests are reproducible, and gives the
EOS token a fixed bonus so generation terminates as soon as the constraints
allow it — without a constraint, an unconstrained request would stop
immediately, which keeps the mock cheap.
"""

from __future__ import annotations

import hashlib

#: Logit bonus for EOS; larger than any hash-derived score so EOS is chosen
#: whenever the constraints permit it.
EOS_BONUS = 2.0


class MockModel:
    def __init__(self, seed: int = 0, eos_token_id: int = -1) -> None:
        self._seed = seed
        self._eos_token_id = eos_token_id

    def logits(self, context: tuple[int, ...], vocab_size: int) -> list[float]:
        # Hash the last few tokens only: scores stay deterministic while the
        # per-step work remains bounded regardless of context length.
        window = context[-8:]
        base = hashlib.sha256(
            self._seed.to_bytes(8, "big", signed=False)
            + len(window).to_bytes(2, "big")
            + b"".join(t.to_bytes(4, "big") for t in window)
        ).digest()
        scores: list[float] = []
        for token_id in range(vocab_size):
            digest = hashlib.sha256(base + token_id.to_bytes(4, "big")).digest()
            score = int.from_bytes(digest[:8], "big") / 2**64
            if token_id == self._eos_token_id:
                score += EOS_BONUS
            scores.append(score)
        return scores
