"""Token vocabulary used by the local generator."""

from __future__ import annotations

import string
from dataclasses import dataclass

EOS_TOKEN = "<eos>"


@dataclass(frozen=True)
class Vocab:
    tokens: tuple[str, ...]
    eos_id: int

    @property
    def size(self) -> int:
        return len(self.tokens)


def default_vocab() -> Vocab:
    """Character-level vocabulary over printable ASCII plus an EOS token."""
    tokens = tuple(sorted(string.printable[:95])) + (EOS_TOKEN,)
    return Vocab(tokens=tokens, eos_id=len(tokens) - 1)
