"""Token vocabulary shared by the toy local model and the automata.

Tokens are single characters; this keeps the candidate set explicit and makes
masking semantics easy to reason about.  The alphabet is the printable ASCII
band, which covers every JSON construct the schema subset can emit.
"""
from __future__ import annotations

import string

#: Sentinel returned by the engine when the automaton sits in an accept state
#: and the model chooses to stop.  Never part of the real alphabet.
STOP: str = "<stop>"

#: Every token a constraint automaton may ever consume.
ALPHABET: frozenset[str] = frozenset(string.printable[:95])  # 0x20 .. 0x7e

_TOKENS: list[str] = sorted(ALPHABET)


def tokens() -> list[str]:
    """Return the sorted candidate-token list (without the STOP sentinel)."""
    return list(_TOKENS)
