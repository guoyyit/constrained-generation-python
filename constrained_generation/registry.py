"""Cache of compiled constraints.

Compilation (regex -> trimmed DFA) is the expensive step; identical structure
definitions must not be recompiled per request.  The registry is an
immutable-value LRU cache keyed by the canonical definition key, safe to share
across concurrent requests because compiled automata are immutable.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Callable

from .regex import DFA


class ConstraintRegistry:
    def __init__(self, maxsize: int = 256) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be positive")
        self.maxsize = maxsize
        self._cache: OrderedDict[str, DFA] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_compile(self, canonical_key: str, compile_fn: Callable[[], DFA]) -> DFA:
        """Return the cached DFA for ``canonical_key`` or compile and cache it."""
        with self._lock:
            hit = self._cache.get(canonical_key)
            if hit is not None:
                self._cache.move_to_end(canonical_key)
                return hit
        # Compile outside the lock: compilation is pure and idempotent, and
        # holding the lock would serialise unrelated requests.
        compiled = compile_fn()
        with self._lock:
            existing = self._cache.get(canonical_key)
            if existing is not None:
                self._cache.move_to_end(canonical_key)
                return existing
            self._cache[canonical_key] = compiled
            self._cache.move_to_end(canonical_key)
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
        return compiled

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
