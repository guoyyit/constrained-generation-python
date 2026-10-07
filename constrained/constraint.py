"""Compiled constraints, per-request constraint state, and the compile cache.

A compiled constraint is immutable and safely shared across requests and
threads; every request gets its own :class:`ConstraintState`.  Repeated
identical definitions compile once and are served from a bounded LRU cache,
and token masks are memoized per NFA state-set so filtering stays bounded as
concurrency grows.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from constrained.errors import ConstraintError, FailureReason
from constrained.regex_nfa import NFA, epsilon_closure, parse_regex, step_char
from constrained.schema import schema_to_regex
from constrained.vocab import Vocab

KIND_REGEX = "regex"
KIND_JSON_SCHEMA = "json_schema"
_KINDS = frozenset({KIND_REGEX, KIND_JSON_SCHEMA})

#: Bound on memoized token masks per compiled constraint.
MAX_MASK_CACHE_ENTRIES = 4096


def _canonical_definition(kind: str, definition: Any) -> str:
    try:
        return json.dumps(
            {"kind": kind, "definition": definition},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ConstraintError(
            FailureReason.INVALID_DEFINITION, f"definition is not JSON-serializable: {exc}"
        ) from exc


def compile_constraint(kind: str, definition: Any) -> "CompiledConstraint":
    """Compile and fully validate a constraint definition.

    Raises :class:`ConstraintError` with a distinguishable reason for any
    invalid, unsupported, contradictory, unsatisfiable or too-complex
    definition — before any generation work starts.
    """
    if kind not in _KINDS:
        raise ConstraintError(
            FailureReason.INVALID_DEFINITION,
            f"unknown constraint kind {kind!r}; expected one of {sorted(_KINDS)}",
        )
    canonical = _canonical_definition(kind, definition)
    if kind == KIND_REGEX:
        if not isinstance(definition, str):
            raise ConstraintError(
                FailureReason.INVALID_DEFINITION, "regex definition must be a string"
            )
        pattern = definition
    else:
        if not isinstance(definition, (dict, bool)):
            raise ConstraintError(
                FailureReason.INVALID_DEFINITION,
                "json_schema definition must be an object",
            )
        pattern = schema_to_regex(definition)
    nfa = parse_regex(pattern)
    if nfa.accept not in nfa.live:
        raise ConstraintError(
            FailureReason.UNSATISFIABLE,
            "constraint cannot match any output (no path to an accepting state)",
        )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return CompiledConstraint(
        constraint_id=f"c_{digest[:24]}",
        kind=kind,
        definition_hash=digest,
        nfa=nfa,
    )


@dataclass(frozen=True)
class CompiledConstraint:
    """An immutable, shareable compiled constraint (cached by definition hash)."""

    constraint_id: str
    kind: str  # "regex" | "json_schema"
    definition_hash: str
    nfa: NFA
    _mask_cache: OrderedDict[frozenset[int], frozenset[int]] = field(
        default_factory=OrderedDict, compare=False, hash=False, repr=False
    )
    _mask_lock: threading.Lock = field(
        default_factory=threading.Lock, compare=False, hash=False, repr=False
    )

    def initial_state(self) -> "ConstraintState":
        states = epsilon_closure(self.nfa, {self.nfa.start}) & self.nfa.live
        return ConstraintState(self, states)

    def allowed_tokens(self, vocab: Vocab, states: frozenset[int]) -> frozenset[int]:
        """Token ids keeping at least one live path alive (EOS excluded)."""
        with self._mask_lock:
            cached = self._mask_cache.get(states)
            if cached is not None:
                self._mask_cache.move_to_end(states)
                return cached
        allowed = []
        for token_id, token in enumerate(vocab.tokens):
            if token_id == vocab.eos_id:
                continue
            if self._advance_states(states, token) is not None:
                allowed.append(token_id)
        result = frozenset(allowed)
        with self._mask_lock:
            self._mask_cache[states] = result
            self._mask_cache.move_to_end(states)
            while len(self._mask_cache) > MAX_MASK_CACHE_ENTRIES:
                self._mask_cache.popitem(last=False)
        return result

    def _advance_states(self, states: frozenset[int], token: str) -> frozenset[int] | None:
        nfa = self.nfa
        current = states
        for ch in token:
            nxt = epsilon_closure(nfa, step_char(nfa, current, ch)) & nfa.live
            if not nxt:
                return None
            current = nxt
        return current


class ConstraintState:
    """Immutable per-request position inside a compiled constraint."""

    __slots__ = ("_compiled", "_states")

    def __init__(self, compiled: CompiledConstraint, states: frozenset[int]) -> None:
        self._compiled = compiled
        self._states = states

    @property
    def compiled(self) -> CompiledConstraint:
        return self._compiled

    def allowed_tokens(self, vocab: Vocab) -> frozenset[int]:
        """Token ids that keep at least one live NFA path alive."""
        return self._compiled.allowed_tokens(vocab, self._states)

    def advance(self, token: str) -> "ConstraintState | None":
        """Return the next state after consuming ``token`` or None if dead."""
        nxt = self._compiled._advance_states(self._states, token)
        if nxt is None:
            return None
        return ConstraintState(self._compiled, nxt)

    def is_accepting(self) -> bool:
        return self._compiled.nfa.accept in self._states


class ConstraintCache:
    """Bounded LRU cache of compiled constraints keyed by definition hash."""

    def __init__(self, max_entries: int = 256) -> None:
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, CompiledConstraint] = OrderedDict()

    def get_or_compile(self, kind: str, definition: Any) -> CompiledConstraint:
        canonical = _canonical_definition(kind, definition)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        with self._lock:
            cached = self._entries.get(digest)
            if cached is not None:
                self._entries.move_to_end(digest)
                return cached
        # Compile outside the lock; concurrent duplicate compiles are
        # harmless (last insert wins) and keep rejection latency bounded.
        compiled = compile_constraint(kind, definition)
        with self._lock:
            self._entries[digest] = compiled
            self._entries.move_to_end(digest)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return compiled

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
