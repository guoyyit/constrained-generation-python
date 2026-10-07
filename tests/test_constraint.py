"""Compiled constraints: token filtering, caching, bounded recomputation."""

from __future__ import annotations

import pytest

from constrained.constraint import ConstraintCache, compile_constraint
from constrained.errors import ConstraintError
from constrained.vocab import default_vocab

VOCAB = default_vocab()


def token_id(ch: str) -> int:
    return VOCAB.tokens.index(ch)


def test_allowed_tokens_track_live_paths() -> None:
    compiled = compile_constraint("regex", "a(b|c)*d")
    state = compiled.initial_state()
    assert state.allowed_tokens(VOCAB) == frozenset({token_id("a")})
    state = state.advance("a")
    assert state is not None
    assert state.allowed_tokens(VOCAB) == frozenset(
        {token_id("b"), token_id("c"), token_id("d")}
    )
    assert not state.is_accepting()
    state = state.advance("b")
    state = state.advance("c")
    state = state.advance("d")
    assert state.is_accepting()
    # EOS is never part of the NFA mask; the engine adds it when accepting
    assert VOCAB.eos_id not in state.allowed_tokens(VOCAB)


def test_advance_returns_none_on_dead_token() -> None:
    compiled = compile_constraint("regex", "ab")
    state = compiled.initial_state()
    assert state.advance("x") is None


def test_token_masks_are_memoized() -> None:
    compiled = compile_constraint("regex", "(ab|cd)+")
    s1 = compiled.initial_state()
    s2 = compiled.initial_state()
    assert s1.allowed_tokens(VOCAB) is s2.allowed_tokens(VOCAB)


def test_cache_deduplicates_identical_definitions() -> None:
    cache = ConstraintCache()
    a = cache.get_or_compile("json_schema", {"type": "integer", "minimum": 1})
    b = cache.get_or_compile("json_schema", {"minimum": 1, "type": "integer"})
    c = cache.get_or_compile("json_schema", {"type": "integer", "minimum": 2})
    assert a is b  # key order does not matter
    assert a is not c
    assert len(cache) == 2


def test_cache_eviction_is_bounded() -> None:
    cache = ConstraintCache(max_entries=4)
    for i in range(10):
        cache.get_or_compile("regex", f"a{{{i}}}")
    assert len(cache) == 4


def test_failed_definitions_are_not_cached() -> None:
    cache = ConstraintCache()
    for _ in range(2):
        with pytest.raises(ConstraintError):
            cache.get_or_compile("regex", "a(")
    assert len(cache) == 0


def test_unsatisfiable_regex_rejected_before_generation() -> None:
    # boolean schema false -> unsatisfiable
    with pytest.raises(ConstraintError) as excinfo:
        compile_constraint("json_schema", False)
    assert excinfo.value.reason == "unsatisfiable"
