"""Automaton invariants: trimming guarantees no dead-end paths."""
from collections import deque

from constrained_generation.regex import (
    Alt,
    Chars,
    Star,
    Seq,
    compile_regex,
    literal,
    product,
)
from constrained_generation.vocab import ALPHABET


def _assert_fully_co_reachable(dfa):
    """Every state must be able to reach an accept state (no dead ends)."""
    reverse = {}
    for src, edges in dfa.transitions.items():
        for dst in edges.values():
            reverse.setdefault(dst, set()).add(src)
    seen = set(dfa.accepts)
    queue = deque(seen)
    while queue:
        state = queue.popleft()
        for prev in reverse.get(state, ()):
            if prev not in seen:
                seen.add(prev)
                queue.append(prev)
    assert dfa.states <= seen


def test_every_allowed_token_stays_completable():
    # "ab" | "ac" | "b": after 'a', only {b, c} may follow; after 'b' (from
    # start) we are already at an accept state.
    dfa = compile_regex(Alt((literal("ab"), literal("ac"), literal("b"))), ALPHABET)
    _assert_fully_co_reachable(dfa)
    assert dfa.allowed_tokens(dfa.start) == frozenset({"a", "b"})
    state = dfa.step(dfa.start, "a")
    assert dfa.allowed_tokens(state) == frozenset({"b", "c"})
    assert not dfa.is_accept(state)


def test_trimmed_states_all_reach_accept():
    dfa = compile_regex(
        Seq((literal('"'), Star(Chars(frozenset("ab"))), literal('"'))), ALPHABET
    )
    _assert_fully_co_reachable(dfa)


def test_product_intersection_semantics():
    a = compile_regex(Alt((literal("ax"), literal("ay"))), ALPHABET)
    b = compile_regex(Alt((literal("ay"), literal("az"))), ALPHABET)
    combined = product([a, b])
    assert combined.accepts_text("ay")
    assert not combined.accepts_text("ax") and not combined.accepts_text("az")
    _assert_fully_co_reachable(combined)


def test_product_of_contradictory_constraints_is_empty():
    a = compile_regex(literal('{"a":1}'), ALPHABET)
    b = compile_regex(literal('{"a":2}'), ALPHABET)
    assert product([a, b]).is_empty()


def test_product_prunes_mid_path_dead_ends():
    # A = {ax, by}, B = {ay, bx}: token-level intersection would allow 'a'
    # then get stuck; the trimmed product must be empty instead.
    a = compile_regex(Alt((literal("ax"), literal("by"))), ALPHABET)
    b = compile_regex(Alt((literal("ay"), literal("bx"))), ALPHABET)
    assert product([a, b]).is_empty()
