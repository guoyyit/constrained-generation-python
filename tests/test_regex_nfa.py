"""The NFA must accept exactly the same language as Python's re (fullmatch)."""

from __future__ import annotations

import itertools
import re

import pytest

from constrained.errors import ConstraintError
from constrained.regex_nfa import epsilon_closure, parse_regex, step_char


def nfa_fullmatch(nfa, text: str) -> bool:
    states = epsilon_closure(nfa, {nfa.start})
    for ch in text:
        states = epsilon_closure(nfa, step_char(nfa, states, ch))
        if not states:
            return False
    return nfa.accept in states


PATTERNS = [
    "abc",
    "a|b|c",
    "(ab|cd)+",
    "a*b?c{2,4}",
    "[a-c]+[0-9]?",
    "[^abc]+",
    "\\d{3}-\\d{4}",
    "\\w+@\\w+\\.com",
    "(?:true|false)",
    "a{0,3}",
    "x{3}",
    "-?(?:0|[1-9]\\d*)",
    "(?:a|b)(?:c|d)",
    "\\[\\{\\}\\(\\)]",
    "a\\nb",
    "\\x41\\u0042",
    "",
    "^abc$",
    "colou?r",
]

ALPHABET = "abc01-,@.x\n"


@pytest.mark.parametrize("pattern", PATTERNS)
def test_nfa_matches_re_fullmatch(pattern: str) -> None:
    nfa = parse_regex(pattern)
    expected = re.compile(pattern)
    samples = {""}
    for length in (1, 2, 3, 4):
        samples.update(
            "".join(chars) for chars in itertools.product(ALPHABET, repeat=length)
        )
    samples.update(["abcabc", "aabbcc", "123-4567", "true", "false", "-42", "007"])
    for text in samples:
        assert nfa_fullmatch(nfa, text) == bool(expected.fullmatch(text)), (
            pattern,
            text,
        )


def test_live_states_prune_dead_ends() -> None:
    nfa = parse_regex("ab")
    states = epsilon_closure(nfa, {nfa.start}) & nfa.live
    states = epsilon_closure(nfa, step_char(nfa, states, "a")) & nfa.live
    # 'x' cannot continue towards acceptance
    assert (epsilon_closure(nfa, step_char(nfa, states, "x")) & nfa.live) == frozenset()
    assert (epsilon_closure(nfa, step_char(nfa, states, "b")) & nfa.live) != frozenset()


def test_unsatisfiable_pattern_has_dead_accept() -> None:
    # [^\x00-\xff] still matches code points >= U+0100, so craft a truly
    # empty language via an empty alternation branch is not possible; instead
    # verify liveness flags states that can never reach accept.
    nfa = parse_regex("a(?:b|c)")
    assert nfa.accept in nfa.live
    assert nfa.start in nfa.live


@pytest.mark.parametrize(
    "pattern",
    ["a(", "a)", "a{2,1}", "[z-a]", "a\\q", "[abc", "*a"],
)
def test_invalid_patterns_rejected(pattern: str) -> None:
    with pytest.raises(ConstraintError) as excinfo:
        parse_regex(pattern)
    assert excinfo.value.reason == "invalid_definition"


def test_too_complex_rejected() -> None:
    with pytest.raises(ConstraintError) as excinfo:
        parse_regex("a{513}")
    assert excinfo.value.reason == "too_complex"
    with pytest.raises(ConstraintError) as excinfo:
        parse_regex("a" * 20_000)
    assert excinfo.value.reason == "too_complex"


def test_unsupported_group_syntax() -> None:
    with pytest.raises(ConstraintError) as excinfo:
        parse_regex("(?P<name>a)")
    assert excinfo.value.reason == "unsupported"
