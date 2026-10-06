"""Regular-language machinery over single-character tokens.

A constraint definition compiles down to a :class:`Regex` AST, which is then
compiled to a trimmed DFA.  "Trimmed" means every remaining state can still
reach an accepting state, so any token the DFA allows can always be extended
into a complete, valid output -- no dead-end paths.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable


# ---------------------------------------------------------------------------
# Regex AST
# ---------------------------------------------------------------------------

class Regex:
    """Base class of the regex AST."""


@dataclass(frozen=True)
class Empty(Regex):
    """Matches only the empty string."""


@dataclass(frozen=True)
class Never(Regex):
    """Matches nothing (empty language)."""


@dataclass(frozen=True)
class Chars(Regex):
    """Matches exactly one character from ``chars``."""

    chars: frozenset[str]


@dataclass(frozen=True)
class Seq(Regex):
    """Concatenation."""

    parts: tuple[Regex, ...]


@dataclass(frozen=True)
class Alt(Regex):
    """Union."""

    options: tuple[Regex, ...]


@dataclass(frozen=True)
class Star(Regex):
    """Kleene star."""

    inner: Regex


@dataclass(frozen=True)
class Plus(Regex):
    """One or more repetitions."""

    inner: Regex


@dataclass(frozen=True)
class Opt(Regex):
    """Zero or one occurrence."""

    inner: Regex


def literal(text: str) -> Regex:
    """Build a regex matching ``text`` exactly."""
    if not text:
        return Empty()
    return Seq(tuple(Chars(frozenset(ch)) for ch in text))


# ---------------------------------------------------------------------------
# DFA
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DFA:
    """A trimmed deterministic automaton over single-character tokens.

    Invariants (enforced at construction time):
      * every state in ``states`` can reach an accepting state;
      * ``transitions`` therefore only ever leads to co-reachable states.
    """

    states: frozenset[int]
    start: int
    accepts: frozenset[int]
    transitions: dict[int, dict[str, int]]
    alphabet: frozenset[str]

    def allowed_tokens(self, state: int) -> frozenset[str]:
        """Tokens that may follow ``state`` while keeping completion possible."""
        if state not in self.states:
            return frozenset()
        return frozenset(self.transitions.get(state, ()))

    def step(self, state: int, token: str) -> int | None:
        """Advance the automaton, or ``None`` if ``token`` is not allowed."""
        return self.transitions.get(state, {}).get(token)

    def is_accept(self, state: int) -> bool:
        return state in self.accepts

    def is_empty(self) -> bool:
        """Whether the recognised language is empty."""
        return self.start not in self.states

    def accepts_text(self, text: str) -> bool:
        """Membership test, mainly for tests and validation cross-checks."""
        state = self.start
        if self.is_empty():
            return False
        for ch in text:
            nxt = self.step(state, ch)
            if nxt is None:
                return False
            state = nxt
        return self.is_accept(state)


def _empty_dfa(alphabet: frozenset[str]) -> DFA:
    return DFA(
        states=frozenset(),
        start=-1,
        accepts=frozenset(),
        transitions={},
        alphabet=alphabet,
    )


# ---------------------------------------------------------------------------
# Thompson NFA
# ---------------------------------------------------------------------------

class _NFA:
    def __init__(self) -> None:
        self.transitions: dict[int, dict[str, set[int]]] = {}
        self.epsilon: dict[int, set[int]] = {}
        self._next = 0

    def new_state(self) -> int:
        state = self._next
        self._next += 1
        return state

    def add(self, src: int, ch: str, dst: int) -> None:
        self.transitions.setdefault(src, {}).setdefault(ch, set()).add(dst)

    def add_eps(self, src: int, dst: int) -> None:
        self.epsilon.setdefault(src, set()).add(dst)

    def build(self, regex: Regex) -> tuple[int, int]:
        """Return (start, accept) fragment for ``regex``."""
        if isinstance(regex, Never):
            start, accept = self.new_state(), self.new_state()
            return start, accept  # no path between them
        if isinstance(regex, Empty):
            start, accept = self.new_state(), self.new_state()
            self.add_eps(start, accept)
            return start, accept
        if isinstance(regex, Chars):
            start, accept = self.new_state(), self.new_state()
            for ch in regex.chars:
                self.add(start, ch, accept)
            return start, accept
        if isinstance(regex, Seq):
            start, accept = self.new_state(), self.new_state()
            cursor = start
            for part in regex.parts:
                p_start, p_accept = self.build(part)
                self.add_eps(cursor, p_start)
                cursor = p_accept
            self.add_eps(cursor, accept)
            return start, accept
        if isinstance(regex, Alt):
            start, accept = self.new_state(), self.new_state()
            for option in regex.options:
                o_start, o_accept = self.build(option)
                self.add_eps(start, o_start)
                self.add_eps(o_accept, accept)
            return start, accept
        if isinstance(regex, Star):
            start, accept = self.new_state(), self.new_state()
            i_start, i_accept = self.build(regex.inner)
            self.add_eps(start, accept)
            self.add_eps(start, i_start)
            self.add_eps(i_accept, i_start)
            self.add_eps(i_accept, accept)
            return start, accept
        if isinstance(regex, Plus):
            i_start, i_accept = self.build(regex.inner)
            accept = self.new_state()
            self.add_eps(i_accept, i_start)
            self.add_eps(i_accept, accept)
            return i_start, accept
        if isinstance(regex, Opt):
            start, accept = self.new_state(), self.new_state()
            i_start, i_accept = self.build(regex.inner)
            self.add_eps(start, accept)
            self.add_eps(start, i_start)
            self.add_eps(i_accept, accept)
            return start, accept
        raise TypeError(f"unknown regex node: {type(regex)!r}")

    def closure(self, states: Iterable[int]) -> frozenset[int]:
        seen = set(states)
        stack = list(seen)
        while stack:
            state = stack.pop()
            for nxt in self.epsilon.get(state, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return frozenset(seen)


def _trim(
    states: Iterable[int],
    start: int,
    accepts: Iterable[int],
    transitions: dict[int, dict[str, int]],
    alphabet: frozenset[str],
) -> DFA:
    """Drop every state that cannot reach an accept state."""
    accepts = frozenset(a for a in accepts if a in states)
    reverse: dict[int, set[int]] = {}
    for src, edges in transitions.items():
        for dst in edges.values():
            reverse.setdefault(dst, set()).add(src)
    co_reachable = set(accepts)
    queue = deque(co_reachable)
    while queue:
        state = queue.popleft()
        for prev in reverse.get(state, ()):
            if prev not in co_reachable:
                co_reachable.add(prev)
                queue.append(prev)
    kept = frozenset(s for s in states if s in co_reachable)
    if start not in kept:
        return _empty_dfa(alphabet)
    trimmed_transitions = {
        s: {ch: dst for ch, dst in transitions.get(s, {}).items() if dst in kept}
        for s in kept
    }
    return DFA(
        states=kept,
        start=start,
        accepts=accepts & kept,
        transitions=trimmed_transitions,
        alphabet=alphabet,
    )


def compile_regex(regex: Regex, alphabet: Iterable[str]) -> DFA:
    """Compile ``regex`` into a trimmed DFA over ``alphabet``."""
    alphabet = frozenset(alphabet)
    nfa = _NFA()
    start, accept = nfa.build(regex)

    start_set = nfa.closure({start})
    subset_ids: dict[frozenset[int], int] = {start_set: 0}
    dfa_states: set[int] = {0}
    dfa_accepts: set[int] = set()
    dfa_transitions: dict[int, dict[str, int]] = {}
    if accept in start_set:
        dfa_accepts.add(0)
    queue = deque([start_set])
    while queue:
        subset = queue.popleft()
        src_id = subset_ids[subset]
        for ch in alphabet:
            moved = {
                dst
                for s in subset
                for dst in nfa.transitions.get(s, {}).get(ch, set())
            }
            if not moved:
                continue
            target = nfa.closure(moved)
            if target not in subset_ids:
                subset_ids[target] = len(subset_ids)
                dfa_states.add(subset_ids[target])
                if accept in target:
                    dfa_accepts.add(subset_ids[target])
                queue.append(target)
            dfa_transitions.setdefault(src_id, {})[ch] = subset_ids[target]
    return _trim(dfa_states, 0, dfa_accepts, dfa_transitions, alphabet)


def product(dfas: list[DFA]) -> DFA:
    """Intersection of ``dfas`` as a trimmed product DFA.

    The result is empty iff the constraints are mutually contradictory.
    """
    if not dfas:
        raise ValueError("product requires at least one DFA")
    alphabet = dfas[0].alphabet
    if any(d.is_empty() for d in dfas):
        return _empty_dfa(alphabet)

    start = tuple(d.start for d in dfas)
    state_ids: dict[tuple[int, ...], int] = {start: 0}
    transitions: dict[int, dict[str, int]] = {}
    accepts: set[int] = set()
    if all(d.is_accept(s) for d, s in zip(dfas, start)):
        accepts.add(0)
    queue = deque([start])
    while queue:
        combo = queue.popleft()
        src_id = state_ids[combo]
        for ch in alphabet:
            nxt: list[int] = []
            for dfa, state in zip(dfas, combo):
                target = dfa.step(state, ch)
                if target is None:
                    break
                nxt.append(target)
            else:
                nxt_combo = tuple(nxt)
                if nxt_combo not in state_ids:
                    state_ids[nxt_combo] = len(state_ids)
                    if all(d.is_accept(s) for d, s in zip(dfas, nxt_combo)):
                        accepts.add(state_ids[nxt_combo])
                    queue.append(nxt_combo)
                transitions.setdefault(src_id, {})[ch] = state_ids[nxt_combo]
    return _trim(set(state_ids.values()), 0, accepts, transitions, alphabet)
