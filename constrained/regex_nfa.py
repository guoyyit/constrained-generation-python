"""Regex -> NFA compilation with dead-end (non-live) state elimination.

The NFA is simulated as a *set of states* so that, at any generation step,
we can compute exactly which next characters still allow the pattern to be
completed.  States from which no accepting state is reachable are pruned up
front, guaranteeing no "looks valid but can never close" path is taken.

Supported syntax: literals, ``.``, character classes (ranges, negation),
escapes (``\\d \\D \\w \\W \\s \\S \\n \\t \\r \\f \\v \\xHH \\uHHHH`` and
escaped punctuation), ``* + ? {m} {m,} {m,n}`` (lazy suffixes accepted and
ignored, since acceptance is existential), groups ``(...)`` / ``(?:...)``,
alternation ``|`` and anchors ``^`` ``$`` (no-ops: matching is always
full-string).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from constrained.errors import ConstraintError, FailureReason

#: Bounded-work limits: any pattern exceeding them is rejected up front.
MAX_PATTERN_LEN = 10_000
MAX_STATES = 100_000
MAX_REPEAT = 512

_DIGITS = frozenset("0123456789")
_WORD = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
_SPACE = frozenset(" \t\n\r\f\v")
_HEX = frozenset("0123456789abcdefABCDEF")


@dataclass(frozen=True)
class CharClass:
    """A set of characters, possibly negated (e.g. ``[^abc]``)."""

    chars: frozenset[str]
    negated: bool = False

    def matches(self, ch: str) -> bool:
        return (ch in self.chars) != self.negated


@dataclass(frozen=True)
class NFA:
    """Thompson NFA with precomputed liveness information."""

    n_states: int
    start: int
    accept: int
    #: state -> tuple of (char class or None for epsilon, target state)
    transitions: dict[int, tuple[tuple[CharClass | None, int], ...]] = field(
        default_factory=dict
    )
    #: states from which the accept state is reachable
    live: frozenset[int] = frozenset()


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Lit:
    cc: CharClass


@dataclass(frozen=True)
class _Cat:
    parts: tuple


@dataclass(frozen=True)
class _Alt:
    options: tuple


@dataclass(frozen=True)
class _Rep:
    node: object
    lo: int
    hi: int | None


@dataclass(frozen=True)
class _Empty:
    pass


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class _Parser:
    def __init__(self, pattern: str) -> None:
        self.s = pattern
        self.i = 0

    def error(self, msg: str) -> None:
        raise ConstraintError(
            FailureReason.INVALID_DEFINITION,
            f"invalid regex: {msg} (position {self.i} of {self.s!r})",
        )

    def _peek(self) -> str | None:
        return self.s[self.i] if self.i < len(self.s) else None

    def parse(self):
        node = self._parse_alt()
        if self.i != len(self.s):
            self.error(f"unexpected {self.s[self.i]!r}")
        return node

    def _parse_alt(self):
        options = [self._parse_cat()]
        while self._peek() == "|":
            self.i += 1
            options.append(self._parse_cat())
        return options[0] if len(options) == 1 else _Alt(tuple(options))

    def _parse_cat(self):
        parts = []
        while self.i < len(self.s) and self._peek() not in "|)":
            parts.append(self._parse_rep())
        if not parts:
            return _Empty()
        return parts[0] if len(parts) == 1 else _Cat(tuple(parts))

    def _parse_rep(self):
        node = self._parse_atom()
        while True:
            c = self._peek()
            if c == "*":
                self.i += 1
                node = _Rep(node, 0, None)
            elif c == "+":
                self.i += 1
                node = _Rep(node, 1, None)
            elif c == "?":
                self.i += 1
                node = _Rep(node, 0, 1)
            elif c == "{":
                parsed = self._try_parse_braces()
                if parsed is None:
                    break
                lo, hi = parsed
                node = _Rep(node, lo, hi)
            else:
                break
            if self._peek() == "?":  # lazy suffix: irrelevant for fullmatch
                self.i += 1
        return node

    def _try_parse_braces(self) -> tuple[int, int | None] | None:
        s, i = self.s, self.i
        j = i + 1
        k = j
        while k < len(s) and s[k].isdigit():
            k += 1
        lo: int | None = int(s[j:k]) if k > j else None
        hi: int | None = None
        if k < len(s) and s[k] == ",":
            k += 1
            j2 = k
            while k < len(s) and s[k].isdigit():
                k += 1
            hi = int(s[j2:k]) if k > j2 else None
            if lo is None:
                lo = 0
        else:
            hi = lo
        if k >= len(s) or s[k] != "}" or lo is None:
            return None
        if hi is not None and hi < lo:
            self.error(f"quantifier min {lo} greater than max {hi}")
        if lo > MAX_REPEAT or (hi is not None and hi > MAX_REPEAT):
            raise ConstraintError(
                FailureReason.TOO_COMPLEX,
                f"repetition count exceeds limit of {MAX_REPEAT}",
            )
        self.i = k + 1
        return lo, hi

    def _parse_atom(self):
        c = self._peek()
        if c is None:
            self.error("unexpected end of pattern")
        if c == "(":
            self.i += 1
            if self.s[self.i : self.i + 2] == "?:":
                self.i += 2
            elif self.s[self.i : self.i + 1] == "?":
                raise ConstraintError(
                    FailureReason.UNSUPPORTED,
                    "only capturing groups (...) and (?:...) are supported",
                )
            node = self._parse_alt()
            if self._peek() != ")":
                self.error("unbalanced parenthesis")
            self.i += 1
            return node
        if c == "[":
            return _Lit(self._parse_class())
        if c == ".":
            self.i += 1
            return _Lit(CharClass(frozenset({"\n"}), negated=True))
        if c in "^$":
            self.i += 1
            return _Empty()
        if c == "\\":
            self.i += 1
            return _Lit(self._parse_escape())
        if c in "*+?":
            self.error(f"nothing to repeat before {c!r}")
        if c == ")":
            self.error("unbalanced parenthesis")
        self.i += 1
        return _Lit(CharClass(frozenset(c)))

    def _parse_escape(self) -> CharClass:
        if self.i >= len(self.s):
            self.error("trailing backslash")
        c = self.s[self.i]
        self.i += 1
        simple = {"n": "\n", "t": "\t", "r": "\r", "f": "\f", "v": "\v", "0": "\0"}
        if c in simple:
            return CharClass(frozenset(simple[c]))
        if c == "d":
            return CharClass(_DIGITS)
        if c == "D":
            return CharClass(_DIGITS, negated=True)
        if c == "w":
            return CharClass(_WORD)
        if c == "W":
            return CharClass(_WORD, negated=True)
        if c == "s":
            return CharClass(_SPACE)
        if c == "S":
            return CharClass(_SPACE, negated=True)
        if c == "x":
            return CharClass(frozenset(self._parse_hex(2)))
        if c == "u":
            return CharClass(frozenset(self._parse_hex(4)))
        if c.isalnum():
            self.error(f"unknown escape \\{c}")
        return CharClass(frozenset(c))

    def _parse_hex(self, n: int) -> str:
        h = self.s[self.i : self.i + n]
        if len(h) != n or any(ch not in _HEX for ch in h):
            self.error(f"invalid hex escape of length {n}")
        self.i += n
        return chr(int(h, 16))

    def _parse_class_item(self) -> frozenset[str]:
        c = self.s[self.i]
        if c == "\\":
            self.i += 1
            cc = self._parse_escape()
            if cc.negated:
                raise ConstraintError(
                    FailureReason.UNSUPPORTED,
                    "negated escapes such as \\D \\W \\S are not supported inside [...]",
                )
            return cc.chars
        self.i += 1
        return frozenset(c)

    def _parse_class(self) -> CharClass:
        # self.s[self.i] == "["
        self.i += 1
        negated = False
        if self._peek() == "^":
            negated = True
            self.i += 1
        chars: set[str] = set()
        first = True
        while True:
            if self.i >= len(self.s):
                self.error("unterminated character class")
            if self.s[self.i] == "]" and not first:
                self.i += 1
                break
            first = False
            item = self._parse_class_item()
            if (
                self._peek() == "-"
                and self.i + 1 < len(self.s)
                and self.s[self.i + 1] != "]"
            ):
                self.i += 1  # consume '-'
                hi_item = self._parse_class_item()
                if len(item) != 1 or len(hi_item) != 1:
                    self.error("range endpoints must be single characters")
                lo, hi = next(iter(item)), next(iter(hi_item))
                if ord(hi) < ord(lo):
                    self.error(f"bad range {lo!r}-{hi!r}")
                chars.update(chr(o) for o in range(ord(lo), ord(hi) + 1))
            else:
                chars |= item
        return CharClass(frozenset(chars), negated)


# ---------------------------------------------------------------------------
# Thompson construction
# ---------------------------------------------------------------------------


class _Builder:
    def __init__(self) -> None:
        self.transitions: dict[int, list[tuple[CharClass | None, int]]] = {}
        self.n = 0

    def new_state(self) -> int:
        s = self.n
        self.n += 1
        if self.n > MAX_STATES:
            raise ConstraintError(
                FailureReason.TOO_COMPLEX,
                f"pattern requires more than {MAX_STATES} NFA states",
            )
        return s

    def add(self, src: int, cc: CharClass | None, dst: int) -> None:
        self.transitions.setdefault(src, []).append((cc, dst))

    def build(self, node) -> tuple[int, int]:
        if isinstance(node, _Empty):
            s = self.new_state()
            e = self.new_state()
            self.add(s, None, e)
            return s, e
        if isinstance(node, _Lit):
            s = self.new_state()
            e = self.new_state()
            self.add(s, node.cc, e)
            return s, e
        if isinstance(node, _Cat):
            start = end = None
            for part in node.parts:
                ps, pe = self.build(part)
                if start is None:
                    start, end = ps, pe
                else:
                    self.add(end, None, ps)
                    end = pe
            return start, end
        if isinstance(node, _Alt):
            s = self.new_state()
            e = self.new_state()
            for opt in node.options:
                os_, oe = self.build(opt)
                self.add(s, None, os_)
                self.add(oe, None, e)
            return s, e
        if isinstance(node, _Rep):
            return self._build_rep(node)
        raise AssertionError(f"unknown node {node!r}")

    def _build_rep(self, node: _Rep) -> tuple[int, int]:
        lo, hi = node.lo, node.hi
        s = self.new_state()
        e = self.new_state()
        prev_end = s
        for _ in range(lo):
            cs, ce = self.build(node.node)
            self.add(prev_end, None, cs)
            prev_end = ce
        if hi is None:
            loop = self.new_state()
            self.add(prev_end, None, loop)
            cs, ce = self.build(node.node)
            self.add(loop, None, cs)
            self.add(ce, None, loop)
            self.add(loop, None, e)
        else:
            for _ in range(hi - lo):
                cs, ce = self.build(node.node)
                self.add(prev_end, None, cs)
                self.add(prev_end, None, e)  # skip the remaining optional copies
                prev_end = ce
            self.add(prev_end, None, e)
        return s, e


def _compute_live(
    transitions: dict[int, tuple[tuple[CharClass | None, int], ...]], accept: int
) -> frozenset[int]:
    rev: dict[int, list[int]] = {}
    for src, trans in transitions.items():
        for _, dst in trans:
            rev.setdefault(dst, []).append(src)
    live = {accept}
    stack = [accept]
    while stack:
        s = stack.pop()
        for p in rev.get(s, ()):
            if p not in live:
                live.add(p)
                stack.append(p)
    return frozenset(live)


def parse_regex(pattern: str) -> NFA:
    """Compile a regex pattern into an NFA with liveness computed."""
    if not isinstance(pattern, str):
        raise ConstraintError(FailureReason.INVALID_DEFINITION, "pattern must be a string")
    if len(pattern) > MAX_PATTERN_LEN:
        raise ConstraintError(
            FailureReason.TOO_COMPLEX,
            f"pattern length exceeds limit of {MAX_PATTERN_LEN}",
        )
    ast = _Parser(pattern).parse()
    builder = _Builder()
    frag_start, accept = builder.build(ast)
    start = builder.new_state()
    builder.add(start, None, frag_start)
    transitions = {s: tuple(t) for s, t in builder.transitions.items()}
    live = _compute_live(transitions, accept)
    return NFA(
        n_states=builder.n,
        start=start,
        accept=accept,
        transitions=transitions,
        live=live,
    )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def epsilon_closure(nfa: NFA, states: Iterable[int]) -> frozenset[int]:
    result = set(states)
    stack = list(states)
    while stack:
        s = stack.pop()
        for cc, dst in nfa.transitions.get(s, ()):
            if cc is None and dst not in result:
                result.add(dst)
                stack.append(dst)
    return frozenset(result)


def step_char(nfa: NFA, states: Iterable[int], ch: str) -> frozenset[int]:
    out: set[int] = set()
    for s in states:
        for cc, dst in nfa.transitions.get(s, ()):
            if cc is not None and cc.matches(ch):
                out.add(dst)
    return frozenset(out)
