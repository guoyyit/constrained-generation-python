"""Output-structure definitions: validation and compilation to regex.

Supported definition kinds:
  * ``json_schema`` -- a deliberately small JSON Schema subset;
  * ``regex``       -- a small regular-expression dialect, matched against the
                       whole output.

Rejection classes (all raised before generation starts):
  * ConstraintInvalidError       -- malformed definition;
  * ConstraintUnsupportedError   -- valid but outside the supported subset;
  * ConstraintUnsatisfiableError -- well-formed, yet no output can satisfy it.

Generated JSON uses compact separators (``{"a":1}``); object properties are
emitted in declaration order.  Both are documented subset semantics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .errors import (
    ConstraintInvalidError,
    ConstraintUnsatisfiableError,
    ConstraintUnsupportedError,
)
from .regex import Alt, Chars, Empty, Opt, Regex, Seq, Star, literal
from .vocab import ALPHABET

MAX_OBJECT_PROPERTIES = 8
MAX_ARRAY_ITEMS = 16
MAX_STRING_LENGTH = 64
MAX_REPEAT = 64

SUPPORTED_KINDS = ("json_schema", "regex")

_JSON_TYPES = ("object", "array", "string", "integer", "number", "boolean", "null")
_COMMON_KEYS = {"type", "enum", "const"}
_TYPE_KEYS = {
    "object": {"properties", "required", "additionalProperties"},
    "array": {"items", "minItems", "maxItems"},
    "string": {"minLength", "maxLength"},
    "integer": set(),
    "number": set(),
    "boolean": set(),
    "null": set(),
}

_STRING_CHARS = frozenset(ALPHABET - {'"', "\\"})
_DIGITS = frozenset("0123456789")
_NONZERO_DIGITS = frozenset("123456789")


@dataclass(frozen=True)
class ConstraintDefinition:
    """A single, already-parsed constraint definition."""

    kind: str
    body: Any

    def canonical_key(self) -> str:
        """Stable cache key for this definition."""
        return json.dumps(
            {"kind": self.kind, "body": self.body},
            sort_keys=True,
            separators=(",", ":"),
        )


def parse_definition(raw: Any) -> ConstraintDefinition:
    """Validate the envelope of one constraint definition."""
    if not isinstance(raw, dict):
        raise ConstraintInvalidError("constraint definition must be an object")
    kind = raw.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ConstraintInvalidError("constraint 'kind' must be a non-empty string")
    if kind not in SUPPORTED_KINDS:
        raise ConstraintUnsupportedError(
            f"unsupported constraint kind: {kind!r}",
            details={"supported": list(SUPPORTED_KINDS)},
        )
    if "body" not in raw:
        raise ConstraintInvalidError("constraint definition requires a 'body'")
    return ConstraintDefinition(kind=kind, body=raw["body"])


def compile_definition(defn: ConstraintDefinition) -> Regex:
    """Compile a validated definition into a regex AST."""
    if defn.kind == "json_schema":
        return _compile_json_schema(defn.body)
    if defn.kind == "regex":
        return _compile_pattern(defn.body)
    raise ConstraintUnsupportedError(f"unsupported constraint kind: {defn.kind!r}")


# ---------------------------------------------------------------------------
# JSON Schema subset
# ---------------------------------------------------------------------------

def _json_literal(value: Any) -> Regex:
    return literal(json.dumps(value, separators=(",", ":")))


def _check_unsupported_keys(schema: dict, type_name: str | None) -> None:
    allowed = set(_COMMON_KEYS)
    if type_name in _TYPE_KEYS:
        allowed |= _TYPE_KEYS[type_name]
    unknown = sorted(k for k in schema if k not in allowed)
    if unknown:
        raise ConstraintUnsupportedError(
            f"unsupported schema keyword(s): {', '.join(unknown)}",
            details={"keywords": unknown},
        )


def _compile_json_schema(schema: Any) -> Regex:
    if isinstance(schema, bool):
        raise ConstraintUnsupportedError("boolean schemas are not supported")
    if not isinstance(schema, dict):
        raise ConstraintInvalidError("schema must be an object")

    type_name = schema.get("type")
    if type_name is not None and type_name not in _JSON_TYPES:
        raise ConstraintInvalidError(
            f"'type' must be one of {list(_JSON_TYPES)}",
            details={"type": type_name},
        )
    _check_unsupported_keys(schema, type_name)
    if type_name is None and "enum" not in schema and "const" not in schema:
        raise ConstraintInvalidError(
            "schema requires a 'type' (or an enum/const)",
            details={"types": list(_JSON_TYPES)},
        )

    enum_values: list[Any] | None = None
    if "const" in schema:
        enum_values = [schema["const"]]
    if "enum" in schema:
        raw_enum = schema["enum"]
        if not isinstance(raw_enum, list):
            raise ConstraintInvalidError("'enum' must be an array")
        enum_values = list(raw_enum)
        if "const" in schema and schema["const"] not in enum_values:
            raise ConstraintUnsatisfiableError("'const' is not a member of 'enum'")

    if enum_values is not None:
        extra = set(schema) - {"type", "enum", "const"}
        if extra:
            raise ConstraintUnsupportedError(
                "'enum'/'const' cannot be combined with other keywords in this subset",
                details={"keywords": sorted(extra)},
            )
        kept = [v for v in enum_values if _value_matches_type(v, type_name)]
        if not kept:
            raise ConstraintUnsatisfiableError(
                "no enum/const value satisfies the declared type"
            )
        return Alt(tuple(_json_literal(v) for v in kept))

    assert type_name is not None
    if type_name == "object":
        return _compile_object(schema)
    if type_name == "array":
        return _compile_array(schema)
    if type_name == "string":
        return _compile_string(schema)
    if type_name == "integer":
        return _integer_regex()
    if type_name == "number":
        return _number_regex()
    if type_name == "boolean":
        return Alt((literal("true"), literal("false")))
    if type_name == "null":
        return literal("null")
    raise ConstraintInvalidError(f"unsupported type: {type_name!r}")


def _value_matches_type(value: Any, type_name: str | None) -> bool:
    if type_name is None:
        return True
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "null":
        return value is None
    return False


def _compile_object(schema: dict) -> Regex:
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ConstraintInvalidError("'properties' must be an object")
    if len(properties) > MAX_OBJECT_PROPERTIES:
        raise ConstraintUnsupportedError(
            f"objects support at most {MAX_OBJECT_PROPERTIES} properties",
            details={"properties": len(properties)},
        )
    additional = schema.get("additionalProperties", False)
    if additional is not False:
        raise ConstraintUnsupportedError(
            "only 'additionalProperties: false' is supported"
        )
    required_raw = schema.get("required", [])
    if not isinstance(required_raw, list) or not all(
        isinstance(r, str) for r in required_raw
    ):
        raise ConstraintInvalidError("'required' must be an array of strings")
    required = set(required_raw)
    missing = sorted(required - set(properties))
    if missing:
        raise ConstraintUnsatisfiableError(
            "required properties are not declared in 'properties'",
            details={"missing": missing},
        )

    compiled: list[tuple[str, Regex]] = [
        (name, _compile_json_schema(sub)) for name, sub in properties.items()
    ]
    if not compiled:
        return literal("{}")

    def pair(name: str, value: Regex) -> Regex:
        return Seq((literal(json.dumps(name) + ":"), value))

    # Chains are order-preserving subsequences of the declared properties that
    # cover every required property.  The empty chain is included when nothing
    # is required.
    chains: list[Regex] = []
    n = len(compiled)
    for mask in range(1 << n):
        chosen = [compiled[i] for i in range(n) if mask & (1 << i)]
        if not required <= {name for name, _ in chosen}:
            continue
        if not chosen:
            chains.append(Empty())
            continue
        parts: list[Regex] = [pair(*chosen[0])]
        for name, value in chosen[1:]:
            parts.append(literal(","))
            parts.append(pair(name, value))
        chains.append(Seq(tuple(parts)))
    return Seq((literal("{"), Alt(tuple(chains)), literal("}")))


def _bounded_repeat(item: Regex, sep: Regex, mn: int, mx: int | None) -> Regex:
    """``item`` repeated ``mn..mx`` times (``mx=None`` = unbounded), separated."""

    def exactly(k: int) -> Regex:
        if k == 0:
            return Empty()
        parts: list[Regex] = [item]
        for _ in range(k - 1):
            parts.extend((sep, item))
        return Seq(tuple(parts))

    head = exactly(mn)
    if mx is None:
        tail: Regex = Star(Seq((sep, item)))
        if mn == 0:
            return Opt(Seq((item, tail)))
        return Seq((head, tail))
    if mx == 0:
        return Empty()
    if mn == 0:
        # Zero occurrences, or one item followed by up to ``mx - 1`` pairs.
        return Opt(Seq((item, _opt_pairs(item, sep, mx - 1))))
    return Seq((head, _opt_pairs(item, sep, mx - mn)))


def _opt_pairs(item: Regex, sep: Regex, n: int) -> Regex:
    """Match 0..n repetitions of ``sep item``."""
    tail: Regex = Empty()
    for _ in range(n):
        tail = Opt(Seq((sep, item, tail)))
    return tail


def _compile_array(schema: dict) -> Regex:
    items = schema.get("items")
    if items is None:
        raise ConstraintUnsupportedError("arrays require an 'items' schema")
    min_items = schema.get("minItems", 0)
    max_items = schema.get("maxItems")
    for name, value in (("minItems", min_items), ("maxItems", max_items)):
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ConstraintInvalidError(f"'{name}' must be a non-negative integer")
        if value > MAX_ARRAY_ITEMS:
            raise ConstraintUnsupportedError(
                f"'{name}' supports at most {MAX_ARRAY_ITEMS}",
                details={name: value},
            )
    if max_items is not None and min_items > max_items:
        raise ConstraintUnsatisfiableError("'minItems' exceeds 'maxItems'")
    item_regex = _compile_json_schema(items)
    body = _bounded_repeat(item_regex, literal(","), min_items, max_items)
    return Seq((literal("["), body, literal("]")))


def _compile_string(schema: dict) -> Regex:
    min_length = schema.get("minLength", 0)
    max_length = schema.get("maxLength")
    for name, value in (("minLength", min_length), ("maxLength", max_length)):
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ConstraintInvalidError(f"'{name}' must be a non-negative integer")
        if value > MAX_STRING_LENGTH:
            raise ConstraintUnsupportedError(
                f"'{name}' supports at most {MAX_STRING_LENGTH}",
                details={name: value},
            )
    if max_length is not None and min_length > max_length:
        raise ConstraintUnsatisfiableError("'minLength' exceeds 'maxLength'")
    cell = Chars(_STRING_CHARS)
    body = _bounded_repeat(cell, Empty(), min_length, max_length)
    return Seq((literal('"'), body, literal('"')))


def _integer_regex() -> Regex:
    unsigned = Alt(
        (literal("0"), Seq((Chars(_NONZERO_DIGITS), Star(Chars(_DIGITS)))))
    )
    return Seq((Opt(literal("-")), unsigned))


def _number_regex() -> Regex:
    fraction = Seq((literal("."), Chars(_DIGITS), Star(Chars(_DIGITS))))
    return Seq((_integer_regex(), Opt(fraction)))


# ---------------------------------------------------------------------------
# Pattern dialect
# ---------------------------------------------------------------------------

class _PatternParser:
    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
        self.pos = 0

    def error(self, message: str) -> ConstraintInvalidError:
        return ConstraintInvalidError(
            f"invalid pattern: {message}", details={"position": self.pos}
        )

    def peek(self) -> str | None:
        return self.pattern[self.pos] if self.pos < len(self.pattern) else None

    def take(self) -> str:
        ch = self.pattern[self.pos]
        self.pos += 1
        return ch

    def parse(self) -> Regex:
        node = self.parse_alt()
        if self.pos != len(self.pattern):
            raise self.error("unexpected trailing input")
        return node

    def parse_alt(self) -> Regex:
        options = [self.parse_concat()]
        while self.peek() == "|":
            self.take()
            options.append(self.parse_concat())
        if len(options) == 1:
            return options[0]
        return Alt(tuple(options))

    def parse_concat(self) -> Regex:
        parts: list[Regex] = []
        while (ch := self.peek()) is not None and ch not in ")|":
            parts.append(self.parse_repeat())
        if not parts:
            return Empty()
        if len(parts) == 1:
            return parts[0]
        return Seq(tuple(parts))

    def parse_repeat(self) -> Regex:
        atom = self.parse_atom()
        while (ch := self.peek()) is not None and ch in "*+?{":
            self.take()
            if ch == "*":
                atom = Star(atom)
            elif ch == "+":
                atom = Seq((atom, Star(atom)))
            elif ch == "?":
                atom = Opt(atom)
            else:
                atom = self.parse_braces(atom)
        return atom

    def parse_braces(self, atom: Regex) -> Regex:
        if self.peek() == "}":
            raise self.error("empty repetition")
        digits = ""
        while (ch := self.peek()) is not None and ch.isdigit():
            digits += self.take()
        if not digits:
            raise self.error("expected a number in repetition")
        mn = int(digits)
        mx: int | None = mn
        if self.peek() == ",":
            self.take()
            digits = ""
            while (ch := self.peek()) is not None and ch.isdigit():
                digits += self.take()
            mx = int(digits) if digits else None
        if self.peek() != "}":
            raise self.error("expected '}'")
        self.take()
        if mn > MAX_REPEAT or (mx is not None and mx > MAX_REPEAT):
            raise ConstraintUnsupportedError(
                f"repetitions support at most {MAX_REPEAT}"
            )
        if mx is not None and mn > mx:
            raise ConstraintInvalidError("repetition lower bound exceeds upper bound")
        return _bounded_repeat(atom, Empty(), mn, mx)

    def parse_atom(self) -> Regex:
        ch = self.take()
        if ch == "(":
            node = self.parse_alt()
            if self.peek() != ")":
                raise self.error("unbalanced parenthesis")
            self.take()
            return node
        if ch == ")":
            raise self.error("unbalanced parenthesis")
        if ch == "[":
            return self.parse_class()
        if ch == ".":
            return Chars(frozenset(ALPHABET))
        if ch in "^$":
            raise self.error("anchors are not supported; matching is always full-output")
        if ch == "\\":
            return self.parse_escape()
        if ch in "*+?{}|":
            raise self.error(f"unexpected {ch!r}")
        return self._char(ch)

    def parse_escape(self) -> Regex:
        ch = self.peek()
        if ch is None:
            raise self.error("trailing backslash")
        self.take()
        classes = {
            "d": frozenset("0123456789"),
            "w": frozenset(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
            ),
            "s": frozenset(" "),
        }
        if ch in classes:
            return Chars(classes[ch])
        if ch.isalnum():
            raise self.error(f"unknown escape \\{ch}")
        return self._char(ch)

    def parse_class(self) -> Regex:
        negate = False
        if self.peek() == "^":
            self.take()
            negate = True
        chars: set[str] = set()
        first = True
        while True:
            ch = self.peek()
            if ch is None:
                raise self.error("unterminated character class")
            if ch == "]" and not first:
                self.take()
                break
            first = False
            self.take()
            if ch == "\\":
                esc = self.parse_escape()
                if isinstance(esc, Chars):
                    chars |= set(esc.chars)
                    continue
                raise self.error("unsupported escape in character class")
            if self.peek() == "-" and self.pos + 1 < len(self.pattern) and self.pattern[self.pos + 1] != "]":
                self.take()  # '-'
                hi = self.take()
                if ord(hi) < ord(ch):
                    raise self.error("descending character range")
                chars |= {chr(c) for c in range(ord(ch), ord(hi) + 1)}
            else:
                chars.add(ch)
        if not chars:
            raise self.error("empty character class")
        unknown = chars - ALPHABET
        if unknown:
            raise ConstraintInvalidError(
                "character class contains tokens outside the vocabulary",
                details={"characters": sorted(unknown)},
            )
        if negate:
            chars = set(ALPHABET) - chars
            if not chars:
                raise ConstraintUnsatisfiableError(
                    "negated character class excludes the whole vocabulary"
                )
        return Chars(frozenset(chars))

    def _char(self, ch: str) -> Regex:
        if ch not in ALPHABET:
            raise ConstraintInvalidError(
                "pattern contains a token outside the vocabulary",
                details={"character": ch},
            )
        return Chars(frozenset(ch))


def _compile_pattern(pattern: Any) -> Regex:
    if not isinstance(pattern, str):
        raise ConstraintInvalidError("regex constraint body must be a string")
    if not pattern:
        raise ConstraintInvalidError("regex constraint body must not be empty")
    return _PatternParser(pattern).parse()
