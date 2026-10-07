"""Compile a supported subset of JSON Schema into a regex over compact JSON text.

The generated pattern matches *canonical* JSON: no insignificant whitespace,
object keys ordered with required properties first (in declared order) then
present optional properties (in declared order), strings/keys in
``json.dumps(..., ensure_ascii=True)`` form.  Anything outside the supported
subset is rejected up front with a distinguishable failure reason instead of
silently degrading to an unconstrained or weakened constraint.
"""

from __future__ import annotations

import json
import math
from typing import Any

from constrained.errors import ConstraintError, FailureReason
from constrained.numrange import int_range_to_regex
from constrained.regex_nfa import MAX_REPEAT

#: Maximum nesting depth of a schema.
MAX_DEPTH = 32

_VALID_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}

#: Keywords that would change the meaning of the constraint and that this
#: engine cannot enforce regularly.  Their presence is an error, never a
#: silent no-op.
_UNSUPPORTED_KEYWORDS = {
    "oneOf",
    "anyOf",
    "allOf",
    "not",
    "$ref",
    "$defs",
    "definitions",
    "pattern",
    "contains",
    "propertyNames",
    "patternProperties",
    "dependentRequired",
    "dependentSchemas",
    "if",
    "then",
    "else",
    "unevaluatedProperties",
    "prefixItems",
    "uniqueItems",
    "multipleOf",
    "minProperties",
    "maxProperties",
}

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1

#: One JSON string character: an unescaped char or an escape sequence.
_STRING_UNIT = r'(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))'


def _literal(text: str) -> str:
    """Regex-escape a literal string (all chars are printable ASCII)."""
    return "".join(ch if ch.isalnum() else "\\" + ch for ch in text)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _join(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return "(?:" + "|".join(parts) + ")"


def _quant(lo: int, hi: int | None) -> str:
    if hi is None:
        if lo == 0:
            return "*"
        if lo == 1:
            return "+"
        return f"{{{lo},}}"
    if lo == hi:
        return f"{{{lo}}}"
    if lo == 0 and hi == 1:
        return "?"
    return f"{{{lo},{hi}}}"


def _check_count(schema: dict[str, Any], lo_key: str, hi_key: str) -> tuple[int, int | None]:
    lo = schema.get(lo_key, 0)
    hi = schema.get(hi_key)
    for key, val in ((lo_key, lo), (hi_key, hi)):
        if val is not None and (not isinstance(val, int) or isinstance(val, bool) or val < 0):
            raise ConstraintError(
                FailureReason.INVALID_DEFINITION, f"{key} must be a non-negative integer"
            )
    if hi is not None and lo > hi:
        raise ConstraintError(
            FailureReason.CONTRADICTORY, f"{lo_key} {lo} is greater than {hi_key} {hi}"
        )
    for key, val in ((lo_key, lo), (hi_key, hi)):
        if val is not None and val > MAX_REPEAT:
            raise ConstraintError(
                FailureReason.TOO_COMPLEX, f"{key} {val} exceeds limit of {MAX_REPEAT}"
            )
    return lo, hi


def _matches_type(value: Any, type_: str) -> bool:
    if type_ == "null":
        return value is None
    if type_ == "boolean":
        return isinstance(value, bool)
    if type_ == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_ == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_ == "string":
        return isinstance(value, str)
    if type_ == "object":
        return isinstance(value, dict)
    if type_ == "array":
        return isinstance(value, list)
    return False


def _object_regex(schema: dict[str, Any], depth: int) -> str:
    props = schema.get("properties", {})
    if not isinstance(props, dict):
        raise ConstraintError(
            FailureReason.INVALID_DEFINITION, "'properties' must be an object"
        )
    additional = schema.get("additionalProperties", False)
    if additional is not False:
        raise ConstraintError(
            FailureReason.UNSUPPORTED,
            "only closed objects are supported: 'additionalProperties' must be false",
        )
    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(r, str) for r in required):
        raise ConstraintError(
            FailureReason.INVALID_DEFINITION, "'required' must be a list of strings"
        )
    missing = [k for k in required if k not in props]
    if missing:
        raise ConstraintError(
            FailureReason.CONTRADICTORY,
            f"required properties {missing!r} are not defined in 'properties' "
            "of a closed object and can never be produced",
        )
    required_set = set(required)

    def member_re(key: str) -> str:
        return _literal(json.dumps(key, ensure_ascii=True)) + ":" + _compile(
            props[key], depth + 1
        )

    required_res = [member_re(k) for k in props if k in required_set]
    optional_res = [member_re(k) for k in props if k not in required_set]
    if not required_res and not optional_res:
        return "\\{\\}"
    parts: list[str] = [",".join(required_res)]
    if optional_res:
        if required_res:
            # After a required member every optional member is simply a
            # comma-prefixed optional group.
            parts.append("".join(f"(?:,{r})?" for r in optional_res))
        else:
            # No required member: the first present optional member must not
            # carry a leading comma, so alternate over which member is first.
            alts = []
            for i, r in enumerate(optional_res):
                tail = "".join(f"(?:,{t})?" for t in optional_res[i + 1 :])
                alts.append(r + tail)
            parts.append("(?:" + "|".join(alts) + ")?")
    return "\\{" + "".join(parts) + "\\}"


def _array_regex(schema: dict[str, Any], depth: int) -> str:
    if "items" not in schema:
        raise ConstraintError(
            FailureReason.UNSUPPORTED, "array schemas must declare 'items'"
        )
    min_items, max_items = _check_count(schema, "minItems", "maxItems")
    item = "(?:" + _compile(schema["items"], depth + 1) + ")"
    if max_items == 0:
        return "\\[\\]"
    if min_items == 0:
        if max_items is None:
            return "\\[(?:" + item + "(?:," + item + ")*)?\\]"
        return "\\[(?:" + item + "(?:," + item + ")" + _quant(0, max_items - 1) + ")?\\]"
    head = "\\[" + item
    extra = _quant(min_items - 1, None if max_items is None else max_items - 1)
    if extra not in ("{0}", "{0,0}"):
        head += "(?:," + item + ")" + extra
    return head + "\\]"


def _string_regex(schema: dict[str, Any]) -> str:
    min_len, max_len = _check_count(schema, "minLength", "maxLength")
    return '"' + _STRING_UNIT + _quant(min_len, max_len) + '"'


def _integer_regex(schema: dict[str, Any]) -> str:
    lo: int | None = None
    hi: int | None = None
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        val = schema.get(key)
        if val is None:
            continue
        if isinstance(val, bool):
            raise ConstraintError(
                FailureReason.UNSUPPORTED,
                f"boolean form of {key} is not supported",
            )
        if not isinstance(val, (int, float)):
            raise ConstraintError(
                FailureReason.INVALID_DEFINITION, f"{key} must be a number"
            )
        if key == "minimum":
            lo = math.ceil(val)
        elif key == "exclusiveMinimum":
            lo = math.floor(val) + 1
        elif key == "maximum":
            hi = math.floor(val)
        else:
            hi = math.ceil(val) - 1
    if lo is None and hi is None:
        return "-?(?:0|[1-9]\\d*)"
    lo = lo if lo is not None else _INT64_MIN
    hi = hi if hi is not None else _INT64_MAX
    if lo < _INT64_MIN or hi > _INT64_MAX:
        raise ConstraintError(
            FailureReason.TOO_COMPLEX,
            "integer bounds must fit in the signed 64-bit range",
        )
    return int_range_to_regex(lo, hi)


def _compile(schema: Any, depth: int) -> str:
    if depth > MAX_DEPTH:
        raise ConstraintError(
            FailureReason.TOO_COMPLEX, f"schema nesting exceeds {MAX_DEPTH} levels"
        )
    if isinstance(schema, bool):
        # JSON Schema boolean form: true = anything (not regular), false = nothing.
        if schema is False:
            raise ConstraintError(
                FailureReason.UNSATISFIABLE, "schema 'false' matches no output"
            )
        raise ConstraintError(
            FailureReason.UNSUPPORTED, "schema 'true' (match anything) is not supported"
        )
    if not isinstance(schema, dict):
        raise ConstraintError(
            FailureReason.INVALID_DEFINITION, "schema must be an object"
        )
    unsupported = sorted(k for k in schema if k in _UNSUPPORTED_KEYWORDS)
    if unsupported:
        raise ConstraintError(
            FailureReason.UNSUPPORTED,
            f"unsupported keyword(s): {', '.join(unsupported)}",
        )

    enum: list[Any] | None = None
    if "const" in schema:
        enum = [schema["const"]]
    elif "enum" in schema:
        raw = schema["enum"]
        if not isinstance(raw, list):
            raise ConstraintError(
                FailureReason.INVALID_DEFINITION, "'enum' must be a list"
            )
        enum = raw

    type_ = schema.get("type")
    if type_ is not None and type_ not in _VALID_TYPES:
        raise ConstraintError(
            FailureReason.INVALID_DEFINITION, f"unknown type {type_!r}"
        )
    if type_ is None and enum is None:
        raise ConstraintError(
            FailureReason.UNSUPPORTED,
            "schema must declare 'type' (or 'enum'/'const'); "
            "unconstrained values are not supported",
        )

    if enum is not None:
        if not enum:
            raise ConstraintError(
                FailureReason.CONTRADICTORY, "'enum' is empty and matches no output"
            )
        if type_ is not None:
            enum = [v for v in enum if _matches_type(v, type_)]
            if not enum:
                raise ConstraintError(
                    FailureReason.CONTRADICTORY,
                    f"no 'enum' value conforms to type {type_!r}",
                )
        return _join([_literal(_canonical_json(v)) for v in enum])

    if type_ == "object":
        return _object_regex(schema, depth)
    if type_ == "array":
        return _array_regex(schema, depth)
    if type_ == "string":
        return _string_regex(schema)
    if type_ == "integer":
        return _integer_regex(schema)
    if type_ == "number":
        for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            if key in schema:
                raise ConstraintError(
                    FailureReason.UNSUPPORTED,
                    f"numeric bounds on 'number' are not supported; use 'integer'",
                )
        return "-?(?:0|[1-9]\\d*)(?:\\.\\d+)?(?:[eE][+-]?\\d+)?"
    if type_ == "boolean":
        return "(?:true|false)"
    if type_ == "null":
        return "null"
    raise AssertionError(f"unhandled type {type_!r}")


def schema_to_regex(schema: dict[str, Any]) -> str:
    """Translate a JSON Schema (supported subset) into a regex string."""
    return _compile(schema, 0)
