"""Constraint definitions are rejected, with distinguishable codes, before
any generation starts."""
import pytest

from constrained_generation.errors import (
    ConstraintInvalidError,
    ConstraintUnsatisfiableError,
    ConstraintUnsupportedError,
)
from constrained_generation.regex import compile_regex
from constrained_generation.schema import compile_definition, parse_definition
from constrained_generation.vocab import ALPHABET


def compile_dfa(raw):
    return compile_regex(compile_definition(parse_definition(raw)), ALPHABET)


# ---------------------------------------------------------------------------
# invalid: malformed definitions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw",
    [
        "not-an-object",
        {"kind": "json_schema"},  # missing body
        {"body": {}},  # missing kind
        {"kind": "", "body": {}},
        {"kind": "json_schema", "body": {"type": "text"}},  # unknown type
        {"kind": "json_schema", "body": {"type": "string", "maxLength": -1}},
        {"kind": "json_schema", "body": {"type": "string", "minLength": "x"}},
        {"kind": "json_schema", "body": {"type": "object", "required": "name"}},
        {"kind": "json_schema", "body": {"type": "array", "items": {"type": "null"}, "minItems": -1}},
        {"kind": "regex", "body": 42},
        {"kind": "regex", "body": ""},
        {"kind": "regex", "body": "(unclosed"},
        {"kind": "regex", "body": "a{3,2}"},
        {"kind": "regex", "body": "^anchored"},
        {"kind": "regex", "body": "trailing$"},
        {"kind": "regex", "body": "é"},  # outside the token vocabulary
        {"kind": "regex", "body": "[z-a]"},
    ],
)
def test_invalid_definitions(raw):
    with pytest.raises(ConstraintInvalidError) as excinfo:
        compile_dfa(raw)
    assert excinfo.value.code.value == "constraint/invalid"


# ---------------------------------------------------------------------------
# unsupported: valid but outside the supported subset
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw",
    [
        {"kind": "frobnicate", "body": {}},
        {"kind": "json_schema", "body": True},  # boolean schemas
        {"kind": "json_schema", "body": {"type": "string", "pattern": "a+"}},
        {"kind": "json_schema", "body": {"type": "integer", "minimum": 3}},
        {"kind": "json_schema", "body": {"type": "object", "additionalProperties": True}},
        {"kind": "json_schema", "body": {"type": "object", "properties": {str(i): {"type": "null"} for i in range(9)}}},
        {"kind": "json_schema", "body": {"type": "array", "items": {"type": "null"}, "maxItems": 100}},
        {"kind": "json_schema", "body": {"type": "string", "maxLength": 1000}},
        {"kind": "json_schema", "body": {"allOf": [{"type": "null"}]}},
        {"kind": "regex", "body": "a{200}"},
    ],
)
def test_unsupported_definitions(raw):
    with pytest.raises(ConstraintUnsupportedError) as excinfo:
        compile_dfa(raw)
    assert excinfo.value.code.value == "constraint/unsupported"


# ---------------------------------------------------------------------------
# unsatisfiable: well-formed, but no output can satisfy them
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw",
    [
        {"kind": "json_schema", "body": {"enum": []}},
        {"kind": "json_schema", "body": {"type": "string", "enum": []}},
        {"kind": "json_schema", "body": {"type": "integer", "enum": ["not-an-int"]}},
        {"kind": "json_schema", "body": {"type": "string", "const": "a", "enum": ["b"]}},
        {"kind": "json_schema", "body": {"type": "string", "minLength": 5, "maxLength": 2}},
        {"kind": "json_schema", "body": {"type": "array", "items": {"type": "null"}, "minItems": 3, "maxItems": 2}},
        {"kind": "json_schema", "body": {"type": "object", "properties": {"a": {"type": "null"}}, "required": ["ghost"]}},
    ],
)
def test_unsatisfiable_definitions(raw):
    with pytest.raises(ConstraintUnsatisfiableError) as excinfo:
        compile_dfa(raw)
    assert excinfo.value.code.value == "constraint/unsatisfiable"


# ---------------------------------------------------------------------------
# compilation correctness
# ---------------------------------------------------------------------------

def test_object_with_optional_properties():
    dfa = compile_dfa(
        {
            "kind": "json_schema",
            "body": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "maxLength": 4},
                    "age": {"type": "integer"},
                },
                "required": ["name"],
            },
        }
    )
    assert dfa.accepts_text('{"name":"ab"}')
    assert dfa.accepts_text('{"name":"ab","age":-3}')
    assert not dfa.accepts_text('{"age":1}')  # required missing
    assert not dfa.accepts_text('{"name":"ab","x":1}')  # additional property
    assert not dfa.accepts_text('{"name":"abcde"}')  # maxLength exceeded


def test_array_bounds():
    dfa = compile_dfa(
        {
            "kind": "json_schema",
            "body": {
                "type": "array",
                "items": {"type": "boolean"},
                "minItems": 1,
                "maxItems": 2,
            },
        }
    )
    assert dfa.accepts_text("[true]") and dfa.accepts_text("[true,false]")
    assert not dfa.accepts_text("[]") and not dfa.accepts_text("[true,false,true]")


def test_nested_schema():
    dfa = compile_dfa(
        {
            "kind": "json_schema",
            "body": {
                "type": "object",
                "properties": {
                    "tags": {"type": "array", "items": {"type": "string", "maxLength": 2}}
                },
                "required": ["tags"],
            },
        }
    )
    assert dfa.accepts_text('{"tags":["a","bc"]}')
    assert not dfa.accepts_text('{"tags":["abc"]}')


def test_enum_and_const():
    dfa = compile_dfa({"kind": "json_schema", "body": {"enum": ["red", 1, None]}})
    assert dfa.accepts_text('"red"') and dfa.accepts_text("1") and dfa.accepts_text("null")
    assert not dfa.accepts_text('"blue"')
    const = compile_dfa({"kind": "json_schema", "body": {"const": "only"}})
    assert const.accepts_text('"only"') and not const.accepts_text('"other"')


def test_regex_kind():
    dfa = compile_dfa({"kind": "regex", "body": '"(ab|cd)+"'})
    assert dfa.accepts_text('"ababcd"')
    assert not dfa.accepts_text('"abx"')
