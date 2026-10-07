from __future__ import annotations

import json
import re

import pytest
from jsonschema import validate

from constrained.errors import ConstraintError
from constrained.schema import schema_to_regex

PERSON = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 20},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "active": {"type": "boolean"},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}


def canonical(value) -> str:
    """Canonical output form: required keys first (declared order), then
    optional keys (declared order), compact separators, ASCII-escaped."""
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: kv[0])
        return "{" + ",".join(
            json.dumps(k, ensure_ascii=True) + ":" + canonical(v) for k, v in items
        ) + "}"
    if isinstance(value, list):
        return "[" + ",".join(canonical(v) for v in value) + "]"
    return json.dumps(value, ensure_ascii=True)


def test_object_pattern_accepts_canonical_instances() -> None:
    rx = re.compile(schema_to_regex(PERSON))
    instances = [
        {"name": "a", "age": 0},
        {"name": "x" * 20, "age": 150, "active": False},
        {"name": "n", "age": 42, "tags": ["u", "v"], "active": True},
        {"name": 'quote"and\\back', "age": 7, "tags": []},
        {"name": "uni-é", "age": 9},
    ]
    for value in instances:
        validate(value, PERSON)
        # canonical key order for this schema: name, age, tags, active
        ordered = {}
        for key in ("name", "age", "tags", "active"):
            if key in value:
                ordered[key] = value[key]
        text = json.dumps(ordered, ensure_ascii=True, separators=(",", ":"))
        assert rx.fullmatch(text), text


def test_object_pattern_rejects_nonconforming() -> None:
    rx = re.compile(schema_to_regex(PERSON))
    bad = [
        '{"name":"a"}',  # missing required age
        '{"name":"a","age":151}',
        '{"name":"a","age":-1}',
        '{"name":"","age":1}',
        '{"name":"a","age":1,"x":1}',
        '{"age":1,"name":"a"}',  # non-canonical key order
        '{ "name": "a", "age": 1 }',  # whitespace
        '{"name":"a","age":1,"active":true,"tags":[]}',  # optional out of order
        '{"name":"a","age":1,"tags":["x","y","z","w"]}',  # maxItems
    ]
    for text in bad:
        assert not rx.fullmatch(text), text


def test_all_optional_object_any_subsequence() -> None:
    rx = re.compile(
        schema_to_regex(
            {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "boolean"},
                    "c": {"type": "null"},
                },
                "additionalProperties": False,
            }
        )
    )
    for ok in [
        "{}",
        '{"a":1}',
        '{"b":true}',
        '{"c":null}',
        '{"a":1,"c":null}',
        '{"a":-2,"b":false,"c":null}',
    ]:
        assert rx.fullmatch(ok), ok
    for no in ['{"b":true,"a":1}', '{"a":1,}', '{,"a":1}', '{"a":1,"a":2}', '{"d":1}']:
        assert not rx.fullmatch(no), no


def test_string_escapes() -> None:
    rx = re.compile(schema_to_regex({"type": "string"}))
    assert rx.fullmatch('""')
    assert rx.fullmatch('"plain"')
    assert rx.fullmatch('"esc \\n \\t \\" \\\\ \\/ é"')
    assert not rx.fullmatch('"raw\nnewline"')
    assert not rx.fullmatch('"bad \\x escape"')
    assert not rx.fullmatch('"unterminated')


def test_enum_and_const() -> None:
    rx = re.compile(schema_to_regex({"enum": ["red", "green", 1, None, True]}))
    for ok in ['"red"', '"green"', "1", "null", "true"]:
        assert rx.fullmatch(ok), ok
    for no in ['"blue"', "2", "false"]:
        assert not rx.fullmatch(no), no
    assert re.compile(schema_to_regex({"const": "x"})).fullmatch('"x"')


def test_number_pattern() -> None:
    rx = re.compile(schema_to_regex({"type": "number"}))
    for ok in ["0", "-1", "3.25", "-0.5e10", "1E-3", "123"]:
        assert rx.fullmatch(ok), ok
    for no in ["01", "1.", ".5", "1e", "0x1", ""]:
        assert not rx.fullmatch(no), no


FAILURE_CASES = [
    ({"type": "integer", "minimum": 5, "maximum": 3}, "contradictory"),
    ({"type": "integer", "exclusiveMinimum": 5, "maximum": 5}, "contradictory"),
    ({"type": "string", "minLength": 4, "maxLength": 2}, "contradictory"),
    ({"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 1}, "contradictory"),
    ({"enum": []}, "contradictory"),
    ({"enum": ["a"], "type": "integer"}, "contradictory"),
    (
        {
            "type": "object",
            "properties": {},
            "required": ["a"],
            "additionalProperties": False,
        },
        "contradictory",
    ),
    ({"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": True}, "unsupported"),
    ({"type": "array"}, "unsupported"),
    ({"type": "string", "pattern": "x"}, "unsupported"),
    ({"type": "number", "minimum": 0}, "unsupported"),
    ({"type": "string", "uniqueItems": True}, "unsupported"),
    ({"oneOf": [{"type": "string"}]}, "unsupported"),
    ({"$ref": "#/definitions/x"}, "unsupported"),
    (True, "unsupported"),
    ({}, "unsupported"),
    ({"type": "foo"}, "invalid_definition"),
    ({"type": "string", "minLength": -1}, "invalid_definition"),
    ({"type": "string", "maxLength": 100000}, "too_complex"),
    ({"type": "array", "items": {"type": "string"}, "maxItems": 1000}, "too_complex"),
    (False, "unsatisfiable"),
]


@pytest.mark.parametrize("schema, reason", FAILURE_CASES)
def test_failure_classification(schema, reason: str) -> None:
    with pytest.raises(ConstraintError) as excinfo:
        schema_to_regex(schema)
    assert excinfo.value.reason == reason, (schema, excinfo.value.reason)


def test_deep_nesting_rejected() -> None:
    schema = {"type": "string"}
    for _ in range(40):
        schema = {"type": "array", "items": schema}
    with pytest.raises(ConstraintError) as excinfo:
        schema_to_regex(schema)
    assert excinfo.value.reason == "too_complex"
