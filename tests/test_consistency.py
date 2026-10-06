"""Every constrained result must pass full, independent validation.

The engine masks candidates with the compiled automaton; here the generated
text is validated *from scratch* against the original definition (the
``jsonschema`` library for schemas, ``re.fullmatch`` for regex constraints)
to prove the masking never lets an invalid output through.
"""
import json
import re

import jsonschema
import pytest

from constrained_generation.manager import GenerationManager, GenerationParams
from constrained_generation.store import RequestStatus, RequestStore

SCHEMAS = [
    {"type": "object",
     "properties": {"name": {"type": "string", "maxLength": 6},
                    "age": {"type": "integer"},
                    "admin": {"type": "boolean"}},
     "required": ["name", "age"]},
    {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 4},
    {"type": "object",
     "properties": {"tags": {"type": "array",
                             "items": {"type": "string", "maxLength": 3},
                             "maxItems": 3}},
     "required": ["tags"]},
    {"enum": ["red", "green", 1, 2, None]},
    {"type": "number"},
    {"type": "object",
     "properties": {"nested": {"type": "object",
                               "properties": {"ok": {"type": "boolean"}},
                               "required": ["ok"]}},
     "required": ["nested"]},
]

PATTERNS = ['"(ab|cd)+"', '"[A-Z][a-z]{2,5}"', "x[0-9]{2,4}y"]


@pytest.fixture()
def manager():
    mgr = GenerationManager(RequestStore(None))
    yield mgr
    mgr.shutdown()


def generate(manager, constraints, seed):
    record = manager.submit(
        "t", GenerationParams(constraints=constraints, max_tokens=1024, seed=seed)
    )
    assert record.status == RequestStatus.COMPLETED, record.error
    return record.result


@pytest.mark.parametrize("schema", SCHEMAS, ids=[str(s)[:40] for s in SCHEMAS])
def test_schema_constrained_output_passes_full_validation(manager, schema):
    for seed in range(6):
        text = generate(manager, [{"kind": "json_schema", "body": schema}], seed)
        instance = json.loads(text)  # must be parseable JSON
        jsonschema.validate(instance=instance, schema=schema)  # raises on mismatch


@pytest.mark.parametrize("pattern", PATTERNS)
def test_regex_constrained_output_fullmatches(manager, pattern):
    for seed in range(6):
        text = generate(manager, [{"kind": "regex", "body": pattern}], seed)
        assert re.fullmatch(pattern, text), text


def test_multiple_constraints_all_hold(manager):
    schema = {"type": "string", "minLength": 2, "maxLength": 8}
    pattern = '"(ab)+"'
    for seed in range(6):
        text = generate(
            manager,
            [{"kind": "json_schema", "body": schema}, {"kind": "regex", "body": pattern}],
            seed,
        )
        jsonschema.validate(instance=json.loads(text), schema=schema)
        assert re.fullmatch(pattern, text), text


def test_seed_makes_generation_reproducible(manager):
    a = generate(manager, [{"kind": "json_schema", "body": SCHEMAS[0]}], 11)
    b = generate(manager, [{"kind": "json_schema", "body": SCHEMAS[0]}], 11)
    assert a == b
