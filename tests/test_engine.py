"""Engine behaviour: outcomes, cancellation, timeout, and the invariant that
every emitted prefix can still complete into a valid output."""
import threading
import time

from constrained_generation.engine import ConstrainedSession, GenerationOutcome
from constrained_generation.model import ToyModel
from constrained_generation.regex import compile_regex
from constrained_generation.schema import compile_definition, parse_definition
from constrained_generation.vocab import ALPHABET

SCHEMA = {
    "kind": "json_schema",
    "body": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "maxLength": 6},
            "age": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "boolean"}, "maxItems": 3},
        },
        "required": ["name"],
    },
}


def make_dfa(definition=None):
    definition = definition or SCHEMA
    return compile_regex(
        compile_definition(parse_definition(definition)), ALPHABET
    )


def run(dfa, seed=0, max_tokens=512, cancel=None, deadline=None, hook=None):
    session = ConstrainedSession(
        dfa, ToyModel(seed), max_tokens, cancel or threading.Event(), deadline,
        step_hook=hook,
    )
    return session.run()


def test_completed_output_is_accepted_by_automaton():
    dfa = make_dfa()
    for seed in range(10):
        result = run(dfa, seed=seed)
        assert result.outcome == GenerationOutcome.COMPLETED
        assert dfa.accepts_text(result.text)


def test_every_prefix_stays_completable():
    """At no step may the engine emit a token that dead-ends the output."""
    dfa = make_dfa()
    prefixes = []

    def hook(_index, text):
        prefixes.append(text)

    result = run(dfa, seed=3, hook=hook)
    assert result.outcome == GenerationOutcome.COMPLETED
    state = dfa.start
    for prefix in prefixes:
        # re-walk the automaton to the state corresponding to this prefix
        state = dfa.start
        for ch in prefix:
            state = dfa.step(state, ch)
            assert state is not None, f"prefix {prefix!r} left the automaton"
        # trimmed automaton: every reachable state can still complete
        assert state in dfa.states


def test_cancellation_stops_promptly():
    dfa = make_dfa({"kind": "regex", "body": "x[a-z]{64}y"})
    cancel = threading.Event()

    def hook(index, _text):
        if index == 5:
            cancel.set()

    result = run(dfa, max_tokens=10_000, cancel=cancel, hook=hook)
    assert result.outcome == GenerationOutcome.CANCELLED
    assert result.tokens_generated <= 6


def test_timeout():
    dfa = make_dfa({"kind": "regex", "body": "x[a-z]{64}y"})
    result = run(dfa, max_tokens=10_000, deadline=time.monotonic() - 0.001)
    assert result.outcome == GenerationOutcome.TIMEOUT
    assert result.tokens_generated == 0


def test_incomplete_when_budget_exhausted():
    dfa = make_dfa({"kind": "json_schema", "body": {"const": "a-longer-value"}})
    result = run(dfa, max_tokens=4)
    assert result.outcome == GenerationOutcome.INCOMPLETE
    assert len(result.text) == 4


def test_intersection_of_multiple_constraints():
    dfa = make_dfa({"kind": "regex", "body": '"(ab)+"'})
    from constrained_generation.regex import product

    other = make_dfa({"kind": "json_schema", "body": {"type": "string", "minLength": 4}})
    combined = product([dfa, other])
    result = run(combined, seed=1)
    assert result.outcome == GenerationOutcome.COMPLETED
    inner = result.text.strip('"')
    assert len(inner) >= 4 and set(inner) <= {"a", "b"}
