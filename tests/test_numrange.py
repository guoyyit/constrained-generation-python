from __future__ import annotations

import random
import re

import pytest

from constrained.errors import ConstraintError
from constrained.numrange import int_range_to_regex


@pytest.mark.parametrize(
    "lo, hi",
    [
        (0, 0),
        (0, 9),
        (5, 5),
        (0, 10),
        (5, 15),
        (0, 100),
        (1, 999),
        (97, 122),
        (-1, 1),
        (-9, -1),
        (-100, -1),
        (-10, -10),
        (-123, 456),
        (1000, 9999),
        (2**62, 2**63 - 1),
        (-(2**63), -(2**62)),
    ],
)
def test_range_matches_exactly(lo: int, hi: int) -> None:
    pattern = int_range_to_regex(lo, hi)
    rx = re.compile(pattern)
    rng = random.Random(lo * 31 + hi)
    inside = [lo, hi] + [rng.randint(lo, hi) for _ in range(200)]
    for n in inside:
        assert rx.fullmatch(str(n)), (lo, hi, n, pattern)
    outside = [lo - 1, hi + 1, lo - 2, hi + 2]
    for _ in range(200):
        n = rng.randint(-(2**63), 2**63 - 1)
        if not lo <= n <= hi:
            outside.append(n)
    for n in outside:
        assert not rx.fullmatch(str(n)), (lo, hi, n, pattern)
    for bad in ["-0", "01", "007", "+1", " 1", "1 ", "1.0", ""]:
        assert not rx.fullmatch(bad), (lo, hi, bad, pattern)


def test_empty_range_is_contradictory() -> None:
    with pytest.raises(ConstraintError) as excinfo:
        int_range_to_regex(5, 3)
    assert excinfo.value.reason == "contradictory"


def test_huge_range_is_too_complex() -> None:
    with pytest.raises(ConstraintError) as excinfo:
        int_range_to_regex(0, 2**64)
    assert excinfo.value.reason == "too_complex"
