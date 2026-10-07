"""Compile integer ranges [lo, hi] into regex alternations.

The generated regex matches exactly the base-10 representations of the
integers in the range, with no leading zeros (JSON number syntax), so it can
be embedded directly into a JSON Schema derived pattern.
"""

from __future__ import annotations

from constrained.errors import ConstraintError, FailureReason

#: Integer bounds wider than signed 64-bit are rejected to keep the
#: generated pattern (and therefore the NFA) bounded.
MAX_RANGE_BITS = 63


def _dig(n: int) -> str:
    if n == 0:
        return ""
    if n == 1:
        return "\\d"
    return f"\\d{{{n}}}"


def _class_for(a: int, b: int) -> str:
    if a == b:
        return str(a)
    if a == 0 and b == 9:
        return "\\d"
    return f"[{a}-{b}]"


def _fixed(lo: int, hi: int, width: int) -> list[str]:
    """Patterns matching exactly the ``width``-digit (zero-padded) strings of
    the numbers in [lo, hi]."""
    assert 0 <= lo <= hi <= 10**width - 1
    if lo == hi:
        return [str(lo).zfill(width)]
    if lo == 0 and hi == 10**width - 1:
        return [_dig(width)]
    s = str(lo).zfill(width)
    e = str(hi).zfill(width)
    i = 0
    while s[i] == e[i]:
        i += 1
    prefix = s[:i]
    d_lo, d_hi = int(s[i]), int(e[i])
    rest = width - i - 1
    pats: list[str] = []
    # block for the lowest digit: suffixes from s[i+1:] up to 9..9
    low_val = int(s[i + 1 :]) if rest else 0
    for p in _fixed(low_val, 10**rest - 1, rest) if rest else [""]:
        pats.append(prefix + str(d_lo) + p)
    # middle digits cover the full suffix range
    if d_hi - d_lo >= 2:
        pats.append(prefix + _class_for(d_lo + 1, d_hi - 1) + _dig(rest))
    # block for the highest digit: suffixes from 0..0 up to e[i+1:]
    high_val = int(e[i + 1 :]) if rest else 0
    for p in _fixed(0, high_val, rest) if rest else [""]:
        pats.append(prefix + str(d_hi) + p)
    return pats


def _nonneg(lo: int, hi: int) -> list[str]:
    """Patterns for [lo, hi] with 0 <= lo <= hi and no leading zeros."""
    pats: list[str] = []
    first_len = len(str(lo)) if lo > 0 else 1
    for length in range(first_len, len(str(hi)) + 1):
        a = max(lo, 10 ** (length - 1)) if length > 1 else lo
        b = min(hi, 10**length - 1)
        if a > b:
            continue
        pats.extend(_fixed(a, b, length))
    return pats


def int_range_to_regex(lo: int, hi: int) -> str:
    """Return a regex matching exactly the integers in [lo, hi] (no leading zeros)."""
    if lo > hi:
        raise ConstraintError(
            FailureReason.CONTRADICTORY,
            f"integer range is empty: minimum {lo} > maximum {hi}",
        )
    if lo < -(2**63) or hi > 2**63 - 1:
        raise ConstraintError(
            FailureReason.TOO_COMPLEX,
            f"integer bounds must fit in signed {MAX_RANGE_BITS + 1}-bit range",
        )
    parts: list[str] = []
    if lo < 0:
        if hi < 0:
            parts.append("-" + _join(_nonneg(-hi, -lo)))
        else:
            parts.append("-" + _join(_nonneg(1, -lo)))
            parts.extend(_nonneg(0, hi))
    else:
        parts.extend(_nonneg(lo, hi))
    return _join(parts)


def _join(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return "(?:" + "|".join(parts) + ")"
