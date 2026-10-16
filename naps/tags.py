"""Tag id normalisation: tag ids leave NAPS as plain Python ints.

The evidence layer, the closed-set and quadrant paths and the SLEAP output normalise the tags they
handle. The legacy path (no evidence sources, no closed set, no quadrants) hands the raw decodes
straight to the cost matrix, so raw decode values only become plain ints when the tracks are named.
"""

import math
from numbers import Integral
from typing import Literal, SupportsFloat, TypeVar, overload

__all__ = ["TAG_PREFIX", "normalize_tag", "require_tag", "tag_name"]

_T = TypeVar("_T")

TAG_PREFIX = "ArUcoTag#"
"""Prefix of the SLEAP track names written by NAPS (``ArUcoTag#<int>``)."""

# Strings (compared lower-cased after stripping) that mean "no tag".
_NULL_STRINGS = frozenset({"", "none", "nan"})


@overload
def normalize_tag(tag: object, strict: Literal[True]) -> int | None: ...
@overload
def normalize_tag(tag: _T, strict: Literal[False] = False) -> int | _T | None: ...
@overload
def normalize_tag(tag: _T, strict: bool = False) -> int | _T | None: ...
def normalize_tag(tag: object, strict: bool = False) -> object:
    """Return ``tag`` as a Python ``int``.

    Integral values of any type (``int``, ``numpy.integer``, ``bool``) are converted exactly,
    with no float round trip, so even ids above ``2**53`` survive. Integral floats such as
    ``21.0`` (or the string ``"21.0"``) become ``21``.

    Args:
        tag: A tag id as it came out of OpenCV, numpy, pandas or a user-supplied string.
        strict: If False (default) a value that is not an integral number (for example the
            label ``"a"`` used by some unit tests, or ``21.5``) is returned unchanged. If True
            it is mapped to ``None`` instead.

    Returns:
        The tag as an ``int``; ``None`` for ``None``, the empty string, ``"none"``/``"nan"``,
        NaN and infinities; otherwise the unchanged input (``strict=False``) or ``None``
        (``strict=True``).
    """
    if tag is None or (isinstance(tag, str) and tag.strip().lower() in _NULL_STRINGS):
        return None
    if isinstance(tag, Integral):
        return int(tag)
    if isinstance(tag, str):
        try:
            return int(tag)  # exact for any length; int() ignores surrounding whitespace
        except ValueError:
            pass  # not a plain integer string: try the float spelling ("21.0", "1e3") below
    value = _to_float(tag)
    if value is not None and not math.isfinite(value):
        return None
    if value is None or value != int(value):
        return None if strict else tag
    return int(value)


def _to_float(tag: object) -> float | None:
    """Return ``float(tag)``, or None if ``tag`` is not number-like."""
    number: SupportsFloat | str
    if isinstance(tag, str):
        number = tag.strip()
    elif isinstance(tag, SupportsFloat):
        number = tag
    else:
        return None
    try:
        return float(number)
    except (TypeError, ValueError, OverflowError):  # OverflowError: e.g. Fraction(10**400)
        return None


def require_tag(tag: object) -> int:
    """Return ``tag`` as a Python ``int``, refusing anything that is not an integral tag id.

    Used for user-supplied tag sets, where silently truncating ``21.5`` (what ``int()`` does) or
    failing with an unrelated ``TypeError`` for ``None`` would hide a mistake.

    Args:
        tag: A tag id (anything :func:`normalize_tag` accepts in strict mode).

    Returns:
        The tag as an ``int``.

    Raises:
        ValueError: If ``tag`` is not a valid tag id (``None``, NaN, non-integral, non-numeric).
    """
    value = normalize_tag(tag, strict=True)
    if value is None:
        msg = f"not a valid tag id: {tag!r}"
        raise ValueError(msg)
    return value


def tag_name(tag: object) -> str:
    """Return the SLEAP track name for ``tag``: always ``ArUcoTag#<int>``.

    Args:
        tag: A tag id (anything :func:`normalize_tag` accepts in strict mode, e.g. ``21.0``).

    Returns:
        The track name, e.g. ``"ArUcoTag#21"``.

    Raises:
        ValueError: If ``tag`` is not a valid tag id (``None``, NaN, non-integral, non-numeric).
    """
    return f"{TAG_PREFIX}{require_tag(tag)}"
