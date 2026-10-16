import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from naps.tags import TAG_PREFIX, normalize_tag, require_tag, tag_name


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (21, 21),
        (0, 0),
        (-3, -3),
        (True, 1),
        (21.0, 21),
        (np.int8(21), 21),
        (np.uint64(21), 21),
        (np.int64(21), 21),
        (np.float32(21.0), 21),
        (np.float64(21.0), 21),
        ("21", 21),
        (" 21 ", 21),
        ("21.0", 21),
        ("2.1e1", 21),
        (Decimal("21"), 21),
        (Fraction(21, 1), 21),
        (np.array(21.0), 21),  # 0-d arrays are number-like
    ],
)
def test_integral_values_become_plain_ints(raw, expected):
    for strict in (False, True):
        result = normalize_tag(raw, strict=strict)
        assert result == expected
        assert type(result) is int


@pytest.mark.parametrize("big", [2**53 + 1, 2**63 - 1, 2**64 + 5, 10**30])
def test_large_integers_are_exact(big):
    # a float round trip would round these to the nearest multiple of a power of two
    assert normalize_tag(big) == big
    assert normalize_tag(np.int64(big) if big < 2**63 else big) == big
    assert normalize_tag(str(big)) == big


@pytest.mark.parametrize(
    "raw", [None, "", "  ", "None", "none", " NaN ", float("nan"), np.nan, math.inf, -math.inf]
)
def test_missing_and_non_finite_values_are_none_in_both_modes(raw):
    assert normalize_tag(raw) is None
    assert normalize_tag(raw, strict=True) is None


@pytest.mark.parametrize("raw", [21.5, "21.5", "a", "ArUcoTag#21", [21], (21,), object(), b"x"])
def test_other_values_are_returned_unchanged_unless_strict(raw):
    assert normalize_tag(raw) is raw
    assert normalize_tag(raw, strict=True) is None


def test_non_strict_keeps_non_numeric_labels_used_by_tests():
    assert normalize_tag("a") == "a"
    assert normalize_tag(2.5) == 2.5


def test_objects_whose_float_conversion_fails_are_handled():
    class Broken:
        def __float__(self):
            msg = "no"
            raise ValueError(msg)

    broken = Broken()
    assert normalize_tag(broken) is broken
    assert normalize_tag(broken, strict=True) is None


def test_value_too_large_for_a_float_is_not_a_tag_instead_of_overflowing():
    huge = Fraction(10**400)  # float(huge) raises OverflowError
    assert normalize_tag(huge) is huge
    assert normalize_tag(huge, strict=True) is None


@pytest.mark.parametrize("raw", [21, 21.0, np.int32(21), np.float64(21.0), "21", "21.0", " 21 "])
def test_require_tag_returns_a_plain_int(raw):
    assert require_tag(raw) == 21
    assert type(require_tag(raw)) is int


@pytest.mark.parametrize("raw", [None, "", float("nan"), math.inf, 21.5, "a", [21]])
def test_require_tag_refuses_anything_else_instead_of_truncating(raw):
    with pytest.raises(ValueError, match="not a valid tag id"):
        require_tag(raw)


@pytest.mark.parametrize("raw", [21, 21.0, np.int32(21), np.float64(21.0), "21", "21.0"])
def test_tag_name_is_always_an_integer_name(raw):
    assert tag_name(raw) == f"{TAG_PREFIX}21" == "ArUcoTag#21"


@pytest.mark.parametrize("raw", [None, float("nan"), 2.5, "a", ""])
def test_tag_name_rejects_invalid_ids(raw):
    with pytest.raises(ValueError, match="not a valid tag id"):
        tag_name(raw)
