"""The source codes are part of the sidecar format: they must never be renumbered."""

from naps.sources import SOURCE_NAMES, Source


def test_codes_are_the_documented_integers():
    assert {s.name: int(s) for s in Source} == {
        "NONE": 0,
        "DECODE": 1,
        "TRACKLET": 2,
        "ELIMINATION": 3,
        "APPEARANCE": 4,
        "INTERPOLATED": 5,
    }


def test_names_are_lower_case_and_in_code_order():
    assert SOURCE_NAMES == (
        "none",
        "decode",
        "tracklet",
        "elimination",
        "appearance",
        "interpolated",
    )
    assert all(SOURCE_NAMES[int(s)] == s.name.lower() for s in Source)


def test_recovered_sources_order_after_decode():
    # the consistency stage treats every code >= TRACKLET as a recovered label
    assert Source.DECODE < Source.TRACKLET < Source.ELIMINATION < Source.APPEARANCE
    assert Source.APPEARANCE < Source.INTERPOLATED
