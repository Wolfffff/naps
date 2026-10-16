"""The reason codes are part of the sidecar format: they must never be renumbered."""

from naps.identity import SlotFlag
from naps.reasons import REASON_NAMES, Reason


def test_codes_are_the_documented_integers():
    assert {r.name: int(r) for r in Reason} == {
        "NONE": 0,
        "GHOST_INSTANCE": 1,
        "DUPLICATE_WITHIN_DEDUPE_PX": 2,
        "EXCESS_INSTANCE": 3,
        "BELOW_MIN_CONFIDENCE": 4,
        "UNTAGGED_BEE": 5,
        "BLOCKED_BY_ANCHOR": 6,
        "CONSISTENCY_DROPPED": 7,
        "UNTRUSTED_WINDOW_WITHHELD": 8,
        "LOW_CONFIDENCE_IN_PILE": 9,
        "NO_ANCHOR_IN_TRACKLET": 10,
        "NO_POSITION": 11,
        "OUTSIDE_REGIONS": 12,
        "REGION_NO_DATA": 13,
        "REGION_UNSUPPORTED_TAGSET": 14,
    }


def test_names_are_lower_case_and_in_code_order():
    assert all(REASON_NAMES[int(r)] == r.name.lower() for r in Reason)
    assert len(REASON_NAMES) == len(Reason)


def test_codes_0_to_8_coincide_with_the_slot_flags():
    # the engine's flags are turned into reasons by identity (low_confidence is refined later)
    pairs = {
        SlotFlag.NONE: Reason.NONE,
        SlotFlag.GHOST: Reason.GHOST_INSTANCE,
        SlotFlag.DUPLICATE_INSTANCE: Reason.DUPLICATE_WITHIN_DEDUPE_PX,
        SlotFlag.EXCESS_INSTANCE: Reason.EXCESS_INSTANCE,
        SlotFlag.LOW_CONFIDENCE: Reason.BELOW_MIN_CONFIDENCE,
        SlotFlag.UNTAGGED_BEE: Reason.UNTAGGED_BEE,
        SlotFlag.BLOCKED_BY_ANCHOR: Reason.BLOCKED_BY_ANCHOR,
        SlotFlag.CONSISTENCY_DROPPED: Reason.CONSISTENCY_DROPPED,
        SlotFlag.WITHHELD: Reason.UNTRUSTED_WINDOW_WITHHELD,
    }
    assert all(int(flag) == int(reason) for flag, reason in pairs.items())
    assert len(pairs) == len(SlotFlag.NAMES)
