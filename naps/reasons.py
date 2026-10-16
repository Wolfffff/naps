"""Reason codes: why an instance of the joint identity engine carries no tag.

Every instance that holds a position but ends up without a label gets exactly one reason in the
identity sidecar (``reason``, names in ``reason_names``) and in the report. The integer codes are
part of the output format: **codes are never renumbered or reused**, new reasons are appended.
Codes ``0..8`` coincide with :class:`naps.identity.SlotFlag` (code 4 is that flag's
``low_confidence``, refined below); everything after it is determined from information the engine
already has.

A reason is only given for what the code can determine; a label that is missing for any other
reason (there is none today) would show up as reason ``none`` with ``tag == -1``, which the tests
forbid.
"""

from __future__ import annotations

from enum import IntEnum

__all__ = ["REASON_NAMES", "Reason"]


class Reason(IntEnum):
    """Why a valid instance has no tag.

    Attributes:
        NONE: The instance is labelled, or the slot holds no instance.
        GHOST_INSTANCE: A duplicate detection of a decoded neighbour (an unclaimed instance
            closer than ``ConsistencyParams.ghost_gate_px`` to a decoded one, or to a
            higher-scored unclaimed one, or with an overlapping pose). Also given to a ghost that
            was in addition culled for exceeding the colony size.
        DUPLICATE_WITHIN_DEDUPE_PX: Closer than ``IdentityParams.dedupe_px`` to a higher-ranked
            instance of the frame: the same bee detected twice.
        EXCESS_INSTANCE: More instances than bees in the region in this frame; the lowest-ranked
            (anchored instances rank first, then the SLEAP score) are culled.
        BELOW_MIN_CONFIDENCE: The best posterior of the instance is below ``--min-confidence``
            and neither of the two refinements below applies.
        UNTAGGED_BEE: The model assigns the instance to a bee without a tag (a phantom tag of a
            region with fewer tags than ``--colony-size``).
        BLOCKED_BY_ANCHOR: The model's tag for the instance is already decoded on another
            instance of the frame, so the label is not given twice.
        CONSISTENCY_DROPPED: A recovered label removed by the temporal-consistency stage
            (duplicate tag, teleport or chain veto).
        UNTRUSTED_WINDOW_WITHHELD: A recovered label inside a window whose quality criteria
            failed (``--untrusted-policy withhold``); decoded labels are never withheld.
        LOW_CONFIDENCE_IN_PILE: Like ``BELOW_MIN_CONFIDENCE``, and another bee is within
            ``IdentityParams.nn_near_px`` in this frame (the regime in which the model prices a
            label swap cheaply, so identities are close to exchangeable).
        NO_ANCHOR_IN_TRACKLET: Like ``BELOW_MIN_CONFIDENCE`` (not in a pile), and no decode
            was ever accepted as an anchor anywhere on the instance's tracklet.
        NO_POSITION: The SLEAP instance has no finite node, so it cannot be tracked.
        OUTSIDE_REGIONS: The instance lies in none of the regions (quadrant modes).
        REGION_NO_DATA: The instance's region is ``no_data`` (no instances or no decoded tag).
        REGION_UNSUPPORTED_TAGSET: The instance's region is ``unsupported_tagset`` (not 3 to
            ``--colony-size`` tags, or crowded by other animals).
    """

    NONE = 0
    GHOST_INSTANCE = 1
    DUPLICATE_WITHIN_DEDUPE_PX = 2
    EXCESS_INSTANCE = 3
    BELOW_MIN_CONFIDENCE = 4
    UNTAGGED_BEE = 5
    BLOCKED_BY_ANCHOR = 6
    CONSISTENCY_DROPPED = 7
    UNTRUSTED_WINDOW_WITHHELD = 8
    LOW_CONFIDENCE_IN_PILE = 9
    NO_ANCHOR_IN_TRACKLET = 10
    NO_POSITION = 11
    OUTSIDE_REGIONS = 12
    REGION_NO_DATA = 13
    REGION_UNSUPPORTED_TAGSET = 14


#: Names in code order (``REASON_NAMES[int(Reason.GHOST_INSTANCE)] == "ghost_instance"``).
REASON_NAMES = tuple(r.name.lower() for r in Reason)
