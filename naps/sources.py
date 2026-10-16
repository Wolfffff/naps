"""Source codes of the labels produced by the joint identity engine.

Every label of a bee-frame says *why* it was given. The integer codes are stored in the identity
sidecar (``source``), so they are part of the output format and must not be renumbered.
"""

from __future__ import annotations

from enum import IntEnum

__all__ = ["SOURCE_NAMES", "Source"]


class Source(IntEnum):
    """Why an instance carries its tag.

    Attributes:
        NONE: No label (the instance is left untracked / the tag is absent in this frame).
        DECODE: The tag was decoded on this instance (an immutable anchor).
        TRACKLET: Inferred: the instance's tracklet carries decoded anchors of this tag.
        ELIMINATION: Inferred by the closed set alone (the other tags are all taken elsewhere).
        APPEARANCE: Inferred and only resolved by external appearance evidence.
        INTERPOLATED: No instance; the position was interpolated across a short gap.
    """

    NONE = 0
    DECODE = 1
    TRACKLET = 2
    ELIMINATION = 3
    APPEARANCE = 4
    INTERPOLATED = 5


#: Names in code order (``SOURCE_NAMES[int(Source.DECODE)] == "decode"``).
SOURCE_NAMES = tuple(s.name.lower() for s in Source)
