"""Closed-set identity elimination.

Every quadrant holds a known set of N tagged animals. In a frame where N-1 of
the expected tags are assigned, exactly one tag is unassigned and exactly one
SLEAP instance is unassigned, that instance must carry the missing tag. The
step is a pure post-pass over the assignments produced by
:class:`naps.cost_matrix.CostMatrix` and is deliberately fail-safe: whenever the
frame is not exactly that situation, or the candidate is not spatially
continuous with where the missing tag was last/next seen, it abstains.

Abstention reasons (keys of :attr:`EliminationResult.abstained`): ``duplicate_tag`` (the input
used a tag twice in the frame), ``untracked_instance``, ``not_exactly_one``, ``no_position``,
``no_reference`` and ``discontinuous``.
"""

import logging
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from naps.cost_matrix import MatchedDict
from naps.tags import normalize_tag, require_tag

__all__ = ["EliminationResult", "eliminate_identities"]

logger = logging.getLogger(__name__)

# ``positions[frame][track] -> (x, y)``
Positions = Mapping[int, Mapping[Hashable, Sequence[float]]]
# tag -> (frames, xy): the sorted frames in which the tag was assigned and the track position then
_Trajectory = tuple[NDArray[np.int64], NDArray[np.float64]]


@dataclass
class EliminationResult:
    """Outcome of :func:`eliminate_identities`.

    Attributes:
        matched: ``matched[frame][track] -> tag``; same shape as the cost-matrix output.
        assigned: ``(frame, track, tag, gap, dist)`` of every instance that was assigned by
            elimination; ``gap`` is the frame distance to the reference position and ``dist``
            the pixel distance to it.
        abstained: Number of frames per abstention reason.
        dropped_foreign: Number of out-of-set tags removed from the output.
        abstained_frames: ``{frame: abstention reason}``.
    """

    matched: MatchedDict
    assigned: list[tuple[int, Hashable, int, int, float]] = field(default_factory=list)
    abstained: Counter[str] = field(default_factory=Counter)
    dropped_foreign: int = 0
    abstained_frames: dict[int, str] = field(default_factory=dict)


def _as_xy(pos: Sequence[float] | None) -> NDArray[np.float64] | None:
    """Return ``pos`` as a finite ``(2,)`` float array, or None if it is missing/NaN/malformed."""
    if pos is None:
        return None
    arr = np.asarray(pos, dtype=float)
    if arr.shape != (2,) or not np.isfinite(arr).all():
        return None
    return arr


def _tag_trajectories(
    matched: Mapping[int, Mapping[Hashable, Hashable]], positions: Positions, closed: set[int]
) -> dict[int, _Trajectory]:
    """Map each closed-set tag to the frames and positions where it was genuinely assigned."""
    rows: defaultdict[int, list[tuple[int, NDArray[np.float64]]]] = defaultdict(list)
    for frame in sorted(matched):
        for track, raw_tag in matched[frame].items():
            tag = normalize_tag(raw_tag, strict=True)
            if tag is not None and tag in closed:
                pos = _as_xy(positions.get(frame, {}).get(track))
                if pos is not None:
                    rows[tag].append((frame, pos))
    return {
        tag: (np.array([f for f, _ in items]), np.array([p for _, p in items]))
        for tag, items in rows.items()
    }


def _reference(
    traj: _Trajectory | None, frame: int, max_look: int
) -> tuple[NDArray[np.float64], int] | None:
    """Interpolated/nearest position of a tag around ``frame`` and the frame gap to it.

    Args:
        traj: The tag's trajectory, or None if it was never assigned.
        frame: The frame (in which the tag was *not* assigned) to find a reference for.
        max_look: Maximum number of frames to look back/forward.

    Returns:
        ``(xy, gap)``, or None if the tag has no assignment within ``max_look`` frames. With
        assignments on both sides the position is linearly interpolated and ``gap`` is the
        nearer distance; with one side only, that side's position is used.
    """
    if traj is None:
        return None
    frames, xy = traj
    i = int(np.searchsorted(frames, frame))
    prev_i = i - 1 if i > 0 else None
    # frames[i] may equal ``frame`` only if the tag was assigned here, which it was not
    next_i = i if i < len(frames) else None
    if prev_i is not None and frame - frames[prev_i] > max_look:
        prev_i = None
    if next_i is not None and frames[next_i] - frame > max_look:
        next_i = None
    if prev_i is not None and next_i is not None:
        # frames[next_i] > frame > frames[prev_i], so the denominator is positive
        f = (frame - frames[prev_i]) / (frames[next_i] - frames[prev_i])
        gap = min(frame - frames[prev_i], frames[next_i] - frame)
        return xy[prev_i] * (1 - f) + xy[next_i] * f, int(gap)
    if prev_i is not None:
        j = prev_i
    elif next_i is not None:
        j = next_i
    else:
        return None
    return xy[j], int(abs(frame - frames[j]))


def _keep_closed_set_tags(
    in_frame: Mapping[Hashable, Hashable],
    closed: set[int],
    out: defaultdict[Hashable, Hashable],
    result: EliminationResult,
) -> tuple[set[int], bool]:
    """Copy the closed-set tags of one frame into ``out`` and drop the rest.

    Tags are normalised with :func:`naps.tags.normalize_tag` before the membership test, so
    ``21.0`` or ``"21"`` count as tag 21 and are written to ``out`` as the plain ``int``.
    A tag is never reused within a frame: only its first track keeps it.

    Returns:
        ``(used tags, whether a duplicate tag was seen)``. Out-of-set tags are counted in
        ``result.dropped_foreign``.
    """
    used: set[int] = set()
    duplicate = False
    for track, raw_tag in in_frame.items():
        tag = normalize_tag(raw_tag, strict=True)
        if tag is not None and tag in closed:
            if tag in used:
                duplicate = True
                continue
            used.add(tag)
            out[track] = tag
        elif raw_tag != "" and raw_tag is not None:
            result.dropped_foreign += 1
    return used, duplicate


def _abstain(result: EliminationResult, frame: int, reason: str) -> None:
    """Record that elimination abstained in ``frame``."""
    result.abstained[reason] += 1
    result.abstained_frames[frame] = reason


def _check_continuity(
    frame: int,
    tag: int,
    pos: Sequence[float] | None,
    trajs: Mapping[int, _Trajectory],
    *,
    max_jump: float,
    max_look: int,
    max_distance: float | None,
) -> tuple[int, float] | str:
    """Decide whether ``pos`` is spatially continuous with where ``tag`` was seen.

    Returns:
        ``(gap, distance)`` if it is, else the abstention reason (``no_position``,
        ``no_reference`` or ``discontinuous``).
    """
    xy = _as_xy(pos)
    if xy is None:
        return "no_position"
    reference = _reference(trajs.get(tag), frame, max_look)
    if reference is None:
        return "no_reference"
    ref, gap = reference
    dist = float(np.linalg.norm(xy - ref))
    limit = max_jump * max(gap, 1)
    if max_distance is not None:
        limit = min(limit, max_distance)
    if dist > limit:
        return "discontinuous"
    return gap, dist


def eliminate_identities(  # noqa: PLR0917 (public API: tuning parameters stay positional-capable)
    matched: Mapping[int, Mapping[Hashable, Hashable]],
    positions: Positions,
    closed_set: Iterable[int],
    max_jump: float = 120.0,
    max_look: int = 25,
    untracked_counts: Mapping[int, int] | None = None,
    max_distance: float | None = None,
) -> EliminationResult:
    """Assign the single missing tag to the single unassigned instance of a frame.

    Args:
        matched: ``matched[frame][track] -> tag`` from the cost matrix. Not modified.
        positions: ``positions[frame][track] -> (x, y)`` for every SLEAP track present.
        closed_set: The expected tags of the quadrant.
        max_jump: Continuity gate; the instance must lie within ``max_jump * gap``
            pixels of the missing tag's reference position, where ``gap`` is the
            distance in frames to the nearest frames in which that tag was assigned.
        max_look: Frames searched on each side for the reference position. No
            reference within this range means abstain.
        untracked_counts: ``{frame: n}`` of SLEAP instances without a track; a
            frame with any is skipped (an unseen instance could be the missing bee).
        max_distance: Optional absolute cap (pixels) on the continuity threshold, which
            otherwise grows without bound with the gap (``max_jump * 17`` frames is
            already 2000 px). None (default) = no cap.

    Returns:
        An :class:`EliminationResult`. Tags outside the closed set are removed from the output
        (counted in ``dropped_foreign``); a tag is never present twice in a frame. Reference
        positions only come from genuine cost-matrix assignments, never from earlier
        eliminations (no chaining).

    Raises:
        ValueError: If an entry of ``closed_set`` is not a valid tag id (``None``, NaN,
            non-integral, non-numeric).
        RuntimeError: If the output would reuse a tag within a frame (an internal invariant).
    """
    closed = {require_tag(t) for t in closed_set}
    untracked_counts = untracked_counts or {}
    trajs = _tag_trajectories(matched, positions, closed)
    result_matched: MatchedDict = defaultdict(lambda: defaultdict(str))
    result = EliminationResult(result_matched)

    for frame in sorted(set(matched) | set(positions)):
        present = positions.get(frame, {})
        out = result_matched[frame]
        used, duplicate = _keep_closed_set_tags(matched.get(frame, {}), closed, out, result)
        if duplicate:
            _abstain(result, frame, "duplicate_tag")
            continue

        missing = closed - used
        leftover = [t for t in present if t not in out]
        if not missing or not leftover:
            continue
        if untracked_counts.get(frame, 0):
            _abstain(result, frame, "untracked_instance")
            continue
        if len(missing) != 1 or len(leftover) != 1:
            _abstain(result, frame, "not_exactly_one")
            continue

        tag, track = next(iter(missing)), leftover[0]
        outcome = _check_continuity(
            frame,
            tag,
            present[track],
            trajs,
            max_jump=max_jump,
            max_look=max_look,
            max_distance=max_distance,
        )
        if isinstance(outcome, str):
            _abstain(result, frame, outcome)
            continue
        gap, dist = outcome
        out[track] = tag
        result.assigned.append((frame, track, tag, gap, dist))

    for frame, row in result_matched.items():
        tags = list(row.values())
        if len(tags) != len(set(tags)):
            msg = f"tag reused within frame {frame}"
            raise RuntimeError(msg)
    logger.info(
        "Closed-set elimination assigned %s instances; abstained %s; dropped %s foreign tags",
        len(result.assigned),
        dict(result.abstained),
        result.dropped_foreign,
    )
    return result
