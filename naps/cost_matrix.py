"""Rolling-window track/tag assignment (the ``CostMatrix``).

Every video frame carries *votes* ``track -> [tag, ...]`` (ArUco decodes, optionally
weighted extra evidence, see :mod:`naps.evidence`). For each frame the votes inside a
window of ``2 * half_rolling_window_size + 1`` frames centred on it are accumulated into a
track x tag cost matrix (a vote is a *negative* cost) and the optimal one-to-one assignment
is found with the Jonker-Volgenant algorithm. A carry-forward pass then extends identities
into the frames the window cannot reach.

Abstention semantics: a track is only ever given a tag that received at least one vote inside
its window, a tag is used at most once per window, and a track without any vote in its window
stays unassigned (it is *not* guessed from its neighbours).

Known limitation: the window needs ``half_rolling_window_size`` frames on each side, so the
first ``half_rolling_window_size`` frames of a job are never assigned (the carry-forward pass
only moves forward in time) and the last ``half_rolling_window_size`` frames only receive a
tag by carry-forward.
"""

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from naps.rolling_votes import RollingVotes

__all__ = ["CostMatrix", "MatchedDict", "VoteEntry", "iter_votes"]

# One entry of a vote list: a bare tag (weight 1), a ``(tag, weight)`` pair or ``None`` (no decode).
VoteEntry = Hashable | tuple[Hashable, float] | None
# ``votes[frame][track] -> [entry, ...]``
Votes = Mapping[int, Mapping[Hashable, Sequence[VoteEntry]]]
# ``matched[frame][track] -> tag``; a missing track reads as ``""`` (legacy ``defaultdict(str)``).
MatchedDict = defaultdict[int, defaultdict[Hashable, Hashable]]


def iter_votes(tag_list: Iterable[VoteEntry]) -> Iterator[tuple[Hashable, float]]:
    """Yield ``(tag, weight)`` from a list of vote entries.

    Args:
        tag_list: Entries that are either a bare tag (weight 1, the legacy format), a
            ``(tag, weight)`` pair, or ``None`` (no decode).

    Yields:
        ``(tag, weight)`` pairs. ``None`` entries, and pairs whose tag or weight is ``None``,
        are skipped.
    """
    for entry in tag_list:
        if entry is None:
            continue
        if isinstance(entry, tuple):
            tag, weight = entry
            if tag is None or weight is None:
                continue
            yield tag, weight
        else:
            yield entry, 1


class CostMatrix:
    """Rolling-window assignment of tags to tracks.

    Args:
        unmatched_dict: ``unmatched_dict[frame][track] -> [vote entry, ...]`` for every frame in
            ``first_frame..last_frame``. A track key present in a frame means "this track exists
            in this frame" even if its vote list is empty or only holds ``None``.
        first_frame: First frame (inclusive) to process.
        last_frame: Last frame (inclusive) to process.
        half_rolling_window_size: Number of frames upstream and downstream of a frame whose votes
            are pooled; the window spans ``2 * half_rolling_window_size + 1`` frames.
        strict_carry_forward: Legacy carry-forward (``False``, default) lets the previous frame's
            tag override a fresh assignment (sticky tags). ``True`` only carries into tracks the
            assignment left unmatched. On the real Beszel Trial2 data the strict variant flipped
            tags 2-8x more often along a SLEAP track than legacy, so legacy stays the default.
        **kwargs: Ignored; accepted for backward compatibility.

    Attributes:
        assignment_method: Callable mapping a (tags x tracks) cost array to ``(track_index,
            tag_index)`` pairs; defaults to the Jonker-Volgenant solver.
    """

    def __init__(
        self,
        unmatched_dict: Votes,
        first_frame: int,
        last_frame: int,
        half_rolling_window_size: int,
        strict_carry_forward: bool = False,
        **kwargs: object,
    ) -> None:
        """Store the arguments; the assignment is computed by :meth:`assignTrackTagPairs`."""
        self.unmatched_dict = unmatched_dict
        self.strict_carry_forward = strict_carry_forward
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.half_rolling_window_size = half_rolling_window_size
        self.assignment_method: Callable[[NDArray[np.float64]], Iterable[tuple[int, int]]] = (
            self._linearAssignment
        )

    @classmethod
    def fromDict(  # noqa: N802 (public camelCase API)
        cls,
        unmatched_dict: Votes,
        first_frame: int,
        last_frame: int,
        half_rolling_window_size: int,
        strict_carry_forward: bool = False,
        **kwargs: object,
    ) -> "CostMatrix":
        """Build a :class:`CostMatrix`; same arguments as the constructor."""
        return cls(
            unmatched_dict,
            first_frame,
            last_frame,
            half_rolling_window_size,
            strict_carry_forward=strict_carry_forward,
            **kwargs,
        )

    def assignTrackTagPairs(self) -> MatchedDict:  # noqa: N802 (public camelCase API)
        """Assign a tag to each track of each frame.

        Returns:
            ``matched[frame][track] -> tag`` with plain Python ``int`` tags (non-numeric labels
            are returned unchanged). Every frame in ``first_frame..last_frame - 1`` has a
            (possibly empty) row; tracks that never received a vote are absent from their row
            (a missing track reads as ``""``). Within one frame no tag appears twice.

        Raises:
            KeyError: If ``unmatched_dict`` lacks a frame in ``first_frame..last_frame``.
        """
        window = RollingVotes(self.assignment_method, default_solver=self._linearAssignment)
        matched_dict: MatchedDict = defaultdict(lambda: defaultdict(str))
        half = self.half_rolling_window_size

        for frame in range(self.first_frame, self.last_frame + 1):
            window.add(self.unmatched_dict[frame])

            # The window around ``match_frame`` is complete once ``frame`` (its upper edge) is in.
            match_frame = frame - half
            if match_frame < self.first_frame + half or match_frame > self.last_frame - half:
                continue

            start_of_window = match_frame - half
            if start_of_window > self.first_frame:
                window.remove(self.unmatched_dict[start_of_window - 1])

            # Drop all-zero tags (rows) and tracks (columns) so they cannot be "assigned"; keep
            # only the tracks of the frame being matched, in first-seen column order.
            for track, tag in window.assign(self.unmatched_dict[match_frame]):
                matched_dict[match_frame][track] = tag

        self._carry_forward(matched_dict)
        return matched_dict

    def _carry_forward(self, matched_dict: MatchedDict) -> None:
        """Extend each frame's assignments into the next frame (in place).

        A tag is carried from frame ``f - 1`` to frame ``f`` when the track exists in ``f`` and
        the tag is not already used in ``f``. In strict mode the track must also be unmatched in
        ``f``; in legacy mode the carried tag may override a fresh assignment (sticky tags).
        """
        for frame in range(self.first_frame + 1, self.last_frame + 1):
            for track, tag in matched_dict[frame - 1].items():
                if (
                    track in self.unmatched_dict[frame]
                    and (not self.strict_carry_forward or track not in matched_dict[frame])
                    and tag not in matched_dict[frame].values()
                ):
                    matched_dict[frame][track] = tag

    @staticmethod
    def _linearAssignment(  # noqa: N802 (name kept for tests/subclasses)
        value_array: NDArray[np.float64],
    ) -> Iterator[tuple[int, int]]:
        """Solve the linear assignment problem (Jonker-Volgenant) for a tags x tracks array.

        Args:
            value_array: Cost array of shape ``(n_tags, n_tracks)``; lower is better.

        Yields:
            ``(track_index, tag_index)`` pairs, ordered by ascending ``tag_index``.
        """
        tag_indices, track_indices = linear_sum_assignment(value_array)
        for track_index, tag_index in zip(track_indices, tag_indices, strict=True):
            yield int(track_index), int(tag_index)
