"""Incremental rolling-window vote accumulator (the fast core of :mod:`naps.cost_matrix`).

The legacy :class:`~naps.cost_matrix.CostMatrix` kept the window as a ``defaultdict`` of
``defaultdict`` and, for **every** frame, rebuilt a :class:`pandas.DataFrame` from it
(``DataFrame.from_dict``, ``fillna``, two ``.loc`` filters, a column selection and
``to_numpy``), an O(frames x tracks ever seen) job with a large constant factor.

:class:`RollingVotes` keeps exactly the same state (one ordered ``{tag: weight}`` dict per
track) but builds the small dense ``tags x tracks`` array of the window directly, in the row and
column order the DataFrame code produced. ``scipy.optimize.linear_sum_assignment`` breaks ties
by input order, so reproducing that order (and not just the matrix values) is what makes the
output **identical** to the legacy output, including in the many tied cases (for example the
documented zero-vote pairing). The order rules, copied from the DataFrame code path, are:

* columns: tracks in the order in which they first received a vote anywhere in the job;
* rows: tags in the order in which they first appear when walking the tracks in column order and
  each track's own ``{tag: weight}`` dict in insertion order. A tag whose weight returns to zero
  is deleted from its track's dict, so it re-enters at the *end* when it is voted for again;
* all-zero rows and all-zero columns are dropped, then only the tracks present in the frame being
  matched are kept (rows that became all-zero by that selection stay in).

Only tracks with a live (non-empty) dict are visited, so the cost per frame depends on the
number of animals in the window, not on how many tracks the SLEAP file has used in the hour.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from naps.tags import normalize_tag

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from naps.cost_matrix import VoteEntry

__all__ = ["WEIGHT_EPSILON", "RollingVotes"]

#: Accumulated weights smaller than this are treated as zero and dropped from the window, so that
#: fractional weights (0.1 + 0.25 - 0.25 - 0.1 != 0.0 in floating point) cannot leave residue.
WEIGHT_EPSILON = 1e-9

# The solver maps a (tags x tracks) cost array to (track_index, tag_index) pairs.
Solver = Callable[["NDArray[np.float64]"], Iterable[tuple[int, int]]]


class RollingVotes:
    """Votes of the frames currently in the window, as ``{track: {tag: -weight}}``.

    Args:
        solver: Callable mapping a ``(n_tags, n_tracks)`` float array to ``(track_index,
            tag_index)`` pairs. A solver that is ``default_solver`` (identified by
            ``solver is default_solver``) is a pure function of the array, which allows
            :meth:`assign` to reuse the previous answer when the array did not change.
        default_solver: The module's own solver, to recognise it by identity.

    Attributes:
        n_solved: Number of times the solver was actually called.
        n_reused: Number of times the previous solution was reused for an identical array.
    """

    def __init__(self, solver: Solver, default_solver: Solver | None = None) -> None:
        """Create an empty window."""
        self._solver = solver
        self._cacheable = default_solver is not None and solver is default_solver
        self._cells: dict[Hashable, dict[Hashable, float]] = {}
        self._rank: dict[Hashable, int] = {}
        self._last_array: NDArray[np.float64] | None = None
        self._last_pairs: list[tuple[int, int]] = []
        self.n_solved = 0
        self.n_reused = 0

    # ------------------------------------------------------------------ window updates
    def add(self, frame_votes: Mapping[Hashable, Sequence[VoteEntry]]) -> None:
        """Add one frame to the window: every vote lowers the cost of its track/tag pair."""
        cells = self._cells
        for track, entries in frame_votes.items():
            inner = None
            for entry in entries:
                if entry is None:
                    continue
                if isinstance(entry, tuple):
                    tag, weight = entry
                    if tag is None or weight is None:
                        continue
                else:
                    tag, weight = entry, 1
                if inner is None:
                    inner = cells.get(track)
                    if inner is None:
                        inner = cells[track] = {}
                        self._rank.setdefault(track, len(self._rank))
                inner[tag] = inner.get(tag, 0) - weight

    def remove(self, frame_votes: Mapping[Hashable, Sequence[VoteEntry]]) -> None:
        """Remove one frame from the window (mirror of :meth:`add`)."""
        cells = self._cells
        for track, entries in frame_votes.items():
            for entry in entries:
                if entry is None:
                    continue
                if isinstance(entry, tuple):
                    tag, weight = entry
                    if tag is None or weight is None:
                        continue
                else:
                    tag, weight = entry, 1
                inner = cells.get(track)
                if inner is None:
                    inner = cells[track] = {}
                    self._rank.setdefault(track, len(self._rank))
                value = inner.get(tag, 0) + weight
                if abs(value) < WEIGHT_EPSILON:
                    inner.pop(tag, None)
                else:
                    inner[tag] = value
                if not inner:
                    del cells[track]

    # ------------------------------------------------------------------ assignment
    def assign(self, present_tracks: Mapping[Hashable, object]) -> list[tuple[Hashable, Hashable]]:
        """Solve the assignment for the current window.

        Args:
            present_tracks: The tracks that exist in the frame being matched; others are not
                candidates.

        Returns:
            ``(track, normalised tag)`` pairs in solver order (ascending tag index).
        """
        cells = self._cells
        rank = self._rank
        order = sorted(cells, key=rank.__getitem__)

        # rows in first-seen order; a row is kept if any weight in it is non-zero
        row_nonzero: dict[Hashable, bool] = {}
        columns: list[Hashable] = []
        for track in order:
            inner = cells[track]
            column_nonzero = False
            for tag, value in inner.items():
                if value != 0:
                    column_nonzero = True
                    row_nonzero[tag] = True
                else:
                    row_nonzero.setdefault(tag, False)
            if column_nonzero and track in present_tracks:
                columns.append(track)
        rows = [tag for tag, nonzero in row_nonzero.items() if nonzero]

        row_index = {tag: i for i, tag in enumerate(rows)}
        array = np.zeros((len(rows), len(columns)), dtype=np.float64)
        for j, track in enumerate(columns):
            for tag, value in cells[track].items():
                i = row_index.get(tag)
                if i is not None:
                    array[i, j] = value

        pairs = self._solve(array)
        # normalise: a float index must never become "ArUcoTag#21.0"
        return [
            (columns[track_index], normalize_tag(rows[tag_index]))
            for track_index, tag_index in pairs
        ]

    def _solve(self, array: NDArray[np.float64]) -> list[tuple[int, int]]:
        """Run the solver, or reuse its previous answer for an identical array."""
        if self._cacheable:
            last = self._last_array
            if last is not None and last.shape == array.shape and np.array_equal(last, array):
                self.n_reused += 1
                return self._last_pairs
        pairs = [(int(t), int(g)) for t, g in self._solver(array)]
        self.n_solved += 1
        if self._cacheable:
            self._last_array = array
            self._last_pairs = pairs
        return pairs
