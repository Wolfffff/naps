"""Parity of the incremental cost matrix with the legacy pandas implementation (seeded)."""

from __future__ import annotations

import numpy as np
import pytest

from naps.cost_matrix import CostMatrix
from naps.rolling_votes import WEIGHT_EPSILON, RollingVotes

from .legacy_cost_matrix import LegacyCostMatrix


def _frozen(matched):
    """``matched`` as nested lists, so the comparison also covers every insertion order."""
    return [(frame, list(row.items())) for frame, row in matched.items()]


def _random_votes(seed, n_frames, n_tracks, n_tags, *, weighted=False, churn=False, strings=False):
    """Random ``{frame: {track: [entry]}}`` with ties, empty and None-only tracks, track churn."""
    rng = np.random.default_rng(seed)
    track_names = [f"t{i}" if strings else 100 + i for i in range(n_tracks)]
    fav = rng.integers(0, n_tags, n_tracks)
    votes = {}
    for frame in range(n_frames):
        row = {}
        for k, track in enumerate(track_names):
            if churn and ((frame + 7 * k) // 17) % 3 == 0:
                continue  # the track is absent from this frame
            entries = []
            for _ in range(int(rng.integers(0, 4))):
                u = rng.random()
                if u < 0.15:
                    entries.append(None)
                    continue
                tag = int(fav[k]) if u < 0.55 else int(rng.integers(0, n_tags))
                if churn and (frame // 31) % 2 == k % 2:
                    tag = int(rng.integers(0, n_tags))  # tags drop out and come back
                if weighted:
                    w = float(rng.choice([1.0, 0.5, 0.25, 0.1, 0.3, 2.0]))
                    entries.append((tag, w))
                    if u > 0.97:
                        entries.append((None, 1.0))
                        entries.append((tag, None))
                else:
                    entries.append(tag)
            row[track] = entries
        votes[frame] = row
    return votes


CASES = [
    {"n_frames": 120, "n_tracks": 4, "n_tags": 4},
    {"n_frames": 150, "n_tracks": 9, "n_tags": 12},
    {"n_frames": 100, "n_tracks": 16, "n_tags": 6, "churn": True},
    {"n_frames": 100, "n_tracks": 5, "n_tags": 5, "weighted": True},
    {"n_frames": 100, "n_tracks": 7, "n_tags": 9, "weighted": True, "churn": True},
    {"n_frames": 90, "n_tracks": 6, "n_tags": 4, "strings": True, "churn": True},
]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("half", [1, 3, 8])
@pytest.mark.parametrize("strict", [False, True])
def test_matches_legacy_exactly(seed, case, half, strict):
    votes = _random_votes(seed, **CASES[case])
    first = 3 if seed % 2 else 0
    last = len(votes) - 1
    new = CostMatrix.fromDict(votes, first, last, half, strict_carry_forward=strict)
    old = LegacyCostMatrix.fromDict(votes, first, last, half, strict_carry_forward=strict)
    assert _frozen(new.assignTrackTagPairs()) == _frozen(old.assignTrackTagPairs())


def test_ties_are_broken_like_the_legacy_code():
    # four tracks voting for the same single tag, and tags nobody voted for after removal:
    # everything is a tie, so any change of row/column order would change the output.
    votes = {f: {k: [1] if f % 2 else [1, 2] for k in "abcd"} for f in range(60)}
    new = CostMatrix.fromDict(votes, 0, 59, 4).assignTrackTagPairs()
    old = LegacyCostMatrix.fromDict(votes, 0, 59, 4).assignTrackTagPairs()
    assert _frozen(new) == _frozen(old)


def test_zero_weight_votes_keep_their_row_position():
    # a cell can be present with value 0 (weight 0.0); it still fixes the row order
    votes = {f: {"a": [(7, 0.0), (8, 1.0)], "b": [(8, 1.0), (7, 1.0)]} for f in range(30)}
    new = CostMatrix.fromDict(votes, 0, 29, 3).assignTrackTagPairs()
    old = LegacyCostMatrix.fromDict(votes, 0, 29, 3).assignTrackTagPairs()
    assert _frozen(new) == _frozen(old)


def test_empty_window_returns_nothing():
    window = RollingVotes(CostMatrix._linearAssignment, CostMatrix._linearAssignment)
    assert window.assign({"a": 1}) == []
    window.add({"a": [None], "b": []})
    assert window.assign({"a": 1, "b": 1}) == []


def test_custom_solver_sees_the_float_array_every_time():
    seen = []

    def solver(array):
        seen.append(array)
        return iter(())

    window = RollingVotes(solver, CostMatrix._linearAssignment)
    window.add({"a": [3, 4], "b": [4]})
    assert window.assign({"a": 0, "b": 0}) == []
    assert window.assign({"a": 0, "b": 0}) == []
    assert len(seen) == 2 and seen[0].dtype == np.float64 and seen[0].shape == (2, 2)
    assert window.n_reused == 0  # a custom solver may be stateful: never cached


def test_identical_arrays_reuse_the_last_solution():
    window = RollingVotes(CostMatrix._linearAssignment, CostMatrix._linearAssignment)
    window.add({"a": [3], "b": [4]})
    first = window.assign({"a": 0, "b": 0})
    again = window.assign({"a": 0, "b": 0})
    assert first == again == [("a", 3), ("b", 4)]
    assert (window.n_solved, window.n_reused) == (1, 1)
    window.add({"a": [3]})  # different weights: solve again
    window.assign({"a": 0, "b": 0})
    assert window.n_solved == 2
    # same shape, different values, then the shape changes
    window.assign({"a": 0})
    assert window.n_solved == 3


def test_removing_an_unknown_vote_creates_it_like_the_legacy_code():
    window = RollingVotes(CostMatrix._linearAssignment)
    window.remove({"a": [5, None, (6, 1.0), (None, 1.0), (7, None)]})
    # the legacy `+=` on a missing key created +1; it is not near zero, so it stays
    assert window.assign({"a": 0}) == [("a", 5)]
    tiny = RollingVotes(CostMatrix._linearAssignment)
    tiny.remove({"a": [(5, WEIGHT_EPSILON / 10)]})
    assert tiny.assign({"a": 0}) == []


def test_cancelling_weights_drop_the_cell():
    window = RollingVotes(CostMatrix._linearAssignment)
    window.add({"a": [(1, 0.1), (1, 0.25)]})
    window.remove({"a": [(1, 0.1), (1, 0.25)]})
    assert window.assign({"a": 0}) == []
    window.add({"a": [(2, 1)]})
    assert window.assign({"a": 0}) == [("a", 2)]


def test_zero_weight_only_tag_gets_no_row():
    window = RollingVotes(CostMatrix._linearAssignment)
    window.add({"a": [(7, 0.0), (8, 1.0)]})  # tag 7 is present in "a" but all-zero: no row
    assert window.assign({"a": 0}) == [("a", 8)]
