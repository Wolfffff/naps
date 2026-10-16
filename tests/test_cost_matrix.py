import numpy as np
import pandas as pd
import pytest

from naps.cost_matrix import CostMatrix, iter_votes


def _constant_votes(n_frames, tracks):
    """``{frame: {track: [tag]}}`` where each track always votes for its own tag."""
    return {f: {track: [tag] for track, tag in tracks.items()} for f in range(n_frames)}


def test_cost_matrix_basic_assignment():
    value_dict = {
        0: {"a": [1, 2], "b": [2], "c": [1, 3]},
        1: {"a": [3], "b": [1, 2], "c": [1, 3]},
        2: {"a": [3], "b": [1, 2, 3], "c": [3]},
        3: {"a": [2], "b": [1, 2, 3], "c": []},
        4: {"a": [2], "b": [1, 2, 3], "c": []},
        5: {"a": [1, 2], "b": [2], "c": [1, 3]},
    }

    assignment_dict = CostMatrix.fromDict(value_dict, 0, 5, 2).assignTrackTagPairs()

    assert 2 in assignment_dict
    assert sorted(assignment_dict) == [0, 1, 2, 3, 4, 5]
    assert list(assignment_dict[2]) == ["b", "a", "c"]
    assert list(assignment_dict[3]) == ["b", "a", "c"]


def test_linear_assignment_returns_track_then_tag_index():
    value_dict = {
        "a": {1: 4, 2: 2, 3: 3},
        "b": {1: 1, 2: 0, 3: 2},
        "c": {1: 3, 2: 5, 3: 2},
    }
    value_array = pd.DataFrame.from_dict(value_dict).fillna(0).to_numpy()

    pairs = list(CostMatrix._linearAssignment(value_array))

    assert [track_index for track_index, _ in pairs] == [1, 0, 2]
    assert [tag_index for _, tag_index in pairs] == [0, 1, 2]  # ordered by ascending tag index
    assert all(type(i) is int for pair in pairs for i in pair)


def test_none_votes_do_not_leak():
    # A track whose frames carry None (no decode) alongside real votes must never
    # produce a None/NaN/float tag, including after the rolling window starts
    # removing old frames (the removal path used to test `tag != -1`, not `is not None`).
    value_dict = {frame: {"a": [None, 4]} for frame in range(30)}

    assignment_dict = CostMatrix.fromDict(value_dict, 0, 29, 3).assignTrackTagPairs()

    tags = [tag for frame in assignment_dict.values() for tag in frame.values()]
    assert tags
    assert all(tag == 4 and type(tag) is int for tag in tags)


def test_carry_forward_does_not_override_fresh_assignment_when_strict():
    # Tag 10 is decoded on the track for 40 frames, then tag 99 for 40 frames.
    # Once the rolling window is well past the switch, the fresh assignment (99)
    # must win; carry-forward may only fill tracks the linear assignment left unmatched.
    value_dict = {frame: {"a": [10] if frame < 40 else [99]} for frame in range(80)}

    assignment_dict = CostMatrix.fromDict(
        value_dict, 0, 79, 3, strict_carry_forward=True
    ).assignTrackTagPairs()

    assert assignment_dict[30]["a"] == 10
    assert assignment_dict[70]["a"] == 99


def test_legacy_carry_forward_is_sticky_by_default():
    # Default (legacy) behaviour: the carried tag may override a fresh assignment. Kept as the
    # default because on real data the strict variant flipped tags 2-8x more often.
    value_dict = {frame: {"a": [10] if frame < 40 else [99]} for frame in range(80)}

    assignment_dict = CostMatrix.fromDict(value_dict, 0, 79, 3).assignTrackTagPairs()

    assert assignment_dict[70]["a"] == 10


def test_head_frames_are_never_assigned_and_tail_is_carried():
    # The window needs `half` frames on each side: frames 0..half-1 get no identity (the
    # carry-forward pass only moves forward in time) and the last `half` frames are carried.
    half, n_frames = 3, 20
    value_dict = _constant_votes(n_frames, {"a": 4})

    assignment_dict = CostMatrix.fromDict(value_dict, 0, n_frames - 1, half).assignTrackTagPairs()

    assert [f for f in range(n_frames) if assignment_dict[f] == {}] == list(range(half))
    assert all(assignment_dict[f] == {"a": 4} for f in range(half, n_frames))


@pytest.mark.parametrize(("first", "last", "half"), [(0, 5, 7), (0, 5, 3), (0, 0, 1), (10, 12, 2)])
def test_job_too_short_for_the_window_assigns_nothing(first, last, half):
    value_dict = {f: {"a": [4]} for f in range(first, last + 1)}

    assignment_dict = CostMatrix.fromDict(value_dict, first, last, half).assignTrackTagPairs()

    assert all(not row for row in assignment_dict.values())


def test_job_just_long_enough_assigns_one_frame_and_carries_it():
    value_dict = {f: {"a": [4]} for f in range(5)}  # exactly 2 * half + 1 frames

    assignment_dict = CostMatrix.fromDict(value_dict, 0, 4, 2).assignTrackTagPairs()

    assert {f: dict(row) for f, row in assignment_dict.items()} == {
        0: {},
        1: {},
        2: {"a": 4},
        3: {"a": 4},
        4: {"a": 4},
    }


def test_all_none_and_empty_votes_abstain():
    value_dict = {f: {"a": [None], "b": [], "c": [None, None]} for f in range(30)}

    assignment_dict = CostMatrix.fromDict(value_dict, 0, 29, 3).assignTrackTagPairs()

    assert all(not row for row in assignment_dict.values())


def test_frame_without_tracks_is_fine():
    value_dict = {f: ({} if 10 <= f < 14 else {"a": [4]}) for f in range(30)}

    assignment_dict = CostMatrix.fromDict(value_dict, 0, 29, 3).assignTrackTagPairs()

    assert assignment_dict[5] == {"a": 4}
    assert assignment_dict[12] == {}  # a track absent from a frame is not carried into it
    assert assignment_dict[20] == {"a": 4}


def test_missing_frame_raises_key_error():
    value_dict = {f: {"a": [4]} for f in range(10) if f != 6}

    with pytest.raises(KeyError):
        CostMatrix.fromDict(value_dict, 0, 9, 2).assignTrackTagPairs()


@pytest.mark.parametrize("first_tracks", [("x", "y"), ("y", "x")])
def test_ties_go_to_the_first_seen_track(first_tracks):
    # Two tracks vote for the same single tag equally often: the optimum is not unique. The
    # winner must be the track seen first, not whichever a set happens to iterate first.
    value_dict = {f: {first_tracks[0]: [7], first_tracks[1]: [7]} for f in range(20)}

    runs = [CostMatrix.fromDict(value_dict, 0, 19, 3).assignTrackTagPairs()[10] for _ in range(3)]

    assert all(run == {first_tracks[0]: 7} for run in runs)


@pytest.mark.parametrize(
    "tag", [np.int64(21), np.int32(21), np.uint8(21), np.float64(21.0), 21.0, 21]
)
def test_numeric_tag_types_come_out_as_plain_ints(tag):
    value_dict = {f: {"a": [tag]} for f in range(20)}

    assignment_dict = CostMatrix.fromDict(value_dict, 0, 19, 3).assignTrackTagPairs()
    tags = [t for row in assignment_dict.values() for t in row.values()]

    assert tags
    assert all(t == 21 and type(t) is int for t in tags)


@pytest.mark.parametrize("tag", [0, 4095, 10**9, 2**31, 2**40])
def test_large_tag_ids(tag):
    value_dict = {f: {"a": [tag]} for f in range(20)}

    row = CostMatrix.fromDict(value_dict, 0, 19, 3).assignTrackTagPairs()[10]

    assert row == {"a": tag}


def test_weighted_and_bare_votes_mix():
    # (tag, weight) pairs, bare tags and None can share a list; 3 x 0.5 beats 1 x 1
    votes = [(5, 0.5), (5, 0.5), (5, 0.5), 6, None, (7, None), (None, 1.0)]
    value_dict = {f: {"a": list(votes)} for f in range(20)}

    row = CostMatrix.fromDict(value_dict, 0, 19, 3).assignTrackTagPairs()[10]

    assert row == {"a": 5}


def test_track_without_votes_stays_unassigned():
    # Track b never votes: it must stay unassigned, not inherit the tag of track a
    value_dict = {f: {"a": [4], "b": []} for f in range(20)}

    assignment_dict = CostMatrix.fromDict(value_dict, 0, 19, 3).assignTrackTagPairs()

    assert all("b" not in row for row in assignment_dict.values())


def test_a_tag_goes_to_one_track_only():
    # Both tracks decode tag 4 (a neighbour's tag in the crop), but track a decodes it more often
    value_dict = {f: {"a": [4, 4], "b": [4]} for f in range(20)}

    row = CostMatrix.fromDict(value_dict, 0, 19, 3).assignTrackTagPairs()[10]

    assert row == {"a": 4}


def test_zero_vote_assignment_is_legacy_behaviour():
    # CHARACTERISATION of a known property, not an endorsement: when a track loses its voted tag
    # to a stronger track and a tag voted for by a track that is absent from the matched frame is
    # left over, the solver pairs them at zero cost. Track b never decoded tag 2.
    value_dict = {f: {"a": [1, 1, 1], "b": [1, 1], "c": [2]} for f in range(20)}
    del value_dict[10]["c"]

    row = CostMatrix.fromDict(value_dict, 0, 19, 3).assignTrackTagPairs()[10]

    assert row == {"a": 1, "b": 2}


def test_custom_assignment_method_is_used():
    cost_matrix = CostMatrix.fromDict(_constant_votes(10, {"a": 4}), 0, 9, 2)
    cost_matrix.assignment_method = lambda _array: iter(())

    assert all(not row for row in cost_matrix.assignTrackTagPairs().values())


def test_extra_keyword_arguments_are_ignored():
    CostMatrix(_constant_votes(3, {"a": 1}), 0, 2, 1, unknown_option=True)


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        ([], []),
        ([None], []),
        ([3, None, 4], [(3, 1), (4, 1)]),
        ([(3, 0.5), (4, 2)], [(3, 0.5), (4, 2)]),
        ([(None, 1.0), (3, None)], []),
        ([np.int64(3)], [(3, 1)]),
    ],
)
def test_iter_votes(entries, expected):
    assert list(iter_votes(entries)) == expected
