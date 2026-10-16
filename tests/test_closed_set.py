import numpy as np
import pytest

from naps.closed_set import eliminate_identities

TAGS = [10, 11, 12, 13]


def world(n=100):
    """Four bees on lanes; track k is TAGS[k]. All assigned by default."""
    positions = {f: {k: np.array([50.0 + f, 100.0 * k]) for k in range(4)} for f in range(n)}
    matched = {f: {k: TAGS[k] for k in range(4)} for f in range(n)}
    return matched, positions


def test_assigns_single_missing_tag_to_single_leftover_instance():
    matched, positions = world()
    del matched[50][2]  # tag 12 lost on track 2 in frame 50
    res = eliminate_identities(matched, positions, TAGS)
    assert res.matched[50] == {0: 10, 1: 11, 2: 12, 3: 13}
    assert [(a[0], a[1], a[2]) for a in res.assigned] == [(50, 2, 12)]
    assert res.assigned[0][3] == 1  # reference one frame away
    assert 2 not in matched[50]  # input untouched


def test_recovers_foreign_tag_and_drops_it_if_abstaining():
    matched, positions = world()
    matched[50][2] = 77  # out-of-set decode on the instance that is really tag 12
    res = eliminate_identities(matched, positions, TAGS)
    assert res.matched[50][2] == 12
    # Two bees unassigned -> abstain, and the foreign tag does not survive either
    matched, positions = world()
    matched[50][2] = 77
    del matched[50][1]
    res = eliminate_identities(matched, positions, TAGS)
    assert 2 not in res.matched[50] and 1 not in res.matched[50]
    assert res.abstained["not_exactly_one"] == 1 and res.dropped_foreign == 1


@pytest.mark.parametrize("raw", [12.0, np.float64(12.0), "12", " 12 "])
def test_other_spellings_of_a_closed_set_tag_are_normalised_to_plain_ints(raw):
    matched, positions = world()
    matched[50][2] = raw
    res = eliminate_identities(matched, positions, TAGS)
    assert res.matched[50][2] == 12
    assert type(res.matched[50][2]) is int
    assert res.dropped_foreign == 0 and not res.assigned  # kept, not dropped and re-eliminated


def test_bool_tag_counts_as_the_int_one_and_is_written_as_int():
    positions = {0: {0: (0.0, 0.0), 1: (10.0, 10.0)}}
    res = eliminate_identities({0: {0: True, 1: 2}}, positions, [1, 2])
    assert res.matched[0][0] == 1
    assert type(res.matched[0][0]) is int
    # the same tag spelled two ways is still a duplicate
    res = eliminate_identities({0: {0: 1, 1: True}}, positions, [1, 2])
    assert res.abstained["duplicate_tag"] == 1


def test_unhashable_tags_are_foreign_not_a_crash():
    matched, positions = world()
    matched[50][2] = [12]
    res = eliminate_identities(matched, positions, TAGS)
    assert res.dropped_foreign == 1
    assert res.matched[50][2] == 12  # the real tag is then recovered by elimination


def test_abstains_when_more_than_one_unassigned():
    matched, positions = world()
    del matched[50][2], matched[50][3]
    res = eliminate_identities(matched, positions, TAGS)
    assert res.matched[50] == {0: 10, 1: 11}
    assert res.abstained["not_exactly_one"] == 1 and not res.assigned


def test_abstains_when_untracked_instance_could_be_the_missing_bee():
    matched, positions = world()
    del matched[50][2]
    res = eliminate_identities(matched, positions, TAGS, untracked_counts={50: 1})
    assert 2 not in res.matched[50] and res.abstained["untracked_instance"] == 1


def test_extra_instance_means_two_leftovers_so_abstain():
    matched, positions = world()
    del matched[50][2]
    positions[50][9] = np.array([0.0, 900.0])  # a fifth tracked instance
    res = eliminate_identities(matched, positions, TAGS)
    assert 2 not in res.matched[50] and 9 not in res.matched[50]
    assert res.abstained["not_exactly_one"] == 1


def test_continuity_gate_blocks_distant_instance():
    matched, positions = world()
    del matched[50][2]
    positions[50][2] = np.array([5000.0, 5000.0])  # nowhere near where tag 12 was
    res = eliminate_identities(matched, positions, TAGS, max_jump=120.0)
    assert 2 not in res.matched[50] and res.abstained["discontinuous"] == 1


def test_gap_scales_the_continuity_gate_and_look_limit():
    matched, positions = world()
    for f in range(40, 60):
        del matched[f][2]
    positions[50][2] = positions[50][2] + [400.0, 0.0]  # moved far during a 10 frame gap
    res = eliminate_identities(matched, positions, TAGS, max_jump=120.0, max_look=25)
    rec = {a[0]: a for a in res.assigned}
    assert res.matched[50][2] == 12 and rec[50][3] == 10
    # no reference within max_look -> abstain
    res = eliminate_identities(matched, positions, TAGS, max_jump=120.0, max_look=5)
    assert 2 not in res.matched[50] and res.abstained["no_reference"] >= 1


def test_nan_position_abstains():
    matched, positions = world()
    del matched[50][2]
    positions[50][2] = np.array([np.nan, np.nan])
    res = eliminate_identities(matched, positions, TAGS)
    assert 2 not in res.matched[50] and res.abstained["no_position"] == 1


def test_never_reuses_a_tag_within_a_frame():
    matched, positions = world()
    matched[50][3] = 10  # duplicate of track 0's tag
    res = eliminate_identities(matched, positions, TAGS)
    assert res.abstained["duplicate_tag"] == 1
    for row in res.matched.values():
        assert len(set(row.values())) == len(row)
    assert list(res.matched[50].values()).count(10) == 1


def test_does_not_chain_from_its_own_inferences():
    # Tag 12 missing for frames 30-70. Reference frames are only genuine assignments, so
    # frames farther than max_look from one abstain instead of chaining off earlier inferences.
    matched, positions = world()
    for f in range(30, 71):
        del matched[f][2]
    res = eliminate_identities(matched, positions, TAGS, max_look=5)
    assigned = sorted(a[0] for a in res.assigned)
    assert assigned == list(range(30, 35)) + list(range(66, 71))
    assert not set(range(40, 60)) & set(assigned)
    assert res.abstained["no_reference"] > 0


def test_max_distance_caps_the_gap_scaled_continuity_gate():
    matched, positions = world()
    for f in range(40, 60):  # tag 12 unseen for 20 frames; track 2 wanders 1000 px away meanwhile
        del matched[f][2]
        positions[f][2] = np.array([50.0 + f, 1200.0])
    # uncapped, the gate is max_jump * gap = 120 * 10 = 1200 px and the instance is accepted
    res = eliminate_identities(matched, positions, TAGS, max_jump=120.0)
    assert any(a[1] == 2 and a[2] == 12 for a in res.assigned)
    # with an absolute cap it is refused
    res = eliminate_identities(matched, positions, TAGS, max_jump=120.0, max_distance=400.0)
    assert not [a for a in res.assigned if a[1] == 2 and a[0] >= 45]
    assert res.abstained["discontinuous"] >= 1


@pytest.mark.parametrize("bad", [None, 21.5, "x", float("nan")])
def test_a_malformed_closed_set_is_refused_not_truncated(bad):
    with pytest.raises(ValueError, match="not a valid tag id"):
        eliminate_identities({}, {}, [10, bad])


def test_empty_input_gives_empty_result():
    res = eliminate_identities({}, {}, TAGS)
    assert not res.matched
    assert not res.assigned and not res.abstained and res.dropped_foreign == 0


def test_blank_assignments_are_neither_foreign_nor_kept():
    matched, positions = world()
    matched[50][2] = ""  # the cost matrix's "unassigned" default
    matched[51][2] = None
    res = eliminate_identities(matched, positions, TAGS)
    assert res.matched[50][2] == 12 and res.matched[51][2] == 12
    assert res.dropped_foreign == 0


def test_float_and_numpy_tags_match_the_closed_set():
    matched, positions = world()
    matched[50][2] = 12.0
    matched[51][2] = np.int64(12)
    res = eliminate_identities(matched, positions, TAGS)
    assert not res.assigned and res.dropped_foreign == 0
    assert res.matched[50][2] == 12.0 and res.matched[51][2] == 12


def test_reference_interpolates_between_both_sides_and_uses_the_nearer_gap():
    matched, positions = world()
    for f in range(40, 56):  # frames 39 and 56 are the references: gaps 1..8
        del matched[f][2]
    res = eliminate_identities(matched, positions, TAGS)
    gaps = {a[0]: a[3] for a in res.assigned}
    assert gaps[40] == 1 and gaps[47] == 8 and gaps[48] == 8 and gaps[55] == 1
    assert all(res.matched[f][2] == 12 for f in range(40, 56))


@pytest.mark.parametrize("missing", [range(10), range(90, 100)])
def test_reference_from_one_side_only_at_the_edges(missing):
    matched, positions = world()
    for f in missing:
        del matched[f][2]
    res = eliminate_identities(matched, positions, TAGS)
    assert sorted(a[0] for a in res.assigned) == list(missing)
    assert all(a[3] >= 1 for a in res.assigned)


def test_tag_never_assigned_has_no_reference():
    matched, positions = world()
    for f in matched:
        del matched[f][2]
    res = eliminate_identities(matched, positions, TAGS)
    assert not res.assigned
    assert res.abstained["no_reference"] == len(matched)


def test_positions_without_assignments_abstain():
    _, positions = world(5)
    res = eliminate_identities({}, positions, TAGS)
    assert res.abstained["not_exactly_one"] == 5
    assert all(not row for row in res.matched.values())


def test_assignments_without_positions_are_kept_and_nothing_is_inferred():
    matched, _ = world(5)
    del matched[2][1]
    res = eliminate_identities(matched, {}, TAGS)
    assert res.matched[2] == {0: 10, 2: 12, 3: 13}
    assert not res.assigned and not res.abstained


def test_duplicate_tag_keeps_the_first_track_and_reports_the_frame():
    matched, positions = world()
    matched[50][3] = 10
    res = eliminate_identities(matched, positions, TAGS)
    assert res.abstained_frames == {50: "duplicate_tag"}
    assert res.matched[50] == {0: 10, 1: 11, 2: 12}


def test_input_is_not_modified_and_the_result_is_deterministic():
    matched, positions = world()
    del matched[50][2]
    matched[60][1] = 99
    snapshot = {f: dict(row) for f, row in matched.items()}
    first = eliminate_identities(matched, positions, TAGS)
    second = eliminate_identities(matched, positions, TAGS)
    assert {f: dict(row) for f, row in matched.items()} == snapshot
    assert first.assigned == second.assigned
    assert {f: dict(r) for f, r in first.matched.items()} == {
        f: dict(r) for f, r in second.matched.items()
    }


def test_positions_may_be_plain_sequences():
    matched, positions = world()
    positions = {f: {k: tuple(p) for k, p in row.items()} for f, row in positions.items()}
    del matched[50][2]
    res = eliminate_identities(matched, positions, TAGS)
    assert res.matched[50][2] == 12


def test_unknown_position_of_the_leftover_instance_abstains():
    matched, positions = world()
    del matched[50][2]
    positions[50][2] = None
    res = eliminate_identities(matched, positions, TAGS)
    assert res.abstained["no_position"] == 1


def test_internal_invariant_violation_is_reported(monkeypatch):
    def duplicating(_in_frame, _closed, out, _result):
        out[0] = 10
        out[1] = 10
        return set(), False

    monkeypatch.setattr("naps.closed_set._keep_closed_set_tags", duplicating)

    with pytest.raises(RuntimeError, match="tag reused within frame 0"):
        eliminate_identities({0: {0: 10}}, {0: {0: (0.0, 0.0)}}, [10, 11])
