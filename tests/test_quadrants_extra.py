"""Edge cases of naps.quadrants: parsing, region lookup, vote filtering and the report."""

import json

import numpy as np
import pytest

from naps.quadrants import (
    QuadrantScope,
    build_report,
    eliminate_by_quadrant,
    locate,
    make_grid,
    parse_roi_args,
    parse_tag_map,
)

ROIS = {"a": (0.0, 0.0, 10.0, 10.0), "b": (10.0, 0.0, 20.0, 10.0)}


def prepared_scope(tag_sets, positions=None, **kwargs):
    scope = QuadrantScope(tag_sets=tag_sets, rois=dict(ROIS), **kwargs)
    scope.prepare((20, 10), positions or {})
    return scope


# ------------------------------------------------------------------------ regions
@pytest.mark.parametrize(("rows", "cols"), [(0, 2), (2, 0), (-1, -1)])
def test_make_grid_needs_at_least_one_row_and_column(rows, cols):
    with pytest.raises(ValueError, match="at least 1 row"):
        make_grid(100, 100, rows=rows, cols=cols)


def test_make_grid_shapes_and_names():
    assert list(make_grid(10, 10, rows=1, cols=1)) == ["r0c0"]
    names = [f"r{i}c{j}" for i in range(3) for j in range(3)]
    assert list(make_grid(30, 30, rows=3, cols=3)) == names
    rois = make_grid(30, 30, rows=1, cols=3)
    found = [locate((x, 5), rois) for x in (0, 9.9, 10, 25, 99)]
    assert found == ["r0c0", "r0c0", "r0c1", "r0c2", "r0c2"]


@pytest.mark.parametrize("pos", [None, (np.nan, 1.0), (1.0, np.inf), (1.0,), (1.0, 2.0, 3.0)])
def test_locate_rejects_unusable_positions(pos):
    assert locate(pos, ROIS) is None


def test_locate_overlapping_regions_are_ambiguous():
    overlapping = {"a": (0.0, 0.0, 10.0, 10.0), "b": (5.0, 0.0, 15.0, 10.0)}
    assert locate((2.0, 5.0), overlapping) == "a"
    assert locate((7.0, 5.0), overlapping) is None
    assert locate([12.0, 5.0], overlapping) == "b"  # lists work like tuples


# ------------------------------------------------------------------------ parsing
def test_parse_tag_map_accepts_inline_specs_longer_than_a_file_name():
    spec = ";".join(f"r{i}:{4 * i + 1},{4 * i + 2}" for i in range(80))
    assert len(spec) > 255
    sets, rois = parse_tag_map(spec)
    assert len(sets) == 80 and sets["r79"] == [317, 318] and rois is None


def test_parse_tag_map_files(tmp_path):
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps({"a": [1, 2.0, "3"]}))
    assert parse_tag_map(str(plain)) == ({"a": [1, 2, 3]}, None)

    no_rois = tmp_path / "tags_only.json"
    no_rois.write_text(json.dumps({"tags": {"a": [1], "b": [2]}}))
    assert parse_tag_map(str(no_rois)) == ({"a": [1], "b": [2]}, None)

    with_rois = tmp_path / "with_rois.json"
    with_rois.write_text(json.dumps({"tags": {"a": [1]}, "rois": {"a": [0, 0, 5, 5]}}))
    assert parse_tag_map(str(with_rois)) == ({"a": [1]}, {"a": (0.0, 0.0, 5.0, 5.0)})


def test_parse_tag_map_file_errors(tmp_path):
    not_a_mapping = tmp_path / "list.json"
    not_a_mapping.write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="expected a mapping"):
        parse_tag_map(str(not_a_mapping))

    short_roi = tmp_path / "short_roi.json"
    short_roi.write_text(json.dumps({"tags": {"a": [1]}, "rois": {"a": [0, 0, 5]}}))
    with pytest.raises(ValueError, match="roi a needs"):
        parse_tag_map(str(short_roi))

    # a path that does not exist is read as an inline spec and rejected as such
    with pytest.raises(ValueError, match="bad --closed-set-map entry"):
        parse_tag_map(str(tmp_path / "missing.json"))


@pytest.mark.parametrize("args", [None, []])
def test_parse_roi_args_without_arguments(args):
    assert parse_roi_args(args) is None


def test_parse_roi_args_validates():
    assert parse_roi_args([["a", "0", "0", "5", "5"], ["b", "5", "0", "9", "5"]]) == {
        "a": (0.0, 0.0, 5.0, 5.0),
        "b": (5.0, 0.0, 9.0, 5.0),
    }
    with pytest.raises(ValueError, match="expected X0 Y0 X1 Y1"):
        parse_roi_args([["a", "0", "0", "5"]])
    with pytest.raises(ValueError, match="given twice"):
        parse_roi_args([["a", "0", "0", "5", "5"], ["a", "6", "0", "9", "5"]])
    with pytest.raises(ValueError, match="empty rectangle"):
        parse_roi_args([["a", "0", "0", "5", "0"]])
    with pytest.raises(ValueError):
        parse_roi_args([["a", "0", "0", "5", "x"]])


# ------------------------------------------------------------------------- scope
def test_scope_needs_tag_sets_or_inference():
    with pytest.raises(ValueError, match="tag_sets or infer"):
        QuadrantScope()


def test_scope_refuses_malformed_tag_sets_and_normalises_the_rest():
    with pytest.raises(ValueError, match="not a valid tag id"):
        QuadrantScope(tag_sets={"a": [1, "x"]})
    assert QuadrantScope(tag_sets={"a": [2.0, np.int64(1), "3"]}).active == {"a": [1, 2, 3]}


def test_scope_lookups():
    positions = {0: {0: (1.0, 1.0), 1: (11.0, 1.0), 2: (50.0, 50.0), 3: (np.nan, 1.0)}}
    scope = prepared_scope({"a": [2, 1]}, positions)
    assert scope.regions == ["a", "b"]
    assert scope.active == {"a": [1, 2]}
    assert [scope.quadrant_of(0, t) for t in range(5)] == ["a", "b", None, None, None]
    assert scope.tag_set("a") == [1, 2] and scope.tag_set("b") is None
    assert scope.owner(1) == "a" and scope.owner(9) is None
    assert scope.union_tags() == {1, 2}
    assert QuadrantScope(infer=True).union_tags() is None


def test_scope_cannot_place_anything_before_prepare():
    scope = QuadrantScope(tag_sets={"a": [1]})
    assert scope.quadrant_of(0, 0) is None
    assert scope.regions == []


def test_vote_filter_causes_are_counted():
    positions = {0: {0: (1.0, 1.0), 1: (11.0, 1.0), 2: (50.0, 50.0)}}
    scope = prepared_scope({"a": [1, 2]}, positions)  # region b has no tag set (legacy)

    assert scope.vote_filter(0, 0, 1) is True
    assert scope.vote_filter(0, 0, np.float64(2.0)) is True  # normalised first
    assert scope.vote_filter(0, 0, 5) is False  # not in the region's set
    assert scope.vote_filter(0, 0, "x") is False  # not a tag: rejected without being counted
    assert scope.vote_filter(0, 0, None) is False
    assert scope.vote_filter(0, 2, 1) is False  # track outside every region
    assert scope.vote_filter(0, 1, 1) is False  # tag owned by a, seen in legacy region b
    assert scope.vote_filter(0, 1, 5) is True  # unowned tag in a legacy region is fine
    assert scope.vote_filter(0, 99, 1) is False  # unknown track

    assert dict(scope.rejected) == {
        "not_in_quadrant_set": 1,
        "track_outside_regions": 2,
        "wrong_quadrant": 1,
    }
    assert dict(scope.wrong_quadrant) == {(1, "b", "a"): 1}


def test_vote_filter_wrong_quadrant_inside_a_region_with_a_tag_set():
    positions = {0: {0: (1.0, 1.0)}}
    scope = prepared_scope({"a": [1, 2], "b": [3, 4]}, positions)
    assert scope.vote_filter(0, 0, 3) is False
    assert dict(scope.wrong_quadrant) == {(3, "a", "b"): 1}


# ------------------------------------------------------------------- elimination
def test_unplaceable_instances_keep_owned_tags_and_drop_foreign_ones():
    positions = {0: {0: (1.0, 1.0), 1: (11.0, 1.0)}}  # tracks 5 and 6 have no position
    matched = {0: {0: 1, 1: 3, 5: 99, 6: 2, 7: ""}}
    scope = prepared_scope({"a": [1, 2], "b": [3, 4]}, positions)

    res = eliminate_by_quadrant(matched, positions, scope)

    assert dict(res.matched[0]) == {0: 1, 1: 3, 6: 2}
    assert res.dropped_foreign == 1
    # the unplaceable instances could be the missing bees: no region may infer a tag in this frame
    assert not res.n_assigned


def test_untracked_positions_need_a_prepared_scope():
    scope = QuadrantScope(tag_sets={"a": [1, 2]})  # no rois, and prepare() not called
    with pytest.raises(RuntimeError, match=r"prepare\(\) must be called"):
        eliminate_by_quadrant({}, {}, scope, untracked_positions={0: [(1.0, 1.0)]})


# ---------------------------------------------------------------------------- report
def test_report_lists_instances_without_a_region_instead_of_crashing():
    positions = {0: {0: (1.0, 1.0)}}
    scope = prepared_scope({"a": [1], "b": [2]}, positions)
    # track 7 has no position, and the last instance has neither a track nor a position
    instances = {0: [(0, (1.0, 1.0)), (7, None), (None, None), (None, (50.0, 50.0))]}

    report = build_report(scope, None, 0, 0, {0: {0: 1}}, {0: {0: 1}}, instances)

    assert report["n_instances"] == 4
    assert report["n_instances_without_tag_after_elimination"] == 3
    assert report["n_none_instances_by_cause"] == {
        "undeterminable_quadrant": 1,
        "sleap_untracked_instance": 2,
    }
    assert report["n_none_instances_by_region_and_cause"] == {
        "none": {"undeterminable_quadrant": 1, "sleap_untracked_instance": 2}
    }
    json.dumps(report)  # serialisable
    assert report["rois"]["a"] == [0.0, 0.0, 10.0, 10.0]


def test_report_needs_a_prepared_scope_to_place_untracked_instances():
    scope = QuadrantScope(tag_sets={"a": [1]})
    with pytest.raises(RuntimeError, match=r"prepare\(\) must be called"):
        build_report(scope, None, 0, 0, {}, {}, {0: [(None, (1.0, 1.0))]})


def test_report_blames_foreign_tags_and_ignores_blank_ones():
    positions = {0: {0: (1.0, 1.0), 1: (2.0, 2.0)}}
    scope = prepared_scope({"a": [1], "b": [2]}, positions)
    instances = {0: [(0, (1.0, 1.0)), (1, (2.0, 2.0))]}
    # track 0 got tag 2 (it belongs to region b) and nothing survived; track 1 has a blank tag
    report = build_report(scope, None, 0, 0, {0: {0: 2, 1: ""}}, {0: {0: "", 1: ""}}, instances)

    assert report["n_instances_without_tag_before_elimination"] == 2
    assert report["n_none_instances_by_region_and_cause"] == {
        "a": {"wrong_quadrant_or_foreign_tag_dropped": 1, "no_decode_evidence": 1}
    }


def test_untracked_instances_in_inactive_regions_do_not_block_other_regions():
    positions = {0: {0: (1.0, 1.0)}}
    scope = prepared_scope({"a": [1, 2]}, positions)  # region b has no tag set
    matched = {0: {0: 1}}

    inactive = eliminate_by_quadrant(
        matched, positions, scope, untracked_positions={0: [(11.0, 1.0)]}
    )
    in_a = eliminate_by_quadrant(matched, positions, scope, untracked_positions={0: [(1.0, 1.0)]})
    unknown = eliminate_by_quadrant(matched, positions, scope, untracked_positions={0: [None]})

    assert "untracked_instance" not in inactive.abstained_frames.values()
    # with one tag missing and no leftover instance there is nothing to abstain about either
    assert dict(in_a.matched[0]) == dict(unknown.matched[0]) == {0: 1}


def test_scope_positions_may_differ_from_the_ones_given_to_elimination():
    scope = prepared_scope({"a": [1, 2]}, {0: {0: (1.0, 1.0)}})
    res = eliminate_by_quadrant({0: {0: 1}}, {}, scope)  # the tracked instance has no position here
    assert dict(res.matched[0]) == {0: 1}


def test_inference_ignores_tracks_it_cannot_place():
    positions = {f: {0: (1.0, 1.0), 1: (99.0, 99.0)} for f in range(40)}
    scope = QuadrantScope(infer=True, expected_size=1, min_support=5, rois=dict(ROIS))
    scope.prepare((20, 10), positions)
    decodes = {f: {0: [7], 1: [8]} for f in range(40)}  # track 1 is outside every region

    scope.infer_from_decodes(decodes)

    assert scope.active == {"a": [7]}
    assert 8 not in scope.inference["excluded_tags"]


def test_report_without_a_scope_uses_a_single_region():
    report = build_report(None, [1, 2], 0, 1, {0: {0: 1}}, {0: {0: 1}, 1: {0: 2}}, {})
    assert report["mode"] == "flat_closed_set"
    assert list(report["regions"]) == ["all"]
    assert build_report(None, None, 0, 0, {}, {}, {})["mode"] == "legacy"
    assert build_report(None, None, 0, 0, {}, {}, {}, extra={"video": "v.mp4"})["video"] == "v.mp4"
