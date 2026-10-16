import json

import numpy as np
import pytest

from naps.closed_set import eliminate_identities
from naps.evidence import collect_votes
from naps.quadrants import (
    QuadrantScope,
    build_report,
    eliminate_by_quadrant,
    locate,
    make_grid,
    parse_roi_args,
    parse_tag_map,
)
from naps.tags import normalize_tag, tag_name

NAMES = ["top_left", "top_right", "bottom_left", "bottom_right"]
ORIGIN = {
    "top_left": (0, 0),
    "top_right": (1832, 0),
    "bottom_left": (0, 1832),
    "bottom_right": (1832, 1832),
}
TAGS = {q: [10 + 4 * i + k for k in range(4)] for i, q in enumerate(NAMES)}
SIZE = (3664, 3664)


def track_id(q, k):
    return 10 * NAMES.index(q) + k


def world(n=60):
    """16 bees, 4 per quadrant of a 3664x3664 frame. Everything assigned."""
    positions, matched = {}, {}
    for f in range(n):
        positions[f], matched[f] = {}, {}
        for q in NAMES:
            ox, oy = ORIGIN[q]
            for k in range(4):
                t = track_id(q, k)
                positions[f][t] = np.array(
                    [ox + 300 + 600 * (k % 2) + 0.5 * f, oy + 300 + 600 * (k // 2)]
                )
                matched[f][t] = TAGS[q][k]
    return matched, positions


def scope_for(positions, tag_sets=TAGS, **kw):
    scope = QuadrantScope(tag_sets=tag_sets, **kw)
    scope.prepare(SIZE, positions)
    return scope


def tagged(res, f):
    return dict(res.matched[f])


# ------------------------------------------------------------------ grid / parsing


def test_default_grid_follows_frame_size_and_names():
    rois = make_grid(*SIZE)
    assert set(rois) == set(NAMES)
    assert locate((10, 10), rois) == "top_left"
    assert locate((1831.9, 10), rois) == "top_left"
    assert locate((1832, 10), rois) == "top_right"
    assert locate((10, 1832), rois) == "bottom_left"
    assert locate((3000, 3000), rois) == "bottom_right"
    # outer edges are open: a prediction slightly outside the frame stays in the nearest tile
    assert locate((-5, 3700), rois) == "bottom_left"
    # a different frame size gives different tiles (nothing hard-coded to 1832)
    assert locate((1500, 10), make_grid(3672, 3672)) == "top_left"
    assert locate((1840, 10), make_grid(3672, 3672)) == "top_right"
    assert locate((np.nan, 10), rois) is None and locate(None, rois) is None
    assert set(make_grid(300, 100, rows=1, cols=3)) == {"r0c0", "r0c1", "r0c2"}


def test_parse_tag_map_inline_json_yaml(tmp_path):
    inline = "top_left:45,46,47,48; top_right: 49, 50,51,52"
    sets, rois = parse_tag_map(inline)
    assert sets == {"top_left": [45, 46, 47, 48], "top_right": [49, 50, 51, 52]} and rois is None

    (tmp_path / "m.json").write_text(json.dumps({"top_left": [1, 2, 3, 4]}))
    assert parse_tag_map(str(tmp_path / "m.json"))[0] == {"top_left": [1, 2, 3, 4]}

    (tmp_path / "m.yaml").write_text(
        "tags:\n  a: [1, 2]\n  b: [3, 4]\nrois:\n  a: [0, 0, 10, 10]\n  b: [10, 0, 20, 10]\n"
    )
    sets, rois = parse_tag_map(str(tmp_path / "m.yaml"))
    assert sets == {"a": [1, 2], "b": [3, 4]}
    assert rois is not None and rois["b"] == (10.0, 0.0, 20.0, 10.0)


def test_parse_tag_map_rejects_bad_input():
    with pytest.raises(ValueError, match="both"):
        parse_tag_map("a:1,2;b:2,3")  # tag in two regions
    with pytest.raises(ValueError, match="duplicate"):
        parse_tag_map("a:1,1")
    with pytest.raises(ValueError, match="not a tag"):
        parse_tag_map("a:1,x")
    with pytest.raises(ValueError):
        parse_tag_map("just-some-text")
    with pytest.raises(ValueError, match="empty"):
        parse_tag_map(";")
    with pytest.raises(ValueError, match="given twice"):
        parse_tag_map("a:1,2;a:3,4")  # would silently drop the first entry
    with pytest.raises(ValueError, match="given twice"):
        parse_tag_map("a:1,2; a :3,4")  # names are compared after stripping
    with pytest.raises(ValueError, match="empty rectangle"):
        parse_roi_args([["a", "5", "5", "5", "9"]])


def test_unknown_region_in_map_is_refused():
    _, positions = world(3)
    with pytest.raises(ValueError, match="not among"):
        scope_for(positions, tag_sets={"nowhere": [1, 2, 3, 4]})


def test_normalize_tag():
    assert normalize_tag(21.0) == 21 and type(normalize_tag(np.float64(21.0))) is int
    assert type(normalize_tag(np.int32(7))) is int and normalize_tag("21.0") == 21
    for bad in (None, "", float("nan"), float("inf"), "None", "nan"):
        assert normalize_tag(bad, strict=True) is None
    assert normalize_tag(2.5, strict=True) is None
    assert normalize_tag("a") == "a" and normalize_tag("a", strict=True) is None
    assert tag_name(np.float64(21.0)) == "ArUcoTag#21"
    with pytest.raises(ValueError):
        tag_name(None)


# ----------------------------------------------------------------------- elimination


def test_flat_closed_set_abstains_on_16_bees_but_quadrants_recover():
    matched, positions = world()
    del matched[30][track_id("top_left", 2)]
    flat = eliminate_identities(matched, positions, TAGS["top_left"])
    assert track_id("top_left", 2) not in flat.matched[30]  # the documented limitation
    res = eliminate_by_quadrant(matched, positions, scope_for(positions))
    assert tagged(res, 30)[track_id("top_left", 2)] == TAGS["top_left"][2]
    assert res.n_assigned == 1
    for q in NAMES:
        for k in range(4):
            assert tagged(res, 30)[track_id(q, k)] == TAGS[q][k]  # all 16 bees tagged


def test_one_bee_in_one_quadrant_gets_the_right_tag():
    matched, positions = world()
    del matched[30][track_id("bottom_left", 1)]
    res = eliminate_by_quadrant(matched, positions, scope_for(positions))
    assert [(a[0], a[1], a[2]) for a in res.per_quadrant["bottom_left"].assigned] == [
        (30, track_id("bottom_left", 1), TAGS["bottom_left"][1])
    ]
    assert all(not r.assigned for q, r in res.per_quadrant.items() if q != "bottom_left")
    assert track_id("bottom_left", 1) not in matched[30]  # input untouched


def test_two_quadrants_recovered_in_the_same_frame():
    matched, positions = world()
    del matched[30][track_id("top_left", 0)]
    del matched[30][track_id("bottom_right", 3)]
    res = eliminate_by_quadrant(matched, positions, scope_for(positions))
    assert tagged(res, 30)[track_id("top_left", 0)] == TAGS["top_left"][0]
    assert tagged(res, 30)[track_id("bottom_right", 3)] == TAGS["bottom_right"][3]
    assert res.n_assigned == 2


def test_quadrant_with_two_missing_tags_abstains_but_neighbours_still_work():
    matched, positions = world()
    del matched[30][track_id("top_right", 0)], matched[30][track_id("top_right", 1)]
    del matched[30][track_id("top_left", 2)]
    res = eliminate_by_quadrant(matched, positions, scope_for(positions))
    assert track_id("top_right", 0) not in res.matched[30]
    assert track_id("top_right", 1) not in res.matched[30]
    assert res.abstained["not_exactly_one"] == 1
    assert res.abstained_frames[(30, "top_right")] == "not_exactly_one"
    assert tagged(res, 30)[track_id("top_left", 2)] == TAGS["top_left"][2]


def test_untracked_instance_blocks_only_its_own_quadrant():
    matched, positions = world()
    del matched[30][track_id("top_left", 2)]
    del matched[30][track_id("top_right", 2)]
    scope = scope_for(positions)
    # a SLEAP instance without a track sits in the top_left tile in frame 30
    res = eliminate_by_quadrant(
        matched, positions, scope, untracked_positions={30: [np.array([500.0, 500.0])]}
    )
    assert track_id("top_left", 2) not in res.matched[30]
    assert res.abstained_frames[(30, "top_left")] == "untracked_instance"
    assert tagged(res, 30)[track_id("top_right", 2)] == TAGS["top_right"][2]
    # an untracked instance without a usable position could be anywhere: it blocks every quadrant
    res = eliminate_by_quadrant(matched, positions, scope, untracked_positions={30: [None]})
    assert res.n_assigned == 0 and res.abstained["untracked_instance"] == 2
    # legacy frame-global counts are accepted and conservative
    res = eliminate_by_quadrant(matched, positions, scope, untracked_counts={30: 1})
    assert res.n_assigned == 0


def test_tag_in_the_wrong_quadrant_is_dropped_and_does_not_block_its_owner():
    matched, positions = world()
    # tag of top_left sits on a top_right bee for one frame (and that bee's own tag is gone)
    matched[30][track_id("top_right", 1)] = TAGS["top_left"][0]
    del matched[30][track_id("top_left", 3)]
    res = eliminate_by_quadrant(matched, positions, scope_for(positions))
    # the foreign tag was removed from top_right and the bee got its own tag back by elimination
    assert tagged(res, 30)[track_id("top_right", 1)] == TAGS["top_right"][1]
    assert tagged(res, 30)[track_id("top_left", 3)] == TAGS["top_left"][3]
    assert res.dropped_foreign == 1
    assert res.per_quadrant["top_right"].dropped_foreign == 1
    # and the tag may not appear twice in a frame
    tags = list(tagged(res, 30).values())
    assert len(tags) == len(set(tags)) == 16


def test_vote_filter_drops_and_flags_wrong_quadrant_votes():
    _, positions = world(10)
    scope = scope_for(positions)
    t_tr = track_id("top_right", 0)
    decodes = {
        f: {track_id("top_left", 0): [10], t_tr: [14 if f != 4 else 10, None]} for f in range(10)
    }
    votes = collect_votes(decodes, positions, [], vote_filter=scope.vote_filter)
    assert votes[4][t_tr] == []  # tag 10 (top_left's) decoded in top_right is ignored
    assert votes[3][t_tr] == [(14, 1.0)] and votes[3][track_id("top_left", 0)] == [(10, 1.0)]
    assert scope.rejected["wrong_quadrant"] == 1
    assert dict(scope.wrong_quadrant) == {(10, "top_right", "top_left"): 1}
    # tracks are kept even when every vote is dropped
    assert t_tr in votes[4]
    # a tag in no set is "not_in_quadrant_set"
    assert not scope.vote_filter(0, t_tr, 99) and scope.rejected["not_in_quadrant_set"] == 1


def test_regions_without_a_tag_set_stay_in_legacy_mode():
    matched, positions = world()
    del matched[30][track_id("top_left", 2)]
    del matched[30][track_id("top_right", 2)]
    sets = {"top_left": TAGS["top_left"]}
    scope = scope_for(positions, tag_sets=sets)
    res = eliminate_by_quadrant(matched, positions, scope)
    assert tagged(res, 30)[track_id("top_left", 2)] == TAGS["top_left"][2]
    assert track_id("top_right", 2) not in res.matched[30]  # no tag set: nothing invented
    assert res.matched[30][track_id("top_right", 0)] == TAGS["top_right"][0]  # but kept
    # votes in the unlisted region are not filtered, except for tags owned by a listed region
    assert scope.vote_filter(30, track_id("top_right", 0), 77)
    assert not scope.vote_filter(30, track_id("top_right", 0), TAGS["top_left"][0])


def test_nan_position_instance_is_located_by_the_fallback_position():
    # the position dict (tag node, else thorax) is what is located and checked for continuity
    matched, positions = world()
    t = track_id("top_left", 2)
    del matched[30][t]
    positions[30][t] = positions[30][t] + np.array([8.0, 8.0])  # thorax instead of the tag node
    res = eliminate_by_quadrant(matched, positions, scope_for(positions))
    assert tagged(res, 30)[t] == TAGS["top_left"][2]
    # no usable position at all -> not located, not assigned
    positions[30][t] = np.array([np.nan, np.nan])
    res = eliminate_by_quadrant(matched, positions, scope_for(positions))
    assert t not in res.matched[30] and res.n_assigned == 0


# ------------------------------------------------------------------------- inference


def decodes_for(n=60):
    return {f: {track_id(q, k): [TAGS[q][k]] for q in NAMES for k in range(4)} for f in range(n)}


def infer(decodes, positions=None, **kw):
    positions = positions or world(len(decodes))[1]
    scope = QuadrantScope(infer=True, min_support=10, **kw)
    scope.prepare(SIZE, positions)
    scope.infer_from_decodes(decodes)
    return scope


def test_inference_recovers_the_sets():
    scope = infer(decodes_for())
    assert scope.active == {q: sorted(TAGS[q]) for q in NAMES}
    assert not scope.inference["abstained"]


def test_inference_ignores_a_tag_planted_in_the_wrong_tile_for_a_few_frames():
    d = decodes_for()
    for f in (20, 21):  # tag of top_left decoded on a top_right bee, twice
        d[f][track_id("top_right", 0)].append(TAGS["top_left"][0])
    scope = infer(d)
    assert scope.active["top_left"] == sorted(TAGS["top_left"])
    assert scope.active["top_right"] == sorted(TAGS["top_right"])
    assert (
        scope.inference["excluded_tags"].get(TAGS["top_left"][0]) is None
    )  # still attributed to its tile


def test_inference_refuses_a_tag_strongly_seen_in_two_quadrants():
    d = decodes_for()
    for f in range(30, 60):
        d[f][track_id("top_right", 0)].append(TAGS["top_left"][0])
    scope = infer(d)
    ex = scope.inference["excluded_tags"][TAGS["top_left"][0]]
    assert ex["reason"] == "multiple_quadrants"
    # the tag is attributed to neither quadrant, and both abstain (a bee we cannot attribute)
    for q in ("top_left", "top_right"):
        assert q not in scope.active and q in scope.inference["abstained"]
    assert scope.inference["abstained"]["top_right"].startswith("ambiguous_tags_in_multiple")
    assert scope.active["bottom_left"] == sorted(TAGS["bottom_left"])  # others unaffected
    # abstaining regions stay in legacy mode: no filtering there
    assert scope.vote_filter(0, track_id("top_left", 0), TAGS["top_left"][0])


def test_inference_impure_tag_is_refused():
    d = decodes_for()
    for f in range(
        8
    ):  # 8 stray decodes: below min_support (10) but a big share of the tag's support
        d[f][track_id("top_right", 0)].append(TAGS["bottom_left"][0])
    scope = infer(d, min_purity=0.95)  # 60 / 68 = 0.88 -> impure
    assert scope.inference["excluded_tags"][TAGS["bottom_left"][0]]["reason"] == "impure"
    assert "bottom_left" in scope.inference["abstained"]


def test_inference_abstains_on_too_few_or_too_many_tags():
    d = decodes_for()
    for f in d:  # one bee in bottom_right is never decoded
        d[f][track_id("bottom_right", 3)] = [None]
    scope = infer(d)
    assert "bottom_right" in scope.inference["abstained"]
    assert scope.inference["abstained"]["bottom_right"].startswith("fewer_than_4_tags")
    assert "bottom_right" not in scope.active

    d = decodes_for()
    for f in d:  # a fifth tag, equally well supported, appears in top_left
        d[f][track_id("top_left", 0)].append(99)
    scope = infer(d)
    assert "top_left" in scope.inference["abstained"] and "top_left" not in scope.active
    # ... but a fifth, much weaker (still supported) tag does not stop the clear four
    d = decodes_for()
    for f in range(12):
        d[f][track_id("top_left", 0)].append(99)
    scope = infer(d, dominance_ratio=3.0)
    assert scope.active["top_left"] == sorted(TAGS["top_left"])


def test_inference_low_support_noise_is_ignored():
    d = decodes_for()
    d[3][track_id("top_left", 1)].append(55)
    scope = infer(d)
    assert scope.inference["excluded_tags"][55]["reason"] == "low_support"
    assert scope.active["top_left"] == sorted(TAGS["top_left"])


# --------------------------------------------------------------------------- report


def test_report_histograms_and_causes():
    matched, positions = world(20)
    # frames 5-9: one bee of top_left lost; frame 12: two lost; frame 15: one of top_right + a
    # foreign tag
    for f in range(5, 10):
        del matched[f][track_id("top_left", 1)]
    del matched[12][track_id("top_left", 0)], matched[12][track_id("top_left", 1)]
    del matched[15][track_id("top_right", 0)]
    matched[16][track_id("bottom_left", 0)] = 77  # a foreign tag on a bee of bottom_left
    scope = scope_for(positions)
    res = eliminate_by_quadrant(matched, positions, scope)
    instances = {
        f: [(t, tuple(p)) for t, p in positions[f].items()] + [(None, (3000.0, 100.0))]
        for f in positions
    }
    rep = build_report(scope, None, 0, 19, matched, res.matched, instances, elimination=res)
    tl = rep["regions"]["top_left"]
    assert tl["frames_by_n_tags_before_elimination"] == {"4": 14, "3": 5, "2": 1, "1": 0, "0": 0}
    assert tl["frames_by_n_tags_after_elimination"] == {"4": 19, "3": 0, "2": 1, "1": 0, "0": 0}
    assert (
        rep["elimination"]["n_assigned"] == 5 + 1 + 1
    )  # five TL frames + TR frame 15 + BL frame 16
    assert rep["elimination"]["abstention_reasons"] == {"not_exactly_one": 1}
    # 20 untracked instances (one per frame), 2 undecided bees in frame 12
    assert rep["n_none_instances_by_cause"]["sleap_untracked_instance"] == 20
    assert rep["n_none_instances_by_cause"]["elimination_abstained:not_exactly_one"] == 2
    assert rep["n_instances_without_tag_before_elimination"] == 20 + 5 + 2 + 1 + 1
    assert rep["n_instances_without_tag_after_elimination"] == 20 + 2
    json.dumps(rep, default=int)  # serialisable
    assert rep["tag_sets_source"] == "explicit" and rep["rois"]["top_left"][0] is None


def test_report_without_scope_is_a_single_region():
    matched = {0: {1: 5, 2: 6}, 1: {1: 5}}
    rep = build_report(
        None,
        [5, 6],
        0,
        1,
        matched,
        matched,
        {0: [(1, (0, 0)), (2, (1, 1))], 1: [(1, (0, 0)), (2, (1, 1))]},
    )
    assert list(rep["regions"]) == ["all"] and rep["mode"] == "flat_closed_set"
    assert rep["regions"]["all"]["frames_by_n_tags_after_elimination"] == {"2": 1, "1": 1, "0": 0}
    assert rep["n_instances_without_tag_after_elimination"] == 1


def test_max_distance_is_passed_through_to_every_quadrant():
    matched, positions = world()
    for f in range(40, 60):
        t = track_id("top_left", 2)
        del matched[f][t]
        positions[f][t] = positions[f][t] + np.array([0.0, 700.0])
    scope = scope_for(positions)
    assert eliminate_by_quadrant(matched, positions, scope).n_assigned > 0
    res = eliminate_by_quadrant(matched, positions, scope, max_distance=300.0)
    assert res.n_assigned == 0 and res.abstained["discontinuous"] >= 1


def test_tracked_instance_without_a_position_keeps_its_tag_and_blocks_elimination():
    matched, positions = world()
    t = track_id("top_left", 1)  # has a tag from the cost matrix but no usable position this frame
    positions[30][t] = np.array([np.nan, np.nan])
    del matched[30][track_id("top_left", 3)]  # a different top_left bee lost its tag
    res = eliminate_by_quadrant(matched, positions, scope_for(positions))
    assert tagged(res, 30)[t] == TAGS["top_left"][1]  # identity is not thrown away
    assert track_id("top_left", 3) not in res.matched[30]  # and nothing is invented
    assert res.n_assigned == 0 and res.abstained["untracked_instance"] >= 1
    # it blocks the frame in every region, but only that frame
    assert res.abstained_frames[(30, "top_left")] == "untracked_instance"
    matched2, positions2 = world()
    del matched2[31][track_id("top_left", 3)]
    positions2[30][t] = np.array([np.nan, np.nan])
    assert eliminate_by_quadrant(matched2, positions2, scope_for(positions2)).n_assigned == 1


def test_report_histogram_without_a_tag_set_covers_the_observed_counts():
    matched = {0: {i: 100 + i for i in range(30)}}
    inst = {0: [(i, (float(i), 0.0)) for i in range(30)]}
    rep = build_report(None, None, 0, 0, matched, matched, inst)
    assert rep["regions"]["all"]["frames_by_n_tags_after_elimination"] == {"30": 1}
