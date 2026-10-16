"""``JointMatching``: decode, split into regions, solve, scatter; plus the sidecar writer."""

import json

import numpy as np
import pytest

from naps.diagnostics import DiagnosticsParams, apply_policy, assess_result
from naps.evidence import EvidenceSource
from naps.identity import SlotFlag, Votes
from naps.instances import InstanceTable
from naps.joint import (
    FLAT_REGION,
    SIDECAR_SCHEMA_VERSION,
    JointConfig,
    JointMatching,
    qualify_tag,
    region_reasons,
    write_sidecar,
)
from naps.quadrants import QuadrantScope
from naps.reasons import REASON_NAMES, Reason
from naps.sources import Source
from tests.scenes import make_scene

TAGS = [10, 11, 12, 13]
QUADRANTS = ["top_left", "top_right", "bottom_left", "bottom_right"]
OFFSETS = [(0.0, 0.0), (1800.0, 0.0), (0.0, 1800.0), (1800.0, 1800.0)]
QUAD_TAGS = {q: [10 + 4 * i + k for k in range(4)] for i, q in enumerate(QUADRANTS)}


class FakeSource:
    """A ``DecodeSource`` that returns scripted decodes and records what it was asked."""

    def __init__(self, decodes, frame_size=(1800, 1800)):
        self.decodes = decodes
        self.frame_size = frame_size
        self.calls = []

    def decode(self, first_frame, last_frame, coords):
        self.calls.append((first_frame, last_frame, coords))
        return self.decodes, self.frame_size


def decodes_of(scene, tag_ids, slot_offset=0, into=None):
    """``decodes[frame][slot] -> [tag ids]`` from the votes of a synthetic scene."""
    out = {} if into is None else into
    v = scene.votes
    for f, s, t in zip(v.frame, v.slot, v.tag, strict=True):
        out.setdefault(int(f), {}).setdefault(int(s) + slot_offset, []).append(tag_ids[int(t)])
    return out


def combined(scenes, offsets):
    """One table (and decodes / truth) holding several scenes side by side in slot space."""
    parts = [s.table for s in scenes]
    n_slots = sum(p.n_slots for p in parts)
    n_frames = parts[0].n_frames
    xy = np.full((n_frames, n_slots, 2), np.nan)
    tag_xy = np.full_like(xy, np.nan)
    score = np.full((n_frames, n_slots), np.nan)
    inst = np.full((n_frames, n_slots), -1, dtype=np.int64)
    truth = np.full((n_frames, n_slots), -1, dtype=np.int64)
    start = 0
    slot_offsets = []
    for part, scene, (ox, oy) in zip(parts, scenes, offsets, strict=True):
        sl = slice(start, start + part.n_slots)
        shift = np.array([ox, oy])
        xy[:, sl], tag_xy[:, sl] = part.xy + shift, part.tag_xy + shift
        score[:, sl] = part.score
        inst[:, sl] = np.where(part.inst_index >= 0, part.inst_index + start, -1)
        truth[:, sl] = scene.truth
        slot_offsets.append(start)
        start += part.n_slots
    hint = np.full((n_frames, n_slots), -1, dtype=np.int64)
    table = InstanceTable(0, xy, tag_xy, score, hint, inst, np.full((n_frames, n_slots), np.nan))
    return table, truth, slot_offsets


def flat_run(scene, tags=TAGS, *, closed_set=None, decodes=None, config=None, sources=()):
    decodes = decodes_of(scene, tags) if decodes is None else decodes
    source = FakeSource(decodes)
    jm = JointMatching(
        scene.table, source, closed_set=closed_set or tags, config=config,
        evidence_sources=sources,
    )  # fmt: skip
    return jm.match(), source


def quad_scenes(overrides=None):
    overrides = overrides or {}
    scenes = []
    for i in range(4):
        kw = {"seed": 100 + i, "n_frames": 300, "absent": {i % 4: (40, 260)}}
        kw.update(overrides.get(i, {}))
        scenes.append(make_scene(**kw))
    return scenes


# ----------------------------------------------------------------------------- flat mode
def test_flat_region_labels_instances_with_tag_ids():
    scene = make_scene(2, absent={3: (50, 349)})
    result, source = flat_run(scene)
    run = result.regions[FLAT_REGION]
    assert run.status == "ok"
    assert run.solved
    assert run.tags == TAGS
    assert run.n_bees == 4
    labelled = result.slot_tag >= 0
    assert set(np.unique(result.slot_tag[labelled])) <= set(TAGS)
    truth_tag = np.array(TAGS)[np.maximum(scene.truth, 0)]
    assert (result.slot_tag[labelled] == truth_tag[labelled]).all()
    absent = (scene.truth == 3) & (np.arange(400)[:, None] >= 50) & (np.arange(400)[:, None] < 350)
    assert (labelled & absent).sum() / absent.sum() > 0.95
    sources = set(np.unique(result.slot_source[labelled & absent]))
    assert sources <= {Source.TRACKLET, Source.ELIMINATION}
    assert result.counters["decodes"] == len(scene.votes)
    assert (result.slot_region[scene.table.valid] == 0).all()
    assert result.frame_size == (1800, 1800)
    assert result.region_names == [FLAT_REGION]
    assert np.isnan(result.slot_conf[~labelled]).all()
    assert source.calls[0][:2] == (0, 399)


def test_the_decoder_gets_the_tag_node_of_every_instance_in_every_frame():
    scene = make_scene(3, n_frames=40, miss_rate=0.2)
    _, source = flat_run(scene)
    ((first, last, coords),) = source.calls
    assert (first, last) == (0, 39)
    t = scene.table
    assert sorted(coords) == list(range(40))
    for row in range(40):
        assert sorted(coords[row]) == sorted(np.flatnonzero(t.inst_index[row] >= 0).tolist())
        for slot, xy in coords[row].items():
            np.testing.assert_array_equal(xy, t.tag_xy[row, slot])


def test_decode_bookkeeping_counts_what_was_dropped():
    scene = make_scene(4, n_frames=60, miss_rate=0.25)
    decodes = decodes_of(scene, TAGS)
    n_votes = len(scene.votes)
    decodes[5][next(iter(decodes[5]))].extend([99, None])  # out of the set, and no decode
    empty = next((f, s) for f in range(60) for s in range(4) if not scene.table.valid[f, s])
    decodes.setdefault(empty[0], {}).setdefault(empty[1], []).append(10)  # on a non-instance
    result, _ = flat_run(scene, decodes=decodes)
    assert result.counters["decodes_out_of_set"] == 1
    assert result.counters["decodes_outside_regions"] == 1
    assert result.counters["decodes"] == n_votes + 2
    assert "decodes_wrong_region" not in result.counters


@pytest.mark.parametrize(
    ("closed_set", "message"),
    [
        ([10, 11], "supports 3..4 tags per region, found 2"),
        ([10, 11, 12, 13, 14], "supports 3..4 tags per region, found 5"),
    ],
)
def test_unsupported_tag_set_sizes(closed_set, message):
    scene = make_scene(5, n_frames=50)
    result, _ = flat_run(scene, closed_set=closed_set)
    run = result.regions[FLAT_REGION]
    assert run.status == "unsupported_tagset"
    assert message in run.reasons[-1]
    assert not run.solved
    assert (result.slot_tag == -1).all()


def test_a_region_without_decodes_or_instances_is_no_data():
    scene = make_scene(6, n_frames=30)
    result, _ = flat_run(scene, decodes={})
    run = result.regions[FLAT_REGION]
    assert run.status == "no_data"
    assert run.reasons == ["no tag was decoded in the region"]
    nothing = np.full((10, 2, 2), np.nan)
    table = InstanceTable(
        0, nothing, nothing.copy(), np.full((10, 2), np.nan), np.full((10, 2), -1),
        np.full((10, 2), -1), np.full((10, 2), np.nan),
    )  # fmt: skip
    empty = JointMatching(table, FakeSource({}), closed_set=TAGS).match()
    assert empty.regions[FLAT_REGION].status == "no_data"
    assert empty.regions[FLAT_REGION].reasons == ["no instances in the region"]
    with pytest.raises(RuntimeError, match="cannot be solved"):
        empty.regions[FLAT_REGION].solve()


def test_a_crowded_region_is_not_a_closed_set():
    scene = make_scene(7, n_frames=60, n_bees=7, n_real=4)  # 7 bees > colony size 4 + slack 2
    result, _ = flat_run(scene)
    run = result.regions[FLAT_REGION]
    assert run.status == "unsupported_tagset"
    assert "median of 7 instances per frame" in run.reasons[-1]
    assert "only these animals" in run.reasons[-1]


def test_three_tags_and_an_untagged_bee_use_a_phantom_tag():
    scene = make_scene(8, n_frames=200, n_bees=4, n_real=3)
    result, _ = flat_run(scene, tags=[10, 11, 12])
    run = result.regions[FLAT_REGION]
    assert run.status == "ok"
    assert run.tags == [10, 11, 12]
    assert run.n_bees == 4
    untagged = scene.truth == 3
    assert (result.slot_tag[untagged] == -1).all()
    tagged = scene.truth >= 0
    tagged &= ~untagged
    assert (result.slot_tag[tagged] == np.array([10, 11, 12])[scene.truth[tagged]]).all()
    flags = result.slot_flag[untagged]
    allowed = {SlotFlag.UNTAGGED_BEE, SlotFlag.LOW_CONFIDENCE, SlotFlag.GHOST}
    assert set(np.unique(flags)) <= allowed


def test_a_smaller_colony_has_no_phantoms():
    scene = make_scene(9, n_frames=100, n_bees=3)
    config = JointConfig(colony_size=3)
    result, _ = flat_run(scene, tags=[10, 11, 12], config=config)
    assert result.regions[FLAT_REGION].n_bees == 3
    assert (result.slot_tag[scene.truth >= 0] >= 10).mean() > 0.95


def test_constructor_errors():
    scene = make_scene(1, n_frames=10)
    source = FakeSource({})
    scope = QuadrantScope(tag_sets={"top_left": TAGS})
    with pytest.raises(ValueError, match="exactly one of closed_set and quadrants"):
        JointMatching(scene.table, source)
    with pytest.raises(ValueError, match="exactly one of closed_set and quadrants"):
        JointMatching(scene.table, source, closed_set=TAGS, quadrants=scope)
    for size in (0, 7):
        with pytest.raises(ValueError, match=r"colony_size must be 1\.\.6"):
            JointMatching(
                scene.table, source, closed_set=TAGS, config=JointConfig(colony_size=size)
            )


# ------------------------------------------------------------------------ region runs
def test_resolving_with_replacement_votes_keeps_the_stored_labels():
    scene = make_scene(11, n_frames=120)
    result, _ = flat_run(scene)
    run = result.regions[FLAT_REGION]
    stored = run.labels
    kept = stored.slot_tag.copy()
    held_out = Votes(
        scene.votes.frame[scene.votes.tag != 0],
        scene.votes.slot[scene.votes.tag != 0],
        scene.votes.tag[scene.votes.tag != 0],
        scene.votes.weight[scene.votes.tag != 0],
    )
    trial = run.solve(held_out)
    assert run.labels is stored
    np.testing.assert_array_equal(stored.slot_tag, kept)
    assert trial is not stored
    assert (trial.slot_source == Source.DECODE).sum() < (stored.slot_source == Source.DECODE).sum()
    again = run.solve()
    assert run.labels is again
    np.testing.assert_array_equal(again.slot_tag, kept)


def test_refresh_rebuilds_the_instance_arrays_from_the_region_labels():
    scene = make_scene(12, n_frames=100, n_bees=4, n_real=3)
    result, _ = flat_run(scene, tags=[10, 11, 12])
    before = (result.slot_tag.copy(), result.slot_conf.copy(), result.slot_source.copy())
    result.refresh()
    for a, b in zip(before, (result.slot_tag, result.slot_conf, result.slot_source), strict=True):
        np.testing.assert_array_equal(a, b)
    result.regions[FLAT_REGION].labels.slot_tag[:] = -1
    result.refresh()
    assert (result.slot_tag == -1).all()


def test_refresh_skips_regions_without_labels():
    scene = make_scene(13, n_frames=30)
    result, _ = flat_run(scene, closed_set=[10, 11])
    result.slot_tag[:] = 5
    result.refresh()
    assert (result.slot_tag == -1).all()


def test_qualify_tag():
    assert qualify_tag("s1", 7) == "s1:ArUcoTag#7"
    assert qualify_tag(None, 7) == "ArUcoTag#7"
    assert qualify_tag("", 7) == "ArUcoTag#7"


# ------------------------------------------------------------------ external evidence
class VoteSource(EvidenceSource):
    """Soft votes for tag 13 on the slots of bee 3, plus junk (unknown tag, empty slot)."""

    name = "fake"

    def __init__(self, truth, valid, active=True):
        self.truth, self.valid, self.active = truth, valid, active
        self.prepared = []

    def prepare(self, positions, prior_votes):
        self.prepared.append((positions, prior_votes))

    def votes(self, frame, track_positions, prior=None):
        if not self.active:
            return {}
        out = {int(s): [(13, 0.5), (99, 1.0)] for s in np.flatnonzero(self.truth[frame] == 3)}
        empty = np.flatnonzero(~self.valid[frame])
        if len(empty):
            out[int(empty[0])] = [(13, 0.5)]
        return out


def test_external_evidence_is_soft_and_never_anchors():
    scene = make_scene(14, n_frames=120, absent={3: (0, 119)}, miss_rate=0.1)
    source = VoteSource(scene.truth, scene.table.valid)
    result, _ = flat_run(scene, sources=[source])
    run = result.regions[FLAT_REGION]
    assert len(source.prepared) == 1
    positions, prior = source.prepared[0]
    assert set(positions) == set(range(120))
    assert set(prior) <= set(range(120))
    assert run.extra is not None
    assert len(run.extra) > 0
    assert (run.extra.tag == 3).all()  # the unknown tag 99 and the empty slot are dropped
    bee3 = (scene.truth == 3) & scene.table.valid
    candidates = np.full(bee3.shape, -1)  # decode anchors, back in input-slot space
    rows, cols = np.nonzero(run.source_slot >= 0)
    candidates[rows, run.source_slot[rows, cols]] = run.labels.candidates[rows, cols]
    assert bee3.any()
    assert (candidates[bee3] == -1).all()
    assert not (result.slot_source[bee3] == Source.DECODE).any()
    assert (result.slot_tag[bee3 & (result.slot_tag >= 0)] == 13).all()


def test_an_evidence_source_without_votes_leaves_no_extra():
    scene = make_scene(15, n_frames=60)
    source = VoteSource(scene.truth, scene.table.valid, active=False)
    result, _ = flat_run(scene, sources=[source])
    assert result.regions[FLAT_REGION].extra is None


# -------------------------------------------------------------------------- quadrants
def build_quadrants(scenes, tag_sets=None, scope=None, frame_size=(3600, 3600), colony=4):
    table, truth, offsets = combined(scenes, OFFSETS[: len(scenes)])
    decodes = {}
    for i, (scene, off) in enumerate(zip(scenes, offsets, strict=True)):
        decodes_of(scene, QUAD_TAGS[QUADRANTS[i]], off, decodes)
    scope = scope or QuadrantScope(tag_sets=tag_sets or QUAD_TAGS)
    source = FakeSource(decodes, frame_size)
    jm = JointMatching(table, source, quadrants=scope, config=JointConfig(colony_size=colony))
    return jm, truth, source


def test_four_colonies_are_solved_independently():
    scenes = quad_scenes()
    jm, truth, _ = build_quadrants(scenes)
    result = jm.match()
    assert result.region_names == QUADRANTS
    assert result.frame_size == (3600, 3600)
    for i, q in enumerate(QUADRANTS):
        run = result.regions[q]
        assert run.status == "ok"
        assert run.tags == QUAD_TAGS[q]
        cols = slice(
            sum(s.table.n_slots for s in scenes[:i]), sum(s.table.n_slots for s in scenes[: i + 1])
        )
        sl_tag, sl_truth = result.slot_tag[:, cols], truth[:, cols]
        labelled = sl_tag >= 0
        assert (result.slot_region[:, cols][scenes[i].table.valid] == i).all()
        expected = np.array(QUAD_TAGS[q])[np.maximum(sl_truth, 0)]
        assert (sl_tag[labelled] == expected[labelled]).all()
        absent = (
            (sl_truth == i % 4) & (np.arange(300)[:, None] >= 40) & (np.arange(300)[:, None] <= 260)
        )
        assert (labelled & absent).sum() / absent.sum() > 0.95
    assert result.counters.get("decodes_wrong_region", 0) == 0


def test_a_tag_owned_by_another_region_is_counted_and_ignored():
    scenes = quad_scenes()
    jm, _, source = build_quadrants(scenes)
    foreign = QUAD_TAGS["bottom_right"][0]
    source.decodes[7][0].append(foreign)  # slot 0 lies in top_left
    result = jm.match()
    assert result.counters["decodes_wrong_region"] == 1


def test_inferred_sets_accept_three_tags_and_abstain_without_decodes():
    scenes = quad_scenes({1: {"n_real": 3, "absent": {}}, 3: {"decode_rate": 0.0, "absent": {}}})
    scenes = [*scenes[:2], scenes[3]]  # top_left, top_right (3 tags), bottom_left -> no decodes
    scope = QuadrantScope(infer=True, expected_size=4, min_size=3, min_support=30)
    # the third scene sits in the bottom_left tile; tags of the first two are inferred
    table, truth, offsets = combined(scenes, [OFFSETS[0], OFFSETS[1], OFFSETS[2]])
    decodes = {}
    for i, (scene, off) in enumerate(zip(scenes, offsets, strict=True)):
        decodes_of(scene, QUAD_TAGS[QUADRANTS[i]], off, decodes)
    result = JointMatching(table, FakeSource(decodes, (3600, 3600)), quadrants=scope).match()
    assert scope.inference["inferred"]["top_left"] == QUAD_TAGS["top_left"]
    assert scope.inference["inferred"]["top_right"] == QUAD_TAGS["top_right"][:3]
    top_right = result.regions["top_right"]
    assert top_right.status == "ok"
    assert top_right.tags == QUAD_TAGS["top_right"][:3]
    assert top_right.n_bees == 4
    untagged = np.zeros(truth.shape, dtype=bool)
    cols = slice(scenes[0].table.n_slots, scenes[0].table.n_slots + scenes[1].table.n_slots)
    untagged[:, cols] = truth[:, cols] == 3
    assert (result.slot_tag[untagged] == -1).all()
    bottom_left = result.regions["bottom_left"]
    assert bottom_left.status == "unsupported_tagset"
    assert "fewer_than_3_tags" in bottom_left.reasons[0]
    bottom_right = result.regions["bottom_right"]
    assert bottom_right.status == "no_data"
    assert bottom_right.reasons[-1] == "no instances in the region"
    assert (result.slot_tag[:, scenes[0].table.n_slots + scenes[1].table.n_slots :] == -1).all()


def test_a_region_with_too_many_tags_is_unsupported_and_the_others_still_run():
    scenes = quad_scenes()
    sets = dict(QUAD_TAGS)
    sets["top_right"] = [*sets["top_right"], 99]
    jm, _, _ = build_quadrants(scenes, tag_sets=sets)
    result = jm.match()
    assert result.regions["top_right"].status == "unsupported_tagset"
    assert result.regions["top_left"].status == "ok"
    assert result.regions["bottom_left"].status == "ok"
    cols = slice(scenes[0].table.n_slots, scenes[0].table.n_slots + scenes[1].table.n_slots)
    assert (result.slot_tag[:, cols] == -1).all()
    assert (result.slot_tag[:, : scenes[0].table.n_slots] >= 0).any()


def test_a_listed_region_without_instances_or_decodes_is_no_data():
    scenes = quad_scenes()[:3]
    jm, _, _ = build_quadrants(scenes, tag_sets=QUAD_TAGS)
    result = jm.match()
    assert result.regions["bottom_right"].status == "no_data"
    scenes = quad_scenes({2: {"decode_rate": 0.0, "absent": {}}})[:3]
    result = build_quadrants(scenes)[0].match()
    assert result.regions["bottom_left"].status == "no_data"
    assert result.regions["bottom_left"].reasons == ["no tag was decoded in the region"]


def test_instances_outside_every_region_stay_untracked():
    scenes = quad_scenes()
    scope = QuadrantScope(tag_sets={"tl": QUAD_TAGS["top_left"]}, rois={"tl": (0, 0, 1800, 1800)})
    jm, _, source = build_quadrants(scenes, scope=scope)
    result = jm.match()
    n_tl = scenes[0].table.n_slots
    assert result.counters["decodes_outside_regions"] > 0
    assert (result.slot_tag[:, n_tl:] == -1).all()
    assert (result.slot_region[:, n_tl:] == -1).all()
    assert (result.slot_tag[:, :n_tl] >= 0).any()
    assert source.calls  # the frame size is not needed with explicit ROIs


def test_quadrant_mode_needs_the_frame_size_unless_rois_are_given():
    scenes = quad_scenes()
    jm, _, _ = build_quadrants(scenes, frame_size=None)
    with pytest.raises(RuntimeError, match="needs the video frame size"):
        jm.match()
    scope = QuadrantScope(tag_sets={"tl": QUAD_TAGS["top_left"]}, rois={"tl": (0, 0, 1800, 1800)})
    jm, _, _ = build_quadrants(scenes, scope=scope, frame_size=None)
    assert jm.match().frame_size is None


def test_inference_in_a_region_less_scope_uses_the_body_positions():
    scenes = quad_scenes()
    scope = QuadrantScope(infer=True, expected_size=4, min_size=3, min_support=30)
    jm, _, _ = build_quadrants(scenes, scope=scope)
    result = jm.match()
    assert {q: r.tags for q, r in result.regions.items()} == QUAD_TAGS


# ------------------------------------------------------------------------------ sidecar
def test_sidecar_round_trip_flat(tmp_path):
    scene = make_scene(6, miss_rate=0.05, absent={1: (30, 380)})
    result, _ = flat_run(scene)
    path = tmp_path / "x.identity.npz"
    write_sidecar(path, result, session="s1", meta={"options": {"a": 1}, "x": np.int64(3)})
    side = np.load(path)
    assert side["frames"].tolist() == list(range(400))
    for key in ("tag", "confidence", "source", "disputed", "flag", "region", "inst_index"):
        assert side[key].shape == (400, 4)
    np.testing.assert_array_equal(side["tag"], result.slot_tag)
    assert side["region_names"].tolist() == [FLAT_REGION]
    assert side["source_names"].tolist()[1] == "decode"
    assert side["flag_names"].tolist() == list(SlotFlag.NAMES)
    assert str(side["session"]) == "s1"
    assert side["tags__all"].tolist() == TAGS
    assert side["tags_qualified__all"].tolist() == [f"s1:ArUcoTag#{t}" for t in TAGS]
    assert str(side["status__all"]) == "ok"
    meta = json.loads(str(side["meta"]))
    assert meta["options"] == {"a": 1}
    assert meta["x"] == "3" or meta["x"] == 3
    tag_slot, tag_xy, tag_src = side["tag_slot__all"], side["tag_xy__all"], side["tag_source__all"]
    assert tag_slot.shape == (400, 4)
    for f, k in zip(*np.nonzero(tag_slot >= 0), strict=True):
        assert result.slot_tag[f, tag_slot[f, k]] == TAGS[k]
    interpolated = tag_src == Source.INTERPOLATED
    assert interpolated.any()
    assert (tag_slot[interpolated] == -1).all()
    assert np.isfinite(tag_xy[interpolated]).all()
    assert side["tag_confidence__all"].shape == (400, 4)


def test_sidecar_without_session_and_with_unsolved_regions(tmp_path):
    scenes = quad_scenes()
    sets = dict(QUAD_TAGS)
    sets["top_right"] = [20, 21]
    result = build_quadrants(scenes, tag_sets=sets)[0].match()
    path = tmp_path / "q.npz"
    write_sidecar(path, result)
    side = np.load(path)
    assert str(side["session"]) == ""
    assert side["region_names"].tolist() == QUADRANTS
    assert side["tags_qualified__top_left"].tolist() == [
        f"ArUcoTag#{t}" for t in QUAD_TAGS["top_left"]
    ]
    assert str(side["status__top_right"]) == "unsupported_tagset"
    assert "tag_xy__top_right" not in side.files
    assert "tag_xy__top_left" in side.files
    assert json.loads(str(side["meta"])) == {}


# -------------------------------------------------- output contract: reason, tracklet_id, schema
def test_sidecar_schema_reason_and_tracklet_arrays(tmp_path):
    scene = make_scene(61, n_frames=300, decode_rate=0.5, ghost_rate=0.3, miss_rate=0.05)
    result, _ = flat_run(scene)
    path = tmp_path / "x.identity.npz"
    write_sidecar(path, result)
    side = np.load(path)
    assert int(side["schema_version"]) == SIDECAR_SCHEMA_VERSION == 2
    assert side["reason_names"].tolist() == list(REASON_NAMES)
    reason, tag = side["reason"], side["tag"]
    valid = scene.table.valid
    # a reason exactly where a valid instance has no tag
    assert ((reason > 0) == (valid & (tag < 0))).all()
    assert reason.dtype == np.int8
    assert (reason[tag >= 0] == 0).all()
    assert reason.max() > 0
    assert set(np.unique(reason)) <= {int(r) for r in Reason}
    # reasons agree with the flag where the flag explains the abstention
    flag = side["flag"]
    plain = (flag > 0) & (flag != SlotFlag.LOW_CONFIDENCE) & (tag < 0)
    assert (reason[plain] == flag[plain]).all()
    low = flag == SlotFlag.LOW_CONFIDENCE
    assert set(np.unique(reason[low & (tag < 0)])) <= {
        Reason.BELOW_MIN_CONFIDENCE,
        Reason.LOW_CONFIDENCE_IN_PILE,
        Reason.NO_ANCHOR_IN_TRACKLET,
    }
    # tracklet ids: one per instance of the kept set, -1 elsewhere, shared by the frames of a bee
    tid = side["tracklet_id"]
    labels = result.regions[FLAT_REGION].labels
    assert labels is not None and labels.tracklets is not None
    assert int(side["tracklet_offset__all"]) == 0
    # (the region table is left-packed, so compare per tracklet, not per slot)
    np.testing.assert_array_equal(np.bincount(tid[tid >= 0]), labels.tracklets.length)
    assert (tid[~valid] == -1).all()
    assert (tid >= 0).any() and (tid < 0)[valid].any()  # ghosts / duplicates are in no tracklet
    assert side["decode_demoted"].dtype == bool
    assert side["decode_demoted"].shape == tag.shape


def test_reasons_of_unsolved_regions_and_instances_outside_every_region(tmp_path):
    scenes = quad_scenes()
    sets = dict(QUAD_TAGS)
    sets["top_right"] = [20, 21]  # unsupported
    result = build_quadrants(scenes, tag_sets=sets)[0].match()
    n = [s.table.n_slots for s in scenes]
    right = slice(n[0], n[0] + n[1])
    valid = result.table.valid
    unsolved = result.slot_reason[:, right][valid[:, right]]
    assert (unsolved == Reason.REGION_UNSUPPORTED_TAGSET).all() and unsolved.size > 0
    assert "top_right" not in result.tracklet_offset
    assert (result.slot_tracklet[:, right] == -1).all()
    # the tracklet ids of the solved regions tile 0..total-1 without collisions
    start = 0
    for name in ("top_left", "bottom_left", "bottom_right"):
        run = result.regions[name]
        assert run.labels is not None and run.labels.tracklets is not None
        assert result.tracklet_offset[name] == start
        ids = np.unique(result.slot_tracklet[run.member])
        ids = ids[ids >= 0]
        assert ids.min() == start
        assert ids.max() == start + run.labels.tracklets.n_tracklets - 1
        start += run.labels.tracklets.n_tracklets
    assert result.slot_tracklet.max() == start - 1
    path = tmp_path / "q.npz"
    write_sidecar(path, result)
    side = np.load(path)
    assert int(side["tracklet_offset__top_right"]) == -1
    assert int(side["tracklet_offset__bottom_left"]) == result.tracklet_offset["bottom_left"]
    # an instance in no region
    only_tl = QuadrantScope(tag_sets={"tl": QUAD_TAGS["top_left"]}, rois={"tl": (0, 0, 1800, 1800)})
    outside = build_quadrants(scenes, scope=only_tl)[0].match()
    n_tl = scenes[0].table.n_slots
    out_valid = outside.table.valid[:, n_tl:]
    assert (outside.slot_reason[:, n_tl:][out_valid] == Reason.OUTSIDE_REGIONS).all()
    assert (outside.slot_tracklet[:, n_tl:] == -1).all()


def test_reasons_of_a_region_without_data_and_of_instances_without_a_position():
    scene = make_scene(62, n_frames=30)
    result, _ = flat_run(scene, decodes={})
    valid = scene.table.valid
    assert (result.slot_reason[valid] == Reason.REGION_NO_DATA).all()
    assert (result.slot_reason[~valid] == 0).all()
    assert (result.slot_tracklet == -1).all()
    # a SLEAP instance without any finite node is listed in the table but cannot be tracked
    solved = make_scene(63, n_frames=40)
    solved.table.xy[3, 0] = np.nan
    solved.table.tag_xy[3, 0] = np.nan
    solved.table.inst_index[3, 0] = 0
    result, _ = flat_run(solved)
    assert result.slot_reason[3, 0] == Reason.NO_POSITION
    assert result.slot_tag[3, 0] == -1


def test_region_reasons_refine_low_confidence_only_where_a_detail_exists():
    # two bees are never decoded and nothing tells them apart: they stay low-confidence
    scene = make_scene(64, n_frames=100, absent={2: (0, 99), 3: (0, 99)})
    result, _ = flat_run(scene)
    labels = result.regions[FLAT_REGION].labels
    assert labels is not None and labels.low_conf_reason is not None
    detailed = region_reasons(labels)
    low = labels.slot_flag == SlotFlag.LOW_CONFIDENCE
    assert low.any() and (detailed[low] == labels.low_conf_reason[low]).all()
    labels.low_conf_reason = None  # hand-built labels: the flag's own code
    assert (region_reasons(labels)[low] == Reason.BELOW_MIN_CONFIDENCE).all()
    # hand-built labels without tracklets / demotions still scatter onto the instances
    labels.tracklets = labels.demoted = None
    result.refresh()
    assert (result.slot_tracklet == -1).all() and not result.slot_demoted.any()


def test_withheld_labels_get_their_reason():
    scene = make_scene(65, n_frames=400, decode_rate=0.4, absent={3: (0, 399)})
    result, _ = flat_run(scene)
    params = DiagnosticsParams(min_median_tracklet_len=1e9)  # every window is uncertain
    statuses, _ = assess_result(result, params)
    assert apply_policy(result, statuses, params) > 0
    withheld = result.slot_flag == SlotFlag.WITHHELD
    assert withheld.any()
    assert (result.slot_reason[withheld] == Reason.UNTRUSTED_WINDOW_WITHHELD).all()
    assert ((result.slot_reason > 0) == (result.table.valid & (result.slot_tag < 0))).all()


def test_an_empty_table_and_empty_vote_lists_are_handled():
    nothing = np.full((0, 2, 2), np.nan)
    table = InstanceTable(
        0, nothing, nothing.copy(), np.full((0, 2), np.nan), np.full((0, 2), -1),
        np.full((0, 2), -1), np.full((0, 2), np.nan),
    )  # fmt: skip
    with pytest.raises(ValueError, match="no frames"):
        JointMatching(table, FakeSource({}), closed_set=TAGS).match()
    votes = JointMatching._to_votes([], np.zeros((3, 2), dtype=np.int64))
    assert len(votes) == 0
