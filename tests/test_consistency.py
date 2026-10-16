"""Temporal-consistency stage: ghosts, duplicates, chain veto, interpolation, metrics."""

import numpy as np
import pytest

from naps import consistency as cs
from naps.consistency import (
    ConsistencyParams,
    allowed_displacement,
    anchors_off_chain,
    apply_consistency,
    bridge_violations,
    drop_duplicates,
    duplicate_stats,
    ghost_mask,
    interpolate,
    jump_matrix,
    recovered_run_stats,
    resolve_tag,
    resolve_tracks,
)
from naps.sources import Source

DECODE, TRACKLET, ELIM, INTERP = (
    int(Source.DECODE),
    int(Source.TRACKLET),
    int(Source.ELIMINATION),
    int(Source.INTERPOLATED),
)


def track(n, *segments):
    """Positions ``(n, 2)`` from ``(first, last, x)`` segments (y = 0); NaN elsewhere."""
    pos = np.full((n, 2), np.nan)
    for first, last, x in segments:
        pos[first : last + 1] = [x, 0.0]
    return pos


def kinds(n, anchors=(), cands=()):
    """``(is_anchor, cand)`` boolean vectors from ``(first, last)`` ranges."""
    is_anchor, cand = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    for a, b in anchors:
        is_anchor[a : b + 1] = True
    for a, b in cands:
        cand[a : b + 1] = True
    return is_anchor, cand


# --------------------------------------------------------------------- allowed_displacement
def test_allowed_displacement_follows_the_table():
    for lag, px in zip(cs.GATE_FRAMES, cs.GATE_PX, strict=True):
        assert allowed_displacement(lag) == pytest.approx(px)
    assert allowed_displacement(7.5) == pytest.approx(160 + (237 - 160) * 0.5)
    assert allowed_displacement(0) == pytest.approx(100.0)  # a lag below 1 counts as 1
    assert allowed_displacement(np.array([1, 2, 3])).tolist() == [100.0, 133.0, 142.0]


def test_allowed_displacement_caps_and_scale():
    assert allowed_displacement(10_000) == pytest.approx(1420.0)  # table saturates
    assert allowed_displacement(10_000, ConsistencyParams(gate_scale=3.0)) == pytest.approx(1450.0)
    assert allowed_displacement(1, ConsistencyParams(gate_scale=0.5)) == pytest.approx(50.0)
    assert allowed_displacement(1, ConsistencyParams(hard_px=40.0)) == pytest.approx(40.0)
    assert allowed_displacement(5, ConsistencyParams(hard_px=20.0)) == pytest.approx(100.0)


# --------------------------------------------------------------------------------- ghosts
def frame_of(*points):
    xy = np.array([points], dtype=float)
    return xy, np.isfinite(xy[..., 0])


def test_unclaimed_instance_next_to_a_claimed_one_is_a_ghost():
    xy, valid = frame_of([0, 0], [30, 0], [200, 0])
    claimed = np.array([[True, False, False]])
    score = np.array([[0.5, 0.9, 0.9]])
    assert ghost_mask(xy, valid, claimed, score).tolist() == [[False, True, False]]


def test_claimed_instances_are_never_ghosts():
    xy, valid = frame_of([0, 0], [10, 0])
    claimed = np.array([[True, True]])
    score = np.array([[0.1, 0.1]])
    assert not ghost_mask(xy, valid, claimed, score).any()


def test_a_neighbour_beyond_the_gate_is_not_a_ghost_without_a_pose():
    xy, valid = frame_of([0, 0], [60, 0])
    claimed = np.array([[True, False]])
    score = np.array([[0.5, 0.5]])
    assert not ghost_mask(xy, valid, claimed, score).any()


def pose_pair(offset_second, n_nodes=4, missing=()):
    base = np.array([[0, 0], [5, 0], [0, 5], [5, 5]], dtype=np.float32)[:n_nodes]
    first = base.copy()
    second = base + np.array(offset_second, dtype=np.float32)
    for node in missing:
        second[node] = np.nan
    return np.stack([first, second])[None]  # (1, 2, N, 2)


def test_pose_overlap_marks_a_ghost_up_to_the_pose_gate():
    xy, valid = frame_of([0, 0], [60, 0])
    claimed = np.array([[True, False]])
    score = np.array([[0.5, 0.5]])
    overlapping = pose_pair([10, 0])  # nodes 10 px apart: a copy
    apart = pose_pair([40, 0])  # nodes 40 px apart: a different bee
    assert ghost_mask(xy, valid, claimed, score, pose=overlapping).tolist() == [[False, True]]
    assert not ghost_mask(xy, valid, claimed, score, pose=apart).any()


def test_pose_overlap_needs_enough_shared_nodes():
    xy, valid = frame_of([0, 0], [60, 0])
    claimed = np.array([[True, False]])
    score = np.array([[0.5, 0.5]])
    two_shared = pose_pair([10, 0], missing=(2, 3))
    assert not ghost_mask(xy, valid, claimed, score, pose=two_shared).any()
    three_shared = pose_pair([10, 0], missing=(3,))
    assert ghost_mask(xy, valid, claimed, score, pose=three_shared).any()


def test_pose_beyond_the_pose_gate_is_ignored():
    xy, valid = frame_of([0, 0], [95, 0])
    claimed = np.array([[True, False]])
    score = np.array([[0.5, 0.5]])
    assert not ghost_mask(xy, valid, claimed, score, pose=pose_pair([10, 0])).any()


def test_of_two_unclaimed_neighbours_the_lower_score_is_the_ghost():
    xy, valid = frame_of([0, 0], [30, 0])
    claimed = np.zeros((1, 2), dtype=bool)
    assert ghost_mask(xy, valid, claimed, np.array([[0.9, 0.2]])).tolist() == [[False, True]]
    assert ghost_mask(xy, valid, claimed, np.array([[0.2, 0.9]])).tolist() == [[True, False]]


def test_equal_scores_keep_the_lower_slot_and_nan_counts_as_zero():
    xy, valid = frame_of([0, 0], [30, 0])
    claimed = np.zeros((1, 2), dtype=bool)
    assert ghost_mask(xy, valid, claimed, np.array([[0.5, 0.5]])).tolist() == [[False, True]]
    assert ghost_mask(xy, valid, claimed, np.array([[np.nan, 0.0]])).tolist() == [[False, True]]


def test_unclaimed_pairs_beyond_the_gate_are_not_ghosts():
    xy, valid = frame_of([0, 0], [70, 0])
    claimed = np.zeros((1, 2), dtype=bool)
    assert not ghost_mask(xy, valid, claimed, np.array([[0.9, 0.1]])).any()


def test_a_ghost_next_to_two_claimed_instances_is_handled_once():
    xy, valid = frame_of([0, 0], [20, 0], [10, 10])
    claimed = np.array([[True, True, False]])
    assert ghost_mask(xy, valid, claimed, np.full((1, 3), 0.5)).tolist() == [[False, False, True]]


def test_invalid_slots_never_take_part():
    xy, valid = frame_of([0, 0], [np.nan, np.nan], [10, 0])
    claimed = np.array([[True, False, False]])
    assert ghost_mask(xy, valid, claimed, np.full((1, 3), 0.5)).tolist() == [[False, False, True]]


def random_frames(seed, n_frames=40, n_slots=4):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 120, size=(n_frames, n_slots, 2))
    valid = rng.random((n_frames, n_slots)) < 0.85
    xy[~valid] = np.nan
    claimed = (rng.random((n_frames, n_slots)) < 0.4) & valid
    score = rng.random((n_frames, n_slots))
    pose = rng.uniform(0, 120, size=(n_frames, n_slots, 4, 2)).astype(np.float32)
    return xy, valid, claimed, score, pose


@pytest.mark.parametrize("seed", range(5))
def test_ghost_mask_gives_the_same_result_in_blocks(seed, monkeypatch):
    xy, valid, claimed, score, pose = random_frames(seed)
    whole = ghost_mask(xy, valid, claimed, score, pose=pose)
    monkeypatch.setattr(cs, "_PAIR_CHUNK", 7)
    assert np.array_equal(whole, ghost_mask(xy, valid, claimed, score, pose=pose))
    assert not (whole & claimed).any()
    assert not (whole & ~valid).any()


# ---------------------------------------------------------------------------- duplicates
def two_tags(n, src0, src1, x0, x1):
    xy = np.full((n, 2, 2), np.nan)
    xy[:, 0] = [x0, 0.0]
    xy[:, 1] = [x1, 0.0]
    src = np.zeros((n, 2), dtype=np.int8)
    src[:, 0], src[:, 1] = src0, src1
    return xy, src, np.full((n, 2), 0.99, dtype=np.float32)


def test_a_candidate_on_top_of_a_decoded_tag_is_dropped():
    xy, src, conf = two_tags(5, DECODE, TRACKLET, 0.0, 30.0)
    anchor_xy = np.where((src == DECODE)[..., None], xy, np.nan)
    drop = drop_duplicates(xy, src, conf, anchor_xy)
    assert drop[:, 1].all()
    assert not drop[:, 0].any()  # the anchor is never dropped


def test_far_apart_candidates_are_kept_and_nothing_to_do_without_candidates():
    xy, src, conf = two_tags(5, DECODE, TRACKLET, 0.0, 300.0)
    anchor_xy = np.where((src == DECODE)[..., None], xy, np.nan)
    assert not drop_duplicates(xy, src, conf, anchor_xy).any()
    xy2, src2, conf2 = two_tags(5, DECODE, DECODE, 0.0, 10.0)
    assert not drop_duplicates(xy2, src2, conf2, xy2).any()


def test_of_two_close_candidates_the_less_confident_is_dropped():
    xy, src, conf = two_tags(3, TRACKLET, ELIM, 0.0, 30.0)
    no_anchor = np.full_like(xy, np.nan)
    conf[:, 0], conf[:, 1] = 0.995, 0.991
    assert drop_duplicates(xy, src, conf, no_anchor).tolist() == [[False, True]] * 3
    conf[:, 0], conf[:, 1] = 0.991, 0.995
    assert drop_duplicates(xy, src, conf, no_anchor).tolist() == [[True, False]] * 3


def test_two_equally_confident_close_candidates_are_both_dropped():
    xy, src, conf = two_tags(3, TRACKLET, ELIM, 0.0, 30.0)
    no_anchor = np.full_like(xy, np.nan)
    assert drop_duplicates(xy, src, conf, no_anchor).all()
    conf[:] = np.nan  # NaN confidence counts as 0: a tie as well
    assert drop_duplicates(xy, src, conf, no_anchor).all()


def test_tags_that_were_close_in_the_decoded_data_are_exempt():
    n = 60
    xy = np.full((n, 2, 2), np.nan)
    xy[:, 0] = [0.0, 0.0]
    xy[:, 1] = [30.0, 0.0]
    src = np.zeros((n, 2), dtype=np.int8)
    src[:, 0] = DECODE
    src[:, 1] = TRACKLET
    src[5, 1] = DECODE  # both decoded 30 px apart at frame 5: bees really touch
    conf = np.full((n, 2), 0.99, dtype=np.float32)
    anchor_xy = np.where((src == DECODE)[..., None], xy, np.nan)
    drop = drop_duplicates(xy, src, conf, anchor_xy)
    exempt = np.zeros(n, dtype=bool)
    exempt[0:16] = True  # frame 5 +- 10
    assert not drop[exempt, 1].any()
    assert drop[~exempt, 1][src[~exempt, 1] == TRACKLET].all()
    # without the window nothing is exempt
    strict = ConsistencyParams(anchor_close_window=0)
    assert drop_duplicates(xy, src, conf, anchor_xy, strict)[6, 1]


def test_anchor_closeness_is_computed_in_blocks(monkeypatch):
    n = 30
    xy, src, conf = two_tags(n, DECODE, TRACKLET, 0.0, 30.0)
    src[3, 1] = DECODE
    anchor_xy = np.where((src == DECODE)[..., None], xy, np.nan)
    whole = drop_duplicates(xy, src, conf, anchor_xy)
    monkeypatch.setattr(cs, "_PAIR_CHUNK", 4)
    assert np.array_equal(whole, drop_duplicates(xy, src, conf, anchor_xy))


# ----------------------------------------------------------------------------- chains
def test_a_consistent_candidate_run_next_to_anchors_is_kept():
    n = 30
    pos = track(n, (0, 29, 0.0))
    is_anchor, cand = kinds(n, anchors=[(0, 9), (20, 29)], cands=[(10, 19)])
    drop, reason = resolve_tag(pos, is_anchor, cand)
    assert not drop.any()
    assert not reason.any()


def test_no_candidates_or_no_positions_means_nothing_to_drop():
    pos = track(10, (0, 9, 0.0))
    only_anchors = kinds(10, anchors=[(0, 9)])
    assert not resolve_tag(pos, *only_anchors)[0].any()
    nothing = np.full((10, 2), np.nan)
    assert not resolve_tag(nothing, *kinds(10, cands=[(0, 9)]))[0].any()


@pytest.mark.parametrize("policy", ["truncate", "drop"])
def test_a_teleporting_candidate_run_is_dropped_with_the_anchor_conflict_reason(policy):
    pos = track(30, (0, 9, 0.0), (10, 14, 1000.0), (20, 29, 0.0))
    is_anchor, cand = kinds(30, anchors=[(0, 9), (20, 29)], cands=[(10, 14)])
    drop, reason = resolve_tag(pos, is_anchor, cand, ConsistencyParams(run_policy=policy))
    assert drop[10:15].all()
    assert not drop[~cand].any()
    assert (reason[10:15] == cs.REASON_ANCHOR_CONFLICT).all()


def test_two_conflicting_floating_runs_keep_the_longer_one():
    pos = track(30, (0, 9, 0.0), (20, 24, 2000.0))
    _, cand = kinds(30, cands=[(0, 9), (20, 24)])
    drop, reason = resolve_tag(pos, np.zeros(30, dtype=bool), cand)
    assert drop[20:25].all()
    assert not drop[0:10].any()
    assert (reason[20:25] == cs.REASON_CANDIDATE_CONFLICT).all()


def test_the_drop_policy_removes_a_whole_run_with_an_impossible_step():
    pos = track(40, (0, 4, 0.0), (5, 9, 1000.0), (20, 39, 1000.0))
    is_anchor, cand = kinds(40, anchors=[(20, 39)], cands=[(0, 9)])
    dropped, reason = resolve_tag(pos, is_anchor, cand, ConsistencyParams(run_policy="drop"))
    assert dropped[0:10].all()
    assert (reason[0:10] == cs.REASON_RUN).all()
    # 'truncate' only drops the part that disagrees with the anchors
    kept, why = resolve_tag(pos, is_anchor, cand, ConsistencyParams(run_policy="truncate"))
    assert kept[0:5].all()
    assert not kept[5:10].any()
    assert (why[0:5] == cs.REASON_ANCHOR_CONFLICT).all()


def test_the_drop_policy_with_nothing_consistent_keeps_nothing():
    pos = track(20, (0, 4, 0.0), (5, 9, 1000.0))
    _, cand = kinds(20, cands=[(0, 9)])
    drop, reason = resolve_tag(
        pos, np.zeros(20, dtype=bool), cand, ConsistencyParams(run_policy="drop")
    )
    assert drop[0:10].all()
    assert (reason[0:10] == cs.REASON_RUN).all()


def test_short_floating_chunks_are_dropped_when_min_seg_len_is_set():
    params = ConsistencyParams(min_seg_len=3)
    # a lone two-frame floating chunk
    pos = track(10, (0, 1, 0.0))
    drop, reason = resolve_tag(pos, np.zeros(10, dtype=bool), kinds(10, cands=[(0, 1)])[1], params)
    assert drop[0:2].all()
    assert (reason[0:2] == cs.REASON_SHORT).all()
    # in the middle of a floating chain: no anchor on either side
    mid = track(40, (0, 9, 0.0), (12, 13, 0.0), (16, 25, 0.0))
    cand = kinds(40, cands=[(0, 9), (12, 13), (16, 25)])[1]
    dropped, why = resolve_tag(mid, np.zeros(40, dtype=bool), cand, params)
    assert dropped[12:14].all()
    assert not dropped[0:10].any()
    assert not dropped[16:26].any()
    assert (why[12:14] == cs.REASON_SHORT).all()
    # last in the chain, previous chunk is floating too
    tail = track(30, (0, 9, 0.0), (12, 13, 0.0))
    dropped2, _ = resolve_tag(
        tail, np.zeros(30, dtype=bool), kinds(30, cands=[(0, 9), (12, 13)])[1], params
    )
    assert dropped2[12:14].all()


def test_short_chunks_next_to_an_anchor_are_kept():
    params = ConsistencyParams(min_seg_len=3)
    pos = track(30, (0, 9, 0.0), (10, 11, 0.0), (20, 21, 0.0), (22, 29, 0.0))
    is_anchor, cand = kinds(30, anchors=[(0, 9), (22, 29)], cands=[(10, 11), (20, 21)])
    drop, _ = resolve_tag(pos, is_anchor, cand, params)
    assert not drop.any()  # (10, 11) follows an anchor, (20, 21) precedes one


def test_resolve_tracks_works_per_tag():
    n = 30
    xy = np.full((n, 2, 2), np.nan)
    src = np.zeros((n, 2), dtype=np.int8)
    xy[:, 0] = track(n, (0, 29, 0.0))
    src[0:10, 0], src[20:30, 0] = DECODE, DECODE
    src[10:15, 0] = TRACKLET
    xy[10:15, 0] = [1000.0, 0.0]
    xy[:, 1] = track(n, (0, 29, 500.0))
    src[:, 1] = DECODE
    drop, reason = resolve_tracks(xy, src)
    assert drop[10:15, 0].all()
    assert not drop[:, 1].any()
    assert reason[10:15, 0].tolist() == [cs.REASON_ANCHOR_CONFLICT] * 5


def test_anchors_off_chain_flags_a_decoder_glitch():
    n = 101
    xy = np.full((n, 3, 2), np.nan)
    src = np.zeros((n, 3), dtype=np.int8)
    xy[:, 0] = track(n, (0, 100, 0.0), (50, 52, 1000.0))
    src[:, 0] = DECODE
    xy[:, 1] = track(n, (0, 100, 300.0))
    src[:, 1] = DECODE
    # tag 2 has no anchor at all
    xy[5:9, 2] = [10.0, 10.0]
    src[5:9, 2] = TRACKLET
    off = anchors_off_chain(xy, src)
    assert off[:, 0].tolist() == [50 <= f <= 52 for f in range(n)]
    assert not off[:, 1].any()
    assert not off[:, 2].any()


# ----------------------------------------------------------------------- apply_consistency
def test_disabled_or_candidate_free_input_is_untouched():
    xy, src, conf = two_tags(6, DECODE, DECODE, 0.0, 10.0)
    result = apply_consistency(xy, src, conf, xy)
    assert not result.drop.any()
    assert result.info["dropped_total"] == 0
    assert result.info["n_candidates"] == 0
    xy2, src2, conf2 = two_tags(6, DECODE, TRACKLET, 0.0, 10.0)
    off = apply_consistency(xy2, src2, conf2, xy2, ConsistencyParams(enabled=False))
    assert not off.drop.any()
    assert off.info["enabled"] is False
    assert off.info["n_candidates"] == 6


def test_apply_consistency_counts_the_reasons():
    n = 40
    xy = np.full((n, 3, 2), np.nan)
    src = np.zeros((n, 3), dtype=np.int8)
    conf = np.full((n, 3), 0.995, dtype=np.float32)
    xy[:, 0] = track(n, (0, 39, 0.0))
    src[:, 0] = DECODE
    # tag 1: a teleporting candidate run among anchors
    xy[:, 1] = track(n, (0, 39, 500.0), (10, 14, 1900.0))
    src[:, 1] = DECODE
    src[10:15, 1] = TRACKLET
    # tag 2: a candidate sitting 20 px from tag 0 in frames 30-34
    xy[30:35, 2] = [20.0, 0.0]
    src[30:35, 2] = ELIM
    anchor_xy = np.where((src == DECODE)[..., None], xy, np.nan)
    result = apply_consistency(xy, src, conf, anchor_xy)
    info = result.info
    assert info["n_candidates"] == 10
    assert info["dropped_total"] == 10
    assert info["dropped_duplicate"] == 5
    assert info["dropped_anchor_conflict"] == 5
    assert info["dropped_fraction"] == 1.0
    assert (result.reason[30:35, 2] == cs.REASON_DUPLICATE).all()
    assert (result.reason[10:15, 1] == cs.REASON_ANCHOR_CONFLICT).all()
    assert not result.drop[src == DECODE].any()


# ------------------------------------------------------------------------- interpolation
def test_interpolate_fills_a_plausible_gap_linearly():
    xy = np.full((10, 1, 2), np.nan)
    xy[0, 0], xy[5, 0] = [0.0, 0.0], [30.0, 10.0]
    filled, out, refused = interpolate(xy)
    assert filled[:, 0].tolist() == [False, True, True, True, True, False] + [False] * 4
    assert out[3, 0].tolist() == pytest.approx([18.0, 6.0])
    assert refused == 0
    assert np.isnan(xy[1, 0, 0])  # the input is not modified
    assert np.isnan(out[6:, 0]).all()


def test_interpolate_refuses_distant_ends_but_not_when_disabled():
    xy = np.full((10, 1, 2), np.nan)
    xy[0, 0], xy[5, 0] = [0.0, 0.0], [200.0, 0.0]
    filled, _, refused = interpolate(xy)
    assert not filled.any()
    assert refused == 4
    filled2, out2, refused2 = interpolate(xy, ConsistencyParams(enabled=False))
    assert filled2.sum() == 4
    assert refused2 == 0
    assert out2[1, 0, 0] == pytest.approx(40.0)


def test_interpolate_refuses_points_on_top_of_another_tag():
    xy = np.full((10, 2, 2), np.nan)
    xy[0, 0], xy[5, 0] = [0.0, 0.0], [30.0, 0.0]
    xy[:, 1] = [15.0, 10.0]  # another bee sits on the path
    filled, _, refused = interpolate(xy)
    assert not filled[:, 0].any()
    assert refused == 4
    xy[:, 1] = [15.0, 100.0]
    assert interpolate(xy)[0][:, 0].sum() == 4
    partial = xy.copy()
    partial[2, 1] = [15.0, 10.0]  # only frame 2 collides
    filled3, _, refused3 = interpolate(partial)
    assert filled3[:, 0].tolist()[1:5] == [True, False, True, True]
    assert refused3 == 1


def test_interpolate_ignores_long_and_zero_gaps():
    xy = np.full((40, 1, 2), np.nan)
    xy[0, 0], xy[1, 0] = [0.0, 0.0], [1.0, 0.0]  # adjacent: no gap
    xy[30, 0] = [2.0, 0.0]  # 28 missing frames: beyond gap_max
    filled, _, refused = interpolate(xy)
    assert not filled.any()
    assert refused == 0
    assert interpolate(xy, ConsistencyParams(gap_max=40))[0].sum() == 28


# ----------------------------------------------------------------------------- metrics
def metric_tracks():
    xy = np.full((6, 2, 2), np.nan)
    src = np.zeros((6, 2), dtype=np.int8)
    xy[:, 0, 0] = [0, 10, 500, 510, 520, 1000]  # steps 10, 490, 10, 10, 480
    xy[:, 0, 1] = 0
    src[:, 0] = [DECODE, DECODE, TRACKLET, TRACKLET, TRACKLET, DECODE]
    xy[:, 1, 0] = [0, 150, 300, np.nan, 400, 420]
    xy[:, 1, 1] = 100
    src[:, 1] = [DECODE, DECODE, DECODE, 0, TRACKLET, TRACKLET]
    return xy, src


def test_jump_matrix_counts_by_class():
    xy, src = metric_tracks()
    out = jump_matrix(xy, src)
    assert out["anchor_anchor"]["n"] == 3  # tag 0: one step; tag 1: two 150 px steps
    assert out["anchor_anchor"]["gt100"] == 2
    assert out["rec_rec"]["n"] == 3
    assert out["rec_rec"]["gt100"] == 0
    assert out["anchor_rec"]["n"] == 2  # tag 0: the 490 px and the 480 px step
    assert out["anchor_rec"]["gt300"] == 2
    assert out["anchor_rec"]["rate_gt300"] == pytest.approx(1.0)
    empty = jump_matrix(xy[:2, :1], np.zeros((2, 1), dtype=np.int8))
    assert empty["rec_rec"]["n"] == 0
    assert empty["rec_rec"]["rate_gt100"] is None


def test_jump_matrix_attributes_glitch_steps():
    xy, src = metric_tracks()
    off = np.zeros(src.shape, dtype=bool)
    off[5, 0] = True  # the last anchor of tag 0 contradicts its own track
    out = jump_matrix(xy, src, off=off)["anchor_rec"]
    assert out["n_touching_anchor_glitch"] == 1
    assert out["gt300_anchor_glitch"] == 1
    assert out["gt300_own"] == 1  # the 490 px step into the recovered run is the engine's


def test_recovered_run_stats():
    xy, src = metric_tracks()
    stats = recovered_run_stats(xy, src)
    assert stats["n_runs"] == 2
    assert stats["n_recovered_labels"] == 3 + 2
    assert stats["n_runs_with_jump"] == 0
    xy[3, 0, 0] = 3000.0
    stats = recovered_run_stats(xy, src)
    assert stats["n_runs_with_jump"] == 1
    assert stats["labels_in_runs_with_jump"] == 3
    assert stats["frac_labels_in_runs_with_jump"] == pytest.approx(3 / 5)
    single = np.zeros_like(src)
    single[2, 0] = TRACKLET  # a one-frame run has no step to judge
    assert recovered_run_stats(xy, single)["n_runs"] == 1
    assert recovered_run_stats(xy, single)["n_runs_with_jump"] == 0
    none = recovered_run_stats(xy, np.zeros_like(src))
    assert none["frac_labels_in_runs_with_jump"] == 0.0
    assert none["n_runs"] == 0


def test_duplicate_stats_split_by_class():
    xy = np.full((2, 3, 2), np.nan)
    src = np.zeros((2, 3), dtype=np.int8)
    xy[0, 0], xy[0, 1], xy[0, 2] = [0, 0], [20, 0], [30, 0]
    src[0] = [DECODE, DECODE, TRACKLET]
    xy[1, 0], xy[1, 1] = [0, 0], [500, 0]
    src[1] = [TRACKLET, TRACKLET, 0]
    stats = duplicate_stats(xy, src, gate=40.0)
    assert stats["anchor_pairs_near"] == 1
    assert stats["anchor_recovered_pairs_near"] == 2
    assert stats["recovered_recovered_pairs_near"] == 0
    assert stats["recovered_labels_near_other_tag"] == 1
    assert stats["recovered_labels_total"] == 3


def test_bridge_violations():
    xy, src = metric_tracks()
    out = bridge_violations(xy, src)
    # tag 0: steps 490 (anchor->rec, g=1) and 480 (rec->anchor); tag 1: 3 -> 4 bridged over a gap
    assert out["n_violations"] == 2
    assert out["n_gt300_consecutive"] == 2
    assert out["n_violations_own"] == 2
    assert out["n_violations_anchor_glitch"] == 0
    assert out["n_links_with_recovered"] == 4 + 2
    off = np.zeros(src.shape, dtype=bool)
    off[5, 0] = True
    glitch = bridge_violations(xy, src, off=off)
    assert glitch["n_violations_own"] == 1
    assert glitch["n_violations_anchor_glitch"] == 1
    lone = bridge_violations(xy[:1], src[:1])  # a single frame has no links
    assert lone["n_violations"] == 0


# --------------------------------------------------------------------- property tests
def random_tags(seed):
    """Smooth tracks of 3 tags with anchors, plausible and teleporting candidate runs."""
    rng = np.random.default_rng(seed)
    n = 160
    xy = np.full((n, 3, 2), np.nan)
    src = np.zeros((n, 3), dtype=np.int8)
    conf = np.full((n, 3), 0.995, dtype=np.float32)
    for k in range(3):
        path = np.cumsum(rng.normal(0, 3, size=(n, 2)), axis=0) + np.array([0.0, 600.0 * k])
        f = 0
        while f < n:
            length = int(rng.integers(1, 12))
            kind = rng.choice(["anchor", "cand", "cand_bad", "none"], p=[0.4, 0.25, 0.1, 0.25])
            sl = slice(f, min(f + length, n))
            if kind == "anchor":
                xy[sl, k], src[sl, k] = path[sl], DECODE
            elif kind == "cand":
                xy[sl, k], src[sl, k] = path[sl], rng.choice([TRACKLET, ELIM])
            elif kind == "cand_bad":
                xy[sl, k], src[sl, k] = path[sl] + [1500.0, 0.0], TRACKLET
            f += length
    return xy, src, conf


@pytest.mark.parametrize("seed", range(60))
@pytest.mark.parametrize("policy", ["truncate", "drop"])
def test_anchors_survive_and_the_result_has_no_violations(seed, policy):
    xy, src, conf = random_tags(seed)
    params = ConsistencyParams(run_policy=policy)
    anchor_xy = np.where((src == DECODE)[..., None], xy, np.nan)
    result = apply_consistency(xy, src, conf, anchor_xy, params)
    assert not result.drop[src == DECODE].any(), "an anchor was dropped"
    assert not result.drop[src == 0].any()
    assert (result.reason[~result.drop] == 0).all()
    kept_xy = np.where(result.drop[..., None], np.nan, xy)
    kept_src = np.where(result.drop, 0, src).astype(np.int8)
    # every link that involves a recovered label is plausible, except where an anchor itself
    # contradicts its track (then the engine is not responsible)
    off = anchors_off_chain(kept_xy, kept_src, params)
    assert bridge_violations(kept_xy, kept_src, params, off)["n_violations_own"] == 0
    again = apply_consistency(xy, src, conf, anchor_xy, params)
    assert np.array_equal(result.drop, again.drop)
    dropped, reason = resolve_tracks(xy, src, params)
    assert not dropped[src == DECODE].any()
    assert (reason[dropped] > 0).all()
