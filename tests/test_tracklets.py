"""Tracklet linking: invariants, near-pass breaks, gates, SLEAP ids as hints, pose terms."""

import numpy as np
import pytest

from naps.tracklets import (
    TrackletParams,
    Tracklets,
    build_tracklets,
    dedupe_instances,
)
from tests.scenes import make_scene

NO_BREAKS = TrackletParams(amb_ratio=0.0, amb_add=0.0)  # switch the competing-instance break off


def field(*tracks):
    """Positions ``(F, S, 2)`` from one ``(F, 2)`` array per bee (slot k = bee k)."""
    xy = np.stack([np.asarray(t, dtype=float) for t in tracks], axis=1)
    return xy, np.isfinite(xy[..., 0])


def line(n, start, step):
    return np.array(start, dtype=float) + np.arange(n)[:, None] * np.array(step, dtype=float)


# ------------------------------------------------------------------------------ Tracklets
def test_tracklets_extents():
    tid = np.array([[0, 1], [0, -1], [-1, 1]], dtype=np.int32)
    tr = Tracklets(tid)
    assert tr.n_tracklets == 2
    assert tr.first.tolist() == [0, 0]
    assert tr.last.tolist() == [1, 2]
    assert tr.length.tolist() == [2, 2]


def test_tracklets_without_any_instance():
    tr = Tracklets(np.full((5, 3), -1, dtype=np.int32))
    assert tr.n_tracklets == 0
    assert tr.first.shape == tr.last.shape == tr.length.shape == (0,)


# ------------------------------------------------------------------------------ dedupe
def test_dedupe_keeps_the_higher_score_and_ignores_empty_slots():
    xy = np.array([[[0.0, 0.0], [10.0, 0.0], [np.nan, np.nan], [200.0, 0.0]]])
    score = np.array([[0.5, 0.9, 0.99, 0.1]])
    assert dedupe_instances(xy, score, 25.0).tolist() == [[False, True, False, True]]


def test_dedupe_tie_keeps_the_lower_slot_and_nan_counts_as_zero():
    xy = np.array([[[0.0, 0.0], [5.0, 0.0]], [[0.0, 0.0], [5.0, 0.0]]])
    score = np.array([[0.7, 0.7], [np.nan, 0.3]])
    assert dedupe_instances(xy, score, 25.0).tolist() == [[True, False], [False, True]]


def test_dedupe_a_dropped_duplicate_does_not_shield_its_neighbour():
    # A(0) B(20) C(40): B is a duplicate of A; C is 40 from A, so it stays
    xy = np.array([[[0.0, 0.0], [20.0, 0.0], [40.0, 0.0]]])
    score = np.array([[0.9, 0.8, 0.7]])
    assert dedupe_instances(xy, score, 25.0).tolist() == [[True, False, True]]


def test_dedupe_distance_is_strict_and_single_instances_pass():
    xy = np.array([[[0.0, 0.0], [25.0, 0.0]], [[0.0, 0.0], [np.nan, np.nan]]])
    score = np.array([[0.9, 0.8], [0.9, 0.8]])
    assert dedupe_instances(xy, score, 25.0).tolist() == [[True, True], [True, False]]


# ------------------------------------------------------------------------------ linking
def test_a_straight_walk_is_one_tracklet():
    xy, valid = field(line(30, (0, 0), (5, 2)))
    tr = build_tracklets(xy, valid)
    assert tr.n_tracklets == 1
    assert (tr.tid == 0).all()
    assert tr.length.tolist() == [30]


def test_two_bees_far_apart_are_two_tracklets():
    xy, valid = field(line(40, (0, 0), (3, 0)), line(40, (0, 400), (3, 0)))
    tr = build_tracklets(xy, valid)
    assert tr.n_tracklets == 2
    assert (tr.tid[:, 0] == tr.tid[0, 0]).all()
    assert (tr.tid[:, 1] == tr.tid[0, 1]).all()
    assert tr.tid[0, 0] != tr.tid[0, 1]


def test_invalid_slots_get_minus_one():
    xy, valid = field(line(10, (0, 0), (3, 0)))
    valid[4, 0] = False
    tr = build_tracklets(xy, valid)
    assert tr.tid[4, 0] == -1
    assert (tr.tid[[0, 1, 2, 3, 5, 6, 7, 8, 9], 0] == 0).all()  # the gap is bridged


def test_a_near_pass_breaks_the_tracklets_instead_of_swapping_them():
    # two bees cross at 8 px distance; slot = bee, so a swap would put both slots in one tracklet
    n = 80
    a = np.stack([np.linspace(0, 400, n), np.full(n, 0.0)], axis=1)
    b = np.stack([np.linspace(400, 0, n), np.full(n, 8.0)], axis=1)
    xy, valid = field(a, b)
    tr = build_tracklets(xy, valid)
    assert tr.n_tracklets > 2  # the crossing broke both
    for tid in range(tr.n_tracklets):
        slots = np.unique(np.nonzero(tr.tid == tid)[1])
        assert len(slots) == 1, "a tracklet changed bee at the near-pass"
    mid = n // 2
    assert tr.tid[mid - 30, 0] != tr.tid[mid + 30, 0]


def test_gate_grows_with_the_bridged_frames():
    # the bee is missing for 2 frames (gap 3 -> gate 45 + 2 * 12 = 69 px)
    def run(jump, gap_frames=2):
        pos = np.zeros((6 + gap_frames, 2))
        pos[3 + gap_frames :, 0] = jump
        valid = np.ones(len(pos), dtype=bool)
        valid[3 : 3 + gap_frames] = False
        xy, _ = field(pos)
        return build_tracklets(xy, valid[:, None])

    assert run(65.0).n_tracklets == 1
    assert run(80.0).n_tracklets == 2
    assert run(40.0, gap_frames=0).n_tracklets == 1
    assert run(50.0, gap_frames=0).n_tracklets == 2  # no growth without a gap


def test_gate_is_capped_at_gate_max():
    pos = np.zeros((20, 2))
    pos[14:, 0] = 150.0
    valid = np.ones(20, dtype=bool)
    valid[2:14] = False
    xy, _ = field(pos)
    capped = TrackletParams(gate_max=100.0)
    assert build_tracklets(xy, valid[:, None]).n_tracklets == 1
    assert build_tracklets(xy, valid[:, None], capped).n_tracklets == 2


@pytest.mark.parametrize(("gap", "expected"), [(19, 1), (20, 2)])
def test_a_tracklet_expires_after_gap_max_missing_frames(gap, expected):
    pos = np.zeros((4 + gap, 2))
    valid = np.ones(len(pos), dtype=bool)
    valid[2 : 2 + gap] = False
    xy, _ = field(pos)
    params = TrackletParams(gate_grow=0.0, gap_max=20)
    assert build_tracklets(xy, valid[:, None], params).n_tracklets == expected


def test_velocity_prediction_bridges_a_gap_and_vel_clip_limits_it():
    # 30 px/frame, then 4 missing frames: the velocity prediction makes the reappearance plausible
    n_walk, n_gap = 6, 4
    pos = np.zeros((n_walk + n_gap + 3, 2))
    pos[:, 0] = np.arange(len(pos)) * 30.0
    valid = np.ones(len(pos), dtype=bool)
    valid[n_walk : n_walk + n_gap] = False
    xy, _ = field(pos)
    assert build_tracklets(xy, valid[:, None]).n_tracklets == 1
    # with the speed clipped to 0 the prediction stays at the last position: 150 px > gate
    assert build_tracklets(xy, valid[:, None], TrackletParams(vel_clip=0.0)).n_tracklets == 2


def test_empty_and_single_frame_inputs():
    xy = np.full((5, 2, 2), np.nan)
    tr = build_tracklets(xy, np.zeros((5, 2), dtype=bool))
    assert tr.n_tracklets == 0
    assert (tr.tid == -1).all()
    one, valid = field([[3.0, 4.0]])
    tr1 = build_tracklets(one, valid)
    assert tr1.tid.tolist() == [[0]]
    zero = build_tracklets(np.zeros((0, 3, 2)), np.zeros((0, 3), dtype=bool))
    assert zero.tid.shape == (0, 3)
    assert zero.n_tracklets == 0


def test_errors():
    xy, valid = field(line(5, (0, 0), (1, 0)))
    with pytest.raises(ValueError, match="unknown tracklet mode"):
        build_tracklets(xy, valid, mode="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="xy has shape"):
        build_tracklets(xy[:3], valid)
    with pytest.raises(ValueError, match="hint has shape"):
        build_tracklets(xy, valid, hint=np.zeros((4, 1), dtype=np.int64))
    with pytest.raises(ValueError, match="heading has shape"):
        build_tracklets(xy, valid, heading=np.zeros((5, 2)))


def test_instances_without_a_finite_position_get_no_tracklet():
    xy, valid = field(line(6, (0, 0), (1, 0)))
    xy[2, 0] = [np.nan, 3.0]
    xy[4, 0, 1] = np.inf
    tr = build_tracklets(xy, np.ones_like(valid))  # the caller says "valid" for every slot
    assert tr.tid[:, 0].tolist() == [0, 0, -1, 0, -1, 0]
    assert tr.n_tracklets == 1


def test_mode_sleap_unlabelled_instances_open_a_new_tracklet_every_frame():
    # documented limitation: "sleap" mode needs equal ids, so an untracked input gives singletons
    xy, valid = field(line(8, (0, 0), (1, 0)))
    tr = build_tracklets(xy, valid, hint=np.full((8, 1), -1, dtype=np.int64), mode="sleap")
    assert tr.n_tracklets == 8
    assert (tr.length == 1).all()
    assert build_tracklets(xy, valid, mode="built").n_tracklets == 1


# --------------------------------------------------------------------- SLEAP ids as hints
def test_mode_sleap_follows_the_ids_and_splits_at_a_jump():
    pos = np.zeros((10, 2))
    pos[5:, 0] = 300.0  # a teleport that SLEAP's tracker did not notice
    xy, valid = field(pos)
    hint = np.zeros((10, 1), dtype=np.int64)
    tr = build_tracklets(xy, valid, hint=hint, mode="sleap")
    assert tr.n_tracklets == 2
    assert tr.tid[:5, 0].tolist() == [0] * 5
    assert tr.tid[5:, 0].tolist() == [1] * 5


def test_mode_sleap_splits_at_a_long_gap_and_never_links_unnamed_instances():
    pos = np.zeros((30, 2))
    valid = np.ones(30, dtype=bool)
    valid[3:26] = False
    xy, _ = field(pos)
    hint = np.zeros((30, 1), dtype=np.int64)
    assert build_tracklets(xy, valid[:, None], hint=hint, mode="sleap").n_tracklets == 2
    nameless = np.full((30, 1), -1, dtype=np.int64)
    every_frame = build_tracklets(xy, np.ones((30, 1), dtype=bool), hint=nameless, mode="sleap")
    assert every_frame.n_tracklets == 30


def test_mode_sleap_does_not_link_across_different_ids():
    a, b = line(10, (0, 0), (2, 0)), line(10, (0, 5), (2, 0))
    xy, valid = field(a, b)
    hint = np.tile(np.array([0, 1]), (10, 1))
    hint[5:] = hint[5:, ::-1]  # SLEAP swaps the two ids at frame 5
    tr = build_tracklets(xy, valid, NO_BREAKS, hint=hint, mode="sleap")
    # ids are followed strictly: each id forms a tracklet across the swap (slots exchange)
    assert tr.tid[4, 0] == tr.tid[5, 1]
    assert tr.tid[4, 1] == tr.tid[5, 0]


def test_mode_built_hint_breaks_ties_between_equally_plausible_links():
    # one track at the origin; two candidates next frame, the farther one has the matching id
    xy = np.array([[[0.0, 0.0], [np.nan, np.nan]], [[10.0, 0.0], [14.0, 0.0]]])
    valid = np.isfinite(xy[..., 0])
    hint = np.array([[7, -1], [3, 7]], dtype=np.int64)
    with_hint = build_tracklets(xy, valid, NO_BREAKS, hint=hint)
    assert with_hint.tid[1].tolist() == [1, 0]  # the matching id (slot 1) continues tracklet 0
    plain = build_tracklets(xy, valid, NO_BREAKS)
    assert plain.tid[1].tolist() == [0, 1]  # geometry alone prefers the nearer one


def test_mode_built_hint_never_lifts_the_distance_gate():
    pos = np.zeros((6, 2))
    pos[3:, 0] = 500.0
    xy, valid = field(pos)
    hint = np.full((6, 1), 4, dtype=np.int64)
    assert build_tracklets(xy, valid, hint=hint).n_tracklets == 2


def test_mode_built_hint_penalty_prefers_a_matching_id_over_a_nearer_stranger():
    xy = np.array([[[0.0, 0.0], [np.nan, np.nan]], [[12.0, 0.0], [8.0, 0.0]]])
    valid = np.isfinite(xy[..., 0])
    hint = np.array([[1, -1], [1, 2]], dtype=np.int64)
    tr = build_tracklets(xy, valid, NO_BREAKS, hint=hint)
    assert tr.tid[1].tolist() == [0, 1]  # 12 * 0.5 = 6 beats 8 * 1.5 = 12


# ----------------------------------------------------------------------------- pose terms
def two_candidates(heading_next, head_start=0.0):
    xy = np.array([[[0.0, 0.0], [np.nan, np.nan]], [[10.0, 0.0], [12.0, 0.0]]])
    valid = np.isfinite(xy[..., 0])
    heading = np.array([[head_start, np.nan], heading_next], dtype=float)
    return xy, valid, heading


def test_pose_weight_penalises_a_heading_flip():
    xy, valid, heading = two_candidates([np.pi, 0.0])
    plain = build_tracklets(xy, valid, NO_BREAKS)
    assert plain.tid[1].tolist() == [0, 1]
    with_pose = build_tracklets(
        xy,
        valid,
        TrackletParams(amb_ratio=0.0, amb_add=0.0, pose_weight=20.0),
        heading=heading,
    )
    assert with_pose.tid[1].tolist() == [1, 0]


def test_pose_max_turn_forbids_a_big_turn():
    xy, valid, heading = two_candidates([np.pi, 0.0])
    params = TrackletParams(amb_ratio=0.0, amb_add=0.0, pose_max_turn=1.0)
    assert build_tracklets(xy, valid, params, heading=heading).tid[1].tolist() == [1, 0]


def test_unknown_headings_neither_cost_nor_forbid():
    xy, valid, heading = two_candidates([np.nan, np.nan])
    params = TrackletParams(amb_ratio=0.0, amb_add=0.0, pose_weight=50.0, pose_max_turn=0.1)
    assert build_tracklets(xy, valid, params, heading=heading).tid[1].tolist() == [0, 1]
    # a track that never had a heading
    xy2, valid2, heading2 = two_candidates([0.0, 0.0], head_start=np.nan)
    assert build_tracklets(xy2, valid2, params, heading=heading2).tid[1].tolist() == [0, 1]


def test_heading_is_remembered_across_missing_headings():
    # frame 1 has no heading; frame 2 compares against the heading seen in frame 0
    xy = np.array([[[0.0, 0.0], [np.nan, np.nan]]] * 2 + [[[10.0, 0.0], [12.0, 0.0]]])
    valid = np.isfinite(xy[..., 0])
    heading = np.array([[0.0, np.nan], [np.nan, np.nan], [np.pi, 0.0]])
    params = TrackletParams(amb_ratio=0.0, amb_add=0.0, pose_weight=20.0)
    tr = build_tracklets(xy, valid, params, heading=heading)
    assert tr.tid[2].tolist() == [1, 0]


def test_heading_change_wraps_around():
    xy = np.array([[[0.0, 0.0]], [[1.0, 0.0]]])
    valid = np.ones((2, 1), dtype=bool)
    heading = np.array([[3.0], [-3.0]])  # 0.28 rad apart across +-pi, not 6 rad
    params = TrackletParams(pose_max_turn=0.5)
    assert build_tracklets(xy, valid, params, heading=heading).n_tracklets == 1


# ------------------------------------------------------------------------ property tests
def random_scene_input(seed):
    rng = np.random.default_rng(seed)
    n_bees = int(rng.integers(2, 6))
    n_frames = int(rng.integers(20, 90))
    passes = ((0, 1, n_frames // 2),) if rng.random() < 0.4 else ()
    scene = make_scene(
        seed,
        n_frames=n_frames,
        n_bees=n_bees,
        ghost_rate=float(rng.choice([0.0, 0.2])),
        miss_rate=float(rng.choice([0.0, 0.1, 0.3])),
        near_passes=passes,
        box=float(rng.choice([300.0, 1800.0])),
    )
    return scene.table


@pytest.mark.parametrize("seed", range(220))
def test_invariants_on_random_scenes(seed):
    table = random_scene_input(seed)
    keep = dedupe_instances(table.xy, table.score, 25.0)
    params = TrackletParams()
    tr = build_tracklets(table.xy, keep, params)
    again = build_tracklets(table.xy, keep, params)
    assert np.array_equal(tr.tid, again.tid), "not deterministic"
    # every kept instance is in exactly one tracklet, every other slot in none
    assert ((tr.tid >= 0) == keep).all()
    # ids are contiguous 0..n-1 and all used
    used = np.unique(tr.tid[tr.tid >= 0])
    assert used.tolist() == list(range(tr.n_tracklets))
    for tid in used:
        rows, slots = np.nonzero(tr.tid == tid)
        assert len(np.unique(rows)) == len(rows), "two instances of a tracklet in one frame"
        order = np.argsort(rows)
        rows, slots = rows[order], slots[order]
        assert (np.diff(rows) <= params.gap_max + 1).all(), "a tracklet crossed a closed gap"
        steps = np.linalg.norm(np.diff(table.xy[rows, slots], axis=0), axis=-1)
        limit = params.gate_max + params.vel_clip * (np.diff(rows))
        assert (steps <= limit + 1e-9).all(), "a link beyond the gate"
    assert tr.length.sum() == keep.sum()


@pytest.mark.parametrize("seed", range(10))
def test_hint_modes_keep_the_invariants(seed):
    table = random_scene_input(seed)
    keep = dedupe_instances(table.xy, table.score, 25.0)
    rng = np.random.default_rng(seed)
    hint = rng.integers(-1, 4, size=keep.shape).astype(np.int64)
    for mode in ("built", "sleap"):
        tr = build_tracklets(table.xy, keep, hint=hint, mode=mode)  # type: ignore[arg-type]
        assert ((tr.tid >= 0) == keep).all()
        for tid in range(tr.n_tracklets):
            rows, _ = np.nonzero(tr.tid == tid)
            assert len(np.unique(rows)) == len(rows)
