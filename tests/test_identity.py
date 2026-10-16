"""Unit tests of the joint identity engine (naps.identity)."""

import itertools
import logging
import math

import numpy as np
import pytest

from naps import identity
from naps.consistency import ConsistencyParams
from naps.identity import (
    MAX_TAGS,
    IdentityParams,
    RegionLabels,
    SlotFlag,
    TransitionModel,
    Votes,
    build_emissions,
    build_entities,
    candidate_anchors,
    cull_excess,
    demote_strays,
    forward_backward,
    permutation_table,
    single_decodes,
    solve_region,
    votes_from_frame_votes,
)
from naps.reasons import Reason
from naps.sources import Source
from naps.tracklets import Tracklets
from tests.scenes import (
    anchors_of,
    assert_invariants,
    bee_slot,
    evaluate,
    info_of,
    make_scene,
    valid_of,
    vote_rows,
    with_votes,
)

P = IdentityParams()


def tracklets_from(tid):
    return Tracklets(np.asarray(tid, dtype=np.int32))


def entities_from(tid, xy=None, n_bees=3, with_virtual=True):
    tr = tracklets_from(tid)
    if xy is None:
        xy = np.zeros((*tr.tid.shape, 2))
    return build_entities(tr, np.asarray(xy, dtype=float), n_bees, with_virtual)


def votes_of(*rows):
    arr = np.array(rows, dtype=float).reshape(-1, 4)
    return Votes(
        arr[:, 0].astype(np.int64),
        arr[:, 1].astype(np.int64),
        arr[:, 2].astype(np.int64),
        arr[:, 3],
    )


# ------------------------------------------------------------------------------- votes
def test_empty_votes_has_no_entries():
    v = Votes.empty()
    assert len(v) == 0
    assert v.frame.dtype == np.int64 and v.weight.dtype == np.float64


def test_votes_from_frame_votes_filters_and_sorts():
    frame_votes = {
        12: {1: [(7, 1.0), (99, 1.0)], 0: [(5, 0.5)]},  # tag 99 is not in the set
        10: {2: [(5, 2.0)]},
        50: {0: [(5, 1.0)]},  # outside the rows
        9: {0: [(5, 1.0)]},  # before the first frame
    }
    v = votes_from_frame_votes(frame_votes, 10, 5, {5: 0, 7: 1})
    assert v.frame.tolist() == [0, 2, 2]
    assert v.slot.tolist() == [2, 0, 1]
    assert v.tag.tolist() == [0, 0, 1]
    assert v.weight.tolist() == [2.0, 0.5, 1.0]


def test_votes_from_frame_votes_empty_inputs():
    assert len(votes_from_frame_votes({}, 0, 10, {1: 0})) == 0
    assert len(votes_from_frame_votes({3: {0: [(8, 1.0)]}}, 0, 10, {1: 0})) == 0


# ---------------------------------------------------------------------------- anchors
def test_candidate_anchors_rules():
    votes = votes_of(
        (0, 0, 1, 1.0),  # a clean decode
        (1, 0, 0, 1.0), (1, 0, 1, 1.0),  # two tags on one instance: ambiguous
        (2, 0, 2, 1.0), (2, 1, 2, 1.0),  # the same tag on two instances: conflict
        (3, 0, 0, 0.4),  # too weak
        (4, 0, 2, 0.3), (4, 0, 2, 0.3),  # weights add up
        (5, 0, 1, 1.0), (5, 1, 0, 1.0),  # two instances, different tags: both fine
    )  # fmt: skip
    a = candidate_anchors(votes, (6, 2), 3)
    assert a[0].tolist() == [1, -1]
    assert a[1].tolist() == [-1, -1]
    assert a[2].tolist() == [-1, -1]
    assert a[3].tolist() == [-1, -1]
    assert a[4].tolist() == [2, -1]
    assert a[5].tolist() == [1, 0]
    assert a.dtype == np.int16
    weak = candidate_anchors(votes_of((0, 0, 1, 0.4)), (1, 1), 3, min_weight=0.3)
    assert weak[0, 0] == 1
    assert (candidate_anchors(Votes.empty(), (2, 2), 3) == -1).all()


def test_demote_strays_minority_of_a_strong_majority():
    tid = np.zeros((200, 1), dtype=np.int32)
    anchor = np.full((200, 1), -1, dtype=np.int16)
    anchor[:100, 0] = 0
    anchor[100:102, 0] = 1  # 2 of 100: a stray
    demote = demote_strays(anchor, tracklets_from(tid), 3, P)
    assert demote[100:102].all() and not demote[:100].any()
    anchor[102:106, 0] = 1  # now 6 of 100: more than 3 %, a real second tag
    assert not demote_strays(anchor, tracklets_from(tid), 3, P).any()


def test_demote_strays_leaves_balanced_and_small_tracklets_alone():
    tid = np.zeros((20, 1), dtype=np.int32)
    anchor = np.full((20, 1), -1, dtype=np.int16)
    anchor[:10, 0] = 0
    anchor[10:20, 0] = 1  # two well-supported tags
    assert not demote_strays(anchor, tracklets_from(tid), 3, P).any()
    small = np.full((20, 1), -1, dtype=np.int16)
    small[:2, 0] = 0
    small[5, 0] = 1  # majority 2 < min_majority
    loose = IdentityParams(minority_fraction=0.9)
    assert not demote_strays(small, tracklets_from(tid), 3, loose).any()


def test_demote_strays_without_anchors_or_tracklets():
    tid = -np.ones((5, 2), dtype=np.int32)
    anchor = np.full((5, 2), 1, dtype=np.int16)  # anchors on instances that are in no tracklet
    out = demote_strays(anchor, Tracklets(tid), 3, P)
    assert out.shape == (5, 2) and not out.any()
    empty = demote_strays(
        np.full((5, 2), -1, dtype=np.int16), tracklets_from(np.zeros((5, 2))), 3, P
    )
    assert not empty.any()


# -------------------------------------------------------------------------- entities
def test_build_entities_sorted_by_tracklet_with_slot_map_and_nn():
    tid = [[2, 0, -1], [1, -1, 0]]
    xy = np.zeros((2, 3, 2))
    xy[0, 0], xy[0, 1] = (100, 0), (0, 0)
    xy[1, 0], xy[1, 2] = (50, 50), (0, 0)
    e = entities_from(tid, xy, n_bees=3)
    assert e.count.tolist() == [2, 2]
    assert e.tid[0, :2].tolist() == [0, 2] and e.tid[1, :2].tolist() == [0, 1]
    assert e.slot[0, :2].tolist() == [1, 0]
    assert e.of_slot[0].tolist() == [1, 0, -1]
    assert e.of_slot[1].tolist() == [1, -1, 0]
    assert np.allclose(e.pos[0, 0], (0, 0))
    assert e.nn[0, 0] == pytest.approx(100.0) and e.nn[1, 0] == pytest.approx(math.hypot(50, 50))
    assert e.tid[0, 2] == -1 and np.isnan(e.pos[0, 2]).all()


def test_build_entities_single_entity_has_far_neighbour():
    e = entities_from([[0, -1], [0, -1]], n_bees=2)
    assert (e.nn[:, 0] == identity._BIG_DISTANCE).all()


def test_build_entities_bridges_gaps_with_virtual_entities():
    xy = np.zeros((5, 1, 2))
    xy[0, 0], xy[4, 0] = (0, 0), (40, 0)
    e = entities_from([[0], [-1], [-1], [-1], [0]], xy, n_bees=2)
    assert e.count.tolist() == [1, 1, 1, 1, 1]
    assert e.slot[2, 0] == -1  # virtual
    assert np.allclose(e.pos[2, 0], (20, 0)) and np.allclose(e.pos[1, 0], (10, 0))
    assert e.of_slot[2, 0] == -1 and e.of_slot[0, 0] == 0
    none = entities_from([[0], [-1], [-1], [-1], [0]], xy, n_bees=2, with_virtual=False)
    assert none.count.tolist() == [1, 0, 0, 0, 1]


def test_build_entities_capacity_drops_virtual_entities():
    # tracklet 0 has a gap at frame 1 while two real instances fill the frame (capacity 2)
    tid = [[0, 1, -1], [-1, 1, 2], [0, 1, -1]]
    xy = np.zeros((3, 3, 2))
    xy[:, 1] = (500, 0)
    xy[1, 2] = (0, 500)
    e = entities_from(tid, xy, n_bees=2)
    assert e.n_dropped_virtual == 1
    assert e.count.tolist() == [2, 2, 2]
    assert e.tid[1, :2].tolist() == [1, 2]


# ---------------------------------------------------------------------------- culling
def test_cull_excess_keeps_the_best_ranked_instances():
    valid = np.array([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=bool)
    rank = np.array([[0.5, 0.9, 0.9, 1000.0], [0.1, 0.2, 0, 0]])
    keep = cull_excess(valid, rank, 2)
    assert keep[0].tolist() == [False, True, False, True]  # ties: lower slot first
    assert keep[1].tolist() == [True, True, False, False]
    assert valid[0].all()  # the input is not modified


# ----------------------------------------------------------------------- state spaces
@pytest.mark.parametrize(("n_tags", "n"), [(3, 0), (3, 1), (3, 2), (3, 3), (4, 3), (6, 6), (5, 2)])
def test_permutation_table_sizes_and_injectivity(n_tags, n):
    table = permutation_table(n_tags, n)
    assert table.shape == (math.perm(n_tags, n), n)
    assert not table.flags.writeable
    assert all(len(set(row)) == n for row in table.tolist())
    assert permutation_table(n_tags, n) is table  # cached


def test_one_hot_and_differs_tables():
    onehot = identity._one_hot(4, 2)
    table = permutation_table(4, 2)
    assert onehot.shape == (12, 2, 4)
    assert (onehot.argmax(axis=2) == table).all() and (onehot.sum(axis=2) == 1).all()
    differs = identity._differs(3, 2, 3, 0, 1)
    a, b = permutation_table(3, 2), permutation_table(3, 3)
    assert differs.shape == (len(a), len(b))
    assert differs[1, 4] == (a[1, 0] != b[4, 1])
    assert not differs.flags.writeable


# ---------------------------------------------------------------------- transitions
def test_log_eps_endpoints_and_interpolation():
    model = TransitionModel(entities_from([[0]], n_bees=3), 3, P)
    assert model.log_eps(0.0) == pytest.approx(math.log(P.eps_near))
    assert model.log_eps(P.nn_near_px) == pytest.approx(math.log(P.eps_near))
    assert model.log_eps(P.nn_far_px) == pytest.approx(math.log(P.eps_far))
    assert model.log_eps(1e4) == pytest.approx(math.log(P.eps_far))
    mid = 0.5 * (P.nn_near_px + P.nn_far_px)
    assert model.log_eps(mid) == pytest.approx(0.5 * (math.log(P.eps_near) + math.log(P.eps_far)))


def test_matrix_penalises_label_changes_of_continuing_entities():
    xy = np.zeros((2, 2, 2))
    xy[:, 1] = (500, 0)  # far apart: eps_far
    e = entities_from([[0, 1], [0, 1]], xy, n_bees=3)
    trans = TransitionModel(e, 3, P).matrix(0)
    table = permutation_table(3, 2)
    assert trans.shape == (6, 6)
    for a, b in itertools.product(range(6), repeat=2):
        n_diff = int((table[a] != table[b]).sum())
        assert trans[a, b] == pytest.approx(math.exp(0.5 * math.log(P.eps_far) * n_diff))
    assert (np.diag(trans) == 1).all()


def test_matrix_uses_the_nearer_neighbour_distance():
    xy = np.zeros((2, 2, 2))
    xy[0, 1] = (500, 0)
    xy[1, 1] = (10, 0)  # close in the second frame
    e = entities_from([[0, 1], [0, 1]], xy, n_bees=3)
    trans = TransitionModel(e, 3, P).matrix(0)
    swap = np.flatnonzero((permutation_table(3, 2) == [1, 0]).all(axis=1))[0]
    assert trans[0, swap] == pytest.approx(math.exp(math.log(P.eps_near)))


@pytest.mark.parametrize(
    ("dist", "bonus"), [(0.0, 20.0), (40.0, 20 * math.exp(-0.5)), (400.0, 0.0)]
)
def test_matrix_hand_off_bonus_for_a_new_entity_next_to_an_ended_one(dist, bonus):
    xy = np.zeros((2, 1, 2))
    xy[1, 0] = (dist, 0)
    e = entities_from([[0], [1]], xy, n_bees=3)  # tracklet 0 ends, tracklet 1 starts
    trans = TransitionModel(e, 3, P).matrix(0)
    expected = np.ones((3, 3)) + bonus * np.eye(3)
    assert np.allclose(trans, expected)


def test_matrix_mixes_continuing_ended_and_new_entities():
    xy = np.zeros((2, 2, 2))
    xy[0, 0], xy[0, 1] = (0, 0), (300, 0)
    xy[1, 0], xy[1, 1] = (300, 0), (5, 0)
    e = entities_from([[0, 1], [1, 2]], xy, n_bees=3)  # 0 ended, 1 continues, 2 is new (near 0)
    trans = TransitionModel(e, 3, P).matrix(0)
    a, b = permutation_table(3, 2), permutation_table(3, 2)
    for s, t in itertools.product(range(6), repeat=2):
        # entity 1 continues (index 1 -> 0); entity 0 ended; the new entity has index 1
        keep = a[s, 1] != b[t, 0]
        expect = math.exp(0.5 * math.log(P.eps_far) * keep)
        expect *= 1 + 20 * math.exp(-0.5 * (5 / 40) ** 2) * (a[s, 0] == b[t, 1])
        assert trans[s, t] == pytest.approx(expect)


# ---------------------------------------------------------------------------- emissions
def test_build_emissions_weights_demotion_and_extra():
    xy = np.zeros((2, 2, 2))
    xy[:, 1] = (500, 0)
    e = entities_from([[0, 1], [0, 1]], xy, n_bees=3)
    votes = votes_of((0, 0, 2, 1.0), (0, 1, 0, 2.0), (1, 0, 1, 1.0), (1, 1, 5, 1.0))
    demoted = np.zeros((2, 2), dtype=bool)
    demoted[0, 1] = True
    extra = votes_of((1, 0, 1, 0.5), (1, 0, 2, 1.0))
    ev = build_emissions(votes, demoted, e, 3, P, extra=extra)
    assert ev.shape == (2, 3, 3)
    assert ev[0, 0, 2] == pytest.approx(2.0)
    assert ev[0, 1, 0] == pytest.approx(2.0 * 2.0 * P.demoted_weight)
    assert ev[1, 0, 1] == pytest.approx(2.0 + 1.0)  # decode + undemoted extra vote
    assert ev[1, 0, 2] == pytest.approx(2.0)
    assert ev[1, 1].sum() == 0  # tag 5 is outside the set
    plain = build_emissions(Votes.empty(), demoted, e, 3, P, extra=Votes.empty())
    assert (plain == 0).all()


def test_build_emissions_ignores_votes_on_instances_without_an_entity():
    e = entities_from([[0, -1]], n_bees=3)
    ev = build_emissions(votes_of((0, 1, 0, 1.0)), np.zeros((1, 2), dtype=bool), e, 3, P)
    assert (ev == 0).all()


def test_build_emissions_appearance_gate_floor_and_nan_rows():
    # entity pairs 30 px, 80 px and 500 px apart in three frames
    xy = np.zeros((4, 2, 2))
    xy[0, 1], xy[1, 1], xy[2, 1], xy[3, 1] = (30, 0), (80, 0), (500, 0), (500, 0)
    e = entities_from([[0, 1]] * 4, xy, n_bees=3)
    app = np.zeros((4, 2, 3))
    app[:, :, :] = [-0.5, -10.0, -1.0]  # tag 1 is clipped to the floor
    app[3, 0] = np.nan  # an instance without appearance scores
    params = IdentityParams(appearance_weight=1.0)
    ev = build_emissions(Votes.empty(), np.ones((4, 2), dtype=bool), e, 3, params, appearance=app)
    clipped = np.array([-0.5, -4.0, -1.0])
    assert np.allclose(ev[0, 0], 0.3 * clipped) and np.allclose(ev[0, 1], 0.3 * clipped)
    assert np.allclose(ev[1, 0], 0.6 * clipped)
    assert np.allclose(ev[2, 0], clipped)
    assert (ev[3, 0] == 0).all() and np.allclose(ev[3, 1], clipped)


# ----------------------------------------------------------------------- forward-backward
def brute_force_marginals(evidence, entities, k, params):
    """Marginals by enumerating every state sequence (exponential; tiny inputs only)."""
    model = TransitionModel(entities, k, params)
    n_frames = evidence.shape[0]
    tables = [permutation_table(k, int(n)) for n in entities.count]
    trans = [model.matrix(f) for f in range(n_frames - 1)]
    out = np.zeros((n_frames, k, k))
    total = 0.0
    for seq in itertools.product(*[range(len(t)) for t in tables]):
        w = 1.0
        for f, s in enumerate(seq):
            w *= math.exp(
                sum(evidence[f, i, tables[f][s][i]] for i in range(int(entities.count[f])))
            )
            if f:
                w *= trans[f - 1][seq[f - 1], s]
        total += w
        for f, s in enumerate(seq):
            for i in range(int(entities.count[f])):
                out[f, i, tables[f][s][i]] += w
    return out / total


LAYOUTS = {
    "appearing_disappearing": [[0, 1, -1], [0, 1, 2], [-1, 1, 2], [-1, -1, -1], [3, 4, -1]],
    "gap_bridged": [[0, 1, -1], [-1, 1, -1], [0, 1, -1], [0, -1, -1], [0, 2, -1]],
    "steady": [[0, 1, 2]] * 5,
}


@pytest.mark.parametrize("name", sorted(LAYOUTS))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_forward_backward_matches_brute_force(name, seed):
    rng = np.random.default_rng(seed)
    tid = np.array(LAYOUTS[name])
    xy = rng.uniform(0, 90, size=(*tid.shape, 2))
    e = entities_from(tid, xy, n_bees=3)
    ev = rng.normal(0, 1.5, size=(len(tid), 3, 3))
    got = forward_backward(ev, e, 3, P)
    want = brute_force_marginals(ev, e, 3, P)
    assert got.dtype == np.float32
    assert np.allclose(got, want, atol=1e-5)
    for f in range(len(tid)):  # rows beyond the entities are zero, present rows sum to one
        n = int(e.count[f])
        assert (got[f, n:] == 0).all()
        assert np.allclose(got[f, :n].sum(), n, atol=1e-5)  # every entity has exactly one tag


def test_forward_backward_rebuilds_transitions_when_they_do_not_fit(monkeypatch):
    rng = np.random.default_rng(5)
    tid = np.array(LAYOUTS["appearing_disappearing"])
    e = entities_from(tid, rng.uniform(0, 90, size=(*tid.shape, 2)), n_bees=3)
    ev = rng.normal(0, 1.5, size=(len(tid), 3, 3))
    cached = forward_backward(ev, e, 3, P)
    monkeypatch.setattr(identity, "_MAX_CACHE_BYTES", 0)
    assert np.array_equal(forward_backward(ev, e, 3, P), cached)


def test_forward_backward_with_a_prebuilt_transition_model_and_one_frame():
    e = entities_from([[0, 1, -1]], n_bees=3)
    ev = np.zeros((1, 3, 3))
    ev[0, 0, 2] = 5.0
    model = TransitionModel(e, 3, P)
    got = forward_backward(ev, e, 3, P, transitions=model)
    want = brute_force_marginals(ev, e, 3, P)
    assert np.allclose(got, want, atol=1e-6)


def test_forward_backward_survives_extreme_evidence():
    e = entities_from([[0, 1, 2]] * 3, n_bees=3)
    ev = np.zeros((3, 3, 3))
    ev[:, 0, 0] = ev[:, 1, 1] = ev[:, 2, 2] = 2000.0
    got = forward_backward(ev, e, 3, P)
    assert np.isfinite(got).all()
    assert np.allclose(got[0, np.arange(3), np.arange(3)], 1.0)


# ------------------------------------------------------------------------- solve_region
@pytest.mark.parametrize(("n_real", "n_bees"), [(0, 4), (5, 4), (2, 7), (7, 7), (-1, 3)])
def test_solve_region_rejects_bad_sizes(n_real, n_bees):
    scene = make_scene(1, n_frames=10)
    with pytest.raises(ValueError, match="n_real"):
        solve_region(scene.table, scene.votes, n_real, n_bees)
    assert MAX_TAGS == 6


def test_head_and_tail_frames_are_labelled():
    # decodes only in the middle of the video: the first and last 100 frames must still be labelled
    scene = make_scene(11, n_frames=400, absent=dict.fromkeys(range(4), (0, 99)))
    rows = [r for r in vote_rows(scene) if r[0] < 300]
    scene = with_votes(scene, rows)
    labels = solve_region(scene.table, scene.votes, 4, 4)
    assert (labels.slot_tag[0] >= 0).all() and (labels.slot_tag[-1] >= 0).all()
    ev = evaluate(scene, labels)
    assert ev["precision"] == 1.0 and ev["coverage"] > 0.99
    assert_invariants(labels, 4)


def test_phantom_tag_for_an_untagged_bee_is_never_emitted():
    scene = make_scene(12, n_frames=300, n_bees=4, n_real=3)
    labels = solve_region(scene.table, scene.votes, 3, 4)
    ev = evaluate(scene, labels)
    assert ev["untagged_labelled"] == 0 and ev["wrong"] == 0 and ev["coverage"] > 0.99
    untagged = bee_slot(scene, 3)
    flags = labels.slot_flag[np.arange(300), untagged]
    assert (flags == SlotFlag.UNTAGGED_BEE).sum() > 250
    assert labels.tag_xy.shape[1] == 3
    assert_invariants(labels, 3)


def test_all_bees_tagged_without_phantoms_uses_elimination():
    scene = make_scene(13, n_frames=300, absent={3: (0, 299)})
    labels = solve_region(scene.table, scene.votes, 4, 4)
    ev = evaluate(scene, labels)
    assert ev["rec_precision"] == 1.0 and ev["rec_coverage"] > 0.99
    recovered = labels.slot_source[labels.slot_tag == 3]
    assert recovered.size > 250 and set(recovered.tolist()) == {int(Source.ELIMINATION)}


def test_extra_votes_are_soft_evidence_and_never_anchors():
    scene = make_scene(14, n_frames=200, decode_rate=0.0)
    truth_at = zip(*np.nonzero(scene.truth >= 0), strict=True)
    rows = [(f, int(s), int(scene.truth[f, s]), 1.0) for f, s in truth_at]
    extra = with_votes(scene, rows).votes
    labels = solve_region(scene.table, scene.votes, 4, 4, extra=extra)
    assert labels.candidates is not None and (labels.candidates == -1).all()
    assert (anchors_of(labels) == -1).all()
    assert (labels.slot_source != Source.DECODE).all()
    ev = evaluate(scene, labels)
    assert ev["wrong"] == 0 and ev["coverage"] > 0.99
    assert_invariants(labels, 4)


def test_without_any_evidence_nothing_is_labelled():
    scene = make_scene(15, n_frames=120, decode_rate=0.0)
    labels = solve_region(scene.table, scene.votes, 4, 4)
    assert (labels.slot_tag == -1).all()
    assert (labels.slot_flag[labels.valid] == SlotFlag.LOW_CONFIDENCE).all()
    assert (labels.guess_conf[labels.valid] < 0.99).all()
    assert labels.info["n_candidate_anchors"] == 0


def test_appearance_evidence_tips_an_otherwise_symmetric_region():
    scene = make_scene(16, n_frames=200, n_bees=2, decode_rate=0.0)
    app = np.full((200, 2, 2), -4.0)
    frames, slots = np.nonzero(scene.truth >= 0)
    app[frames, slots, scene.truth[frames, slots]] = -0.05
    params = IdentityParams(appearance_weight=1.0)
    plain = solve_region(scene.table, scene.votes, 2, 2, params)
    assert (plain.slot_tag == -1).all()
    labels = solve_region(scene.table, scene.votes, 2, 2, params, appearance=app)
    ev = evaluate(scene, labels)
    assert ev["wrong"] == 0 and ev["coverage"] > 0.99
    assert set(labels.slot_source[labels.slot_tag >= 0].tolist()) == {int(Source.APPEARANCE)}
    assert_invariants(labels, 2)


def test_a_decode_that_contradicts_the_closed_set_is_flagged_and_kept():
    scene = make_scene(17, n_frames=160, decode_rate=0.8)
    slot_b, slot_a = bee_slot(scene, 2), bee_slot(scene, 0)
    # bee 2 is decoded only twice; one frame carries a lone decode of bee 0's tag on bee 2
    stray = next(f for f in range(20, 160) if not scene.decode_on[f, 0] and slot_b[f] >= 0)
    rows = [r for r in vote_rows(scene) if r[2] != 2]
    rows += [(5, int(slot_b[5]), 2, 1.0), (6, int(slot_b[6]), 2, 1.0)]
    rows += [(stray, int(slot_b[stray]), 0, 1.0)]
    scene = with_votes(scene, rows)
    labels = solve_region(scene.table, scene.votes, 4, 4)
    sb = int(slot_b[stray])
    assert labels.slot_tag[stray, sb] == 0 and labels.slot_source[stray, sb] == Source.DECODE
    assert labels.slot_disputed[stray, sb]
    sa = int(slot_a[stray])
    assert labels.slot_tag[stray, sa] == -1
    assert labels.slot_flag[stray, sa] == SlotFlag.BLOCKED_BY_ANCHOR
    assert labels.slot_disputed.sum() == 1
    assert_invariants(labels, 4)


def test_stray_decodes_on_a_long_minority_are_demoted():
    scene = make_scene(18, n_frames=300, decode_rate=0.9)
    slot_b = bee_slot(scene, 1)
    stray_frames = (100, 150, 200)
    # in these frames bee 1 has no decode of its own and bee 3 none either: only the stray remains
    rows = [r for r in vote_rows(scene) if not (r[0] in stray_frames and r[2] in (1, 3))]
    rows += [(f, int(slot_b[f]), 3, 1.0) for f in stray_frames]
    scene = with_votes(scene, rows)
    labels = solve_region(scene.table, scene.votes, 4, 4)
    assert info_of(labels)["n_stray_anchors_demoted"] >= 1
    ev = evaluate(scene, labels)
    assert ev["wrong"] == 0
    assert_invariants(labels, 4)


@pytest.mark.parametrize("n_bees", [5, 6])
def test_five_and_six_bees_run_with_lazy_transitions(n_bees):
    scene = make_scene(20 + n_bees, n_frames=40, n_bees=n_bees, absent={n_bees - 1: (0, 39)})
    labels = solve_region(scene.table, scene.votes, n_bees, n_bees)
    ev = evaluate(scene, labels)
    assert ev["wrong"] == 0 and ev["coverage"] > 0.9
    assert_invariants(labels, n_bees)


def test_six_bees_with_untagged_phantoms():
    scene = make_scene(27, n_frames=30, n_bees=6, n_real=4)
    labels = solve_region(scene.table, scene.votes, 4, 6)
    assert evaluate(scene, labels)["wrong"] == 0
    assert labels.marginals is not None and labels.marginals.shape == (30, 6, 6)


def test_region_labels_are_consistent_and_documented():
    scene = make_scene(30, n_frames=250, absent={2: (30, 220)}, near_passes=((0, 1, 120),))
    labels = solve_region(scene.table, scene.votes, 4, 4)
    assert_invariants(labels, 4)
    info = labels.info
    assert info["n_instances"] == int(scene.table.valid.sum())
    assert {"consistency", "n_tracklets", "n_ghost_instances"} <= set(info)
    assert labels.tracklets is not None and labels.entities is not None
    assert labels.guess_tag.shape == labels.slot_tag.shape
    unlabelled = (labels.slot_tag < 0) & valid_of(labels)
    assert (labels.slot_flag[unlabelled] > 0).all()
    assert len(SlotFlag.NAMES) == SlotFlag.WITHHELD + 1


# ------------------------------------------------------------------- consistency stage
def test_a_teleporting_recovered_tag_is_dropped_by_the_chain_veto():
    scene = make_scene(31, n_frames=300, absent={3: (0, 299)}, box=1200.0)
    slot = bee_slot(scene, 3)
    late = np.arange(300) >= 150
    for arr in (scene.table.xy, scene.table.tag_xy):
        arr[late, slot[late]] += (-500.0 if scene.table.xy[200, slot[200], 0] > 600 else 500.0, 0)
    labels = solve_region(scene.table, scene.votes, 4, 4)
    dropped = info_of(labels, "consistency")["dropped_total"]
    assert dropped > 100
    assert (labels.slot_flag == SlotFlag.CONSISTENCY_DROPPED).sum() == dropped
    assert evaluate(scene, labels)["wrong"] == 0
    xy3 = labels.tag_xy[:, 3]
    steps = np.linalg.norm(np.diff(xy3, axis=0), axis=-1)
    assert np.nanmax(steps) < 100
    assert_invariants(labels, 4)


def test_consistency_switches():
    scene = make_scene(32, n_frames=150, absent={2: (0, 149)}, miss_rate=0.1)
    off = solve_region(scene.table, scene.votes, 4, 4, None, ConsistencyParams(enabled=False))
    no_ghost = solve_region(scene.table, scene.votes, 4, 4, None, ConsistencyParams(ghost=False))
    no_gap = solve_region(scene.table, scene.votes, 4, 4, None, ConsistencyParams(gap_max=0))
    assert info_of(off, "consistency")["dropped_total"] == 0
    assert no_ghost.info["n_ghost_instances"] == 0
    assert info_of(no_gap, "consistency")["interpolation_refused_frames"] == 0
    # bridged tracklet gaps (virtual entities) are still labelled; no extra gap filling happens
    assert info_of(no_gap, "consistency")["interpolated_frames"] == int(
        (no_gap.tag_src == Source.INTERPOLATED).sum()
    )
    for labels in (off, no_ghost, no_gap):
        assert evaluate(scene, labels)["wrong"] == 0


def test_bridged_gaps_are_interpolated_or_left_unlabelled():
    scene = make_scene(33, n_frames=200, absent={1: (0, 199)}, miss_rate=0.15)
    labels = solve_region(scene.table, scene.votes, 4, 4)
    assert info_of(labels, "consistency")["interpolated_frames"] > 0
    filled = labels.tag_src == Source.INTERPOLATED
    assert (labels.tag_slot[filled] == -1).all() and np.isfinite(labels.tag_xy[filled]).all()
    assert np.isfinite(labels.tag_conf[filled]).all()
    assert_invariants(labels, 4)


def test_virtual_entities_without_evidence_stay_unlabelled():
    scene = make_scene(34, n_frames=120, decode_rate=0.0, miss_rate=0.1)
    labels = solve_region(scene.table, scene.votes, 4, 4)
    assert labels.entities is not None and (labels.entities.slot == -1).any()
    assert not np.isfinite(labels.tag_xy).any()
    phantom = make_scene(35, n_frames=150, n_real=2, miss_rate=0.1)  # two untagged bees' gaps
    labels = solve_region(phantom.table, phantom.votes, 2, 4)
    assert evaluate(phantom, labels)["wrong"] == 0
    assert_invariants(labels, 2)


def test_a_bridged_gap_on_top_of_another_bee_is_dropped_as_a_duplicate():
    # bee 3 (never decoded) is missed for 11 frames while it meets bee 0: the bridged position
    # would sit on bee 0's decoded position, so the duplicate rule removes that label
    scene = make_scene(
        12, n_frames=220, absent={3: (0, 219)}, near_passes=((0, 3, 105, 35),), min_separation=45.0
    )
    slot = bee_slot(scene, 3)
    for f in range(100, 111):
        scene.table.xy[f, slot[f]] = np.nan
        scene.table.tag_xy[f, slot[f]] = np.nan
    labels = solve_region(scene.table, scene.votes, 4, 4)
    assert info_of(labels, "consistency")["dropped_duplicate"] > 0
    assert evaluate(scene, labels)["wrong_inferred"] == 0
    assert_invariants(labels, 4)


# --------------------------------------------------- audit fixes: demotions, gating, counters
def test_a_conflicting_decode_is_demoted_counted_and_the_instance_relabelled():
    scene = make_scene(41, n_frames=240, decode_rate=1.0, min_separation=100.0)
    slot0 = bee_slot(scene, 0)
    rows = [
        r for r in vote_rows(scene) if not (100 <= r[0] < 111 and r[1] == slot0[r[0]] and r[2] == 0)
    ]
    rows += [(f, int(slot0[f]), 1, 1.0) for f in range(100, 111)]  # bee 0 reads as tag 1 ...
    noisy = with_votes(scene, rows)  # ... where bee 1 really carries (and is decoded as) tag 1
    labels = solve_region(noisy.table, noisy.votes, 4, 4)
    info = info_of(labels)
    # both instances lose their anchor: the same tag is decoded on two instances of one frame
    assert info["n_conflict_decodes_demoted"] == 22
    assert labels.demoted is not None and int(labels.demoted.sum()) == 22
    assert labels.demoted[np.arange(100, 111), slot0[100:111]].all()
    assert (anchors_of(labels)[labels.demoted] == -1).all()
    # bee 0's misread is overruled by the model, bee 1's correct decode is confirmed
    assert info["n_demoted_decodes_relabelled"] >= 9
    assert info["n_demoted_decodes_confirmed"] >= 9
    assert info["n_demoted_decodes_relabelled"] + info["n_demoted_decodes_confirmed"] <= 22
    assert evaluate(scene, labels)["wrong"] == 0
    assert_invariants(labels, 4)


def test_stray_and_conflict_demotions_partition_the_demoted_decodes():
    scene = make_scene(42, n_frames=300, decode_rate=0.9, stray_rate=0.04, min_separation=100.0)
    labels = solve_region(scene.table, scene.votes, 4, 4)
    info = info_of(labels)
    assert info["n_stray_anchors_demoted"] > 0
    assert labels.demoted is not None
    assert int(labels.demoted.sum()) == (
        info["n_stray_anchors_demoted"] + info["n_conflict_decodes_demoted"]
    )
    assert info["n_demoted_decodes_relabelled"] + info["n_demoted_decodes_confirmed"] <= int(
        labels.demoted.sum()
    )
    assert_invariants(labels, 4)


def test_single_decodes_are_the_decodes_before_the_conflict_test():
    rows = [(0, 0, 1, 1.0), (0, 1, 1, 1.0), (0, 2, 0, 1.0), (0, 2, 2, 1.0), (1, 0, 3, 0.2)]
    votes = votes_of(*rows)
    single = single_decodes(votes, (2, 3), 4, 0.5)
    # slots 0 and 1 of frame 0 both read tag 1 (kept here), slot 2 has two tags, frame 1 is weak
    assert single.tolist() == [[1, 1, -1], [-1, -1, -1]]
    assert identity.candidate_anchors(votes, (2, 3), 4, 0.5).tolist() == [[-1, -1, -1]] * 2


def _gap_labels(n_frames=10):
    """One tag decoded in frames 0-3 and 7-9 on a line, with a 3-frame gap in between."""
    shape = (n_frames, 1)
    tag_xy = np.full((n_frames, 1, 2), np.nan)
    present = [0, 1, 2, 3, 7, 8, 9]
    tag_xy[present, 0] = [[10.0 * f, 0.0] for f in present]
    tag_src = np.zeros(shape, dtype=np.int8)
    tag_src[present] = Source.DECODE
    conf = np.full(shape, np.nan, dtype=np.float32)
    conf[present] = 1.0
    return RegionLabels(
        slot_tag=np.full(shape, -1, dtype=np.int16),
        slot_conf=np.full(shape, np.nan, dtype=np.float32),
        slot_source=np.zeros(shape, dtype=np.int8),
        slot_disputed=np.zeros(shape, dtype=bool),
        slot_flag=np.zeros(shape, dtype=np.int8),
        guess_tag=np.full(shape, -1, dtype=np.int16),
        guess_conf=np.zeros(shape, dtype=np.float32),
        tag_xy=tag_xy,
        tag_slot=np.full(shape, -1, dtype=np.int64),
        tag_src=tag_src,
        tag_conf=conf,
    )


def test_gap_filling_is_part_of_the_consistency_stage():
    on = _gap_labels()
    assert identity._fill_gaps(on, ConsistencyParams()) == 0
    assert (on.tag_src[4:7] == Source.INTERPOLATED).all()
    assert np.allclose(on.tag_xy[4:7, 0, 0], [40.0, 50.0, 60.0])
    off = _gap_labels()
    assert identity._fill_gaps(off, ConsistencyParams(enabled=False)) == 0
    assert (off.tag_src[4:7] == Source.NONE).all()
    assert np.isnan(off.tag_xy[4:7]).all()
    none = _gap_labels()
    assert identity._fill_gaps(none, ConsistencyParams(gap_max=0)) == 0
    assert np.isnan(none.tag_xy[4:7]).all()


def test_no_consistency_does_not_interpolate_ungated(monkeypatch):
    scene = make_scene(33, n_frames=200, absent={1: (0, 199)}, miss_rate=0.15)
    calls = []
    real = identity.interpolate

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(identity, "interpolate", spy)
    off = solve_region(scene.table, scene.votes, 4, 4, None, ConsistencyParams(enabled=False))
    assert not calls  # the ungated v1-style interpolation is never reached
    assert info_of(off, "consistency")["interpolation_refused_frames"] == 0
    solve_region(scene.table, scene.votes, 4, 4)
    assert calls  # with the stage on, it runs (gated)


def test_excess_counter_includes_culled_ghosts_and_flags_keep_the_ghost_cause():
    scene = make_scene(43, n_frames=300, ghost_rate=0.5, min_separation=100.0)
    labels = solve_region(scene.table, scene.votes, 4, 4)
    info = info_of(labels)
    removed = scene.table.valid & ~valid_of(labels)  # duplicates and culled instances
    flags = labels.slot_flag
    assert info["n_duplicate_instances"] == int((flags == SlotFlag.DUPLICATE_INSTANCE).sum())
    assert info["n_excess_instances"] == int(removed.sum()) - info["n_duplicate_instances"]
    assert info["n_ghost_instances_culled"] > 0
    # the flag of a culled ghost stays "ghost": the specific cause wins over "excess"
    culled_ghosts = int((flags[removed] == SlotFlag.GHOST).sum())
    assert info["n_ghost_instances_culled"] == culled_ghosts
    flagged_excess = int((flags == SlotFlag.EXCESS_INSTANCE).sum())
    assert flagged_excess == info["n_excess_instances"] - culled_ghosts


def test_low_confidence_is_refined_into_pile_no_anchor_and_below_threshold():
    # nothing is ever decoded: every instance lacks an anchor, except where a pile is the cause
    pile = make_scene(44, n_frames=200, decode_rate=0.0, near_passes=((0, 1, 100, 15),))
    labels = solve_region(pile.table, pile.votes, 4, 4)
    low = labels.slot_flag == SlotFlag.LOW_CONFIDENCE
    detail = labels.low_conf_reason
    assert detail is not None and low.any()
    assert (detail[~low] == 0).all()
    ent = labels.entities
    assert ent is not None
    rows, slots = np.nonzero(low)
    nn = ent.nn[rows, ent.of_slot[rows, slots]]
    in_pile = nn <= P.nn_near_px
    expect = np.where(in_pile, Reason.LOW_CONFIDENCE_IN_PILE, Reason.NO_ANCHOR_IN_TRACKLET)
    assert (detail[rows, slots] == expect).all()
    assert Reason.LOW_CONFIDENCE_IN_PILE in set(expect.tolist())
    assert Reason.NO_ANCHOR_IN_TRACKLET in set(expect.tolist())
    # anchored tracklets whose posterior cannot reach the threshold: plain "below min confidence"
    scene = make_scene(45, n_frames=200, decode_rate=0.6, min_separation=100.0)
    strict = solve_region(scene.table, scene.votes, 4, 4, IdentityParams(threshold=1.01))
    low = strict.slot_flag == SlotFlag.LOW_CONFIDENCE
    assert low.any()
    assert strict.low_conf_reason is not None
    assert (strict.low_conf_reason[low] == Reason.BELOW_MIN_CONFIDENCE).all()


def test_forward_backward_needs_a_frame():
    empty = entities_from(np.zeros((0, 3), dtype=np.int32))
    with pytest.raises(ValueError, match="at least one frame"):
        forward_backward(np.zeros((0, 3, 3)), empty, 3, P)


def test_the_sleap_tracklet_mode_warns_when_most_instances_have_no_track(caplog):
    scene = make_scene(46, n_frames=60)  # the scenes have no SLEAP track ids
    with caplog.at_level(logging.WARNING, logger="naps.identity"):
        solve_region(scene.table, scene.votes, 4, 4, tracklet_mode="sleap")
    assert "no SLEAP track" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="naps.identity"):
        solve_region(scene.table, scene.votes, 4, 4)
    assert "no SLEAP track" not in caplog.text
    scene.table.hint[:] = 3  # tracked input: no warning in sleap mode either
    with caplog.at_level(logging.WARNING, logger="naps.identity"):
        solve_region(scene.table, scene.votes, 4, 4, tracklet_mode="sleap")
    assert "no SLEAP track" not in caplog.text
