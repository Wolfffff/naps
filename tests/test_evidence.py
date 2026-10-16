import numpy as np
import pytest

from naps.cost_matrix import CostMatrix
from naps.evidence import ArucoEvidence, EvidenceSource, IdtrackerEvidence, collect_votes

TAGS = [10, 11, 12, 13]


def lane_world(n_frames=300, perm=(0, 1, 2, 3), prob=0.95):
    """Four bees on well separated lanes. Track k carries TAGS[k].

    Idtracker animal a follows the bee with tag TAGS[perm[a]] (labels are
    arbitrary per video, so perm is a label permutation).
    Returns positions[frame][track], traj (F, 4, 2), id_prob (F, 4), truth {animal: tag}.
    """
    frames = np.arange(n_frames)
    bee_xy = np.zeros((n_frames, 4, 2))
    for k in range(4):
        bee_xy[:, k, 0] = 200 + 0.5 * frames
        bee_xy[:, k, 1] = 300 * k + 100
    positions = {f: {k: bee_xy[f, k] + 0.3 for k in range(4)} for f in range(n_frames)}
    traj = np.stack([bee_xy[:, perm[a]] + 10.0 for a in range(4)], axis=1)  # centroid != tag node
    id_prob = np.full((n_frames, 4), prob)
    truth = {a: TAGS[perm[a]] for a in range(4)}
    return positions, traj, id_prob, truth


def decodes_every(n_frames, step, tags=TAGS, tracks=range(4), skip=lambda f, k: False):
    """ArUco decode of the right tag every `step` frames, [None] otherwise."""
    out = {}
    for f in range(n_frames):
        out[f] = {k: [tags[k]] if (f % step == 0 and not skip(f, k)) else [None] for k in tracks}
    return out


# ------------------------------------------------------------------ weights
def test_aruco_evidence_weights_closed_set_and_keys():
    decodes = {0: {0: [None], 1: [10, 99], 2: [11, 11]}}
    ev = ArucoEvidence(decodes, allowed_tags=[10, 11], weight=1.0)
    votes = ev.votes(0)
    assert votes[0] == []  # track key kept even with nothing usable
    assert votes[1] == [(10, 1.0)]  # 99 is outside the closed set
    assert votes[2] == [(11, 1.0), (11, 1.0)]
    assert ev.n_out_of_set == 1
    assert all(isinstance(t, int) for v in votes.values() for t, _ in v)
    assert ArucoEvidence(decodes, weight=0.5).votes(0)[1] == [(10, 0.5), (99, 0.5)]


def test_collect_votes_without_sources_is_legacy_equivalent():
    decodes = {f: {0: [5, None], 1: [None]} for f in range(40)}
    positions = {f: {0: np.array([0.0, 0.0]), 1: np.array([9.0, 9.0])} for f in decodes}
    merged = collect_votes(decodes, positions)
    legacy = CostMatrix.fromDict(decodes, 0, 39, 3).assignTrackTagPairs()
    new = CostMatrix.fromDict(merged, 0, 39, 3).assignTrackTagPairs()
    assert {f: dict(v) for f, v in legacy.items()} == {f: dict(v) for f, v in new.items()}
    assert set(merged[0]) == {0, 1}  # tracks without usable decodes stay present


def test_cost_matrix_weighted_votes_and_float_residue():
    # Many light votes for tag 5 vs fewer heavy votes for tag 6: the larger total wins,
    # and fractional bookkeeping over a rolling window leaves no residue.
    value_dict = {f: {"a": [(5, 0.1), (6, 0.25)] if f % 2 else [(5, 0.1)]} for f in range(200)}
    assignment = CostMatrix.fromDict(value_dict, 0, 199, 5).assignTrackTagPairs()
    tags = [t for fr in assignment.values() for t in fr.values()]
    assert tags and set(tags) == {6}
    assert all(isinstance(t, (int, np.integer)) for t in tags)

    value_dict = {f: {"a": [(5, 0.3), (6, 0.1)]} for f in range(100)}
    assignment = CostMatrix.fromDict(value_dict, 0, 99, 5).assignTrackTagPairs()
    assert {t for fr in assignment.values() for t in fr.values()} == {5}


# ---------------------------------------------------------------- idtracker
def build(perm, n=300, **kw):
    positions, traj, prob, truth = lane_world(n, perm)
    decodes = decodes_every(n, 3)
    ev = IdtrackerEvidence(traj, prob, offset=(-10, -10), allowed_tags=TAGS, **kw)
    aruco_votes = ArucoEvidence(decodes, TAGS).votes
    prior = {f: aruco_votes(f) for f in decodes}
    return ev, positions, prior, truth


@pytest.mark.parametrize("perm", [(0, 1, 2, 3), (2, 0, 3, 1), (3, 2, 1, 0)])
def test_alignment_recovers_label_permutation(perm):
    ev, positions, prior, truth = build(perm)
    ev.prepare(positions, prior)
    assert ev.id_to_tag == truth
    assert all(e.accepted and e.purity == 1.0 for e in ev.alignment.values())


def test_votes_use_aligned_tag_weight_and_only_fill_aruco_gaps():
    perm = (1, 3, 0, 2)
    ev, positions, prior, _ = build(perm, weight=0.5)
    ev.prepare(positions, prior)
    n_with = 0
    for f in range(300):
        votes = ev.votes(f, positions[f], prior[f])
        if f % 3 == 0:
            assert votes == {}  # ArUco decoded every track: idtracker stays out of it
            continue
        assert set(votes) == {0, 1, 2, 3}
        for track, vs in votes.items():
            assert vs == [(TAGS[track], pytest.approx(0.5 * 0.95))]
            n_with += 1
    assert n_with > 0
    # without the fallback-only rule it also votes on decoded frames
    ev2, positions, prior, _ = build(perm, only_when_aruco_missing=False)
    ev2.prepare(positions, prior)
    assert set(ev2.votes(0, positions[0], prior[0])) == {0, 1, 2, 3}


def test_nan_id_probability_is_gated_but_still_blocks_neighbours():
    ev, positions, prior, _ = build((0, 1, 2, 3))
    ev.prepare(positions, prior)
    # Animal 2 is only an interpolated position (NaN) in frames 100-199, animal 3 is absent (0)
    ev.id_probabilities[100:200, 2] = np.nan
    ev.id_probabilities[100:200, 3] = 0.0
    ev._join_cache.clear()
    for f in range(100, 200):
        assert not {2, 3} & set(ev.votes(f, positions[f], {k: [] for k in range(4)}))
    for f in (50, 250):
        assert {2, 3} <= set(ev.votes(f, positions[f], {k: [] for k in range(4)}))


def test_interpolated_animal_cannot_steal_a_track():
    # Track 0 sits 3px from an interpolated (NaN-prob) animal 1 and 40px from the
    # real animal 0. The interpolated animal wins the join; it must emit nothing,
    # and the real animal must not be forced onto track 0.
    traj = np.array([[[40.0, 0.0], [3.0, 0.0]]])
    prob = np.array([[0.99, np.nan]])
    ev = IdtrackerEvidence(traj, prob, id_to_tag={0: 10, 1: 11}, ambiguity_ratio=0.9)
    votes = ev.votes(0, {0: np.array([0.0, 0.0])}, {})
    assert votes == {}


def test_low_purity_alignment_refuses_to_emit():
    positions, traj, prob, _ = lane_world(300)
    rng = np.random.default_rng(0)
    decodes = {
        f: {k: [int(rng.choice(TAGS))] for k in range(4)} for f in range(300)
    }  # ArUco unrelated to idtracker label
    prior = {f: ArucoEvidence(decodes, TAGS).votes(f) for f in decodes}
    ev = IdtrackerEvidence(traj, prob, offset=(-10, -10), allowed_tags=TAGS)
    ev.prepare(positions, prior)
    assert ev.id_to_tag == {}
    assert all(not e.accepted and "purity" in e.reason for e in ev.alignment.values())
    assert all(ev.votes(f, positions[f], {}) == {} for f in range(300))


def test_low_support_alignment_refuses_to_emit():
    positions, traj, prob, _ = lane_world(300)
    decodes = decodes_every(300, 100)  # only 3 decodes per bee
    prior = {f: ArucoEvidence(decodes, TAGS).votes(f) for f in decodes}
    ev = IdtrackerEvidence(traj, prob, offset=(-10, -10), allowed_tags=TAGS, min_support=30)
    ev.prepare(positions, prior)
    assert ev.id_to_tag == {}
    assert all("support" in e.reason for e in ev.alignment.values())


def test_duplicate_tag_claim_refuses_both_animals():
    # idtracker split one bee over two labels: both animals follow tag 10's bee
    positions, traj, prob, _ = lane_world(300)
    traj[:, 1] = traj[:, 0] + 1.0
    decodes = decodes_every(300, 2)
    prior = {f: ArucoEvidence(decodes, TAGS).votes(f) for f in decodes}
    ev = IdtrackerEvidence(traj, prob, offset=(-10, -10), allowed_tags=TAGS)
    ev.prepare(positions, prior)
    assert 0 not in ev.id_to_tag and 1 not in ev.id_to_tag


def test_neighbour_cross_votes_do_not_poison_alignment():
    # Track 0's crop also sees neighbour tag 11 in 40% of frames; those frames carry two
    # distinct tags and are skipped, so purity stays 1.0
    positions, traj, prob, truth = lane_world(300)
    decodes = decodes_every(300, 2)
    for f in range(0, 300, 5):
        decodes[f][0] = [TAGS[0], TAGS[1]]
    prior = {f: ArucoEvidence(decodes, TAGS).votes(f) for f in decodes}
    ev = IdtrackerEvidence(traj, prob, offset=(-10, -10), allowed_tags=TAGS)
    ev.prepare(positions, prior)
    assert ev.id_to_tag == truth
    assert ev.alignment[0].purity == 1.0


def test_swap_safety_on_crossing_never_links_wrong():
    # Two bees (tags 10, 11) approach, pass through each other (centroids within 2 px) and
    # separate. The idtracker centroid carries a different offset from the SLEAP tag node, so a
    # naive nearest-neighbour join would swap them. Every emitted vote must be correct.
    n = 200
    f = np.arange(n)
    x0 = 100 + 3.0 * (f - 100)  # bee A moves right
    x1 = 100 - 3.0 * (f - 100)  # bee B moves left
    bee = np.zeros((n, 2, 2))
    bee[:, 0, 0], bee[:, 1, 0] = x0 + 300, x1 + 300
    bee[:, 0, 1], bee[:, 1, 1] = 50.0, 52.0
    positions = {i: {0: bee[i, 0] + [5.0, 0], 1: bee[i, 1] + [-5.0, 0]} for i in range(n)}
    traj = bee[:, ::-1]  # animal 0 follows bee B (labels arbitrary)
    prob = np.full((n, 2), 0.9)
    ev = IdtrackerEvidence(
        traj, prob, id_to_tag={0: 11, 1: 10}, max_dist=60.0, ambiguity_ratio=0.5, min_margin=20.0
    )
    truth = {0: 10, 1: 11}
    emitted, crossing_emitted = 0, 0
    for i in range(n):
        for track, vs in ev.votes(i, positions[i], {}).items():
            emitted += 1
            assert vs == [(truth[track], pytest.approx(0.45))]  # never a wrong link
            if 96 <= i <= 103:
                crossing_emitted += 1
    assert emitted > 100  # it does work away from the crossing
    assert crossing_emitted == 0  # and abstains around it


def test_exactly_equidistant_animal_abstains():
    traj = np.array([[[10.0, 0.0]], [[10.0, 0.0]]])
    prob = np.array([[0.99], [0.99]])
    ev = IdtrackerEvidence(traj, prob, id_to_tag={0: 10}, max_dist=50.0)
    tracks = {0: np.array([0.0, 0.0]), 1: np.array([20.0, 0.0])}
    assert ev.votes(0, tracks, {}) == {}
    tracks[1] = np.array([60.0, 0.0])  # now clearly closer to track 0
    assert ev.votes(1, tracks, {}) == {0: [(10, pytest.approx(0.495))]}


def test_distance_gate_and_frame_offset():
    traj = np.zeros((4, 1, 2))
    prob = np.full((4, 1), 1.0)
    ev = IdtrackerEvidence(traj, prob, id_to_tag={0: 10}, max_dist=30.0, frame_offset=5)
    near, far = {0: np.array([10.0, 0.0])}, {0: np.array([40.0, 0.0])}
    assert ev.votes(5, near, {}) and ev.votes(6, near, {})
    assert ev.votes(7, far, {}) == {}  # beyond max_dist
    assert ev.votes(4, near, {}) == {} and ev.votes(9, near, {}) == {}  # outside the file


def test_from_file_roundtrip_and_prob_shape(tmp_path):
    _, traj, prob, _ = lane_world(100)
    path = tmp_path / "trajectories.npy"
    # idtracker.ai v6 stores a pickled dict inside the .npy
    payload = {"trajectories": traj, "id_probabilities": prob[..., None], "version": "6"}
    np.save(path, np.array(payload, dtype=object))
    ev = IdtrackerEvidence.from_file(path, offset=(-10, -10), allowed_tags=TAGS)
    assert ev.id_probabilities.shape == (100, 4)
    np.save(tmp_path / "bad.npy", np.array({"trajectories": traj}, dtype=object))
    with pytest.raises(ValueError, match="id_probabilities"):
        IdtrackerEvidence.from_file(tmp_path / "bad.npy")
    np.save(tmp_path / "worse.npy", np.array({"id_probabilities": prob}, dtype=object))
    with pytest.raises(ValueError, match="trajectories"):
        IdtrackerEvidence.from_file(tmp_path / "worse.npy")


def test_idtracker_bridges_tracklet_restart_through_cost_matrix():
    # Bee 0's SLEAP tracklet ends at frame 100 and the bee continues as new tracklet 4,
    # whose tag is never decoded (e.g. a huddle). Legacy NAPS leaves tracklet 4 unidentified;
    # idtracker continuity, aligned on the earlier decodes, identifies it.
    perm = (2, 0, 3, 1)
    positions, traj, prob, truth = lane_world(300, perm)
    decodes = decodes_every(300, 3)
    for f in range(300):
        if f >= 100:
            positions[f][4] = positions[f].pop(0)
            decodes[f][4] = [None]
            del decodes[f][0]
    ev = IdtrackerEvidence(traj, prob, offset=(-10, -10), allowed_tags=TAGS)
    base = CostMatrix.fromDict(collect_votes(decodes, positions, [], TAGS), 0, 299, 5)
    fused = CostMatrix.fromDict(collect_votes(decodes, positions, [ev], TAGS), 0, 299, 5)
    base_tags = base.assignTrackTagPairs()
    fused_tags = fused.assignTrackTagPairs()
    assert 4 not in base_tags[150]
    assert fused_tags[150] == {4: 10, 1: 11, 2: 12, 3: 13}
    assert ev.id_to_tag == truth
    for f in range(110, 290):
        assert fused_tags[f] == {4: 10, 1: 11, 2: 12, 3: 13}
    for f in range(10, 90):
        assert fused_tags[f] == {k: TAGS[k] for k in range(4)}


# ------------------------------------------------------------ validation and edge cases
def test_aruco_evidence_normalises_tags_and_applies_the_vote_filter():
    decodes = {0: {0: [10.0, np.int64(11), "x", None, np.nan, 12.5, "13"]}}
    ev = ArucoEvidence(decodes, vote_filter=lambda frame, track, tag: tag != 11)
    assert ev.votes(0)[0] == [(10, 1.0), (13, 1.0)]
    assert ev.votes(99) == {}  # unknown frame


def test_closed_sets_are_validated():
    with pytest.raises(ValueError, match="not a valid tag id"):
        ArucoEvidence({}, allowed_tags=[10, None])
    with pytest.raises(ValueError, match="not a valid tag id"):
        IdtrackerEvidence(np.zeros((1, 1, 2)), np.ones((1, 1)), allowed_tags=[10.5])
    assert IdtrackerEvidence(
        np.zeros((1, 1, 2)), np.ones((1, 1)), allowed_tags=[12.0, "11"]
    ).allowed_tags == [11, 12]


def test_idtracker_validates_array_shapes():
    with pytest.raises(ValueError, match="frames, animals, 2"):
        IdtrackerEvidence(np.zeros((5, 2, 3)), np.ones((5, 2)))
    with pytest.raises(ValueError, match="does not match"):
        IdtrackerEvidence(np.zeros((5, 2, 2)), np.ones((5, 3)))


def test_idtracker_with_fixed_mapping_skips_alignment():
    traj = np.zeros((3, 2, 2))
    ev = IdtrackerEvidence(traj, np.ones((3, 2)), id_to_tag={0: 10, 1: 11})
    ev.prepare({}, {})
    assert ev.id_to_tag == {0: 10, 1: 11}
    assert ev.alignment == {}


def test_idtracker_fixed_mapping_tags_are_normalised_and_validated():
    traj = np.zeros((3, 2, 2))
    ev = IdtrackerEvidence(traj, np.ones((3, 2)), id_to_tag={0: 10.0, 1: "11"})
    assert ev.id_to_tag == {0: 10, 1: 11}
    assert all(type(tag) is int for tag in ev.id_to_tag.values())
    ev.prepare({}, {})
    assert ev.id_to_tag == {0: 10, 1: 11}
    assert all(type(tag) is int for tag in ev.id_to_tag.values())
    for bad in (None, float("nan"), 10.5, "x"):
        with pytest.raises(ValueError, match="not a valid tag id"):
            IdtrackerEvidence(traj, np.ones((3, 2)), id_to_tag={0: bad})


def test_idtracker_emits_nothing_before_alignment_and_without_cooccurrence():
    positions, traj, prob, _ = lane_world(60)
    ev = IdtrackerEvidence(traj, prob, offset=(-10, -10), allowed_tags=TAGS)
    assert ev.votes(0, positions[0], {}) == {}  # prepare() has not run
    no_decodes = {f: {k: [] for k in range(4)} for f in range(60)}
    ev.prepare(positions, no_decodes)
    assert ev.id_to_tag == {}
    assert {e.reason for e in ev.alignment.values()} == {"support 0 < 30"}
    assert ev.votes(0, positions[0], {}) == {}
    # without a closed set there are no tags to align to at all
    free = IdtrackerEvidence(traj, prob, offset=(-10, -10))
    free.prepare(positions, no_decodes)
    assert free.id_to_tag == {}
    assert {e.reason for e in free.alignment.values()} == {"no co-occurrence"}


def test_idtracker_animal_without_cooccurrence_is_refused_not_a_crash():
    # More animals than tags and no thresholds: an animal that never co-occurs with a decode
    # has no best tag. It must be refused (this path used to hit int(None)).
    positions, traj, prob, _ = lane_world(120)
    decodes = {f: {0: [TAGS[0]], 1: [None], 2: [None], 3: [None]} for f in range(120)}
    prior = {f: ArucoEvidence(decodes, TAGS).votes(f) for f in decodes}
    ev = IdtrackerEvidence(
        traj, prob, offset=(-10, -10), allowed_tags=[TAGS[0]], min_support=0, min_purity=0.0
    )
    ev.prepare(positions, prior)
    assert ev.id_to_tag == {0: TAGS[0]}
    assert [ev.alignment[a].accepted for a in range(4)] == [True, False, False, False]
    assert {ev.alignment[a].reason for a in (1, 2, 3)} == {"no co-occurrence"}


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"support": 5}, "support 5 < 30"),
        ({"purity": 0.5}, "purity 0.50 < 0.9"),
        ({"top": None}, "no co-occurrence"),
        ({"n_claims": 2}, "tag 3 claimed by 2 animals"),
        ({"one_to_one": False}, "one-to-one assignment disagrees with best tag"),
        ({}, "ok"),
    ],
)
def test_alignment_refusal_reasons_and_their_order(kwargs, reason):
    ev = IdtrackerEvidence(np.zeros((1, 1, 2)), np.ones((1, 1)))
    arguments = {"support": 40, "purity": 1.0, "n_claims": 1, "one_to_one": True, "top": 3}
    arguments.update(kwargs)
    assert ev._refusal_reason(**arguments) == reason


class _FixedSource(EvidenceSource):
    name = "fixed"

    def __init__(self, votes_by_frame):
        self.votes_by_frame = votes_by_frame
        self.prepared_with = None

    def prepare(self, positions, prior_votes):
        self.prepared_with = (positions, prior_votes)

    def votes(self, frame, track_positions, prior=None):
        return self.votes_by_frame.get(frame, {})


def test_collect_votes_extra_sources_see_aruco_as_prior_and_cannot_add_tracks():
    decodes = {0: {"a": [1]}, 1: {"a": [None]}}
    positions = {0: {"a": (0, 0)}}
    source = _FixedSource({0: {"a": [(2, 0.5)], "ghost": [(3, 0.5)]}, 1: {"a": [(2, 0.5)]}})

    merged = collect_votes(decodes, positions, [source])

    assert merged == {0: {"a": [(1, 1.0), (2, 0.5)]}, 1: {"a": [(2, 0.5)]}}
    assert source.prepared_with is not None
    prepared_positions, prior = source.prepared_with
    assert prepared_positions is positions
    assert prior == {0: {"a": [(1, 1.0)]}, 1: {"a": []}}


def test_collect_votes_applies_the_vote_filter_to_every_source():
    decodes = {0: {"a": [1, 5]}}
    source = _FixedSource({0: {"a": [(2, 0.5), (6, 0.5)]}})

    merged = collect_votes(
        decodes, {}, [source], vote_filter=lambda frame, track, tag: tag < 5, aruco_weight=2.0
    )

    assert merged == {0: {"a": [(1, 2.0), (2, 0.5)]}}


def test_collect_votes_restricts_aruco_to_the_closed_set():
    merged = collect_votes({0: {"a": [1, 9]}}, {}, allowed_tags=[1])
    assert merged == {0: {"a": [(1, 1.0)]}}


def test_alignment_ignores_frames_outside_the_file_and_unconfident_animals():
    positions, traj, prob, _ = lane_world(300)
    decodes = decodes_every(300, 2)
    prior = {f: ArucoEvidence(decodes, TAGS).votes(f) for f in decodes}
    prior[10_000] = {0: [(TAGS[0], 1.0)]}  # a frame the trajectories do not cover
    prob[:, 3] = 0.1  # animal 3 is never confident enough to count
    ev = IdtrackerEvidence(traj, prob, offset=(-10, -10), allowed_tags=TAGS)

    ev.prepare(positions, prior)

    assert set(ev.id_to_tag) == {0, 1, 2}
    assert ev.alignment[3].support == 0 and not ev.alignment[3].accepted


def test_join_skips_tracks_without_a_usable_position():
    ev = IdtrackerEvidence(np.zeros((1, 1, 2)), np.ones((1, 1)), id_to_tag={0: 10})
    tracks = {"nan": np.array([np.nan, 0.0]), "short": np.array([1.0]), "ok": np.array([5.0, 0.0])}

    assert ev.votes(0, tracks, {}) == {"ok": [(10, pytest.approx(0.5))]}

    # the join is cached per frame, so use fresh sources for the degenerate inputs
    for degenerate in ({}, {"nan": np.array([np.nan, 0.0])}):
        fresh = IdtrackerEvidence(np.zeros((1, 1, 2)), np.ones((1, 1)), id_to_tag={0: 10})
        assert fresh.votes(0, degenerate, {}) == {}


def test_evidence_source_is_abstract_and_prepare_is_optional():
    with pytest.raises(TypeError):
        EvidenceSource()  # ty: ignore[call-non-callable]
    _FixedSource({}).votes(0, {})
    ArucoEvidence({}).prepare({}, {})
