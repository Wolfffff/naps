"""Seeded property tests of the joint identity engine on random synthetic scenes.

Every property is checked on scenes drawn from a seeded generator (``random_scene``): 3-5 bees,
sometimes an untagged bee, absent bees, near passes (40 px), stray decodes, ghost instances and
fragmentation. Bees keep >= 45 px apart: two instances closer than 25 px are one detection to the
engine (duplicate removal), so overlapping bees are outside the model. No hypothesis dependency:
the seeds are the parameters.
"""

import numpy as np
import pytest

from naps.consistency import (
    ConsistencyParams,
    allowed_displacement,
    anchors_off_chain,
    bridge_violations,
    duplicate_stats,
)
from naps.identity import IdentityParams, SlotFlag, Votes, solve_region
from naps.instances import InstanceTable
from naps.sources import Source
from tests.scenes import anchors_of, assert_invariants, bee_slot, evaluate, make_scene

N_INVARIANT_SEEDS = 160
N_TRANSFORM_SEEDS = 24
THRESHOLD = IdentityParams().threshold
NEAR_THRESHOLD = 1e-4  # labels whose confidence is this close to the threshold may flip


def random_scene(seed):
    """A random region (a pure function of ``seed``)."""
    rng = np.random.default_rng(10_000 + seed)
    n_bees = int(rng.choice([3, 4, 4, 4, 5]))
    n_real = n_bees - int(rng.integers(0, 2)) if n_bees > 3 else n_bees
    n_frames = int(rng.integers(20, 140))
    absent = {}
    for bee in range(n_real):
        if rng.random() < 0.4:
            lo = int(rng.integers(0, n_frames))
            absent[bee] = (lo, int(rng.integers(lo, n_frames)))
    passes = ()
    if rng.random() < 0.4:
        a, b = rng.choice(n_bees, size=2, replace=False)
        passes = ((int(a), int(b), int(rng.integers(0, n_frames)), 40),)
    miss_rate = float(rng.choice([0.0, 0.0, 0.05]))
    scene = make_scene(
        seed,
        n_frames=n_frames,
        n_bees=n_bees,
        n_real=n_real,
        decode_rate=float(rng.uniform(0.0, 0.9)),
        absent=absent,
        near_passes=passes,
        stray_rate=float(rng.choice([0.0, 0.0, 0.02])),
        ghost_rate=float(rng.choice([0.0, 0.0, 0.1])),
        miss_rate=miss_rate,
        min_separation=float(rng.choice([45.0, 60.0])),
    )
    scene.extras["miss_rate"] = miss_rate
    return scene


def solve(scene, **kwargs):
    return solve_region(scene.table, scene.votes, scene.n_real, scene.n_bees, **kwargs)


def same(a, b):
    return np.array_equal(a, b, equal_nan=True)


# ------------------------------------------------------------------- structural invariants
@pytest.mark.parametrize("seed", range(N_INVARIANT_SEEDS))
def test_invariants_hold_on_random_scenes(seed):
    scene = random_scene(seed)
    labels = solve(scene)
    assert_invariants(labels, scene.n_real, THRESHOLD)
    # an instance that took no part is never labelled; ghosts and duplicates carry their flag
    assert (labels.slot_tag[~labels.valid] == -1).all()
    removed = scene.table.valid & ~labels.valid
    assert (labels.slot_flag[removed] != SlotFlag.NONE).all()
    # untagged bees are never emitted, ghosts never labelled
    ev = evaluate(scene, labels)
    assert ev["untagged_labelled"] == 0
    if scene.extras["miss_rate"] == 0:  # with a missed bee the ghost can fill its place
        assert ev["ghost_labelled"] == 0
    # the temporal-consistency stage leaves no implausible link involving a recovered label
    off = anchors_off_chain(labels.tag_xy, labels.tag_src)
    violations = bridge_violations(labels.tag_xy, labels.tag_src, ConsistencyParams(), off)
    assert violations["n_violations_own"] == 0
    recovered = labels.tag_src >= Source.TRACKLET
    step = np.linalg.norm(np.diff(labels.tag_xy, axis=0), axis=-1)
    touches = (recovered[1:] | recovered[:-1]) & ~(off[1:] | off[:-1])
    assert (step[touches & np.isfinite(step)] <= allowed_displacement(1.0)).all()
    # flags and the guess agree: a flagged instance has a guess below the threshold or a reason
    low = labels.slot_flag == SlotFlag.LOW_CONFIDENCE
    assert (labels.guess_conf[low] < THRESHOLD).all()


@pytest.mark.parametrize("seed", range(12))
def test_six_bees_hold_the_invariants(seed):
    scene = make_scene(
        seed, n_frames=30, n_bees=6, n_real=5, absent={2: (0, 29)}, min_separation=45.0
    )
    labels = solve(scene)
    assert_invariants(labels, 5)
    assert evaluate(scene, labels)["wrong_inferred"] == 0


@pytest.mark.parametrize("seed", range(24))
def test_clean_separated_bees_never_get_recovered_labels_next_to_another_tag(seed):
    scene = make_scene(
        seed, n_frames=120, absent={seed % 4: (10, 110)}, miss_rate=0.05, min_separation=60.0
    )
    labels = solve(scene)
    stats = duplicate_stats(labels.tag_xy, labels.tag_src, gate=50.0)
    assert stats["recovered_labels_near_other_tag"] == 0
    assert evaluate(scene, labels)["wrong_inferred"] == 0


# ------------------------------------------------------------------- abstain, do not guess
@pytest.mark.parametrize("seed", range(20))
def test_two_inseparable_undecoded_bees_are_never_labelled_wrongly(seed):
    scene = make_scene(
        seed, n_frames=100, absent={2: (0, 99), 3: (0, 99)}, min_separation=60.0, miss_rate=0.0
    )
    labels = solve(scene)
    ev = evaluate(scene, labels)
    assert ev["wrong"] == 0
    for bee in (2, 3):
        assert (labels.slot_tag[np.arange(100), bee_slot(scene, bee)] == -1).all()
    assert_invariants(labels, 4)


# ------------------------------------------------------------------- determinism
@pytest.mark.parametrize("seed", range(N_TRANSFORM_SEEDS))
def test_two_runs_are_identical(seed):
    scene = random_scene(1000 + seed)
    a, b = solve(scene), solve(scene)
    for name in ("slot_tag", "slot_conf", "slot_source", "slot_flag", "slot_disputed"):
        assert same(getattr(a, name), getattr(b, name)), name
    for name in ("tag_xy", "tag_src", "tag_conf", "tag_slot", "guess_tag", "guess_conf"):
        assert same(getattr(a, name), getattr(b, name)), name
    assert same(a.marginals, b.marginals)


# ------------------------------------------------------------------- equivariance
def agree(base, other, mapped_tags=None):
    """Labels of two runs agree, except where the original is within 1e-4 of the threshold."""
    ref = base.slot_tag if mapped_tags is None else mapped_tags
    unstable = np.abs(base.guess_conf - THRESHOLD) < NEAR_THRESHOLD
    unstable |= np.abs(other.guess_conf - THRESHOLD) < NEAR_THRESHOLD
    return ((ref == other.slot_tag) | unstable).all()


@pytest.mark.parametrize("seed", range(N_TRANSFORM_SEEDS))
def test_relabelling_the_tags_relabels_the_output(seed):
    scene = random_scene(2000 + seed)
    rng = np.random.default_rng(seed)
    perm = np.arange(scene.n_bees)
    perm[: scene.n_real] = rng.permutation(scene.n_real)
    votes = Votes(scene.votes.frame, scene.votes.slot, perm[scene.votes.tag], scene.votes.weight)
    base = solve(scene)
    other = solve_region(scene.table, votes, scene.n_real, scene.n_bees)
    mapped = np.where(base.slot_tag >= 0, perm[np.maximum(base.slot_tag, 0)], -1)
    assert agree(base, other, mapped)
    both = (mapped >= 0) & (other.slot_tag >= 0)
    assert np.allclose(base.slot_conf[both], other.slot_conf[both], atol=1e-4)
    assert (base.slot_source[both] == other.slot_source[both]).mean() > 0.99
    assert (anchors_of(base) >= 0).sum() == (anchors_of(other) >= 0).sum()


@pytest.mark.parametrize("seed", range(N_TRANSFORM_SEEDS))
def test_reordering_the_slots_does_not_change_the_labels_of_the_bees(seed):
    scene = random_scene(3000 + seed)
    t = scene.table
    n_frames, n_slots = t.xy.shape[:2]
    rng = np.random.default_rng(seed)
    order = np.stack([rng.permutation(n_slots) for _ in range(n_frames)])  # new slot j <- old order
    inverse = np.argsort(order, axis=1)
    rows = np.arange(n_frames)[:, None]

    shuffled = InstanceTable(
        t.first_frame, t.xy[rows, order], t.tag_xy[rows, order], t.score[rows, order],
        t.hint[rows, order], t.inst_index[rows, order], t.heading[rows, order],
    )  # fmt: skip
    votes = Votes(
        scene.votes.frame,
        inverse[scene.votes.frame, scene.votes.slot],
        scene.votes.tag,
        scene.votes.weight,
    )
    base = solve(scene)
    other = solve_region(shuffled, votes, scene.n_real, scene.n_bees)
    moved_tag = np.take_along_axis(other.slot_tag, inverse, axis=1)  # old slot layout again
    moved_guess = np.take_along_axis(other.guess_conf, inverse, axis=1)
    unstable = (np.abs(base.guess_conf - THRESHOLD) < NEAR_THRESHOLD) | (
        np.abs(moved_guess - THRESHOLD) < NEAR_THRESHOLD
    )
    assert ((base.slot_tag == moved_tag) | unstable).all()


# ------------------------------------------------------------------- head and tail
@pytest.mark.parametrize("seed", range(40))
def test_first_and_last_frames_are_labelled_when_their_tracklet_is_anchored(seed):
    scene = make_scene(
        seed, n_frames=150, n_bees=4, decode_rate=0.5, absent={seed % 4: (0, 149)},
        min_separation=45.0,
    )  # fmt: skip
    labels = solve(scene)
    tr = labels.tracklets
    for row, window in ((0, slice(0, 30)), (149, slice(120, 150))):
        for slot in np.flatnonzero(tr.tid[row] >= 0):
            tid = tr.tid[row, slot]
            anchored = ((tr.tid[window] == tid) & (labels.anchors[window] >= 0)).any()
            if anchored:
                assert labels.slot_tag[row, slot] >= 0, (row, slot)


def test_random_scene_generator_is_a_pure_function_of_the_seed():
    a, b = random_scene(5), random_scene(5)
    assert np.array_equal(a.table.xy, b.table.xy, equal_nan=True)
    assert np.array_equal(a.votes.slot, b.votes.slot)


def test_inferred_labels_are_almost_always_right_over_many_scenes():
    # seeds 0-399 of random_scene: 8 wrong of 56,198 inferred labels (1.4e-4), all at confidence
    # >= 0.99; the bound is 7x that rate. (The model's own claim is <= 1 %.)
    inferred = wrong = 0
    for seed in range(120):
        scene = random_scene(seed)
        labels = solve(scene)
        inferred += int(((labels.slot_tag >= 0) & (labels.slot_source != Source.DECODE)).sum())
        wrong += int(evaluate(scene, labels)["wrong_inferred"])
    assert inferred > 10_000
    assert wrong <= 0.001 * inferred, (wrong, inferred)
