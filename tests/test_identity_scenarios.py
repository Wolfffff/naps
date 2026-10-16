"""Synthetic scenarios for the joint identity engine (failure modes found in the research).

Every scenario is a pure function of its seed (``tests/scenes.make_scene``). The thresholds in
``EXPECTED`` were calibrated by running seeds 0-9 of each scenario (see the comment above each
entry: the observed worst case) and loosened by a margin; they are bounds, not targets.
"""

import numpy as np
import pytest

from naps.identity import SlotFlag, solve_region
from naps.sources import Source
from tests.scenes import (
    Scene,
    assert_invariants,
    bee_slot,
    evaluate,
    info_of,
    make_scene,
    vote_rows,
    with_votes,
)

SEEDS = (0, 1, 2)
# Real bees are never closer than ~50 px (p0.01 of the inter-tag distance in real data), and the
# engine drops the lower-scored of two instances closer than 25 px as a duplicate: the scenes keep
# bees 45 px apart and let them meet at 40 px.
MIN_SEPARATION = 45.0
PASS_GAP = 40


def scene_of(seed, **kwargs):
    return make_scene(seed, min_separation=MIN_SEPARATION, **kwargs)


def build_clean(seed):
    return scene_of(seed)


def build_one_bee_absent(seed):
    return scene_of(seed, n_frames=400, absent={3: (50, 349)})


def build_two_bees_symmetric(seed):
    # bees 2 and 3 are never decoded and nothing tells them apart
    return scene_of(seed, n_frames=300, absent={2: (0, 299), 3: (0, 299)})


def build_two_bees_distinguishable(seed):
    # bee 2 is decoded before, bee 3 after the stretch in which both are undecoded
    return scene_of(seed, n_frames=400, absent={2: (100, 399), 3: (0, 299)})


def build_near_pass(seed):
    return scene_of(
        seed, n_frames=400, near_passes=((0, 1, 200, PASS_GAP),), absent={3: (150, 399)}
    )


def build_dropout_across_near_pass(seed):
    return scene_of(
        seed, n_frames=400, near_passes=((0, 1, 200, PASS_GAP),), absent={0: (100, 300)}
    )


def build_fragmentation(seed):
    return scene_of(seed, n_frames=400, miss_rate=0.1, absent={1: (30, 380)})


def build_ghosts(seed):
    return scene_of(seed, n_frames=400, ghost_rate=0.15, absent={0: (0, 399)})


def build_three_tags(seed):
    return scene_of(seed, n_frames=300, n_real=3)


def build_strays(seed):
    return scene_of(seed, n_frames=400, stray_rate=0.01, absent={3: (50, 399)})


# name -> (builder, n_real, n_bees); the observed worst case over seeds 0-9 is noted per scenario
SCENARIOS = {
    "clean": (build_clean, 4, 4),
    "one_bee_absent": (build_one_bee_absent, 4, 4),
    "two_bees_symmetric": (build_two_bees_symmetric, 4, 4),
    "two_bees_distinguishable": (build_two_bees_distinguishable, 4, 4),
    "near_pass": (build_near_pass, 4, 4),
    "dropout_across_near_pass": (build_dropout_across_near_pass, 4, 4),
    "fragmentation": (build_fragmentation, 4, 4),
    "ghosts": (build_ghosts, 4, 4),
    "three_tags": (build_three_tags, 3, 4),
    "strays": (build_strays, 4, 4),
}

# name -> (min coverage, max wrong inferred labels, max wrong decoded labels, min recovered
# coverage). Observed worst case over seeds 0-9 in the comment (coverage / recovered coverage /
# wrong inferred / wrong decoded); a wrong *decode* is an immutable anchor, not an engine error.
EXPECTED = {
    "clean": (0.98, 0, 0, 0.97),  # 0.994 / 0.984 / 0 / 0
    "one_bee_absent": (0.98, 0, 0, 0.97),  # 0.994 / 0.988 / 0 / 0
    "two_bees_symmetric": (0.45, 0, 0, 0.2),  # 0.4975 / 0.271 (bees 0,1 only) / 0 / 0
    "two_bees_distinguishable": (0.98, 0, 0, 0.97),  # 0.994 / 0.990 / 0 / 0
    "near_pass": (0.98, 0, 0, 0.97),  # 0.9925 / 0.985 / 0 / 0
    "dropout_across_near_pass": (0.97, 2, 0, 0.96),  # 0.992 / 0.983 / 1 (seed 4) / 0
    "fragmentation": (0.97, 0, 0, 0.96),  # 0.990 / 0.980 / 0 / 0
    "ghosts": (0.98, 0, 0, 0.97),  # 0.9956 / 0.992 / 0 / 0
    "three_tags": (0.98, 0, 0, 0.96),  # 0.990 / 0.975 / 0 / 0
    "strays": (0.98, 0, 10, 0.97),  # 0.995 / 0.9905 / 0 / 5 (the strays themselves)
}


def run(name, seed):
    build, n_real, n_bees = SCENARIOS[name]
    scene = build(seed)
    return scene, solve_region(scene.table, scene.votes, n_real, n_bees)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
@pytest.mark.parametrize("seed", SEEDS)
def test_scenario_meets_its_bounds(name, seed):
    scene, labels = run(name, seed)
    ev = evaluate(scene, labels)
    min_cov, max_inferred, max_decoded, min_rec = EXPECTED[name]
    assert ev["wrong_inferred"] <= max_inferred, ev
    assert ev["wrong"] - ev["wrong_inferred"] <= max_decoded, ev
    assert ev["coverage"] >= min_cov, ev
    assert ev["rec_coverage"] >= min_rec, ev
    assert ev["untagged_labelled"] == 0 and ev["ghost_labelled"] == 0
    assert_invariants(labels, SCENARIOS[name][1])


def test_two_symmetric_bees_are_left_unlabelled():
    scene, labels = run("two_bees_symmetric", 0)
    for bee in (2, 3):
        slot = bee_slot(scene, bee)
        assert (labels.slot_tag[np.arange(300), slot] == -1).all()
        flags = labels.slot_flag[np.arange(300), slot]
        assert (flags == SlotFlag.LOW_CONFIDENCE).all()
    assert (labels.guess_conf[labels.valid] <= 1.0).all()


def test_distinguishable_bees_are_recovered_from_their_tracklets():
    scene, labels = run("two_bees_distinguishable", 0)
    ev = evaluate(scene, labels)
    assert ev["rec_precision"] == 1.0
    sources = labels.slot_source[(labels.slot_tag >= 0) & (labels.slot_source != Source.DECODE)]
    assert (sources == Source.TRACKLET).sum() > 0.5 * len(sources)


def test_ghost_instances_never_receive_a_tag():
    scene, labels = run("ghosts", 0)
    ghosts = (scene.truth < 0) & scene.table.valid
    assert ghosts.sum() > 20
    assert (labels.slot_tag[ghosts] == -1).all()
    assert (labels.slot_flag[ghosts] == SlotFlag.GHOST).sum() >= 0.9 * ghosts.sum()
    assert labels.info["n_ghost_instances"] > 20


def test_spurious_far_instances_beyond_the_colony_size_are_culled():
    scene = scene_of(40, n_frames=200, n_slots=5, absent={0: (0, 199)})
    rng = np.random.default_rng(7)
    for f in range(50, 100):
        free = int(np.flatnonzero(~np.isfinite(scene.table.xy[f, :, 0]))[0])
        scene.table.xy[f, free] = rng.uniform(0, 1800, 2)
        scene.table.tag_xy[f, free] = scene.table.xy[f, free]
        scene.table.score[f, free] = 0.1
    labels = solve_region(scene.table, scene.votes, 4, 4)
    spurious = (scene.truth < 0) & scene.table.valid
    assert spurious.sum() == 50
    assert (labels.slot_tag[spurious] == -1).all()
    info = info_of(labels)
    assert info["n_excess_instances"] + info["n_ghost_instances"] >= 40
    assert (labels.slot_flag[spurious] > 0).all()
    ev = evaluate(scene, labels)
    assert ev["wrong"] == 0 and ev["coverage"] > 0.99
    assert_invariants(labels, 4)


def test_untagged_bee_is_never_labelled_and_is_flagged():
    scene, labels = run("three_tags", 0)
    slot = bee_slot(scene, 3)
    flags = labels.slot_flag[np.arange(300), slot]
    assert (labels.slot_tag[np.arange(300), slot] == -1).all()
    assert (flags == SlotFlag.UNTAGGED_BEE).sum() >= 0.95 * 300


# ------------------------------------------------------------- the h047 failure mode
def h047_scene(seed, n_strays, wrong_tag=1):
    """A bee undecoded for 2000+ frames while a few decodes of ANOTHER bee's tag land on it.

    The other bee (``wrong_tag``) is lost by SLEAP for 40 frames (longer than a tracklet gap), so
    its tag is free while the strays appear: this is how a stray anchor can look plausible.
    """
    n_frames = 2400
    scene = scene_of(seed, n_frames=n_frames, absent={3: (100, n_frames - 1)}, decode_rate=0.5)
    slot_lost, slot_target = bee_slot(scene, wrong_tag), bee_slot(scene, 3)
    lost_frames = np.arange(1000, 1040)
    for f in lost_frames:  # SLEAP misses the other bee
        scene.table.xy[f, slot_lost[f]] = np.nan
        scene.table.tag_xy[f, slot_lost[f]] = np.nan
    stray_frames = list(range(1005, 1005 + 6 * n_strays, 6))[:n_strays]
    rows = [r for r in vote_rows(scene) if not (r[0] in lost_frames and r[2] == wrong_tag)]
    rows += [(f, int(slot_target[f]), wrong_tag, 1.0) for f in stray_frames]
    out = with_votes(scene, rows)
    out.extras["strays"] = stray_frames
    return out


# observed over seeds 0-3 with 3 and 6 strays: the only wrong labels are the stray decodes
# themselves (immutable anchors; the model's confidence in them is 1e-4 .. <0.99);
# 95.7-100 % of the bee's frames are labelled, all of the others correctly.
@pytest.mark.parametrize("n_strays", [3, 6])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stray_anchors_do_not_propagate_over_thousands_of_frames(seed, n_strays):
    scene = h047_scene(seed, n_strays)
    labels = solve_region(scene.table, scene.votes, 4, 4)
    slot = bee_slot(scene, 3)
    strays = scene.extras["strays"]
    frames = np.arange(100, 2400)
    got = labels.slot_tag[frames, slot[frames]]
    labelled = got >= 0
    wrong = labelled & (got != 3)
    # the strays are decoded labels (kept) that the model does not endorse ...
    assert set(frames[wrong]) == set(strays)
    assert (labels.slot_source[strays, slot[strays]] == Source.DECODE).all()
    assert (labels.slot_conf[strays, slot[strays]] < 0.99).all()
    # ... and nothing else is wrong: the wrong tag does not propagate over the stretch
    ev = evaluate(scene, labels)
    assert ev["wrong_inferred"] == 0 and ev["rec_precision"] >= 0.99
    assert labelled.mean() > 0.9
    assert_invariants(labels, 4)


def test_scene_helpers_roundtrip():
    scene = make_scene(3, n_frames=30)
    rows = vote_rows(scene)
    again = with_votes(scene, rows)
    assert np.array_equal(again.votes.frame, scene.votes.frame)
    assert isinstance(again, Scene)
    assert len(with_votes(scene, []).votes) == 0
