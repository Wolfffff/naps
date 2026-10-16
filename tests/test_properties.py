"""Randomised invariant tests (fixed seeds) for the cost matrix and closed-set elimination."""

import copy
import random

import numpy as np
import pytest

from naps.closed_set import eliminate_identities
from naps.cost_matrix import CostMatrix, iter_votes

SEEDS = range(12)


def random_votes(rng, n_frames):
    """Random ``{frame: {track: [vote entry, ...]}}`` with None, weights, repeats and gaps."""
    n_tracks = rng.randint(1, 5)
    tag_pool = [rng.randint(0, 30) for _ in range(rng.randint(1, 6))] + [10**6]
    value_dict = {}
    for frame in range(n_frames):
        row = {}
        for track in range(n_tracks):
            if rng.random() < 0.1:
                continue  # track absent from this frame
            entries = []
            for _ in range(rng.choice([0, 0, 1, 1, 2, 3])):
                kind = rng.random()
                if kind < 0.2:
                    entries.append(None)
                elif kind < 0.5:
                    entries.append((rng.choice(tag_pool), rng.choice([0.25, 0.5, 1.0])))
                else:
                    entries.append(rng.choice(tag_pool))
            row[track] = entries
        value_dict[frame] = row
    return value_dict


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_cost_matrix_invariants(seed, strict):
    rng = random.Random(seed)  # noqa: S311 (reproducible test data, not security)
    n_frames = rng.randint(1, 60)
    half = rng.randint(0, 4)
    value_dict = random_votes(rng, n_frames)
    voted = {}  # track -> every tag it ever voted for
    for row in value_dict.values():
        for track, entries in row.items():
            voted.setdefault(track, set()).update(tag for tag, _ in iter_votes(entries))
    all_voted = set().union(*voted.values()) if voted else set()

    def run():
        return CostMatrix.fromDict(
            value_dict, 0, n_frames - 1, half, strict_carry_forward=strict
        ).assignTrackTagPairs()

    result = run()

    for frame, row in result.items():
        assert 0 <= frame < n_frames
        tags = list(row.values())
        assert len(tags) == len(set(tags)), "one tag per track per frame"
        for track, tag in row.items():
            assert type(tag) is int
            assert track in value_dict[frame], "never invents a track the frame did not have"
            assert track in voted, "a track that never received a vote is never assigned"
            assert tag in all_voted, "never invents a tag nobody voted for"
    for frame in range(min(half, n_frames)):
        assert not result.get(frame), "the head frames never get an identity"
    # deterministic
    assert {f: dict(r) for f, r in run().items()} == {f: dict(r) for f, r in result.items()}


@pytest.mark.parametrize("seed", SEEDS)
def test_cost_matrix_unvoted_tracks_stay_unassigned(seed):
    rng = random.Random(1000 + seed)  # noqa: S311 (reproducible test data, not security)
    n_frames = rng.randint(10, 40)
    value_dict = random_votes(rng, n_frames)
    for row in value_dict.values():
        row["silent"] = rng.choice([[], [None], [None, None]])

    result = CostMatrix.fromDict(value_dict, 0, n_frames - 1, 2).assignTrackTagPairs()

    assert all("silent" not in row for row in result.values())


def random_closed_world(rng, n_frames=40):
    closed = rng.sample(range(100, 200), rng.randint(2, 5))
    n = len(closed)
    positions, matched = {}, {}
    for frame in range(n_frames):
        positions[frame], matched[frame] = {}, {}
        for k in range(n + rng.choice([0, 0, 0, 1])):  # sometimes an extra instance
            position = np.array([50.0 + frame + rng.uniform(-1, 1), 100.0 * k])
            if rng.random() < 0.05:
                position = np.array([np.nan, np.nan])
            positions[frame][k] = position
            roll = rng.random()
            if k >= n or roll < 0.15:
                continue  # unassigned
            if roll < 0.2:
                matched[frame][k] = 999  # foreign tag
            elif roll < 0.25:
                matched[frame][k] = rng.choice(closed)  # possible duplicate
            else:
                matched[frame][k] = closed[k]
    untracked = {f: 1 for f in range(n_frames) if rng.random() < 0.1}
    return closed, matched, positions, untracked


@pytest.mark.parametrize("seed", SEEDS)
def test_elimination_invariants(seed):
    rng = random.Random(2000 + seed)  # noqa: S311 (reproducible test data, not security)
    closed, matched, positions, untracked = random_closed_world(rng)
    before = copy.deepcopy((matched, untracked))
    max_jump = 120.0

    res = eliminate_identities(
        matched, positions, closed, max_jump=max_jump, untracked_counts=untracked
    )

    assert (matched, untracked) == before, "inputs are not modified"
    assigned = {(a[0], a[1]): a for a in res.assigned}
    for frame, row in res.matched.items():
        tags = list(row.values())
        assert len(tags) == len(set(tags)), "never reuses a tag within a frame"
        assert set(tags) <= set(closed), "only closed-set tags survive"
        assert set(row) <= set(positions.get(frame, {})) | set(matched.get(frame, {}))
        for track, tag in row.items():
            if (frame, track) not in assigned:
                assert matched[frame][track] == tag, "only elimination changes an assignment"

    for frame, track, tag, gap, dist in res.assigned:
        in_frame = {t: g for t, g in matched.get(frame, {}).items() if g in closed}
        kept = {}
        for t, g in in_frame.items():
            kept.setdefault(g, t)  # duplicates keep the first track only
        leftover = [t for t in positions[frame] if t not in kept.values()]
        assert frame not in res.abstained_frames
        assert untracked.get(frame, 0) == 0, "abstains where an unseen instance could be the bee"
        assert leftover == [track], "exactly one unassigned instance"
        assert set(closed) - set(kept) == {tag}, "exactly one missing tag"
        assert np.isfinite(positions[frame][track]).all()
        assert dist <= max_jump * max(gap, 1) + 1e-9


@pytest.mark.parametrize("seed", SEEDS)
def test_elimination_is_deterministic(seed):
    rng = random.Random(3000 + seed)  # noqa: S311 (reproducible test data, not security)
    closed, matched, positions, untracked = random_closed_world(rng)

    first = eliminate_identities(matched, positions, closed, untracked_counts=untracked)
    second = eliminate_identities(matched, positions, closed, untracked_counts=untracked)

    assert first.assigned == second.assigned
    assert first.abstained == second.abstained
    assert {f: dict(r) for f, r in first.matched.items()} == {
        f: dict(r) for f, r in second.matched.items()
    }
