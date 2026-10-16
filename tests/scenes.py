"""Seeded synthetic scenes for the joint identity engine (tests, stress tests, benchmarks).

A :class:`Scene` simulates ``n_bees`` bees walking in a box. Every frame each bee is detected as an
instance in a random slot (so slot order carries no information), tags are decoded with a
per-bee probability that can drop to zero for stretches (a bee "absent from NAPS"), and the
failure modes the engine must survive can be switched on: near-passes, stray decodes of other tags,
ghost instances, fragmentation (missed detections) and swapped neighbour tags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from naps.identity import Votes
from naps.instances import InstanceTable
from naps.sources import Source


@dataclass
class Scene:
    """A simulated region.

    Attributes:
        table: Instances of every frame.
        votes: Decodes (tag index of the true tags).
        truth: ``(F, S)`` true tag index of every slot, -1 for ghosts / empty slots.
        positions: ``(F, K, 2)`` true bee positions.
        decode_on: ``(F, K)`` frames in which the bee was decodable.
    """

    table: InstanceTable
    votes: Votes
    truth: NDArray[np.int64]
    positions: NDArray[np.float64]
    decode_on: NDArray[np.bool_]
    n_bees: int
    n_real: int
    extras: dict[str, object] = field(default_factory=dict)


def separate(pos: NDArray[np.float64], min_sep: float) -> None:
    """Push bees that are closer than ``min_sep`` apart along their connecting line, in place."""
    for _ in range(4):
        moved = False
        for a in range(len(pos)):
            for b in range(a + 1, len(pos)):
                delta = pos[b] - pos[a]
                dist = float(np.linalg.norm(delta))
                if dist < min_sep:
                    direction = delta / dist if dist > 1e-9 else np.array([1.0, 0.0])
                    shift = 0.5 * (min_sep - dist) * direction
                    pos[a] -= shift
                    pos[b] += shift
                    moved = True
        if not moved:
            break


def random_walks(  # noqa: PLR0917 (a small simulation helper)
    rng: np.random.Generator,
    n_frames: int,
    n_bees: int,
    box: float = 1800.0,
    speed: float = 3.0,
    min_sep: float = 0.0,
) -> NDArray[np.float64]:
    """Smooth random walks (damped velocity noise) reflected at the box walls.

    With ``min_sep > 0`` bees are kept at least that far apart (real bees do not overlap: the
    distance between two tags is below 50 px in 0.01 % of frames).
    """
    pos = rng.uniform(0.15 * box, 0.85 * box, size=(n_bees, 2))
    vel = rng.normal(0, speed, size=(n_bees, 2))
    out = np.empty((n_frames, n_bees, 2))
    for f in range(n_frames):
        vel = 0.92 * vel + rng.normal(0, 0.6 * speed, size=vel.shape)
        pos = pos + vel
        low, high = pos < 20, pos > box - 20
        vel = np.where(low | high, -vel, vel)
        pos = np.clip(pos, 20, box - 20)
        if min_sep > 0:
            separate(pos, min_sep)
        out[f] = pos
    return out


def add_near_pass(  # noqa: PLR0917 (small helper called with explicit values)
    positions: NDArray[np.float64], a: int, b: int, frame: int, half: int = 30, gap: float = 12.0
) -> None:
    """Make bees ``a`` and ``b`` meet at ``frame`` (within ``gap`` px) and part again, in place."""
    n_frames = len(positions)
    lo, hi = max(frame - half, 0), min(frame + half, n_frames - 1)
    centre = 0.5 * (positions[frame, a] + positions[frame, b])
    direction = positions[hi, a] - positions[lo, a]
    direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
    for f in range(lo, hi + 1):
        w = 1.0 - abs(f - frame) / half
        sign = np.sign(f - frame) or 1.0
        meet_a = centre + sign * direction * gap * 0.5
        meet_b = centre - sign * direction * gap * 0.5
        positions[f, a] = (1 - w) * positions[f, a] + w * meet_a
        positions[f, b] = (1 - w) * positions[f, b] + w * meet_b


def make_scene(  # noqa: PLR0917 (a simulator with many independent switches)
    seed: int,
    n_frames: int = 400,
    n_bees: int = 4,
    n_real: int | None = None,
    decode_rate: float = 0.6,
    absent: dict[int, tuple[int, int]] | None = None,
    near_passes: tuple[tuple[int, ...], ...] = (),
    stray_rate: float = 0.0,
    ghost_rate: float = 0.0,
    miss_rate: float = 0.0,
    noise: float = 1.5,
    box: float = 1800.0,
    n_slots: int | None = None,
    min_separation: float = 0.0,
) -> Scene:
    """Simulate a region.

    Args:
        seed: Seed of the generator (the scene is a pure function of its arguments).
        n_frames: Number of frames.
        n_bees: Number of bees.
        n_real: Number of tagged bees (the first ``n_real``); the others have no tag.
        decode_rate: Probability that a tagged bee's tag is decoded in a frame.
        absent: ``{bee: (first_row, last_row)}`` frames in which a bee is never decoded.
        near_passes: ``(bee_a, bee_b, row)`` or ``(bee_a, bee_b, row, gap_px)`` close encounters.
        stray_rate: Probability per frame that an instance also gets a decode of a wrong tag.
        ghost_rate: Probability per frame that a ghost copy of a random bee is added.
        miss_rate: Probability per frame and bee that SLEAP misses the bee.
        noise: Standard deviation of the position noise (px).
        box: Side of the arena.
        n_slots: Slots per frame (default ``n_bees + 1`` when ghosts are on).
        min_separation: Minimum distance (px) between bees in the random walks (0 = none).
    """
    rng = np.random.default_rng(seed)
    n_real = n_bees if n_real is None else n_real
    positions = random_walks(rng, n_frames, n_bees, box, min_sep=min_separation)
    for a, b, row, *gap in near_passes:
        add_near_pass(positions, a, b, row, gap=float(gap[0]) if gap else 12.0)
    n_slots = n_slots or (n_bees + 1 if ghost_rate > 0 else n_bees)
    xy = np.full((n_frames, n_slots, 2), np.nan)
    tag_xy = np.full((n_frames, n_slots, 2), np.nan)
    score = np.full((n_frames, n_slots), np.nan)
    truth = np.full((n_frames, n_slots), -1, dtype=np.int64)
    decode_on = rng.random((n_frames, n_bees)) < decode_rate
    decode_on[:, n_real:] = False
    for bee, (lo, hi) in (absent or {}).items():
        decode_on[lo : hi + 1, bee] = False
    rows, slots, tags, weights = [], [], [], []
    for f in range(n_frames):
        detected = [b for b in range(n_bees) if rng.random() >= miss_rate]
        occupied = list(rng.permutation(n_slots)[: len(detected)])
        for bee, slot in zip(detected, occupied, strict=True):
            xy[f, slot] = positions[f, bee] + rng.normal(0, noise, 2)
            tag_xy[f, slot] = xy[f, slot]
            score[f, slot] = rng.uniform(0.7, 0.99)
            truth[f, slot] = bee
            if decode_on[f, bee]:
                rows.append(f), slots.append(slot), tags.append(bee), weights.append(1.0)
            if stray_rate > 0 and n_real > 1 and rng.random() < stray_rate:
                wrong = int(rng.choice([t for t in range(n_real) if t != bee]))
                rows.append(f), slots.append(slot), tags.append(wrong), weights.append(1.0)
        free = [s for s in range(n_slots) if s not in occupied]
        if free and detected and rng.random() < ghost_rate:
            bee = int(rng.choice(detected))
            slot = free[0]
            offset = rng.normal(0, 1, 2)
            offset = 30.0 * offset / max(float(np.linalg.norm(offset)), 1e-9)
            xy[f, slot] = positions[f, bee] + offset
            tag_xy[f, slot] = xy[f, slot]
            score[f, slot] = rng.uniform(0.2, 0.5)
    table = InstanceTable(
        first_frame=0,
        xy=xy,
        tag_xy=tag_xy,
        score=score,
        hint=np.full((n_frames, n_slots), -1, dtype=np.int64),
        inst_index=np.where(np.isfinite(xy[..., 0]), np.arange(n_slots)[None, :], -1),
        heading=np.full((n_frames, n_slots), np.nan),
    )
    votes = Votes(
        np.array(rows, dtype=np.int64),
        np.array(slots, dtype=np.int64),
        np.array(tags, dtype=np.int64),
        np.array(weights, dtype=np.float64),
    )
    return Scene(table, votes, truth, positions, decode_on, n_bees, n_real)


def info_of(labels, key: str = "") -> Any:
    """``labels.info`` (or its ``consistency`` block with ``key="consistency"``) as a plain dict."""
    return cast("Any", labels.info[key] if key else labels.info)


def anchors_of(labels) -> NDArray[np.int16]:
    """``labels.anchors``, which :func:`naps.identity.solve_region` always sets."""
    assert labels.anchors is not None
    return labels.anchors


def valid_of(labels) -> NDArray[np.bool_]:
    """``labels.valid``, which :func:`naps.identity.solve_region` always sets."""
    assert labels.valid is not None
    return labels.valid


def with_votes(scene: Scene, rows: list[tuple[int, int, int, float]]) -> Scene:
    """A copy of ``scene`` whose decodes are the ``(frame, slot, tag, weight)`` rows given."""
    arr = np.array(sorted(rows), dtype=float).reshape(-1, 4)
    votes = Votes(
        arr[:, 0].astype(np.int64),
        arr[:, 1].astype(np.int64),
        arr[:, 2].astype(np.int64),
        arr[:, 3],
    )
    return Scene(
        scene.table, votes, scene.truth, scene.positions, scene.decode_on, scene.n_bees,
        scene.n_real, dict(scene.extras),
    )  # fmt: skip


def vote_rows(scene: Scene) -> list[tuple[int, int, int, float]]:
    """The decodes of ``scene`` as ``(frame, slot, tag, weight)`` rows."""
    v = scene.votes
    return list(
        zip(v.frame.tolist(), v.slot.tolist(), v.tag.tolist(), v.weight.tolist(), strict=True)
    )


def bee_slot(scene: Scene, bee: int) -> NDArray[np.int64]:
    """``(F,)`` slot that holds ``bee`` in every frame (-1 if SLEAP missed it)."""
    hit = scene.truth == bee
    return np.where(hit.any(axis=1), hit.argmax(axis=1), -1)


def evaluate(scene: Scene, labels) -> dict[str, float]:
    """Score the labels of ``scene`` against the truth.

    ``rec_*`` count the bee-frames of tagged bees that were *not* decoded (what the engine has to
    recover); ``wrong`` counts labelled instances with the wrong tag; ``ghost_labelled`` counts
    labelled instances that belong to no bee.
    """
    truth, lab = scene.truth, labels.slot_tag
    real = (truth >= 0) & (truth < scene.n_real)
    frames, slots = np.nonzero(real)
    bee = truth[frames, slots]
    emitted = lab[frames, slots] >= 0
    correct = emitted & (lab[frames, slots] == bee)
    rec = ~scene.decode_on[frames, bee]
    rec_emitted, rec_correct = emitted & rec, correct & rec
    n_rec = int(rec.sum())
    inferred = (lab >= 0) & (labels.slot_source != Source.DECODE)
    return {
        "n": float(len(frames)),
        "coverage": float(emitted.mean()) if len(frames) else 1.0,
        "precision": float(correct.sum() / max(emitted.sum(), 1)),
        "wrong": float((emitted & ~correct).sum()),
        # wrong labels the engine inferred itself (a wrong *decode* is an immutable anchor)
        "wrong_inferred": float(
            (emitted & ~correct & (labels.slot_source[frames, slots] != Source.DECODE)).sum()
        ),
        "n_rec": float(n_rec),
        "rec_coverage": float(rec_emitted.sum() / max(n_rec, 1)),
        "rec_precision": float(rec_correct.sum() / max(rec_emitted.sum(), 1)),
        "rec_wrong": float((rec_emitted & ~correct).sum()),
        # inferred labels on a ghost / on an untagged bee (a stray *decode* on one is an anchor)
        "ghost_labelled": float((inferred & ~real & (truth < 0)).sum()),
        "untagged_labelled": float((inferred & (truth >= scene.n_real)).sum()),
    }


def assert_invariants(labels, n_real: int, threshold: float = 0.99) -> None:
    """Structural invariants every :class:`naps.identity.RegionLabels` must satisfy."""
    lab = labels.slot_tag
    labelled = lab >= 0
    assert labels.valid is not None and labels.anchors is not None
    assert not (labelled & ~labels.valid).any(), "label on an instance that took no part"
    for k in range(n_real):  # one tag per slot is implicit; no tag twice in a frame
        assert ((lab == k).sum(axis=1) <= 1).all(), f"tag {k} twice in a frame"
    assert (lab[labelled] < n_real).all()
    # per-tag arrays agree with per-slot arrays
    for f, k in zip(*np.nonzero(labels.tag_slot >= 0), strict=True):
        assert lab[f, labels.tag_slot[f, k]] == k
    for f, s in zip(*np.nonzero(labelled), strict=True):
        assert labels.tag_slot[f, lab[f, s]] == s
    # source / flag / confidence bookkeeping
    src = labels.slot_source
    assert set(np.unique(src)) <= {int(s) for s in Source}
    assert (src[labelled] > 0).all() and (src[~labelled] == 0).all()
    assert (labels.slot_flag[labelled] == 0).all()
    assert labels.slot_flag.max() <= 8
    conf = labels.slot_conf
    assert np.isfinite(conf[labelled]).all() and np.isnan(conf[~labelled]).all()
    assert ((conf[labelled] >= 0) & (conf[labelled] <= 1 + 1e-6)).all()
    recovered = labelled & (src != Source.DECODE)
    assert (conf[recovered] >= threshold - 1e-6).all(), "recovered label below the threshold"
    # decoded anchors are immutable
    anchored = labels.anchors >= 0
    assert (lab[anchored] == labels.anchors[anchored]).all()
    assert (src[anchored] == Source.DECODE).all()
    assert not (src[~anchored & labelled] == Source.DECODE).any()
    # tag arrays
    present = np.isfinite(labels.tag_xy[..., 0])
    assert ((labels.tag_src > 0) == present).all()
    assert set(np.unique(labels.tag_src)) <= {int(s) for s in Source}
