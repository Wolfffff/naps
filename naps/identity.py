"""Joint identity inference: which tag does every instance carry?

The legacy engine assigns, frame by frame, the tag that received the most ArUco decodes inside a
rolling window on each SLEAP track. The joint engine instead solves the whole video of a *region*
(a colony / quadrant with a known set of N tags) as one exact probabilistic inference problem:

* **Nodes** are tracklets (:mod:`naps.tracklets`) turned into *entities*; a gap of a few frames
  inside a tracklet is bridged by a virtual entity at the interpolated position.
* **State** of a frame is an *injective* map from the entities present to the N tags. The closed set
  is therefore hard: two entities never carry the same tag, and when N-1 tags are taken the last
  entity must carry the remaining one (*elimination*). There are N!/(N-n)! states; for N = 4 the
  tables are tiny, up to N = 6 (720 states) they are built lazily.
* **Emissions**: every decode is evidence that the entity carries the tag (a log-likelihood ratio of
  ``anchor_llr`` per decode, scaled by the decode weight); optional appearance log-probabilities
  and external votes (:mod:`naps.evidence`) add to it. Decodes that are contradicted - the same tag
  on two instances of a frame, or a tag that is a tiny minority on a tracklet (a stray anchor) - are
  *demoted* to soft evidence (weight ``demoted_weight``) and are no longer anchors.
* **Transitions**: a continuing entity keeps its tag; a label change costs ``sqrt(eps)`` per entity
  (a swap of two costs ``eps``), where ``eps`` grows from 1e-9 to 3e-2 as another bee comes within
  40..150 px. A *new* entity may take any tag no continuing entity holds, with a bonus for taking
  the tag of an entity that ended nearby in the previous frame (hand-off at tracklet joins).
* **Inference** is an exact forward-backward pass; the marginal of (entity, tag) is the confidence.

Labels are emitted where the best assignment of the frame has marginal >= ``threshold``
(default 0.99, the operating point at which the band was reliably calibrated). Clean, unique
decodes are *anchors* and are immutable: the model may *dispute* an anchor (``disputed``), never
change it. A decode that is not an anchor is **not** protected: an instance whose decode was
demoted (conflict or stray, see above) is treated like an instance without a decode and may be
labelled with a different tag by the model. Those demotions are counted
(``n_conflict_decodes_demoted``, ``n_stray_anchors_demoted``, ``n_demoted_decodes_relabelled``,
``n_demoted_decodes_confirmed`` in ``RegionLabels.info``; ``RegionLabels.demoted`` marks the
instances). The first and last frames are labelled like all others (there is no half-window dead
zone).

Everything is deterministic and permutation-equivariant in the tag labels.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from naps.consistency import (
    ConsistencyParams,
    apply_consistency,
    ghost_mask,
    interpolate,
)
from naps.instances import InstanceTable
from naps.reasons import Reason
from naps.sources import Source
from naps.tracklets import (
    TrackletMode,
    TrackletParams,
    Tracklets,
    build_tracklets,
    dedupe_instances,
)

__all__ = [
    "MAX_TAGS",
    "Entities",
    "IdentityParams",
    "RegionLabels",
    "SlotFlag",
    "TransitionModel",
    "Votes",
    "build_entities",
    "candidate_anchors",
    "demote_strays",
    "forward_backward",
    "permutation_table",
    "single_decodes",
    "solve_region",
    "votes_from_frame_votes",
]

logger = logging.getLogger(__name__)

#: Largest number of bees (tags + untagged bees) of one region the engine supports.
MAX_TAGS = 6

_BIG_DISTANCE = 1e4  # nearest-neighbour distance of an entity with no neighbour (px)
_MAX_CACHE_BYTES = 400_000_000  # forward transition matrices kept in memory up to this size
_MIN_HANDOFF = 1e-3  # hand-off bonus below this is ignored


class SlotFlag:
    """Why an instance has no label (``RegionLabels.slot_flag``). 0 = labelled or no instance."""

    NONE = 0
    GHOST = 1
    DUPLICATE_INSTANCE = 2
    EXCESS_INSTANCE = 3
    LOW_CONFIDENCE = 4
    UNTAGGED_BEE = 5
    BLOCKED_BY_ANCHOR = 6
    CONSISTENCY_DROPPED = 7
    WITHHELD = 8
    NAMES = (
        "none",
        "ghost",
        "duplicate_instance",
        "excess_instance",
        "low_confidence",
        "untagged_bee",
        "blocked_by_anchor",
        "consistency_dropped",
        "withheld",
    )


@dataclass(frozen=True)
class IdentityParams:
    """Parameters of the joint model (defaults: the validated research values).

    Attributes:
        threshold: A label is emitted only if its marginal is at least this (default 0.99). On
            38 held-out windows the marginal was well calibrated (ECE 0.0028), so lowering it
            trades precision for coverage; see ``docs/identity_pipeline.md``.
        anchor_llr: Log-likelihood ratio contributed by one decode of weight 1.
        demoted_weight: Weight multiplier of decodes demoted to soft evidence.
        minority_fraction: A tag decoded on a tracklet at most this fraction as often as the
            majority tag is a stray (decoder noise) and is demoted.
        min_majority: ... provided the majority tag has at least this many decodes.
        min_anchor_weight: A decode needs at least this weight to be an anchor.
        min_tracklet_anchors: A recovered label counts as ``TRACKLET`` evidence (not elimination)
            if its tracklet carries at least this many anchors of the tag.
        eps_far: Probability scale of a label change when no other bee is near.
        eps_near: ... when another bee is within ``nn_near_px``.
        nn_near_px: Nearest-neighbour distance at and below which ``eps_near`` applies.
        nn_far_px: ... at and above which ``eps_far`` applies (log-linear in between).
        handoff_gain: Bonus factor for a new entity to take the tag of an entity that just ended.
        handoff_sigma_px: Distance scale of the hand-off bonus.
        appearance_weight: Weight of appearance log-probabilities (a tie-breaker by default).
        appearance_floor: Log-probabilities are clipped below this.
        appearance_gate: ``((nn_px, factor), ...)``: factor applied while the nearest neighbour is
            closer than ``nn_px`` (appearance is least reliable in piles).
        dedupe_px: Instances closer than this are duplicates (see
            :func:`naps.tracklets.dedupe_instances`).
    """

    threshold: float = 0.99
    anchor_llr: float = 2.0
    demoted_weight: float = 0.25
    minority_fraction: float = 0.03
    min_majority: int = 3
    min_anchor_weight: float = 0.5
    min_tracklet_anchors: int = 3
    eps_far: float = 1e-9
    eps_near: float = 3e-2
    nn_near_px: float = 40.0
    nn_far_px: float = 150.0
    handoff_gain: float = 20.0
    handoff_sigma_px: float = 40.0
    appearance_weight: float = 0.001
    appearance_floor: float = -4.0
    appearance_gate: tuple[tuple[float, float], ...] = ((60.0, 0.3), (100.0, 0.6), (1e9, 1.0))
    dedupe_px: float = 25.0


@dataclass
class Votes:
    """Sparse identity votes: evidence that slot ``slot`` of frame row ``frame`` carries ``tag``.

    Attributes:
        frame: Frame row (0-based within the region table).
        slot: Slot of the region table.
        tag: Tag *index* (0..N-1) in the region's sorted tag list.
        weight: Vote weight (a decode of weight 1 is one ArUco read).
    """

    frame: NDArray[np.int64]
    slot: NDArray[np.int64]
    tag: NDArray[np.int64]
    weight: NDArray[np.float64]

    @classmethod
    def empty(cls) -> Votes:
        """Votes without any entry."""
        return cls(
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )

    def __len__(self) -> int:
        """Number of votes."""
        return len(self.frame)


def votes_from_frame_votes(
    frame_votes: Mapping[int, Mapping[int, Sequence[tuple[int, float]]]],
    first_frame: int,
    n_frames: int,
    tag_index: Mapping[int, int],
) -> Votes:
    """Turn ``frame_votes[frame][slot] -> [(tag, weight)]`` into :class:`Votes`.

    Frames outside ``first_frame .. first_frame + n_frames - 1`` and tags that are not in
    ``tag_index`` (``{tag id: index}``) are skipped.

    Args:
        frame_votes: Votes keyed by video frame number and region-table slot.
        first_frame: Video frame number of row 0.
        n_frames: Number of rows.
        tag_index: Mapping from tag id to tag index.

    Returns:
        The votes, sorted by frame row and slot.
    """
    rows: list[tuple[int, int, int, float]] = []
    for frame, per_slot in frame_votes.items():
        row = frame - first_frame
        if not 0 <= row < n_frames:
            continue
        for slot, entries in per_slot.items():
            rows.extend(
                (row, int(slot), tag_index[tag], float(weight))
                for tag, weight in entries
                if tag in tag_index
            )
    if not rows:
        return Votes.empty()
    rows.sort()
    arr = np.array(rows, dtype=float)
    return Votes(
        arr[:, 0].astype(np.int64),
        arr[:, 1].astype(np.int64),
        arr[:, 2].astype(np.int64),
        arr[:, 3],
    )


# --------------------------------------------------------------------------- anchors
def _vote_weights(votes: Votes, shape: tuple[int, int], n_tags: int) -> NDArray[np.float64]:
    """Dense ``(F, S, N)`` sum of the vote weights."""
    dense = np.zeros((*shape, n_tags))
    if len(votes):
        np.add.at(dense, (votes.frame, votes.slot, votes.tag), votes.weight)
    return dense


def single_decodes(
    votes: Votes, shape: tuple[int, int], n_tags: int, min_weight: float = 0.5
) -> NDArray[np.int16]:
    """The unambiguous decode of every instance, before the same-frame conflict test.

    Args:
        votes: The decode votes.
        shape: ``(F, S)`` of the region table.
        n_tags: Number of tags.
        min_weight: Minimum total weight of the tag.

    Returns:
        ``(F, S)`` tag index of the one tag decoded on the instance (with weight >=
        ``min_weight``), else -1 (nothing decoded, a weak decode, or two different tags).
    """
    dense = _vote_weights(votes, shape, n_tags)
    exactly_one = (dense > 0).sum(axis=2) == 1
    strong = dense.max(axis=2) >= min_weight
    return np.where(exactly_one & strong, dense.argmax(axis=2), -1).astype(np.int16)


def candidate_anchors(
    votes: Votes, shape: tuple[int, int], n_tags: int, min_weight: float = 0.5
) -> NDArray[np.int16]:
    """Decodes that qualify as anchors before any cross-checking.

    An instance carries an anchor tag if exactly one distinct tag was decoded on it with total
    weight >= ``min_weight``; an instance with two different decoded tags (a crop that shows a
    neighbour's tag) is ambiguous and has no anchor. A tag decoded on two instances of the same
    frame is a conflict and is anchored on neither.

    Args:
        votes: The decode votes (only decodes: soft sources must not anchor).
        shape: ``(F, S)`` of the region table.
        n_tags: Number of tags.
        min_weight: Minimum total weight of the anchor tag.

    Returns:
        ``(F, S)`` tag index or -1.
    """
    anchor = single_decodes(votes, shape, n_tags, min_weight)
    for tag in range(n_tags):
        holders = anchor == tag
        conflict = holders.sum(axis=1) >= 2
        anchor[holders & conflict[:, None]] = -1
    return anchor


def demote_strays(
    anchor: NDArray[np.int16],
    tracklets: Tracklets,
    n_tags: int,
    params: IdentityParams,
) -> NDArray[np.bool_]:
    """Anchors that are decoder noise on their tracklet.

    On every tracklet the anchors are counted per tag. If the majority tag has at least
    ``min_majority`` anchors, an anchor of another tag that makes up at most ``minority_fraction``
    of the majority is a stray (e.g. a crop that happened to show a neighbour's marker) and is
    demoted. A tracklet with two well-supported tags is *not* touched: that is a real identity
    conflict the model resolves through the transition costs.

    Args:
        anchor: ``(F, S)`` candidate anchors.
        tracklets: The tracklets the slots belong to.
        n_tags: Number of tags.
        params: Thresholds.

    Returns:
        ``(F, S)`` boolean mask of the anchors to demote.
    """
    frames, slots = np.nonzero((anchor >= 0) & (tracklets.tid >= 0))
    demote = np.zeros(anchor.shape, dtype=bool)
    if len(frames) == 0:
        return demote
    tids = tracklets.tid[frames, slots].astype(np.int64)
    tags = anchor[frames, slots].astype(np.int64)
    counts = np.zeros((tracklets.n_tracklets, n_tags))
    np.add.at(counts, (tids, tags), 1)
    majority = counts.max(axis=1)
    minority = (counts <= params.minority_fraction * majority[:, None]) & (
        majority[:, None] >= params.min_majority
    )
    own_majority = counts == majority[:, None]
    stray = minority & ~own_majority
    demote[frames, slots] = stray[tids, tags]
    return demote


# ------------------------------------------------------------------------- entities
@dataclass
class Entities:
    """Entities (tracklets, plus virtual bridged gaps) present in every frame.

    Attributes:
        tid: ``(F, N)`` tracklet id of the entity in position ``i`` of the frame (``-1`` = unused).
        slot: ``(F, N)`` table slot of the instance, ``-1`` for a virtual entity.
        pos: ``(F, N, 2)`` position (interpolated for virtual entities).
        count: ``(F,)`` number of entities in the frame.
        nn: ``(F, N)`` distance of every entity to its nearest other entity.
        of_slot: ``(F, S)`` position ``i`` of the entity holding the slot, ``-1`` if none.
        n_dropped_virtual: Virtual entities not created because the frame was full.
    """

    tid: NDArray[np.int64]
    slot: NDArray[np.int64]
    pos: NDArray[np.float64]
    count: NDArray[np.int64]
    nn: NDArray[np.float64]
    of_slot: NDArray[np.int64]
    n_dropped_virtual: int = 0


def _virtual_gaps(
    tracklets: Tracklets, xy: NDArray[np.float64]
) -> dict[int, list[tuple[int, NDArray[np.float64]]]]:
    """Interpolated positions of the frames inside a tracklet that has no instance there.

    The gaps are bounded by :attr:`TrackletParams.gap_max`: a tracklet is closed when it was not
    seen for more than that many frames.
    """
    gaps: dict[int, list[tuple[int, NDArray[np.float64]]]] = {}
    for tid in np.flatnonzero(tracklets.last - tracklets.first + 1 > tracklets.length):
        rows, slots = np.nonzero(tracklets.tid == tid)
        order = np.argsort(rows)
        rows, slots = rows[order], slots[order]
        points = xy[rows, slots]
        span = np.arange(tracklets.first[tid], tracklets.last[tid] + 1)
        missing = np.setdiff1d(span, rows)
        px = np.interp(missing, rows, points[:, 0])
        py = np.interp(missing, rows, points[:, 1])
        for frame, x, y in zip(missing, px, py, strict=True):
            gaps.setdefault(int(frame), []).append((int(tid), np.array([x, y])))
    return gaps


def build_entities(
    tracklets: Tracklets, xy: NDArray[np.float64], n_bees: int, with_virtual: bool = True
) -> Entities:
    """Collect the entities of every frame, sorted by tracklet id.

    Args:
        tracklets: The tracklets.
        xy: ``(F, S, 2)`` body positions.
        n_bees: Capacity per frame (tags + untagged bees). Real instances beyond the capacity must
            have been culled before; virtual entities are only added while there is room.
        with_virtual: Bridge gaps inside tracklets by virtual entities.

    Returns:
        The :class:`Entities`.
    """
    n_frames, n_slots = tracklets.tid.shape
    ent_tid = np.full((n_frames, n_bees), -1, dtype=np.int64)
    ent_slot = np.full((n_frames, n_bees), -1, dtype=np.int64)
    ent_pos = np.full((n_frames, n_bees, 2), np.nan)
    count = np.zeros(n_frames, dtype=np.int64)
    gaps = _virtual_gaps(tracklets, xy) if with_virtual else {}
    dropped = 0
    for frame in range(n_frames):
        members = [
            (int(tracklets.tid[frame, s]), int(s), xy[frame, s])
            for s in np.flatnonzero(tracklets.tid[frame] >= 0)
        ]
        for tid, point in gaps.get(frame, []):
            if len(members) < n_bees:
                members.append((tid, -1, point))
            else:
                dropped += 1
        members.sort(key=lambda m: m[0])
        count[frame] = len(members)
        for i, (tid, slot, point) in enumerate(members):
            ent_tid[frame, i] = tid
            ent_slot[frame, i] = slot
            ent_pos[frame, i] = point
    of_slot = np.full((n_frames, n_slots), -1, dtype=np.int64)
    rows, cols = np.nonzero(ent_slot >= 0)
    of_slot[rows, ent_slot[rows, cols]] = cols
    dist = np.linalg.norm(ent_pos[:, :, None, :] - ent_pos[:, None, :, :], axis=-1)
    dist[:, np.arange(n_bees), np.arange(n_bees)] = np.inf
    dist = np.where(np.isnan(dist), np.inf, dist)
    nn = np.minimum(dist.min(axis=2), _BIG_DISTANCE)
    return Entities(ent_tid, ent_slot, ent_pos, count, nn, of_slot, dropped)


def cull_excess(
    valid: NDArray[np.bool_], rank: NDArray[np.float64], capacity: int
) -> NDArray[np.bool_]:
    """Keep at most ``capacity`` instances per frame (highest ``rank`` first, ties: lower slot).

    Args:
        valid: ``(F, S)`` instances present.
        rank: ``(F, S)`` priority (anchored instances should rank above everything else).
        capacity: Number of bees the region holds.

    Returns:
        The reduced mask.
    """
    keep = valid.copy()
    for frame in np.flatnonzero(valid.sum(axis=1) > capacity):
        slots = np.flatnonzero(valid[frame])
        order = sorted(slots, key=lambda s: (-rank[frame, s], s))
        keep[frame, order[capacity:]] = False
    return keep


# --------------------------------------------------------------------- state spaces
@lru_cache(maxsize=64)
def permutation_table(n_tags: int, n_entities: int) -> NDArray[np.int8]:
    """All injective maps from ``n_entities`` entities to ``n_tags`` tags, in lexicographic order.

    Args:
        n_tags: Number of tags.
        n_entities: Number of entities (<= ``n_tags``).

    Returns:
        ``(P, n_entities)`` read-only array; row ``s`` gives the tag of entity ``i`` in state ``s``.
    """
    rows = list(itertools.permutations(range(n_tags), n_entities))
    table = np.array(rows, dtype=np.int8).reshape(len(rows), n_entities)
    table.setflags(write=False)
    return table


@lru_cache(maxsize=64)
def _one_hot(n_tags: int, n_entities: int) -> NDArray[np.float64]:
    """``(P, n, N)`` indicator ``[state s assigns tag k to entity i]``."""
    table = permutation_table(n_tags, n_entities)
    onehot = (table[:, :, None] == np.arange(n_tags)[None, None, :]).astype(np.float64)
    onehot.setflags(write=False)
    return onehot


@lru_cache(maxsize=4096)
def _differs(n_tags: int, n_from: int, n_to: int, i: int, j: int) -> NDArray[np.bool_]:
    """``(P_from, P_to)``: entity ``i`` of the earlier frame differs from ``j`` of the next."""
    a, b = permutation_table(n_tags, n_from), permutation_table(n_tags, n_to)
    mask = a[:, i][:, None] != b[:, j][None, :]
    mask.setflags(write=False)
    return mask


class TransitionModel:
    """Frame-to-frame transition matrices between the state spaces of consecutive frames.

    Args:
        entities: The entities of every frame.
        n_tags: Number of tags (= bees) of the region.
        params: Model parameters.
    """

    def __init__(self, entities: Entities, n_tags: int, params: IdentityParams) -> None:
        """Store the entities and parameters."""
        self.entities = entities
        self.n_tags = n_tags
        self.params = params

    def log_eps(self, nn: float) -> float:
        """Log of the label-change probability scale for a nearest-neighbour distance ``nn``."""
        p = self.params
        u = float(np.clip((p.nn_far_px - nn) / (p.nn_far_px - p.nn_near_px), 0.0, 1.0))
        return float(np.log(p.eps_far)) * (1 - u) + float(np.log(p.eps_near)) * u

    def matrix(self, frame: int) -> NDArray[np.float64]:
        """The (unnormalised) transition weights from ``frame`` to ``frame + 1``.

        Returns:
            ``(P_frame, P_next)`` weights.
        """
        e = self.entities
        n_a, n_b = int(e.count[frame]), int(e.count[frame + 1])
        k = self.n_tags
        tid_a, tid_b = list(e.tid[frame, :n_a]), list(e.tid[frame + 1, :n_b])
        index_a = {t: i for i, t in enumerate(tid_a)}
        log_w = np.zeros((len(permutation_table(k, n_a)), len(permutation_table(k, n_b))))
        for j, tid in enumerate(tid_b):
            i = index_a.get(tid)
            if i is not None:  # continuing entity: a label change is penalised
                nn = min(e.nn[frame, i], e.nn[frame + 1, j])
                log_w += 0.5 * self.log_eps(float(nn)) * _differs(k, n_a, n_b, i, j)
        ended = [i for i, t in enumerate(tid_a) if t not in tid_b]
        for j, tid in enumerate(tid_b):
            if tid in index_a:
                continue
            # new entity: free choice among the tags nobody holds, hand-off bonus if one just ended
            weight = np.ones_like(log_w)
            for i in ended:
                gap = float(np.linalg.norm(e.pos[frame, i] - e.pos[frame + 1, j]))
                bonus = self.params.handoff_gain * np.exp(
                    -0.5 * (gap / self.params.handoff_sigma_px) ** 2
                )
                if bonus > _MIN_HANDOFF:
                    weight += bonus * ~_differs(k, n_a, n_b, i, j)
            log_w += np.log(weight)
        return np.exp(log_w)


# ------------------------------------------------------------------------ emissions
def build_emissions(
    votes: Votes,
    demoted: NDArray[np.bool_],
    entities: Entities,
    n_tags: int,
    params: IdentityParams,
    *,
    extra: Votes | None = None,
    appearance: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Log-evidence ``(F, N, N)`` that entity ``i`` of frame ``f`` carries tag ``k``.

    Args:
        votes: Decode votes.
        demoted: ``(F, S)`` slots whose decodes were demoted to soft evidence.
        entities: The entities.
        n_tags: Number of tags (= bees).
        params: Model parameters.
        extra: Soft votes of external evidence sources (never anchors).
        appearance: Optional ``(F, S, N)`` appearance log-probabilities per slot.

    Returns:
        Additive log-evidence per (frame, entity, tag).
    """
    n_frames, n_ent = entities.tid.shape
    evidence = np.zeros((n_frames, n_ent, n_tags))
    for source, soft in ((votes, True), (extra, False)):
        if source is None or len(source) == 0:
            continue
        ent = entities.of_slot[source.frame, source.slot]
        weight = source.weight.copy()
        if soft:
            weight = np.where(
                demoted[source.frame, source.slot], params.demoted_weight * weight, weight
            )
        usable = (ent >= 0) & (source.tag < n_tags)
        np.add.at(
            evidence,
            (source.frame[usable], ent[usable], source.tag[usable]),
            params.anchor_llr * weight[usable],
        )
    if appearance is not None:
        frames, slots = np.nonzero(entities.of_slot >= 0)
        ent = entities.of_slot[frames, slots]
        gate = np.ones(len(frames))
        lower = 0.0
        nn = entities.nn[frames, ent]
        for limit, factor in params.appearance_gate:
            gate[(nn >= lower) & (nn < limit)] = factor
            lower = limit
        logp = np.clip(appearance[frames, slots, :n_tags], params.appearance_floor, 0.0)
        ok = np.isfinite(logp).all(axis=1)
        evidence[frames[ok], ent[ok]] += params.appearance_weight * gate[ok, None] * logp[ok]
    return evidence


def _state_log_emission(
    evidence: NDArray[np.float64], entities: Entities, n_tags: int
) -> NDArray[np.float64]:
    """``(F, P_max)`` log-emission of every state (``-inf`` for states that do not exist)."""
    n_frames = evidence.shape[0]
    p_max = len(permutation_table(n_tags, n_tags))
    out = np.full((n_frames, p_max), -np.inf)
    for n in range(int(np.max(entities.count, initial=0)) + 1):
        frames = np.flatnonzero(entities.count == n)
        if len(frames) == 0:
            continue
        table = permutation_table(n_tags, n)
        total = np.zeros((len(frames), len(table)))
        for i in range(n):
            total += evidence[frames, i][:, table[:, i]]
        out[np.ix_(frames, np.arange(len(table)))] = total
    return out


def forward_backward(
    evidence: NDArray[np.float64],
    entities: Entities,
    n_tags: int,
    params: IdentityParams,
    transitions: TransitionModel | None = None,
) -> NDArray[np.float32]:
    """Exact marginals ``P(entity i carries tag k)`` for every frame.

    Args:
        evidence: ``(F, N, N)`` log-evidence from :func:`build_emissions`.
        entities: The entities.
        n_tags: Number of tags (= bees).
        params: Model parameters.
        transitions: Optional pre-built transition model.

    Returns:
        ``(F, N, N)`` float32 marginals; rows beyond ``entities.count`` are 0.

    Raises:
        ValueError: If there are no frames.

    Note:
        Memory: the state emissions and forward messages are each an ``(F, N!)`` float64 array
        (N = 6: 720 states, 72,000 frames: about 0.4 GB each), and the transition matrices are
        cached up to 400 MB (else rebuilt in the backward pass). Solve long videos of large
        regions in chunks of frames if that is too much.
    """
    n_frames = evidence.shape[0]
    if n_frames == 0:
        msg = "forward_backward needs at least one frame"
        raise ValueError(msg)
    model = transitions or TransitionModel(entities, n_tags, params)
    log_em = _state_log_emission(evidence, entities, n_tags)
    peak = np.max(np.where(np.isfinite(log_em), log_em, -np.inf), axis=1, keepdims=True)
    em = np.exp(log_em - np.where(np.isfinite(peak), peak, 0.0))
    counts = entities.count
    sizes = [len(permutation_table(n_tags, int(n))) for n in range(n_tags + 1)]
    alpha = np.zeros_like(em)
    state = em[0, : sizes[counts[0]]].copy()
    state /= state.sum()
    alpha[0, : len(state)] = state
    # the matrices are kept for the backward pass when they fit in memory, else rebuilt
    store = (n_frames - 1) * sizes[n_tags] ** 2 * 8 <= _MAX_CACHE_BYTES
    cache: list[NDArray[np.float64]] = []
    for f in range(1, n_frames):
        trans = model.matrix(f - 1)
        if store:
            cache.append(trans)
        state = (state @ trans) * em[f, : trans.shape[1]]
        state /= state.sum()
        alpha[f, : len(state)] = state
    marg = np.zeros((n_frames, n_tags, n_tags), dtype=np.float32)
    beta = np.ones(sizes[counts[-1]])
    for f in range(n_frames - 1, -1, -1):
        n = int(counts[f])
        gamma = alpha[f, : sizes[n]] * beta
        gamma /= gamma.sum()
        if n:
            marg[f, :n, :] = np.einsum("p,pik->ik", gamma, _one_hot(n_tags, n))
        if f:
            trans = cache[f - 1] if store else model.matrix(f - 1)
            beta = trans @ (em[f, : trans.shape[1]] * beta)
            beta /= beta.sum()
    return marg


# ----------------------------------------------------------------------- the result
@dataclass
class RegionLabels:
    """Labels of one region.

    Attributes:
        slot_tag: ``(F, S)`` tag index per instance slot, ``-1`` = untracked.
        slot_conf: ``(F, S)`` marginal of the emitted label (``NaN`` if none).
        slot_source: ``(F, S)`` :class:`~naps.sources.Source` code.
        slot_disputed: ``(F, S)`` an anchored instance whose tag the model confidently disputes.
        slot_flag: ``(F, S)`` :class:`SlotFlag` of an unlabelled instance.
        guess_tag: ``(F, S)`` best tag of an unlabelled instance (``-1`` if none).
        guess_conf: ``(F, S)`` marginal of the guess.
        tag_xy: ``(F, N_real, 2)`` position of every tag (``NaN`` = absent / abstained).
        tag_slot: ``(F, N_real)`` slot carrying the tag, ``-1`` for none or interpolated.
        tag_src: ``(F, N_real)`` source code of the tag's label.
        tag_conf: ``(F, N_real)`` confidence of the tag's label.
        info: Counters for the report.
        tracklets: The tracklets used.
        entities: The entities used.
        candidates: ``(F, S)`` clean decodes before the stray test (-1 none).
        anchors: ``(F, S)`` final anchor tags (-1 none) after demotion.
        valid: ``(F, S)`` instances that took part in the inference.
        marginals: ``(F, N, N)`` marginals of the final inference.
        demoted: ``(F, S)`` instances with a clean decode that is not an anchor (demoted as a
            same-frame conflict or as a stray on its tracklet): the decode is only weak evidence
            and the model may label the instance with another tag.
        low_conf_reason: ``(F, S)`` :class:`~naps.reasons.Reason` code that refines
            ``SlotFlag.LOW_CONFIDENCE`` (0 where the flag is another one).
        consistency_candidates: ``(F,)`` number of recovered tag labels the consistency stage
            examined in every frame.
        consistency_dropped: ``(F,)`` number of those it dropped.
    """

    slot_tag: NDArray[np.int16]
    slot_conf: NDArray[np.float32]
    slot_source: NDArray[np.int8]
    slot_disputed: NDArray[np.bool_]
    slot_flag: NDArray[np.int8]
    guess_tag: NDArray[np.int16]
    guess_conf: NDArray[np.float32]
    tag_xy: NDArray[np.float64]
    tag_slot: NDArray[np.int64]
    tag_src: NDArray[np.int8]
    tag_conf: NDArray[np.float32]
    info: dict[str, object] = field(default_factory=dict)
    tracklets: Tracklets | None = None
    entities: Entities | None = None
    candidates: NDArray[np.int16] | None = None
    anchors: NDArray[np.int16] | None = None
    valid: NDArray[np.bool_] | None = None
    marginals: NDArray[np.float32] | None = None
    demoted: NDArray[np.bool_] | None = None
    low_conf_reason: NDArray[np.int8] | None = None
    consistency_candidates: NDArray[np.int64] | None = None
    consistency_dropped: NDArray[np.int64] | None = None


def _best_assignment(
    marginals: NDArray[np.float32], entities: Entities
) -> tuple[NDArray[np.int16], NDArray[np.float32]]:
    """Per frame, the assignment of entities to tags with the largest total marginal."""
    n_frames, n_ent = entities.tid.shape
    tag = np.full((n_frames, n_ent), -1, dtype=np.int16)
    conf = np.zeros((n_frames, n_ent), dtype=np.float32)
    for f in np.flatnonzero(entities.count > 0):
        n = int(entities.count[f])
        rows, cols = linear_sum_assignment(-marginals[f, :n, :])
        tag[f, rows] = cols
        conf[f, rows] = marginals[f, rows, cols]
    return tag, conf


def _anchor_counts(
    anchors: NDArray[np.int16], tracklets: Tracklets, n_tags: int
) -> NDArray[np.int64]:
    """``(n_tracklets, N)`` number of anchors of every tag on every tracklet."""
    counts = np.zeros((max(tracklets.n_tracklets, 1), n_tags), dtype=np.int64)
    frames, slots = np.nonzero((anchors >= 0) & (tracklets.tid >= 0))
    np.add.at(counts, (tracklets.tid[frames, slots], anchors[frames, slots]), 1)
    return counts


class _Assembler:
    """Turns anchors and model marginals into per-slot and per-tag labels.

    Args:
        table_xy: ``(F, S)`` body positions of the region table.
        entities: The entities.
        tracklets: The tracklets.
        anchors: ``(F, S)`` final anchors.
        marginals: ``(F, N, N)`` marginals of the final inference.
        plain: ``(F, N, N)`` marginals without appearance evidence.
        n_real: Number of real (emitted) tags.
        params: Model parameters.
        ghost: ``(F, S)`` instances that are duplicate copies of a decoded neighbour: they stay in
            the inference (a ghost that is dropped would fragment its bee's tracklet) but never
            receive a label.
    """

    def __init__(  # noqa: PLR0917 (one aligned array per argument)
        self,
        table_xy: NDArray[np.float64],
        entities: Entities,
        tracklets: Tracklets,
        anchors: NDArray[np.int16],
        marginals: NDArray[np.float32],
        plain: NDArray[np.float32],
        n_real: int,
        params: IdentityParams,
        ghost: NDArray[np.bool_],
    ) -> None:
        """Allocate the label arrays and the per-frame lookups."""
        n_frames, n_slots = anchors.shape
        self.ghost = ghost
        self.table_xy, self.entities, self.tracklets = table_xy, entities, tracklets
        self.anchors, self.marginals, self.plain = anchors, marginals, plain
        self.n_real, self.params = n_real, params
        self.ent_tag, self.ent_conf = _best_assignment(marginals, entities)
        self.counts = _anchor_counts(anchors, tracklets, marginals.shape[1])
        shape = (n_frames, n_slots)
        self.slot_tag = np.full(shape, -1, dtype=np.int16)
        self.slot_conf = np.full(shape, np.nan, dtype=np.float32)
        self.slot_source = np.zeros(shape, dtype=np.int8)
        self.disputed = np.zeros(shape, dtype=bool)
        self.flag = np.zeros(shape, dtype=np.int8)
        self.low_reason = np.zeros(shape, dtype=np.int8)
        self.guess_tag = np.full(shape, -1, dtype=np.int16)
        self.guess_conf = np.zeros(shape, dtype=np.float32)
        self.tag_xy = np.full((n_frames, n_real, 2), np.nan)
        self.tag_src = np.zeros((n_frames, n_real), dtype=np.int8)
        self.tag_conf = np.full((n_frames, n_real), np.nan, dtype=np.float32)
        self.anchored = np.zeros((n_frames, marginals.shape[1]), dtype=bool)
        rows, cols = np.nonzero(anchors >= 0)
        self.anchored[rows, anchors[rows, cols]] = True

    def virtual(self, f: int, i: int, k: int, conf: float) -> None:
        """Label a virtual entity (a bridged gap inside a tracklet) with tag ``k``."""
        if conf >= self.params.threshold and k < self.n_real and not self.anchored[f, k]:
            self.tag_xy[f, k] = self.entities.pos[f, i]
            self.tag_src[f, k] = Source.INTERPOLATED
            self.tag_conf[f, k] = conf

    def decoded(self, f: int, i: int, slot: int, k: int, conf: float) -> None:
        """Label a decoded instance; the model may dispute the tag but never change it."""
        anchor = int(self.anchors[f, slot])
        self.slot_tag[f, slot], self.slot_source[f, slot] = anchor, Source.DECODE
        self.slot_conf[f, slot] = self.marginals[f, i, anchor]
        self.disputed[f, slot] = conf >= self.params.threshold and k != anchor

    def low_confidence_reason(self, f: int, i: int, slot: int) -> int:
        """Why a posterior below the threshold: a pile, a tracklet without anchors, or neither."""
        if self.entities.nn[f, i] <= self.params.nn_near_px:
            return Reason.LOW_CONFIDENCE_IN_PILE
        if self.counts[int(self.tracklets.tid[f, slot])].sum() == 0:
            return Reason.NO_ANCHOR_IN_TRACKLET
        return Reason.BELOW_MIN_CONFIDENCE

    def recovered(self, f: int, i: int, slot: int, k: int, conf: float) -> None:
        """Label an instance without a decode with the model's tag, or record why not."""
        if self.ghost[f, slot]:
            self.flag[f, slot] = SlotFlag.GHOST
        elif conf < self.params.threshold:
            self.flag[f, slot] = SlotFlag.LOW_CONFIDENCE
            self.low_reason[f, slot] = self.low_confidence_reason(f, i, slot)
        elif k >= self.n_real:
            self.flag[f, slot] = SlotFlag.UNTAGGED_BEE
        elif self.anchored[f, k]:
            self.flag[f, slot] = SlotFlag.BLOCKED_BY_ANCHOR
        else:
            self.slot_tag[f, slot], self.slot_conf[f, slot] = k, conf
            tid = int(self.tracklets.tid[f, slot])
            if self.plain[f, i, k] < self.params.threshold:
                self.slot_source[f, slot] = Source.APPEARANCE
            elif self.counts[tid, k] >= self.params.min_tracklet_anchors:
                self.slot_source[f, slot] = Source.TRACKLET
            else:
                self.slot_source[f, slot] = Source.ELIMINATION

    def run(self) -> RegionLabels:
        """Label every entity and derive the per-tag arrays."""
        ent = self.entities
        for f in map(int, np.flatnonzero(ent.count > 0)):
            for i in range(int(ent.count[f])):
                slot, k = int(ent.slot[f, i]), int(self.ent_tag[f, i])
                conf = float(self.ent_conf[f, i])
                if slot < 0:
                    self.virtual(f, i, k, conf)
                    continue
                self.guess_tag[f, slot], self.guess_conf[f, slot] = k, conf
                if self.anchors[f, slot] >= 0:
                    self.decoded(f, i, slot, k, conf)
                else:
                    self.recovered(f, i, slot, k, conf)
        ff, ss = np.nonzero(self.slot_tag >= 0)
        kk = self.slot_tag[ff, ss]
        real = kk < self.n_real
        ff, ss, kk = ff[real], ss[real], kk[real]
        tag_slot = np.full(self.tag_src.shape, -1, dtype=np.int64)
        self.tag_xy[ff, kk] = self.table_xy[ff, ss]
        tag_slot[ff, kk] = ss
        self.tag_src[ff, kk] = self.slot_source[ff, ss]
        self.tag_conf[ff, kk] = self.slot_conf[ff, ss]
        return RegionLabels(
            self.slot_tag, self.slot_conf, self.slot_source, self.disputed, self.flag,
            self.guess_tag, self.guess_conf, self.tag_xy, tag_slot, self.tag_src, self.tag_conf,
            low_conf_reason=self.low_reason,
        )  # fmt: skip


# ------------------------------------------------------------------- consistency stage
def _blank_labels(labels: RegionLabels, drop: NDArray[np.bool_]) -> None:
    """Remove the dropped tag labels from the per-tag and per-slot arrays (in place)."""
    frames, tags = np.nonzero(drop)
    for f, k in zip(frames, tags, strict=True):
        slot = int(labels.tag_slot[f, k])
        if slot >= 0:
            labels.slot_tag[f, slot] = -1
            labels.slot_conf[f, slot] = np.nan
            labels.slot_source[f, slot] = Source.NONE
            labels.slot_flag[f, slot] = SlotFlag.CONSISTENCY_DROPPED
    labels.tag_xy[drop] = np.nan
    labels.tag_slot[drop] = -1
    labels.tag_src[drop] = Source.NONE
    labels.tag_conf[drop] = np.nan


def _fill_gaps(labels: RegionLabels, consistency: ConsistencyParams) -> int:
    """Interpolate short gaps of every tag (in place); returns the number of refused frames.

    Nothing is filled when the consistency stage is off (``enabled=False``: without its
    plausibility gates the interpolation would bridge crossings) or ``gap_max`` is 0.
    """
    if not consistency.enabled or consistency.gap_max <= 0:
        return 0
    filled, xy_new, refused = interpolate(labels.tag_xy, consistency)
    for k in range(labels.tag_xy.shape[1]):
        present = np.flatnonzero(np.isfinite(labels.tag_xy[:, k, 0]))
        frames = np.flatnonzero(filled[:, k])
        if len(frames) == 0:
            continue
        after = present[np.searchsorted(present, frames)]
        before = present[np.searchsorted(present, frames) - 1]
        ends = np.stack([labels.tag_conf[before, k], labels.tag_conf[after, k]])
        labels.tag_conf[frames, k] = np.fmin(ends[0], ends[1])
    labels.tag_xy[:] = xy_new
    labels.tag_src[filled] = Source.INTERPOLATED
    return refused


def _apply_consistency(labels: RegionLabels, consistency: ConsistencyParams) -> dict[str, object]:
    """Duplicate / chain veto and gated interpolation on the recovered labels (in place)."""
    anchor_xy = np.where((labels.tag_src == Source.DECODE)[..., None], labels.tag_xy, np.nan)
    candidate = (labels.tag_src >= Source.TRACKLET) & np.isfinite(labels.tag_xy[..., 0])
    result = apply_consistency(
        labels.tag_xy, labels.tag_src, labels.tag_conf, anchor_xy, consistency
    )
    labels.consistency_candidates = candidate.sum(axis=1).astype(np.int64)
    labels.consistency_dropped = result.drop.sum(axis=1).astype(np.int64)
    _blank_labels(labels, result.drop)
    info: dict[str, object] = dict(result.info)
    info["interpolation_refused_frames"] = _fill_gaps(labels, consistency)
    info["interpolated_frames"] = int((labels.tag_src == Source.INTERPOLATED).sum())
    return info


# -------------------------------------------------------------------------- orchestration
def solve_region(  # noqa: PLR0917 (public API: the two parameter objects stay positional)
    table: InstanceTable,
    votes: Votes,
    n_real: int,
    n_bees: int,
    params: IdentityParams | None = None,
    consistency: ConsistencyParams | None = None,
    *,
    tracklet_params: TrackletParams | None = None,
    tracklet_mode: TrackletMode = "built",
    extra: Votes | None = None,
    appearance: NDArray[np.float64] | None = None,
) -> RegionLabels:
    """Label every instance of one region with the joint model.

    Args:
        table: The region's instances (see :meth:`InstanceTable.subset`).
        votes: Decode votes (tag *indices* 0..``n_real``-1).
        n_real: Number of real (decodable) tags of the region.
        n_bees: Number of bees in the region (tags plus untagged bees, <= :data:`MAX_TAGS`).
            The tags ``n_real..n_bees-1`` are phantom tags for the untagged bees.
        params: Model parameters.
        consistency: Temporal-consistency thresholds (``enabled=False`` switches the stage off).
        tracklet_params: Tracklet parameters.
        tracklet_mode: ``"built"`` or ``"sleap"`` (see :func:`naps.tracklets.build_tracklets`).
        extra: Soft votes of external sources (never anchors).
        appearance: Optional ``(F, S, n_bees)`` appearance log-probabilities.

    Returns:
        The :class:`RegionLabels`.

    Raises:
        ValueError: If ``n_real`` or ``n_bees`` are out of range.
    """
    p = params or IdentityParams()
    cons = consistency or ConsistencyParams()
    if not 1 <= n_real <= n_bees <= MAX_TAGS:
        msg = f"need 1 <= n_real <= n_bees <= {MAX_TAGS}, got n_real={n_real}, n_bees={n_bees}"
        raise ValueError(msg)
    shape = (table.n_frames, table.n_slots)
    candidate = candidate_anchors(votes, shape, n_real, p.min_anchor_weight)
    claimed = candidate >= 0
    rank = np.where(np.isfinite(table.score), table.score, 0.0) + 1e3 * claimed

    present = table.valid
    valid = dedupe_instances(table.xy, rank, p.dedupe_px)
    flag = np.zeros(shape, dtype=np.int8)
    flag[present & ~valid] = SlotFlag.DUPLICATE_INSTANCE
    ghost = np.zeros(shape, dtype=bool)
    if cons.enabled and cons.ghost:
        ghost = ghost_mask(table.xy, valid, claimed, table.score, params=cons, pose=table.pose)
    # ghosts stay in the inference (removing them fragments the tracklet of the bee they touch)
    # but rank last when the region is over capacity, and never receive a label
    kept = cull_excess(valid, rank - 1e2 * ghost, n_bees)
    culled = valid & ~kept
    flag[culled] = SlotFlag.EXCESS_INSTANCE
    flag[ghost] = SlotFlag.GHOST  # a ghost that was also culled stays a ghost (the specific cause)
    n_ghost_culled = int((ghost & culled).sum())
    ghost &= kept

    if tracklet_mode == "sleap" and kept.any() and float((table.hint[kept] < 0).mean()) > 0.5:
        logger.warning(
            "--tracklets sleap: most instances have no SLEAP track, and in this mode an instance "
            "without a track id never continues a tracklet; expect fragmented tracklets and many "
            "abstentions (use --tracklets built for untracked input)."
        )
    tracklets = build_tracklets(
        table.xy, kept, tracklet_params, hint=table.hint, heading=table.heading, mode=tracklet_mode
    )
    single = np.where(kept, single_decodes(votes, shape, n_real, p.min_anchor_weight), -1)
    candidate = np.where(kept, candidate, -1).astype(np.int16)
    stray = demote_strays(candidate, tracklets, n_real, p)
    anchors = np.where(stray, -1, candidate).astype(np.int16)
    demoted = (single >= 0) & (anchors < 0)
    entities = build_entities(tracklets, table.xy, n_bees)
    evidence = build_emissions(
        votes, anchors < 0, entities, n_bees, p, extra=extra, appearance=appearance
    )
    marginals = forward_backward(evidence, entities, n_bees, p)
    plain = marginals
    if appearance is not None:
        plain = forward_backward(
            build_emissions(votes, anchors < 0, entities, n_bees, p, extra=extra),
            entities,
            n_bees,
            p,
        )
    labels = _Assembler(
        table.xy, entities, tracklets, anchors, marginals, plain, n_real, p, ghost
    ).run()
    labels.slot_flag = np.where(labels.slot_flag == 0, flag, labels.slot_flag).astype(np.int8)
    labels.info = {
        "n_instances": int(present.sum()),
        "n_duplicate_instances": int((flag == SlotFlag.DUPLICATE_INSTANCE).sum()),
        "n_ghost_instances": int((flag == SlotFlag.GHOST).sum()),
        # every instance culled for exceeding the colony size, ghosts that were culled included
        # (their flag stays "ghost"); the flag-based count is ``abstentions["excess_instance"]``
        "n_excess_instances": int(culled.sum()),
        "n_ghost_instances_culled": n_ghost_culled,
        "n_candidate_anchors": int(claimed.sum()),
        # decodes that were not anchors: the same tag on two instances of a frame, or a minority
        # tag on a tracklet. They only weigh `demoted_weight` and the instance may be relabelled.
        "n_conflict_decodes_demoted": int(((single >= 0) & (candidate < 0)).sum()),
        "n_stray_anchors_demoted": int(stray.sum()),
        "n_virtual_dropped": entities.n_dropped_virtual,
        "n_tracklets": tracklets.n_tracklets,
    }
    labels.info["consistency"] = _apply_consistency(labels, cons)
    labels.info["n_demoted_decodes_relabelled"] = int(
        (demoted & (labels.slot_tag >= 0) & (labels.slot_tag != single)).sum()
    )
    labels.info["n_demoted_decodes_confirmed"] = int((demoted & (labels.slot_tag == single)).sum())
    labels.tracklets, labels.entities = tracklets, entities
    labels.candidates, labels.anchors = candidate, anchors
    labels.valid, labels.marginals, labels.demoted = kept, marginals, demoted
    return labels
