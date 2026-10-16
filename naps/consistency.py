"""Temporal-consistency stage of the joint identity engine.

The probabilistic model labels the right bee ~99.8% of the time on held-out data, but labels
*recovered* by it (as opposed to decoded) used to be temporally implausible: tags teleported,
landed on a bee that already carried another tag, or were interpolated through crossings. This
stage removes exactly those, using thresholds that were calibrated on 44 windows / 1.2 million
decoded tag-pairs of real NAPS data (every number below is a documented default of
:class:`ConsistencyParams` and can be overridden):

* **ghosts** (:func:`ghost_mask`): an unclaimed instance < 50 px from a decoded instance (or
  < 80 px with an overlapping pose, or < 50 px from a higher-scored unclaimed instance) is a
  duplicate detection and never receives a tag.
* **duplicates** (:func:`drop_duplicates`): a recovered label < 50 px from another tag's position in
  the same frame is suppressed (inter-tag distance of decoded tags: p0.01 = 50 px).
* **chain veto** (:func:`resolve_tracks`): per tag, decoded positions are immutable anchors and
  recovered positions are candidates; the maximum-weight chain of segments in which successive
  positions are within ``allowed_displacement(gap)`` (the p99.9 of decoded displacements over a
  lag of ``gap`` frames) is kept and all other candidates are dropped.
* **gated interpolation** (:func:`interpolate`): a gap of <= 20 frames is filled linearly only if
  both ends are plausibly linked, <= 60 px apart (P(error < 40 px) = 0.984), and the filled point
  is not within 50 px of another tag.

Decoded (anchor) labels are never modified or dropped here.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from naps.sources import Source

__all__ = [
    "GATE_FRAMES",
    "GATE_PX",
    "ConsistencyParams",
    "ConsistencyResult",
    "allowed_displacement",
    "anchors_off_chain",
    "apply_consistency",
    "bridge_violations",
    "drop_duplicates",
    "duplicate_stats",
    "ghost_mask",
    "interpolate",
    "jump_matrix",
    "recovered_run_stats",
    "resolve_tag",
    "resolve_tracks",
]

#: Lag (frames) and the p99.9 displacement (px) of one decoded tag over that lag, pooled over 44
#: windows (922k..731k pairs). The lag-1 value is rounded from 116 to 100 px so that "jump" and
#: "gate violation" coincide.
GATE_FRAMES = np.array([1, 2, 3, 5, 10, 20, 40, 100, 200], dtype=float)
GATE_PX = np.array([100, 133, 142, 160, 237, 409, 708, 1172, 1420], dtype=float)

#: Reason codes of a dropped candidate label (``ConsistencyResult.reason``).
REASON_DUPLICATE = 1
REASON_ANCHOR_CONFLICT = 2
REASON_CANDIDATE_CONFLICT = 3
REASON_RUN = 4
REASON_SHORT = 5

_CODE_ANCHOR = 1  # chunk kind: decoded anchor
_CODE_CANDIDATE = 2  # chunk kind: recovered candidate
_PAIR_CHUNK = 4096  # frames per block when computing pairwise distances


@dataclass(frozen=True)
class ConsistencyParams:
    """Thresholds of the temporal-consistency stage (calibrated defaults, see module docstring).

    Attributes:
        enabled: ``False`` switches the whole stage off (ablation / legacy-like behaviour).
        gate_scale: Multiplies the p99.9 lag table (0.5 is about the p99 table; it lost coverage).
        hard_px: Displacement per frame that is physically impossible (~19 mm/frame).
        arena_cap_px: Empirical saturation of the lag displacement (a quadrant is 1832 px).
        dup_gate_px: Two tags closer than this in one frame are duplicates (decoded: 0.0097%).
        anchor_close_window: A recovered label near an anchor is kept if the decoded tags were
            that close within +-this many frames anyway.
        run_policy: ``"truncate"`` keeps the optimal consistent chain; ``"drop"`` removes a whole
            recovered run if it contains an impossible step.
        anchor_weight: Weight of an anchor frame relative to a recovered frame in the chain.
        min_seg_len: Floating (unanchored) recovered segments shorter than this are dropped.
        gap_max: Longest gap (frames) that is interpolated.
        interp_max_disp_px: Interpolate only if the two ends are at most this far apart.
        ghost: Apply the ghost mask.
        ghost_gate_px: Unclaimed instance this close to a decoded one is a ghost.
        ghost_pose_gate_px: ... or this close with an overlapping pose.
        ghost_pose_dist_px: Pose overlap = median node distance below this.
        ghost_min_shared: ... over at least this many shared nodes.
    """

    enabled: bool = True
    gate_scale: float = 1.0
    hard_px: float = 300.0
    arena_cap_px: float = 1450.0
    dup_gate_px: float = 50.0
    anchor_close_window: int = 10
    run_policy: Literal["truncate", "drop"] = "truncate"
    anchor_weight: float = 50.0
    min_seg_len: int = 1
    gap_max: int = 20
    interp_max_disp_px: float = 60.0
    ghost: bool = True
    ghost_gate_px: float = 50.0
    ghost_pose_gate_px: float = 80.0
    ghost_pose_dist_px: float = 25.0
    ghost_min_shared: int = 3


@dataclass
class ConsistencyResult:
    """Outcome of :func:`apply_consistency`.

    Attributes:
        drop: ``(F, K)`` recovered labels that were removed.
        reason: ``(F, K)`` reason code of every removed label (0 = kept).
        info: Counts for the report.
    """

    drop: NDArray[np.bool_]
    reason: NDArray[np.int8]
    info: dict[str, int | float | bool]


def allowed_displacement(
    gap: NDArray[np.float64] | float, params: ConsistencyParams | None = None
) -> NDArray[np.float64]:
    """Largest plausible displacement (px) of one tag between two positions ``gap`` frames apart."""
    p = params or ConsistencyParams()
    lag = np.maximum(np.asarray(gap, dtype=float), 1.0)
    table = p.gate_scale * np.interp(lag, GATE_FRAMES, GATE_PX)
    return np.minimum(np.minimum(table, p.arena_cap_px), p.hard_px * lag)


# ------------------------------------------------------------------------------- ghosts
def _pairwise(xy: NDArray[np.float64], valid: NDArray[np.bool_]) -> NDArray[np.float64]:
    """``(F, S, S)`` distances between valid instances (``inf`` elsewhere and on the diagonal)."""
    dist = np.linalg.norm(xy[:, :, None, :] - xy[:, None, :, :], axis=-1)
    dist[~valid[:, :, None] | ~valid[:, None, :]] = np.inf
    n = xy.shape[1]
    dist[:, np.arange(n), np.arange(n)] = np.inf
    return dist


def _poses_overlap(
    pose_a: NDArray[np.float32], pose_b: NDArray[np.float32], params: ConsistencyParams
) -> bool:
    """Whether two poses coincide (median node distance below the gate over enough nodes)."""
    shared = np.isfinite(pose_a[:, 0]) & np.isfinite(pose_b[:, 0])
    if int(shared.sum()) < params.ghost_min_shared:
        return False
    median = np.median(np.linalg.norm(pose_a[shared] - pose_b[shared], axis=-1))
    return bool(median < params.ghost_pose_dist_px)


def ghost_mask(
    xy: NDArray[np.float64],
    valid: NDArray[np.bool_],
    claimed: NDArray[np.bool_],
    score: NDArray[np.float64],
    *,
    params: ConsistencyParams | None = None,
    pose: NDArray[np.float32] | None = None,
) -> NDArray[np.bool_]:
    """Mark instances that are duplicate / partial copies of a neighbouring instance.

    An instance can only be a ghost if it is not *claimed* (it carries no decoded tag). It is a
    ghost if (a) it is closer than ``ghost_gate_px`` to a claimed instance, (b) closer than
    ``ghost_pose_gate_px`` to a claimed instance and their poses overlap, or (c) closer than
    ``ghost_gate_px`` to a higher-scored unclaimed instance (ties: the lower slot wins).

    Args:
        xy: ``(F, S, 2)`` body positions.
        valid: ``(F, S)`` instances present.
        claimed: ``(F, S)`` instances that carry a decoded tag.
        score: ``(F, S)`` instance scores (``NaN`` counts as 0).
        params: Thresholds (default :class:`ConsistencyParams`).
        pose: Optional ``(F, S, N, 2)`` poses for rule (b).

    Returns:
        ``(F, S)`` boolean mask of ghosts.
    """
    p = params or ConsistencyParams()
    ghost = np.zeros(valid.shape, dtype=bool)
    ranking = np.where(np.isfinite(score), score, 0.0)
    unclaimed = valid & ~claimed
    for lo in range(0, valid.shape[0], _PAIR_CHUNK):
        hi = lo + _PAIR_CHUNK
        dist = _pairwise(xy[lo:hi], valid[lo:hi])
        events = np.nonzero((dist < p.ghost_pose_gate_px) & unclaimed[lo:hi, :, None])
        for rel, s, c in zip(*events, strict=True):
            f = lo + int(rel)
            if ghost[f, s]:
                continue
            d = dist[rel, s, c]
            if claimed[f, c]:
                near = d < p.ghost_gate_px
                overlap = pose is not None and _poses_overlap(pose[f, s], pose[f, c], p)
                ghost[f, s] = near or overlap
            elif d < p.ghost_gate_px:
                rival = ranking[f, c]
                ghost[f, s] = rival > ranking[f, s] or (rival == ranking[f, s] and c < s)
    return ghost


# --------------------------------------------------------------------------- duplicates
def _anchors_close_nearby(
    anchor_xy: NDArray[np.float64], params: ConsistencyParams
) -> NDArray[np.bool_]:
    """``(F, K, K)``: the decoded tags k and j were closer than ``dup_gate_px`` within +-window."""
    n = anchor_xy.shape[1]
    close = np.zeros((*anchor_xy.shape[:1], n, n), dtype=bool)
    for lo in range(0, anchor_xy.shape[0], _PAIR_CHUNK):
        sl = slice(lo, lo + _PAIR_CHUNK)
        dist = np.linalg.norm(anchor_xy[sl, :, None, :] - anchor_xy[sl, None, :, :], axis=-1)
        close[sl] = np.nan_to_num(dist, nan=np.inf) < params.dup_gate_px
    close[:, np.arange(n), np.arange(n)] = False
    window = int(params.anchor_close_window)
    if window <= 0 or not close.any():
        return close
    out = close.copy()
    for shift in range(1, window + 1):
        out[shift:] |= close[:-shift]
        out[:-shift] |= close[shift:]
    return out


def drop_duplicates(
    xy: NDArray[np.float64],
    src: NDArray[np.int8],
    conf: NDArray[np.float32],
    anchor_xy: NDArray[np.float64],
    params: ConsistencyParams | None = None,
) -> NDArray[np.bool_]:
    """Recovered labels that sit on top of another tag.

    A candidate (``src >= TRACKLET``) closer than ``dup_gate_px`` to the decoded position of another
    tag is dropped; of two candidates that close the lower-confidence one is dropped (a tie drops
    both). Pairs of tags that were that close in the decoded data within +-``anchor_close_window``
    frames are exempt (bees really touching).

    Args:
        xy: ``(F, K, 2)`` tag positions (``NaN`` = none).
        src: ``(F, K)`` source codes.
        conf: ``(F, K)`` label confidences.
        anchor_xy: ``(F, K, 2)`` decoded positions (``NaN`` where the tag was not decoded).
        params: Thresholds.

    Returns:
        ``(F, K)`` boolean mask of the candidates to drop.
    """
    p = params or ConsistencyParams()
    candidate = (src >= Source.TRACKLET) & np.isfinite(xy[..., 0])
    drop = np.zeros(candidate.shape, dtype=bool)
    if not candidate.any():
        return drop
    anchored = np.isfinite(anchor_xy[..., 0]) & (src == Source.DECODE)
    n = xy.shape[1]
    dist = np.linalg.norm(xy[:, :, None, :] - xy[:, None, :, :], axis=-1)
    dist[:, np.arange(n), np.arange(n)] = np.inf
    close = np.nan_to_num(dist, nan=np.inf) < p.dup_gate_px
    exempt = _anchors_close_nearby(anchor_xy, p)
    on_anchor = close & anchored[:, None, :] & candidate[:, :, None] & ~exempt
    drop |= on_anchor.any(axis=2)
    recovered = (src >= Source.TRACKLET) & np.isfinite(xy[..., 0])
    pair = close & recovered[:, :, None] & recovered[:, None, :] & candidate[:, :, None] & ~exempt
    conf0 = np.where(np.isfinite(conf), conf, 0.0)
    for f in np.flatnonzero(pair.any(axis=(1, 2))):
        for k, j in zip(*np.nonzero(pair[f]), strict=True):
            if drop[f, k] or drop[f, j]:
                continue
            if conf0[f, k] < conf0[f, j]:
                drop[f, k] = True
            elif conf0[f, k] > conf0[f, j]:
                drop[f, j] = drop[f, j] or candidate[f, j]
            else:
                drop[f, k] = True
                drop[f, j] |= candidate[f, j]
    return drop & candidate


# ------------------------------------------------------------------------------- chains
def _chunks(
    pos: NDArray[np.float64],
    kind: NDArray[np.int8],
    split_candidates: bool,
    params: ConsistencyParams,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int8], NDArray[np.bool_]]:
    """Maximal runs of consecutive frames of one kind (1 anchor, 2 candidate).

    Anchor runs are cut at their own impossible steps; candidate runs are cut there too if
    ``split_candidates``, else kept whole with a ``bad`` flag. Returns the first and last frame,
    the kind and the ``bad`` flag of every chunk.
    """
    idx = np.flatnonzero(kind > 0)
    if len(idx) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=np.int8), np.zeros(0, dtype=bool)
    step = np.linalg.norm(pos[idx[1:]] - pos[idx[:-1]], axis=-1)
    step_bad = step > allowed_displacement(1, params)
    adjacent = (np.diff(idx) == 1) & (kind[idx[1:]] == kind[idx[:-1]])
    cut = ~adjacent | (step_bad & (split_candidates | (kind[idx[1:]] == _CODE_ANCHOR)))
    starts = np.flatnonzero(np.r_[True, cut])
    ends = np.r_[starts[1:] - 1, len(idx) - 1]
    inner_bad = np.r_[0, np.cumsum(step_bad & adjacent & ~cut)]
    bad = (inner_bad[ends] - inner_bad[starts]) > 0
    return idx[starts], idx[ends], kind[idx[starts]], bad


def _best_chain(
    first: NDArray[np.int64],
    last: NDArray[np.int64],
    weight: NDArray[np.float64],
    pos: NDArray[np.float64],
    params: ConsistencyParams,
) -> NDArray[np.bool_]:
    """Maximum-weight chain of chunks whose successive members are plausibly linked.

    Chunks are ordered in time and disjoint; weight-0 chunks are excluded. O(n^2) dynamic
    programme over the chunks.
    """
    n = len(first)
    start_pos, end_pos = pos[first], pos[last]
    best = np.zeros(n)
    parent = np.full(n, -1)
    for i in range(n):
        if weight[i] == 0.0:
            continue
        best[i] = weight[i]
        if i == 0:
            continue
        before = np.arange(i)
        gap = first[i] - last[before]
        linked = (weight[before] > 0) & (
            np.linalg.norm(end_pos[before] - start_pos[i], axis=-1)
            <= allowed_displacement(gap, params)
        )
        if linked.any():
            candidates = before[linked]
            top = candidates[np.argmax(best[candidates])]
            best[i] = weight[i] + best[top]
            parent[i] = top
    keep = np.zeros(n, dtype=bool)
    node = int(np.argmax(best)) if n and float(np.max(best)) > 0 else -1
    while node >= 0:
        keep[node] = True
        node = int(parent[node])
    return keep


def _tag_kinds(
    pos: NDArray[np.float64], is_anchor: NDArray[np.bool_], cand: NDArray[np.bool_]
) -> NDArray[np.int8]:
    """Chunk kind per frame: 1 anchor, 2 candidate, 0 none."""
    finite = np.isfinite(pos[:, 0])
    return np.where(finite & is_anchor, 1, np.where(finite & cand, 2, 0)).astype(np.int8)


def _chain_weights(
    kind: NDArray[np.int8],
    first: NDArray[np.int64],
    last: NDArray[np.int64],
    params: ConsistencyParams,
) -> NDArray[np.float64]:
    """Chunk weights: an anchor frame counts ``anchor_weight`` candidate frames."""
    frames = (last - first + 1).astype(float)
    return np.where(kind == _CODE_ANCHOR, params.anchor_weight * frames, frames)


def anchors_off_chain(
    xy: NDArray[np.float64], src: NDArray[np.int8], params: ConsistencyParams | None = None
) -> NDArray[np.bool_]:
    """Anchor frames that contradict the rest of their tag's track.

    Runs the chain optimisation of :func:`resolve_tag` on the final tracks; anchors that end up
    off the maximum-weight chain are decoded identity glitches (a tag that teleports for a few
    frames). Only used to attribute gate violations to the decoder rather than to the engine.

    Returns:
        ``(F, K)`` boolean mask.
    """
    p = params or ConsistencyParams()
    off = np.zeros(src.shape, dtype=bool)
    for k in range(xy.shape[1]):
        kind = _tag_kinds(xy[:, k], src[:, k] == Source.DECODE, src[:, k] >= Source.TRACKLET)
        if (kind == _CODE_ANCHOR).sum() < 1 or (kind > 0).sum() < 2:
            continue
        first, last, kd, _ = _chunks(xy[:, k], kind, True, p)
        weight = _chain_weights(kd, first, last, p)
        keep = _best_chain(first, last, weight, xy[:, k], p)
        for i in np.flatnonzero(~keep & (kd == _CODE_ANCHOR)):
            off[first[i] : last[i] + 1, k] = True
    return off


def _short_floating(
    keep: NDArray[np.bool_],
    kd: NDArray[np.int8],
    first: NDArray[np.int64],
    last: NDArray[np.int64],
    params: ConsistencyParams,
) -> list[int]:
    """Kept candidate chunks shorter than ``min_seg_len`` with no anchor neighbour in the chain."""
    chain = np.flatnonzero(keep)
    short = []
    for a, i in enumerate(chain):
        if kd[i] != _CODE_CANDIDATE or (last[i] - first[i] + 1) >= params.min_seg_len:
            continue
        before = a > 0 and kd[chain[a - 1]] == _CODE_ANCHOR
        after = a + 1 < len(chain) and kd[chain[a + 1]] == _CODE_ANCHOR
        if not (before or after):
            short.append(int(i))
    return short


def resolve_tag(
    pos: NDArray[np.float64],
    is_anchor: NDArray[np.bool_],
    cand: NDArray[np.bool_],
    params: ConsistencyParams | None = None,
) -> tuple[NDArray[np.bool_], NDArray[np.int8]]:
    """Optimal temporal-consistency selection for one tag.

    Nodes are chunks (runs of consecutive frames of one kind). A chain of chunks is consistent if
    every successive pair is linked (distance <= ``allowed_displacement(gap)``). The consistent
    chain with the maximum weight is kept (candidate chunk = its number of frames, anchor chunk =
    ``anchor_weight`` x frames, so anchors win whenever they are consistent at all); candidate
    chunks off the chain are dropped.

    Args:
        pos: ``(F, 2)`` positions of the tag.
        is_anchor: ``(F,)`` decoded (immutable) frames.
        cand: ``(F,)`` recovered candidate frames.
        params: Thresholds.

    Returns:
        ``(drop, reason)``: the candidate frames to drop and why (2 conflicts with an anchor
        chunk, 3 conflicts with another candidate, 4 run dropped by the ``"drop"`` policy, 5 short
        floating chunk).
    """
    p = params or ConsistencyParams()
    drop = np.zeros(len(pos), dtype=bool)
    reason = np.zeros(len(pos), dtype=np.int8)
    kind = _tag_kinds(pos, is_anchor, cand)
    split = p.run_policy != "drop"
    first, last, kd, bad = _chunks(pos, kind, split, p)
    if len(first) == 0 or not (kd == _CODE_CANDIDATE).any():
        return drop, reason
    weight = _chain_weights(kd, first, last, p)
    if not split:
        weight = np.where((kd == _CODE_CANDIDATE) & bad, 0.0, weight)
    start_pos, end_pos = pos[first], pos[last]
    keep = _best_chain(first, last, weight, pos, p)
    dropped = [int(i) for i in np.flatnonzero((kd == _CODE_CANDIDATE) & ~keep)]
    short = _short_floating(keep, kd, first, last, p) if p.min_seg_len > 1 else []
    kept_anchor = np.flatnonzero(keep & (kd == _CODE_ANCHOR))
    for i in sorted(set(dropped + short)):
        drop[first[i] : last[i] + 1] = True
        if i in short:
            reason[first[i] : last[i] + 1] = REASON_SHORT
            continue
        code = REASON_RUN if (not split and bad[i]) else REASON_CANDIDATE_CONFLICT
        for nb in (kept_anchor[kept_anchor < i][-1:], kept_anchor[kept_anchor > i][:1]):
            for other in nb:
                a, b = (int(other), i) if other < i else (i, int(other))
                gap = first[b] - last[a]
                if np.linalg.norm(end_pos[a] - start_pos[b]) > allowed_displacement(gap, p):
                    code = REASON_ANCHOR_CONFLICT if code == REASON_CANDIDATE_CONFLICT else code
        reason[first[i] : last[i] + 1] = code
    return drop, reason


def resolve_tracks(
    xy: NDArray[np.float64], src: NDArray[np.int8], params: ConsistencyParams | None = None
) -> tuple[NDArray[np.bool_], NDArray[np.int8]]:
    """:func:`resolve_tag` for every tag; returns ``(drop, reason)`` of shape ``(F, K)``."""
    drop = np.zeros(src.shape, dtype=bool)
    reason = np.zeros(src.shape, dtype=np.int8)
    for k in range(xy.shape[1]):
        drop[:, k], reason[:, k] = resolve_tag(
            xy[:, k], src[:, k] == Source.DECODE, src[:, k] >= Source.TRACKLET, params
        )
    return drop, reason


def apply_consistency(
    xy: NDArray[np.float64],
    src: NDArray[np.int8],
    conf: NDArray[np.float32],
    anchor_xy: NDArray[np.float64],
    params: ConsistencyParams | None = None,
) -> ConsistencyResult:
    """Duplicate suppression and chain veto on the recovered labels (no interpolation).

    Args:
        xy: ``(F, K, 2)`` tag positions.
        src: ``(F, K)`` source codes (anchors = ``Source.DECODE``, recovered ``>= TRACKLET``).
        conf: ``(F, K)`` confidences.
        anchor_xy: ``(F, K, 2)`` decoded positions.
        params: Thresholds.

    Returns:
        The labels to drop with reasons and counts. Anchors are never dropped.
    """
    p = params or ConsistencyParams()
    candidate = (src >= Source.TRACKLET) & np.isfinite(xy[..., 0])
    n_candidates = int(candidate.sum())
    none = np.zeros(src.shape, dtype=bool)
    info: dict[str, int | float | bool] = {"enabled": p.enabled, "n_candidates": n_candidates}
    if not p.enabled or n_candidates == 0:
        info.update(
            dropped_total=0,
            dropped_duplicate=0,
            dropped_anchor_conflict=0,
            dropped_candidate_conflict=0,
            dropped_run=0,
            dropped_short=0,
            dropped_fraction=0.0,
        )
        return ConsistencyResult(none, np.zeros(src.shape, dtype=np.int8), info)
    dup = drop_duplicates(xy, src, conf, anchor_xy, p)
    xy_left = np.where(dup[..., None], np.nan, xy)
    src_left = np.where(dup, 0, src).astype(np.int8)
    chain_drop, why = resolve_tracks(xy_left, src_left, p)
    drop = dup | chain_drop
    reason = np.where(dup, REASON_DUPLICATE, why).astype(np.int8)
    info.update(
        dropped_total=int(drop.sum()),
        dropped_duplicate=int(dup.sum()),
        dropped_anchor_conflict=int(((why == REASON_ANCHOR_CONFLICT) & ~dup).sum()),
        dropped_candidate_conflict=int(((why == REASON_CANDIDATE_CONFLICT) & ~dup).sum()),
        dropped_run=int(((why == REASON_RUN) & ~dup).sum()),
        dropped_short=int(((why == REASON_SHORT) & ~dup).sum()),
        dropped_fraction=round(float(drop.sum() / n_candidates), 4),
    )
    return ConsistencyResult(drop, reason, info)


# ------------------------------------------------------------------------- interpolation
def interpolate(
    xy: NDArray[np.float64],
    params: ConsistencyParams | None = None,
) -> tuple[NDArray[np.bool_], NDArray[np.float64], int]:
    """Fill short gaps of every tag by linear interpolation, where that is plausible.

    A gap of 1..``gap_max`` frames between two present positions of one tag is filled only if the
    two ends are linked (``allowed_displacement``), at most ``interp_max_disp_px`` apart, and the
    interpolated point is not within ``dup_gate_px`` of another tag's position in that frame.
    With ``enabled=False`` the plausibility gates are skipped (v1-like).

    Args:
        xy: ``(F, K, 2)`` tag positions (``NaN`` = absent).
        params: Thresholds.

    Returns:
        ``(filled, xy_new, n_refused)``: the filled ``(F, K)`` mask, the positions with the filled
        points, and the number of gap frames that were refused by a gate.
    """
    p = params or ConsistencyParams()
    out = xy.copy()
    filled = np.zeros(xy.shape[:2], dtype=bool)
    refused = 0
    n_tags = xy.shape[1]
    for k in range(n_tags):
        present = np.flatnonzero(np.isfinite(xy[:, k, 0]))
        for a, b in pairwise(present):
            missing = b - a - 1
            if missing <= 0 or missing > p.gap_max:
                continue
            frames = np.arange(a + 1, b)
            ends = float(np.linalg.norm(xy[b, k] - xy[a, k]))
            if p.enabled and ends > min(
                float(allowed_displacement(int(b - a), p)), p.interp_max_disp_px
            ):
                refused += len(frames)
                continue
            weight = ((frames - a) / (b - a))[:, None]
            points = (1 - weight) * xy[a, k][None] + weight * xy[b, k][None]
            if p.enabled:
                others = np.delete(np.arange(n_tags), k)
                dist = np.linalg.norm(out[frames][:, others] - points[:, None, :], axis=-1)
                near = (np.nan_to_num(dist, nan=np.inf) < p.dup_gate_px).any(axis=1)
                refused += int(near.sum())
                frames, points = frames[~near], points[~near]
            out[frames, k] = points
            filled[frames, k] = True
    return filled, out, refused


# ----------------------------------------------------------------------------- metrics
def jump_matrix(
    xy: NDArray[np.float64],
    src: NDArray[np.int8],
    thresholds: tuple[float, ...] = (100.0, 300.0),
    off: NDArray[np.bool_] | None = None,
) -> dict[str, dict[str, float | int | None]]:
    """Step statistics between consecutive frames by class of the pair of labels.

    Classes: ``anchor_anchor``, ``rec_rec`` (recovered incl. interpolated) and ``anchor_rec``.

    Args:
        xy: ``(F, K, 2)`` positions.
        src: ``(F, K)`` source codes.
        thresholds: Jump sizes (px) to count.
        off: Optional :func:`anchors_off_chain` mask: steps touching such an anchor are counted
            separately as ``..._anchor_glitch`` (the jump is the decoder's, not the engine's).
    """
    s0, s1 = src[:-1], src[1:]
    step = np.linalg.norm(xy[1:] - xy[:-1], axis=-1)
    finite = np.isfinite(step)
    a0, a1 = s0 == Source.DECODE, s1 == Source.DECODE
    r0, r1 = s0 >= Source.TRACKLET, s1 >= Source.TRACKLET
    classes = {
        "anchor_anchor": a0 & a1,
        "rec_rec": r0 & r1,
        "anchor_rec": (a0 & r1) | (r0 & a1),
    }
    out: dict[str, dict] = {}
    for name, members in classes.items():
        mask = members & finite
        n = int(mask.sum())
        entry: dict[str, float | int | None] = {"n": n}
        for t in thresholds:
            entry[f"gt{int(t)}"] = int((step[mask] > t).sum())
            entry[f"rate_gt{int(t)}"] = float((step[mask] > t).mean()) if n else None
        out[name] = entry
    if off is not None:
        glitch = (a0 & r1 & off[:-1]) | (r0 & a1 & off[1:])
        mask = classes["anchor_rec"] & finite
        out["anchor_rec"]["n_touching_anchor_glitch"] = int((mask & glitch).sum())
        for t in thresholds:
            out["anchor_rec"][f"gt{int(t)}_anchor_glitch"] = int(((step > t) & mask & glitch).sum())
            out["anchor_rec"][f"gt{int(t)}_own"] = int(((step > t) & mask & ~glitch).sum())
    return out


def recovered_run_stats(
    xy: NDArray[np.float64], src: NDArray[np.int8], jump: float = 100.0
) -> dict[str, int | float]:
    """Recovered labels grouped into contiguous runs per tag; the share in runs with a big jump."""
    n_labels = n_bad = n_runs = n_bad_runs = 0
    for k in range(xy.shape[1]):
        flag = np.r_[False, (src[:, k] >= Source.TRACKLET) & np.isfinite(xy[:, k, 0]), False]
        starts = np.flatnonzero(~flag[:-1] & flag[1:])
        ends = np.flatnonzero(flag[:-1] & ~flag[1:])
        for a, b in zip(starts, ends, strict=True):
            n_runs += 1
            n_labels += int(b - a)
            if b - a > 1:
                steps = np.linalg.norm(np.diff(xy[a:b, k], axis=0), axis=-1)
                if np.isfinite(steps).any() and np.nanmax(steps) > jump:
                    n_bad += int(b - a)
                    n_bad_runs += 1
    return {
        "n_recovered_labels": n_labels,
        "n_runs": n_runs,
        "n_runs_with_jump": n_bad_runs,
        "labels_in_runs_with_jump": n_bad,
        "frac_labels_in_runs_with_jump": round(n_bad / n_labels, 4) if n_labels else 0.0,
    }


def duplicate_stats(
    xy: NDArray[np.float64], src: NDArray[np.int8], gate: float = 40.0
) -> dict[str, float | int]:
    """Bee-frames closer than ``gate`` px to another tag in the same frame, split by label class."""
    n = xy.shape[1]
    dist = np.linalg.norm(xy[:, :, None, :] - xy[:, None, :, :], axis=-1)
    dist[:, np.arange(n), np.arange(n)] = np.inf
    close = np.nan_to_num(dist, nan=np.inf) < gate
    rec = src >= Source.TRACKLET
    anchor = src == Source.DECODE
    return {
        "gate_px": gate,
        "recovered_labels_near_other_tag": int((close.any(axis=2) & rec).sum()),
        "recovered_labels_total": int(rec.sum()),
        "anchor_pairs_near": int((close & anchor[:, :, None] & anchor[:, None, :]).sum() // 2),
        "anchor_recovered_pairs_near": int((close & anchor[:, :, None] & rec[:, None, :]).sum()),
        "recovered_recovered_pairs_near": int(
            (close & rec[:, :, None] & rec[:, None, :]).sum() // 2
        ),
    }


def bridge_violations(
    xy: NDArray[np.float64],
    src: NDArray[np.int8],
    params: ConsistencyParams | None = None,
    off: NDArray[np.bool_] | None = None,
) -> dict[str, int]:
    """Links between successive positions of a tag that exceed the gate (with a recovered label).

    ``n_violations_own`` leaves out links touching an anchor glitch (``off``): those are the ones
    the engine is responsible for. Without ``off`` every violation counts as own.
    """
    totals = {"links": 0, "bad": 0, "bad300": 0, "own": 0, "own300": 0}
    for k in range(xy.shape[1]):
        idx = np.flatnonzero(np.isfinite(xy[:, k, 0]))
        if len(idx) < 2:
            continue
        step = np.linalg.norm(xy[idx[1:], k] - xy[idx[:-1], k], axis=-1)
        gap = np.diff(idx)
        rec = (src[idx[1:], k] >= Source.TRACKLET) | (src[idx[:-1], k] >= Source.TRACKLET)
        glitch = (
            np.zeros(len(step), dtype=bool) if off is None else (off[idx[1:], k] | off[idx[:-1], k])
        )
        bad = rec & (step > allowed_displacement(gap, params))
        bad300 = rec & (gap == 1) & (step > 300)
        totals["links"] += int(rec.sum())
        totals["bad"] += int(bad.sum())
        totals["bad300"] += int(bad300.sum())
        totals["own"] += int((bad & ~glitch).sum())
        totals["own300"] += int((bad300 & ~glitch).sum())
    return {
        "n_links_with_recovered": totals["links"],
        "n_violations": totals["bad"],
        "n_gt300_consecutive": totals["bad300"],
        "n_violations_own": totals["own"],
        "n_gt300_consecutive_own": totals["own300"],
        "n_violations_anchor_glitch": totals["bad"] - totals["own"],
    }
