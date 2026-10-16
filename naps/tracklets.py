"""Tracklets: short, high-purity position tracks built from SLEAP instances.

The legacy assignment keys identity on the ``track_N`` ids written by SLEAP's tracker and is only
as good as that tracker. The joint engine links the instances of consecutive frames itself:

* frame-to-frame Hungarian linking on the body position, with a constant-velocity prediction, a
  distance gate that grows with the number of bridged frames, and gap bridging;
* **competing-instance breaks**: when a second instance (or a second tracklet) is almost as close
  as the chosen link, the link is *not* made, so a near-pass cannot silently swap two bees;
* optional **pose-aware** terms (heading continuity) and SLEAP track ids as a **hint**: ids make
  a link cheaper when they agree and dearer when they disagree, but never lift the distance
  gate, so a track id that jumps across the arena is split, not trusted;
* ``mode="sleap"`` follows the SLEAP ids strictly: a link needs *equal* ids on both sides, so an
  instance without a SLEAP track (hint -1) never continues a tracklet and opens a new one in every
  frame (an untracked ``.slp`` therefore gives only single-frame tracklets in this mode; use
  ``mode="built"`` there). It still splits at implausible jumps or long gaps.

The defaults reproduce the research code that was validated on held-out NAPS data (see
:class:`TrackletParams`). Everything is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

__all__ = ["TrackletParams", "Tracklets", "build_tracklets", "dedupe_instances"]

TrackletMode = Literal["built", "sleap"]

# A cost no real link can have; pairs at this cost are ignored after the assignment.
_FORBIDDEN = 1e6


@dataclass(frozen=True)
class TrackletParams:
    """Parameters of :func:`build_tracklets` (defaults: the validated research values).

    Attributes:
        gate0: Distance gate (px) between the predicted and the observed position of a track that
            was seen in the previous frame.
        gate_grow: The gate grows by this many px per additionally bridged frame.
        gate_max: Upper limit of the gate.
        gap_max: Longest frame distance between two instances of one tracklet: a tracklet is
            closed when it was not seen for more than this many frames, so at most ``gap_max - 1``
            consecutive frames are missing inside it.
        amb_ratio: Competing-instance break: a link of length ``d`` is refused if another
            candidate lies within ``amb_ratio * d + amb_add`` of the same track or instance. The
            refused link is not made *and the track is closed* (the instance opens a new
            tracklet), so the identity of a bee cannot be carried across a near-pass.
        amb_add: See ``amb_ratio``.
        vel_clip: Maximum speed (px/frame) used by the constant-velocity prediction.
        vel_damp: Smoothing of the velocity estimate (0 = last step only, 1 = never updates).
        hint_bonus: Link cost factor when the SLEAP ids of track and instance agree.
        hint_penalty: Link cost factor when both have an id and the ids differ.
        pose_weight: Extra cost (px per radian) of a heading change; 0 switches the pose term off.
        pose_max_turn: Heading change (radians, absolute, not scaled by the number of bridged
            frames) above which a link is refused; ``None`` = no limit. Duplicates are removed
            before linking by :func:`dedupe_instances` (``IdentityParams.dedupe_px``).
    """

    gate0: float = 45.0
    gate_grow: float = 12.0
    gate_max: float = 200.0
    gap_max: int = 20
    amb_ratio: float = 2.0
    amb_add: float = 30.0
    vel_clip: float = 20.0
    vel_damp: float = 0.5
    hint_bonus: float = 0.5
    hint_penalty: float = 1.5
    pose_weight: float = 0.0
    pose_max_turn: float | None = None


@dataclass
class Tracklets:
    """Result of :func:`build_tracklets`.

    Attributes:
        tid: ``(F, S)`` tracklet id of every slot, ``-1`` for an empty or rejected slot.
        n_tracklets: Number of tracklets (ids are ``0..n_tracklets-1``).
        first: ``(n_tracklets,)`` first frame row of each tracklet.
        last: ``(n_tracklets,)`` last frame row of each tracklet.
        length: ``(n_tracklets,)`` number of instances of each tracklet.
    """

    tid: NDArray[np.int32]
    n_tracklets: int = field(init=False)
    first: NDArray[np.int64] = field(init=False)
    last: NDArray[np.int64] = field(init=False)
    length: NDArray[np.int64] = field(init=False)

    def __post_init__(self) -> None:
        """Derive the per-tracklet extents from ``tid``."""
        frames, slots = np.nonzero(self.tid >= 0)
        ids = self.tid[frames, slots].astype(np.int64)
        self.n_tracklets = int(ids.max()) + 1 if len(ids) else 0
        self.first = np.full(self.n_tracklets, self.tid.shape[0], dtype=np.int64)
        self.last = np.full(self.n_tracklets, -1, dtype=np.int64)
        np.minimum.at(self.first, ids, frames)
        np.maximum.at(self.last, ids, frames)
        self.length = np.bincount(ids, minlength=self.n_tracklets).astype(np.int64)


@dataclass
class _Track:
    """State of an open tracklet while linking."""

    frame: int
    pos: NDArray[np.float64]
    vel: NDArray[np.float64]
    hint: int
    heading: float


def dedupe_instances(
    xy: NDArray[np.float64], score: NDArray[np.float64], min_distance: float
) -> NDArray[np.bool_]:
    """Mask of instances that are not a near-duplicate of a higher-scored instance.

    Args:
        xy: ``(F, S, 2)`` positions, ``NaN`` for empty slots.
        score: ``(F, S)`` instance scores (``NaN`` counts as 0). Ties keep the lower slot.
        min_distance: Instances closer than this (px) are duplicates.

    Returns:
        ``(F, S)`` boolean mask of the instances to keep (finite position, not a duplicate).
    """
    keep = np.isfinite(xy[..., 0]) & np.isfinite(xy[..., 1])
    ranking = np.where(np.isfinite(score), score, 0.0)
    for frame in np.flatnonzero(keep.sum(axis=1) >= 2):
        slots = np.flatnonzero(keep[frame])
        # visit the best-scored first; each later one is dropped if it is too close to a kept one
        ordered = sorted(map(int, slots), key=lambda s: (-ranking[frame, s], s))
        kept: list[int] = []
        for slot in ordered:
            distances = [float(np.linalg.norm(xy[frame, slot] - xy[frame, k])) for k in kept]
            if distances and min(distances) < min_distance:
                keep[frame, slot] = False
            else:
                kept.append(slot)
    return keep


def _heading_change(a: float, b: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """Absolute wrapped angle difference between ``a`` and ``b`` (radians, 0..pi)."""
    diff = np.abs(np.asarray(b, dtype=float) - a) % (2 * np.pi)
    return np.minimum(diff, 2 * np.pi - diff)


class _Linker:
    """Frame-to-frame linker holding the open tracks (one instance per ``build_tracklets``)."""

    def __init__(self, params: TrackletParams, mode: TrackletMode) -> None:
        self.params = params
        self.mode = mode
        self.tracks: dict[int, _Track] = {}
        self.next_id = 0

    def expire(self, frame: int) -> None:
        """Close the tracks that were not seen for more than ``gap_max`` frames."""
        gone = [k for k, t in self.tracks.items() if frame - t.frame > self.params.gap_max]
        for key in gone:
            del self.tracks[key]

    def _costs(
        self,
        frame: int,
        active: list[int],
        pos: NDArray[np.float64],
        hint: NDArray[np.int64],
        heading: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
        """Return link costs, feasibility mask and raw distances (tracks x instances)."""
        p = self.params
        tracks = [self.tracks[k] for k in active]
        gaps = np.array([frame - t.frame for t in tracks])
        predicted = np.array([t.pos + t.vel * (frame - t.frame) for t in tracks])
        gate = np.minimum(p.gate0 + p.gate_grow * (gaps - 1), p.gate_max)
        dist = np.linalg.norm(predicted[:, None, :] - pos[None, :, :], axis=-1)
        feasible = dist <= gate[:, None]
        cost = dist.copy()
        track_hint = np.array([t.hint for t in tracks])
        both = (track_hint[:, None] >= 0) & (hint[None, :] >= 0)
        same = both & (track_hint[:, None] == hint[None, :])
        differ = both & ~same
        if self.mode == "sleap":
            feasible &= same
        else:
            cost = np.where(same, cost * p.hint_bonus, cost)
            cost = np.where(differ, cost * p.hint_penalty, cost)
        if p.pose_weight > 0 or p.pose_max_turn is not None:
            turn = np.array(
                [_heading_change(t.heading, heading) for t in tracks], dtype=float
            ).reshape(len(tracks), len(heading))
            known = np.isfinite(turn)
            if p.pose_weight > 0:
                cost = cost + np.where(known, p.pose_weight * np.nan_to_num(turn), 0.0)
            if p.pose_max_turn is not None:
                feasible &= ~known | (turn <= p.pose_max_turn)
        return cost, feasible, dist

    def link(
        self,
        frame: int,
        slots: NDArray[np.int64],
        pos: NDArray[np.float64],
        hint: NDArray[np.int64],
        heading: NDArray[np.float64],
    ) -> dict[int, int]:
        """Link one frame's instances to open tracks; returns ``{index in slots: track id}``."""
        active = [k for k in sorted(self.tracks) if frame - self.tracks[k].frame >= 1]
        if len(slots) == 0 or not active:
            return {}
        p = self.params
        cost, feasible, dist = self._costs(frame, active, pos, hint, heading)
        rows, cols = linear_sum_assignment(np.where(feasible, cost, _FORBIDDEN))
        matched: dict[int, int] = {}
        for i, j in zip(rows, cols, strict=True):
            if not feasible[i, j]:
                continue
            threshold = p.amb_ratio * dist[i, j] + p.amb_add
            rival_instance = feasible[i].copy()
            rival_instance[j] = False
            rival_track = feasible[:, j].copy()
            rival_track[i] = False
            ambiguous = bool(
                (rival_instance & (dist[i] < threshold)).any()
                or (rival_track & (dist[:, j] < threshold)).any()
            )
            if ambiguous:
                # break: the identity of this instance cannot be carried over
                del self.tracks[active[i]]
                continue
            matched[int(j)] = active[i]
        return matched

    def update(
        self,
        frame: int,
        track_id: int | None,
        pos: NDArray[np.float64],
        hint: int,
        heading: float,
    ) -> int:
        """Extend ``track_id`` with the instance at ``pos`` (or open a new track); return its id."""
        p = self.params
        if track_id is None:
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = _Track(frame, pos.copy(), np.zeros(2), hint, heading)
            return track_id
        track = self.tracks[track_id]
        step = (pos - track.pos) / (frame - track.frame)
        speed = float(np.linalg.norm(step))
        if speed > p.vel_clip:
            step = step * p.vel_clip / speed
        track.vel = p.vel_damp * track.vel + (1 - p.vel_damp) * step
        track.pos = pos.copy()
        track.frame = frame
        track.hint = hint if hint >= 0 else track.hint
        track.heading = heading if np.isfinite(heading) else track.heading
        return track_id


def build_tracklets(
    xy: NDArray[np.float64],
    valid: NDArray[np.bool_],
    params: TrackletParams | None = None,
    *,
    hint: NDArray[np.int64] | None = None,
    heading: NDArray[np.float64] | None = None,
    mode: TrackletMode = "built",
) -> Tracklets:
    """Link the instances of consecutive frames into tracklets.

    Args:
        xy: ``(F, S, 2)`` body positions.
        valid: ``(F, S)`` instances to link (e.g. after :func:`dedupe_instances`); the others, and
            instances whose position is not finite, get tracklet id ``-1``.
        params: Linking parameters (default: :class:`TrackletParams`).
        hint: Optional ``(F, S)`` SLEAP track numbers (``-1`` = none), used as a soft cost
            (``mode="built"``) or as a hard condition (``mode="sleap"``).
        heading: Optional ``(F, S)`` body axis angles for the pose-aware terms.
        mode: ``"built"`` links by geometry (ids only bias it); ``"sleap"`` follows the SLEAP ids
            and only splits them at jumps and gaps.

    Returns:
        The :class:`Tracklets`. Every slot is in at most one tracklet and no tracklet has two
        instances in a frame.

    Raises:
        ValueError: If ``mode`` is unknown or the array shapes disagree.
    """
    if mode not in ("built", "sleap"):
        msg = f"unknown tracklet mode {mode!r}; use 'built' or 'sleap'"
        raise ValueError(msg)
    n_frames, n_slots = valid.shape
    expected = {"xy": (n_frames, n_slots, 2), "hint": (n_frames, n_slots)}
    expected["heading"] = expected["hint"]
    for name, array in (("xy", xy), ("hint", hint), ("heading", heading)):
        if array is not None and array.shape != expected[name]:
            msg = f"{name} has shape {array.shape}, expected {expected[name]}"
            raise ValueError(msg)
    # an instance without a finite position cannot be linked: it gets no tracklet
    valid = valid & np.isfinite(xy[..., 0]) & np.isfinite(xy[..., 1])
    hint = np.full((n_frames, n_slots), -1, dtype=np.int64) if hint is None else hint
    heading = np.full((n_frames, n_slots), np.nan) if heading is None else heading
    linker = _Linker(params or TrackletParams(), mode)
    tid = np.full((n_frames, n_slots), -1, dtype=np.int32)
    for frame in range(n_frames):
        slots = np.flatnonzero(valid[frame])
        linker.expire(frame)
        matched = linker.link(
            frame, slots, xy[frame, slots], hint[frame, slots], heading[frame, slots]
        )
        for j, slot in enumerate(slots):
            tid[frame, slot] = linker.update(
                frame,
                matched.get(j),
                xy[frame, slot],
                int(hint[frame, slot]),
                float(heading[frame, slot]),
            )
    return Tracklets(tid)
