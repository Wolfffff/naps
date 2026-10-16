"""Dense per-frame table of SLEAP instances, the input of the joint identity engine.

The legacy pipeline keys everything by SLEAP track id and never looks at instances without a
track. The joint engine (:mod:`naps.identity`) builds its own tracklets, so it needs *every*
instance of every frame. :class:`InstanceTable` stores them as padded ``(frames, slots, ...)``
arrays; a slot is just the position of the instance in its frame (``NaN`` position = empty).

Tables are normally built from a loaded :class:`sleap_io.Labels` with :func:`table_from_labels`
(the same ``Labels`` object is later handed to the writer, so instance order cannot drift) or
directly from arrays in tests and benchmarks.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import sleap_io as sio
from numpy.typing import NDArray

__all__ = ["InstanceTable", "table_from_labels"]

logger = logging.getLogger(__name__)

# Separator of the numeric suffix in SLEAP track names (``track_0``, ``track_1``, ...).
_TRACK_NAME_SEPARATOR = "_"


@dataclass
class InstanceTable:
    """Padded arrays describing the instances of consecutive frames.

    Attributes:
        first_frame: Video frame number of row 0.
        xy: ``(F, S, 2)`` position used for tracking (body node); ``NaN`` marks an empty slot or an
            instance without any finite node.
        tag_xy: ``(F, S, 2)`` position of the ArUco tag node (where the tag is decoded).
        score: ``(F, S)`` SLEAP instance score (``NaN`` if unknown).
        hint: ``(F, S)`` SLEAP track number of the instance, ``-1`` if it has no track.
        inst_index: ``(F, S)`` index of the instance in ``LabeledFrame.instances``, ``-1`` for an
            empty slot (this is what the writer needs to put labels back on instances).
        heading: ``(F, S)`` body axis angle in radians (abdomen to head), ``NaN`` if unknown.
        pose: Optional ``(F, S, N, 2)`` full pose, used by the ghost test.
    """

    first_frame: int
    xy: NDArray[np.float64]
    tag_xy: NDArray[np.float64]
    score: NDArray[np.float64]
    hint: NDArray[np.int64]
    inst_index: NDArray[np.int64]
    heading: NDArray[np.float64]
    pose: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        """Check that the arrays agree in shape.

        Raises:
            ValueError: If the shapes of the arrays are inconsistent.
        """
        n_frames, n_slots = self.xy.shape[:2]
        shapes = {
            "tag_xy": self.tag_xy.shape == (n_frames, n_slots, 2),
            "score": self.score.shape == (n_frames, n_slots),
            "hint": self.hint.shape == (n_frames, n_slots),
            "inst_index": self.inst_index.shape == (n_frames, n_slots),
            "heading": self.heading.shape == (n_frames, n_slots),
            "pose": self.pose is None or self.pose.shape[:2] == (n_frames, n_slots),
        }
        bad = [name for name, ok in shapes.items() if not ok]
        if bad or self.xy.ndim != 3 or self.xy.shape[2] != 2:
            msg = f"InstanceTable arrays disagree in shape: {bad or ['xy']}"
            raise ValueError(msg)

    @property
    def n_frames(self) -> int:
        """Number of frames (rows)."""
        return int(self.xy.shape[0])

    @property
    def n_slots(self) -> int:
        """Number of slots per frame."""
        return int(self.xy.shape[1])

    @property
    def valid(self) -> NDArray[np.bool_]:
        """``(F, S)`` mask of slots that hold an instance with a position."""
        return np.isfinite(self.xy[..., 0]) & np.isfinite(self.xy[..., 1])

    @property
    def frames(self) -> NDArray[np.int64]:
        """Video frame numbers of the rows."""
        return self.first_frame + np.arange(self.n_frames, dtype=np.int64)

    def subset(self, member: NDArray[np.bool_]) -> tuple[InstanceTable, NDArray[np.int64]]:
        """Keep only the instances flagged in ``member`` and left-pack them.

        Args:
            member: ``(F, S)`` mask of the slots to keep.

        Returns:
            ``(table, source_slot)``: the packed table and, for each of its slots, the slot of
            this table it came from (``-1`` for padding).
        """
        member = member & self.valid
        n_slots = max(int(np.max(member.sum(axis=1), initial=0)), 1)
        # stable sort so the kept slots of a frame come first, in their original order
        order = np.argsort(~member, axis=1, kind="stable")[:, :n_slots]
        keep = np.take_along_axis(member, order, axis=1)
        source = np.where(keep, order, -1).astype(np.int64)

        def pick(array: NDArray, fill: float) -> NDArray:
            """Gather ``array`` (``(F, S, ...)``) at ``order`` and blank the padding."""
            extra = array.ndim - 2
            index = order.reshape(order.shape + (1,) * extra)
            taken = np.take_along_axis(
                array, np.broadcast_to(index, order.shape + array.shape[2:]), 1
            )
            mask = keep.reshape(keep.shape + (1,) * extra)
            return np.where(mask, taken, fill).astype(array.dtype)

        table = InstanceTable(
            first_frame=self.first_frame,
            xy=pick(self.xy, np.nan),
            tag_xy=pick(self.tag_xy, np.nan),
            score=pick(self.score, np.nan),
            hint=pick(self.hint, -1),
            inst_index=pick(self.inst_index, -1),
            heading=pick(self.heading, np.nan),
            pose=None if self.pose is None else pick(self.pose, np.nan),
        )
        return table, source


def _node_index(names: Sequence[str], wanted: str | None) -> int | None:
    """Index of the node called ``wanted`` (exact match first, then case-insensitive)."""
    if wanted is None:
        return None
    if wanted in names:
        return list(names).index(wanted)
    lowered = [n.lower() for n in names]
    return lowered.index(wanted.lower()) if wanted.lower() in lowered else None


def _body_position(
    points: NDArray[np.float64], position_idx: int | None, tag_idx: int | None
) -> NDArray[np.float64]:
    """Position node, else the tag node, else the mean of the finite nodes (``NaN`` if none)."""
    for idx in (position_idx, tag_idx):
        if idx is not None and np.isfinite(points[idx]).all():
            return np.asarray(points[idx], dtype=float)
    finite = points[np.isfinite(points).all(axis=1)]
    return finite.mean(axis=0) if len(finite) else np.full(2, np.nan)


def _heading(points: NDArray[np.float64], head_idx: int | None, tail_idx: int | None) -> float:
    """Angle (radians) of the vector from the tail node to the head node, ``NaN`` if unknown."""
    if head_idx is None or tail_idx is None:
        return float("nan")
    head, tail = points[head_idx], points[tail_idx]
    if not (np.isfinite(head).all() and np.isfinite(tail).all()):
        return float("nan")
    return float(np.arctan2(head[1] - tail[1], head[0] - tail[0]))


def _track_hint(track: sio.Track | None, labels: sio.Labels) -> int:
    """SLEAP track number (``track_<n>``, else the index in ``labels.tracks``), -1 if none."""
    if track is None:
        return -1
    try:
        return int(track.name.split(_TRACK_NAME_SEPARATOR)[-1])
    except ValueError:
        return labels.tracks.index(track) if track in labels.tracks else -1


def table_from_labels(
    labels: sio.Labels,
    first_frame: int,
    last_frame: int,
    tag_node: str,
    *,
    position_node: str | None = "Thorax",
    head_node: str | None = "head",
    tail_node: str | None = "abdomen",
    keep_pose: bool = True,
) -> InstanceTable:
    """Build the :class:`InstanceTable` of frames ``first_frame..last_frame`` (inclusive).

    Every instance of a labeled frame gets a slot, tracked or not. Frames without labels are
    empty rows.

    Args:
        labels: The loaded SLEAP labels.
        first_frame: First frame (inclusive).
        last_frame: Last frame (inclusive).
        tag_node: Name of the skeleton node on the ArUco tag.
        position_node: Node used as the body position (case-insensitive); falls back to the tag
            node, then to the mean of the finite nodes.
        head_node: Node for the heading (with ``tail_node``); ignored if the skeleton lacks it.
        tail_node: See ``head_node``.
        keep_pose: Also store the full pose (needed by the ghost test).

    Returns:
        The table. ``score`` is the SLEAP instance score, ``NaN`` for instances without one.

    Raises:
        ValueError: If the skeleton has no node called ``tag_node``.
    """
    n_frames = last_frame - first_frame + 1
    per_frame: dict[int, list[tuple[int, sio.Instance | sio.PredictedInstance]]] = {}
    for lf in labels.labeled_frames:
        if first_frame <= lf.frame_idx <= last_frame:
            per_frame.setdefault(lf.frame_idx, []).extend(enumerate(lf.instances))
    n_slots = max((len(v) for v in per_frame.values()), default=1)
    n_slots = max(n_slots, 1)
    n_nodes = len(labels.skeletons[0].nodes) if labels.skeletons else 0

    xy = np.full((n_frames, n_slots, 2), np.nan)
    tag_xy = np.full((n_frames, n_slots, 2), np.nan)
    score = np.full((n_frames, n_slots), np.nan)
    hint = np.full((n_frames, n_slots), -1, dtype=np.int64)
    inst_index = np.full((n_frames, n_slots), -1, dtype=np.int64)
    heading = np.full((n_frames, n_slots), np.nan)
    pose = np.full((n_frames, n_slots, n_nodes, 2), np.nan, dtype=np.float32) if keep_pose else None

    for frame, entries in per_frame.items():
        row = frame - first_frame
        for slot, (index, inst) in enumerate(entries):
            names = inst.skeleton.node_names
            tag_idx = _node_index(names, tag_node)
            if tag_idx is None:
                msg = f"tag node {tag_node!r} is not a node of the SLEAP skeleton {list(names)}"
                raise ValueError(msg)
            points = np.asarray(inst.numpy(), dtype=float)[:, :2]
            pos_idx = _node_index(names, position_node)
            xy[row, slot] = _body_position(points, pos_idx, tag_idx)
            tag_xy[row, slot] = points[tag_idx]
            raw_score = getattr(inst, "score", None)
            score[row, slot] = np.nan if raw_score is None else float(raw_score)
            hint[row, slot] = _track_hint(inst.track, labels)
            inst_index[row, slot] = index
            heading[row, slot] = _heading(
                points, _node_index(names, head_node), _node_index(names, tail_node)
            )
            if pose is not None:
                pose[row, slot, : len(points)] = points
    return InstanceTable(first_frame, xy, tag_xy, score, hint, inst_index, heading, pose)
