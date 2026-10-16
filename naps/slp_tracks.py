"""Read the tag-node positions of a ``.slp`` file without building ``sleap_io`` objects.

``naps-track`` only needs, for every instance of the frames it processes, the SLEAP track number,
the position of the tag node and a fallback position. ``sio.load_slp`` builds one Python object
per instance and per point array to give that (about a minute for the 1.1 million instances of
an hour of 16 animals). :func:`iter_window_instances` reads the same numbers straight from the HDF5
tables with numpy and yields them in the order of the eager loop::

    for lf in labels.labeled_frames:            # file order
        for instance in lf.instances:           # instance order
            ...

Everything it yields is checked against the eager loader by ``tests/test_slp_tracks.py``. A file
it does not understand makes it raise :class:`UnsupportedSlpError`; the caller then falls back to
the eager loader, so the result never depends on which reader ran.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import h5py
import numpy as np
from numpy.typing import NDArray
from sleap_io.io import slp as sleap_io_slp
from sleap_io.model.instance import Track
from sleap_io.model.skeleton import Skeleton

__all__ = ["InstanceRow", "UnsupportedSlpError", "iter_window_instances"]

logger = logging.getLogger(__name__)

_USER = 0
_PREDICTED = 1
_SEPARATOR = "_"

#: ``(frame_idx, SLEAP track number or None, tag-node row (2,), position (2,) or None)``
InstanceRow: TypeAlias = tuple[int, int | None, NDArray[np.float64], NDArray[np.float64] | None]
PositionFn = Callable[[NDArray[np.float64], int, Sequence[str], str], NDArray[np.float64] | None]


class UnsupportedSlpError(Exception):
    """The file's layout is not one :func:`iter_window_instances` reproduces exactly."""


def _gather_xy(table: np.ndarray, starts: NDArray[np.int64], n_nodes: int) -> NDArray[np.float64]:
    """Return ``(len(starts), n_nodes, 2)`` xy of the points ``starts[i] .. starts[i] + n_nodes``.

    The xy is NaN where the point is not visible (what ``Instance.numpy()`` returns).
    """
    index = starts[:, None] + np.arange(n_nodes)[None, :]
    xy = np.stack([table["x"][index], table["y"][index]], axis=-1).astype(np.float64, copy=False)
    xy[~table["visible"][index]] = np.nan
    return xy


@dataclass
class _Tables:
    """The raw tables of a ``.slp`` file."""

    frames: np.ndarray
    instances: np.ndarray
    points: np.ndarray
    pred_points: np.ndarray
    skeletons: list[Skeleton]
    tracks: list[Track]


def _read_tables(path: str) -> _Tables:
    """Read and sanity-check the tables.

    Raises:
        UnsupportedSlpError: If a table is missing or has an unexpected layout.
    """
    try:
        with h5py.File(path, "r") as f:
            frames = f["frames"][()]
            instances = f["instances"][()]
            points = f["points"][()]
            pred_points = f["pred_points"][()]
        skeletons = sleap_io_slp.read_skeletons(path)
        tracks = sleap_io_slp.read_tracks(path)
    except (KeyError, OSError, ValueError) as exc:
        msg = f"cannot read the tables of {path}: {exc}"
        raise UnsupportedSlpError(msg) from exc
    frame_fields = {"frame_idx", "instance_id_start", "instance_id_end"}
    instance_fields = {"instance_type", "skeleton", "track", "point_id_start", "point_id_end"}
    if not frame_fields <= set(frames.dtype.names or ()) or not instance_fields <= set(
        instances.dtype.names or ()
    ):
        msg = f"unexpected table layout in {path}"
        raise UnsupportedSlpError(msg)
    return _Tables(frames, instances, points, pred_points, skeletons, tracks)


def _window_instances(
    tables: _Tables, start_frame: int, end_frame: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """``(frame_idx, instance row)`` of the instances of the window, in the eager loop's order."""
    frames = tables.frames
    in_window = (frames["frame_idx"] >= start_frame) & (frames["frame_idx"] <= end_frame)
    first = frames["instance_id_start"][in_window].astype(np.int64)
    count = frames["instance_id_end"][in_window].astype(np.int64) - first
    frame_of = np.repeat(frames["frame_idx"][in_window].astype(np.int64), count)
    offsets = np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count)
    return frame_of, np.repeat(first, count) + offsets


def _positions(
    tables: _Tables,
    rows: NDArray[np.int64],
    tag_node_name: str,
    position_node_name: str,
    position_fn: PositionFn,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Tag-node xy, position and "has a position" flag of the instances ``rows``."""
    instances = tables.instances
    n = len(rows)
    skeleton_ids = instances["skeleton"][rows].astype(np.int64)
    kinds = instances["instance_type"][rows]
    if np.any((kinds != _USER) & (kinds != _PREDICTED)):
        msg = "unknown instance type"
        raise UnsupportedSlpError(msg)
    point_start = instances["point_id_start"][rows].astype(np.int64)
    point_stop = instances["point_id_end"][rows].astype(np.int64)

    tag_xy = np.full((n, 2), np.nan)
    position = np.full((n, 2), np.nan)
    has_position = np.zeros(n, dtype=bool)
    for skeleton_id in np.unique(skeleton_ids):
        node_names = tables.skeletons[int(skeleton_id)].node_names
        try:
            tag_idx = node_names.index(tag_node_name)
        except ValueError as exc:
            msg = f"tag node {tag_node_name!r} is not a node of the SLEAP skeleton {node_names}"
            raise ValueError(msg) from exc
        fallback_idx = (
            node_names.index(position_node_name) if position_node_name in node_names else None
        )
        for kind, table in ((_USER, tables.points), (_PREDICTED, tables.pred_points)):
            sel = np.flatnonzero((skeleton_ids == skeleton_id) & (kinds == kind))
            if not len(sel):
                continue
            if np.any(point_stop[sel] - point_start[sel] != len(node_names)):
                msg = "an instance does not have as many points as its skeleton has nodes"
                raise UnsupportedSlpError(msg)
            xy = _gather_xy(table, point_start[sel], len(node_names))
            tag_xy[sel] = xy[:, tag_idx]
            usable = np.isfinite(tag_xy[sel]).all(axis=1)
            position[sel[usable]] = tag_xy[sel[usable]]
            if fallback_idx is not None:
                fallback = xy[:, fallback_idx]
                use_fallback = ~usable & np.isfinite(fallback).all(axis=1)
                position[sel[use_fallback]] = fallback[use_fallback]
                usable |= use_fallback
            has_position[sel[usable]] = True
            for local in np.flatnonzero(~usable):  # rare: neither node is usable
                result = position_fn(xy[local], tag_idx, node_names, position_node_name)
                if result is not None:
                    position[sel[local]] = result
                    has_position[sel[local]] = True
    return tag_xy, position, has_position


def iter_window_instances(  # noqa: PLR0917 (public, like load_sleap_tracks)
    slp_path: str | Path,
    start_frame: int,
    end_frame: int,
    tag_node_name: str,
    position_node_name: str,
    position_fn: PositionFn,
) -> Iterator[InstanceRow]:
    """Yield the instances of frames ``start_frame..end_frame`` as the eager loader sees them.

    Args:
        slp_path: The ``.slp`` file.
        start_frame: First frame index (inclusive).
        end_frame: Last frame index (inclusive).
        tag_node_name: Skeleton node on the ArUco tag.
        position_node_name: Fallback node for the instance position.
        position_fn: ``position_fn(points, tag_index, node_names, fallback_name)``: the position
            of an instance whose tag node *and* fallback are not finite (the mean of its finite
            nodes, or None); called with the instance's ``(n_nodes, 2)`` points.

    Yields:
        ``(frame_idx, track number | None, tag-node xy, position | None)`` per instance.

    Raises:
        UnsupportedSlpError: If the file lacks the tables this reader needs, an instance does not
            have as many points as its skeleton has nodes, or an instance type is unknown.
        ValueError: If the skeleton of an instance in the window has no ``tag_node_name`` node
            (like the eager loader) or a track name does not end in ``_<int>``.
    """
    tables = _read_tables(str(slp_path))
    frame_of, rows = _window_instances(tables, start_frame, end_frame)
    tag_xy, position, has_position = _positions(
        tables, rows, tag_node_name, position_node_name, position_fn
    )
    track_ids = tables.instances["track"][rows].astype(np.int64)

    numbers: dict[int, int] = {}
    for i in range(len(rows)):
        track_id = int(track_ids[i])
        number = None
        if track_id >= 0:
            number = numbers.get(track_id)
            if number is None:
                name = tables.tracks[track_id].name
                number = numbers[track_id] = int(name.split(_SEPARATOR)[-1])
        yield (
            int(frame_of[i]),
            number,
            tag_xy[i],
            position[i] if has_position[i] else None,
        )
