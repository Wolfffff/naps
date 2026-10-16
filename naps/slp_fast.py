"""Faster ``.slp`` writing for the re-tracked output of ``naps-track``.

``sleap_io.io.slp.write_lfs`` (sleap-io 0.9.x) writes the ``frames``, ``instances``, ``points`` and
``pred_points`` datasets with a Python loop that appends one list per *point*
(about 20 per instance, 1.1 million instances in an hour of 16 animals). :func:`write_lfs_fast`
produces exactly the same four datasets (same dtypes, same row order, same values) from whole
arrays, an order of magnitude faster. :func:`save_labels` saves a :class:`sleap_io.Labels` with it
by swapping the module function for the duration of the call, and falls back to the stock writer
if the installed sleap-io is not a version this replacement was verified against.

The replacement is a mirror of one private function, so it is guarded twice: by the version check
and by ``tests/test_slp_fast.py``, which compares every dataset of a file written both ways.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Generator
from typing import Any

import h5py
import numpy as np
import sleap_io as sio
from sleap_io.io import slp as sleap_io_slp
from sleap_io.model.instance import Instance, PredictedInstance

__all__ = ["VERIFIED_SLEAP_IO", "fast_writer_supported", "save_labels", "write_lfs_fast"]

logger = logging.getLogger(__name__)

#: sleap-io minor versions whose ``write_lfs`` layout this module reproduces.
VERIFIED_SLEAP_IO = ("0.9",)

_INSTANCE_DTYPE = np.dtype(
    [
        ("instance_id", "i8"),
        ("instance_type", "u1"),
        ("frame_id", "u8"),
        ("skeleton", "u4"),
        ("track", "i4"),
        ("from_predicted", "i8"),
        ("score", "f4"),
        ("point_id_start", "u8"),
        ("point_id_end", "u8"),
        ("tracking_score", "f4"),
    ]
)
_FRAME_DTYPE = np.dtype(
    [
        ("frame_id", "u8"),
        ("video", "u4"),
        ("frame_idx", "u8"),
        ("instance_id_start", "u8"),
        ("instance_id_end", "u8"),
    ]
)
_POINT_DTYPE = np.dtype([("x", "f8"), ("y", "f8"), ("visible", "?"), ("complete", "?")])
_PRED_POINT_DTYPE = np.dtype(
    [("x", "f8"), ("y", "f8"), ("visible", "?"), ("complete", "?"), ("score", "f8")]
)
_USER = 0
_PREDICTED = 1
_NO_LINK = -1


def fast_writer_supported() -> bool:
    """Whether the installed sleap-io is one this module's writer was verified against."""
    return str(sio.__version__).startswith(VERIFIED_SLEAP_IO) and hasattr(sleap_io_slp, "write_lfs")


def _video_ids(labels: sio.Labels) -> dict[int, int]:
    """``{video index: video id}``; embedded videos keep their sparse id (``video15/video``)."""
    ids = {}
    for video_idx, video in enumerate(labels.videos):
        ids[video_idx] = video_idx
        dataset = getattr(getattr(video, "backend", None), "dataset", None)
        if dataset is not None:
            group = dataset.split("/")[0]
            if group.startswith("video"):
                with contextlib.suppress(ValueError):
                    ids[video_idx] = int(group[5:])
    return ids


def write_lfs_fast(  # noqa: PLR0915 (one pass mirroring the single upstream function it replaces)
    labels_path: str | os.PathLike[str], labels: sio.Labels
) -> None:
    """Write the frame, instance and point datasets of ``labels`` (drop-in for ``write_lfs``).

    Args:
        labels_path: The ``.slp`` file to append the datasets to (it must already exist).
        labels: The labels to write.

    Raises:
        ValueError: If an instance is neither a user nor a predicted instance.
    """
    video_ids = _video_ids(labels)
    video_slot: dict[int, int] = {}
    skeleton_cache: dict[int, int] = {}
    track_cache: dict[int, int] = {}

    frames: list[tuple[int, int, int, int, int]] = []
    rows: list[tuple[Any, ...]] = []
    user_points: list[Any] = []
    pred_points: list[Any] = []
    n_user = n_pred = 0
    links: list[tuple[int, Instance]] = []
    ids_of: dict[int, int] = {}
    n_instances = 0

    for frame_id, lf in enumerate(labels):
        start = n_instances
        for inst in lf:
            skeleton = skeleton_cache.get(id(inst.skeleton))
            if skeleton is None:
                skeleton = skeleton_cache[id(inst.skeleton)] = labels.skeletons.index(inst.skeleton)
            if inst.track:
                track = track_cache.get(id(inst.track))
                if track is None:
                    track = track_cache[id(inst.track)] = labels.tracks.index(inst.track)
            else:
                track = -1
            ids_of[id(inst)] = n_instances
            if inst.from_predicted:
                links.append((n_instances, inst.from_predicted))
            tracking = np.nan if inst.tracking_score is None else inst.tracking_score
            n_points = len(inst.points)
            if type(inst) is Instance:
                rows.append(
                    (n_instances, _USER, frame_id, skeleton, track, _NO_LINK, 0.0,
                     n_user, n_user + n_points, tracking)
                )  # fmt: skip
                user_points.append(inst.points)
                n_user += n_points
            elif type(inst) is PredictedInstance:
                rows.append(
                    (n_instances, _PREDICTED, frame_id, skeleton, track, _NO_LINK, inst.score,
                     n_pred, n_pred + n_points, tracking)
                )  # fmt: skip
                pred_points.append(inst.points)
                n_pred += n_points
            else:
                msg = f"Unknown instance type: {type(inst)}"
                raise ValueError(msg)
            n_instances += 1
        video = video_slot.get(id(lf.video))
        if video is None:
            video = video_slot[id(lf.video)] = labels.videos.index(lf.video)
        frames.append((frame_id, video_ids[video], lf.frame_idx, start, n_instances))

    instances = np.array(rows, dtype=_INSTANCE_DTYPE) if rows else np.empty(0, _INSTANCE_DTYPE)
    for instance_id, source in links:
        # the source instance may have been removed from the labels: drop the link
        instances["from_predicted"][instance_id] = ids_of.get(id(source), _NO_LINK)
    frame_array = np.array(frames, dtype=_FRAME_DTYPE) if frames else np.empty(0, _FRAME_DTYPE)

    with h5py.File(labels_path, "a") as f:
        f.create_dataset("points", data=_concat(user_points, _POINT_DTYPE, scored=False))
        f.create_dataset("pred_points", data=_concat(pred_points, _PRED_POINT_DTYPE, scored=True))
        f.create_dataset("instances", data=instances, dtype=instances.dtype)
        f.create_dataset("frames", data=frame_array, dtype=frame_array.dtype)


def _concat(arrays: list[Any], dtype: np.dtype[Any], *, scored: bool) -> np.ndarray:
    """Stack the per-instance point arrays into one array of ``dtype`` (``x, y, visible, ...``)."""
    out = np.empty(sum(len(a) for a in arrays), dtype=dtype)
    if not arrays:
        return out
    stacked = np.concatenate(arrays)
    out["x"] = stacked["xy"][:, 0]
    out["y"] = stacked["xy"][:, 1]
    out["visible"] = stacked["visible"]
    out["complete"] = stacked["complete"]
    if scored:
        out["score"] = stacked["score"]
    return out


@contextlib.contextmanager
def _fast_writer() -> Generator[None]:
    """Temporarily replace sleap-io's ``write_lfs`` by :func:`write_lfs_fast`."""
    original = sleap_io_slp.write_lfs
    sleap_io_slp.write_lfs = write_lfs_fast  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        sleap_io_slp.write_lfs = original


def save_labels(labels: sio.Labels, path: str | os.PathLike[str], *, fast: bool = True) -> None:
    """Save ``labels`` to ``path`` (like ``labels.save``), using the fast frame writer if possible.

    Args:
        labels: The labels to save.
        path: Output ``.slp`` path.
        fast: Use :func:`write_lfs_fast` (default). Ignored, with a log line, when the installed
            sleap-io is not a verified version, or when ``labels`` is lazy (sleap-io then has its
            own array-based writer).
    """
    if fast and (labels.is_lazy or not fast_writer_supported()):
        logger.info("Using the stock sleap-io writer (lazy labels or unverified sleap-io version)")
        fast = False
    if fast:
        with _fast_writer():
            labels.save(str(path))
    else:
        labels.save(str(path))
