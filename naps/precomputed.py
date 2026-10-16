"""Precomputed tag decodes: a :class:`~naps.matching.DecodeSource` that never reads the video.

A GPU pipeline that decodes each hour-video once (NVDEC) can run SLEAP, cut the tag crops around the
SLEAP tag nodes and decode the tags in the same pass. It stores the SLEAP ``.slp`` and a *decodes
file* with the votes of every instance. :class:`PrecomputedDecodeSource` serves these votes to
``naps-track --decodes`` (both identity engines), so NAPS does not re-read the video.

The decodes file is a plain ``.npz`` (no pickled objects). Instance ``j`` of a frame is the ``j``-th
instance of the frame in the matching ``.slp`` (``LabeledFrame.instances[j]``, the slot of the joint
engine). Arrays (``F`` frames, ``S`` instance columns, ``V`` votes):

======================  ==========  =============================================================
``schema_version``      ``()``      ``1``
``frame_size``          ``(2,)``    ``(width, height)`` of the video frames
``frame_idx``           ``(F,)``    video frame number of each row; consecutive
``n_inst``              ``(F,)``    number of instances of the frame
``tag_xy``              ``(F,S,2)`` tag-node position the crop was cut around (``NaN``: none)
``vote_frame``          ``(V,)``    video frame number of each vote
``vote_inst``           ``(V,)``    instance ``j`` of the vote
``vote_tag``            ``(V,)``    decoded tag id (several votes per instance are allowed)
======================  ==========  =============================================================

Instances without a vote (no tag node, or nothing decoded) get ``[None]``, exactly as the video
decoders do. Further arrays (``video``, ``vote_conf``, ``vote_dist``, ``inst_slot``, ``inst_state``,
``crop_origin``, ``region``, ``region_names``, ``region_tags``, ``meta``) are documentation and
diagnostics; the source ignores them except ``video`` (compared with the video path, warning only).

The votes are only valid for the ``.slp`` they were computed with. The source therefore refuses
frame ranges the file does not cover, instance counts that differ from the requested ``coords``
and tag-node positions that differ from the stored ones.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from naps.matching import Decodes

__all__ = ["SCHEMA_VERSION", "DecodesError", "PrecomputedDecodeSource"]

logger = logging.getLogger(__name__)

#: The decodes-file layout this module reads.
SCHEMA_VERSION = 1

# Tag-node positions (pixels) may differ by this much between the file (float32) and the request.
_POSITION_TOLERANCE = 1e-3
# A mismatching run reports at most this many frames (the first ones).
_MAX_PROBLEMS = 5
_REQUIRED = (
    "schema_version",
    "frame_size",
    "frame_idx",
    "n_inst",
    "tag_xy",
    "vote_frame",
    "vote_inst",
    "vote_tag",
)


class DecodesError(ValueError):
    """The decodes file is unusable, or does not match the SLEAP file / frame range of the run."""


class PrecomputedDecodeSource:
    """:class:`~naps.matching.DecodeSource` backed by a decodes file (see the module docstring).

    Args:
        path: The ``.npz`` decodes file.
        video: Path of the video the run is for. Only its file name is compared with the one
            stored in the file (a mismatch logs a warning); the video is never opened.
        keys: Optional ``keys[frame][key] -> j`` mapping the keys of ``coords`` to instance
            indices. Without it the keys must be the instance indices ``0..n_inst-1`` of the
            frame (the joint engine's slots). The legacy matcher keys by SLEAP track id and
            passes the instances it tracked.

    Attributes:
        frame_size: ``(width, height)`` of the video frames.
        first_frame: First frame the file covers.
        last_frame: Last frame the file covers (inclusive).

    Raises:
        DecodesError: If the file lacks required arrays, has another schema version, or its
            arrays are inconsistent.
    """

    def __init__(
        self,
        path: str | Path,
        video: str | Path | None = None,
        keys: Mapping[int, Mapping[Hashable, int]] | None = None,
    ) -> None:
        """Load and validate the file."""
        self.path = Path(path)
        self.keys = keys
        with np.load(self.path, allow_pickle=False) as data:
            missing = [k for k in _REQUIRED if k not in data.files]
            if missing:
                msg = f"{self.path}: not a decodes file (missing arrays {missing})"
                raise DecodesError(msg)
            version = int(data["schema_version"])
            if version != SCHEMA_VERSION:
                msg = (
                    f"{self.path}: decodes schema version {version}, "
                    f"this NAPS reads {SCHEMA_VERSION}"
                )
                raise DecodesError(msg)
            self._frames: NDArray[np.int64] = np.asarray(data["frame_idx"], dtype=np.int64)
            self._n_inst: NDArray[np.int64] = np.asarray(data["n_inst"], dtype=np.int64)
            self._tag_xy: NDArray[np.float64] = np.asarray(data["tag_xy"], dtype=np.float64)
            vote_frame = np.asarray(data["vote_frame"], dtype=np.int64)
            vote_inst = np.asarray(data["vote_inst"], dtype=np.int64)
            vote_tag = np.asarray(data["vote_tag"], dtype=np.int64)
            width, height = (int(v) for v in data["frame_size"])
            stored_video = str(data["video"]) if "video" in data.files else ""
        self.frame_size = (width, height)
        self._check_arrays(vote_frame, vote_inst, vote_tag)
        self.first_frame = int(self._frames[0])
        self.last_frame = int(self._frames[-1])
        if video is not None and stored_video and Path(stored_video).name != Path(video).name:
            logger.warning(
                "decodes %s were computed from video %s, but the run uses %s",
                self.path, Path(stored_video).name, Path(video).name,
            )  # fmt: skip
        # votes grouped per frame once: the votes of row r are [_starts[r], _stops[r])
        order = np.argsort(vote_frame, kind="stable")
        self._vote_inst = vote_inst[order]
        self._vote_tag = vote_tag[order]
        self._starts = np.searchsorted(vote_frame[order], self._frames, side="left")
        self._stops = np.searchsorted(vote_frame[order], self._frames, side="right")

    def _check_arrays(
        self,
        vote_frame: NDArray[np.int64],
        vote_inst: NDArray[np.int64],
        vote_tag: NDArray[np.int64],
    ) -> None:
        """Check that the loaded arrays agree in shape and describe consecutive frames.

        Raises:
            DecodesError: On any inconsistency.
        """
        n = self._frames.shape[0]
        if n == 0:
            msg = f"{self.path}: the decodes file holds no frames"
            raise DecodesError(msg)
        if not (np.diff(self._frames) == 1).all():
            msg = f"{self.path}: frame_idx is not a run of consecutive frames"
            raise DecodesError(msg)
        if self._n_inst.shape != (n,) or self._tag_xy.ndim != 3 or self._tag_xy.shape[0] != n:
            msg = f"{self.path}: n_inst / tag_xy do not match frame_idx ({n} frames)"
            raise DecodesError(msg)
        if np.max(self._n_inst) > self._tag_xy.shape[1]:
            msg = f"{self.path}: n_inst exceeds the {self._tag_xy.shape[1]} instance columns"
            raise DecodesError(msg)
        if not vote_frame.shape == vote_inst.shape == vote_tag.shape:
            msg = f"{self.path}: the vote arrays differ in length"
            raise DecodesError(msg)
        rows = vote_frame - int(self._frames[0])
        if len(rows) and (
            np.min(rows) < 0 or np.max(rows) >= n or (vote_inst >= self._n_inst[rows]).any()
        ):
            msg = f"{self.path}: a vote refers to a frame or instance that does not exist"
            raise DecodesError(msg)

    def decode(
        self,
        first_frame: int,
        last_frame: int,
        coords: Mapping[int, Mapping[Hashable, Sequence[float]]],
    ) -> tuple[Decodes, tuple[int, int] | None]:
        """Serve the stored votes of frames ``first_frame..last_frame`` (see ``DecodeSource``).

        Args:
            first_frame: First frame (inclusive).
            last_frame: Last frame (inclusive).
            coords: ``coords[frame][key] -> (x, y)`` tag nodes of the instances to serve; checked
                against the stored positions.

        Returns:
            ``(decodes, frame_size)``; ``decodes[frame][key]`` lists the stored tags, or
            ``[None]`` if the instance has none.

        Raises:
            DecodesError: If the file does not cover the frame range, a frame of ``coords`` has a
                different number of instances than the file, or a tag node differs.
        """
        if first_frame < self.first_frame or last_frame > self.last_frame:
            msg = (
                f"{self.path} covers frames {self.first_frame}..{self.last_frame} but the run "
                f"needs {first_frame}..{last_frame}"
            )
            raise DecodesError(msg)
        decodes: Decodes = {}
        problems: list[str] = []
        for frame in range(first_frame, last_frame + 1):
            row = frame - self.first_frame
            wanted = coords[frame]
            keys = list(wanted)
            insts = self._instances(frame, keys, row, problems)
            if insts is not None and keys:
                asked = np.asarray([wanted[k] for k in keys], dtype=np.float64).reshape(-1, 2)
                self._check_positions(frame, keys, asked, self._tag_xy[row, insts], problems)
            if len(problems) >= _MAX_PROBLEMS:
                break
            if insts is None:
                continue
            lo, hi = int(self._starts[row]), int(self._stops[row])
            voted: defaultdict[int, list[int]] = defaultdict(list)
            for inst, tag in zip(
                self._vote_inst[lo:hi].tolist(), self._vote_tag[lo:hi].tolist(), strict=True
            ):
                voted[inst].append(tag)
            per_key: defaultdict[Hashable, list[Hashable | None]] = defaultdict(list)
            for key, inst in zip(keys, insts.tolist(), strict=True):
                per_key[key].extend(voted.get(inst) or [None])
            decodes[frame] = per_key
        if problems:
            msg = f"{self.path} does not match the SLEAP file: " + "; ".join(problems)
            raise DecodesError(msg)
        return decodes, self.frame_size

    def _instances(
        self, frame: int, keys: list[Hashable], row: int, problems: list[str]
    ) -> NDArray[np.int64] | None:
        """Instance index of every key in ``frame`` (``None``: unmappable, problem noted)."""
        n_inst = int(self._n_inst[row])
        if self.keys is None:
            if set(keys) != set(range(n_inst)):
                problems.append(
                    f"frame {frame}: the SLEAP file has {len(keys)} instances, the decodes {n_inst}"
                )
                return None
            return np.asarray(cast("list[int]", keys), dtype=np.int64)
        mapped = self.keys.get(frame, {})
        index = [mapped.get(k, -1) for k in keys]
        if any(not 0 <= i < n_inst for i in index):
            problems.append(
                f"frame {frame}: tracks {keys} are not all instances of the decodes ({n_inst})"
            )
            return None
        return np.asarray(index, dtype=np.int64)

    @staticmethod
    def _check_positions(
        frame: int,
        keys: list[Hashable],
        asked: NDArray[np.float64],
        stored: NDArray[np.float64],
        problems: list[str],
    ) -> None:
        """Note a problem if the requested tag nodes differ from the stored ones (NaN == NaN)."""
        same = (np.isnan(asked) == np.isnan(stored)).all(axis=1) & np.isclose(
            asked, stored, atol=_POSITION_TOLERANCE, rtol=0.0, equal_nan=True
        ).all(axis=1)
        if not same.all():
            bad = int(np.argmin(same))
            problems.append(
                f"frame {frame} instance {keys[bad]}: tag node {asked[bad].tolist()} "
                f"vs stored {stored[bad].tolist()}"
            )
