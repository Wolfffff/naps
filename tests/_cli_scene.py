"""A tiny synthetic scene for the speed-option CLI tests (3 static bees with ArUco markers)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import sleap_io as sio


def make_args(tmp_path: Path, colony, slp_factory, video_factory) -> Callable[..., list[str]]:
    """Return ``args(output_name, *extra) -> argv`` for ``naps-track`` on the colony."""
    slp = slp_factory(tmp_path / "bees.slp", colony)
    video = video_factory(tmp_path / "bees.avi", colony)

    def args(out_name: str, *extra: str) -> list[str]:
        return [
            "--slp-path", str(slp), "--video-path", str(video), "--tag-node-name", "Tag",
            "--start-frame", "0", "--end-frame", str(colony.n_frames - 1),
            "--aruco-marker-set", "DICT_4X4_100", "--aruco-crop-size", "60",
            "--output-path", str(tmp_path / out_name), *extra,
        ]  # fmt: skip

    return args


def tags_of(path: Path) -> dict[int, list[str | None]]:
    """``{frame: [track name per instance]}`` of a written ``.slp``."""
    return {
        lf.frame_idx: [None if i.track is None else i.track.name for i in lf.instances]
        for lf in sio.load_slp(str(path))
    }
