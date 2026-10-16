"""Shared helpers of the parallel-decode tests: a video whose tags change every 2 frames."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from naps.aruco import ArUcoModel
from naps.decode import DecodeParams, TagDecoder

N_FRAMES = 47
SIZE = 160
POSITIONS = {0: (45.0, 45.0), 1: (115.0, 45.0), 2: (45.0, 115.0), 3: (115.0, 115.0)}
CROP = 30


def _marker(tag: int) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    img = cv2.aruco.generateImageMarker(dictionary, tag, 40)
    return cv2.copyMakeBorder(img, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)


def tag_at(frame: int, track: int) -> int:
    """The tag shown by ``track`` in ``frame``: changes every 2 frames, so a seam error shows."""
    return (frame // 2 * 3 + track * 7) % 60


def make_video(directory: Path) -> Path:
    """Write the test video (intra-only MJPG, so every seek is exact) into ``directory``."""
    path = directory / "varying.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 10.0, (SIZE, SIZE))
    if not writer.isOpened():  # pragma: no cover - environment without an MJPG encoder
        pytest.skip("OpenCV build cannot write MJPG video")
    for f in range(N_FRAMES):
        img = np.full((SIZE, SIZE), 255, np.uint8)
        for track, (x, y) in POSITIONS.items():
            m = _marker(tag_at(f, track))
            h = m.shape[0]
            img[int(y) - h // 2 : int(y) - h // 2 + h, int(x) - h // 2 : int(x) - h // 2 + h] = m
        writer.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    writer.release()
    return path


def make_detector():
    """The legacy ArUco detector function used by the tests."""
    return ArUcoModel.withTagSet("DICT_4X4_100", 3, 10, 3, 10, 0.1, 0.1).detect


def make_tag_decoder() -> TagDecoder:
    """The decode-stage decoder with the same detector settings."""
    return TagDecoder(
        DecodeParams(
            tag_set="DICT_4X4_100",
            win_min=3,
            win_max=10,
            win_step=3,
            thresh_const=10,
            margin=0.1,
            error_correction_rate=0.1,
        )
    )


def coords(first: int = 0, last: int = N_FRAMES - 1) -> dict[int, dict[int, tuple[float, float]]]:
    """``coords[frame][track] -> (x, y)`` for ``first..last``."""
    return {f: dict(POSITIONS) for f in range(first, last + 1)}


def plain(decodes):
    """Comparable form of a ``decodes[frame][track] -> votes`` mapping (keeps the order)."""
    return [(f, [(t, list(v)) for t, v in row.items()]) for f, row in decodes.items()]
