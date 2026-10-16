"""Helpers for the video/decode tests: tiny real videos and a scriptable fake ``VideoCapture``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pytest

#: limited-range luma level -> the intensity ``read()`` produces (swscale floors)
LIMITED_LUT = np.clip((np.arange(256) - 16) * 255 // 219, 0, 255).astype(np.uint8)


def textured_frames(n: int, height: int = 64, width: int = 80, seed: int = 7) -> list[np.ndarray]:
    """``n`` deterministic grayscale frames that differ from each other (so order matters)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(30, 220, (height, width), dtype=np.uint8)
    return [np.roll(base, 3 * i, axis=1) ^ np.uint8(i) for i in range(n)]


def write_gray_video(path: Path, frames: list[np.ndarray]) -> Path:
    """Write ``frames`` as a lossy (MJPG) grayscale video; skips the test if it cannot."""
    height, width = frames[0].shape
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter.fourcc(*"MJPG"), 10.0, (width, height), isColor=False
    )
    if not writer.isOpened():  # pragma: no cover - environment without an MJPG encoder
        pytest.skip("OpenCV build cannot write MJPG video")
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def write_colour_video(path: Path, frames: list[np.ndarray]) -> Path:
    """Write ``frames`` (BGR) as a colour MJPG video; skips the test if it cannot."""
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 10.0, (width, height))
    if not writer.isOpened():  # pragma: no cover - environment without an MJPG encoder
        pytest.skip("OpenCV build cannot write MJPG video")
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def read_all_blue(path: Path) -> list[np.ndarray]:
    """Every frame of ``path`` as ``cv2.VideoCapture.read()`` returns it, first channel only."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame[:, :, 0])
    cap.release()
    return frames


@dataclass
class FakeSpec:
    """Behaviour script of :class:`FakeCapture`.

    Attributes:
        frames: Luma planes of the video (what a limited-range grayscale video decodes to).
        raw: What ``retrieve()`` returns with ``CONVERT_RGB=0``: ``"plane"`` (H, W), ``"i420"``
            (3H/2, W), ``"bgr"`` (H, W, 3), ``"odd"`` (wrong 2-D shape), ``"float"`` (float
            plane) or ``"fail"`` (``(False, None)``).
        colour: Make the converted frames non-gray (blue differs from green and red).
        shifted: Make the luma plane a different picture than the converted frame.
        inverted: Converted frames are the negative of the luma remap (a decreasing function).
        opens: Whether ``isOpened()`` is true.
        fail_grab: Frame indices whose ``grab()`` fails (end of stream when it is the last).
        corrupt: Frame indices whose ``retrieve()`` fails.
        seek_error: Offset added to the position after a seek (a backend that mis-seeks).
        read_ok: Whether the converted ``read()`` of frame 0 succeeds.
        full_range: Converted frames equal the luma plane (full-range video).
        offset: Added to the converted intensities (a remap that is neither full nor limited).
        odd_from: From this frame on the raw frame has a wrong 2-D shape.
        log: ``("seek" | "grab", index)`` calls, in order.
    """

    frames: list[np.ndarray]
    raw: str = "plane"
    colour: bool = False
    shifted: bool = False
    inverted: bool = False
    opens: bool = True
    fail_grab: set[int] = field(default_factory=set)
    corrupt: set[int] = field(default_factory=set)
    seek_error: int = 0
    read_ok: bool = True
    full_range: bool = False
    offset: int = 0
    odd_from: int | None = None
    log: list[tuple[str, int]] = field(default_factory=list)


class FakeCapture:
    """A scriptable stand-in for ``cv2.VideoCapture`` (install with ``monkeypatch``)."""

    spec: ClassVar[FakeSpec]

    def __init__(self, _path: str, *_open_params: object) -> None:
        self.convert = True
        self.pos = 0
        self.last = 0
        self.released = False

    @property
    def frames(self) -> list[np.ndarray]:
        return self.spec.frames

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV's name
        return self.spec.opens

    def release(self) -> None:
        self.released = True

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_CONVERT_RGB:
            self.convert = bool(value)
        elif prop == cv2.CAP_PROP_POS_FRAMES:
            self.pos = int(value) + self.spec.seek_error
            self.spec.log.append(("seek", int(value)))
        return True

    def get(self, prop: int) -> float:
        height, width = self.frames[0].shape
        return {
            cv2.CAP_PROP_FRAME_COUNT: float(len(self.frames)),
            cv2.CAP_PROP_FPS: 10.0,
            cv2.CAP_PROP_FRAME_WIDTH: float(width),
            cv2.CAP_PROP_FRAME_HEIGHT: float(height),
            cv2.CAP_PROP_POS_FRAMES: float(self.pos),
        }[prop]

    def grab(self) -> bool:
        if self.pos >= len(self.frames) or self.pos in self.spec.fail_grab:
            return False
        self.last = self.pos
        self.pos += 1
        self.spec.log.append(("grab", self.last))
        return True

    def _converted(self, index: int) -> np.ndarray:
        luma = self.frames[index]
        blue = luma if self.spec.full_range else LIMITED_LUT[luma]
        if self.spec.offset:
            blue = np.clip(blue.astype(np.int16) + self.spec.offset, 0, 255).astype(np.uint8)
        if self.spec.inverted:
            blue = 255 - blue
        if self.spec.shifted:
            blue = np.roll(255 - blue, 7, axis=0)
        green = blue
        if self.spec.colour:
            green = np.clip(blue.astype(np.int16) + 40, 0, 255).astype(np.uint8)
        return np.dstack([blue, green, blue])

    def retrieve(self) -> tuple[bool, np.ndarray | None]:  # noqa: PLR0911 (one return per kind)
        if self.last in self.spec.corrupt:
            return False, None
        if self.convert:
            return True, self._converted(self.last)
        luma = self.frames[self.last]
        kind = self.spec.raw
        if self.spec.odd_from is not None and self.last >= self.spec.odd_from:
            kind = "odd"
        if kind == "plane":
            return True, luma.copy()
        if kind == "i420":
            chroma = np.full((luma.shape[0] // 2, luma.shape[1]), 128, dtype=np.uint8)
            return True, np.vstack([luma, chroma])
        if kind == "bgr":
            return True, self._converted(self.last)
        if kind == "odd":
            return True, luma[:-1].copy()
        if kind == "float":
            return True, luma.astype(np.float32)
        return False, None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.pos == 0 and not self.spec.read_ok:
            return False, None
        if not self.grab():
            return False, None
        return self.retrieve()


def install_fake_video(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, spec: FakeSpec) -> Path:
    """Route ``cv2.VideoCapture`` to a :class:`FakeCapture` playing ``spec``; return its path."""
    FakeCapture.spec = spec
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    path = tmp_path / "fake.mp4"
    path.write_bytes(b"not a real video")
    return path
