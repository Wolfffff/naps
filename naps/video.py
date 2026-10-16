"""Frame reading for the decode stage: grayscale frames without the colour-conversion cost.

:class:`FastVideoReader` wraps ``cv2.VideoCapture`` and always hands out 2-D ``uint8`` grayscale
frames. When the backend can return the decoder's raw luma (Y) plane (``CAP_PROP_CONVERT_RGB=0``,
which OpenCV 5 does for ``yuv420p`` video) the reader skips the YUV to BGR conversion; otherwise it
falls back to a normal ``read()`` and its blue channel (what NAPS always decoded). Either way the
frames come out in exactly the order and number ``read()`` would produce.

Measured on the production videos (3664 x 3664 HEVC, 20 fps; see ``docs/decode_measurements.md``,
Beszel Trial2 camera0) the raw luma path saves the conversion
(about 1.3 to 1.5 times faster than ``read()``); decoding the HEVC stream itself dominates, and a
random seek costs seconds, so always walk forward with ``grab()`` instead of seeking to many frames
(this reader does that for you).

The luma plane of limited-range video differs from the blue channel ``read()`` returns by a fixed
intensity remap. The reader measures that remap on the first frame
(:attr:`FastVideoReader.legacy_lut`), so a caller that must reproduce the old NAPS pixels exactly
can apply :meth:`FastVideoReader.to_legacy` to a crop.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Generator, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from itertools import pairwise
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DEFAULT_MAX_WALK",
    "FastVideoReader",
    "Frame",
    "PixelRange",
    "VideoReadError",
    "open_capture",
    "split_range",
]

logger = logging.getLogger(__name__)

Frame = NDArray[np.uint8]
"""A 2-D grayscale frame."""
PixelRange = Literal["limited", "full", "unknown"]
"""Intensity range of the luma plane relative to the BGR frame ``read()`` gives."""
Mode = Literal["auto", "y", "bgr"]

DEFAULT_MAX_WALK = 256
"""Forward jumps of at most this many frames are walked with ``grab()`` instead of seeking."""

# Fraction of the first frame's pixels that the measured luma to converted table must reproduce
# for the luma plane to be accepted as the same picture.
_MIN_EXPLAINED = 0.999
_N_LEVELS = 256
_LIMITED_LOW = 16
_LIMITED_SPAN = 219
_FULL_MAX = 255
_PLANE_NDIM = 2
_COLOUR_NDIM = 3


class VideoReadError(RuntimeError):
    """The video cannot be opened, seeked or read."""


def split_range(
    start: int, stop: int, parts: int, *, align: int = 1, offset: int = 0
) -> list[tuple[int, int]]:
    """Split ``range(start, stop)`` into at most ``parts`` contiguous, near-equal chunks.

    Use it to decode one video in several processes (one :class:`FastVideoReader` each): seeking
    is exact, so the chunks tile the range with no gap or overlap.

    Args:
        start: First frame (inclusive).
        stop: End frame (exclusive).
        parts: Number of chunks wanted.
        align: Move every boundary between two chunks to the nearest frame ``b`` with
            ``b % align == offset % align`` (for example the keyframe spacing of the video: a
            chunk then starts where the seek is cheap). The chunks stay contiguous, but are no
            longer equal, and fewer than ``parts`` come back if two boundaries merge or leave the
            range. 1 (default): no alignment.
        offset: Position of the boundaries inside the alignment period. OpenCV's seek lands on the
            keyframe *before* a target less than about a second after a keyframe, so boundaries a
            few frames *after* the keyframes (``offset`` 25 at 250 frames per keyframe) seek
            fastest.

    Returns:
        The non-empty ``(chunk_start, chunk_stop)`` pairs, in order.

    Raises:
        ValueError: If ``parts`` or ``align`` is not positive.
    """
    if parts < 1 or align < 1:
        msg = "parts and align must be at least 1"
        raise ValueError(msg)
    n = max(stop - start, 0)
    bounds = [start + (n * i) // parts for i in range(parts + 1)]
    if align > 1:
        inner = {((b - offset + align // 2) // align) * align + offset for b in bounds[1:-1]}
        bounds = [bounds[0], *sorted(b for b in inner if bounds[0] < b < bounds[-1]), bounds[-1]]
    return [(a, b) for a, b in pairwise(bounds) if b > a]


def open_capture(path: str, threads: int | None = None) -> cv2.VideoCapture:
    """``cv2.VideoCapture(path)``, optionally with a fixed number of decoder threads.

    OpenCV's FFmpeg backend starts one decoder thread per visible CPU, so ``N`` parallel workers
    would run ``N * N`` threads on ``N`` CPUs (extra CPU time and gigabytes of frame buffers, both
    measured at ``N = 16``). The thread count can only be given when the file is opened (setting
    ``CAP_PROP_N_THREADS`` afterwards has no effect); the decoded pixels do not depend on it.

    Args:
        path: Video file.
        threads: Decoder threads, or ``None`` for OpenCV's default. Ignored (the default open is
            used) if this OpenCV build has no ``CAP_PROP_N_THREADS`` or cannot open the file with
            it.

    Returns:
        The capture (check ``isOpened()``).
    """
    prop = getattr(cv2, "CAP_PROP_N_THREADS", None)
    if threads is not None and prop is not None:
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [prop, threads])
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(path)


@contextmanager
def _quiet_opencv() -> Generator[None, None, None]:
    """Silence OpenCV's per-frame "treated as 8UC1" warning while retrieving raw luma."""
    previous = cv2.utils.logging.getLogLevel()
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    try:
        yield
    finally:
        cv2.utils.logging.setLogLevel(previous)


def _to_gray(image: NDArray[np.generic]) -> Frame:
    """Return a retrieved frame as a 2-D uint8 image.

    A 3-channel frame yields its first (blue) channel, which is what NAPS has always decoded
    (identical to any grayscale conversion for grayscale video).
    """
    if image.ndim == _COLOUR_NDIM:
        return np.ascontiguousarray(image[:, :, 0], dtype=np.uint8)
    return image.astype(np.uint8, copy=False)


class FastVideoReader:
    """Sequential and random-access reader of grayscale frames.

    Use it as a context manager. One reader owns one ``VideoCapture``; concurrent calls are
    serialised by a lock (safe, but pointless), so give every worker thread or process its own
    reader and divide the frames with :func:`split_range`. Instances can be pickled: the unpickled
    copy re-opens the file and continues at the same frame.

    Args:
        path: Video file.
        mode: ``"auto"`` (default) uses the luma plane when the first frame proves it is a
            faithful grayscale picture and silently falls back to ``read()`` otherwise; ``"y"``
            insists on the luma plane (:class:`VideoReadError` if impossible); ``"bgr"`` always
            uses the normal ``read()`` path (what NAPS did before).
        calibrate: Measure the luma to BGR remap on the first frame (one extra ``read()``, about
            0.2 s for 3664 x 3664) to set :attr:`pixel_range` and :attr:`legacy_lut`; this is also
            what lets ``"auto"`` verify the fast path. With ``False``, ``"auto"`` trusts the shape
            of the retrieved frame.
        max_walk: Forward jumps of at most this many frames are walked with ``grab()``; longer
            jumps and any jump backwards seek.
        threads: Decoder threads of the FFmpeg backend, fixed when the file is opened (see
            :func:`open_capture`); ``None`` keeps OpenCV's default of one thread per visible CPU.
            Use 1 in each of several parallel workers.

    Attributes:
        path: The video path.
        frame_count: Frame count according to the container (may be 0 or inexact).
        fps: Frames per second according to the container.
        width: Frame width in pixels.
        height: Frame height in pixels.
        luma_plane: True if frames come from the raw luma plane (the fast path).
        pixel_range: ``"full"`` if the frames equal what ``read()`` gives, ``"limited"`` if the
            luma plane is the 16-235 to 0-255 expansion of it, ``"unknown"`` if not measured or
            not a pure remap.
        legacy_lut: 256-entry ``uint8`` table mapping this reader's intensities to the old
            ``read()`` channel-0 intensities, measured on the first frame; ``None`` if not
            measured or the video is not a pure remap (colour video).
        truncated_at: Index of the frame a read could not obtain because the stream ended there
            (reading to the end sets it to the frame count; a read after a seek past the end sets
            it to the requested frame; the seek itself does not); ``None`` if the last read or
            iteration did not hit the end.

    Raises:
        FileNotFoundError: If ``path`` is not an existing file.
        ValueError: If ``mode`` is not ``"auto"``, ``"y"`` or ``"bgr"``.
        VideoReadError: If the file cannot be opened, or ``mode="y"`` cannot be honoured.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        mode: Mode = "auto",
        calibrate: bool = True,
        max_walk: int = DEFAULT_MAX_WALK,
        threads: int | None = None,
    ) -> None:
        """Open the video and (unless ``calibrate=False``) check the fast path on frame 0."""
        if mode not in ("auto", "y", "bgr"):
            msg = f"unknown mode {mode!r}; use 'auto', 'y' or 'bgr'"
            raise ValueError(msg)
        self.path = Path(path)
        if not self.path.is_file():
            msg = f"{self.path} does not exist"
            raise FileNotFoundError(msg)
        self._mode: Mode = mode
        self._calibrate = calibrate
        self._max_walk = max_walk
        self._threads = threads
        self._lock = threading.RLock()
        self._cap: cv2.VideoCapture | None = None
        self._next = 0  # index of the frame the next grab() returns
        self.luma_plane = False
        self.pixel_range: PixelRange = "unknown"
        self.legacy_lut: NDArray[np.uint8] | None = None
        self.truncated_at: int | None = None
        self._open()

    # ------------------------------------------------------------------ lifecycle
    def _open(self) -> None:
        """Open the capture, choose the pixel path and (optionally) calibrate on frame 0."""
        cap = self._new_capture(luma=self._mode != "bgr")
        self.frame_count = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
        self.fps = float(cap.get(cv2.CAP_PROP_FPS))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cap = cap
        self.luma_plane = False
        if self._mode != "bgr":
            first = self._first_luma_frame(cap)
            if first is not None and (not self._calibrate or self._matches_decoded(first)):
                self.luma_plane = True
            elif self._mode == "y":
                self.close()
                msg = f"{self.path}: the backend cannot return a faithful luma plane (mode='y')"
                raise VideoReadError(msg)
            else:
                cap.release()
                self._cap = self._new_capture(luma=False)
        if not self.luma_plane:
            self.pixel_range = "full"
        self._require_open().set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._next = 0

    def _new_capture(self, *, luma: bool) -> cv2.VideoCapture:
        """Open the file, with raw luma output if ``luma``."""
        cap = open_capture(str(self.path), self._threads)
        if not cap.isOpened():
            cap.release()
            msg = f"cannot open video {self.path}"
            raise VideoReadError(msg)
        if luma:
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        return cap

    def _first_luma_frame(self, cap: cv2.VideoCapture) -> Frame | None:
        """Frame 0 as luma, or None if the backend does not return a usable luma plane."""
        with _quiet_opencv():
            ok, frame = cap.retrieve() if cap.grab() else (False, None)
        return self._luma_from(frame) if ok and frame is not None else None

    def _luma_from(self, frame: NDArray[np.generic]) -> Frame | None:
        """Return the luma plane contained in a raw retrieved frame, or None if it holds none."""
        if frame.dtype != np.uint8 or frame.ndim != _PLANE_NDIM:
            return None
        if frame.shape == (self.height, self.width):
            return cast("Frame", frame)
        if frame.shape == (self.height * 3 // 2, self.width):  # whole I420 buffer: Y comes first
            return cast("Frame", frame[: self.height])
        return None

    def _matches_decoded(self, luma: Frame) -> bool:
        """Compare frame 0 of the luma path with the converted frame; learn the remap.

        The luma plane replaces the old ``read()`` pixels exactly (up to a fixed intensity remap)
        only for grayscale video, where all three converted channels are equal. For colour video
        ``"auto"`` mode reads the old way; ``"y"`` mode keeps the luma plane without a remap.

        Returns:
            False if the luma plane is not a faithful stand-in for the decoded picture.
        """
        ref_cap = self._new_capture(luma=False)
        try:
            ok, bgr = ref_cap.read()
        finally:
            ref_cap.release()
        if not ok:
            # nothing to compare with: trust the shape check, but say so (pixel_range stays
            # "unknown" and legacy_lut None, so a caller that needs the legacy pixels will raise)
            logger.warning("%s: no reference frame to verify the luma plane against", self.path)
            return True
        neutral = bgr.ndim != _COLOUR_NDIM or (
            np.array_equal(bgr[..., 0], bgr[..., 1]) and np.array_equal(bgr[..., 1], bgr[..., 2])
        )
        if not neutral:
            return self._mode == "y"
        if not self._learn_remap(luma, _to_gray(bgr)):
            logger.warning("%s: the luma plane does not match the decoded picture", self.path)
            return False
        return True

    def _learn_remap(self, luma: Frame, gray: Frame) -> bool:
        """Set ``legacy_lut`` and ``pixel_range`` from a luma frame and its converted twin.

        For grayscale video the converted value is a function of the luma level alone, so one
        frame determines the table on every level it contains; the other levels are filled with
        the analytic expansion (``"limited"``) or the identity. The luma plane is accepted as a
        stand-in for the decoded picture only if that table is monotonic and reproduces the
        converted frame (a structural test that holds at any brightness, unlike a bound on the
        intensity difference).

        Returns:
            False if the converted frame is not a monotonic per-intensity remap of the luma.
        """
        pairs = luma.astype(np.int64).ravel() * _N_LEVELS + gray.ravel()
        counts = np.bincount(pairs, minlength=_N_LEVELS * _N_LEVELS).reshape(_N_LEVELS, _N_LEVELS)
        seen = counts.sum(axis=1) > 0
        levels = np.arange(_N_LEVELS)
        best = np.argmax(counts, axis=1)
        explained = float(counts[levels, best].sum()) / float(counts.sum())
        if explained < _MIN_EXPLAINED or (np.diff(best[seen]) < 0).any():
            return False
        # swscale's limited to full range expansion floors (verified exact on 145 levels)
        expansion = np.clip(((levels - _LIMITED_LOW) * _FULL_MAX) // _LIMITED_SPAN, 0, _FULL_MAX)
        measured = best
        if np.abs(measured[seen] - expansion[seen]).max() <= 1:
            self.pixel_range = "limited"
            fill = expansion
        else:
            self.pixel_range = "full" if (measured[seen] == levels[seen]).all() else "unknown"
            fill = levels
        self.legacy_lut = np.where(seen, measured, fill).astype(np.uint8)
        return True

    def close(self) -> None:
        """Release the capture (safe to call twice)."""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    def __enter__(self) -> FastVideoReader:
        """Return self."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the reader."""
        self.close()

    def __getstate__(self) -> dict[str, object]:
        """Pickle the configuration and position, not the capture."""
        state = self.__dict__.copy()
        state["_cap"] = None
        state["_lock"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore, re-open the file and continue at the pickled position."""
        self.__dict__.update(state)
        self._lock = threading.RLock()
        position = self._next
        pixel_range, legacy_lut = self.pixel_range, self.legacy_lut
        self._mode = "y" if self.luma_plane else "bgr"
        self._calibrate = False
        self._open()
        self.pixel_range, self.legacy_lut = pixel_range, legacy_lut
        self.seek(position)

    def to_legacy(self, image: Frame) -> Frame:
        """Remap ``image`` (a frame or crop of this reader) to the intensities old NAPS saw.

        Args:
            image: A frame or crop from :meth:`read` or :meth:`frames`.

        Returns:
            ``image`` itself if the frames already are the ``read()`` pixels, otherwise the
            image passed through :attr:`legacy_lut`.

        Raises:
            VideoReadError: If the frames are luma and no remap table could be measured.
        """
        if not self.luma_plane or self.pixel_range == "full":
            return image
        if self.legacy_lut is None:
            msg = f"{self.path}: no luma to read() remap is known (calibrate=False or colour video)"
            raise VideoReadError(msg)
        return np.asarray(cv2.LUT(image, self.legacy_lut), dtype=np.uint8)

    # ------------------------------------------------------------------ reading
    @property
    def position(self) -> int:
        """Index of the frame the next read returns."""
        return self._next

    def _require_open(self) -> cv2.VideoCapture:
        """Return the capture or raise if the reader was closed."""
        if self._cap is None:
            msg = f"{self.path}: the reader is closed"
            raise VideoReadError(msg)
        return self._cap

    def seek(self, index: int) -> None:
        """Position the reader so the next read returns frame ``index`` (exact).

        Raises:
            ValueError: If ``index`` is negative.
            VideoReadError: If the backend cannot position at ``index``.
        """
        if index < 0:
            msg = "frame index must not be negative"
            raise ValueError(msg)
        with self._lock:
            cap = self._require_open()
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            # a position past the end is clamped by the backend: the next read just finds no frame
            beyond_end = 0 < self.frame_count <= index
            if not beyond_end and int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != index:
                msg = f"{self.path}: cannot seek to frame {index}"
                raise VideoReadError(msg)
            self._next = index

    def read(self, index: int | None = None) -> Frame | None:
        """Return frame ``index`` (default: the next one) as 2-D uint8, advancing the position.

        Args:
            index: Frame to return; forward jumps of up to ``max_walk`` frames are walked with
                ``grab()``, anything else seeks.

        Returns:
            The frame; ``None`` if the stream ends before it (:attr:`truncated_at` is then set)
            or the frame is corrupt.

        Raises:
            ValueError: If ``index`` is negative.
            VideoReadError: If a needed seek fails, the reader is closed, or (luma reader) the
                retrieved frame is not the luma plane the reader was calibrated for.
        """
        with self._lock:
            cap = self._require_open()
            self.truncated_at = None
            if index is not None and not self._advance_to(cap, index):
                return None
            return self._grab_retrieve(cap)

    def _advance_to(self, cap: cv2.VideoCapture, index: int) -> bool:
        """Make the next grab return frame ``index``; False if the stream ended on the way."""
        gap = index - self._next
        if gap == 0:
            return True
        if 0 < gap <= self._max_walk:
            for _ in range(gap):
                if not cap.grab():
                    self.truncated_at = self._next
                    return False
                self._next += 1
            return True
        self.seek(index)
        return True

    def _grab_retrieve(self, cap: cv2.VideoCapture) -> Frame | None:
        """Grab and retrieve the next frame; None at the end of the stream or if corrupt."""
        if not cap.grab():
            self.truncated_at = self._next
            return None
        self._next += 1
        quiet: AbstractContextManager[None] = _quiet_opencv() if self.luma_plane else nullcontext()
        with quiet:
            ok, frame = cap.retrieve()
        if not ok or frame is None:
            return None
        if not self.luma_plane:
            return _to_gray(frame)
        luma = self._luma_from(frame)
        if luma is None:
            # never fall back to _to_gray(): a raw 1.5H-row I420 buffer or the first channel of a
            # colour frame is not the picture the caller expects
            msg = (
                f"{self.path}: frame {self._next - 1} is not a luma plane (dtype {frame.dtype}, "
                f"shape {frame.shape}); the reader was calibrated for {self.height}x{self.width}"
            )
            raise VideoReadError(msg)
        return luma

    def frames(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> Iterator[tuple[int, Frame | None]]:
        """Iterate ``(index, frame)`` over ``range(start, stop, step)``.

        Frames between wanted ones are skipped with ``grab()`` (decoded but not copied or
        converted). The iterator ends early if the stream ends (:attr:`truncated_at` is then
        set to the first unreadable frame). A corrupt frame is yielded as ``(index, None)``.

        Args:
            start: First frame index.
            stop: End index (exclusive); ``None`` reads to the end of the stream.
            step: Distance between yielded frames.

        Yields:
            ``(index, frame)`` pairs; ``frame`` is ``None`` only for a corrupt frame.

        Raises:
            ValueError: If ``start`` is negative or ``step`` is not positive.
            VideoReadError: If a needed seek fails, or (luma reader) a retrieved frame is not the
                luma plane the reader was calibrated for.
        """
        if start < 0 or step < 1:
            msg = "start must be >= 0 and step >= 1"
            raise ValueError(msg)
        index = start
        self.truncated_at = None
        while stop is None or index < stop:
            with self._lock:
                cap = self._require_open()
                if not self._advance_to(cap, index):
                    return
                before = self._next
                frame = self._grab_retrieve(cap)
                if frame is None and self._next == before:  # grab() failed: end of stream
                    return
            yield index, frame
            index += step

    def __iter__(self) -> Iterator[tuple[int, Frame | None]]:
        """Iterate over every frame from the current position to the end."""
        return self.frames(self._next)
