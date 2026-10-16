"""Batched crops around tag nodes with exact origin bookkeeping (CPU, optional GPU).

NAPS decodes the ArUco tag of every animal in a small window around its tag node. The legacy code
cut that window one animal at a time with Python slicing. :class:`CropExtractor` cuts the windows
of many animals (and of many frames) in one call and keeps the bookkeeping the decoder needs: where
every crop came from, which crops are usable, and which part of a fixed-size crop is real image.

Two cutting modes share one API:

* **Gather** (default): the legacy window ``[int(c) - half, int(c) + half)`` per axis, padded to a
  fixed ``(2 * half, 2 * half)`` so the batch is one ``(N, S, S)`` array. :meth:`CropBatch.view`
  returns the unpadded legacy window of crop ``i`` as a zero-copy view; it equals
  :func:`naps.decode.crop_around` exactly (the only deliberate deviation is a centre more than
  ``half`` pixels beyond the low edge, where the legacy negative slice stop returned a huge bogus
  crop and this module returns an empty one).
* **Affine** (``scale != 1``, ``subpixel=True`` or per-crop ``angles``): every crop is resampled
  through its own 2 x 3 transform: rotation-normalised (the heading direction becomes the crop's
  +x axis), upsampled and/or sub-pixel centred. With zero angles, ``scale == 1`` and truncated
  centres it reproduces the gather crops exactly.

Frames are 2-D ``uint8`` Y-plane (grayscale) images: a single ``(H, W)`` frame, a ``(K, H, W)``
stack or a sequence of frames. If the frames are ``torch`` tensors (CPU or CUDA) the work is done
by :mod:`naps.crops_torch` on the tensors' device and a ``DeviceCropBatch`` is returned; ``torch``
is imported only then.

Coordinate conventions: pixel centres are at integer coordinates, ``x`` is the column and ``y`` the
row. Crop pixel ``(u, v)`` is frame point ``transforms[i] @ (u, v, 1)``; the crop's centre is at
``(S / 2, S / 2)``, which is the (truncated) tag node.

Example:
    >>> extractor = CropExtractor(half=50)
    >>> batch = extractor.extract(frame, centers_xy)  # doctest: +SKIP
    >>> crops = batch.to_mapping(keys)  # doctest: +SKIP
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CropBatch",
    "CropExtractor",
    "CropMeta",
    "Frame",
    "Interpolation",
    "Mosaic",
    "Pad",
    "heading_angles",
    "resample_windows",
    "stretch_crops",
]

Frame = NDArray[np.uint8]
"""A 2-D ``uint8`` grayscale frame or crop."""
Pad = Literal["constant", "replicate"]
"""How pixels outside the image are filled: a constant, or the nearest edge pixel."""
Interpolation = Literal["nearest", "linear", "cubic"]
"""Resampling of the affine mode (``cubic`` needs the cv2 backend or torch)."""
Backend = Literal["cv2", "numpy"]

_BIG = float(1 << 40)  # centres are clipped to +-_BIG before the integer cast (no overflow)
_UINT8_MAX = 255
_N_XY = 2
_STACK_NDIM = 3
_LUT_SIZE = 256
_CV2_INTERPOLATION = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
}
_MARGIN = {"nearest": 2, "linear": 2, "cubic": 4}  # extra source pixels around a rotated window


def heading_angles(
    head_xy: NDArray[np.floating] | Sequence[Sequence[float]],
    tail_xy: NDArray[np.floating] | Sequence[Sequence[float]],
) -> NDArray[np.float64]:
    """Return the angle (radians) of the vector ``tail -> head`` for each animal.

    The result is what :meth:`CropExtractor.extract` takes as ``angles``: the direction in the
    frame (``atan2(dy, dx)``, ``y`` pointing down) that becomes the +x axis of the crop. A NaN
    node gives a NaN angle, which the extractor treats as "do not rotate".

    Args:
        head_xy: ``(N, 2)`` head positions.
        tail_xy: ``(N, 2)`` abdomen (or any caudal node) positions.

    Returns:
        ``(N,)`` angles in ``(-pi, pi]``.
    """
    head = np.asarray(head_xy, dtype=np.float64).reshape(-1, _N_XY)
    tail = np.asarray(tail_xy, dtype=np.float64).reshape(-1, _N_XY)
    delta = head - tail
    return np.arctan2(delta[:, 1], delta[:, 0])


@dataclass(frozen=True, eq=False)
class CropMeta:
    """Bookkeeping shared by host and device crop batches (all arrays are NumPy).

    Attributes:
        valid: ``(N,)`` bool. ``False`` where the centre is NaN or infinite (the legacy code gave
            ``None`` instead of a crop).
        boxes: ``(N, 4)`` int ``(x0, y0, x1, y1)`` in crop pixels: the part of the crop that is
            real image under the legacy rule (never includes the frame's last row and column).
            ``(0, 0, 0, 0)`` for invalid centres; an empty box means the window missed the frame.
            In affine mode the box is the whole crop whenever the crop touches the frame.
        origins: ``(N, 2)`` float ``(x, y)`` frame coordinates of crop pixel ``(0, 0)``. Integer
            valued in gather mode.
        transforms: ``(N, 2, 3)`` float crop-to-frame affine maps (``[[a, b, tx], [c, d, ty]]``).
        frame_index: ``(N,)`` int, the frame (of the stack/sequence passed in) each crop is from.
        size: Crop side ``S`` in pixels.
    """

    valid: NDArray[np.bool_]
    boxes: NDArray[np.int64]
    origins: NDArray[np.float64]
    transforms: NDArray[np.float64]
    frame_index: NDArray[np.int64]
    size: int

    def __len__(self) -> int:
        """Number of crops."""
        return int(self.valid.shape[0])

    @property
    def nonempty(self) -> NDArray[np.bool_]:
        """``(N,)`` bool: ``valid`` and the real-image box has pixels."""
        wide = self.boxes[:, 2] > self.boxes[:, 0]
        high = self.boxes[:, 3] > self.boxes[:, 1]
        return self.valid & wide & high

    def frame_points(self, index: int, points_xy: Sequence[Sequence[float]]) -> NDArray[np.float64]:
        """Map points from the pixels of crop ``index`` to frame coordinates.

        Args:
            index: Crop number.
            points_xy: ``(M, 2)`` points in crop pixels (for example detected marker corners).

        Returns:
            ``(M, 2)`` points in frame pixels.
        """
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, _N_XY)
        matrix = self.transforms[index]
        return pts @ matrix[:, :_N_XY].T + matrix[:, _N_XY]

    def crop_points(self, index: int, points_xy: Sequence[Sequence[float]]) -> NDArray[np.float64]:
        """Map frame points into the pixels of crop ``index`` (inverse of :meth:`frame_points`).

        Args:
            index: Crop number.
            points_xy: ``(M, 2)`` points in frame pixels (for example the SLEAP tag node).

        Returns:
            ``(M, 2)`` points in crop pixels.
        """
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, _N_XY)
        matrix = self.transforms[index]
        return (pts - matrix[:, _N_XY]) @ np.linalg.inv(matrix[:, :_N_XY]).T


@dataclass(frozen=True, eq=False)
class CropBatch(CropMeta):
    """A batch of crops on the host.

    Attributes:
        crops: ``(N, S, S)`` ``uint8``; pixels outside the image are padded (see ``pad``).
    """

    crops: Frame

    def view(self, index: int) -> Frame | None:
        """Return the real-image part of crop ``index`` as a zero-copy view.

        In gather mode this is exactly the window :func:`naps.decode.crop_around` returns.

        Args:
            index: Crop number.

        Returns:
            ``None`` for an invalid (non-finite) centre, an empty array if the window missed the
            frame, else a view into :attr:`crops`.
        """
        if not self.valid[index]:
            return None
        x0, y0, x1, y1 = (int(v) for v in self.boxes[index])
        return self.crops[index, y0:y1, x0:x1]

    def to_mapping(self, keys: Sequence[Hashable]) -> dict[Hashable, Frame | None]:
        """Return ``{key: view(i)}``, the input of ``TagDecoder.decode_batch``.

        Args:
            keys: One key (for example a track) per crop, in crop order.

        Returns:
            The unpadded views by key (``None`` for invalid centres).

        Raises:
            ValueError: If ``len(keys)`` differs from the number of crops.
        """
        if len(keys) != len(self):
            msg = f"{len(keys)} keys for {len(self)} crops"
            raise ValueError(msg)
        return {key: self.view(i) for i, key in enumerate(keys)}

    def per_frame(self, n_frames: int) -> Frame:
        """Return :attr:`crops` as ``(n_frames, N // n_frames, S, S)`` (a view).

        Args:
            n_frames: Number of frames the batch was extracted from, each with the same number of
                crops in frame order.

        Returns:
            The reshaped crops.

        Raises:
            ValueError: If the crops cannot be split evenly.
        """
        if n_frames < 1 or len(self) % n_frames:
            msg = f"{len(self)} crops cannot be split over {n_frames} frames"
            raise ValueError(msg)
        return self.crops.reshape(n_frames, len(self) // n_frames, self.size, self.size)

    def to_mosaic(self, columns: int | None = None, gap: int = 8, value: int = 0) -> Mosaic:
        """Tile all crops into one image, so a detector is called once instead of ``N`` times.

        Measured on the production crops, one ``detectMarkers`` call on a 4 x 4 mosaic is about 2.9
        times faster than 16 calls on the single crops with identical decoded tags (the black
        gaps stop the dark body outlines at a crop border from spawning marker candidates).

        Args:
            columns: Tiles per row (default: ``ceil(sqrt(N))``).
            gap: Width in pixels of the ``value``-filled lane between (and after) the tiles.
            value: Fill of the gaps and of unused tiles.

        Returns:
            The canvas and the tile layout; use :meth:`Mosaic.tile_of` to attribute detections.

        Raises:
            ValueError: If ``columns`` is not positive or ``gap`` is negative.
        """
        n = len(self)
        cols = max(math.ceil(math.sqrt(n)), 1) if columns is None else columns
        if cols < 1 or gap < 0:
            msg = "columns must be positive and gap non-negative"
            raise ValueError(msg)
        rows = max(math.ceil(n / cols), 1)
        pitch = self.size + gap
        canvas = np.full((rows, pitch, cols, pitch), value, dtype=np.uint8)
        padded = np.full((rows * cols, self.size, self.size), value, dtype=np.uint8)
        padded[:n] = self.crops
        canvas[:, : self.size, :, : self.size] = padded.reshape(
            rows, cols, self.size, self.size
        ).transpose(0, 2, 1, 3)
        index = np.arange(n)
        origins = np.stack([(index % cols) * pitch, (index // cols) * pitch], axis=1)
        flat = canvas.reshape(rows * pitch, cols * pitch)
        return Mosaic(flat, origins.astype(np.int64), self.size, pitch, cols)


@dataclass(frozen=True, eq=False)
class Mosaic:
    """Crops tiled into one image (see :meth:`CropBatch.to_mosaic`).

    Attributes:
        canvas: ``(rows * pitch, columns * pitch)`` ``uint8`` image.
        origins: ``(N, 2)`` int ``(x, y)`` of the top-left pixel of every tile in the canvas.
        size: Tile side in pixels.
        pitch: Distance between neighbouring tile origins (``size + gap``).
        columns: Tiles per row.
    """

    canvas: Frame
    origins: NDArray[np.int64]
    size: int
    pitch: int
    columns: int

    def tile_of(self, points_xy: Sequence[Sequence[float]]) -> NDArray[np.int64]:
        """Return the tile each canvas point lies in (``-1``: gap, unused tile or outside).

        Args:
            points_xy: ``(M, 2)`` canvas points, for example detected marker centres.

        Returns:
            ``(M,)`` tile numbers.
        """
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, _N_XY)
        col = np.floor(pts[:, 0] / self.pitch).astype(np.int64)
        row = np.floor(pts[:, 1] / self.pitch).astype(np.int64)
        number = row * self.columns + col
        inside = (
            (pts >= 0).all(axis=1)
            & (col < self.columns)
            & (pts[:, 0] - col * self.pitch < self.size)
            & (pts[:, 1] - row * self.pitch < self.size)
            & (number < len(self.origins))
        )
        return np.where(inside, number, -1)

    def tile_points(self, index: int, points_xy: Sequence[Sequence[float]]) -> NDArray[np.float64]:
        """Map canvas points into the pixels of tile ``index``.

        Args:
            index: Tile number.
            points_xy: ``(M, 2)`` canvas points.

        Returns:
            ``(M, 2)`` tile (crop) pixels.
        """
        return np.asarray(points_xy, dtype=np.float64).reshape(-1, _N_XY) - self.origins[index]


@dataclass(frozen=True, eq=False)
class Geom:
    """Per-crop geometry before any pixel is touched (host arrays; shared with torch).

    Attributes:
        meta: The public bookkeeping.
        centre: ``(N, 2)`` float crop centres in frame pixels (truncated unless ``subpixel``).
        window_x0: ``(N,)`` int column of the first pixel of the (gather) source window.
        window_y0: ``(N,)`` int row of the first pixel of the (gather) source window.
    """

    meta: CropMeta
    centre: NDArray[np.float64]
    window_x0: NDArray[np.int64]
    window_y0: NDArray[np.int64]

    def __len__(self) -> int:
        """Number of crops."""
        return len(self.meta)


@dataclass(frozen=True, eq=False)
class Frames:
    """The frames of a call as an indexable sequence plus every frame's ``(H, W)``.

    Attributes:
        frames: Indexable frames (a list or a ``(K, H, W)`` array).
        shapes: ``(K, 2)`` int ``(height, width)`` of each frame.
    """

    frames: Sequence[Frame]
    shapes: NDArray[np.int64]

    def __len__(self) -> int:
        """Number of frames."""
        return int(self.shapes.shape[0])


class CropExtractor:
    """Cut many crops (of many frames) around tag nodes in one call.

    Args:
        half: Half-width of the crop in source pixels (``50``: the legacy 100 x 100 window).
        pad: Fill of pixels outside the image: ``"constant"`` (``fill``) or ``"replicate"``.
        fill: Constant fill value (``0``-``255``).
        legacy_clip: Keep the legacy rule that a window never includes the frame's last row and
            column when computing :attr:`CropMeta.boxes`.
        scale: Output pixels per source pixel; the crop side is ``round(2 * half * scale)``.
            ``scale != 1`` selects the affine mode.
        interpolation: Resampling of the affine mode.
        subpixel: Affine mode only: centre on the exact (float) tag node instead of the truncated
            one. ``True`` selects the affine mode.
        backend: ``"cv2"`` (``warpAffine`` straight from the frame, fastest) or ``"numpy"``
            (vectorised float reference that mirrors the torch implementation; no cubic).
        lut: Optional 256-entry ``uint8`` table applied to every crop (for example the luma to
            legacy-BGR remap, :meth:`naps.video.FastVideoReader.to_legacy`).
        stretch: Optional ``(low, high)`` percentiles; each crop is linearly stretched so they map
            to 0 and 255 (applied after ``lut``).

    Raises:
        ValueError: On a non-positive ``half``/``scale``, a bad ``fill`` or ``lut`` or an unknown
            ``pad``, ``interpolation`` or ``backend``.
    """

    def __init__(
        self,
        half: int = 50,
        *,
        pad: Pad = "constant",
        fill: int = 0,
        legacy_clip: bool = True,
        scale: float = 1.0,
        interpolation: Interpolation = "linear",
        subpixel: bool = False,
        backend: Backend = "cv2",
        lut: NDArray[np.uint8] | None = None,
        stretch: tuple[float, float] | None = None,
    ) -> None:
        """Validate and store the configuration."""
        if half < 1 or scale <= 0:
            msg = "half and scale must be positive"
            raise ValueError(msg)
        if pad not in ("constant", "replicate"):
            msg = f"unknown pad mode {pad!r}"
            raise ValueError(msg)
        if interpolation not in _CV2_INTERPOLATION:
            msg = f"unknown interpolation {interpolation!r}"
            raise ValueError(msg)
        if backend not in ("cv2", "numpy"):
            msg = f"unknown backend {backend!r}"
            raise ValueError(msg)
        if backend == "numpy" and interpolation == "cubic":
            msg = "the numpy backend has no cubic interpolation (use cv2 or torch)"
            raise ValueError(msg)
        if not 0 <= fill <= _UINT8_MAX:
            msg = "fill must be in 0..255"
            raise ValueError(msg)
        if lut is not None and (lut.shape != (_LUT_SIZE,) or lut.dtype != np.uint8):
            msg = "lut must be a (256,) uint8 array"
            raise ValueError(msg)
        self.half = int(half)
        self.pad: Pad = pad
        self.fill = int(fill)
        self.legacy_clip = legacy_clip
        self.scale = float(scale)
        self.interpolation: Interpolation = interpolation
        self.subpixel = subpixel
        self.backend: Backend = backend
        self.lut = lut
        self.stretch = stretch
        self.size = round(2 * self.half * self.scale)

    def uses_affine(self, angles: object = None) -> bool:
        """Whether a call is resampled (``scale != 1``, ``subpixel`` or ``angles``), not gathered.

        Args:
            angles: The ``angles`` argument of the call (``None``: no rotation requested).

        Returns:
            ``True`` for the affine mode.
        """
        return self.scale != 1.0 or self.subpixel or angles is not None

    @property
    def source_radius(self) -> int:
        """Half-width (source pixels) of a window that covers any rotation of one crop."""
        reach = math.ceil(self.size / 2.0 / self.scale * math.sqrt(2.0))
        return reach + _MARGIN[self.interpolation]

    # ------------------------------------------------------------------ public API
    def extract(
        self,
        frames: Any,  # noqa: ANN401 (ndarray / sequence of ndarrays / torch tensor)
        centers_xy: Any,  # noqa: ANN401 (array-like or torch tensor)
        *,
        angles: Any = None,  # noqa: ANN401 (array-like or None)
        frame_index: Any = None,  # noqa: ANN401 (array-like or None)
        out: Frame | None = None,
    ) -> Any:  # noqa: ANN401 (CropBatch, or DeviceCropBatch for tensors)
        """Cut one crop per centre.

        Args:
            frames: One ``(H, W)`` frame, a ``(K, H, W)`` stack or a sequence of ``(H, W)``
                frames (``uint8``), or the same as ``torch`` tensors (a ``(B, 1, H, W)`` tensor is
                also accepted) for the device implementation.
            centers_xy: ``(N, 2)`` ``(x, y)`` tag nodes in frame pixels, or ``(K, n, 2)`` for a
                stack of ``K`` frames with ``n`` animals each. NaN or infinite rows are invalid.
            angles: Optional ``(N,)`` radians (see :func:`heading_angles`); the direction that
                becomes the crop's +x axis. NaN means no rotation. Angles select the affine mode.
            frame_index: Optional ``(N,)`` frame of each centre; defaults to frame 0 for ``(N, 2)``
                centres and to the frame of each row for ``(K, n, 2)`` centres.
            out: Optional preallocated ``(N, S, S)`` ``uint8`` C-contiguous buffer to fill and use
                as :attr:`CropBatch.crops` (host mode only).

        Returns:
            A :class:`CropBatch` (a ``DeviceCropBatch`` of :mod:`naps.crops_torch` for tensors).

        Raises:
            ValueError: On inconsistent shapes, non-``uint8`` frames, or a bad ``out``.
        """
        if _is_tensor(frames):
            return importlib.import_module("naps.crops_torch").extract(
                self, frames, centers_xy, angles, frame_index
            )
        centers, index = normalise_centers(centers_xy, frame_index)
        stack = as_frames(frames)
        geom = self.geometry(stack, centers, index, angles)
        shape = (len(geom), self.size, self.size)
        if out is None:
            crops = np.empty(shape, dtype=np.uint8)
        elif out.shape != shape or out.dtype != np.uint8 or not out.flags.c_contiguous:
            msg = f"out must be a C-contiguous uint8 array of shape {shape}"
            raise ValueError(msg)
        else:
            crops = out
        if not self.uses_affine(angles):
            self._fill_gather(stack, geom, crops)
        elif self.backend == "cv2":
            self._fill_affine_cv2(stack, geom, crops)
        else:
            self._fill_affine_numpy(stack, geom, crops)
        if self.lut is not None and crops.size:
            flat_crops = crops.reshape(-1, self.size)
            cv2.LUT(flat_crops, self.lut, dst=flat_crops)
        if self.stretch is not None and crops.size:
            crops[...] = stretch_crops(crops, self.stretch)
        return CropBatch(**meta_fields(geom.meta), crops=crops)

    def legacy_views(
        self,
        frame: Frame,
        centers_xy: Sequence[Sequence[float]] | NDArray[np.floating],
    ) -> list[Frame | None]:
        """Return the legacy windows of one frame as zero-copy views (no batch, no padding).

        This is :func:`naps.decode.crop_around` for all centres at once: the bookkeeping is
        vectorised and the slicing is the only per-crop work.

        Args:
            frame: One ``(H, W)`` ``uint8`` frame.
            centers_xy: ``(N, 2)`` tag nodes.

        Returns:
            Per centre a view into ``frame`` (``None`` for a non-finite centre).
        """
        centers, index = normalise_centers(centers_xy, None)
        geom = self._window(as_frames(frame), centers, index)
        cols = (geom.window_x0[:, None] + geom.meta.boxes[:, (0, 2)]).tolist()
        rows = (geom.window_y0[:, None] + geom.meta.boxes[:, (1, 3)]).tolist()
        return [
            frame[ya:yb, xa:xb] if ok else None
            for ok, (xa, xb), (ya, yb) in zip(geom.meta.valid.tolist(), cols, rows, strict=True)
        ]

    def geometry(
        self,
        stack: Frames,
        centers: NDArray[np.float64],
        index: NDArray[np.int64],
        angles: Any,  # noqa: ANN401
    ) -> Geom:
        """Compute the geometry of the requested crops without touching pixels.

        Args:
            stack: The frames (see :func:`as_frames`).
            centers: ``(N, 2)`` float centres (see :func:`normalise_centers`).
            index: ``(N,)`` frame of each centre.
            angles: ``None`` or ``(N,)`` radians.

        Returns:
            The geometry; in the affine mode its transforms carry rotation and scale.

        Raises:
            ValueError: On an out-of-range ``index`` or ``angles`` of the wrong shape.
        """
        if index.shape != (centers.shape[0],) or (not ((index >= 0) & (index < len(stack))).all()):
            msg = "frame_index must hold one in-range frame number per centre"
            raise ValueError(msg)
        geom = self._window(stack, centers, index)
        if not self.uses_affine(angles):
            return geom
        return self._affine_geometry(stack, centers, geom, angles)

    def _window(
        self, stack: Frames, centers: NDArray[np.float64], index: NDArray[np.int64]
    ) -> Geom:
        """Legacy window bookkeeping of every centre (no pixel work)."""
        size = 2 * self.half
        valid = np.isfinite(centers).all(axis=1)
        safe = np.clip(np.where(valid[:, None], centers, 0.0), -_BIG, _BIG)
        trunc = np.trunc(safe).astype(np.int64)
        win_x0 = trunc[:, 0] - self.half
        win_y0 = trunc[:, 1] - self.half
        edge_x = stack.shapes[index, 1] - int(self.legacy_clip)
        edge_y = stack.shapes[index, 0] - int(self.legacy_clip)
        bx0 = np.clip(-win_x0, 0, size)
        by0 = np.clip(-win_y0, 0, size)
        bx1 = np.maximum(np.clip(edge_x - win_x0, 0, size), bx0)
        by1 = np.maximum(np.clip(edge_y - win_y0, 0, size), by0)
        boxes = np.stack([bx0, by0, bx1, by1], axis=1)
        boxes[~valid] = 0
        origins = np.stack([win_x0, win_y0], axis=1).astype(np.float64)
        transforms = np.zeros((centers.shape[0], 2, 3), dtype=np.float64)
        transforms[:, 0, 0] = transforms[:, 1, 1] = 1.0
        transforms[:, :, 2] = origins
        meta = CropMeta(valid, boxes, origins, transforms, index, size)
        return Geom(meta, trunc.astype(np.float64), win_x0, win_y0)

    def _affine_geometry(
        self,
        stack: Frames,
        centers: NDArray[np.float64],
        geom: Geom,
        angles: Any,  # noqa: ANN401
    ) -> Geom:
        """Replace the legacy windows by per-crop rotation / scale transforms."""
        n = centers.shape[0]
        valid = geom.meta.valid
        theta = np.zeros(n) if angles is None else np.asarray(angles, dtype=np.float64).reshape(-1)
        if theta.shape != (n,):
            msg = f"angles must have shape ({n},)"
            raise ValueError(msg)
        theta = np.where(np.isfinite(theta), theta, 0.0)
        raw = np.where(valid[:, None], centers, 0.0)
        centre = np.clip(raw if self.subpixel else np.trunc(raw), -_BIG, _BIG)
        cos, sin = np.cos(theta) / self.scale, np.sin(theta) / self.scale
        half_out = self.size / 2.0
        transforms = np.zeros((n, 2, 3), dtype=np.float64)
        transforms[:, 0, 0], transforms[:, 0, 1] = cos, -sin
        transforms[:, 1, 0], transforms[:, 1, 1] = sin, cos
        transforms[:, 0, 2] = centre[:, 0] - half_out * (cos - sin)
        transforms[:, 1, 2] = centre[:, 1] - half_out * (sin + cos)
        # the crop's footprint (its bounding square) must overlap the frame to be non-empty
        reach = half_out * (np.abs(cos) + np.abs(sin))  # cos/sin already include 1 / scale
        edge_x = stack.shapes[geom.meta.frame_index, 1] - int(self.legacy_clip)
        edge_y = stack.shapes[geom.meta.frame_index, 0] - int(self.legacy_clip)
        touches = (
            valid
            & (centre[:, 0] + reach > 0)
            & (centre[:, 0] - reach < edge_x)
            & (centre[:, 1] + reach > 0)
            & (centre[:, 1] - reach < edge_y)
        )
        boxes = np.zeros((n, 4), dtype=np.int64)
        boxes[touches] = (0, 0, self.size, self.size)
        radius = self.source_radius
        anchor = np.floor(centre).astype(np.int64) - radius
        meta = CropMeta(
            valid, boxes, transforms[:, :, 2].copy(), transforms, geom.meta.frame_index, self.size
        )
        return Geom(meta, centre, anchor[:, 0], anchor[:, 1])

    def _fill_gather(self, stack: Frames, geom: Geom, crops: Frame) -> None:
        """Copy the legacy windows into ``crops`` (padding outside the image)."""
        size = 2 * self.half
        replicate = self.pad == "replicate"
        for i, (ok, k, x0, y0) in enumerate(
            zip(
                geom.meta.valid.tolist(),
                geom.meta.frame_index.tolist(),
                geom.window_x0.tolist(),
                geom.window_y0.tolist(),
                strict=True,
            )
        ):
            frame = stack.frames[k]
            height, width = frame.shape
            if ok and x0 >= 0 and y0 >= 0 and x0 + size <= width and y0 + size <= height:
                crops[i] = frame[y0 : y0 + size, x0 : x0 + size]
            elif ok and replicate:
                ys = np.clip(np.arange(y0, y0 + size), 0, height - 1)
                xs = np.clip(np.arange(x0, x0 + size), 0, width - 1)
                crops[i] = frame[ys[:, None], xs[None, :]]
            else:
                crops[i] = self.fill
                if ok:
                    ya, yb = max(-y0, 0), min(height - y0, size)
                    xa, xb = max(-x0, 0), min(width - x0, size)
                    if yb > ya and xb > xa:
                        crops[i, ya:yb, xa:xb] = frame[y0 + ya : y0 + yb, x0 + xa : x0 + xb]

    def _fill_affine_cv2(self, stack: Frames, geom: Geom, crops: Frame) -> None:
        """``cv2.warpAffine`` straight from the frame into the output slot."""
        flags = _CV2_INTERPOLATION[self.interpolation] | cv2.WARP_INVERSE_MAP
        border = cv2.BORDER_REPLICATE if self.pad == "replicate" else cv2.BORDER_CONSTANT
        dsize = (self.size, self.size)
        for i, (ok, k) in enumerate(
            zip(geom.meta.valid.tolist(), geom.meta.frame_index.tolist(), strict=True)
        ):
            if ok:
                cv2.warpAffine(
                    stack.frames[k],
                    geom.meta.transforms[i],
                    dsize,
                    dst=crops[i],
                    flags=flags,
                    borderMode=border,
                    borderValue=self.fill,
                )
            else:
                crops[i] = self.fill

    def _fill_affine_numpy(self, stack: Frames, geom: Geom, crops: Frame) -> None:
        """Vectorised float32 resampling of per-crop source windows (mirrors the torch path)."""
        radius = self.source_radius
        win = 2 * radius
        windows = np.empty((len(geom), win, win), dtype=np.uint8)
        gather = CropExtractor(radius, pad=self.pad, fill=self.fill)
        gather._fill_gather(stack, geom, windows)
        matrices = geom.meta.transforms.copy()
        matrices[:, 0, 2] -= geom.window_x0
        matrices[:, 1, 2] -= geom.window_y0
        crops[...] = resample_windows(windows, matrices, self.size, self.interpolation, self.fill)
        crops[~geom.meta.valid] = self.fill


# ---------------------------------------------------------------------- helpers (shared with torch)
def meta_fields(meta: CropMeta) -> dict[str, Any]:
    """The constructor arguments of ``meta`` (to build a subclass from it)."""
    return {
        "valid": meta.valid,
        "boxes": meta.boxes,
        "origins": meta.origins,
        "transforms": meta.transforms,
        "frame_index": meta.frame_index,
        "size": meta.size,
    }


def _is_tensor(obj: object) -> bool:
    """Whether ``obj`` is a ``torch`` tensor (without importing torch)."""
    return type(obj).__module__.split(".")[0] == "torch"


def normalise_centers(
    centers_xy: Any,  # noqa: ANN401
    frame_index: Any,  # noqa: ANN401
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return ``(N, 2)`` float centres and the ``(N,)`` frame of each.

    Args:
        centers_xy: ``(N, 2)`` or ``(K, n, 2)`` array-like (or a ``torch`` tensor).
        frame_index: ``None`` or ``(N,)`` frame numbers (not allowed with ``(K, n, 2)``).

    Returns:
        The flattened centres and the frame index (zeros if none was given).

    Raises:
        ValueError: On a wrong centre shape or a ``frame_index`` given with 3-D centres.
    """
    raw = centers_xy.detach().cpu().numpy() if _is_tensor(centers_xy) else centers_xy
    centers = np.asarray(raw, dtype=np.float64)
    if centers.ndim == _STACK_NDIM and centers.shape[-1] == _N_XY:
        if frame_index is not None:
            msg = "frame_index cannot be combined with (K, n, 2) centres"
            raise ValueError(msg)
        k, n = centers.shape[:2]
        return centers.reshape(-1, _N_XY), np.repeat(np.arange(k, dtype=np.int64), n)
    if centers.ndim != _N_XY or centers.shape[-1] != _N_XY:
        msg = f"centres must have shape (N, 2) or (K, n, 2), got {centers.shape}"
        raise ValueError(msg)
    if frame_index is None:
        return centers, np.zeros(centers.shape[0], dtype=np.int64)
    return centers, np.asarray(frame_index, dtype=np.int64).reshape(-1)


def as_frames(frames: Any) -> Frames:  # noqa: ANN401
    """Normalise ``frames`` to an indexable sequence of 2-D ``uint8`` frames.

    Args:
        frames: One ``(H, W)`` array, a ``(K, H, W)`` array or a sequence of ``(H, W)`` arrays.

    Returns:
        The frames and their shapes.

    Raises:
        ValueError: If a frame is not a 2-D ``uint8`` array.
    """
    if isinstance(frames, np.ndarray) and frames.ndim == _STACK_NDIM:
        if frames.dtype != np.uint8:
            msg = "frames must be uint8"
            raise ValueError(msg)
        shapes = np.tile(np.asarray(frames.shape[1:], dtype=np.int64), (frames.shape[0], 1))
        return Frames(cast("Sequence[Frame]", frames), shapes)
    seq: list[Frame] = [frames] if isinstance(frames, np.ndarray) else list(frames)
    if any(f.dtype != np.uint8 or f.ndim != _N_XY for f in seq):
        msg = "frames must be 2-D uint8 arrays"
        raise ValueError(msg)
    return Frames(seq, np.asarray([f.shape for f in seq], dtype=np.int64).reshape(-1, _N_XY))


def resample_windows(
    windows: Frame,
    matrices: NDArray[np.float64],
    size: int,
    interpolation: Interpolation,
    fill: int,
) -> Frame:
    """Resample ``(N, P, P)`` windows through per-window affine maps (vectorised, float32).

    Args:
        windows: Source windows.
        matrices: ``(N, 2, 3)`` maps from output pixel ``(u, v)`` to window pixel coordinates.
        size: Output side.
        interpolation: ``"nearest"`` or ``"linear"``; samples outside a window read ``fill``.
        fill: Value outside the window.

    Returns:
        ``(N, size, size)`` ``uint8``.

    Raises:
        ValueError: For ``"cubic"``.
    """
    if interpolation == "cubic":
        msg = "cubic interpolation is not implemented for the numpy resampler"
        raise ValueError(msg)
    n, side = windows.shape[0], windows.shape[1]
    grid = np.arange(size, dtype=np.float32)
    mat = matrices.astype(np.float32)
    xs = (
        mat[:, 0, 0, None, None] * grid[None, None, :]
        + mat[:, 0, 1, None, None] * grid[None, :, None]
    )
    ys = (
        mat[:, 1, 0, None, None] * grid[None, None, :]
        + mat[:, 1, 1, None, None] * grid[None, :, None]
    )
    xs = xs + mat[:, 0, 2, None, None]
    ys = ys + mat[:, 1, 2, None, None]
    flat = windows.reshape(n, side * side)

    def tap(ix: NDArray[np.int64], iy: NDArray[np.int64]) -> NDArray[np.float32]:
        inside = (ix >= 0) & (ix < side) & (iy >= 0) & (iy < side)
        idx = (np.clip(iy, 0, side - 1) * side + np.clip(ix, 0, side - 1)).reshape(n, -1)
        got = np.take_along_axis(flat, idx, axis=1).reshape(n, size, size).astype(np.float32)
        return np.where(inside, got, np.float32(fill))

    if interpolation == "nearest":
        value = tap(np.floor(xs + 0.5).astype(np.int64), np.floor(ys + 0.5).astype(np.int64))
    else:
        fx, fy = np.floor(xs), np.floor(ys)
        wx, wy = xs - fx, ys - fy
        ix, iy = fx.astype(np.int64), fy.astype(np.int64)
        value = (
            tap(ix, iy) * (1 - wx) * (1 - wy)
            + tap(ix + 1, iy) * wx * (1 - wy)
            + tap(ix, iy + 1) * (1 - wx) * wy
            + tap(ix + 1, iy + 1) * wx * wy
        )
    return np.clip(np.floor(value + 0.5), 0, _UINT8_MAX).astype(np.uint8)


def stretch_crops(crops: Frame, percentiles: tuple[float, float]) -> Frame:
    """Linearly stretch each crop so its low/high percentiles map to 0 and 255.

    The percentiles are exact (``numpy.percentile`` "linear" definition) but come from a 256-bin
    histogram of every crop, which is several times faster than sorting for ``uint8`` data.

    Args:
        crops: ``(N, S, S)`` ``uint8``.
        percentiles: ``(low, high)`` in percent.

    Returns:
        The stretched crops (a new array); a flat crop maps to 0.
    """
    n = crops.shape[0]
    flat = crops.reshape(n, -1)
    m = flat.shape[1]
    keys = flat.astype(np.int64) + _LUT_SIZE * np.arange(n)[:, None]
    cum = np.bincount(keys.ravel(), minlength=_LUT_SIZE * n).reshape(n, _LUT_SIZE).cumsum(axis=1)

    def percentile(q: float) -> NDArray[np.float64]:
        pos = (m - 1) * q / 100.0
        below = math.floor(pos)
        above = min(below + 1, m - 1)
        v_below = (cum > below).argmax(axis=1)
        v_above = (cum > above).argmax(axis=1)
        return (v_below + (pos - below) * (v_above - v_below)).astype(np.float64)

    low = percentile(percentiles[0]).astype(np.float32)[:, None]
    high = percentile(percentiles[1]).astype(np.float32)[:, None]
    gain = np.float32(_UINT8_MAX) / np.maximum(high - low, np.float32(1.0))
    stretched = np.floor((flat.astype(np.float32) - low) * gain + 0.5)
    return np.clip(stretched, 0, _UINT8_MAX).astype(np.uint8).reshape(crops.shape)
