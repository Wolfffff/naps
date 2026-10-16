"""Device (``torch``) implementation of :class:`naps.crops.CropExtractor`.

Hand the extractor a ``uint8`` tensor of frames (for example Y planes NVDEC decoded into
GPU memory) and it cuts every crop of the batch on that device with one gather kernel (axis-aligned
crops) or one gather plus one ``grid_sample`` (rotated / upsampled crops): the 13 MB frames never
leave the GPU, only the small ``(N, S, S)`` crops are copied back (:meth:`DeviceCropBatch.to_host`).

Results equal the host implementation: exactly for gathered crops, within one grey level for
resampled crops (float32 arithmetic and rounding of ties). This module imports ``torch``; it is
loaded by :meth:`naps.crops.CropExtractor.extract` only when it is given a tensor.

The work is queued on the current CUDA stream, so run it inside ``with torch.cuda.stream(s):`` to
overlap it with decoding or inference on another stream.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from naps.crops import (
    CropBatch,
    CropMeta,
    Frame,
    Frames,
    meta_fields,
    normalise_centers,
)

if TYPE_CHECKING:
    from naps.crops import CropExtractor

__all__ = ["DeviceCropBatch", "extract"]

torch: Any = importlib.import_module("torch")
_functional: Any = importlib.import_module("torch.nn.functional")

_GRID_MODE = {"nearest": "nearest", "linear": "bilinear", "cubic": "bicubic"}
_FRAMES_3D = 3
_FRAMES_4D = 4
_QUANTILE_LIMIT = 1 << 24  # torch.quantile refuses larger inputs
_UINT8_MAX = 255.0


@dataclass(frozen=True, eq=False)
class DeviceCropBatch(CropMeta):
    """A batch of crops on the frames' device.

    Attributes:
        crops: ``(N, S, S)`` ``uint8`` tensor on the device of the input frames.
    """

    crops: Any

    def to_host(self, out: Frame | None = None, *, non_blocking: bool = False) -> CropBatch:
        """Copy the crops to host memory.

        Args:
            out: Optional preallocated ``(N, S, S)`` ``uint8`` C-contiguous array (page-locked
                memory, for example ``torch.empty(shape, dtype=torch.uint8, pin_memory=True)
                .numpy()``, makes the copy asynchronous).
            non_blocking: Do not wait for the copy. The returned array is valid only after the
                current CUDA stream (or an event recorded after this call) has completed.

        Returns:
            A :class:`~naps.crops.CropBatch` with the same bookkeeping.

        Raises:
            ValueError: If ``out`` has the wrong shape, dtype or layout.
        """
        shape = tuple(self.crops.shape)
        if out is None:
            pinned = bool(non_blocking and self.crops.is_cuda)
            host = torch.empty(shape, dtype=torch.uint8, pin_memory=pinned)
            array = host.numpy()
        elif out.shape != shape or out.dtype != np.uint8 or not out.flags.c_contiguous:
            msg = f"out must be a C-contiguous uint8 array of shape {shape}"
            raise ValueError(msg)
        else:
            array = out
            host = torch.from_numpy(out)
        host.copy_(self.crops, non_blocking=non_blocking)
        return CropBatch(**meta_fields(self), crops=array)


def extract(
    extractor: CropExtractor,
    frames: Any,  # noqa: ANN401 (torch tensor)
    centers_xy: Any,  # noqa: ANN401
    angles: Any,  # noqa: ANN401
    frame_index: Any,  # noqa: ANN401
) -> DeviceCropBatch:
    """Cut the crops of ``frames`` on the device of ``frames`` (see ``CropExtractor.extract``).

    Args:
        extractor: The configured extractor (its ``backend`` is ignored).
        frames: ``(H, W)``, ``(K, H, W)`` or ``(K, 1, H, W)`` ``uint8`` tensor.
        centers_xy: ``(N, 2)`` or ``(K, n, 2)`` centres (array-like or tensor).
        angles: ``None`` or ``(N,)`` radians.
        frame_index: ``None`` or ``(N,)`` frame numbers.

    Returns:
        The crops and their bookkeeping.

    Raises:
        ValueError: On frames that are not ``uint8`` tensors of a supported shape.
    """
    stack = _as_stack(frames)
    centers, index = normalise_centers(centers_xy, frame_index)
    shapes = np.tile(np.asarray(stack.shape[1:], dtype=np.int64), (stack.shape[0], 1))
    geom = extractor.geometry(Frames([], shapes), centers, index, angles)
    device = stack.device
    packed = np.stack([geom.meta.frame_index, geom.window_x0, geom.window_y0, geom.meta.valid])
    ints = torch.from_numpy(packed.astype(np.int64)).to(device)  # one host-to-device copy
    valid = ints[3].bool()
    if extractor.uses_affine(angles):
        crops = _affine(extractor, stack, geom, ints, device)
    else:
        crops = _gather(stack, ints[0], ints[1], ints[2], extractor.half, extractor)
    crops = torch.where(valid[:, None, None], crops, _scalar(extractor.fill, device))
    if extractor.lut is not None:
        table = torch.as_tensor(extractor.lut, device=device)
        crops = table[crops.long()]
    if extractor.stretch is not None and crops.numel():
        crops = _stretch(crops, extractor.stretch)
    return DeviceCropBatch(**meta_fields(geom.meta), crops=crops)


def _as_stack(frames: Any) -> Any:  # noqa: ANN401
    """Return ``frames`` as a ``(K, H, W)`` ``uint8`` tensor (a view)."""
    if frames.dtype != torch.uint8:
        msg = "frames must be a uint8 tensor"
        raise ValueError(msg)
    if frames.ndim == _FRAMES_4D and frames.shape[1] == 1:
        return frames[:, 0]
    if frames.ndim == _FRAMES_3D:
        return frames
    if frames.ndim == _FRAMES_3D - 1:
        return frames[None]
    msg = f"frames must be (H, W), (K, H, W) or (K, 1, H, W), got {tuple(frames.shape)}"
    raise ValueError(msg)


def _scalar(value: int, device: Any) -> Any:  # noqa: ANN401
    """A 0-d ``uint8`` tensor on ``device``."""
    return torch.tensor(value, dtype=torch.uint8, device=device)


def _gather(  # noqa: PLR0917 (internal helper)
    stack: Any,  # noqa: ANN401
    frame_index: Any,  # noqa: ANN401
    x0: Any,  # noqa: ANN401
    y0: Any,  # noqa: ANN401
    half: int,
    extractor: CropExtractor,
) -> Any:  # noqa: ANN401
    """Gather ``(N, 2 * half, 2 * half)`` windows with their top-left pixel at ``(x0, y0)``."""
    device = stack.device
    height, width = stack.shape[-2:]
    size = 2 * half
    arange = torch.arange(size, device=device)
    ys = y0[:, None] + arange
    xs = x0[:, None] + arange
    frame = frame_index
    window = stack[
        frame[:, None, None],
        ys.clamp(0, height - 1)[:, :, None],
        xs.clamp(0, width - 1)[:, None, :],
    ]
    if extractor.pad == "constant":
        inside = ((ys >= 0) & (ys < height))[:, :, None] & ((xs >= 0) & (xs < width))[:, None, :]
        window = torch.where(inside, window, _scalar(extractor.fill, device))
    return window


def _affine(
    extractor: CropExtractor,
    stack: Any,  # noqa: ANN401
    geom: Any,  # noqa: ANN401
    ints: Any,  # noqa: ANN401
    device: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Resample every crop from its source window with one ``grid_sample``."""
    radius = extractor.source_radius
    windows = _gather(stack, ints[0], ints[1], ints[2], radius, extractor)
    side = 2 * radius
    mats = geom.meta.transforms.copy()
    mats[:, 0, 2] -= geom.window_x0
    mats[:, 1, 2] -= geom.window_y0
    m = torch.as_tensor(mats, dtype=torch.float32, device=device)
    size = extractor.size
    grid = torch.arange(size, dtype=torch.float32, device=device)
    xs = m[:, 0, 0, None, None] * grid[None, None, :] + m[:, 0, 1, None, None] * grid[None, :, None]
    ys = m[:, 1, 0, None, None] * grid[None, None, :] + m[:, 1, 1, None, None] * grid[None, :, None]
    xs = xs + m[:, 0, 2, None, None]
    ys = ys + m[:, 1, 2, None, None]
    norm = 2.0 / (side - 1)
    sample_grid = torch.stack([xs * norm - 1.0, ys * norm - 1.0], dim=-1)
    sampled = _functional.grid_sample(
        windows[:, None].to(torch.float32),
        sample_grid,
        mode=_GRID_MODE[extractor.interpolation],
        padding_mode="zeros",
        align_corners=True,
    )[:, 0]
    return (sampled + 0.5).floor_().clamp_(0, _UINT8_MAX).to(torch.uint8)


def _stretch(crops: Any, percentiles: tuple[float, float]) -> Any:  # noqa: ANN401
    """Per-crop linear stretch of the low/high percentiles to 0 / 255."""
    n = crops.shape[0]
    flat = crops.reshape(n, -1).to(torch.float32)
    q = torch.tensor([percentiles[0] / 100.0, percentiles[1] / 100.0], device=crops.device)
    per_chunk = max(_QUANTILE_LIMIT // flat.shape[1], 1)
    bounds = torch.cat(
        [torch.quantile(flat[i : i + per_chunk], q, dim=1) for i in range(0, n, per_chunk)], dim=1
    )
    low, high = bounds[0][:, None], bounds[1][:, None]
    gain = _UINT8_MAX / (high - low).clamp_min(1.0)
    out = ((flat - low) * gain + 0.5).floor_().clamp_(0, _UINT8_MAX)
    return out.to(torch.uint8).reshape(crops.shape)
