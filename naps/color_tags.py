r"""Brightness ("colour") tags: per-instance votes for ``naps-track --decodes``.

Some experiments mark animals with paint dots instead of ArUco markers, e.g. one bee with a light
(purple/white) dot and one with a black dot. In grayscale video the light dot is a bright disc on
the dark thorax and the black dot is (nearly) invisible, so the only signal is the brightness at
the tag node. This module measures it for every SLEAP instance, splits bright from dark with a
per-video threshold and writes the votes as a precomputed-decodes file
(:mod:`naps.precomputed`), so both identity engines can use them unchanged::

    naps-color-decodes --slp-path X.slp --video-path V.mp4 --output-path X.decodes.npz \\
        --bright-tag 1 --dark-tag 0
    naps-track --slp-path X.slp --video-path V.mp4 --decodes X.decodes.npz \\
        --closed-set 0 1 --identity-engine joint --colony-size 2 ...

The two kinds of reading are not equally reliable: a bright disc is strong evidence for the
light tag, while a dark reading can also be a light tag that is hidden (wing, tilt, occlusion) or
a tag node placed off the tag. ``--dark-votes`` therefore defaults to off: only bright discs vote
and the closed set names the other animal.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import sleap_io as sio
from numpy.typing import NDArray

from naps.precomputed import SCHEMA_VERSION

__all__ = ["ColorReading", "classify", "main", "measure", "otsu_threshold", "write_decodes"]

logger = logging.getLogger(__name__)


@dataclass
class ColorReading:
    """Brightness of the tag node of every instance of every frame ``0..n_frames-1``.

    Attributes:
        n_inst: ``(F,)`` instances per frame (``LabeledFrame.instances`` order).
        tag_xy: ``(F, S, 2)`` tag-node positions (``NaN``: none).
        value: ``(F, S)`` brightness statistic of the disc around the tag node (``NaN``: none).
        frame_size: ``(width, height)``.
    """

    n_inst: NDArray[np.int64]
    tag_xy: NDArray[np.float32]
    value: NDArray[np.float64]
    frame_size: tuple[int, int]


def _disc_offsets(radius: float) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    r = int(np.ceil(radius))
    dy, dx = np.mgrid[-r : r + 1, -r : r + 1]
    keep = dy**2 + dx**2 <= radius**2
    return dy[keep], dx[keep]


def disc_statistic(
    gray: NDArray[np.uint8], x: float, y: float, radius: float, percentile: float
) -> float:
    """``percentile`` of the pixels within ``radius`` of ``(x, y)`` (``NaN`` if none in frame)."""
    dy, dx = _disc_offsets(radius)
    rows = round(y) + dy
    cols = round(x) + dx
    h, w = gray.shape
    ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    if not ok.any():
        return float("nan")
    return float(np.percentile(gray[rows[ok], cols[ok]], percentile))


def measure(
    slp_path: str | Path,
    video_path: str | Path,
    *,
    tag_node: str = "tag",
    radius: float = 5.0,
    percentile: float = 75.0,
    end_frame: int | None = None,
) -> ColorReading:
    """Read the video once and measure the tag-node brightness of every instance.

    Frames without instances are skipped with ``grab()`` (no decode of the image).
    ``percentile`` (default 75) of a ``radius`` disc tolerates a tag node a few pixels off the dot.
    """
    labels = sio.load_slp(str(slp_path), open_videos=False)
    k = labels.skeleton.node_names.index(tag_node)
    by_frame = {lf.frame_idx: lf for lf in labels.labeled_frames}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        msg = f"cannot open {video_path}"
        raise OSError(msg)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    last = end_frame if end_frame is not None else int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    if end_frame is None and by_frame:
        last = max(last, *by_frame)
    n_frames = last + 1
    slots = max([len(lf.instances) for lf in by_frame.values()] + [1])
    n_inst = np.zeros(n_frames, np.int64)
    tag_xy = np.full((n_frames, slots, 2), np.nan, np.float32)
    value = np.full((n_frames, slots), np.nan)
    frame = 0
    while frame < n_frames:
        lf = by_frame.get(frame)
        if lf is None:
            if not cap.grab():
                break
            frame += 1
            continue
        ok, img = cap.read()
        if not ok:
            break
        gray = img[:, :, 0] if img.ndim == 3 else img
        n_inst[frame] = len(lf.instances)
        for j, inst in enumerate(lf.instances):
            x, y = inst.numpy()[k]
            tag_xy[frame, j] = (x, y)
            if np.isfinite(x) and np.isfinite(y):
                value[frame, j] = disc_statistic(gray, float(x), float(y), radius, percentile)
        frame += 1
    cap.release()
    if frame < n_frames:
        logger.warning("video %s ended at frame %d, expected %d", video_path, frame, n_frames)
        n_inst, tag_xy, value = n_inst[:frame], tag_xy[:frame], value[:frame]
    return ColorReading(n_inst=n_inst, tag_xy=tag_xy, value=value, frame_size=(width, height))


def otsu_threshold(values: NDArray[np.float64]) -> float:
    """Otsu threshold of the finite ``values`` (0..255 grey levels)."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float("nan")
    hist, edges = np.histogram(v, bins=256, range=(0, 256))
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega))
    # the criterion is flat between two well separated modes: take the middle of the plateau
    best = np.flatnonzero(np.isclose(between, np.nanmax(between), rtol=1e-9, atol=0))
    return float(edges[round((best[0] + best[-1]) / 2) + 1])


def classify(value: NDArray[np.float64], threshold: float, margin: float) -> NDArray[np.int8]:
    """``+1`` bright, ``-1`` dark, ``0`` abstain (within ``margin`` of the threshold or ``NaN``)."""
    out = np.zeros(value.shape, np.int8)
    out[value >= threshold + margin] = 1
    out[value <= threshold - margin] = -1
    return out


def write_decodes(
    path: str | Path,
    reading: ColorReading,
    calls: NDArray[np.int8],
    *,
    bright_tag: int,
    dark_tag: int,
    dark_votes: bool,
    video: str | Path,
    meta: str = "",
) -> int:
    """Write the decodes file; returns the number of votes."""
    keep = calls == 1
    if dark_votes:
        keep |= calls == -1
    frames, slots = np.nonzero(keep)
    tags = np.where(calls[frames, slots] == 1, bright_tag, dark_tag).astype(np.int64)
    np.savez_compressed(
        path,
        schema_version=np.int64(SCHEMA_VERSION),
        frame_size=np.array(reading.frame_size, np.int64),
        frame_idx=np.arange(len(reading.n_inst), dtype=np.int64),
        n_inst=reading.n_inst,
        tag_xy=reading.tag_xy,
        vote_frame=frames.astype(np.int64),
        vote_inst=slots.astype(np.int64),
        vote_tag=tags,
        vote_conf=reading.value[frames, slots],
        video=np.array(str(Path(video).resolve())),
        meta=np.array(meta),
    )
    return len(frames)


def build_parser() -> argparse.ArgumentParser:
    """The ``naps-color-decodes`` argument parser."""
    p = argparse.ArgumentParser(
        prog="naps-color-decodes",
        description="Bright/dark paint-dot tags -> precomputed decodes for naps-track --decodes.",
    )
    p.add_argument("--slp-path", required=True, help="SLEAP predictions (.slp) with a tag node.")
    p.add_argument("--video-path", required=True, help="The video the predictions are for.")
    p.add_argument("--output-path", required=True, help="Decodes file to write (.npz).")
    p.add_argument("--tag-node-name", default="tag", help="Tag node name. Default 'tag'.")
    p.add_argument("--radius", type=float, default=5.0, help="Disc radius (px). Default 5.")
    p.add_argument(
        "--percentile", type=float, default=75.0,
        help="Percentile of the disc pixels used as the brightness. Default 75.",
    )  # fmt: skip
    p.add_argument(
        "--threshold", type=float,
        help="Bright/dark grey-level threshold. Default: Otsu over all instances of the video.",
    )  # fmt: skip
    p.add_argument(
        "--margin", type=float, default=10.0,
        help="Abstain within this many grey levels of the threshold. Default 10.",
    )  # fmt: skip
    p.add_argument("--bright-tag", type=int, default=1, help="Tag id of a bright dot. Default 1.")
    p.add_argument("--dark-tag", type=int, default=0, help="Tag id of a dark dot. Default 0.")
    p.add_argument(
        "--dark-votes", action="store_true",
        help="Also vote the dark tag for dark readings (off: hidden light dots also read dark).",
    )  # fmt: skip
    p.add_argument("--end-frame", type=int, help="Last frame (default: last video frame).")
    p.add_argument("--csv", help="Also write frame,inst,x,y,value,call per instance.")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point of ``naps-color-decodes``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = build_parser().parse_args(argv)
    reading = measure(
        a.slp_path, a.video_path, tag_node=a.tag_node_name, radius=a.radius,
        percentile=a.percentile, end_frame=a.end_frame,
    )  # fmt: skip
    thr = a.threshold if a.threshold is not None else otsu_threshold(reading.value)
    calls = classify(reading.value, thr, a.margin)
    meta = (
        f"threshold={thr:.1f} margin={a.margin} radius={a.radius} percentile={a.percentile} "
        f"bright={a.bright_tag} dark={a.dark_tag} dark_votes={a.dark_votes}"
    )
    n = write_decodes(
        a.output_path,
        reading,
        calls,
        bright_tag=a.bright_tag,
        dark_tag=a.dark_tag,
        dark_votes=a.dark_votes,
        video=a.video_path,
        meta=meta,
    )
    finite = int(np.isfinite(reading.value).sum())
    logger.info(
        "%s: %d frames, %d instances, %d with a tag node; threshold %.1f%s; bright %d, dark %d, "
        "abstain %d; %d votes",
        a.slp_path, len(reading.n_inst), int(reading.n_inst.sum()), finite, thr,
        "" if a.threshold is not None else " (Otsu)", int((calls == 1).sum()),
        int((calls == -1).sum()), finite - int((calls != 0).sum()), n,
    )  # fmt: skip
    if a.csv:
        f, s = np.nonzero(np.isfinite(reading.value))
        rows = np.column_stack(
            [f, s, reading.tag_xy[f, s, 0], reading.tag_xy[f, s, 1], reading.value[f, s],
             calls[f, s]]
        )  # fmt: skip
        np.savetxt(
            a.csv, rows, delimiter=",", header="frame,inst,x,y,value,call", comments="",
            fmt=["%d", "%d", "%.2f", "%.2f", "%.1f", "%d"],
        )  # fmt: skip


if __name__ == "__main__":
    main()
