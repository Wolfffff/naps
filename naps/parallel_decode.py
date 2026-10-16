"""Run the per-frame decode of a video in parallel, over contiguous frame ranges.

Reading a frame, cropping the tag window of every track and decoding the ArUco marker is a pure
function of that one frame (and the track positions in it): no state flows from frame ``f`` to
frame ``f + 1``. A video can therefore be cut into contiguous ranges that are decoded
independently, and the per-frame results concatenated, **without any window overlap or seam
stitching**: the concatenation is the serial result by construction. The stateful part of NAPS
(the rolling-window vote accumulation and the carry-forward) is cheap and stays serial, running
on the merged results; it must not be chunked, because its tie-breaking depends on the
first-seen order of tracks and tags over the whole job and a tag can be carried forward from the
first frame to the last.

The only thing a range needs is a reader that can start at an arbitrary frame
(``VideoCapture.set(CAP_PROP_POS_FRAMES)`` or :meth:`naps.video.FastVideoReader.seek`). Seeking
must be frame-accurate; ``tests/test_parallel_decode.py`` checks the merged result against the
serial one. The ranges come from :func:`naps.video.split_range`.

Two back ends are provided. ``"process"`` forks workers that inherit the task (nothing but the
small range tuples goes in, the per-frame results come back pickled); it is the default because it
scales with the number of cores independently of the GIL. ``"thread"`` runs the same task in
threads; OpenCV releases the GIL in ``VideoCapture.read`` and ``detectMarkers`` but the Python
glue around them does not, so it scales less. Each worker should open its decoder with one thread
(:func:`naps.video.open_capture`), otherwise OpenCV starts one decoder thread per CPU in every
worker.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Literal, TypeVar, cast

import cv2

from naps.video import split_range

__all__ = ["Backend", "available_cpus", "decode_in_chunks", "default_chunk_frames", "run_chunks"]

Backend = Literal["process", "thread"]

_T = TypeVar("_T")
_V = TypeVar("_V")

# Minimum frames per chunk: every chunk pays one seek (and the decode of the frames between the
# preceding keyframe and its first frame), so tiny chunks lose more than they gain.
_MIN_CHUNK_FRAMES = 200
# Chunks per worker: more than one so that a slow chunk does not leave the other workers idle.
_CHUNKS_PER_WORKER = 4

# The task of the forked workers; set by run_chunks before the pool is created.
_TASK: Callable[[int, int], object] | None = None


def default_chunk_frames(n_frames: int, workers: int) -> int:
    """Chunk length for ``n_frames`` frames and ``workers`` workers (balanced, not tiny).

    Args:
        n_frames: Number of frames of the job.
        workers: Number of parallel workers.

    Returns:
        About ``n_frames / (workers * 4)`` frames, but at least 200.
    """
    return max(_MIN_CHUNK_FRAMES, -(-n_frames // (max(workers, 1) * _CHUNKS_PER_WORKER)))


def available_cpus() -> int:
    """Number of CPUs this process may use (SLURM/cgroup aware via the affinity mask)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def decode_in_chunks(
    task: Callable[[int, int], tuple[dict[int, _V], tuple[int, int] | None]],
    first_frame: int,
    last_frame: int,
    *,
    workers: int = 1,
    chunk_frames: int | None = None,
    align: int = 1,
    offset: int = 0,
    backend: Backend = "process",
    background: Callable[[], None] | None = None,
) -> tuple[dict[int, _V], tuple[int, int] | None]:
    """Decode ``first_frame..last_frame`` (inclusive) in contiguous chunks and merge them.

    With the defaults (one worker, no chunk size) ``task`` is called once on the whole range: the
    serial run. The merged result is the serial one for every ``workers`` / ``chunk_frames``.

    Args:
        task: ``task(first, last) -> (per-frame results, frame size)`` for a sub-range; the
            results must be picklable for the ``"process"`` back end.
        first_frame: First frame (inclusive).
        last_frame: Last frame (inclusive).
        workers: Parallel workers (1: in this process).
        chunk_frames: Chunk length in frames; default: one chunk for one worker, else balanced
            (:func:`default_chunk_frames`). Chunks are near-equal, no longer than this (before
            alignment).
        align: Start chunks on frames ``b`` with ``b % align == offset % align`` (see
            :func:`naps.video.split_range`), e.g. the keyframe spacing of the video.
        offset: See ``align``.
        backend: ``"process"`` or ``"thread"``, see :func:`run_chunks`.
        background: Optional job run while the chunks are decoded, see :func:`run_chunks`.

    Returns:
        ``(results, frame_size)``: the merged per-frame results and the frame size reported by
        the first chunk.
    """
    n_frames = last_frame - first_frame + 1
    if chunk_frames is None:
        chunk_frames = default_chunk_frames(n_frames, workers) if workers > 1 else max(n_frames, 1)
    parts = max(-(-n_frames // max(chunk_frames, 1)), 1)
    ranges = [
        (a, b - 1)
        for a, b in split_range(first_frame, last_frame + 1, parts, align=align, offset=offset)
    ] or [(first_frame, last_frame)]  # an empty range: the serial loop still reads once
    merged: dict[int, _V] = {}
    frame_size: tuple[int, int] | None = None
    for index, (part, size) in enumerate(
        run_chunks(task, ranges, workers, backend=backend, background=background)
    ):
        merged.update(part)
        if index == 0:
            frame_size = size
    return merged, frame_size


def _call_task(args: tuple[int, int]) -> object:
    """Worker entry point: run the inherited task on one range."""
    if _TASK is None:
        msg = "no task registered"
        raise RuntimeError(msg)
    return _TASK(*args)


def _star(task: Callable[[int, int], object]) -> Callable[[tuple[int, int]], object]:
    """Adapt ``task(start, stop)`` to a function of one ``(start, stop)`` tuple."""

    def run(args: tuple[int, int]) -> object:
        return task(*args)

    return run


def run_chunks(
    task: Callable[[int, int], _T],
    ranges: Sequence[tuple[int, int]],
    workers: int,
    *,
    backend: Backend = "process",
    cv2_threads: int = 1,
    background: Callable[[], None] | None = None,
) -> list[_T]:
    """Run ``task(start, stop)`` for every range and return the results in range order.

    With one worker (or one range) the task runs in this process, in order, and OpenCV keeps its
    own thread setting: this is the serial code path. If a task raises, the exception of the
    *first* range (in order) that failed propagates, like in a serial run; ranges after it are
    cancelled.

    Args:
        task: ``task(start, stop) -> result``. With the ``"process"`` back end it is inherited by
            fork (it may be a closure over unpicklable objects such as an OpenCV detector) and
            its *result* must be picklable.
        ranges: The ``(start, stop)`` pairs, in order.
        workers: Number of workers.
        backend: ``"process"`` (forked workers; only used on Linux, elsewhere it runs threads) or
            ``"thread"``.
        cv2_threads: ``cv2.setNumThreads`` value while the workers run (set before they are
            forked, and restored afterwards).
        background: Optional job to run while the ranges are being decoded: in a thread of this
            process once the workers exist (so they never fork a large heap), or after the ranges
            in a serial run. Use it for work that would otherwise run after the decode (for
            example loading the file the result is written to). An exception it raises propagates
            after the ranges are done.

    Returns:
        The results, one per range, in range order.

    Raises:
        ValueError: If ``backend`` is unknown or ``workers`` is smaller than 1.
    """
    if workers < 1:
        msg = f"workers must be at least 1, got {workers}"
        raise ValueError(msg)
    if backend not in ("process", "thread"):
        msg = f"unknown backend {backend!r}: choose 'process' or 'thread'"
        raise ValueError(msg)
    if workers == 1 or len(ranges) <= 1:
        serial = [task(a, b) for a, b in ranges]
        if background is not None:
            background()
        return serial

    previous_threads = cv2.getNumThreads()
    # OpenCV's thread count is set in THIS process before the workers exist: calling
    # cv2.setNumThreads inside a forked child deadlocks if the parent used OpenCV's thread pool
    # before (the child inherits a lock whose owner thread does not exist in the child).
    cv2.setNumThreads(cv2_threads)
    executor, call = _start_pool(task, min(workers, len(ranges)), backend)
    failure: list[BaseException] = []
    thread: threading.Thread | None = None
    try:
        results = executor.map(call, ranges)  # the workers exist from here on
        if background is not None:
            thread = threading.Thread(
                target=_run_guarded, args=(background, failure), name="naps-background", daemon=True
            )
            thread.start()
        out = cast("list[_T]", list(results))
        if thread is not None:
            thread.join()
        if failure:
            raise failure[0]
        return out
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        _release_task()
        cv2.setNumThreads(previous_threads)


def _run_guarded(job: Callable[[], None], failure: list[BaseException]) -> None:
    """Run ``job`` in a thread, keeping its exception for the caller."""
    try:
        job()
    except BaseException as exc:
        failure.append(exc)


def _start_pool(
    task: Callable[[int, int], object], n_workers: int, backend: Backend
) -> tuple[Executor, Callable[[tuple[int, int]], object]]:
    """Create the executor (forked processes if possible, else threads) and its call adaptor."""
    global _TASK  # noqa: PLW0603 (the forked workers read the task from this module global)
    # fork is used on Linux only: macOS crashes or warns when a multi-threaded process forks
    can_fork = (
        sys.platform.startswith("linux") and "fork" in multiprocessing.get_all_start_methods()
    )
    if backend == "process" and can_fork:
        _TASK = task
        return (
            ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=multiprocessing.get_context("fork"),
            ),
            _call_task,
        )
    return ThreadPoolExecutor(max_workers=n_workers), _star(task)


def _release_task() -> None:
    """Forget the task handed to the forked workers."""
    global _TASK  # noqa: PLW0603
    _TASK = None
