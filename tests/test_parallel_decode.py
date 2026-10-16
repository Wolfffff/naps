"""Chunked / parallel decode: the merged result must equal the serial result exactly."""

from __future__ import annotations

import multiprocessing
import os
import sys
from concurrent.futures.process import BrokenProcessPool

import cv2
import numpy as np
import pytest

import naps.parallel_decode as pd_mod
from naps.matching import ChunkedDecodeSource, OpenCVDecodeSource, TagDecoderSource
from naps.parallel_decode import (
    available_cpus,
    decode_in_chunks,
    default_chunk_frames,
    run_chunks,
)
from naps.video import VideoReadError, open_capture, split_range

from ._perf_fixtures import (
    CROP,
    N_FRAMES,
    SIZE,
    coords,
    make_detector,
    make_tag_decoder,
    make_video,
    plain,
)

# Forked workers are exercised on Linux only (macOS warns about forking a multi-threaded process,
# Windows has no fork: there the "process" back end falls back to threads).
BACKENDS = ["process", "thread"] if sys.platform == "linux" else ["thread"]


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    return make_video(tmp_path_factory.mktemp("pd"))


@pytest.fixture(scope="module")
def detector():
    return make_detector()


def _inclusive(first, last, parts, **kw):
    """The inclusive ranges :func:`decode_in_chunks` derives from ``split_range``."""
    return [(a, b - 1) for a, b in split_range(first, last + 1, parts, **kw)]


def _square(a, b):
    return [x * x for x in range(a, b + 1)]


# ------------------------------------------------------------------------------ helpers
def test_default_chunk_frames():
    assert default_chunk_frames(72_000, 8) == 2250
    assert default_chunk_frames(100, 8) == 200  # never tiny
    assert default_chunk_frames(1000, 0) == 250  # zero workers treated as one


def test_available_cpus(monkeypatch):
    assert available_cpus() >= 1
    monkeypatch.delattr(os, "sched_getaffinity")
    assert available_cpus() >= 1


# ----------------------------------------------------------------------------- run_chunks
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_run_chunks_preserves_order(backend, workers):
    ranges = _inclusive(0, 99, 14)
    assert run_chunks(_square, ranges, workers, backend=backend) == [
        _square(a, b) for a, b in ranges
    ]


def test_run_chunks_validates_arguments():
    with pytest.raises(ValueError, match="workers"):
        run_chunks(_square, [(0, 1)], 0)
    with pytest.raises(ValueError, match="backend"):
        run_chunks(_square, [(0, 1)], 2, backend="gpu")  # type: ignore[arg-type]


def test_run_chunks_first_failing_range_wins():
    def task(a, b):
        if a >= 20:
            raise RuntimeError(f"boom {a}")
        return a

    for backend in BACKENDS:
        with pytest.raises(RuntimeError, match="boom 20"):
            run_chunks(task, _inclusive(0, 59, 6), 3, backend=backend)


def test_run_chunks_falls_back_to_threads_without_fork(monkeypatch):
    monkeypatch.setattr(multiprocessing, "get_all_start_methods", lambda: ["spawn"])
    before = cv2.getNumThreads()
    assert run_chunks(_square, [(0, 3), (4, 6)], 2, backend="process") == [
        _square(0, 3),
        _square(4, 6),
    ]
    assert cv2.getNumThreads() == before  # restored


def test_worker_hooks_run_in_process():
    # these run only inside forked workers during a real pool; call them directly for coverage
    try:
        pd_mod._TASK = None
        with pytest.raises(RuntimeError, match="no task"):
            pd_mod._call_task((0, 1))
        pd_mod._TASK = _square
        assert pd_mod._call_task((1, 3)) == [1, 4, 9]
    finally:
        pd_mod._TASK = None


@pytest.mark.skipif(sys.platform != "linux", reason="needs forked worker processes")
def test_a_crashed_worker_fails_the_run_instead_of_returning_partial_results():
    def task(a, b):
        if a >= 10:
            os._exit(1)  # what the kernel's OOM killer looks like to the pool
        return a

    with pytest.raises(BrokenProcessPool):
        run_chunks(task, _inclusive(0, 39, 4), 2, backend="process")


@pytest.mark.parametrize("backend", BACKENDS)
def test_background_job_runs_during_a_parallel_run(backend):
    seen = []
    ranges = _inclusive(0, 39, 4)
    got = run_chunks(_square, ranges, 2, backend=backend, background=lambda: seen.append(1))
    assert seen == [1]
    assert got == [_square(a, b) for a, b in ranges]


def test_background_job_runs_after_a_serial_run():
    order = []

    def task(a, b):
        order.append("task")
        return a

    run_chunks(task, [(0, 1)], 4, background=lambda: order.append("background"))
    assert order == ["task", "background"]


def test_background_exception_propagates_after_the_ranges():
    def boom():
        raise OSError("disk")

    with pytest.raises(OSError, match="disk"):
        run_chunks(_square, _inclusive(0, 39, 4), 2, backend="thread", background=boom)


# ------------------------------------------------------------------------ decode_in_chunks
def _fake_task(a, b):
    return {f: f * 10 for f in range(a, b + 1)}, (a, 0)


@pytest.mark.parametrize("seed", range(25))
def test_decode_in_chunks_merges_every_frame_once_in_order(seed):
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, 40))
    last = first + int(rng.integers(0, 300))
    workers = int(rng.integers(1, 4))
    chunk = int(rng.choice([1, 7, 50, 1000, 0])) or None
    align = int(rng.choice([1, 5, 50]))
    offset = int(rng.integers(0, 60))
    got, size = decode_in_chunks(
        _fake_task,
        first,
        last,
        workers=workers,
        chunk_frames=chunk,
        align=align,
        offset=offset,
        backend="thread",
    )
    assert list(got) == list(range(first, last + 1))
    assert got == {f: f * 10 for f in range(first, last + 1)}
    assert size == (first, 0)  # the size of the first chunk


def test_an_empty_range_still_calls_the_task_once():
    got, size = decode_in_chunks(_fake_task, 5, 4)
    assert got == {}
    assert size == (5, 0)


# ------------------------------------------------------------------------------- sources
def _serial(source, first=0, last=N_FRAMES - 1):
    return source.decode(first, last, coords(first, last))


def _legacy_source(video, detector, threads=None):
    return OpenCVDecodeSource(str(video), detector, CROP, threads)


def _fast_source(video, threads=None):
    return TagDecoderSource(video, make_tag_decoder(), crop_half=CROP, reader_threads=threads)


@pytest.mark.parametrize("kind", ["legacy", "fast"])
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("workers", "chunk", "align"), [(2, None, 1), (3, 5, 1), (4, 1, 1), (2, 100, 1), (3, 9, 6)]
)
def test_parallel_decode_equals_serial(  # noqa: PLR0917
    video, detector, kind, backend, workers, chunk, align
):
    def build(threads=None):
        return (
            _legacy_source(video, detector, threads)
            if kind == "legacy"
            else _fast_source(video, threads)
        )

    expected, size = _serial(build())
    parallel = ChunkedDecodeSource(
        build(1 if workers > 1 else None),
        workers=workers,
        chunk_frames=chunk,
        align=align,
        offset=2,
        backend=backend,
    )
    got, got_size = _serial(parallel)
    assert plain(got) == plain(expected)
    assert got_size == size == (SIZE, SIZE)
    # sanity: the video really varies, so a seam error would have been noticed
    assert len({tuple(map(tuple, row.values())) for row in got.values()}) > 10


@pytest.mark.parametrize("kind", ["legacy", "fast"])
def test_parallel_decode_with_nonzero_start(video, detector, kind):
    def build():
        return _legacy_source(video, detector, 1) if kind == "legacy" else _fast_source(video, 1)

    expected = _serial(build(), 9, 40)
    got = _serial(ChunkedDecodeSource(build(), workers=3, chunk_frames=7), 9, 40)
    assert plain(got[0]) == plain(expected[0])
    assert got[1] == expected[1]
    assert list(got[0]) == list(range(9, 41))


@pytest.mark.parametrize("kind", ["legacy", "fast"])
def test_past_the_end_error_is_the_same(video, detector, kind):
    # track positions beyond the last readable frame raise the same error as the serial run
    error = RuntimeError if kind == "legacy" else VideoReadError
    for workers in (1, 3):
        inner = _legacy_source(video, detector) if kind == "legacy" else _fast_source(video)
        source = ChunkedDecodeSource(inner, workers=workers, chunk_frames=10, backend="thread")
        with pytest.raises(error, match=rf"frame {N_FRAMES}"):
            source.decode(0, N_FRAMES + 8, coords(0, N_FRAMES + 8))


def test_an_unreadable_video_reports_the_same_failure_with_one_decoder_thread(tmp_path):
    broken = tmp_path / "broken.avi"
    broken.write_bytes(b"not a video")
    inner = OpenCVDecodeSource(str(broken), lambda img: [None], CROP, 1)
    for workers in (1, 2):
        source = ChunkedDecodeSource(inner, workers=workers, chunk_frames=3, backend="thread")
        with pytest.raises(RuntimeError, match="frame 0"):
            source.decode(0, 5, {f: {0: (40.0, 40.0)} for f in range(6)})


def test_the_background_job_runs_with_the_chunked_source(video, detector):
    seen = []
    source = ChunkedDecodeSource(
        _legacy_source(video, detector, 1),
        workers=2,
        chunk_frames=10,
        background=lambda: seen.append(1),
    )
    _serial(source)
    assert seen == [1]


# ------------------------------------------------------------------------- open_capture
def test_open_capture_limits_the_decoder_threads(video):
    for threads in (None, 1):
        cap = open_capture(str(video), threads)
        assert cap.isOpened()
        ok, frame = cap.read()
        assert ok
        assert frame.shape[:2] == (SIZE, SIZE)
        cap.release()


def test_open_capture_without_the_thread_property_or_on_failure(video, tmp_path, monkeypatch):
    monkeypatch.delattr(cv2, "CAP_PROP_N_THREADS", raising=False)
    cap = open_capture(str(video), 1)
    assert cap.isOpened()
    cap.release()
    monkeypatch.undo()
    missing = open_capture(str(tmp_path / "missing.avi"), 1)
    assert not missing.isOpened()
    missing.release()
