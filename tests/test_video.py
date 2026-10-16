import logging
import pickle

import cv2
import numpy as np
import pytest

from naps.video import DEFAULT_MAX_WALK, FastVideoReader, VideoReadError, _to_gray, split_range
from tests.video_helpers import (
    LIMITED_LUT,
    FakeSpec,
    install_fake_video,
    read_all_blue,
    textured_frames,
    write_colour_video,
    write_gray_video,
)

EXAMPLE_VIDEO = "tests/data/example.mp4"
ARUCO_VIDEO = "tests/data/example_ArUco_video.avi"


@pytest.fixture
def fake_video(monkeypatch, tmp_path):
    """Factory ``(spec) -> path`` of a video played by a scripted fake ``VideoCapture``."""
    return lambda spec: install_fake_video(monkeypatch, tmp_path, spec)


def _spec(n=10, **kwargs):
    return FakeSpec(textured_frames(n), **kwargs)


# ---------------------------------------------------------------- split_range


@pytest.mark.parametrize(
    ("start", "stop", "parts"), [(0, 10, 3), (5, 6, 4), (0, 1000, 7), (3, 3, 2)]
)
def test_split_range_tiles_exactly(start, stop, parts):
    chunks = split_range(start, stop, parts)
    assert chunks == sorted(chunks)
    covered = [i for a, b in chunks for i in range(a, b)]
    assert covered == list(range(start, stop))
    assert len(chunks) <= parts
    assert all(b > a for a, b in chunks)


def test_split_range_rejects_no_parts():
    with pytest.raises(ValueError, match="at least 1"):
        split_range(0, 10, 0)
    with pytest.raises(ValueError, match="at least 1"):
        split_range(0, 10, 2, align=0)


@pytest.mark.parametrize("seed", range(40))
def test_aligned_split_range_tiles_and_starts_on_the_alignment(seed):
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, 600))
    stop = start + int(rng.integers(0, 3000))
    parts = int(rng.integers(1, 20))
    align = int(rng.choice([1, 2, 5, 20, 250, 500]))
    offset = int(rng.integers(0, 3 * align))
    chunks = split_range(start, stop, parts, align=align, offset=offset)
    assert [i for a, b in chunks for i in range(a, b)] == list(range(start, stop))
    assert len(chunks) <= parts
    assert all(a % align == offset % align for a, _ in chunks[1:])
    assert all(b > a for a, b in chunks)


def test_keyframe_aligned_chunks_start_after_the_keyframes():
    chunks = split_range(0, 3000, 4, align=250, offset=25)
    assert [a % 250 for a, _ in chunks[1:]] == [25] * (len(chunks) - 1)
    assert chunks[0][0] == 0 and chunks[-1][1] == 3000
    # boundaries that collapse onto the same keyframe merge into one chunk
    assert split_range(0, 300, 6, align=250, offset=25) == [(0, 25), (25, 275), (275, 300)]
    # a boundary that falls outside the range after snapping is dropped
    assert split_range(0, 40, 4, align=250, offset=100) == [(0, 40)]


# ---------------------------------------------------------------- opening


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        FastVideoReader(tmp_path / "nope.mp4")


def test_unopenable_video_raises(fake_video):
    path = fake_video(_spec(opens=False))
    with pytest.raises(VideoReadError, match="cannot open"):
        FastVideoReader(path)


def test_luma_plane_limited_range_and_legacy_lut(fake_video):
    spec = _spec()
    reader = FastVideoReader(fake_video(spec))
    assert reader.luma_plane
    assert reader.pixel_range == "limited"
    assert (reader.frame_count, reader.fps, reader.width, reader.height) == (10, 10.0, 80, 64)
    frames = [f for _, f in reader.frames()]
    assert all(np.array_equal(f, g) for f, g in zip(frames, spec.frames, strict=True))
    assert reader.legacy_lut is not None
    assert np.array_equal(reader.legacy_lut, LIMITED_LUT[np.arange(256)])
    assert np.array_equal(reader.to_legacy(frames[0]), LIMITED_LUT[frames[0]])


def test_full_range_luma_needs_no_remap(fake_video):
    reader = FastVideoReader(fake_video(_spec(full_range=True)))
    assert reader.luma_plane
    assert reader.pixel_range == "full"
    frame = reader.read(0)
    assert frame is not None
    assert reader.to_legacy(frame) is frame
    assert reader.legacy_lut is not None
    assert np.array_equal(reader.legacy_lut, np.arange(256))


def test_unknown_remap_is_measured_not_assumed(fake_video):
    reader = FastVideoReader(fake_video(_spec(full_range=True, offset=5)))
    assert reader.pixel_range == "unknown"
    frame = reader.read(0)
    assert frame is not None
    expected = np.clip(frame.astype(np.int16) + 5, 0, 255).astype(np.uint8)
    assert np.array_equal(reader.to_legacy(frame), expected)


def test_i420_buffer_is_cut_to_its_luma_plane(fake_video):
    spec = _spec(raw="i420")
    reader = FastVideoReader(fake_video(spec))
    assert reader.luma_plane
    assert np.array_equal(reader.read(3), spec.frames[3])


@pytest.mark.parametrize("raw", ["odd", "float", "fail", "bgr"])
def test_unusable_raw_frames_fall_back_to_read(fake_video, raw):
    spec = _spec(raw=raw)
    path = fake_video(spec)
    reader = FastVideoReader(path)
    assert not reader.luma_plane
    assert reader.pixel_range == "full"
    assert np.array_equal(reader.read(2), LIMITED_LUT[spec.frames[2]])
    with pytest.raises(VideoReadError, match="faithful luma plane"):
        FastVideoReader(path, mode="y")


def test_empty_stream_falls_back(fake_video):
    reader = FastVideoReader(fake_video(_spec(fail_grab={0})))
    assert not reader.luma_plane
    assert reader.read() is None


def test_mismatching_luma_plane_is_rejected(fake_video, caplog):
    path = fake_video(_spec(shifted=True))
    with caplog.at_level(logging.WARNING, logger="naps.video"):
        reader = FastVideoReader(path)
    assert not reader.luma_plane
    assert "does not match" in caplog.text
    with pytest.raises(VideoReadError, match="faithful luma plane"):
        FastVideoReader(path, mode="y")


def test_colour_video_is_read_the_old_way(fake_video):
    spec = _spec(colour=True)
    path = fake_video(spec)
    reader = FastVideoReader(path)
    assert not reader.luma_plane
    assert np.array_equal(reader.read(1), LIMITED_LUT[spec.frames[1]])  # the blue channel
    forced = FastVideoReader(path, mode="y")
    assert forced.luma_plane
    assert forced.pixel_range == "unknown"
    assert forced.legacy_lut is None
    frame = forced.read(0)
    assert frame is not None
    with pytest.raises(VideoReadError, match="no luma to read"):
        forced.to_legacy(frame)


def test_without_calibration_the_frame_shape_is_trusted(fake_video):
    path = fake_video(_spec())
    reader = FastVideoReader(path, calibrate=False)
    assert reader.luma_plane
    assert reader.pixel_range == "unknown"
    assert reader.legacy_lut is None
    assert not FastVideoReader(fake_video(_spec(raw="odd")), calibrate=False).luma_plane


def test_missing_reference_frame_trusts_the_shape(fake_video, caplog):
    with caplog.at_level(logging.WARNING, logger="naps.video"):
        reader = FastVideoReader(fake_video(_spec(read_ok=False)))
    assert reader.luma_plane
    assert reader.legacy_lut is None
    assert reader.pixel_range == "unknown"
    assert "no reference frame" in caplog.text


def test_a_two_dimensional_frame_is_already_gray():
    plane = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert _to_gray(plane) is plane
    assert _to_gray(plane.astype(np.int64)).dtype == np.uint8
    assert np.array_equal(_to_gray(np.dstack([plane, plane + 1, plane + 2])), plane)


def test_decoder_threads_are_fixed_when_the_file_is_opened(fake_video, monkeypatch):
    path = fake_video(_spec())
    opened = []
    real = cv2.VideoCapture

    def spy(*args):
        opened.append(args)
        return real(*args)

    monkeypatch.setattr(cv2, "VideoCapture", spy)
    reader = FastVideoReader(path, threads=1)
    assert opened  # luma capture and the calibration reference, all with the thread limit
    assert all(a[1:] == (cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]) for a in opened)
    assert reader.read(2) is not None
    opened.clear()
    FastVideoReader(path)
    assert all(len(a) == 1 for a in opened)  # the default open is untouched


def test_unknown_mode_is_rejected(fake_video):
    with pytest.raises(ValueError, match="unknown mode"):
        FastVideoReader(fake_video(_spec()), mode="gray")  # type: ignore[arg-type]


def test_bgr_mode_never_uses_luma(fake_video):
    spec = _spec()
    reader = FastVideoReader(fake_video(spec), mode="bgr")
    assert not reader.luma_plane
    frame = reader.read(4)
    assert frame is not None
    assert np.array_equal(frame, LIMITED_LUT[spec.frames[4]])
    assert reader.to_legacy(frame) is frame


def test_frame_changing_shape_midstream_is_an_error_not_a_wrong_picture(fake_video):
    spec = _spec(odd_from=4)
    reader = FastVideoReader(fake_video(spec), calibrate=False)
    before = reader.read(3)
    assert before is not None
    assert before.shape == (64, 80)
    with pytest.raises(VideoReadError, match="frame 4 is not a luma plane"):
        reader.read(4)
    with pytest.raises(VideoReadError, match="not a luma plane"):
        list(FastVideoReader(fake_video(spec), calibrate=False).frames(3, 6))


# ---------------------------------------------------------------- reading


def test_frames_follow_range_semantics(fake_video):
    spec = _spec(12)
    reader = FastVideoReader(fake_video(spec))
    got = list(reader.frames(2, 11, 3))
    assert [i for i, _ in got] == [2, 5, 8]
    assert all(np.array_equal(f, spec.frames[i]) for i, f in got)
    assert [i for i, _ in reader.frames(0, 0)] == []
    assert reader.truncated_at is None


def test_frames_validate_arguments(fake_video):
    reader = FastVideoReader(fake_video(_spec()))
    with pytest.raises(ValueError, match="step"):
        next(reader.frames(0, 5, 0))
    with pytest.raises(ValueError, match="step"):
        next(reader.frames(-1, 5))


def test_stream_ending_early_sets_truncated_at(fake_video):
    spec = _spec(6)
    reader = FastVideoReader(fake_video(spec))
    got = [i for i, _ in reader.frames(0, 50)]
    assert got == list(range(6))
    assert reader.truncated_at == 6
    assert list(reader.frames(20, 30)) == []  # nothing left after a long jump past the end
    assert reader.truncated_at is not None


def test_failing_grab_in_the_middle_ends_the_iteration(fake_video):
    reader = FastVideoReader(fake_video(_spec(fail_grab={5})))
    assert [i for i, _ in reader.frames()] == [0, 1, 2, 3, 4]
    assert reader.truncated_at == 5
    # a walk that runs into the failure
    assert reader.read(8) is None
    assert reader.truncated_at is not None


def test_corrupt_frame_is_yielded_as_none_and_iteration_continues(fake_video):
    reader = FastVideoReader(fake_video(_spec(8, corrupt={3})))
    got = dict(reader.frames(0, 8))
    assert sorted(got) == list(range(8))
    assert got[3] is None
    assert got[4] is not None
    assert reader.truncated_at is None


def test_read_walks_short_jumps_and_seeks_otherwise(fake_video):
    spec = _spec(40)
    path = fake_video(spec)
    reader = FastVideoReader(path, max_walk=5)
    spec.log.clear()
    assert reader.read(3) is not None  # forward 3: walked
    assert [kind for kind, _ in spec.log].count("seek") == 0
    assert reader.read(5) is not None  # next frame after 3 is 4: skip one
    assert reader.read(30) is not None  # 24 frames ahead: seek
    assert ("seek", 30) in spec.log
    assert reader.read(2) is not None  # backwards: seek
    assert ("seek", 2) in spec.log
    assert reader.position == 3
    frame = reader.read()  # the next frame
    assert np.array_equal(frame, spec.frames[3])
    assert DEFAULT_MAX_WALK > 5


def test_read_default_walk_limit(fake_video):
    spec = _spec(40)
    reader = FastVideoReader(fake_video(spec))
    spec.log.clear()
    assert reader.read(30) is not None
    assert not [entry for entry in spec.log if entry[0] == "seek"]


def test_seek_errors(fake_video):
    spec = _spec(10)
    reader = FastVideoReader(fake_video(spec))
    with pytest.raises(ValueError, match="negative"):
        reader.seek(-1)
    spec.seek_error = 2
    with pytest.raises(VideoReadError, match="cannot seek to frame 5"):
        reader.seek(5)
    spec.seek_error = 0
    reader.seek(5)
    assert reader.position == 5


def test_seeking_past_the_end_is_the_end_of_the_stream(fake_video):
    reader = FastVideoReader(fake_video(_spec(10)))
    assert reader.read(5000) is None
    assert reader.truncated_at == 5000
    assert list(reader.frames(7000, 7005)) == []


def test_closed_reader_refuses_to_read(fake_video):
    path = fake_video(_spec())
    with FastVideoReader(path) as reader:
        assert reader.read() is not None
    reader.close()  # second close is fine
    with pytest.raises(VideoReadError, match="closed"):
        reader.read()
    with pytest.raises(VideoReadError, match="closed"):
        next(reader.frames())


def test_iteration_continues_from_the_current_position(fake_video):
    spec = _spec(10)
    reader = FastVideoReader(fake_video(spec))
    reader.seek(7)
    assert [i for i, _ in reader] == [7, 8, 9]


def test_pickle_roundtrip_keeps_position_and_calibration(fake_video):
    spec = _spec(10)
    reader = FastVideoReader(fake_video(spec))
    reader.read(4)
    clone = pickle.loads(pickle.dumps(reader))  # noqa: S301 - our own object
    assert clone.position == reader.position == 5
    assert clone.luma_plane
    assert clone.pixel_range == "limited"
    assert np.array_equal(clone.legacy_lut, reader.legacy_lut)
    assert np.array_equal(clone.read(), spec.frames[5])
    bgr_clone = pickle.loads(pickle.dumps(FastVideoReader(fake_video(spec), mode="bgr")))  # noqa: S301
    assert not bgr_clone.luma_plane
    assert bgr_clone.position == 0


# ---------------------------------------------------------------- real videos: order and count


def _expected(path):
    return read_all_blue(path)


def _same(reader, frame, reference):
    """The reader's frame, remapped to the intensities read() gives, equals ``reference``."""
    return frame is not None and np.array_equal(reader.to_legacy(frame), reference)


@pytest.mark.parametrize("mode", ["auto", "y", "bgr"])
def test_gray_video_matches_cv2_read(tmp_path, mode):
    path = write_gray_video(tmp_path / "gray.avi", textured_frames(24))
    expected = _expected(path)
    with FastVideoReader(path, mode=mode) as reader:
        got = list(reader)
        assert reader.luma_plane == (mode != "bgr")
    assert [i for i, _ in got] == list(range(len(expected)))
    for (_, frame), reference in zip(got, expected, strict=True):
        assert frame is not None
        assert frame.dtype == np.uint8
        assert _same(reader, frame, reference)


@pytest.mark.parametrize("seed", range(12))
def test_random_ranges_match_cv2_read(tmp_path, seed):
    """Property: any (start, stop, step) selects exactly the frames read() returns."""
    rng = np.random.default_rng(seed)
    path = write_gray_video(tmp_path / "gray.avi", textured_frames(30, seed=seed))
    expected = _expected(path)
    start = int(rng.integers(0, 28))
    stop = int(rng.integers(start, 34))
    step = int(rng.integers(1, 7))
    with FastVideoReader(path, max_walk=int(rng.integers(1, 6))) as reader:
        got = list(reader.frames(start, stop, step))
        order = rng.permutation(len(expected))[:10]
        random_access = {int(i): reader.read(int(i)) for i in order}
    wanted = list(range(start, min(stop, len(expected)), step))
    assert [i for i, _ in got] == wanted
    for i, frame in got:
        assert _same(reader, frame, expected[i])
    for i, frame in random_access.items():
        assert _same(reader, frame, expected[i])


def test_colour_video_is_read_like_read(tmp_path):
    frames = [np.dstack([f, f // 2, 255 - f]) for f in textured_frames(10)]
    path = write_colour_video(tmp_path / "colour.avi", frames)
    expected = _expected(path)
    with FastVideoReader(path) as reader:
        assert not reader.luma_plane
        got = [f for _, f in reader]
    assert len(got) == len(expected)
    assert all(np.array_equal(a, b) for a, b in zip(got, expected, strict=True))


def test_real_example_clip_matches_cv2_read():
    cap = cv2.VideoCapture(EXAMPLE_VIDEO)
    expected = []
    for _ in range(25):
        ok, frame = cap.read()
        assert ok
        expected.append(frame[:, :, 0])
    cap.release()
    with FastVideoReader(EXAMPLE_VIDEO) as reader:
        got = list(reader.frames(0, 25))
        strided = list(reader.frames(3, 25, 4))
        assert [i for i, _ in got] == list(range(25))
        assert all(_same(reader, f, e) for (_, f), e in zip(got, expected, strict=True))
        assert [i for i, _ in strided] == [3, 7, 11, 15, 19, 23]
        assert all(_same(reader, f, expected[i]) for i, f in strided)


def test_real_aruco_clip_frame_count_and_order():
    expected = read_all_blue(ARUCO_VIDEO)
    with FastVideoReader(ARUCO_VIDEO) as reader:
        got = [f for _, f in reader]
        assert reader.truncated_at == len(expected)
        assert len(got) == len(expected)
        assert all(_same(reader, a, b) for a, b in zip(got, expected, strict=True))


def test_luma_plane_is_accepted_at_any_brightness(fake_video):
    """The acceptance test is structural: bright video (big remap offsets) keeps the fast path."""
    bright = [np.full((64, 80), 210, dtype=np.uint8) for _ in range(3)]
    bright[0][:20] = 235
    bright[0][20:40] = 190
    reader = FastVideoReader(fake_video(FakeSpec(bright)))
    assert reader.luma_plane
    assert reader.pixel_range == "limited"
    assert reader.legacy_lut is not None
    assert reader.legacy_lut[210] == LIMITED_LUT[210]
    assert LIMITED_LUT[210] - 210 > 12  # a bound on the intensity difference would reject this


def test_decreasing_remap_is_rejected(fake_video):
    path = fake_video(_spec(inverted=True))
    assert not FastVideoReader(path).luma_plane
