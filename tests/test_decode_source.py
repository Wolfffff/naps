"""The ``DecodeSource`` protocol and its OpenCV implementation (``decode_video``)."""

import numpy as np
import pytest

from naps.aruco import ArUcoModel
from naps.matching import DecodeSource, Matching, OpenCVDecodeSource, decode_video

VIDEO_PATH = "tests/data/example_ArUco_video.avi"
N_VIDEO_FRAMES = 15
COORDS = {0: (225, 225), 1: (925, 225), 2: (575, 575), 3: (225, 925), 4: (925, 925)}
ARUCO_PARAMS = {
    "adaptiveThreshWinSizeMin": 3,
    "adaptiveThreshWinSizeMax": 10,
    "adaptiveThreshWinSizeStep": 3,
    "adaptiveThreshConstant": 10,
    "perspectiveRemoveIgnoredMarginPerCell": 0.1,
    "errorCorrectionRate": 0.1,
}


@pytest.fixture(scope="module")
def detector():
    return ArUcoModel.withTagSet("DICT_4X4_100", **ARUCO_PARAMS).detect


def coords_for(last_frame, positions=COORDS, first_frame=0):
    return {f: dict(positions) for f in range(first_frame, last_frame + 1)}


def test_decode_video_equals_the_legacy_decode_stage(detector):
    last = N_VIDEO_FRAMES - 1
    coords = coords_for(last)
    matcher = Matching(
        VIDEO_PATH, 0, last, marker_detector=detector, aruco_crop_size=200,
        half_rolling_window_size=7, tag_node_dict=coords,
    )  # fmt: skip
    legacy = matcher._decode_video()
    decodes, frame_size = decode_video(VIDEO_PATH, 0, last, coords, 200, detector)
    assert decodes == legacy
    assert frame_size == matcher.frame_size
    assert frame_size is not None
    assert [decodes[0][k] for k in COORDS] == [[1], [2], [3], [4], [5]]


def test_opencv_decode_source_satisfies_the_protocol_and_decodes(detector):
    source: DecodeSource = OpenCVDecodeSource(VIDEO_PATH, detector, 200)
    decodes, frame_size = source.decode(2, 5, coords_for(5, first_frame=2))
    assert sorted(decodes) == [2, 3, 4, 5]
    assert decodes[3][4] == [5]
    assert frame_size is not None
    assert source.crop_size == 200
    assert source.video_filename == VIDEO_PATH


def test_opencv_decode_source_rejects_a_missing_video(tmp_path, detector):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        OpenCVDecodeSource(str(tmp_path / "nothing.avi"), detector, 100)


def test_frames_past_the_end_are_tolerated_while_no_position_is_there(detector):
    last = N_VIDEO_FRAMES + 3
    coords = {f: (COORDS if f < N_VIDEO_FRAMES else {0: (np.nan, np.nan)}) for f in range(last + 1)}
    decodes, frame_size = decode_video(VIDEO_PATH, 0, last, coords, 200, detector)
    assert sorted(decodes) == list(range(last + 1))
    assert decodes[last] == {0: [None]}
    assert frame_size is not None


def test_a_position_in_an_unreadable_frame_names_the_frame(detector):
    last = N_VIDEO_FRAMES + 1
    with pytest.raises(RuntimeError, match=rf"frame {N_VIDEO_FRAMES}: .*could not be read"):
        decode_video(VIDEO_PATH, 0, last, coords_for(last), 200, detector)


def test_a_first_frame_beyond_the_end_gives_no_frame_size(detector):
    coords = {N_VIDEO_FRAMES + 50: {0: (np.nan, np.nan)}}
    decodes, frame_size = decode_video(
        VIDEO_PATH, N_VIDEO_FRAMES + 50, N_VIDEO_FRAMES + 50, coords, 200, detector
    )
    assert frame_size is None
    assert decodes == {N_VIDEO_FRAMES + 50: {0: [None]}}


def test_a_missing_frame_in_coords_is_a_key_error(detector):
    with pytest.raises(KeyError):
        decode_video(VIDEO_PATH, 0, 3, coords_for(2), 200, detector)


def test_a_pure_python_source_can_stand_in():
    class Scripted:
        def decode(self, first_frame, last_frame, coords):
            return {f: {s: [7] for s in coords[f]} for f in range(first_frame, last_frame + 1)}, (
                10,
                20,
            )

    source: DecodeSource = Scripted()
    decodes, size = source.decode(0, 1, {0: {3: (1.0, 2.0)}, 1: {}})
    assert decodes == {0: {3: [7]}, 1: {}}
    assert size == (10, 20)
