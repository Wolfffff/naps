import logging
from collections import defaultdict

import cv2
import numpy as np
import pytest

from naps.aruco import ArUcoModel
from naps.closed_set import EliminationResult
from naps.matching import MatchFrame, Matching
from naps.quadrants import QuadrantElimination, QuadrantScope

IMAGE_PATH = "tests/data/example_ArUco_image.jpg"
VIDEO_PATH = "tests/data/example_ArUco_video.avi"
N_VIDEO_FRAMES = 15

# Centroid of each of the five tags in the example image/video; tag ids are track + 1
COORDS = {
    0: (225, 225),
    1: (925, 225),
    2: (575, 575),
    3: (225, 925),
    4: (925, 925),
}
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


def make_matcher(detector, last_frame=N_VIDEO_FRAMES - 1, tag_node_dict=None, **kwargs):
    if tag_node_dict is None:
        tag_node_dict = dict.fromkeys(range(last_frame + 1), COORDS)
    return Matching(
        VIDEO_PATH,
        kwargs.pop("first_frame", 0),
        last_frame,
        marker_detector=detector,
        aruco_crop_size=200,
        half_rolling_window_size=kwargs.pop("half_rolling_window_size", 7),
        tag_node_dict=tag_node_dict,
        **kwargs,
    )


# ---------------------------------------------------------------------- MatchFrame


def test_matchframe_from_cv2_keeps_frame_and_flag():
    image = cv2.imread(IMAGE_PATH, 0)
    frame = MatchFrame.fromCV2(True, image)
    assert frame.frame_exists is True
    assert frame.frame is image
    assert frame.frame_images == {}


def test_matchframe_from_cv2_requires_both_arguments():
    image = cv2.imread(IMAGE_PATH, 0)
    for args in [(), (True,), (image,)]:
        with pytest.raises(TypeError):
            MatchFrame.fromCV2(*args)


def test_matchframe_crops_one_image_per_track():
    match_frame = MatchFrame.fromCV2(True, cv2.imread(IMAGE_PATH))
    match_frame.cropMarkerWithCoordsArray(COORDS, 200)

    assert list(match_frame.frame_images) == [0, 1, 2, 3, 4]
    for match_image in match_frame.frame_images.values():
        assert match_image is not None
        assert match_image.shape == (400, 400)


@pytest.mark.parametrize(
    ("coords", "shape"),
    [
        ((0, 0), (200, 200)),  # clipped at the top left corner
        ((-50.7, 10.2), (210, 150)),  # slightly outside the frame; int() truncates toward zero
        ((1149.9, 1149.9), None),  # bottom right: crop is clipped to the last pixel
        (np.array([225.4, 225.9]), (400, 400)),  # truncated toward zero, numpy input
    ],
)
def test_matchframe_crop_is_clipped_to_the_frame(coords, shape):
    image = cv2.imread(IMAGE_PATH)
    assert image is not None
    match_frame = MatchFrame.fromCV2(True, image)
    match_frame.cropMarkerWithCoordsArray({"t": coords}, 200)

    crop = match_frame.frame_images["t"]
    assert crop is not None
    height, width = image.shape[:2]
    assert crop.shape[0] <= min(400, height) and crop.shape[1] <= min(400, width)
    if shape is not None:
        assert crop.shape == shape


def test_matchframe_crop_far_outside_the_frame_is_empty():
    match_frame = MatchFrame.fromCV2(True, cv2.imread(IMAGE_PATH))
    match_frame.cropMarkerWithCoordsArray({"t": (-5000, -5000)}, 200)
    crop = match_frame.frame_images["t"]
    assert crop is not None
    assert 0 in crop.shape


@pytest.mark.parametrize("coords", [(np.nan, 5.0), (5.0, np.nan), (np.nan, np.nan)])
def test_matchframe_nan_coordinates_give_no_image(coords):
    match_frame = MatchFrame.fromCV2(True, cv2.imread(IMAGE_PATH))
    match_frame.cropMarkerWithCoordsArray({"t": coords}, 200)
    assert match_frame.frame_images == {"t": None}


def test_matchframe_without_image_raises_unless_coordinates_are_nan():
    unreadable = MatchFrame.fromCV2(False, None)
    unreadable.cropMarkerWithCoordsArray({"t": (np.nan, np.nan)}, 200)  # nothing to crop
    assert unreadable.frame_images == {"t": None}
    with pytest.raises(RuntimeError, match="could not be read"):
        unreadable.cropMarkerWithCoordsArray({"t": (5.0, 5.0)}, 200)


def test_matchframe_return_marker_tags(detector):
    match_frame = MatchFrame.fromCV2(True, cv2.imread(IMAGE_PATH))
    match_frame.cropMarkerWithCoordsArray(COORDS, 200)
    match_frame.frame_images["gone"] = None

    match_dict = match_frame.returnMarkerTags(detector)

    assert list(match_dict) == [0, 1, 2, 3, 4, "gone"]
    assert list(match_dict.values()) == [[1], [2], [3], [4], [5], [None]]


def test_matchframe_return_marker_tags_keeps_every_decode():
    match_frame = MatchFrame.fromCV2(True, cv2.imread(IMAGE_PATH))
    match_frame.cropMarkerWithCoordsArray({"a": (225, 225)}, 50)
    assert match_frame.returnMarkerTags(lambda _img: [3, 4, None]) == {"a": [3, 4, None]}


# ----------------------------------------------------------------------- Matching


def test_matching_constructs(detector):
    matcher = make_matcher(detector)
    assert matcher.closed_set is None
    assert matcher.position_dict is matcher.tag_node_dict
    assert matcher.elimination_result is None


def test_matching_requires_an_existing_video(detector):
    with pytest.raises(TypeError):
        Matching()  # ty: ignore[missing-argument]
    with pytest.raises(FileNotFoundError, match="does_not_exist"):
        Matching(
            "tests/data/does_not_exist.avi",
            0,
            14,
            marker_detector=detector,
            aruco_crop_size=200,
            half_rolling_window_size=7,
            tag_node_dict={},
        )


def test_matching_closed_set_and_quadrants_are_exclusive(detector):
    scope = QuadrantScope(tag_sets={"all": [1, 2]}, rois={"all": (0, 0, 9, 9)})
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_matcher(detector, closed_set=[1, 2], quadrants=scope)


def test_matching_warns_about_ignored_arguments(detector):
    with pytest.warns(DeprecationWarning, match="min_sleap_score"):
        make_matcher(detector, min_sleap_score=0.3)
    with pytest.warns(UserWarning, match="typo_option"):
        make_matcher(detector, typo_option=1)


def test_matching_normalises_the_closed_set(detector):
    matcher = make_matcher(detector, closed_set=[np.int64(5), 3.0, "4", 3])
    assert matcher.closed_set == [3, 4, 5]
    with pytest.raises(ValueError, match="not a valid tag id"):
        make_matcher(detector, closed_set=[3, 4.5])


def test_match_assigns_every_tag(detector):
    matcher = make_matcher(detector)

    frame_match_dict = matcher.match()

    assert len(frame_match_dict) == N_VIDEO_FRAMES
    assert sorted(frame_match_dict) == list(range(N_VIDEO_FRAMES))
    assert list(frame_match_dict[7]) == [0, 1, 2, 3, 4]
    # tags are plain Python ints (the old `[[1], ...]` only held through numpy broadcasting)
    assert list(frame_match_dict[7].values()) == [1, 2, 3, 4, 5]
    assert all(type(v) is int for v in frame_match_dict[7].values())
    assert matcher.frame_size is not None
    assert matcher.pre_elimination is frame_match_dict
    assert matcher.elimination_result is None


def test_match_head_frames_have_no_identity(detector):
    # half_rolling_window_size=7: frames 0..6 are never assigned (documented limitation)
    frame_match_dict = make_matcher(detector).match()
    assert all(not frame_match_dict[f] for f in range(7))
    assert all(len(frame_match_dict[f]) == 5 for f in range(7, N_VIDEO_FRAMES))


def test_match_tolerates_frames_past_the_end_without_instances(detector):
    # The video has 15 frames; frames 15..19 are requested but hold no tracked instances
    tag_node_dict = defaultdict(dict, dict.fromkeys(range(N_VIDEO_FRAMES), COORDS))
    matcher = make_matcher(detector, last_frame=19, tag_node_dict=tag_node_dict)

    frame_match_dict = matcher.match()

    assert list(frame_match_dict[10].values()) == [1, 2, 3, 4, 5]
    assert not frame_match_dict[19]


def test_match_past_the_end_with_instances_is_a_clear_error(detector):
    tag_node_dict = dict.fromkeys(range(20), COORDS)  # instances in frames the video lacks
    matcher = make_matcher(detector, last_frame=19, tag_node_dict=tag_node_dict)

    with pytest.raises(RuntimeError, match=r"frame 15.*could not be read"):
        matcher.match()


def test_match_with_closed_set_runs_elimination(detector):
    matcher = make_matcher(detector, closed_set=[1, 2, 3, 4, 5])

    frame_match_dict = matcher.match()

    assert list(frame_match_dict[10].values()) == [1, 2, 3, 4, 5]
    assert isinstance(matcher.elimination_result, EliminationResult)
    assert matcher.pre_elimination is not frame_match_dict


def test_match_closed_set_drops_foreign_decodes(detector):
    matcher = make_matcher(detector, closed_set=[1, 2, 3, 4])

    frame_match_dict = matcher.match()

    # Track 4 only ever decodes tag 5, which is outside the closed set: abstain, never guess
    assert 4 not in frame_match_dict[10]
    assert list(frame_match_dict[10].values()) == [1, 2, 3, 4]


def test_match_closed_set_without_elimination(detector):
    matcher = make_matcher(detector, closed_set=[1, 2, 3, 4, 5], eliminate=False)

    matcher.match()

    assert matcher.elimination_result is None


def test_match_warns_when_the_closed_set_is_much_smaller_than_the_tracks(detector, caplog):
    matcher = make_matcher(detector, closed_set=[1, 2])
    with caplog.at_level(logging.WARNING, logger="naps.matching"):
        matcher.match()
    assert "median of 5.0 tracks" in caplog.text


def test_match_with_quadrant_scope(detector):
    scope = QuadrantScope(
        tag_sets={"all": [1, 2, 3, 4, 5]}, rois={"all": (-np.inf, -np.inf, np.inf, np.inf)}
    )
    matcher = make_matcher(detector, quadrants=scope)

    frame_match_dict = matcher.match()

    assert list(frame_match_dict[10].values()) == [1, 2, 3, 4, 5]
    assert isinstance(matcher.elimination_result, QuadrantElimination)
    assert scope.frame_size == matcher.frame_size


def test_match_infers_quadrant_tag_sets_from_the_decodes(detector):
    scope = QuadrantScope(
        infer=True,
        expected_size=5,
        min_support=3,
        rois={"all": (-np.inf, -np.inf, np.inf, np.inf)},
    )
    matcher = make_matcher(detector, quadrants=scope)

    frame_match_dict = matcher.match()

    assert scope.active == {"all": [1, 2, 3, 4, 5]}
    assert list(frame_match_dict[10].values()) == [1, 2, 3, 4, 5]


def test_match_quadrants_without_elimination_returns_cost_matrix_result(detector):
    scope = QuadrantScope(
        tag_sets={"all": [1, 2, 3, 4, 5]}, rois={"all": (-np.inf, -np.inf, np.inf, np.inf)}
    )
    matcher = make_matcher(detector, quadrants=scope, eliminate=False)

    frame_match_dict = matcher.match()

    assert frame_match_dict is matcher.pre_elimination
    assert matcher.elimination_result is None


def test_match_quadrants_need_a_readable_frame(detector):
    scope = QuadrantScope(tag_sets={"all": [1]}, rois={"all": (0, 0, 9, 9)})
    matcher = make_matcher(
        detector,
        first_frame=100,
        last_frame=104,
        tag_node_dict=defaultdict(dict),
        quadrants=scope,
        half_rolling_window_size=1,
    )

    with pytest.raises(RuntimeError, match="frame size"):
        matcher.match()
