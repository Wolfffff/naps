from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from naps.aruco import (
    ARUCO_TAG_SETS,
    CORNER_REFINEMENTS,
    ArUcoModel,
    Detection,
    closed_set_correction_bits,
)

IMAGE_PATH = "tests/data/example_ArUco_image.jpg"


def _params(**overrides):
    params = {
        "adaptiveThreshWinSizeMin": 10,
        "adaptiveThreshWinSizeMax": 10,
        "adaptiveThreshWinSizeStep": 10,
        "adaptiveThreshConstant": 10,
        "perspectiveRemoveIgnoredMarginPerCell": 0.1,
        "errorCorrectionRate": 0.1,
    }
    params.update(overrides)
    return params


def _model(tag_set="DICT_4X4_100", **kwargs):
    return ArUcoModel.withTagSet(
        tag_set,
        **_params(
            adaptiveThreshWinSizeMin=3,
            adaptiveThreshWinSizeMax=23,
            adaptiveThreshConstant=7,
            errorCorrectionRate=0.6,
        ),
        **kwargs,
    )


def _marker_image(tag_set, tag_id, size=200):
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, tag_set))
    marker = cv2.aruco.generateImageMarker(aruco_dict, tag_id, size)
    return np.pad(marker, 40, constant_values=255)


@pytest.mark.parametrize("tag_set", ARUCO_TAG_SETS)
def test_every_tag_set_builds(tag_set):
    model = ArUcoModel.withTagSet(tag_set, **_params())
    model.build()
    assert model.model_built
    assert model.aruco_dict is not None


def test_unknown_tag_set_raises_value_error():
    with pytest.raises(ValueError, match="DICT_4X4_FAIL"):
        ArUcoModel.withTagSet("DICT_4X4_FAIL", **_params()).build()


def test_missing_tag_set_raises():
    with pytest.raises(TypeError):
        ArUcoModel.withTagSet(**_params())


@pytest.mark.parametrize(
    ("param", "value"),
    [
        ("adaptiveThreshWinSizeMin", 0.1),
        ("adaptiveThreshWinSizeMax", 0.1),
        ("adaptiveThreshWinSizeStep", 0.1),
        ("adaptiveThreshConstant", "10"),
        ("perspectiveRemoveIgnoredMarginPerCell", "10"),
        ("errorCorrectionRate", "10"),
    ],
)
def test_wrongly_typed_parameters_fail_at_build(param, value):
    model = ArUcoModel.withTagSet("DICT_4X4_100", **_params(**{param: value}))
    with pytest.raises((cv2.error, TypeError)):
        model.build()


@pytest.mark.parametrize(
    ("coords", "tag"),
    [
        ([100, 350, 100, 350], 1),
        ([100, 350, 800, 1050], 2),
        ([450, 700, 450, 700], 3),
        ([800, 1050, 100, 350], 4),
        ([800, 1050, 800, 1050], 5),
    ],
)
def test_detect_example_image(coords, tag):
    model = ArUcoModel.withTagSet(
        "DICT_4X4_100",
        **_params(adaptiveThreshWinSizeMin=3, adaptiveThreshWinSizeStep=3),
    )
    model.build()
    aruco_image = cv2.imread(IMAGE_PATH, 0)
    assert aruco_image is not None
    tag_image = aruco_image[coords[0] - 100 : coords[1] + 100, coords[2] - 100 : coords[3] + 100]

    tags = model.detect(tag_image)
    assert tags[0] == tag
    assert all(type(t) is int for t in tags)


def test_detects_rendered_marker_and_abstains_on_blank_canvas():
    marker = _marker_image("DICT_5X5_50", 17, size=120)
    canvas = np.full((300, 300), 200, dtype=np.uint8)
    canvas[90:210, 90:210] = marker[40:160, 40:160]
    model = _model("DICT_5X5_50")

    assert model.detect(canvas) == [17]
    assert model.detect(np.full((300, 300), 200, dtype=np.uint8)) == [None]


@pytest.mark.parametrize("shape", [(0, 10), (10, 0), (0, 0)])
def test_empty_image_returns_none_without_building(shape):
    model = _model()
    assert model.detect(np.zeros(shape, dtype=np.uint8)) == [None]
    assert not model.model_built


def test_marker_subset_returns_real_tag_ids():
    img = _marker_image("DICT_4X4_100", 17)
    assert _model().detect(img) == [17]
    # subset containing the tag: decoded index is mapped back to the tag id (not the subset index)
    assert _model(tag_subset_list=[3, 17, 42]).detect(img) == [17]
    # subset excluding the tag: the decoder cannot return it
    assert _model(tag_subset_list=[3, 42, 60]).detect(img) == [None]


def test_marker_subset_accepts_numpy_ints_and_normalises_them():
    model = _model(tag_subset_list=np.array([3, 17, 42]))
    assert model.tag_subset_list == [3, 17, 42]
    assert all(type(t) is int for t in model.tag_subset_list)
    assert _model(tag_subset_list=[]).tag_subset_list is None


def test_marker_subset_validates_ids():
    with pytest.raises(ValueError, match="not a valid tag id"):
        _model(tag_subset_list=[1, 2.5])
    with pytest.raises(ValueError, match="duplicate"):
        _model(tag_subset_list=[1, 1])
    with pytest.raises(ValueError, match="not in DICT_4X4_100"):
        _model(tag_subset_list=[1, 500]).build()
    with pytest.raises(ValueError, match="not in DICT_4X4_100"):
        _model(tag_subset_list=[-1, 5]).build()


def test_extra_keyword_arguments_are_ignored():
    assert _model(unknown_option=3).tag_set == "DICT_4X4_100"


# ---- the OpenCV < 4.7 API paths, simulated by hiding the new API of this build


def test_old_api_detection_function_is_used_without_arucodetector(monkeypatch):
    calls = []

    def fake_detect_markers(img, dictionary, parameters=None):
        calls.append((img.shape, parameters is not None))
        # OpenCV 4.x shapes the ids (N, 1)
        return [np.zeros((1, 4, 2))] * 2, np.array([[7], [9]]), []

    monkeypatch.delattr(cv2.aruco, "ArucoDetector")
    monkeypatch.setattr(cv2.aruco, "detectMarkers", fake_detect_markers, raising=False)
    model = _model()
    assert model.detect(np.zeros((20, 20), dtype=np.uint8)) == [7, 9]
    assert model.aruco_detector is None
    assert calls == [((20, 20), True)]

    monkeypatch.setattr(cv2.aruco, "detectMarkers", lambda *a, **k: ([], None, []), raising=False)
    assert model.detect(np.zeros((20, 20), dtype=np.uint8)) == [None]


def test_detector_returning_flat_ids_and_subset_mapping(monkeypatch):
    # OpenCV 5.x shapes the ids (N,); with a subset, indices map back to the real ids
    model = _model(tag_subset_list=[40, 41, 42])
    model.build()
    assert model.subset_dict == {0: 40, 1: 41, 2: 42}

    class FakeDetector:
        def detectMarkers(self, img):  # noqa: N802 - OpenCV's name
            return [np.zeros((1, 4, 2))] * 2, np.array([2, 0]), []

    monkeypatch.setattr(model, "aruco_detector", FakeDetector())
    assert model.detect(np.zeros((20, 20), dtype=np.uint8)) == [42, 40]


def test_old_api_detector_parameters_and_dictionary(monkeypatch):
    new_parameters = cv2.aruco.DetectorParameters
    new_dictionary = cv2.aruco.getPredefinedDictionary
    calls = []

    def missing(*_args, **_kwargs):
        raise AttributeError

    def parameters_create():
        calls.append("DetectorParameters_create")
        return new_parameters()

    def dictionary_get(dict_id):
        calls.append("Dictionary_get")
        return new_dictionary(dict_id)

    monkeypatch.setattr(cv2.aruco, "DetectorParameters", missing)
    monkeypatch.setattr(cv2.aruco, "getPredefinedDictionary", missing)
    monkeypatch.setattr(cv2.aruco, "DetectorParameters_create", parameters_create, raising=False)
    monkeypatch.setattr(cv2.aruco, "Dictionary_get", dictionary_get, raising=False)
    monkeypatch.delattr(cv2.aruco, "ArucoDetector")

    model = _model()
    model.build()
    assert calls == ["Dictionary_get", "DetectorParameters_create"]
    assert model.model_built


def test_old_api_custom_dictionary_for_subsets(monkeypatch):
    real = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

    def new_dictionary_unsupported(*_args, **_kwargs):
        raise TypeError

    created = []

    def custom_dictionary(n_markers, marker_size, max_correction_bits):
        created.append((n_markers, marker_size, max_correction_bits))
        return SimpleNamespace(bytesList=None)

    monkeypatch.setattr(cv2.aruco, "Dictionary", new_dictionary_unsupported)
    monkeypatch.setattr(cv2.aruco, "custom_dictionary", custom_dictionary, raising=False)

    model = _model(tag_subset_list=[3, 17])
    subset = model._subsetArucoDict(real, [3, 17])
    assert created == [(0, real.markerSize, 1)]
    assert subset.bytesList.shape[0] == 2
    assert model.subset_dict == {0: 3, 1: 17}


# ---- detect_detailed: geometry and confidence


def test_detection_geometry_properties():
    corners = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    detection = Detection(3, corners, 0.9)
    assert detection.center == (5.0, 5.0)
    assert detection.side == 10.0
    assert np.isnan(Detection(3, corners).confidence)


def test_detect_detailed_reports_position_size_and_confidence():
    img = _marker_image("DICT_4X4_100", 17)  # 200 px marker, 40 px quiet zone: centre (140, 140)
    model = _model()
    [detection] = model.detect_detailed(img)
    assert detection.tag == 17
    assert type(detection.tag) is int
    assert detection.corners.shape == (4, 2)
    assert abs(detection.center[0] - 139.5) < 2
    assert abs(detection.center[1] - 139.5) < 2
    assert abs(detection.side - 200) < 4
    assert 0.0 <= detection.confidence <= 1.0
    assert model.detect(img) == [17]  # the legacy method is unaffected


def test_detect_detailed_maps_subset_ids_and_handles_empty_input():
    img = _marker_image("DICT_4X4_100", 17)
    assert [d.tag for d in _model(tag_subset_list=[3, 17, 42]).detect_detailed(img)] == [17]
    assert _model(tag_subset_list=[3, 42]).detect_detailed(img) == []
    model = _model()
    assert model.detect_detailed(np.zeros((0, 5), dtype=np.uint8)) == []
    assert not model.model_built
    assert model.detect_detailed(np.full((200, 200), 200, dtype=np.uint8)) == []


def test_detect_detailed_without_confidence_support(monkeypatch):
    model = _model()
    model.build()
    real = model.aruco_detector

    class NoConfidence:
        def detectMarkers(self, img):  # noqa: N802 - OpenCV's name
            return real.detectMarkers(img)

    monkeypatch.setattr(model, "aruco_detector", NoConfidence())
    [detection] = model.detect_detailed(_marker_image("DICT_4X4_100", 17))
    assert detection.tag == 17
    assert np.isnan(detection.confidence)


def test_detect_detailed_old_api(monkeypatch):
    quad = np.array([[[0, 0], [8, 0], [8, 8], [0, 8]]], dtype=np.float32)
    monkeypatch.delattr(cv2.aruco, "ArucoDetector")
    monkeypatch.setattr(
        cv2.aruco, "detectMarkers", lambda *a, **k: ([quad], np.array([[7]]), []), raising=False
    )
    [detection] = _model().detect_detailed(np.zeros((20, 20), dtype=np.uint8))
    assert detection.tag == 7
    assert detection.center == (4.0, 4.0)
    assert np.isnan(detection.confidence)


# ---- corner refinement, detector overrides, subset capacity


@pytest.mark.parametrize("method", sorted(CORNER_REFINEMENTS))
def test_every_corner_refinement_builds_and_detects(method):
    model = _model(corner_refinement=method)
    assert model.detect(_marker_image("DICT_4X4_100", 17)) == [17]
    assert model.aruco_params.cornerRefinementMethod == CORNER_REFINEMENTS[method]


def test_default_corner_refinement_is_subpix():
    model = _model()
    model.build()
    assert model.aruco_params.cornerRefinementMethod == cv2.aruco.CORNER_REFINE_SUBPIX


def test_unknown_corner_refinement_is_rejected():
    with pytest.raises(ValueError, match="corner refinement 'bogus'"):
        _model(corner_refinement="bogus").build()


def test_detector_overrides_are_applied_and_validated():
    model = _model(
        detector_overrides={"maxErroneousBitsInBorderRate": 0.5, "minDistanceToBorder": 0}
    )
    model.build()
    assert model.aruco_params.maxErroneousBitsInBorderRate == pytest.approx(0.5)
    assert model.aruco_params.minDistanceToBorder == 0
    with pytest.raises(ValueError, match="no field 'notAField'"):
        _model(detector_overrides={"notAField": 1}).build()


def test_subset_correction_bits_sets_the_subset_dictionary_capacity():
    default = _model(tag_subset_list=[3, 17])
    default.build()
    assert (
        default.aruco_dict.maxCorrectionBits
        == cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100).maxCorrectionBits
    )
    wide = _model(tag_subset_list=[3, 17], subset_correction_bits=5)
    wide.build()
    assert wide.aruco_dict.maxCorrectionBits == 5


def test_closed_set_correction_bits():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    capacity = int(dictionary.maxCorrectionBits)
    everything = closed_set_correction_bits("DICT_5X5_100", range(100))
    assert everything == capacity
    four = closed_set_correction_bits("DICT_5X5_100", [56, 57, 58, 59])
    two = closed_set_correction_bits("DICT_5X5_100", [56, 57])
    one = closed_set_correction_bits("DICT_5X5_100", [56])
    assert one >= two >= four >= capacity
    assert type(four) is int
    # numpy ints and iterables are accepted
    assert closed_set_correction_bits("DICT_5X5_100", np.array([56, 57, 58, 59])) == four


def test_closed_set_correction_bits_validates_input():
    with pytest.raises(ValueError, match="Unable to assign"):
        closed_set_correction_bits("DICT_9X9_9", [1])
    with pytest.raises(ValueError, match="not in DICT_4X4_100"):
        closed_set_correction_bits("DICT_4X4_100", [1, 500])
    with pytest.raises(ValueError, match="not in DICT_4X4_100"):
        closed_set_correction_bits("DICT_4X4_100", [-1])
