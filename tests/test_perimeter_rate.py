"""``minMarkerPerimeterRate``: the detector speed knob of the decode stage.

OpenCV discards candidate contours shorter than ``rate * max(width, height)`` pixels. The default
0.03 (3 px on a 100 px crop) lets tiny noise quads through; 0.15 gives the same decodes much
faster (measured on 53,640 crops of 12 colonies/cameras, summarised in
``docs/decode_measurements.md``). These tests pin the behaviour that justifies it (a 45 px tag
decodes identically) and its limit (a tag whose perimeter is below the threshold is rejected), and
that the rate is relative to the image, so scaled crops need no rescaling.
"""

import importlib
from typing import Any, cast

import cv2
import numpy as np
import pytest

from naps.decode import (
    CLOSED_SET_PASSES,
    FAST_MIN_MARKER_PERIMETER_RATE,
    DecodeParams,
    TagDecoder,
)
from naps.matching import OpenCVDecodeSource, TagDecoderSource

naps_track = importlib.import_module("naps.naps_track")

OPENCV_DEFAULT_RATE = cv2.aruco.DetectorParameters().minMarkerPerimeterRate
REQUIRED = [
    "--slp-path", "a.slp", "--video-path", "a.avi", "--start-frame", "0", "--end-frame", "9",
    "--aruco-marker-set", "DICT_5X5_100",
]  # fmt: skip


def _crop(tag=7, side=45, size=100, angle=0.0, tag_set="DICT_5X5_100"):
    """A ``size`` px crop with one ``side`` px marker (border included) on a bright background."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, tag_set))
    marker = cv2.aruco.generateImageMarker(dictionary, tag, side)
    crop = np.full((size, size), 200, dtype=np.uint8)
    lo = (size - side) // 2
    crop[lo : lo + side, lo : lo + side] = marker
    if angle:
        matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
        crop = cv2.warpAffine(crop, matrix, (size, size), borderValue=200)
    return crop


def _signature(detections):
    """Tags, corners and confidences of a decode, for exact comparison."""
    return [(d.tag, d.corners.tolist(), d.confidence) for d in detections]


def _built_rate(decoder):
    [model] = decoder._models.values()
    assert model.aruco_params is not None
    return model.aruco_params.minMarkerPerimeterRate


# ---------------------------------------------------------------- the parameter itself
def test_default_keeps_opencvs_rate():
    assert DecodeParams().min_marker_perimeter_rate is None
    decoder = TagDecoder()
    decoder.decode(_crop())
    assert _built_rate(decoder) == pytest.approx(OPENCV_DEFAULT_RATE)
    assert pytest.approx(0.03) == OPENCV_DEFAULT_RATE  # the premise of the speed-up


def test_rate_reaches_the_detector_and_an_explicit_override_wins():
    decoder = TagDecoder(DecodeParams(min_marker_perimeter_rate=0.2))
    decoder.decode(_crop())
    assert _built_rate(decoder) == pytest.approx(0.2)
    both = TagDecoder(
        DecodeParams(
            min_marker_perimeter_rate=0.2,
            overrides=(("minMarkerPerimeterRate", 0.4), ("minDistanceToBorder", 1)),
        )
    )
    both.decode(_crop())
    assert _built_rate(both) == pytest.approx(0.4)
    [model] = both._models.values()
    assert model.aruco_params is not None
    assert model.aruco_params.minDistanceToBorder == 1


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan")])
def test_rate_must_be_positive(bad):
    with pytest.raises(ValueError, match="min_marker_perimeter_rate must be positive"):
        TagDecoder(DecodeParams(min_marker_perimeter_rate=bad))
    with pytest.raises(ValueError, match="min_marker_perimeter_rate must be positive"):
        TagDecoder.for_closed_set([1, 2, 3, 4], min_marker_perimeter_rate=bad)


# ---------------------------------------------------------------- same decodes, a tag of ~45 px
@pytest.mark.parametrize("tag", [0, 7, 56, 99])
@pytest.mark.parametrize("angle", [0.0, 31.0, 90.0])
@pytest.mark.parametrize("rate", [0.10, FAST_MIN_MARKER_PERIMETER_RATE, 0.30])
def test_a_45_px_tag_decodes_identically_with_the_default_and_a_raised_rate(tag, angle, rate):
    crop = _crop(tag=tag, side=45, angle=angle)
    default = TagDecoder(DecodeParams()).decode(crop)
    fast = TagDecoder(DecodeParams(min_marker_perimeter_rate=rate)).decode(crop)
    assert [d.tag for d in default] == [tag]
    assert _signature(fast) == _signature(default)


def test_closed_set_cascade_decodes_identically_with_the_default_and_the_fast_rate():
    crops = [
        _crop(tag=t, side=s, angle=a) for t, s, a in [(56, 45, 0.0), (58, 38, 17.0), (59, 30, 45.0)]
    ]
    slow = TagDecoder.for_closed_set([56, 57, 58, 59], min_marker_perimeter_rate=None)
    fast = TagDecoder.for_closed_set([56, 57, 58, 59])
    for crop in crops:
        assert _signature(fast.decode(crop)) == _signature(slow.decode(crop))
        assert fast.tags(crop) != [None]
    assert _built_rate(slow) == pytest.approx(OPENCV_DEFAULT_RATE)
    assert fast.decode(crops[0]) and all(
        m.aruco_params is not None
        and m.aruco_params.minMarkerPerimeterRate == pytest.approx(FAST_MIN_MARKER_PERIMETER_RATE)
        for m in fast._models.values()
    )


def test_no_decode_on_noise_changes_with_the_fast_rate():
    noise = np.random.default_rng(0).integers(0, 255, (100, 100), dtype=np.uint8)
    assert TagDecoder().decode(noise) == []
    assert TagDecoder(DecodeParams(min_marker_perimeter_rate=0.15)).decode(noise) == []


# ------------------------------------------------------- the limit: a tag below the threshold
def test_a_tag_below_the_perimeter_threshold_is_rejected():
    crop = _crop(tag=7, side=28)  # perimeter 112 px in a 100 px crop
    assert [d.tag for d in TagDecoder().decode(crop)] == [7]
    assert [
        d.tag for d in TagDecoder(DecodeParams(min_marker_perimeter_rate=0.15)).decode(crop)
    ] == [7]
    assert [
        d.tag for d in TagDecoder(DecodeParams(min_marker_perimeter_rate=1.0)).decode(crop)
    ] == [7]
    # 1.2 * 100 px = 120 px threshold > 112 px perimeter: nothing is detected any more
    assert TagDecoder(DecodeParams(min_marker_perimeter_rate=1.2)).decode(crop) == []
    assert TagDecoder(DecodeParams(min_marker_perimeter_rate=1.2)).tags(crop) == [None]


def test_the_threshold_scales_with_the_crop_size():
    """The same rate rejects the same tag in a small crop and keeps it in a crop half the size."""
    small, large = _crop(side=28, size=100), _crop(side=28, size=200)
    rate = 0.9  # 90 px on the 100 px crop (< 112 px perimeter), 180 px on the 200 px crop
    decoder = TagDecoder(DecodeParams(min_marker_perimeter_rate=rate))
    assert [d.tag for d in decoder.decode(small)] == [7]
    assert decoder.decode(large) == []


# ---------------------------------------------------------------- scaled crops need no rescaling
@pytest.mark.parametrize("scale", [1.5, 2.0])
def test_the_rate_is_relative_so_scaled_crops_need_no_rescaling(scale):
    crop = _crop(tag=56, side=45)
    default = TagDecoder(DecodeParams(scale=scale)).decode(crop)
    fast = TagDecoder(DecodeParams(scale=scale, min_marker_perimeter_rate=0.15)).decode(crop)
    assert [d.tag for d in default] == [56]
    assert _signature(fast) == _signature(default)
    # the limit is the same in original pixels at every scale: 1.0 keeps, 1.2 drops a 28 px tag
    small = _crop(side=28)
    assert [
        d.tag
        for d in TagDecoder(DecodeParams(scale=scale, min_marker_perimeter_rate=1.0)).decode(small)
    ] == [7]
    assert TagDecoder(DecodeParams(scale=scale, min_marker_perimeter_rate=1.2)).decode(small) == []


# ---------------------------------------------------------------- the closed-set preset
def test_the_closed_set_preset_uses_the_fast_rate_and_it_can_be_switched_off():
    assert all(
        p.min_marker_perimeter_rate == FAST_MIN_MARKER_PERIMETER_RATE for p in CLOSED_SET_PASSES
    )
    assert FAST_MIN_MARKER_PERIMETER_RATE == 0.15
    decoder = TagDecoder.for_closed_set([1, 2, 3, 4])
    assert [p.min_marker_perimeter_rate for p in decoder.passes] == [0.15] * 3
    off = TagDecoder.for_closed_set([1, 2, 3, 4], min_marker_perimeter_rate=None)
    assert [p.min_marker_perimeter_rate for p in off.passes] == [None] * 3
    custom = TagDecoder.for_closed_set([1, 2, 3, 4], min_marker_perimeter_rate=0.3, rescue_passes=2)
    assert [p.min_marker_perimeter_rate for p in custom.passes] == [0.3] * 2
    assert [p.min_marker_perimeter_rate for p in decoder.with_closed_set([1, 2]).passes] == [
        0.15
    ] * 3


def test_the_legacy_defaults_are_untouched():
    assert DecodeParams() == DecodeParams(min_marker_perimeter_rate=None)
    assert TagDecoder().passes == (DecodeParams(),)


# ---------------------------------------------------------------- command line
def test_cli_option_defaults_and_validation():
    parser = naps_track.build_parser()
    assert parser.parse_args(REQUIRED).aruco_min_marker_perimeter_rate is None
    given = parser.parse_args([*REQUIRED, "--aruco-min-marker-perimeter-rate", "0.2"])
    assert given.aruco_min_marker_perimeter_rate == pytest.approx(0.2)
    for bad in ("0", "-1", "abc", "inf", "nan"):
        with pytest.raises(SystemExit):
            parser.parse_args([*REQUIRED, "--aruco-min-marker-perimeter-rate", bad])


def test_cli_legacy_decoder_is_unchanged_unless_the_option_is_given():
    parser = naps_track.build_parser()
    default = naps_track.build_aruco_model(parser.parse_args(REQUIRED))
    assert default.detector_overrides == {}
    default.build()
    assert default.aruco_params is not None
    assert default.aruco_params.minMarkerPerimeterRate == pytest.approx(OPENCV_DEFAULT_RATE)
    legacy = parser.parse_args([*REQUIRED, "--aruco-min-marker-perimeter-rate", "0.15"])
    model = naps_track.build_aruco_model(legacy)
    assert model.detector_overrides == {"minMarkerPerimeterRate": 0.15}
    model.build()
    assert model.aruco_params is not None
    assert model.aruco_params.minMarkerPerimeterRate == pytest.approx(0.15)


def test_cli_fast_decoder_defaults_to_the_fast_rate_and_accepts_an_override(tmp_path):
    video = tmp_path / "a.avi"
    video.write_bytes(b"")
    base = [*REQUIRED]
    base[base.index("--video-path") + 1] = str(video)
    parser = naps_track.build_parser()

    def rate_of(extra):
        args = parser.parse_args([*base, "--decoder", "fast", *extra])
        source = cast(TagDecoderSource, naps_track.build_decode_source(args))
        [params] = source.decoder.passes
        return params.min_marker_perimeter_rate

    assert rate_of([]) == FAST_MIN_MARKER_PERIMETER_RATE
    assert rate_of(["--aruco-min-marker-perimeter-rate", "0.3"]) == pytest.approx(0.3)
    assert rate_of(["--aruco-min-marker-perimeter-rate", "0.03"]) == pytest.approx(0.03)
    legacy = cast(OpenCVDecodeSource, naps_track.build_decode_source(parser.parse_args(base)))
    assert type(legacy).__name__ == "OpenCVDecodeSource"
    assert cast(Any, legacy.marker_detector).__self__.detector_overrides == {}
    explicit_args = parser.parse_args([*base, "--aruco-min-marker-perimeter-rate", "0.15"])
    explicit = cast(OpenCVDecodeSource, naps_track.build_decode_source(explicit_args))
    model = cast(Any, explicit.marker_detector).__self__
    assert model.detector_overrides == {"minMarkerPerimeterRate": 0.15}
