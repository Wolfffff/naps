import pickle
from collections import defaultdict

import cv2
import numpy as np
import pytest

from naps.aruco import ArUcoModel, Detection
from naps.decode import (
    CLOSED_SET_PASSES,
    DecodeParams,
    DecodeResult,
    SkipPolicy,
    TagDecoder,
    crop_around,
    decode_frame,
    decode_video,
    detect_dictionary,
    guess_dictionary,
)
from naps.matching import MatchFrame, Matching
from naps.video import FastVideoReader, VideoReadError
from tests.video_helpers import (
    LIMITED_LUT,
    FakeSpec,
    install_fake_video,
    textured_frames,
    write_gray_video,
)


@pytest.fixture
def fake_video(monkeypatch, tmp_path):
    """Factory ``(spec) -> path`` of a video played by a scripted fake ``VideoCapture``."""
    return lambda spec: install_fake_video(monkeypatch, tmp_path, spec)


VIDEO_PATH = "tests/data/example_ArUco_video.avi"
IMAGE_PATH = "tests/data/example_ArUco_image.jpg"
N_VIDEO_FRAMES = 15
COORDS = {0: (225, 225), 1: (925, 225), 2: (575, 575), 3: (225, 925), 4: (925, 925)}
# the settings tests/test_matching.py decodes the example video with
EXAMPLE = DecodeParams(
    tag_set="DICT_4X4_100",
    win_min=3,
    win_max=10,
    win_step=3,
    thresh_const=10,
    margin=0.1,
    error_correction_rate=0.1,
)


def _marker(tag_set="DICT_5X5_100", tag=7, size=60, pad=30, value=255):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, tag_set))
    marker = cv2.aruco.generateImageMarker(dictionary, tag, size)
    return np.pad(marker, pad, constant_values=value)


def _canvas(tags_at, size=200, tag_set="DICT_5X5_100", background=200):
    """A canvas with markers: ``tags_at`` is ``[(tag, cx, cy), ...]`` (60 px markers)."""
    canvas = np.full((size, size), background, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, tag_set))
    for tag, cx, cy in tags_at:
        marker = cv2.aruco.generateImageMarker(dictionary, tag, 60)
        marker = np.pad(marker, 8, constant_values=255)
        h = marker.shape[0] // 2
        canvas[cy - h : cy - h + marker.shape[0], cx - h : cx - h + marker.shape[1]] = marker
    return canvas


# ---------------------------------------------------------------- TagDecoder


def test_defaults_are_the_legacy_production_decoder():
    params = DecodeParams()
    assert (params.tag_set, params.win_min, params.win_max, params.win_step) == (
        "DICT_5X5_100",
        3,
        23,
        10,
    )
    assert (params.thresh_const, params.error_correction_rate, params.margin) == (7.0, 0.6, 0.1)
    assert params.corner_refinement == "subpix"
    assert params.closed_set is None
    assert params.scale == 1.0
    assert params.clahe is None


@pytest.mark.parametrize("tag", [0, 7, 56, 99])
def test_default_decoder_matches_arucomodel_exactly(tag):
    image = _canvas([(tag, 100, 100)])
    legacy = ArUcoModel.withTagSet("DICT_5X5_100", 3, 23, 10, 7.0, 0.1, 0.6)
    decoder = TagDecoder()
    assert decoder.tags(image) == legacy.detect(image) == [tag]
    assert decoder(image) == [tag]
    blank = np.full((120, 120), 90, dtype=np.uint8)
    assert decoder.tags(blank) == legacy.detect(blank) == [None]


def test_decode_returns_geometry_in_input_pixels():
    [detection] = TagDecoder().decode(_canvas([(7, 90, 110)]))
    assert detection.tag == 7
    assert abs(detection.center[0] - 90) < 2
    assert abs(detection.center[1] - 110) < 2
    assert abs(detection.side - 60) < 3


def test_empty_and_missing_images_decode_to_nothing():
    decoder = TagDecoder()
    assert decoder.decode(None) == []
    assert decoder.decode(np.zeros((0, 7), dtype=np.uint8)) == []
    assert decoder.tags(None) == [None]


def test_closed_set_rejects_foreign_tags_and_can_be_set_per_call():
    image = _canvas([(7, 100, 100)])
    assert TagDecoder(DecodeParams(closed_set=(7, 8))).tags(image) == [7]
    assert TagDecoder(DecodeParams(closed_set=(8, 9))).tags(image) == [None]
    decoder = TagDecoder(DecodeParams(closed_set=(8, 9)))
    assert decoder.tags(image, closed_set=[7, 50]) == [7]
    assert decoder.with_closed_set([7]).tags(image) == [7]
    assert decoder.with_closed_set(None).tags(image) == [7]
    assert decoder.with_closed_set([]).tags(image) == [7]  # empty = no restriction on the copy
    # an empty per-call answer never widens a restricted decoder: its own set still applies
    assert decoder.tags(image, closed_set=[]) == [None]
    assert decoder.tags(image, closed_set=None) == [None]


def test_models_are_cached_per_pass_and_closed_set():
    decoder = TagDecoder()
    image = _canvas([(7, 100, 100)])
    decoder.decode(image)
    decoder.decode(image)
    decoder.decode(image, closed_set=[7])
    assert len(decoder._models) == 2


def test_position_radius_picks_the_marker_under_the_tag_node():
    image = _canvas([(7, 60, 100), (8, 140, 100)])
    everything = TagDecoder()
    assert sorted(everything.tags(image)) == [7, 8]
    near_first = TagDecoder(position_radius=20.0)
    assert near_first.tags(image, expected=(60, 100)) == [7]
    assert near_first.tags(image, expected=(140, 100)) == [8]
    assert near_first.tags(image, expected=(100, 10)) == [None]
    # nearest first without a gate
    assert [d.tag for d in everything.decode(image, expected=(140, 100))] == [8, 7]
    assert [d.tag for d in near_first.decode(image)] == []  # the crop centre is between them


def test_min_confidence_filters_but_keeps_unknown_confidence(monkeypatch):
    image = _canvas([(7, 100, 100)])
    assert TagDecoder(min_confidence=0.0).tags(image) == [7]
    assert TagDecoder(min_confidence=1.01).tags(image) == [None]
    corners = np.array([[90, 90], [110, 90], [110, 110], [90, 110]], dtype=np.float32)
    unknown = Detection(7, corners, float("nan"))
    monkeypatch.setattr(ArUcoModel, "detect_detailed", lambda self, img: [unknown])
    assert TagDecoder(min_confidence=0.9).tags(image) == [7]


def test_cascade_runs_later_passes_only_when_needed_and_union_merges():
    image = _canvas([(7, 100, 100)])
    miss = DecodeParams(closed_set=(8,))
    hit = DecodeParams(closed_set=(7,))
    assert TagDecoder([miss, hit]).tags(image) == [7]
    assert TagDecoder([hit, miss]).tags(image) == [7]
    assert TagDecoder([miss]).tags(image) == [None]
    union = TagDecoder([hit, DecodeParams()], strategy="union")
    [detection] = union.decode(image)  # same tag found by both passes: one detection
    assert detection.tag == 7
    cascade = TagDecoder([hit, DecodeParams()])
    cascade.decode(image)
    assert len(cascade._models) == 1  # the second pass never ran


def test_union_keeps_the_most_confident_detection(monkeypatch):
    corners = np.array([[90, 90], [110, 90], [110, 110], [90, 110]], dtype=np.float32)
    answers = iter(
        [
            [Detection(7, corners, 0.5)],
            [Detection(7, corners, 0.9)],
            [Detection(8, corners, float("nan"))],
        ]
    )
    monkeypatch.setattr(ArUcoModel, "detect_detailed", lambda self, img: next(answers))
    decoder = TagDecoder(
        [DecodeParams(), DecodeParams(thresh_const=5.0), DecodeParams(thresh_const=3.0)],
        strategy="union",
    )
    found = {d.tag: d.confidence for d in decoder.decode(np.zeros((200, 200), dtype=np.uint8))}
    assert found[7] == 0.9
    assert np.isnan(found[8])


def test_upsampling_and_clahe_still_decode_and_report_input_pixels():
    image = _canvas([(7, 90, 110)])
    for params in (
        DecodeParams(scale=2.0),
        DecodeParams(scale=1.5, interpolation="linear"),
        DecodeParams(clahe=(2.0, 4)),
        DecodeParams(scale=0.75),
    ):
        decoder = TagDecoder(params)
        [detection] = decoder.decode(image)
        assert detection.tag == 7
        assert abs(detection.center[0] - 90) < 3
        assert abs(detection.center[1] - 110) < 3
        assert abs(detection.side - 60) < 4
        assert decoder.decode(image)[0].tag == 7  # the cached CLAHE object is reusable


def test_threshold_windows_scale_with_the_resampling():
    decoder = TagDecoder(DecodeParams(scale=2.0))
    decoder.decode(_canvas([(7, 100, 100)]))
    [model] = decoder._models.values()
    assert (
        model.adaptiveThreshWinSizeMin,
        model.adaptiveThreshWinSizeMax,
        model.adaptiveThreshWinSizeStep,
    ) == (6, 46, 20)


def test_correction_bits_and_overrides_reach_the_detector():
    decoder = TagDecoder(
        DecodeParams(
            closed_set=(56, 57, 58, 59),
            correction_bits="closed_set",
            error_correction_rate=1.0,
            corner_refinement="none",
            overrides=(("maxErroneousBitsInBorderRate", 0.5),),
        )
    )
    decoder.decode(_canvas([(56, 100, 100)]))
    [model] = decoder._models.values()
    assert model.aruco_params is not None
    assert _capacity(model) >= 3
    assert model.aruco_params.maxErroneousBitsInBorderRate == pytest.approx(0.5)
    fixed = TagDecoder(DecodeParams(closed_set=(56, 57), correction_bits=5))
    fixed.decode(_canvas([(56, 100, 100)]))
    assert _capacity(next(iter(fixed._models.values()))) == 5
    # "closed_set" without a closed set falls back to the dictionary's own capacity
    open_set = TagDecoder(DecodeParams(correction_bits="closed_set"))
    open_set.decode(_canvas([(56, 100, 100)]))
    assert _capacity(next(iter(open_set._models.values()))) == 3


def _capacity(model):
    """Correctable bits of the (subset) dictionary a built ArUcoModel decodes with."""
    assert model.aruco_dict is not None
    return int(model.aruco_dict.maxCorrectionBits)


def test_legacy_lut_is_applied_before_detection():
    image = _canvas([(7, 100, 100)])
    inverted = (255 - np.arange(256)).astype(np.uint8)
    assert TagDecoder(legacy_lut=inverted).tags(image) == [None]  # inverted marker is unreadable
    identity = np.arange(256, dtype=np.uint8)
    assert TagDecoder(legacy_lut=identity).tags(image) == [7]
    assert TagDecoder().with_legacy_lut(identity).legacy_lut is identity


def test_decode_batch_handles_missing_crops_and_expected_positions():
    crops = {0: _canvas([(7, 100, 100)]), 1: None, 2: _canvas([(8, 60, 100), (9, 140, 100)])}
    decoder = TagDecoder(position_radius=15.0)
    out = decoder.decode_batch(crops, expected={2: (140, 100)})
    assert [d.tag for d in out[0]] == [7]
    assert out[1] == []
    assert [d.tag for d in out[2]] == [9]
    plain = TagDecoder().decode_batch(crops, closed_set=[7, 8])
    assert [d.tag for d in plain[2]] == [8]


def test_decoder_pickles_without_its_detectors():
    decoder = TagDecoder(DecodeParams(clahe=(2.0, 4)), position_radius=12.0)
    decoder.decode(_canvas([(7, 100, 100)]))
    assert decoder._models
    clone = pickle.loads(pickle.dumps(decoder))  # noqa: S301 - our own object
    assert not clone._models
    assert clone.position_radius == 12.0
    assert clone.tags(_canvas([(7, 100, 100)])) == [7]


def test_decoder_rejects_bad_configuration():
    with pytest.raises(ValueError, match="at least one"):
        TagDecoder([])
    with pytest.raises(ValueError, match="scale"):
        TagDecoder(DecodeParams(scale=0.0))


# ---------------------------------------------------------------- cropping


def test_crop_around_matches_the_legacy_matchframe_crop():
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 255, (120, 150, 3), dtype=np.uint8)
    for _ in range(40):
        xy = (float(rng.uniform(-10, 160)), float(rng.uniform(-10, 130)))
        match_frame = MatchFrame.fromCV2(True, frame)
        match_frame.cropMarkerWithCoordsArray({0: xy}, 17)
        legacy = match_frame.frame_images[0]
        new = crop_around(frame[:, :, 0], xy, 17)
        assert new is not None
        assert np.array_equal(new, legacy)


def test_crop_around_nan_gives_none():
    frame = np.zeros((50, 50), dtype=np.uint8)
    assert crop_around(frame, (np.nan, 3.0), 10) is None
    assert crop_around(frame, (3.0, np.nan), 10) is None


# ---------------------------------------------------------------- frame and video


def _example_positions(n=N_VIDEO_FRAMES):
    return {f: dict(COORDS) for f in range(n)}


def test_decode_frame_gives_full_frame_geometry():
    image = cv2.imread(IMAGE_PATH, 0)
    coords = {**COORDS, 9: (float("nan"), 5.0)}
    out = decode_frame(image, coords, TagDecoder(EXAMPLE), 200)
    assert [d.tag for d in out[0]] == [1]
    assert out[9] == []
    for track, (x, y) in COORDS.items():
        [detection] = out[track]
        assert detection.tag == track + 1
        assert abs(detection.center[0] - x) < 5
        assert abs(detection.center[1] - y) < 5
    restricted = decode_frame(image, COORDS, TagDecoder(EXAMPLE), 200, closed_set=[1, 2])
    assert restricted[3] == restricted[4] == []


def test_decode_video_reproduces_matching_decode_video():
    detector = ArUcoModel.withTagSet(
        "DICT_4X4_100",
        adaptiveThreshWinSizeMin=3,
        adaptiveThreshWinSizeMax=10,
        adaptiveThreshWinSizeStep=3,
        adaptiveThreshConstant=10,
        perspectiveRemoveIgnoredMarginPerCell=0.1,
        errorCorrectionRate=0.1,
    ).detect
    matcher = Matching(
        VIDEO_PATH,
        0,
        N_VIDEO_FRAMES - 1,
        marker_detector=detector,
        aruco_crop_size=200,
        half_rolling_window_size=7,
        tag_node_dict=_example_positions(),
    )
    expected = matcher._decode_video()
    result = decode_video(
        VIDEO_PATH,
        _example_positions(),
        TagDecoder(EXAMPLE),
        first_frame=0,
        last_frame=N_VIDEO_FRAMES - 1,
        crop_half=200,
    )
    assert isinstance(result, DecodeResult)
    assert result.votes == expected
    assert result.frame_size == matcher.frame_size == (1150, 1150)
    assert result.n_decoded == 5 * N_VIDEO_FRAMES
    assert result.n_skipped == 0
    assert result.truncated_at is None
    assert all(isinstance(votes, defaultdict) for votes in result.votes.values())
    assert [d.tag for d in result.detections[3][2]] == [3]


def test_decode_video_with_a_start_frame_and_an_open_reader():
    positions = {f: dict(COORDS) for f in range(4, 9)}
    with FastVideoReader(VIDEO_PATH) as reader:
        result = decode_video(
            reader, positions, TagDecoder(EXAMPLE), first_frame=4, last_frame=8, crop_half=200
        )
        assert reader.read() is not None  # the caller's reader stays open
    assert sorted(result.votes) == [4, 5, 6, 7, 8]
    assert result.votes[6][1] == [2]


def test_decode_video_closed_set_and_nan_tracks():
    positions = {f: {**COORDS, 7: (float("nan"), float("nan"))} for f in range(3)}
    result = decode_video(
        VIDEO_PATH,
        positions,
        TagDecoder(EXAMPLE),
        first_frame=0,
        last_frame=2,
        crop_half=200,
        closed_set=[1, 2],
    )
    assert result.votes[0][0] == [1]
    assert result.votes[0][3] == [None]
    assert result.votes[0][7] == [None]
    assert result.detections[0][7] == []


def test_decode_video_past_the_end():
    positions = {f: dict(COORDS) for f in range(12, 19)}
    with pytest.raises(VideoReadError, match=r"frame 15: .*could not be read"):
        decode_video(
            VIDEO_PATH, positions, TagDecoder(EXAMPLE), first_frame=12, last_frame=18, crop_half=200
        )
    empty = {f: {0: (float("nan"), float("nan"))} for f in range(12, 19)}
    result = decode_video(
        VIDEO_PATH, empty, TagDecoder(EXAMPLE), first_frame=12, last_frame=18, crop_half=200
    )
    assert result.truncated_at == 15
    assert result.votes[17] == {0: [None]}
    assert result.frame_size == (1150, 1150)


def test_decode_video_missing_positions_raise():
    with pytest.raises(KeyError):
        decode_video(VIDEO_PATH, {}, TagDecoder(EXAMPLE), first_frame=0, last_frame=0)


# ---------------------------------------------------------------- legacy pixels and the luma reader


def _marker_frames(n=6, tag=7, half=40):
    """Gray frames with one 5x5 marker at the centre, on a textured background."""
    base = textured_frames(n, height=160, width=160)
    marker = _marker(tag=tag, size=64, pad=8)
    frames = []
    for i in range(n):
        frame = base[i].copy() // 4 + 100
        frame[48 : 48 + marker.shape[0], 48 : 48 + marker.shape[1]] = marker
        frames.append(frame)
    return frames


def test_luma_reader_with_legacy_pixels_decodes_like_read(tmp_path):
    frames = _marker_frames()
    path = write_gray_video(tmp_path / "markers.avi", frames)
    positions = {f: {0: (88.0, 88.0)} for f in range(len(frames))}
    legacy = decode_video(
        path, positions, TagDecoder(), first_frame=0, last_frame=5, crop_half=50, reader_mode="bgr"
    )
    fast = decode_video(path, positions, TagDecoder(), first_frame=0, last_frame=5, crop_half=50)
    assert fast.votes == legacy.votes
    assert [v[0] for v in fast.votes.values()] == [[7]] * 6


def test_limited_range_luma_is_remapped_for_legacy_pixels(fake_video):
    frames = [_canvas([(7, 50, 50)], size=100) for _ in range(4)]
    path = fake_video(FakeSpec(frames))
    positions = {f: {0: (50.0, 50.0)} for f in range(4)}
    kwargs = {"first_frame": 0, "last_frame": 3, "crop_half": 45}
    legacy = decode_video(path, positions, TagDecoder(), reader_mode="bgr", **kwargs)
    with FastVideoReader(path) as reader:
        assert reader.pixel_range == "limited"
        assert reader.luma_plane
        fast = decode_video(reader, positions, TagDecoder(), **kwargs)
        raw = decode_video(reader, positions, TagDecoder(), legacy_pixels=False, **kwargs)
        decoder = TagDecoder(legacy_lut=reader.legacy_lut)
        crop = crop_around(frames[0], (50.0, 50.0), 45)
        assert crop is not None
        remapped = decoder._preprocess(crop, DecodeParams())
        assert np.array_equal(remapped, LIMITED_LUT[crop])
    assert fast.votes == legacy.votes
    assert [v[0] for v in fast.votes.values()] == [[7]] * 4
    assert len(raw.votes) == 4


def test_path_with_unmeasurable_luma_remap_is_read_the_old_way(fake_video):
    frames = [_canvas([(7, 50, 50)], size=100) for _ in range(3)]
    path = fake_video(FakeSpec(frames, read_ok=False))  # luma accepted on trust, remap unknown
    positions = {f: {0: (50.0, 50.0)} for f in range(3)}
    with FastVideoReader(path) as probe:
        assert probe.luma_plane
        assert probe.legacy_lut is None
    result = decode_video(path, positions, TagDecoder(), first_frame=0, last_frame=2, crop_half=45)
    assert [v[0] for v in result.votes.values()] == [[7]] * 3


def test_passed_in_luma_reader_without_a_remap_cannot_reproduce_legacy_pixels(fake_video):
    path = fake_video(FakeSpec(textured_frames(3), colour=True))
    positions = {f: {0: (30.0, 30.0)} for f in range(3)}
    with FastVideoReader(path, mode="y") as reader:
        with pytest.raises(VideoReadError, match="no luma to read"):
            decode_video(reader, positions, TagDecoder(), first_frame=0, last_frame=2)
        result = decode_video(
            reader, positions, TagDecoder(), first_frame=0, last_frame=2, legacy_pixels=False
        )
    assert len(result.votes) == 3


def test_colour_video_path_is_read_the_old_way(fake_video):
    path = fake_video(FakeSpec(textured_frames(3), colour=True, raw="plane"))
    positions = {f: {0: (30.0, 30.0)} for f in range(3)}
    result = decode_video(path, positions, TagDecoder(), first_frame=0, last_frame=2)
    assert len(result.votes) == 3
    assert result.frame_size == (80, 64)


# ---------------------------------------------------------------- skip policy


def test_skip_policy_state_machine():
    policy = SkipPolicy(confirmations=2, refresh=3, max_step=2.0)
    xy = (10.0, 10.0)
    assert policy.skip_tag("a", 0, xy) is None  # nothing known
    policy.observe("a", 0, xy, 5)
    assert policy.skip_tag("a", 1, xy) is None  # one confirmation only
    policy.observe("a", 1, xy, 5)
    assert policy.skip_tag("a", 2, xy) == 5
    assert policy.skip_tag("a", 3, xy) == 5
    assert policy.skip_tag("a", 4, xy) is None  # refresh: decode again
    assert policy.skip_tag("a", 2, (30.0, 10.0)) is None  # moved too far
    assert policy.skip_tag("a", 1, xy) is None  # same frame as the last decode / earlier
    policy.observe("a", 2, xy, 6)  # a different tag resets the streak
    assert policy.skip_tag("a", 3, xy) is None
    policy.observe("a", 3, xy, None)  # a failed decode forgets the track
    policy.observe("a", 4, xy, None)  # forgetting an unknown track is fine
    assert policy.skip_tag("a", 5, xy) is None
    policy.observe("b", 0, xy, 1)
    policy.reset()
    assert policy.skip_tag("b", 1, xy) is None


def test_decode_video_with_skip_policy_saves_decodes_and_carries_votes():
    positions = _example_positions()
    kwargs = {"first_frame": 0, "last_frame": N_VIDEO_FRAMES - 1, "crop_half": 200}
    dense = decode_video(VIDEO_PATH, positions, TagDecoder(EXAMPLE), **kwargs)
    policy = SkipPolicy(confirmations=2, refresh=4, max_step=3.0)
    sparse = decode_video(VIDEO_PATH, positions, TagDecoder(EXAMPLE), skip=policy, **kwargs)
    assert sparse.n_skipped > 0
    assert sparse.n_decoded + sparse.n_skipped == 5 * N_VIDEO_FRAMES
    assert sparse.votes == dense.votes  # static tags: the carried tag is the decoded tag
    uncarried = decode_video(
        VIDEO_PATH,
        positions,
        TagDecoder(EXAMPLE),
        skip=SkipPolicy(confirmations=2, refresh=4, carry=False),
        **kwargs,
    )
    assert any(votes[0] == [None] for votes in uncarried.votes.values())
    assert any(not dets[0] for dets in uncarried.detections.values())


# ---------------------------------------------------------------- dictionary detection


def test_detect_dictionary_finds_the_dictionary_of_the_tags():
    crops = [_canvas([(56 + k % 4, 100, 100)], tag_set="DICT_5X5_100") for k in range(12)]
    ranking = detect_dictionary(
        crops, ["DICT_4X4_100", "DICT_5X5_50", "DICT_5X5_100", "DICT_6X6_100"]
    )
    best = ranking[0]
    assert best.tag_set == "DICT_5X5_100"
    assert set(best.tags) == {56, 57, 58, 59}
    assert best.n_stable == 12
    assert best.n_decodes >= best.n_stable
    # the 5x5 dictionary without ids >= 50 cannot read these tags
    assert next(g for g in ranking if g.tag_set == "DICT_5X5_50").n_stable == 0


def test_detect_dictionary_prefers_the_smaller_nested_dictionary_on_ties():
    crops = [_canvas([(3 + k % 3, 100, 100)], tag_set="DICT_5X5_100") for k in range(9)]
    seen = []
    ranking = detect_dictionary(
        crops, ["DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250"], progress=seen.append
    )
    assert ranking[0].tag_set == "DICT_5X5_50"
    assert ranking[0].n_stable == ranking[1].n_stable == ranking[2].n_stable
    assert seen == ["DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250"]


def test_detect_dictionary_requires_recurring_ids():
    crops = [_canvas([(t, 100, 100)], tag_set="DICT_5X5_100") for t in (10, 20, 30)]
    [guess] = detect_dictionary(crops, ["DICT_5X5_100"])
    assert guess.n_decodes == 3
    assert guess.n_stable == 0
    assert guess.tags == {}


def test_closed_sets_can_differ_per_track():
    positions = {f: dict(COORDS) for f in range(2)}
    per_track = {0: [1], 1: [9], 2: [3]}  # tracks 3 and 4 are not listed: unrestricted
    result = decode_video(
        VIDEO_PATH,
        positions,
        TagDecoder(EXAMPLE),
        first_frame=0,
        last_frame=1,
        crop_half=200,
        closed_set=per_track,
    )
    assert [result.votes[0][t] for t in range(5)] == [[1], [None], [3], [4], [5]]
    frame = decode_frame(
        cv2.imread(IMAGE_PATH, 0), COORDS, TagDecoder(EXAMPLE), 200, closed_set=per_track
    )
    assert [[d.tag for d in frame[t]] for t in range(5)] == [[1], [], [3], [4], [5]]


def test_one_shot_iterators_serve_every_track_and_frame():
    positions = {f: dict(COORDS) for f in range(2)}

    def run(closed_set):
        return decode_video(
            VIDEO_PATH,
            positions,
            TagDecoder(EXAMPLE),
            first_frame=0,
            last_frame=1,
            crop_half=200,
            closed_set=closed_set,
        ).votes

    expected = run((1, 2))
    # a generator would be used up by the first crop: the other tracks (and frames) would be
    # decoded against the whole dictionary
    assert run(t for t in (1, 2)) == expected
    for f in range(2):
        assert [expected[f][t] for t in range(5)] == [[1], [2], [None], [None], [None]]
    per_track = {0: [1], 1: [9], 2: [3]}
    assert run({k: iter(v) for k, v in per_track.items()}) == run(per_track)
    assert run({0: iter([1]), 1: iter([9])})[1][1] == [None]  # frame 1: still restricted
    frame = decode_frame(
        cv2.imread(IMAGE_PATH, 0), COORDS, TagDecoder(EXAMPLE), 200, closed_set=iter([1, 2])
    )
    assert [[d.tag for d in frame[t]] for t in range(5)] == [[1], [2], [], [], []]
    assert run({0: None, 1: [9]})[0][0] == [1]  # a None entry is no restriction


def test_non_finite_positions_give_no_crop_and_a_none_vote():
    frame = np.zeros((50, 50), dtype=np.uint8)
    assert crop_around(frame, (np.inf, 3.0), 10) is None
    assert crop_around(frame, (3.0, -np.inf), 10) is None
    positions = {0: {**COORDS, 7: (float("inf"), 5.0)}}
    result = decode_video(
        VIDEO_PATH, positions, TagDecoder(EXAMPLE), first_frame=0, last_frame=0, crop_half=200
    )
    assert result.votes[0][7] == [None]
    assert result.detections[0][7] == []


def test_guess_dictionary_returns_a_name_or_none():
    assert guess_dictionary([_canvas([(56, 100, 100)])], []) is None  # nothing to choose from
    crops = [_canvas([(56 + k % 4, 100, 100)]) for k in range(12)]
    assert guess_dictionary(crops, ["DICT_4X4_100", "DICT_5X5_100"]) == "DICT_5X5_100"
    assert guess_dictionary(crops[:2], ["DICT_4X4_100", "DICT_5X5_100"]) is None
    blank = [np.full((100, 100), 120, dtype=np.uint8)] * 6
    assert guess_dictionary(blank, ["DICT_5X5_100"]) is None


# ---------------------------------------------------------------- legacy callables as decoders


def test_decode_video_accepts_a_legacy_marker_detector():
    detector = ArUcoModel.withTagSet(
        "DICT_4X4_100",
        adaptiveThreshWinSizeMin=3,
        adaptiveThreshWinSizeMax=10,
        adaptiveThreshWinSizeStep=3,
        adaptiveThreshConstant=10,
        perspectiveRemoveIgnoredMarginPerCell=0.1,
        errorCorrectionRate=0.1,
    ).detect
    kwargs = {"first_frame": 0, "last_frame": 4, "crop_half": 200}
    legacy = decode_video(VIDEO_PATH, _example_positions(5), detector, **kwargs)
    modern = decode_video(VIDEO_PATH, _example_positions(5), TagDecoder(EXAMPLE), **kwargs)
    assert legacy.votes == modern.votes
    assert legacy.n_decoded == 25
    assert all(not dets for row in legacy.detections.values() for dets in row.values())
    # a callable can return anything hashable; skipping still works on its votes
    policy = SkipPolicy(confirmations=1, refresh=3)
    names = decode_video(
        VIDEO_PATH, _example_positions(5), lambda crop: ["a"], skip=policy, **kwargs
    )
    assert names.n_skipped > 0
    assert names.votes[0][0] == ["a"]


def test_legacy_callable_sees_remapped_crops_for_limited_range_luma(fake_video):
    frames = [_canvas([(7, 50, 50)], size=100) for _ in range(2)]
    path = fake_video(FakeSpec(frames))
    seen = []

    def detector(crop):
        seen.append(crop.copy())
        return [None]

    decode_video(
        path,
        {f: {0: (50.0, 50.0)} for f in range(2)},
        detector,
        first_frame=0,
        last_frame=1,
        crop_half=45,
    )
    expected = LIMITED_LUT[crop_around(frames[0], (50.0, 50.0), 45)]
    assert np.array_equal(seen[0], expected)


def test_closed_set_can_be_a_callable_of_frame_and_track():
    positions = {f: dict(COORDS) for f in range(3)}
    calls = []

    def sets(frame, track):
        calls.append((frame, track))
        return [track + 1] if frame == 0 else [9]  # frame 0: the right tag, later frames: wrong

    result = decode_video(
        VIDEO_PATH,
        positions,
        TagDecoder(EXAMPLE),
        first_frame=0,
        last_frame=2,
        crop_half=200,
        closed_set=sets,
    )
    assert [result.votes[0][t] for t in range(5)] == [[1], [2], [3], [4], [5]]
    assert [result.votes[1][t] for t in range(5)] == [[None]] * 5
    assert (2, 4) in calls
    frame = decode_frame(
        cv2.imread(IMAGE_PATH, 0), COORDS, TagDecoder(EXAMPLE), 200, closed_set=sets, frame_index=0
    )
    assert [[d.tag for d in frame[t]] for t in range(5)] == [[1], [2], [3], [4], [5]]


# ---------------------------------------------------------------- the closed-set preset


def test_for_closed_set_builds_the_measured_cascade():
    decoder = TagDecoder.for_closed_set([59, 56, 58, 57])
    assert len(decoder.passes) == len(CLOSED_SET_PASSES) == 3
    assert all(p.closed_set == (56, 57, 58, 59) for p in decoder.passes)
    assert all(p.error_correction_rate == 1.0 for p in decoder.passes)
    assert decoder.passes[2].clahe == (2.0, 4)
    assert decoder.passes[0].win_max == 23
    assert decoder.passes[1].win_max == 33
    assert TagDecoder.for_closed_set([56], rescue_passes=1).passes == (
        TagDecoder.for_closed_set([56]).passes[0],
    )
    gated = TagDecoder.for_closed_set([56], position_radius=30.0, tag_set="DICT_5X5_250")
    assert gated.position_radius == 30.0
    assert gated.passes[0].tag_set == "DICT_5X5_250"
    lut = np.arange(256, dtype=np.uint8)
    assert TagDecoder.for_closed_set([56], legacy_lut=lut).legacy_lut is lut


def test_for_closed_set_decodes_and_rejects_foreign_tags():
    decoder = TagDecoder.for_closed_set([56, 57, 58, 59])
    assert decoder.tags(_canvas([(57, 100, 100)])) == [57]
    assert decoder.tags(_canvas([(7, 100, 100)])) == [None]
    low_contrast = (_canvas([(57, 100, 100)]).astype(np.float32) * 0.12 + 110).astype(np.uint8)
    assert TagDecoder.for_closed_set([57]).tags(low_contrast) == [57]  # the CLAHE pass rescues it


def test_for_closed_set_validates_its_arguments():
    with pytest.raises(ValueError, match="at least one tag"):
        TagDecoder.for_closed_set([])
    with pytest.raises(ValueError, match="between 1 and 3"):
        TagDecoder.for_closed_set([56], rescue_passes=0)
    with pytest.raises(ValueError, match="between 1 and 3"):
        TagDecoder.for_closed_set([56], rescue_passes=4)
    with pytest.raises(ValueError, match="not a valid tag id"):
        TagDecoder.for_closed_set([56.5])
