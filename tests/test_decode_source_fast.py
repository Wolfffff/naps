"""The decode-stage DecodeSource gives the same decodes as the legacy OpenCV one."""

from naps.aruco import ArUcoModel
from naps.decode import DecodeParams, TagDecoder
from naps.matching import OpenCVDecodeSource, TagDecoderSource

VIDEO_PATH = "tests/data/example_ArUco_video.avi"
N_FRAMES = 6
COORDS = {
    0: (225, 225),
    1: (925, 225),
    2: (575, 575),
    3: (225, 925),
    4: (925, 925),
}
PARAMS = {
    "adaptiveThreshWinSizeMin": 3,
    "adaptiveThreshWinSizeMax": 10,
    "adaptiveThreshWinSizeStep": 3,
    "adaptiveThreshConstant": 10,
    "perspectiveRemoveIgnoredMarginPerCell": 0.1,
    "errorCorrectionRate": 0.1,
}


def test_tag_decoder_source_matches_the_legacy_source():
    coords = dict.fromkeys(range(N_FRAMES), COORDS)
    model = ArUcoModel.withTagSet("DICT_4X4_100", **PARAMS)
    legacy, legacy_size = OpenCVDecodeSource(VIDEO_PATH, model.detect, 200).decode(
        0, N_FRAMES - 1, coords
    )
    decoder = TagDecoder(
        DecodeParams(
            tag_set="DICT_4X4_100",
            win_min=3,
            win_max=10,
            win_step=3,
            thresh_const=10,
            margin=0.1,
            error_correction_rate=0.1,
        )
    )
    source = TagDecoderSource(VIDEO_PATH, decoder, crop_half=200)
    fast, fast_size = source.decode(0, N_FRAMES - 1, coords)
    assert fast_size == legacy_size
    assert source.last_result is not None
    assert set(fast) == set(legacy)
    for frame in legacy:
        for track in COORDS:
            assert fast[frame][track] == legacy[frame][track]
    assert any(tags != [None] for per_track in fast.values() for tags in per_track.values())
