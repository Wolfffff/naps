"""Brightness tags (``naps.color_tags`` / ``naps-color-decodes``)."""

import cv2
import numpy as np
import pytest
import sleap_io as sio

from naps.color_tags import classify, disc_statistic, main, measure, otsu_threshold
from naps.precomputed import PrecomputedDecodeSource

N_FRAMES = 6
BRIGHT, DARK = 200, 30


def _scene(tmp_path):
    """Two animals on a grey background: slot 0 carries a bright dot, slot 1 a dark one.

    Frame 3 has no instances; in frame 4 slot 1 has no tag node.
    """
    skeleton = sio.Skeleton(["thorax", "tag"])
    video_path = tmp_path / "pair.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"FFV1"), 20, (120, 80))
    frames = []
    for f in range(N_FRAMES):
        img = np.full((80, 120, 3), 120, np.uint8)
        a, b = (20 + 2 * f, 40), (90 - 2 * f, 40)
        cv2.circle(img, a, 6, (BRIGHT,) * 3, -1)
        cv2.circle(img, b, 6, (DARK,) * 3, -1)
        writer.write(img)
        if f == 3:
            continue
        instances = []
        for j, (x, y) in enumerate((a, b)):
            tag = (np.nan, np.nan) if (f == 4 and j == 1) else (x, y)
            pts = np.array([[x, y + 5], tag], float)
            instances.append(sio.PredictedInstance.from_numpy(pts, skeleton=skeleton, score=1.0))
        frames.append((f, instances))
    writer.release()
    video = sio.Video.from_filename(str(video_path))
    labels = sio.Labels(
        [sio.LabeledFrame(video=video, frame_idx=f, instances=i) for f, i in frames],
        videos=[video], skeletons=[skeleton],
    )  # fmt: skip
    slp = tmp_path / "pair.slp"
    labels.save(str(slp))
    return slp, video_path


def test_disc_statistic_and_threshold():
    gray = np.full((20, 20), 10, np.uint8)
    gray[5:10, 5:10] = 250
    assert disc_statistic(gray, 7, 7, 2, 50) == 250
    assert disc_statistic(gray, 15, 15, 2, 50) == 10
    assert np.isnan(disc_statistic(gray, -50, -50, 2, 50))
    thr = otsu_threshold(np.array([10.0, 12, 11, 240, 245, np.nan]))
    assert 12 < thr <= 240
    calls = classify(np.array([0.0, thr - 1, thr + 50, np.nan]), thr, 5)
    assert calls.tolist() == [-1, 0, 1, 0]


def test_measure_reads_every_frame(tmp_path):
    slp, video = _scene(tmp_path)
    r = measure(slp, video, radius=3, percentile=50)
    assert r.n_inst.tolist() == [2, 2, 2, 0, 2, 2]
    assert r.frame_size == (120, 80)
    assert np.allclose(r.value[[0, 1, 2, 4, 5], 0], BRIGHT, atol=3)
    assert np.allclose(r.value[[0, 1, 2, 5], 1], DARK, atol=3)
    assert np.isnan(r.value[4, 1]) and np.isnan(r.value[3]).all()


@pytest.mark.parametrize(("dark_votes", "expected"), [(False, 5), (True, 9)])
def test_cli_writes_a_valid_decodes_file(tmp_path, dark_votes, expected):
    slp, video = _scene(tmp_path)
    out = tmp_path / "pair.decodes.npz"
    argv = [
        "--slp-path", str(slp), "--video-path", str(video), "--output-path", str(out),
        "--radius", "3", "--percentile", "50", "--bright-tag", "7", "--dark-tag", "3",
    ]  # fmt: skip
    main(argv + (["--dark-votes"] if dark_votes else []))
    with np.load(out) as d:
        assert len(d["vote_tag"]) == expected
        assert set(d["vote_tag"][d["vote_inst"] == 0].tolist()) == {7}
    source = PrecomputedDecodeSource(out, video=video)
    xy = {f: {0: (20 + 2 * f, 40), 1: (90 - 2 * f, 40)} for f in (0, 1, 2)}
    decodes, size = source.decode(0, 2, xy)
    assert size == (120, 80)
    assert decodes[0][0] == [7]
    assert decodes[0][1] == ([3] if dark_votes else [None])
