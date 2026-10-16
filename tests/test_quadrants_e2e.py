"""End-to-end CLI test on a synthetic 4-quadrant file (16 bees, real ArUco markers)."""

import importlib
import json

import cv2
import numpy as np
import pytest
import sleap_io as sio

# `from naps import naps_track` fails when nothing imported it yet: naps.__name__ is "naps-track"
naps_track = importlib.import_module("naps.naps_track")

SIZE = 800  # 4 tiles of 400 px
N_FRAMES = 40
Q = ["top_left", "top_right", "bottom_left", "bottom_right"]
ORIGIN = {
    "top_left": (0, 0),
    "top_right": (400, 0),
    "bottom_left": (0, 400),
    "bottom_right": (400, 400),
}
TAGS = {q: [10 + 4 * i + k for k in range(4)] for i, q in enumerate(Q)}
ARGS = [
    "--tag-node-name", "Tag", "--start-frame", "0", "--end-frame", str(N_FRAMES - 1),
    "--aruco-marker-set", "DICT_4X4_100", "--aruco-crop-size", "50",
]  # fmt: skip
MAP = ";".join(f"{q}:{','.join(map(str, TAGS[q]))}" for q in Q)


def bee_xy(q, k):
    ox, oy = ORIGIN[q]
    return np.array([ox + 100.0 + 200 * (k % 2), oy + 100.0 + 200 * (k // 2)])


def marker_image(tag):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    img = cv2.aruco.generateImageMarker(dictionary, tag, 40)
    return cv2.copyMakeBorder(img, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)


class Scenario:
    """Who is where, and who is visible, in every frame.

    Bee (q, k) is track ``10 * qi + k``; a *restart* gives the bee a brand-new SLEAP track
    (``100 + id``) from some frame on whose tag node is NaN and whose marker is never visible
    (how a lost bee comes back in real data).
    """

    def __init__(self):
        self.restart = {
            ("top_left", 2): 15,  # one bee lost in one quadrant
            ("top_right", 1): 20,  # a second quadrant at the same time
            ("bottom_right", 0): 15,  # two bees lost in one quadrant -> must abstain
            ("bottom_right", 1): 15,
        }
        # tag of top_left decoded on a bottom_left bee, for two frames only
        self.planted = {(("bottom_left", 3), f): TAGS["top_left"][0] for f in (8, 9)}

    @staticmethod
    def base(q, k):
        return 10 * Q.index(q) + k

    def bees(self, f):
        """[(track id, q, k, tag, visible)] present in frame f."""
        out = []
        for q in Q:
            for k in range(4):
                t0 = self.restart.get((q, k))
                if t0 is not None and f >= t0:
                    out.append((100 + self.base(q, k), q, k, TAGS[q][k], False))
                else:
                    out.append((self.base(q, k), q, k, TAGS[q][k], True))
        return out

    def frame_image(self, f):
        img = np.full((SIZE, SIZE, 3), 120, dtype=np.uint8)
        for _, q, k, tag, visible in self.bees(f):
            if not visible:
                continue
            x, y = bee_xy(q, k).astype(int)
            m = marker_image(self.planted.get(((q, k), f), tag))
            h = m.shape[0] // 2
            img[y - h : y - h + m.shape[0], x - h : x - h + m.shape[1]] = m[:, :, None]
        return img


def write_slp(path, scenario):
    skel = sio.Skeleton(["Tag", "Thorax"])
    video = sio.Video(filename="synthetic.mp4", open_backend=False)
    tracks = {}
    lfs = []
    for f in range(N_FRAMES):
        insts = []
        for tid, q, k, _, visible in scenario.bees(f):
            track = tracks.setdefault(tid, sio.Track(f"track_{tid}"))
            xy = bee_xy(q, k)
            tag_xy = xy if visible else np.array([np.nan, np.nan])  # lost bee: no tag node
            pts = np.array([tag_xy, xy + np.array([0.0, 12.0])])
            insts.append(
                sio.PredictedInstance.from_numpy(pts, skeleton=skel, score=1.0, track=track)
            )
        lfs.append(sio.LabeledFrame(video=video, frame_idx=f, instances=insts))
    sio.Labels(lfs).save(str(path), verbose=False)


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    scenario = Scenario()
    slp = tmp_path / "synthetic.slp"
    write_slp(slp, scenario)
    video = tmp_path / "synthetic.mp4"
    video.write_bytes(b"")

    class FakeCapture:
        def __init__(self, path):
            self.f = 0

        def set(self, prop, value):
            self.f = int(value)

        def get(self, prop):
            return float(SIZE)

        def read(self):
            if self.f >= N_FRAMES:
                return False, None
            img = scenario.frame_image(self.f)
            self.f += 1
            return True, img

        def release(self):
            pass

    monkeypatch.setattr("naps.matching.cv2.VideoCapture", FakeCapture)
    return scenario, ["--slp-path", str(slp), "--video-path", str(video), *ARGS]


def tags_by_frame(path):
    """{frame: {track id: tag or None}}, also checking every track name is ArUcoTag#<int>."""
    out = {}
    for lf in sio.load_slp(str(path)):
        row = []
        for inst in lf.instances:
            if inst.track is None:
                tag = None
            else:
                assert inst.track.name.startswith("ArUcoTag#")
                suffix = inst.track.name.split("#")[1]
                assert suffix.isdigit(), inst.track.name  # no 21.0, None, nan
                tag = int(suffix)
            row.append(tag)
        out[lf.frame_idx] = row
    return out


def tags_of_bee(path, scenario, q, k, frames):
    """Tag per frame of one bee (instances are in scenario.bees order)."""
    order = [(qq, kk) for qq in Q for kk in range(4)].index((q, k))
    labels = sio.load_slp(str(path))
    return {
        lf.frame_idx: (
            None
            if lf.instances[order].track is None
            else int(lf.instances[order].track.name.split("#")[1])
        )
        for lf in labels
        if lf.frame_idx in frames
    }


def test_quadrant_closed_set_recovers_lost_bees_end_to_end(synthetic, tmp_path, caplog):
    scenario, base = synthetic
    legacy, mapped = tmp_path / "legacy.slp", tmp_path / "mapped.slp"
    report_path = tmp_path / "report.json"
    naps_track.main([*base, "--output-path", str(legacy)])
    caplog.set_level("INFO")
    naps_track.main(
        [
            *base,
            *["--closed-set-map", MAP, "--report-json", str(report_path)],
            *["--output-path", str(mapped)],
        ]
    )

    # the first half-window (5 frames) is never assigned by the cost matrix
    interior = range(5, N_FRAMES)
    lost = range(15, N_FRAMES)

    # legacy: the restarted tracks never decode, so they stay untagged
    assert all(t is None for t in tags_of_bee(legacy, scenario, "top_left", 2, lost).values())

    # quadrant-aware closed set: one lost bee in one quadrant, a second quadrant at another time
    assert set(tags_of_bee(mapped, scenario, "top_left", 2, lost).values()) == {TAGS["top_left"][2]}
    assert set(tags_of_bee(mapped, scenario, "top_right", 1, range(20, N_FRAMES)).values()) == {
        TAGS["top_right"][1]
    }
    # before its restart the top_right bee is simply decoded
    assert set(tags_of_bee(mapped, scenario, "top_right", 1, range(5, 20)).values()) == {
        TAGS["top_right"][1]
    }
    # two lost bees in bottom_right: nothing invented
    for k in (0, 1):
        assert set(tags_of_bee(mapped, scenario, "bottom_right", k, lost).values()) == {None}
    # untouched bees are identical to legacy, and every assigned tag is the bee's true tag
    mapped_all, legacy_all = tags_by_frame(mapped), tags_by_frame(legacy)
    truth = {f: [b[3] for b in scenario.bees(f)] for f in range(N_FRAMES)}
    for f in interior:
        for got, want in zip(mapped_all[f], truth[f], strict=True):
            assert got in (None, want)
    # no tag twice in a frame, none outside the sets
    all_tags = {t for tags in TAGS.values() for t in tags}
    for row in mapped_all.values():
        assigned = [t for t in row if t is not None]
        assert len(assigned) == len(set(assigned)) and set(assigned) <= all_tags
    # the tag planted in the wrong tile is ignored: the bee keeps its own tag
    assert set(tags_of_bee(mapped, scenario, "bottom_left", 3, range(5, 35)).values()) == {
        TAGS["bottom_left"][3]
    }
    n_legacy = sum(t is not None for r in legacy_all.values() for t in r)
    n_mapped = sum(t is not None for r in mapped_all.values() for t in r)
    assert n_mapped - n_legacy == (N_FRAMES - 15) + (N_FRAMES - 20)

    # the report
    rep = json.loads(report_path.read_text())
    assert rep["mode"] == "quadrants" and rep["frame_size"] == [SIZE, SIZE]
    assert rep["tag_sets_source"] == "explicit"
    tl = rep["regions"]["top_left"]
    assert tl["frames_by_n_tags_before_elimination"] == {"4": 10, "3": 25, "2": 0, "1": 0, "0": 5}
    assert tl["frames_by_n_tags_after_elimination"] == {"4": 35, "3": 0, "2": 0, "1": 0, "0": 5}
    br = rep["regions"]["bottom_right"]
    assert br["frames_by_n_tags_after_elimination"]["2"] == 25  # abstained
    assert rep["elimination"]["n_assigned"] == (N_FRAMES - 15) + (N_FRAMES - 20)
    assert rep["elimination"]["abstention_reasons"]["not_exactly_one"] >= 25
    # two lost bees x 25 frames, plus the 4 bees of the 5 head frames the window never assigns
    assert (
        rep["n_none_instances_by_region_and_cause"]["bottom_right"][
            "elimination_abstained:not_exactly_one"
        ]
        == 2 * 25 + 4 * 5
    )
    assert rep["cost_matrix_unassigned_head_frames"] == 5
    assert rep["wrong_quadrant_votes"] == [
        {"tag": TAGS["top_left"][0], "seen_in": "bottom_left", "belongs_to": "top_left", "votes": 2}
    ]
    assert any("Instances without a tag" in r.getMessage() for r in caplog.records)


def test_inferred_tag_sets_end_to_end(synthetic, tmp_path):
    scenario, base = synthetic
    out, report_path = tmp_path / "inferred.slp", tmp_path / "report.json"
    naps_track.main(
        [
            *base,
            *["--closed-set-infer", "--closed-set-min-support", "10"],
            *["--report-json", str(report_path), "--output-path", str(out)],
        ]
    )
    rep = json.loads(report_path.read_text())
    assert rep["tag_sets_source"] == "inferred"
    # the two quadrants with a clear 4-tag set are inferred and used; the two bees that never
    # decode after frame 15 still leave bottom_right with 4 supported tags (15 frames each)
    inferred = {q: sorted(v) for q, v in rep["inference"]["inferred"].items()}
    assert inferred == {q: sorted(TAGS[q]) for q in Q}
    assert set(tags_of_bee(out, scenario, "top_left", 2, range(15, N_FRAMES)).values()) == {
        TAGS["top_left"][2]
    }
    assert set(tags_of_bee(out, scenario, "bottom_right", 0, range(15, N_FRAMES)).values()) == {
        None
    }
    # the planted wrong-tile decode did not flip any set
    assert TAGS["top_left"][0] in inferred["top_left"]
    tags_by_frame(out)  # names are ArUcoTag#<int>


def test_inference_abstains_when_support_is_too_low(synthetic, tmp_path):
    scenario, base = synthetic
    out, report_path = tmp_path / "abstain.slp", tmp_path / "report.json"
    # default min support (30 frames) cannot be met by the bees that only decode for 15 frames
    naps_track.main(
        [*base, "--closed-set-infer", "--report-json", str(report_path), "--output-path", str(out)]
    )
    rep = json.loads(report_path.read_text())
    assert "bottom_right" in rep["inference"]["abstained"]
    assert "top_left" in rep["inference"]["abstained"]  # bee (top_left, 2) decodes 15 frames
    assert rep["elimination"] is None or rep["elimination"]["n_assigned"] == 0
    assert set(tags_of_bee(out, scenario, "top_left", 2, range(15, N_FRAMES)).values()) == {None}
    assert rep["n_none_instances_by_cause"]["quadrant_without_tag_set"] > 0


def test_flat_closed_set_still_abstains_on_the_full_frame(synthetic, tmp_path):
    scenario, base = synthetic
    out = tmp_path / "flat.slp"
    naps_track.main([*base, "--closed-set", *map(str, TAGS["top_left"]), "--output-path", str(out)])
    assert set(tags_of_bee(out, scenario, "top_left", 2, range(15, N_FRAMES)).values()) == {None}


def test_option_conflicts_exit(synthetic, tmp_path):
    _, base = synthetic
    out = ["--output-path", str(tmp_path / "x.slp")]
    with pytest.raises(SystemExit):
        naps_track.main([*base, "--closed-set", "1", "2", "--closed-set-map", MAP, *out])
    with pytest.raises(SystemExit):
        naps_track.main([*base, "--closed-set-map", MAP, "--closed-set-infer", *out])
    with pytest.raises(SystemExit):
        naps_track.main([*base, "--closed-set-map", "a:1,2;b:2,3", *out])
    with pytest.raises(SystemExit):  # region names must exist in the grid
        naps_track.main([*base, "--closed-set-map", "centre:1,2,3,4", *out])
