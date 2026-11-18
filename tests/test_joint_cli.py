"""``naps-track --identity-engine joint``: options, outputs, quadrant modes, legacy parity."""

import argparse
import importlib
import json

import cv2
import numpy as np
import pytest
import sleap_io as sio

from naps.matching import OpenCVDecodeSource
from tests.conftest import DATA_DIR

# `from naps import naps_track` fails when nothing imported it yet: naps.__name__ is "naps-track"
naps_track = importlib.import_module("naps.naps_track")

REQUIRED = [
    "--slp-path", "a.slp", "--video-path", "a.avi", "--start-frame", "0", "--end-frame", "9",
    "--aruco-marker-set", "DICT_4X4_100",
]  # fmt: skip


def track_names(path):
    """{frame: [track name or None per instance]} of an output file."""
    return {
        lf.frame_idx: [None if i.track is None else i.track.name for i in lf.instances]
        for lf in sio.load_slp(str(path))
    }


def assert_integer_names(path):
    for row in track_names(path).values():
        for name in row:
            assert name is None or (
                name.startswith("ArUcoTag#") and name.split("#")[1].isdigit()
            ), name


@pytest.fixture
def make_joint(tmp_path, colony, slp_factory, video_factory):
    """Factory ``(tracked=False) -> (args, out, colony)`` for a flat joint run on the colony."""

    def make(tracked=False, out_name="out.slp"):
        slp = slp_factory(tmp_path / "bees.slp", colony, tracked=tracked)
        video = video_factory(tmp_path / "bees.avi", colony)
        out = tmp_path / out_name
        args = [
            "--slp-path", str(slp),
            "--video-path", str(video),
            "--tag-node-name", "Tag",
            "--start-frame", "0",
            "--end-frame", str(colony.n_frames - 1),
            "--aruco-marker-set", "DICT_4X4_100",
            "--aruco-crop-size", "60",
            "--output-path", str(out),
            "--identity-engine", "joint",
            "--closed-set", *map(str, colony.tags),
            "--colony-size", "3",
        ]  # fmt: skip
        return args, out, colony

    return make


@pytest.fixture
def joint_scene(make_joint):
    return make_joint()


# ------------------------------------------------------------------------------ options
def test_joint_options_and_their_defaults():
    args = naps_track.build_parser().parse_args(REQUIRED)
    assert args.identity_engine == "legacy"
    assert args.tracklets == "built"
    assert args.min_confidence == 0.99
    assert args.consistency is True
    assert args.interpolate_max_gap == 20
    assert args.colony_size == 4
    assert args.decoder == "legacy"
    assert args.untrusted_policy == "withhold"
    assert args.self_validate is False
    assert args.window_frames == 6000
    assert args.identity_sidecar is None
    assert args.session is None
    given = naps_track.build_parser().parse_args(
        [
            *REQUIRED,
            *["--identity-engine", "joint", "--tracklets", "sleap", "--min-confidence", "0.95"],
            *["--no-consistency", "--interpolate-max-gap", "5", "--colony-size", "5"],
            *["--decoder", "fast", "--untrusted-policy", "emit", "--self-validate"],
            *["--window-frames", "100", "--identity-sidecar", "s.npz", "--session", "S1"],
        ]
    )
    assert given.identity_engine == "joint"
    assert given.tracklets == "sleap"
    assert given.min_confidence == 0.95
    assert given.consistency is False
    assert given.interpolate_max_gap == 5
    assert given.colony_size == 5
    assert given.decoder == "fast"
    assert given.untrusted_policy == "emit"
    assert given.self_validate is True
    assert given.window_frames == 100
    assert given.identity_sidecar == "s.npz"
    assert given.session == "S1"


def test_help_lists_the_joint_engine_group(capsys):
    with pytest.raises(SystemExit) as exc:
        naps_track.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "joint identity engine" in out
    for flag in (
        "--identity-engine", "--tracklets", "--min-confidence", "--consistency",
        "--no-consistency", "--interpolate-max-gap", "--colony-size", "--self-validate",
        "--identity-sidecar", "--session", "--untrusted-policy", "--window-frames", "--decoder",
    ):  # fmt: skip
        assert flag in out


def test_build_joint_config_maps_the_options():
    args = naps_track.build_parser().parse_args(
        [
            *REQUIRED,
            *["--identity-engine", "joint", "--tracklets", "sleap", "--min-confidence", "0.95"],
            *["--no-consistency", "--interpolate-max-gap", "7", "--colony-size", "5"],
        ]
    )
    config = naps_track.build_joint_config(args)
    assert config.identity.threshold == 0.95
    assert config.consistency.enabled is False
    assert config.consistency.gap_max == 7
    assert config.tracklet_mode == "sleap"
    assert config.colony_size == 5
    default = naps_track.build_joint_config(naps_track.build_parser().parse_args(REQUIRED))
    assert default.identity.threshold == 0.99
    assert default.consistency.enabled is True
    assert default.consistency.gap_max == 20
    assert default.tracklet_mode == "built"


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ([], "--identity-engine joint needs the closed set of tags"),
        (["--closed-set", "5", "6", "7", "--colony-size", "2"], "--colony-size must be between 3"),
        (["--closed-set", "5", "6", "7", "--colony-size", "7"], "--colony-size must be between 3"),
        (["--closed-set", "5", "6", "7", "--min-confidence", "0.5"], "--min-confidence"),
        (["--closed-set", "5", "6", "7", "--min-confidence", "1.5"], "--min-confidence"),
        (["--closed-set", "5", "6", "7", "--interpolate-max-gap", "-1"], "--interpolate-max-gap"),
        (["--closed-set", "5", "6", "7", "--window-frames", "0"], "--window-frames"),
        (["--closed-set", "5"], "--closed-set needs between 2 and --colony-size"),
        (["--closed-set", "5", "6", "7", "8", "9"], "--closed-set needs between 2 and"),
        (["--closed-set-infer", "--colony-size", "2"], "--colony-size must be between 3"),
    ],
)
def test_invalid_joint_options_exit_with_a_message(joint_scene, extra, message):
    args, out, _ = joint_scene
    base = args[: args.index("--closed-set")]  # drop the closed set and the colony size
    with pytest.raises(SystemExit) as exc:
        naps_track.main([*base, *extra])
    assert isinstance(exc.value.code, str)
    assert message in exc.value.code
    assert not out.exists()


def test_a_two_tag_closed_set_is_a_pair():
    parser = naps_track.build_parser()
    pair = parser.parse_args(
        [*REQUIRED, "--identity-engine", "joint", "--closed-set", "0", "1", "--colony-size", "2"]
    )
    naps_track._validate_joint_args(pair, quadrant_mode=False)
    config = naps_track.build_joint_config(pair)
    assert (config.colony_size, config.min_tags) == (2, 2)
    triple = parser.parse_args(
        [*REQUIRED, "--identity-engine", "joint", "--closed-set", "5", "6", "7"]
    )
    assert naps_track.build_joint_config(triple).min_tags == 3


def test_quadrant_scope_wiring_depends_on_the_engine():
    parser = naps_track.build_parser()
    legacy = parser.parse_args([*REQUIRED, "--closed-set-infer"])
    scope = naps_track.build_quadrant_scope(legacy)
    assert scope is not None
    assert (scope.expected_size, scope.min_size) == (4, None)
    joint = parser.parse_args(
        [*REQUIRED, "--closed-set-infer", "--identity-engine", "joint", "--colony-size", "5"]
    )
    scope = naps_track.build_quadrant_scope(joint)
    assert scope is not None
    assert (scope.expected_size, scope.min_size) == (5, 3)
    bare = argparse.Namespace(**vars(legacy))
    del bare.identity_engine
    bare_scope = naps_track.build_quadrant_scope(bare)
    assert bare_scope is not None
    assert bare_scope.min_size is None


def test_run_dispatches_on_the_engine(monkeypatch):
    seen = []
    monkeypatch.setattr(naps_track, "run_joint", seen.append)
    naps_track.run(argparse.Namespace(identity_engine="joint"))
    assert len(seen) == 1


def test_sidecar_path_prefers_the_explicit_option(tmp_path):
    args = argparse.Namespace(identity_sidecar="custom.npz", output_path=str(tmp_path / "out.slp"))
    assert str(naps_track._sidecar_path(args)) == "custom.npz"


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("naps_output.analysis.h5", "naps_output.identity.npz"),
        ("x.slp", "x.identity.npz"),
        ("x.h5", "x.identity.npz"),
        ("plain", "plain.identity.npz"),
    ],
)
def test_default_sidecar_names(tmp_path, output, expected):
    args = argparse.Namespace(identity_sidecar=None, output_path=str(tmp_path / output))
    assert naps_track._sidecar_path(args) == tmp_path / expected


def test_build_decode_source_selects_the_decoder(make_joint):
    args, _, _ = make_joint()
    parsed = naps_track.build_parser().parse_args(args)
    assert isinstance(naps_track.build_decode_source(parsed), OpenCVDecodeSource)
    fast = naps_track.build_parser().parse_args([*args, "--decoder", "fast"])
    assert type(naps_track.build_decode_source(fast)).__name__ == "TagDecoderSource"


# ----------------------------------------------------------------------------- flat runs
def test_joint_engine_labels_untracked_instances_from_frame_zero(joint_scene, tmp_path):
    args, out, colony = joint_scene
    report = tmp_path / "report.json"
    assert naps_track.main([*args, "--report-json", str(report)]) == 0
    labels = sio.load_slp(str(out))
    names = {t.name for t in labels.tracks}
    assert names == {f"ArUcoTag#{t}" for t in colony.tags}
    first = labels.labeled_frames[0]
    assert first.frame_idx == 0
    assert sorted(i.track.name for i in first.instances) == sorted(
        f"ArUcoTag#{t}" for t in colony.tags
    )
    assert all(i.identity_score is not None for i in first.instances)
    rep = json.loads(report.read_text())
    assert rep["engine"] == "joint"
    assert rep["regions"]["all"]["status"] == "ok"
    side = np.load(out.with_name("out.identity.npz"))
    assert side["tag"].shape[0] == colony.n_frames
    assert_integer_names(out)


def test_tracked_input_gives_the_same_labels(make_joint):
    args, out, colony = make_joint(tracked=False)
    naps_track.main(args)
    untracked = track_names(out)
    args2, out2, _ = make_joint(tracked=True, out_name="tracked.slp")
    naps_track.main(args2)
    assert track_names(out2) == untracked
    expected = [f"ArUcoTag#{t}" for t in colony.tags]
    assert all(row == expected for row in untracked.values())


def test_sleap_tracklets_follow_the_sleap_ids(make_joint):
    args, out, colony = make_joint(tracked=True)
    naps_track.main([*args, "--tracklets", "sleap"])
    expected = [f"ArUcoTag#{t}" for t in colony.tags]
    assert all(row == expected for row in track_names(out).values())


@pytest.mark.parametrize(
    "extra",
    [
        ["--no-consistency"],
        ["--interpolate-max-gap", "0"],
        ["--untrusted-policy", "emit"],
        ["--min-confidence", "0.999"],
        ["--window-frames", "7"],
    ],
)
def test_joint_options_run(make_joint, extra):
    args, out, colony = make_joint()
    assert naps_track.main([*args, *extra]) == 0
    expected = [f"ArUcoTag#{t}" for t in colony.tags]
    assert all(row == expected for row in track_names(out).values())


def test_session_qualifies_identities_and_the_sidecar(make_joint, tmp_path):
    args, out, colony = make_joint()
    custom = tmp_path / "side.npz"
    report = tmp_path / "report.json"
    naps_track.main(
        [*args, "--session", "S1", "--identity-sidecar", str(custom), "--report-json", str(report)]
    )
    labels = sio.load_slp(str(out))
    inst = labels.labeled_frames[3].instances[0]
    assert inst.track.name == f"ArUcoTag#{colony.tags[0]}"
    assert inst.identity.name == f"S1:ArUcoTag#{colony.tags[0]}"
    side = np.load(custom)
    assert str(side["session"]) == "S1"
    assert side["tags_qualified__all"].tolist() == [f"S1:ArUcoTag#{t}" for t in colony.tags]
    assert not out.with_name("out.identity.npz").exists()
    rep = json.loads(report.read_text())
    assert rep["session"] == "S1"
    assert rep["sidecar"] == str(custom)


def test_the_default_session_is_the_video_directory(make_joint):
    args, out, _ = make_joint()
    naps_track.main(args)
    side = np.load(out.with_name("out.identity.npz"))
    assert str(side["session"]) == out.parent.name


def test_report_content_of_a_flat_run(make_joint, tmp_path):
    args, _, colony = make_joint()
    report = tmp_path / "report.json"
    naps_track.main([*args, "--report-json", str(report)])
    rep = json.loads(report.read_text())
    assert set(rep) >= {
        "engine", "video", "slp", "session", "sidecar", "runtime_s", "options", "regions",
        "region_status_counts", "n_instances", "n_instances_labelled", "decodes", "frame_size",
    }  # fmt: skip
    assert rep["n_instances"] == rep["n_instances_labelled"] == colony.n_frames * colony.n_bees
    assert rep["frame_size"] == [colony.frame_size, colony.frame_size]
    assert rep["options"]["identity_engine"] == "joint"
    region = rep["regions"]["all"]
    assert region["tags"] == sorted(colony.tags)
    assert region["sources"]["decode"] == colony.n_frames * colony.n_bees
    assert region["completeness_after"]["bee_frame_completeness"] == 1.0
    assert region["windows"][0]["status"] == "unvalidated"


def test_self_validation_end_to_end(make_joint, tmp_path):
    args, out, colony = make_joint()
    report = tmp_path / "report.json"
    naps_track.main([*args, "--self-validate", "--report-json", str(report)])
    region = json.loads(report.read_text())["regions"]["all"]
    assert region["self_validation"]["leave1"]["n_heldout_bee_frames"] > 0
    assert region["windows"][0]["status"] in {"trusted", "uncertain", "no_repair_needed"}
    expected = [f"ArUcoTag#{t}" for t in colony.tags]
    assert all(row == expected for row in track_names(out).values())


def test_the_fast_decoder_gives_the_same_labels(make_joint):
    args, out, colony = make_joint()
    naps_track.main([*args, "--decoder", "fast"])
    expected = [f"ArUcoTag#{t}" for t in colony.tags]
    assert all(row == expected for row in track_names(out).values())


# -------------------------------------------------------------------------- quadrants
SIZE = 800
N_FRAMES = 40
QUADRANTS = ["top_left", "top_right", "bottom_left", "bottom_right"]
ORIGIN = {
    "top_left": (0, 0),
    "top_right": (400, 0),
    "bottom_left": (0, 400),
    "bottom_right": (400, 400),
}
# (tag, marker visible) per bee; a bee without a visible marker is never decoded
SPEC = {
    "top_left": [(10, True), (11, True), (12, True), (13, True)],
    "top_right": [(14, True), (15, True), (16, True), (None, False)],  # 3 tags + 1 untagged bee
    "bottom_left": [],  # an empty quadrant
    "bottom_right": [(22, True), (23, True), (24, True), (25, False)],  # tag 25 never decoded
}
MAP = "top_left:10,11,12,13;top_right:14,15,16;bottom_left:18,19,20,21;bottom_right:22,23,24,25"
QUAD_ARGS = [
    "--tag-node-name", "Tag", "--start-frame", "0", "--end-frame", str(N_FRAMES - 1),
    "--aruco-marker-set", "DICT_4X4_100", "--aruco-crop-size", "50",
    "--identity-engine", "joint",
]  # fmt: skip


def bee_xy(quadrant, k):
    ox, oy = ORIGIN[quadrant]
    return np.array([ox + 100.0 + 200 * (k % 2), oy + 100.0 + 200 * (k // 2)])


def quadrant_frame():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    img = np.full((SIZE, SIZE, 3), 120, dtype=np.uint8)
    for q in QUADRANTS:
        for k, (tag, visible) in enumerate(SPEC[q]):
            if not visible or tag is None:
                continue
            marker = cv2.aruco.generateImageMarker(dictionary, tag, 40)
            marker = cv2.copyMakeBorder(marker, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
            x, y = bee_xy(q, k).astype(int)
            h = marker.shape[0] // 2
            img[y - h : y - h + marker.shape[0], x - h : x - h + marker.shape[1]] = marker[
                :, :, None
            ]
    return img


@pytest.fixture
def quadrant_run(tmp_path, monkeypatch):
    frame = quadrant_frame()
    skeleton = sio.Skeleton(["Tag", "Thorax"])
    video = sio.Video(filename="synthetic.mp4", open_backend=False)
    lfs = []
    for f in range(N_FRAMES):
        insts = []
        for q in QUADRANTS:
            for k in range(len(SPEC[q])):
                xy = bee_xy(q, k)
                pts = np.array([xy, xy + np.array([0.0, 12.0])])
                insts.append(sio.PredictedInstance.from_numpy(pts, skeleton=skeleton, score=1.0))
        lfs.append(sio.LabeledFrame(video=video, frame_idx=f, instances=insts))
    slp = tmp_path / "quads.slp"
    sio.Labels(lfs).save(str(slp), verbose=False)
    video_path = tmp_path / "quads.mp4"
    video_path.write_bytes(b"")

    class FakeCapture:
        def __init__(self, path):
            self.f = 0

        def set(self, prop, value):
            self.f = int(value)

        def read(self):
            if self.f >= N_FRAMES:
                return False, None
            self.f += 1
            return True, frame

        def release(self):
            pass

    monkeypatch.setattr("naps.matching.cv2.VideoCapture", FakeCapture)
    out = tmp_path / "quads_out.slp"
    base = ["--slp-path", str(slp), "--video-path", str(video_path), "--output-path", str(out)]
    return [*base, *QUAD_ARGS], out, tmp_path / "report.json"


def instance_tags(out):
    """{quadrant: [tag per bee, per frame]} from the output file (instances in SPEC order)."""
    rows = {q: [] for q in QUADRANTS}
    for lf in sio.load_slp(str(out)):
        i = 0
        for q in QUADRANTS:
            row = []
            for _ in SPEC[q]:
                inst = lf.instances[i]
                row.append(None if inst.track is None else int(inst.track.name.split("#")[1]))
                i += 1
            rows[q].append(row)
    return rows


def test_quadrant_map_with_three_tags_an_untagged_bee_and_an_empty_quadrant(quadrant_run):
    args, out, report = quadrant_run
    assert naps_track.main([*args, "--closed-set-map", MAP, "--report-json", str(report)]) == 0
    tags = instance_tags(out)
    assert tags["top_left"] == [[10, 11, 12, 13]] * N_FRAMES
    assert tags["top_right"] == [[14, 15, 16, None]] * N_FRAMES  # the untagged bee stays untracked
    assert tags["bottom_left"] == [[]] * N_FRAMES
    # tag 25 is never decoded: found by elimination, from the very first frame
    assert tags["bottom_right"] == [[22, 23, 24, 25]] * N_FRAMES
    assert_integer_names(out)
    rep = json.loads(report.read_text())
    regions = rep["regions"]
    assert {q: r["status"] for q, r in regions.items()} == {
        "top_left": "ok",
        "top_right": "ok",
        "bottom_left": "no_data",
        "bottom_right": "ok",
    }
    assert regions["top_right"]["phantom_tags"] == 1
    assert regions["top_right"]["n_bees"] == 4
    sources = regions["bottom_right"]["sources"]
    assert sources["decode"] == 3 * N_FRAMES
    assert sources["tracklet"] + sources["elimination"] == N_FRAMES
    assert rep["region_status_counts"] == {"ok": 3, "no_data": 1}
    side = np.load(out.with_name("quads_out.identity.npz"))
    assert side["region_names"].tolist() == QUADRANTS


def test_quadrant_inference_accepts_three_tags_and_leaves_an_unseen_bee_untagged(quadrant_run):
    args, out, report = quadrant_run
    naps_track.main([*args, "--closed-set-infer", "--report-json", str(report)])
    tags = instance_tags(out)
    assert tags["top_left"] == [[10, 11, 12, 13]] * N_FRAMES
    assert tags["top_right"] == [[14, 15, 16, None]] * N_FRAMES
    # tag 25 is never decoded, so the inferred set has three tags and the fourth bee is untagged
    assert tags["bottom_right"] == [[22, 23, 24, None]] * N_FRAMES
    rep = json.loads(report.read_text())
    assert rep["regions"]["bottom_right"]["tags"] == [22, 23, 24]
    assert rep["regions"]["bottom_left"]["status"] == "no_data"


# ------------------------------------------------------------------ legacy is unchanged
class SceneRun:
    """Base arguments of legacy runs on the synthetic colony."""

    def __init__(self, tmp_path, colony, slp_factory, video_factory):
        self.slp = slp_factory(tmp_path / "bees.slp", colony)
        self.video = video_factory(tmp_path / "bees.avi", colony)
        self.tmp = tmp_path
        self.base = [
            "--slp-path", str(self.slp), "--video-path", str(self.video),
            "--tag-node-name", "Tag", "--start-frame", "0",
            "--end-frame", str(colony.n_frames - 1),
            "--aruco-marker-set", "DICT_4X4_100", "--aruco-crop-size", "60",
        ]  # fmt: skip

    def run(self, name, *extra):
        out = self.tmp / f"{name}.slp"
        report = self.tmp / f"{name}.json"
        argv = [*self.base, "--output-path", str(out), "--report-json", str(report), *extra]
        naps_track.main(argv)
        return track_names(out), json.loads(report.read_text())


def forbid_joint(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("the legacy path must not touch the joint engine")

    monkeypatch.setattr(naps_track, "JointMatching", boom)
    monkeypatch.setattr(naps_track, "run_joint", boom)


def test_legacy_default_equals_explicit_legacy_on_the_synthetic_scene(
    tmp_path, colony, slp_factory, video_factory, monkeypatch
):
    forbid_joint(monkeypatch)
    scene = SceneRun(tmp_path, colony, slp_factory, video_factory)
    plain_names, plain_report = scene.run("plain")
    explicit_names, explicit_report = scene.run("explicit", "--identity-engine", "legacy")
    assert plain_names == explicit_names
    assert plain_report == explicit_report
    # the pre-existing behaviour: the first half-window is unassigned, then every bee is tagged
    assert plain_names[0] == [None] * colony.n_bees
    assert plain_names[10] == [f"ArUcoTag#{t}" for t in colony.tags]
    closed = ["--closed-set", *map(str, colony.tags)]
    a_names, a_report = scene.run("closed_a", *closed)
    b_names, b_report = scene.run("closed_b", *closed, "--identity-engine", "legacy")
    assert a_names == b_names
    assert a_report == b_report


def test_legacy_default_equals_explicit_legacy_on_the_real_example(tmp_path, monkeypatch):
    forbid_joint(monkeypatch)
    base = [
        "--slp-path", str(DATA_DIR / "example.slp"), "--video-path", str(DATA_DIR / "example.mp4"),
        "--tag-node-name", "Tag", "--start-frame", "0", "--end-frame", "29",
        "--aruco-marker-set", "DICT_4X4_100", "--aruco-crop-size", "50",
    ]  # fmt: skip
    results = []
    for name, extra in (("plain", []), ("explicit", ["--identity-engine", "legacy"])):
        out, report = tmp_path / f"{name}.slp", tmp_path / f"{name}.json"
        naps_track.main([*base, "--output-path", str(out), "--report-json", str(report), *extra])
        results.append((track_names(out), json.loads(report.read_text())))
    assert results[0] == results[1]
    names = results[0][0]
    assert sorted(names) == list(range(30))
    assert any(n is not None for row in names.values() for n in row)
