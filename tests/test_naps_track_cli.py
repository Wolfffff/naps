"""``naps-track`` command line on a tiny synthetic scene (3 static bees with ArUco markers)."""

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import sleap_io as sio

from naps import naps_track

REPO_ROOT = Path(__file__).resolve().parents[1]
FIRST_ASSIGNED_FRAME = 5  # the rolling window (half size 5) does not assign the first frames


class Scene:
    """Synthetic slp + video and the base argument list; ``out`` is the output path."""

    def __init__(self, tmp_path, colony, slp_factory, video_factory, untracked=()):
        self.colony = colony
        self.slp = slp_factory(tmp_path / "bees.slp", colony, untracked=untracked)
        self.video = video_factory(tmp_path / "bees.avi", colony)
        self.out = tmp_path / "out.slp"
        self.args = [
            "--slp-path", str(self.slp),
            "--video-path", str(self.video),
            "--tag-node-name", "Tag",
            "--start-frame", "0",
            "--end-frame", str(colony.n_frames - 1),
            "--aruco-marker-set", "DICT_4X4_100",
            "--aruco-crop-size", "60",
            "--output-path", str(self.out),
        ]  # fmt: skip


@pytest.fixture
def make_scene(tmp_path, colony, slp_factory, video_factory):
    """Factory ``(untracked=()) -> Scene``."""
    return lambda untracked=(): Scene(tmp_path, colony, slp_factory, video_factory, untracked)


@pytest.fixture
def scene(make_scene):
    """The default scene: three tracked bees."""
    return make_scene()


def run_tags(scene, *extra):
    """Run naps-track on the scene and return {frame: [track name per instance]}."""
    assert naps_track.main([*scene.args, *extra]) == 0
    return {
        lf.frame_idx: [None if i.track is None else i.track.name for i in lf.instances]
        for lf in sio.load_slp(str(scene.out))
    }


def test_naps_track_help_lists_groups_and_flags(capsys):
    with pytest.raises(SystemExit) as exc:
        naps_track.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage: naps-track" in out
    for group in ("input and output:", "ArUco detection:", "diagnostics:", "quadrants"):
        assert group in out
    for flag in ("--closed-set-infer", "--idtracker-trajectories", "--report-json", "--verbose"):
        assert flag in out
    assert "DEPRECATED" in out


def test_naps_track_runs_as_a_script(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, "-m", "naps.naps_track", "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage: naps-track" in result.stdout


def test_missing_required_arguments_exit_with_status_2():
    with pytest.raises(SystemExit) as exc:
        naps_track.main([])
    assert exc.value.code == 2


def test_tracks_every_bee_with_its_tag(scene):
    names = run_tags(scene)
    expected = [f"ArUcoTag#{tag}" for tag in scene.colony.tags]
    for frame in range(FIRST_ASSIGNED_FRAME, scene.colony.n_frames):
        assert names[frame] == expected, frame
    all_names = {n for row in names.values() for n in row if n is not None}
    assert all_names and all(re.fullmatch(r"ArUcoTag#\d+", n) for n in all_names)
    # the output reloads with the geometry of the input
    original = sio.load_slp(str(scene.slp))
    for new_lf, old_lf in zip(sio.load_slp(str(scene.out)), original, strict=True):
        for new, old in zip(new_lf.instances, old_lf.instances, strict=True):
            assert (new.numpy() == old.numpy()).all()


def test_untracked_instances_stay_untracked(make_scene):
    scene = make_scene(untracked=[1])
    names = run_tags(scene)
    for frame, row in names.items():
        assert row[1] is None
        if frame >= FIRST_ASSIGNED_FRAME:
            assert row[0] == "ArUcoTag#5"
            assert row[2] == "ArUcoTag#7"


def test_frame_range_limits_the_output(scene):
    names = run_tags(scene, "--start-frame", "2", "--end-frame", "14")
    assert sorted(names) == list(range(2, 15))


def test_unknown_tag_node_is_a_clear_error(scene):
    with pytest.raises(ValueError, match="tag node 'Antenna' is not a node"):
        naps_track.main([*scene.args, "--tag-node-name", "Antenna"])


def test_h5_path_is_deprecated(scene, caplog):
    expected = run_tags(scene)
    bogus = scene.out.parent / "never_read.h5"  # does not exist: proves the file is not opened
    with pytest.warns(DeprecationWarning, match=r"--h5-path is deprecated and ignored"):
        again = run_tags(scene, "--h5-path", str(bogus))
    assert again == expected
    assert any(
        r.levelno == logging.WARNING and "--h5-path is deprecated" in r.getMessage()
        for r in caplog.records
    )


def test_no_deprecation_warning_without_deprecated_flags(scene, recwarn):
    run_tags(scene)
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


def test_deprecated_options_table_matches_the_parser():
    dests = {a.dest for a in naps_track.build_parser()._actions}
    assert set(naps_track.DEPRECATED_OPTIONS) <= dests


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (
            ["--closed-set", "5", "6", "--closed-set-map", "top_left:5,6"],
            "--closed-set (one colony) cannot be combined with",
        ),
        (
            ["--closed-set", "5", "6", "--closed-set-infer"],
            "--closed-set (one colony) cannot be combined with",
        ),
        (
            ["--closed-set-map", "top_left:5,6", "--closed-set-infer"],
            "use either --closed-set-map or --closed-set-infer, not both",
        ),
        (
            ["--idtracker-trajectories", "nothing.npy"],
            "--idtracker-trajectories requires --closed-set",
        ),
        (
            ["--closed-set-map", "a:1,2;b:2,3"],
            "invalid quadrant closed-set configuration",
        ),
        (
            ["--closed-set-map", "centre:5,6"],
            "invalid quadrant closed-set configuration: regions ['centre'] are not among",
        ),
        (
            ["--closed-set-infer", "--quadrant-roi", "a", "10", "10", "5", "5"],
            "invalid quadrant closed-set configuration: roi a: empty rectangle",
        ),
    ],
)
def test_invalid_option_combinations_exit_with_a_message(scene, extra, message):
    with pytest.raises(SystemExit) as exc:
        naps_track.main([*scene.args, *extra])
    assert isinstance(exc.value.code, str)  # a message (exit status 1), not a bare status
    assert message in exc.value.code
    assert not scene.out.exists()


def test_verbose_and_quiet_are_mutually_exclusive(scene, capsys):
    with pytest.raises(SystemExit) as exc:
        naps_track.main([*scene.args, "--verbose", "--quiet"])
    assert exc.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_quiet_silences_info_and_restores_the_log_level(scene, caplog):
    logger = logging.getLogger("NAPS Logger")
    before = logger.level
    with caplog.at_level(logging.INFO):
        run_tags(scene, "--quiet")
    assert not [
        r for r in caplog.records if r.name == "NAPS Logger" and r.levelno < logging.WARNING
    ]
    assert logger.level == before


def test_verbose_enables_debug_messages_and_restores_the_log_level(make_scene, caplog):
    scene = make_scene(untracked=[1])
    logger = logging.getLogger("NAPS Logger")
    before = logger.level
    run_tags(scene)  # default level: the debug message about untracked instances is not emitted
    assert not [r for r in caplog.records if "without a SLEAP track" in r.getMessage()]
    run_tags(scene, "--verbose")
    assert [r for r in caplog.records if "without a SLEAP track" in r.getMessage()]
    assert logger.level == before


def test_report_json_schema_in_legacy_mode(scene):
    report_path = scene.out.parent / "report.json"
    run_tags(scene, "--report-json", str(report_path))
    report = json.loads(report_path.read_text())
    assert set(report) >= {
        "frames", "frame_size", "mode", "regions", "n_instances",
        "n_instances_without_tag_before_elimination",
        "n_instances_without_tag_after_elimination",
        "n_none_instances_by_cause", "n_none_instances_by_region_and_cause", "elimination",
        "video", "slp", "cost_matrix_unassigned_head_frames", "options",
    }  # fmt: skip
    assert report["mode"] == "legacy"
    assert report["frames"] == [0, scene.colony.n_frames - 1]
    assert report["frame_size"] == [scene.colony.frame_size, scene.colony.frame_size]
    assert report["video"] == str(scene.video)
    assert report["slp"] == str(scene.slp)
    assert report["n_instances"] == scene.colony.n_frames * scene.colony.n_bees
    assert report["cost_matrix_unassigned_head_frames"] == 5
    assert set(report["regions"]) == {"all"}
    assert report["elimination"] is None
    # the options block is built from the parsed arguments by prefix; pin its keys
    assert set(report["options"]) == {
        "closed_set", "closed_set_map", "closed_set_infer", "closed_set_size",
        "closed_set_min_support", "closed_set_min_fraction", "closed_set_min_purity",
        "closed_set_elimination", "closed_set_max_jump", "closed_set_max_distance",
        "quadrant_grid", "quadrant_roi", "position_node_name",
    }  # fmt: skip
    assert report["options"]["quadrant_grid"] == [2, 2]


def test_flat_closed_set_eliminates_a_lost_tag_and_reports_it(scene):
    report_path = scene.out.parent / "report.json"
    names = run_tags(
        scene,
        "--closed-set", *map(str, scene.colony.tags),
        "--report-json", str(report_path),
    )  # fmt: skip
    for frame in range(FIRST_ASSIGNED_FRAME, scene.colony.n_frames):
        assert names[frame] == [f"ArUcoTag#{t}" for t in scene.colony.tags]
    report = json.loads(report_path.read_text())
    assert report["mode"] == "flat_closed_set"
    assert report["regions"]["all"]["tag_set"] == sorted(scene.colony.tags)
    assert set(report["elimination"]) >= {
        "n_assigned",
        "abstention_reasons",
        "dropped_foreign_tags",
    }


def test_quadrant_map_with_inline_regions(scene):
    # a 2x2 grid over the 240 px frame: bee 0 top_left, bee 1 top_right, bee 2 bottom_left
    report_path = scene.out.parent / "report.json"
    names = run_tags(
        scene,
        "--closed-set-map", "top_left:5;top_right:6;bottom_left:7",
        "--report-json", str(report_path),
    )  # fmt: skip
    for frame in range(FIRST_ASSIGNED_FRAME, scene.colony.n_frames):
        assert names[frame] == [f"ArUcoTag#{t}" for t in scene.colony.tags]
    report = json.loads(report_path.read_text())
    assert report["mode"] == "quadrants"
    assert report["tag_sets_source"] == "explicit"
    assert report["options"]["closed_set_map"] == "top_left:5;top_right:6;bottom_left:7"


def test_quadrant_roi_replaces_the_grid(scene):
    names = run_tags(
        scene,
        "--closed-set-map", "left:5,7;right:6",
        "--quadrant-roi", "left", "0", "0", "120", "240",
        "--quadrant-roi", "right", "120", "0", "240", "240",
    )  # fmt: skip
    for frame in range(FIRST_ASSIGNED_FRAME, scene.colony.n_frames):
        assert names[frame] == [f"ArUcoTag#{t}" for t in scene.colony.tags]


def test_closed_set_max_distance_override_is_accepted(scene):
    names = run_tags(
        scene,
        "--closed-set", *map(str, scene.colony.tags),
        "--closed-set-max-distance", "50", "--no-closed-set-elimination",
    )  # fmt: skip
    assert names[scene.colony.n_frames - 1] == [f"ArUcoTag#{t}" for t in scene.colony.tags]


# --------------------------------------------------------------------------- helpers


def test_instance_position_prefers_tag_then_fallback_then_mean():
    nan = np.nan
    points = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # tag, Thorax, Head
    names = ["Tag", "Thorax", "Head"]
    pos = naps_track._instance_position
    np.testing.assert_array_equal(pos(points, 0, names, "Thorax"), [1.0, 2.0])
    with_nan_tag = points.copy()
    with_nan_tag[0] = nan
    np.testing.assert_array_equal(pos(with_nan_tag, 0, names, "Thorax"), [3.0, 4.0])
    # fallback node missing from the skeleton or itself NaN: mean of the finite nodes
    np.testing.assert_array_equal(pos(with_nan_tag, 0, names, "Absent"), [4.0, 5.0])
    both_nan = with_nan_tag.copy()
    both_nan[1] = nan
    np.testing.assert_array_equal(pos(both_nan, 0, names, "Thorax"), [5.0, 6.0])
    assert pos(np.full((3, 2), nan), 0, names, "Thorax") is None


def test_json_default_handles_numpy_scalars_sets_and_tuples():
    default = naps_track._json_default
    assert json.dumps({"n": np.int64(3), "x": np.float32(0.5)}, default=default) == (
        '{"n": 3, "x": 0.5}'
    )
    assert json.loads(json.dumps({"s": {1}, "t": (1, 2)}, default=default)) == {
        "s": [1],
        "t": [1, 2],
    }
    with pytest.raises(TypeError, match="not JSON serialisable: object"):
        json.dumps(object(), default=default)


def test_load_sleap_tracks_without_any_finite_node(tmp_path):
    skeleton = sio.Skeleton(["Tag", "Thorax"])
    video = sio.Video(filename="v.mp4", open_backend=False)
    instance = sio.PredictedInstance.from_numpy(
        np.full((2, 2), np.nan), skeleton=skeleton, score=1.0, track=sio.Track("track_3")
    )
    path = tmp_path / "missing.slp"
    sio.Labels([sio.LabeledFrame(video=video, frame_idx=0, instances=[instance])]).save(
        str(path), verbose=False
    )
    tracks = naps_track.load_sleap_tracks(path, 0, 5, "Tag", "Thorax")
    assert list(tracks.tag_locations[0]) == [3]
    assert 0 not in tracks.positions  # no usable position for the instance
    assert tracks.instances_by_frame[0] == [(3, None)]


def parse_minimal(*extra):
    """Parse the required options (and ``extra``) without running anything."""
    required = ["--slp-path", "a", "--video-path", "b", "--start-frame", "0", "--end-frame", "1"]
    return naps_track.build_parser().parse_args(
        [*required, "--aruco-marker-set", "DICT_4X4_100", *extra]
    )


def test_report_requires_the_recorded_pre_elimination_assignments():
    args = parse_minimal("--report-json", "r.json")

    class Unmatched:
        pre_elimination = None

    with pytest.raises(RuntimeError, match="did not record"):
        naps_track.report_diagnostics(args, naps_track.SleapTracks(), Unmatched(), {}, None)


def test_report_is_skipped_without_opt_in_options():
    # no --report-json, no closed set, no scope: returns before touching the matching
    args = parse_minimal()
    assert naps_track.report_diagnostics(args, naps_track.SleapTracks(), None, {}, None) is None


def test_parser_defaults_are_unchanged():
    args = parse_minimal()
    assert (args.tag_node_name, args.half_rolling_window_size) == ("tag", 5)
    assert (args.output_path, args.aruco_crop_size) == ("naps_output.analysis.h5", 100)
    assert (args.adaptiveThreshWinSizeMin, args.adaptiveThreshWinSizeMax) == (10, 30)
    assert (args.adaptiveThreshWinSizeStep, args.adaptiveThreshConstant) == (12, 3)
    assert (args.perspectiveRemoveIgnoredMarginPerCell, args.errorCorrectionRate) == (0.1, 1)
    assert args.closed_set is None
    assert args.closed_set_elimination is True
    assert args.quadrant_grid == (2, 2)
    assert args.idtracker_offset == (0.0, 0.0)
    assert (args.idtracker_weight, args.idtracker_min_prob) == (0.5, 0.5)
    assert args.h5_path is None
    assert (args.verbose, args.quiet) == (False, False)
