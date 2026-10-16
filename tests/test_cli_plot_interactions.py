"""``naps-plot`` and ``naps-interactions`` command lines on a synthetic scene."""

import argparse
import csv
import logging
import os
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

pytest.importorskip("shapely")
pytest.importorskip("yaml")

import yaml

from naps import naps_interactions, naps_plot
from naps.utils.cli import ConfirmFile, log_verbosity
from naps.utils.tracking import TrackingArray
from tests.conftest import SyntheticColony

REPO_ROOT = Path(__file__).resolve().parents[1]
N_FRAMES = 6
MODEL = {
    "Skeleton": [{"name": "big", "shape": "point", "nodes": ["Tag"], "buffer": 25}],
    "Models": [{"name": "disc", "includes": ["big"]}],
    "Interactions": [{"name": "touch", "origin": "disc", "dest": "disc", "symmetical": True}],
}


class Scene:
    """slp, analysis h5, video and YAML model of two touching bees and a distant third."""

    def __init__(self, tmp_path, slp_factory, video_factory, h5_factory):
        colony = SyntheticColony(
            tags=(5, 6, 7), n_frames=N_FRAMES, positions=[(60.0, 60.0), (90.0, 60.0), (60.0, 200.0)]
        )
        self.slp = slp_factory(tmp_path / "bees.slp", colony)
        array = TrackingArray.fromSLEAP(self.slp, model_dict=MODEL)
        self.yaml = tmp_path / "model.yaml"
        self.yaml.write_text(yaml.safe_dump(MODEL))
        self.h5 = h5_factory(
            tmp_path / "bees.analysis.h5", array._frames_coords_array, array._tracks, array._nodes
        )
        self.video = video_factory(tmp_path / "bees.avi", colony)
        self.tsv = tmp_path / "interactions.tsv"

    def interaction_args(self, *extra):
        return [
            "--h5-file", str(self.h5),
            "--yaml-model", str(self.yaml),
            "--out-file", str(self.tsv),
            *extra,
        ]  # fmt: skip

    def read_rows(self):
        with self.tsv.open(newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))


@pytest.fixture
def scene(tmp_path, slp_factory, video_factory, h5_factory):
    return Scene(tmp_path, slp_factory, video_factory, h5_factory)


@pytest.mark.parametrize(
    ("module", "prog", "flag"),
    [
        (naps_plot, "naps-plot", "--models-to-plot"),
        (naps_interactions, "naps-interactions", "--h5-file"),
    ],
)
def test_help(module, prog, flag, capsys):
    with pytest.raises(SystemExit) as exc:
        module.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert f"usage: {prog}" in out
    assert flag in out
    assert "--verbose" in out
    assert "--quiet" in out


@pytest.mark.parametrize("module", ["naps.naps_plot", "naps.naps_interactions"])
def test_modules_run_as_scripts(module, tmp_path):
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(  # noqa: S603 (fixed argv, no shell)
        [sys.executable, "-m", module, "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("module", [naps_plot, naps_interactions])
def test_missing_arguments_exit_with_status_2(module):
    with pytest.raises(SystemExit) as exc:
        module.main([])
    assert exc.value.code == 2


def test_interactions_cli_writes_the_tsv(scene):
    assert naps_interactions.main(scene.interaction_args()) == 0
    rows = scene.read_rows()
    assert len(rows) == N_FRAMES
    pairs = {(r["Origin interactor"], r["Destination interactor"]) for r in rows}
    assert pairs == {("track_0", "track_1")}


def test_interactions_cli_frame_range_and_threads(scene):
    extra = ["--frame-start", "2", "--frame-end", "5", "--threads", "2", "--quiet"]
    naps_interactions.main(scene.interaction_args(*extra))
    assert [int(r["Interaction Frame"]) for r in scene.read_rows()] == [2, 3, 4]


def test_plot_cli_writes_a_video(scene, tmp_path):
    naps_interactions.main(scene.interaction_args())
    out = tmp_path / "plotted.mp4"
    argv = [
        "--sleap-file", str(scene.slp),
        "--yaml-model", str(scene.yaml),
        "--interactions-file", str(scene.tsv),
        "--video-file", str(scene.video),
        "--models-to-plot", "disc",
        "--out-file", str(out),
    ]  # fmt: skip
    assert naps_plot.main(argv) == 0
    capture = cv2.VideoCapture(str(out))
    n_frames = 0
    while capture.read()[0]:
        n_frames += 1
    capture.release()
    assert n_frames == N_FRAMES


def test_interactions_cli_missing_input_file_raises_an_oserror(scene, tmp_path):
    argv = ["--h5-file", str(tmp_path / "missing.h5"), "--yaml-model", str(scene.yaml)]
    with pytest.raises(OSError, match=r"missing\.h5 not found"):
        naps_interactions.main(argv)


def test_plot_cli_missing_input_file_raises_an_oserror(scene, tmp_path):
    argv = [
        "--sleap-file", str(tmp_path / "missing.slp"),
        "--yaml-model", str(scene.yaml),
        "--interactions-file", "x", "--video-file", "x", "--models-to-plot", "m",
        "--out-file", str(tmp_path / "out.mp4"),
    ]  # fmt: skip
    with pytest.raises(OSError, match=r"missing\.slp not found"):
        naps_plot.main(argv)


def test_legacy_parser_functions_still_parse(scene):
    plot_argv = [
        "--sleap-file", str(scene.slp), "--yaml-model", str(scene.yaml),
        "--interactions-file", "i.tsv", "--video-file", "v.avi",
        "--models-to-plot", "a", "b", "--out-file", "o.mp4",
    ]  # fmt: skip
    plot_args = naps_plot.plotParser(plot_argv)
    assert plot_args.models_to_plot == ["a", "b"]
    assert (plot_args.frame_start, plot_args.frame_end) == (0, -1)
    inter_args = naps_interactions.interactionParser(
        ["--h5-file", str(scene.h5), "--yaml-model", str(scene.yaml)]
    )
    assert inter_args.out_file == "out.tsv"
    assert inter_args.threads == 1


def test_confirm_file_action(tmp_path):
    present = tmp_path / "here.txt"
    present.write_text("x")
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", action=ConfirmFile)
    assert parser.parse_args(["--file", str(present)]).file == str(present)
    with pytest.raises(FileNotFoundError, match=r"gone\.txt not found"):
        parser.parse_args(["--file", str(tmp_path / "gone.txt")])


def test_log_verbosity_restores_levels():
    loggers = [logging.getLogger(n) for n in ("NAPS Logger", "naps")]
    before = [lg.level for lg in loggers]
    with log_verbosity(verbose=True):
        assert all(lg.level == logging.DEBUG for lg in loggers)
    with log_verbosity(quiet=True):
        assert all(lg.level == logging.WARNING for lg in loggers)
    with log_verbosity():
        assert [lg.level for lg in loggers] == before
    assert [lg.level for lg in loggers] == before
