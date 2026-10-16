"""``TrackingArray`` (loading, interaction assignment, plotting) on synthetic coordinates."""

import csv
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shapely")
pytest.importorskip("yaml")

from naps.utils.tracking import INTERACTION_COLUMNS, TrackingArray
from tests.conftest import SyntheticColony

N_FRAMES = 6
# bees 0 and 1 are 30 px apart (their discs touch); bee 2 is far from both
POSITIONS = [(60.0, 60.0), (90.0, 60.0), (60.0, 200.0)]
DISC_MODEL = {
    "Skeleton": [
        {"name": "small", "shape": "point", "nodes": ["Tag"], "buffer": 10},
        {"name": "big", "shape": "point", "nodes": ["Tag"], "buffer": 25},
    ],
    "Models": [
        {"name": "small_disc", "includes": ["small"]},
        {"name": "big_disc", "includes": ["big"]},
    ],
    "Interactions": [
        {"name": "touch", "origin": "big_disc", "dest": "big_disc", "symmetical": True},
    ],
}


@pytest.fixture
def colony():
    return SyntheticColony(tags=(5, 6, 7), n_frames=N_FRAMES, positions=POSITIONS)


@pytest.fixture
def slp(tmp_path, colony, slp_factory):
    return slp_factory(tmp_path / "bees.slp", colony)


def read_tsv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def pairs(rows):
    return {(r["Origin interactor"], r["Destination interactor"]) for r in rows}


def test_default_constructor_is_empty(tmp_path):
    array = TrackingArray()
    assert array._frames_coords_array.shape[0] == 0
    out = tmp_path / "empty.tsv"
    array.assign(out)
    assert out.read_text().strip().split("\t") == INTERACTION_COLUMNS


def test_from_slp_loads_tracks_nodes_and_coordinates(slp):
    array = TrackingArray.fromSLEAP(slp, model_dict=DISC_MODEL, threads=2, chunk_size=3)
    assert array._tracks == ["track_0", "track_1", "track_2"]
    assert array._nodes == ["Tag", "Thorax"]
    assert array._frames_coords_array.shape == (N_FRAMES, 2, 2, 3)
    np.testing.assert_allclose(array._frames_coords_array[0, 0, :, 1], POSITIONS[1])
    np.testing.assert_allclose(array._frames_coords_array[0, 1, :, 1], (90.0, 72.0))
    assert (array._threads, array._chunk_size) == (2, 3)


def test_from_h5_matches_from_slp(slp, tmp_path, h5_factory):
    from_slp = TrackingArray.fromSLEAP(slp, model_dict=DISC_MODEL)
    h5 = h5_factory(
        tmp_path / "bees.analysis.h5",
        from_slp._frames_coords_array,
        from_slp._tracks,
        from_slp._nodes,
    )
    from_h5 = TrackingArray.fromH5(h5, model_dict=DISC_MODEL)
    assert from_h5._tracks == from_slp._tracks
    assert from_h5._nodes == from_slp._nodes
    np.testing.assert_array_equal(from_h5._frames_coords_array, from_slp._frames_coords_array)


@pytest.mark.parametrize("loader", ["fromSLEAP", "fromH5"])
def test_loaders_require_exactly_one_model(
    loader, slp, tmp_path, h5_factory, interactor_model_yaml
):
    source = slp
    if loader == "fromH5":
        base = TrackingArray.fromSLEAP(slp, model_dict=DISC_MODEL)
        source = h5_factory(tmp_path / "x.h5", base._frames_coords_array, base._tracks, base._nodes)
    load = getattr(TrackingArray, loader)
    with pytest.raises(ValueError, match="Only a single Interactor model"):
        load(source, model_dict=DISC_MODEL, model_yaml=interactor_model_yaml)
    with pytest.raises(ValueError, match="model is required"):
        load(source)
    assert load(source, model_yaml=interactor_model_yaml)._interactor_model_dict["Skeleton"]


def test_load_yaml_roundtrip_and_errors(tmp_path, interactor_model_dict, interactor_model_yaml):
    assert TrackingArray.loadYAML(interactor_model_yaml) == interactor_model_dict
    broken = tmp_path / "broken.yaml"
    broken.write_text("Skeleton: [unclosed")
    with pytest.raises(ValueError, match="Unable to parse"):
        TrackingArray.loadYAML(broken)


@pytest.mark.parametrize(
    ("text", "found"),
    [("", "NoneType"), ("# only a comment\n", "NoneType"), ("- a\n- b\n", "list"), ("42\n", "int")],
)
def test_load_yaml_requires_a_mapping(tmp_path, text, found):
    path = tmp_path / "model.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match=f"Expected a mapping.*{found}"):
        TrackingArray.loadYAML(path)
    # and the model loaders fail with this clear error, not a late KeyError
    with pytest.raises(ValueError, match="Expected a mapping"):
        TrackingArray._select_model(None, path)


def test_assign_finds_the_touching_pair_in_every_frame(slp, tmp_path):
    out = tmp_path / "interactions.tsv"
    TrackingArray.fromSLEAP(slp, model_dict=DISC_MODEL).assign(out)
    rows = read_tsv(out)
    assert list(rows[0]) == INTERACTION_COLUMNS
    assert pairs(rows) == {("track_0", "track_1")}
    assert [int(r["Interaction Frame"]) for r in rows] == list(range(N_FRAMES))
    assert {r["Interaction Name"] for r in rows} == {"touch"}
    # a buffered point is a 64-gon: just below pi r^2
    assert float(rows[0]["Origin Area"]) == pytest.approx(np.pi * 25**2, rel=0.01)


def test_assign_threads_and_chunks_give_the_same_rows(slp, tmp_path):
    reference = tmp_path / "reference.tsv"
    TrackingArray.fromSLEAP(slp, model_dict=DISC_MODEL).assign(reference)
    for name, kwargs in {
        "chunked": {"chunk_size": 4},
        "threads": {"threads": 2, "chunk_size": 4},
    }.items():
        out = tmp_path / f"{name}.tsv"
        TrackingArray.fromSLEAP(slp, model_dict=DISC_MODEL, **kwargs).assign(out)
        assert read_tsv(out) == read_tsv(reference), name


def test_assign_frame_range_and_missing_data(slp, tmp_path):
    array = TrackingArray.fromSLEAP(slp, model_dict=DISC_MODEL)
    array._frames_coords_array[3, :, :, 0] = np.nan  # bee 0 is not tracked in frame 3
    out = tmp_path / "range.tsv"
    array.assign(out, frame_start=1, frame_end=5)  # frame_end is exclusive
    assert [int(r["Interaction Frame"]) for r in read_tsv(out)] == [1, 2, 4]


def test_check_for_interactions_needs_a_chunk(slp):
    with pytest.raises(RuntimeError, match="no chunk is being processed"):
        TrackingArray.fromSLEAP(slp, model_dict=DISC_MODEL)._check_for_interactions(0, 1)


def test_process_interactions_file(tmp_path):
    tsv = tmp_path / "interactions.tsv"
    pd.DataFrame(
        {
            "Origin interactor": ["a", "a", "b"],
            "Destination interactor": ["b", "c", "c"],
            "Interaction Name": ["x", "y", "x"],
            "Interaction Frame": [1, 1, 2],
            "Origin Area": [1.0, 1.0, 1.0],
            "Destination Area": [1.0, 1.0, 1.0],
        }
    ).to_csv(tsv, sep="\t", index=False)

    frames = TrackingArray.processInteractionsFile(tsv, 0, 5)
    assert list(frames["Interaction Frame"]) == [1, 2]
    assert sorted(frames.loc[0, "Instances"]) == ["a", "a", "b", "c"]
    assert sorted(frames.loc[1, "Instances"]) == ["b", "c"]

    limited = TrackingArray.processInteractionsFile(tsv, 0, 5, limit_interactions=["y"])
    assert list(limited["Interaction Frame"]) == [1]
    assert sorted(limited.loc[0, "Instances"]) == ["a", "c"]

    with pytest.raises(ValueError, match="No interactions found between frames 3 to 9"):
        TrackingArray.processInteractionsFile(tsv, 3, 9)


def test_interpolate_fills_gaps_and_blanks_jumps():
    coords = np.full((6, 1, 2, 2), np.nan)
    coords[:, 0, 0, 0] = [0, np.nan, np.nan, 3, 4, 5]  # gap with a smooth fill
    coords[:, 0, 1, 0] = [0, 1, 2, 10, 11, 12]  # a jump of 8 between frames 2 and 3
    coords[:, 0, 0, 1] = 7.0
    coords[:, 0, 1, 1] = 9.0
    result = TrackingArray.interpolate(coords)
    assert result is coords  # modified in place
    np.testing.assert_allclose(result[:, 0, 0, 0], [0, 1, 2, 3, 4, 5])
    expected_y = [0, 1, 2, np.nan, 11, 12]
    np.testing.assert_allclose(result[:, 0, 1, 0], expected_y)
    assert (result[:, 0, 0, 1] == 7.0).all()


# --------------------------------------------------------------------------------- plot


@pytest.fixture
def plot_inputs(tmp_path, slp, colony, video_factory):
    array = TrackingArray.fromSLEAP(slp, model_dict=DISC_MODEL)
    tsv = tmp_path / "interactions.tsv"
    array.assign(tsv)
    video = video_factory(tmp_path / "bees.avi", colony)
    return array, tsv, video


def read_video(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    ok, frame = capture.read()
    while ok:
        frames.append(frame)
        ok, frame = capture.read()
    capture.release()
    return frames


def coloured_pixels(frame, channel):
    """Pixels where BGR ``channel`` clearly dominates (robust to lossy video compression)."""
    values = frame.astype(int)
    others = [values[:, :, c] for c in range(3) if c != channel]
    return int(((values[:, :, channel] - np.maximum(*others)) > 60).sum())


def red_pixels(frame):
    return coloured_pixels(frame, 2)


def test_plot_draws_only_the_requested_model(plot_inputs, tmp_path):
    array, tsv, video = plot_inputs
    outputs = {}
    for model in ("small_disc", "big_disc"):
        out = tmp_path / f"{model}.mp4"
        array.plot(tsv, [model], video, out)
        outputs[model] = read_video(out)
    for frames in outputs.values():
        assert len(frames) == N_FRAMES
        assert frames[0].shape == (240, 240, 3)
    # the default colour is red; each interacting bee gets a ring, nothing is drawn around bee 2
    small, big = (red_pixels(outputs[m][0]) for m in ("small_disc", "big_disc"))
    assert small > 20
    assert big > small  # the larger disc has the longer outline
    assert red_pixels(outputs["small_disc"][0][150:, :]) == 0  # bee 2 (y=200) is not interacting


def test_plot_unknown_model_raises(plot_inputs, tmp_path):
    array, tsv, video = plot_inputs
    with pytest.raises(KeyError, match="nope"):
        array.plot(tsv, ["nope"], video, tmp_path / "x.mp4")


def test_plot_colour_options(plot_inputs, tmp_path):
    array, tsv, video = plot_inputs
    out = tmp_path / "green.mp4"
    array.plot(tsv, ["big_disc"], video, out, plot_color=(0, 255, 0))
    frame = read_video(out)[0]
    assert red_pixels(frame) == 0
    assert coloured_pixels(frame, 1) > 20

    with pytest.raises(ValueError, match="Iterables must be of length 3"):
        array.plot(tsv, ["big_disc"], video, tmp_path / "bad.mp4", plot_color=(0, 255))


def test_plot_colour_name_is_resolved_by_matplotlib(plot_inputs, tmp_path, monkeypatch):
    """Colour names go through ``matplotlib.colors.to_rgb`` (a stand-in here: it is optional)."""
    fake = types.ModuleType("matplotlib.colors")
    monkeypatch.setattr(fake, "to_rgb", lambda name: {"blue": (0.0, 0.0, 1.0)}[name], raising=False)
    monkeypatch.setitem(sys.modules, "matplotlib.colors", fake)
    array, tsv, video = plot_inputs
    out = tmp_path / "blue.mp4"
    array.plot(tsv, ["big_disc"], video, out, plot_color="blue")
    assert coloured_pixels(read_video(out)[0], 0) > 20


def test_resolve_color():
    assert TrackingArray._resolve_color(None) == (255, 0, 0)
    assert TrackingArray._resolve_color([1, 2, 3]) == (1, 2, 3)
    with pytest.raises(ValueError, match="length 3"):
        TrackingArray._resolve_color([1, 2, 3, 4])


def test_plot_frame_range_and_limits(plot_inputs, tmp_path):
    array, tsv, video = plot_inputs
    out = tmp_path / "range.mp4"
    array.plot(tsv, ["big_disc"], video, out, frame_start=2, frame_end=4)
    frames = read_video(out)
    assert len(frames) == 3
    assert all(red_pixels(f) > 20 for f in frames)

    with pytest.raises(ValueError, match="No interactions found"):
        array.plot(tsv, ["big_disc"], video, tmp_path / "none.mp4", limit_interactions=["other"])
