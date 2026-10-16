"""The table reader must give exactly what the eager (sleap-io objects) loader gives."""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import sleap_io as sio

from naps import naps_track
from naps.slp_tracks import UnsupportedSlpError, iter_window_instances

DATA = Path(__file__).resolve().parent / "data"


def _freeze(value):
    """Nested dict/list/array to comparable plain data (NaN compares equal, order kept)."""
    if isinstance(value, np.ndarray):
        return [("nan" if math.isnan(v) else float(v)) for v in value.ravel()]
    if isinstance(value, dict):
        return [(k, _freeze(v)) for k, v in value.items()]
    if isinstance(value, (list, tuple)):
        return [_freeze(v) for v in value]
    return value


def _tracks_plain(t):
    return {
        "tag_locations": _freeze(dict(t.tag_locations)),
        "positions": _freeze(dict(t.positions)),
        "instances_by_frame": _freeze(dict(t.instances_by_frame)),
        "untracked_counts": dict(t.untracked_counts),
        "untracked_positions": _freeze(dict(t.untracked_positions)),
    }


def _same(path, start, end, tag="Tag", fallback="Thorax"):
    fast = naps_track.load_sleap_tracks(path, start, end, tag, fallback, fast=True)
    eager = naps_track.load_sleap_tracks(path, start, end, tag, fallback, fast=False)
    assert _tracks_plain(fast) == _tracks_plain(eager)
    return fast


def _synthetic_file(tmp_path, seed=0, *, n_frames=40):
    """Two skeletons, user and predicted instances, untracked instances, NaN tags/thoraxes."""
    rng = np.random.default_rng(seed)
    sk1 = sio.Skeleton([sio.Node("Tag"), sio.Node("Thorax"), sio.Node("Head")])
    sk2 = sio.Skeleton([sio.Node("Tag"), sio.Node("Other")])  # no Thorax: no fallback node
    video = sio.Video(filename="v.mp4", open_backend=False)
    tracks = [sio.Track(f"track_{k}") for k in range(5)]
    lfs = []
    for f in range(n_frames):
        insts = []
        for _ in range(int(rng.integers(0, 5))):
            pts = rng.random((3, 2)) * 200
            u = rng.random()
            if u < 0.2:
                pts[0] = np.nan  # tag lost: fallback
            elif u < 0.3:
                pts[0] = np.nan
                pts[1] = np.nan  # tag and thorax lost: mean of the rest
            elif u < 0.35:
                pts[:] = np.nan  # nothing at all
            track = None if rng.random() < 0.2 else tracks[int(rng.integers(0, 5))]
            if rng.random() < 0.3:
                insts.append(sio.Instance.from_numpy(pts, sk1, track=track))
            else:
                insts.append(sio.PredictedInstance.from_numpy(pts, sk1, score=0.9, track=track))
        if f % 7 == 3:
            p2 = rng.random((2, 2)) * 200
            if f % 2:
                p2[0] = np.nan
            insts.append(sio.PredictedInstance.from_numpy(p2, sk2, score=0.5, track=tracks[1]))
        if insts:
            lfs.append(sio.LabeledFrame(video=video, frame_idx=f, instances=insts))
    path = tmp_path / f"synthetic{seed}.slp"
    sio.Labels(labeled_frames=lfs, videos=[video], skeletons=[sk1, sk2], tracks=tracks).save(
        str(path), verbose=False
    )
    return path


def test_example_file_equals_eager():
    labels = sio.load_slp(str(DATA / "example.slp"))
    tag = next(n for n in labels.skeletons[0].node_names if n.lower() == "tag")
    _same(str(DATA / "example.slp"), 0, 10_000, tag, "Thorax")


@pytest.mark.parametrize("seed", range(6))
def test_synthetic_files_equal_eager(tmp_path, seed):
    path = _synthetic_file(tmp_path, seed)
    _same(path, 0, 10_000)
    _same(path, 5, 20)  # a window
    _same(path, 100, 200)  # an empty window


def test_unnamed_fallback_and_window_edges(tmp_path):
    path = _synthetic_file(tmp_path, 3)
    _same(path, 10, 10, fallback="NoSuchNode")


def test_missing_tag_node_raises_like_the_eager_loader(tmp_path):
    path = _synthetic_file(tmp_path)
    with pytest.raises(ValueError, match="tag node 'Nope'"):
        naps_track.load_sleap_tracks(path, 0, 100, "Nope", "Thorax", fast=True)
    with pytest.raises(ValueError, match="tag node 'Nope'"):
        naps_track.load_sleap_tracks(path, 0, 100, "Nope", "Thorax", fast=False)


def test_non_numeric_track_name_raises(tmp_path):
    skeleton = sio.Skeleton([sio.Node("Tag")])
    video = sio.Video(filename="v.mp4", open_backend=False)
    track = sio.Track("alice")
    lf = sio.LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[sio.PredictedInstance.from_numpy(np.ones((1, 2)), skeleton, track=track)],
    )
    path = tmp_path / "x.slp"
    sio.Labels(labeled_frames=[lf], videos=[video], skeletons=[skeleton], tracks=[track]).save(
        str(path), verbose=False
    )
    with pytest.raises(ValueError, match="alice"):
        list(iter_window_instances(path, 0, 5, "Tag", "Thorax", naps_track._instance_position))
    with pytest.raises(ValueError, match="alice"):
        naps_track.load_sleap_tracks(path, 0, 5, "Tag", "Thorax", fast=False)


def test_missing_tables_fall_back_to_the_eager_loader(tmp_path, caplog):
    path = _synthetic_file(tmp_path)
    broken = tmp_path / "broken.slp"
    broken.write_bytes(path.read_bytes())
    with h5py.File(broken, "a") as f:
        del f["pred_points"]
    with pytest.raises(UnsupportedSlpError):
        list(iter_window_instances(broken, 0, 5, "Tag", "Thorax", naps_track._instance_position))
    with pytest.raises(UnsupportedSlpError):
        list(
            iter_window_instances(
                tmp_path / "missing.slp", 0, 5, "Tag", "Thorax", naps_track._instance_position
            )
        )


def test_loader_falls_back_when_the_reader_refuses(tmp_path, monkeypatch, caplog):
    path = _synthetic_file(tmp_path)

    def refuse(*args, **kwargs):
        raise UnsupportedSlpError("test")
        yield  # pragma: no cover - makes this a generator like the real function

    monkeypatch.setattr(naps_track, "iter_window_instances", refuse)
    with caplog.at_level("INFO"):
        fallback = naps_track.load_sleap_tracks(path, 0, 100, "Tag", "Thorax", fast=True)
    assert "Fast .slp reader not used" in caplog.text
    eager = naps_track.load_sleap_tracks(path, 0, 100, "Tag", "Thorax", fast=False)
    assert _tracks_plain(fallback) == _tracks_plain(eager)


@pytest.mark.parametrize("damage", ["layout", "point_count", "instance_type"])
def test_unexpected_layouts_are_refused(tmp_path, damage):
    path = _synthetic_file(tmp_path)
    bad = tmp_path / "bad.slp"
    bad.write_bytes(path.read_bytes())
    with h5py.File(bad, "a") as f:
        if damage == "layout":
            frames = f["frames"][()]
            del f["frames"]
            names = [n for n in frames.dtype.names if n != "frame_idx"]
            f.create_dataset("frames", data=frames[names])
        else:
            inst = f["instances"][()]
            if damage == "point_count":
                inst["point_id_end"][0] += 1
            else:
                inst["instance_type"][0] = 7
            f["instances"][...] = inst
    with pytest.raises(UnsupportedSlpError):
        list(iter_window_instances(bad, 0, 1000, "Tag", "Thorax", naps_track._instance_position))


def test_the_empty_window_yields_nothing(tmp_path):
    path = _synthetic_file(tmp_path)
    assert (
        list(iter_window_instances(path, 500, 600, "Tag", "Thorax", naps_track._instance_position))
        == []
    )
