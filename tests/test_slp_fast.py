"""The fast ``.slp`` frame writer must write the same datasets as sleap-io's own."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import sleap_io as sio

from naps import slp_fast
from naps.slp_fast import fast_writer_supported, save_labels, write_lfs_fast

DATA = Path(__file__).resolve().parent / "data"


def _datasets(path):
    out: dict[str, Any] = {}
    with h5py.File(path, "r") as f:

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = (
                    obj.dtype,
                    obj.shape,
                    obj[()].tobytes() if obj.dtype.hasobject is False else obj[()].tolist(),
                )
            out[name + "@attrs"] = {
                k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in obj.attrs.items()
            }

        f.visititems(visit)
        out["@root"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in f.attrs.items()}
    return out


def _assert_same(a, b):
    da, db = _datasets(a), _datasets(b)
    assert da.keys() == db.keys()
    for key in da:
        assert da[key] == db[key], key


def _both(labels, tmp_path):
    fast, stock = tmp_path / "fast.slp", tmp_path / "stock.slp"
    save_labels(labels, fast, fast=True)
    save_labels(labels, stock, fast=False)
    _assert_same(fast, stock)
    return fast, stock


def test_example_file_is_identical(tmp_path):
    assert fast_writer_supported()
    labels = sio.load_slp(str(DATA / "example.slp"))
    _both(labels, tmp_path)


def test_retracked_example_is_identical(tmp_path):
    labels = sio.load_slp(str(DATA / "example_naps_track_output.h5.slp"))
    fast, stock = _both(labels, tmp_path)
    # the files are in fact byte-identical on this stack; recorded, not required of other builds
    assert (
        hashlib.sha256(fast.read_bytes()).hexdigest()
        == hashlib.sha256(stock.read_bytes()).hexdigest()
    )


def _synthetic(seed):
    rng = np.random.default_rng(seed)
    skeleton = sio.Skeleton(nodes=[sio.Node(n) for n in ("a", "b", "c")], name="s")
    other = sio.Skeleton(nodes=[sio.Node(n) for n in ("a", "b")], name="o")
    videos = [sio.Video(filename=f"v{i}.mp4", open_backend=False) for i in range(2)]
    tracks = [sio.Track(name=f"track_{i}") for i in range(4)]
    lfs = []
    for frame in range(12):
        insts = []
        for k in range(int(rng.integers(0, 5))):
            pts = rng.random((3, 2)) * 100
            if k == 2:
                pts[1] = np.nan
            track = None if k == 3 else tracks[int(rng.integers(0, 4))]
            if k % 2:
                insts.append(
                    sio.PredictedInstance.from_numpy(
                        pts,
                        skeleton,
                        point_scores=rng.random(3),
                        score=float(rng.random()),
                        track=track,
                        tracking_score=None if k == 1 else float(rng.random()),
                    )
                )
            else:
                user = sio.Instance.from_numpy(pts, skeleton, track=track)
                insts.append(user)
        if frame % 5 == 4:
            insts.append(sio.Instance.from_numpy(rng.random((2, 2)), other))
        if insts and frame % 4 == 1 and isinstance(insts[0], sio.PredictedInstance):
            insts.append(
                sio.Instance.from_numpy(rng.random((3, 2)), skeleton, from_predicted=insts[0])
            )
        lfs.append(sio.LabeledFrame(video=videos[frame % 2], frame_idx=frame * 3, instances=insts))
    return sio.Labels(labeled_frames=lfs, videos=videos, skeletons=[skeleton, other], tracks=tracks)


@pytest.mark.parametrize("seed", range(5))
def test_synthetic_mixed_labels_are_identical(seed, tmp_path):
    _both(_synthetic(seed), tmp_path)


def test_removed_prediction_link_is_dropped(tmp_path):
    labels = _synthetic(1)
    # point a link at an instance that is not part of the labels
    ghost = sio.PredictedInstance.from_numpy(np.ones((3, 2)), labels.skeletons[0])
    labels.labeled_frames[0].instances.append(
        sio.Instance.from_numpy(np.ones((3, 2)), labels.skeletons[0], from_predicted=ghost)
    )
    _both(labels, tmp_path)


def test_empty_labels(tmp_path):
    skeleton = sio.Skeleton(nodes=[sio.Node("a")])
    video = sio.Video(filename="v.mp4", open_backend=False)
    labels = sio.Labels(
        labeled_frames=[sio.LabeledFrame(video=video, frame_idx=0)],
        videos=[video],
        skeletons=[skeleton],
    )
    _both(labels, tmp_path)
    _both(sio.Labels(videos=[video], skeletons=[skeleton]), tmp_path)


def test_unknown_instance_type_raises(tmp_path):
    class Odd(sio.Instance):
        pass

    labels = _synthetic(0)
    skeleton = labels.skeletons[0]
    labels.labeled_frames[0].instances.append(Odd.from_numpy(np.ones((3, 2)), skeleton))
    path = tmp_path / "x.slp"
    path.write_bytes(b"")
    with h5py.File(path, "w"):
        pass
    with pytest.raises(ValueError, match="Unknown instance type"):
        write_lfs_fast(path, labels)


def test_embedded_video_ids_are_kept():
    class Backend:
        def __init__(self, dataset):
            self.dataset = dataset

    class FakeVideo:
        def __init__(self, dataset):
            self.backend = Backend(dataset)

    class FakeLabels:
        videos = (
            FakeVideo("video15/video"),
            FakeVideo("videoX/video"),
            FakeVideo(None),
            FakeVideo("images/x"),
            object(),
        )

    assert slp_fast._video_ids(FakeLabels()) == {0: 15, 1: 1, 2: 2, 3: 3, 4: 4}


def test_falls_back_to_stock_writer(tmp_path, monkeypatch):
    labels = _synthetic(2)
    stock = tmp_path / "stock.slp"
    save_labels(labels, stock, fast=False)
    monkeypatch.setattr(slp_fast, "fast_writer_supported", lambda: False)
    out = tmp_path / "out.slp"
    save_labels(labels, out)  # unverified version: stock writer
    _assert_same(out, stock)


def test_lazy_labels_use_the_stock_writer(tmp_path):
    lazy = sio.load_slp(str(DATA / "example.slp"), lazy=True)
    out = tmp_path / "lazy.slp"
    save_labels(lazy, out)
    assert out.exists()


def test_writer_is_restored_after_an_error(tmp_path):
    original = slp_fast.sleap_io_slp.write_lfs
    labels = _synthetic(0)
    with pytest.raises(OSError, match="Unable"):
        save_labels(labels, tmp_path / "missing_dir" / "x.slp")
    assert slp_fast.sleap_io_slp.write_lfs is original
