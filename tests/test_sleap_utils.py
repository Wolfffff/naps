import logging

import h5py
import numpy as np
import pytest
import sleap_io as sio

from naps.sleap_utils import get_location_matrix, load_tracks_from_slp, update_labeled_frames


def _build_slp(path):
    skel = sio.Skeleton(["Tag", "Head"])
    video = sio.Video(filename="missing_video.mp4", open_backend=False)
    t0, t1 = sio.Track("track_0"), sio.Track("track_1")
    odd = sio.Track("weird")  # no integer suffix -> unmappable
    lfs = []
    # Sparse frame indices so list position != frame_idx
    for fi in [3, 4, 5, 6, 7, 8]:
        insts = [
            sio.PredictedInstance.from_numpy(
                np.array([[fi, 1.0], [fi, 2.0]]), skeleton=skel, track=t0, score=1.0
            ),
            sio.PredictedInstance.from_numpy(
                np.array([[fi, 10.0], [fi, 20.0]]), skeleton=skel, track=t1, score=1.0
            ),
        ]
        if fi == 7:
            insts.append(
                sio.PredictedInstance.from_numpy(
                    np.array([[0.0, 0.0], [1.0, 1.0]]),
                    skeleton=skel,
                    track=odd,
                    score=1.0,
                )
            )
            insts.append(
                sio.PredictedInstance.from_numpy(
                    np.array([[2.0, 2.0], [3.0, 3.0]]), skeleton=skel, score=1.0
                )
            )
        lfs.append(sio.LabeledFrame(video=video, frame_idx=fi, instances=insts))
    sio.Labels(lfs).save(str(path), verbose=False)


def test_update_labeled_frames(tmp_path):
    slp = tmp_path / "in.slp"
    _build_slp(slp)

    # Frame 6 has no entry for track_1 (KeyError); frame 4 is outside the range
    matching_dict = {
        5: {0: 12, 1: 34},
        6: {0: 12},
        7: {0: 12, 1: 34},
        4: {0: 99, 1: 98},
    }
    out = update_labeled_frames(str(slp), matching_dict, 5, 7)

    # Selected by frame_idx range (inclusive), not by list position
    assert [lf.frame_idx for lf in out.labeled_frames] == [5, 6, 7]

    names = {
        lf.frame_idx: [i.track.name if i.track else None for i in lf.instances]
        for lf in out.labeled_frames
    }
    assert names[5] == ["ArUcoTag#12", "ArUcoTag#34"]
    # track_1 not in matching_dict[6] -> None
    assert names[6] == ["ArUcoTag#12", None]
    # unparsable track name -> None; no track at all -> None
    assert names[7] == ["ArUcoTag#12", "ArUcoTag#34", None, None]

    # Tag 98/99 only appear outside the range but still create tracks
    # harmlessly; only tracks actually used are registered in the output.
    assert {t.name for t in out.tracks} == {"ArUcoTag#12", "ArUcoTag#34"}

    # Points are preserved
    assert np.allclose(out.labeled_frames[0].instances[0].numpy(), [[5, 1], [5, 2]])

    # Round-trips through save/load and the location-matrix helper
    out_path = tmp_path / "out.slp"
    out.save(str(out_path), verbose=False)
    locs, nodes = load_tracks_from_slp(str(out_path))
    assert nodes == ["Tag", "Head"]
    assert locs.shape[0] == 8  # all_frames=True -> frames 0..7
    assert locs.shape[3] == 2
    assert np.isnan(locs[4]).all()  # frame 4 was excluded


# ------------------------------------------------------------------ in-memory helpers
SKELETON = sio.Skeleton(["Tag", "Head"])
VIDEO = sio.Video(filename="missing_video.mp4", open_backend=False)


def _instance(x, track=None, *, predicted=True):
    points = np.array([[x, 1.0], [x, 2.0]])
    if predicted:
        return sio.PredictedInstance.from_numpy(points, skeleton=SKELETON, track=track, score=1.0)
    return sio.Instance.from_numpy(points, skeleton=SKELETON, track=track)


def _labels(frames, tracks=()):
    """``frames``: ``{frame_idx: [instance, ...]}``."""
    lfs = [
        sio.LabeledFrame(video=VIDEO, frame_idx=fi, instances=insts) for fi, insts in frames.items()
    ]
    return sio.Labels(lfs, tracks=list(tracks), videos=[VIDEO], skeletons=[SKELETON])


# --------------------------------------------------------------- get_location_matrix
def test_location_matrix_all_frames_lines_up_frame_index_and_leaves_nan():
    t0, t1 = sio.Track("track_0"), sio.Track("track_1")
    labels = _labels(
        {2: [_instance(2.0, t0)], 4: [_instance(4.0, t0), _instance(40.0, t1)]}, [t0, t1]
    )

    locs, nodes = get_location_matrix(labels, all_frames=True)

    assert locs.shape == (5, 2, 2, 2)  # frames, nodes, xy, tracks
    assert [n.name for n in nodes] == ["Tag", "Head"]
    assert np.allclose(locs[2, :, :, 0], [[2, 1], [2, 2]])
    assert np.allclose(locs[4, :, :, 1], [[40, 1], [40, 2]])
    assert np.isnan(locs[[0, 1, 3]]).all()
    assert np.isnan(locs[2, :, :, 1]).all()


def test_location_matrix_without_all_frames_starts_at_the_first_labeled_frame():
    t0 = sio.Track("track_0")
    labels = _labels({10: [_instance(10.0, t0)], 12: [_instance(12.0, t0)]}, [t0])

    locs, _ = get_location_matrix(labels, all_frames=False)

    assert locs.shape[0] == 3
    assert np.allclose(locs[0, :, :, 0], [[10, 1], [10, 2]])
    assert np.isnan(locs[1]).all()
    assert np.allclose(locs[2, :, :, 0], [[12, 1], [12, 2]])


def test_location_matrix_prefers_user_instances_over_predicted_ones():
    t0 = sio.Track("track_0")
    labels = _labels(
        {0: [_instance(1.0, t0, predicted=True), _instance(2.0, t0, predicted=False)]}, [t0]
    )

    locs, _ = get_location_matrix(labels, all_frames=True)

    assert np.allclose(locs[0, :, :, 0], [[2, 1], [2, 2]])


def test_location_matrix_without_tracks_uses_a_single_column():
    labels = _labels({0: [_instance(1.0)], 1: [_instance(3.0)]})

    locs, _ = get_location_matrix(labels, all_frames=True)

    assert locs.shape == (2, 2, 2, 1)
    assert np.allclose(locs[1, :, :, 0], [[3, 1], [3, 2]])


def test_location_matrix_duplicate_instances_on_a_track_last_one_wins_and_warns(caplog):
    t0 = sio.Track("track_0")
    labels = _labels({0: [_instance(1.0, t0), _instance(9.0, t0)]}, [t0])

    with caplog.at_level(logging.WARNING, logger="naps.sleap_utils"):
        locs, _ = get_location_matrix(labels, all_frames=True)

    assert np.allclose(locs[0, :, :, 0], [[9, 1], [9, 2]])
    assert "more than one instance per track" in caplog.text


def test_location_matrix_explicit_video_and_errors():
    other = sio.Video(filename="other.mp4", open_backend=False)
    t0 = sio.Track("track_0")
    labels = _labels({0: [_instance(1.0, t0)]}, [t0])
    labels.videos.append(other)

    with pytest.raises(IndexError, match=r"No labeled frames in other\.mp4"):
        get_location_matrix(labels, all_frames=True, video=other)

    with pytest.raises(IndexError, match="no videos"):
        get_location_matrix(sio.Labels([], videos=[], skeletons=[SKELETON]), all_frames=True)
    with pytest.raises(IndexError, match="No labeled frames"):
        get_location_matrix(_labels({}), all_frames=True)


# ------------------------------------------------------------- load_tracks_from_slp
def test_load_tracks_from_analysis_h5(tmp_path):
    path = tmp_path / "tracks.analysis.h5"
    tracks = np.arange(2 * 2 * 3 * 4, dtype=float).reshape(2, 2, 3, 4)  # tracks, xy, nodes, frames
    with h5py.File(path, "w") as f:
        f["tracks"] = tracks
        f["node_names"] = np.array([b"tag", b"head", b"abdomen"])

    locs, names = load_tracks_from_slp(str(path))

    assert names == ["tag", "head", "abdomen"]
    assert locs.shape == (4, 3, 2, 2)  # frames, nodes, xy, tracks
    assert locs[1, 2, 0, 1] == tracks[1, 0, 2, 1]


def test_load_tracks_from_slp_accepts_path_objects(tmp_path):
    t0 = sio.Track("track_0")
    path = tmp_path / "labels.slp"
    _labels({1: [_instance(5.0, t0)]}, [t0]).save(str(path), verbose=False)

    locs, names = load_tracks_from_slp(path)

    assert names == ["Tag", "Head"]
    assert locs.shape == (2, 2, 2, 1)


@pytest.mark.parametrize("name", ["tracks.txt", "tracks", "tracks.SLP"])
def test_load_tracks_rejects_unknown_suffixes(name, tmp_path):
    with pytest.raises(ValueError, match="expected a SLEAP project"):
        load_tracks_from_slp(str(tmp_path / name))


# --------------------------------------------------------- update_labeled_frames
def _save(tmp_path, frames, tracks):
    path = tmp_path / "in.slp"
    _labels(frames, tracks).save(str(path), verbose=False)
    return path


def _names(labels):
    return {
        lf.frame_idx: [i.track.name if i.track else None for i in lf.instances]
        for lf in labels.labeled_frames
    }


def test_update_normalises_tag_types_and_skips_invalid_ones(tmp_path, caplog):
    t = [sio.Track(f"track_{i}") for i in range(6)]
    path = _save(tmp_path, {0: [_instance(float(i), t[i]) for i in range(6)]}, t)
    matching = {0: {0: 21.0, 1: np.int64(22), 2: np.nan, 3: 2.5, 4: "", 5: None}}

    with caplog.at_level(logging.WARNING, logger="naps.sleap_utils"):
        out = update_labeled_frames(path, matching, 0, 0)

    assert _names(out)[0] == ["ArUcoTag#21", "ArUcoTag#22", None, None, None, None]
    assert "Ignored 2 invalid" in caplog.text  # nan and 2.5; blank assignments are not noise


def test_update_never_gives_one_tag_to_two_instances_of_a_frame(tmp_path, caplog):
    t0, t1 = sio.Track("track_0"), sio.Track("track_1")
    # frame 0: two instances on SLEAP track 0 (both map to tag 7); frame 1: tracks 0 and 1 both
    # mapped to tag 7 by the matching; frame 2: an ordinary frame
    path = _save(
        tmp_path,
        {
            0: [_instance(1.0, t0), _instance(2.0, t0), _instance(3.0, t1)],
            1: [_instance(1.0, t0), _instance(2.0, t1)],
            2: [_instance(1.0, t0), _instance(2.0, t1)],
        },
        [t0, t1],
    )
    matching = {0: {0: 7, 1: 8}, 1: {0: 7, 1: 7}, 2: {0: 7, 1: 8}}

    with caplog.at_level(logging.WARNING, logger="naps.sleap_utils"):
        out = update_labeled_frames(str(path), matching, 0, 2)

    names = _names(out)
    assert names[0] == [None, "ArUcoTag#7", "ArUcoTag#8"]  # the last instance keeps the tag
    assert names[1] == [None, "ArUcoTag#7"]
    assert names[2] == ["ArUcoTag#7", "ArUcoTag#8"]
    assert "2 instances shared a tag" in caplog.text
    for lf in out.labeled_frames:
        tags = [i.track.name for i in lf.instances if i.track is not None]
        assert len(tags) == len(set(tags))


def test_update_leaves_untracked_instances_and_missing_frames_untracked(tmp_path):
    t0 = sio.Track("track_0")
    path = _save(tmp_path, {0: [_instance(1.0, t0), _instance(2.0)], 1: [_instance(1.0, t0)]}, [t0])

    out = update_labeled_frames(path, {0: {0: 5}}, 0, 1)  # frame 1 is not in the matching dict

    assert _names(out) == {0: ["ArUcoTag#5", None], 1: [None]}


def test_update_with_an_empty_range_or_matching_returns_no_frames(tmp_path):
    t0 = sio.Track("track_0")
    path = _save(tmp_path, {3: [_instance(1.0, t0)]}, [t0])

    assert update_labeled_frames(path, {3: {0: 5}}, 4, 9).labeled_frames == []
    assert _names(update_labeled_frames(path, {}, 0, 9)) == {3: [None]}
