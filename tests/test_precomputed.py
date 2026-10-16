"""Precomputed decodes: the loader, ``Matching(decode_source=...)`` and ``naps-track --decodes``."""

import importlib
import logging
from collections import defaultdict

import numpy as np
import pytest
import sleap_io as sio

from naps.matching import Matching
from naps.precomputed import SCHEMA_VERSION, DecodesError, PrecomputedDecodeSource

naps_track = importlib.import_module("naps.naps_track")


def write_decodes(path, counts, xy, votes, *, first_frame=0, video="clip.mp4", **override):
    """Write a decodes file in the documented layout.

    ``counts``: instances per frame (list); ``xy``: ``(F, S, 2)`` positions; ``votes``:
    ``[(frame, instance, tag), ...]``.
    """
    n = len(counts)
    arrays = {
        "schema_version": np.array(SCHEMA_VERSION),
        "video": np.array(video),
        "frame_size": np.array([200, 100], dtype=np.int32),
        "frame_idx": np.arange(first_frame, first_frame + n, dtype=np.int32),
        "n_inst": np.asarray(counts, dtype=np.uint8),
        "tag_xy": np.asarray(xy, dtype=np.float32),
        "vote_frame": np.array([v[0] for v in votes], dtype=np.int32),
        "vote_inst": np.array([v[1] for v in votes], dtype=np.uint8),
        "vote_tag": np.array([v[2] for v in votes], dtype=np.int16),
    }
    arrays.update(override)
    np.savez_compressed(path, **arrays)
    return path


def positions(n_frames=3, n_slots=3):
    """``(F, S, 2)`` tag nodes: instance j sits at (10 + 30 j, 20 + frame)."""
    xy = np.zeros((n_frames, n_slots, 2), dtype=np.float32)
    for f in range(n_frames):
        for j in range(n_slots):
            xy[f, j] = (10 + 30 * j, 20 + f)
    return xy


def coords_of(xy, n_inst, first_frame=0):
    """``coords[frame][instance] -> (x, y)`` for the instances of a decodes file."""
    return {
        first_frame + f: {j: tuple(map(float, xy[f, j])) for j in range(n_inst[f])}
        for f in range(len(n_inst))
    }


@pytest.fixture
def basic(tmp_path):
    """3 frames x 3 instances; frame 1 has 2 votes for instance 1 and none for instance 2."""
    n_inst = [3, 3, 3]
    xy = positions()
    xy[2, 2] = np.nan  # instance 2 has no tag node in frame 2
    votes = [(0, 0, 45), (0, 1, 46), (0, 2, 47), (1, 0, 45), (1, 1, 46), (1, 1, 99), (2, 0, 45)]
    path = write_decodes(tmp_path / "d.npz", n_inst, xy, votes)
    return path, n_inst, xy


def test_serves_the_stored_votes_and_none_for_the_rest(basic):
    path, n_inst, xy = basic
    source = PrecomputedDecodeSource(path)
    decodes, frame_size = source.decode(0, 2, coords_of(xy, n_inst))
    assert frame_size == (200, 100)
    assert (source.first_frame, source.last_frame) == (0, 2)
    assert decodes[0] == {0: [45], 1: [46], 2: [47]}
    assert decodes[1] == {0: [45], 1: [46, 99], 2: [None]}  # two markers in one crop; nothing
    assert decodes[2] == {0: [45], 1: [None], 2: [None]}  # nothing; no tag node
    assert all(isinstance(t, int) for row in decodes.values() for v in row.values() for t in v if t)
    assert isinstance(decodes[0], defaultdict)


def test_serves_a_sub_range_and_a_file_that_starts_late(tmp_path):
    n_inst = [2, 2, 2, 2]
    xy = positions(4, 2)
    path = write_decodes(
        tmp_path / "late.npz", n_inst, xy, [(11, 0, 5), (12, 1, 6), (13, 0, 7)], first_frame=10
    )
    source = PrecomputedDecodeSource(path)
    assert (source.first_frame, source.last_frame) == (10, 13)
    decodes, _ = source.decode(11, 12, coords_of(xy, n_inst, first_frame=10))
    assert sorted(decodes) == [11, 12]
    assert decodes[11] == {0: [5], 1: [None]}
    assert decodes[12] == {0: [None], 1: [6]}


def test_unsorted_votes_are_grouped_by_frame(tmp_path):
    xy = positions(2, 1)
    path = write_decodes(
        tmp_path / "u.npz", [1, 1], xy, [(1, 0, 8), (0, 0, 7), (1, 0, 9)]
    )  # stable order within a frame
    decodes, _ = PrecomputedDecodeSource(path).decode(0, 1, coords_of(xy, [1, 1]))
    assert decodes[0] == {0: [7]}
    assert decodes[1] == {0: [8, 9]}


@pytest.mark.parametrize(("first", "last"), [(-1, 2), (0, 3), (5, 9)])
def test_a_frame_range_outside_the_file_is_refused(basic, first, last):
    path = basic[0]
    coords = {f: {} for f in range(first, last + 1)}
    message = rf"covers frames 0..2 but the run needs {first}..{last}"
    with pytest.raises(DecodesError, match=message):
        PrecomputedDecodeSource(path).decode(first, last, coords)


def test_a_different_instance_count_is_refused(basic):
    path, n_inst, xy = basic
    coords = coords_of(xy, n_inst)
    del coords[1][2]
    with pytest.raises(DecodesError, match=r"frame 1: the SLEAP file has 2 instances"):
        PrecomputedDecodeSource(path).decode(0, 2, coords)


def test_a_different_tag_node_is_refused(basic):
    path, n_inst, xy = basic
    coords = coords_of(xy, n_inst)
    coords[1][1] = (coords[1][1][0] + 5.0, coords[1][1][1])
    with pytest.raises(DecodesError, match=r"frame 1 instance 1: tag node"):
        PrecomputedDecodeSource(path).decode(0, 2, coords)


def test_a_nan_that_is_not_stored_as_nan_is_refused(basic):
    path, n_inst, xy = basic
    coords = coords_of(xy, n_inst)
    coords[0][0] = (np.nan, np.nan)
    with pytest.raises(DecodesError, match="frame 0 instance 0"):
        PrecomputedDecodeSource(path).decode(0, 2, coords)


def test_positions_agree_within_float32_rounding(basic):
    path, n_inst, xy = basic
    coords = coords_of(xy, n_inst)
    coords[0][0] = (coords[0][0][0] + 1e-5, coords[0][0][1])
    PrecomputedDecodeSource(path).decode(0, 2, coords)


def test_at_most_a_few_problems_are_reported(tmp_path):
    n = 40
    xy = positions(n, 1)
    path = write_decodes(tmp_path / "m.npz", [1] * n, xy, [])
    coords = {f: {0: (999.0, 999.0)} for f in range(n)}
    with pytest.raises(DecodesError) as exc:
        PrecomputedDecodeSource(path).decode(0, n - 1, coords)
    assert str(exc.value).count("tag node") == 5


def test_keys_map_track_ids_to_instances(tmp_path):
    n_inst = [3, 3]
    xy = positions(2, 3)
    path = write_decodes(tmp_path / "k.npz", n_inst, xy, [(0, 0, 45), (0, 2, 47), (1, 1, 46)])
    keys = {0: {7: 0, 8: 2}, 1: {7: 1}}  # frame 1: only track 7 (= instance 1) is tracked
    coords = {
        0: {7: tuple(xy[0, 0]), 8: tuple(xy[0, 2])},
        1: {7: tuple(xy[1, 1])},
    }
    decodes, _ = PrecomputedDecodeSource(path, keys=keys).decode(0, 1, coords)
    assert decodes[0] == {7: [45], 8: [47]}
    assert decodes[1] == {7: [46]}


def test_keys_of_unknown_tracks_are_refused(tmp_path):
    xy = positions(1, 2)
    path = write_decodes(tmp_path / "k2.npz", [2], xy, [])
    with pytest.raises(DecodesError, match="not all instances of the decodes"):
        PrecomputedDecodeSource(path, keys={0: {7: 5}}).decode(0, 0, {0: {7: tuple(xy[0, 0])}})
    with pytest.raises(DecodesError, match="not all instances of the decodes"):
        PrecomputedDecodeSource(path, keys={}).decode(0, 0, {0: {7: tuple(xy[0, 0])}})


def test_a_frame_without_tracked_instances_is_empty_with_keys(tmp_path):
    path = write_decodes(tmp_path / "k3.npz", [2], positions(1, 2), [])
    decodes, _ = PrecomputedDecodeSource(path, keys={}).decode(0, 0, {0: {}})
    assert decodes == {0: {}}


# ----------------------------------------------------------------------------- file checks
def test_a_missing_array_is_named(tmp_path):
    bad = tmp_path / "bad.npz"
    np.savez(bad, schema_version=np.array(SCHEMA_VERSION), frame_idx=np.arange(3))
    with pytest.raises(DecodesError, match=r"not a decodes file \(missing arrays \['frame_size'"):
        PrecomputedDecodeSource(bad)


def test_another_schema_version_is_refused(tmp_path):
    path = write_decodes(tmp_path / "v.npz", [1], positions(1, 1), [], schema_version=np.array(2))
    with pytest.raises(DecodesError, match="schema version 2"):
        PrecomputedDecodeSource(path)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"frame_idx": np.array([0, 2, 3], dtype=np.int32)}, "consecutive"),
        (
            {"frame_idx": np.array([], dtype=np.int32), "n_inst": np.array([], dtype=np.uint8)},
            "no frames",
        ),
        ({"n_inst": np.array([3, 3], dtype=np.uint8)}, "do not match frame_idx"),
        ({"tag_xy": np.zeros((2, 3, 2), dtype=np.float32)}, "do not match frame_idx"),
        ({"tag_xy": np.zeros((3, 3), dtype=np.float32)}, "do not match frame_idx"),
        ({"n_inst": np.array([4, 3, 3], dtype=np.uint8)}, "exceeds the 3 instance columns"),
        ({"vote_tag": np.array([1, 2], dtype=np.int16)}, "differ in length"),
        ({"vote_frame": np.array([5], dtype=np.int32)}, "does not exist"),
        ({"vote_inst": np.array([9], dtype=np.uint8)}, "does not exist"),
    ],
)
def test_inconsistent_files_are_refused(tmp_path, override, message):
    arrays = {"vote_frame": np.array([1]), "vote_inst": np.array([1]), "vote_tag": np.array([4])}
    path = write_decodes(tmp_path / "x.npz", [3, 3, 3], positions(), [], **{**arrays, **override})
    with pytest.raises(DecodesError, match=message):
        PrecomputedDecodeSource(path)


def test_a_different_video_name_only_warns(basic, caplog):
    path, _, _ = basic
    with caplog.at_level(logging.WARNING, logger="naps.precomputed"):
        PrecomputedDecodeSource(path, video="/data/other.mp4")
    assert "were computed from video clip.mp4, but the run uses other.mp4" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="naps.precomputed"):
        PrecomputedDecodeSource(path, video="/elsewhere/clip.mp4")
        PrecomputedDecodeSource(path)
    assert caplog.text == ""


def test_a_file_without_a_stored_video_name_never_warns(tmp_path, caplog):
    path = write_decodes(tmp_path / "nv.npz", [1], positions(1, 1), [])
    data = dict(np.load(path))
    del data["video"]
    np.savez(path, **data)
    with caplog.at_level(logging.WARNING, logger="naps.precomputed"):
        PrecomputedDecodeSource(path, video="whatever.mp4")
    assert caplog.text == ""


# ----------------------------------------------------------------------------- Matching
def test_matching_uses_the_decode_source_and_never_opens_the_video(tmp_path):
    n_inst = [2] * 12
    xy = positions(12, 2)
    votes = [(f, j, 5 + j) for f in range(12) for j in range(2)]
    path = write_decodes(tmp_path / "m.npz", n_inst, xy, votes)
    coords = {f: {0: tuple(xy[f, 0]), 1: tuple(xy[f, 1])} for f in range(12)}
    matcher = Matching(
        str(tmp_path / "does_not_exist.avi"), 0, 11, marker_detector=lambda _: [],
        aruco_crop_size=50, half_rolling_window_size=2, tag_node_dict=coords,
        decode_source=PrecomputedDecodeSource(path),
    )  # fmt: skip
    matched = matcher.match()
    assert matcher.frame_size == (200, 100)
    assert matched[6] == {0: 5, 1: 6}


def test_matching_without_a_decode_source_still_needs_the_video(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        Matching(
            str(tmp_path / "nothing.avi"), 0, 1, marker_detector=lambda _: [],
            aruco_crop_size=50, half_rolling_window_size=1, tag_node_dict={},
        )  # fmt: skip


# ----------------------------------------------------------------------------- naps-track
def decodes_of_colony(path, colony, n_frames=None, first_frame=0):
    """Truth decodes of the synthetic colony: instance k (bee k) shows tag ``tags[k]``."""
    n_frames = n_frames or colony.n_frames
    xy = np.zeros((n_frames, colony.n_bees, 2), dtype=np.float32)
    xy[:] = np.asarray(colony.positions, dtype=np.float32)[None]
    votes = [
        (first_frame + f, k, colony.tags[k]) for f in range(n_frames) for k in range(colony.n_bees)
    ]
    return write_decodes(path, [colony.n_bees] * n_frames, xy, votes, first_frame=first_frame)


def track_names(path):
    return {
        lf.frame_idx: [None if i.track is None else i.track.name for i in lf.instances]
        for lf in sio.load_slp(str(path))
    }


@pytest.fixture
def cli_args(tmp_path, colony, slp_factory):
    """Arguments of a flat naps-track run on the synthetic colony; the video file does not exist."""
    slp = slp_factory(tmp_path / "bees.slp", colony, tracked=False)
    decodes = decodes_of_colony(tmp_path / "bees.decodes.npz", colony)
    out = tmp_path / "out.slp"
    args = [
        "--slp-path", str(slp),
        "--video-path", str(tmp_path / "no_such_video.avi"),
        "--tag-node-name", "Tag",
        "--start-frame", "0",
        "--end-frame", str(colony.n_frames - 1),
        "--aruco-marker-set", "DICT_4X4_100",
        "--output-path", str(out),
        "--closed-set", *map(str, colony.tags),
        "--decodes", str(decodes),
    ]  # fmt: skip
    return args, out, decodes, colony


def test_decodes_option_is_parsed():
    required = [
        *["--slp-path", "a.slp", "--video-path", "a.avi", "--start-frame", "0", "--end-frame", "9"],
        *["--aruco-marker-set", "DICT_4X4_100"],
    ]
    assert naps_track.build_parser().parse_args(required).decodes is None
    args = naps_track.build_parser().parse_args([*required, "--decodes", "d.npz"])
    assert args.decodes == "d.npz"


def test_joint_engine_runs_from_decodes_without_the_video(cli_args):
    args, out, _, colony = cli_args
    assert naps_track.main([*args, "--identity-engine", "joint", "--colony-size", "3"]) == 0
    expected = [f"ArUcoTag#{t}" for t in colony.tags]
    assert all(row == expected for row in track_names(out).values())


def test_legacy_engine_runs_from_decodes_without_the_video(tmp_path, colony, slp_factory):
    slp = slp_factory(tmp_path / "bees.slp", colony, tracked=True)
    decodes = decodes_of_colony(tmp_path / "bees.decodes.npz", colony)
    out = tmp_path / "out.slp"
    assert naps_track.main(
        [
            *["--slp-path", str(slp), "--video-path", str(tmp_path / "gone.avi")],
            *["--tag-node-name", "Tag", "--start-frame", "0", "--end-frame", "19"],
            *["--aruco-marker-set", "DICT_4X4_100"],
            *["--output-path", str(out), "--decodes", str(decodes)],
        ]
    ) == 0  # fmt: skip
    expected = [f"ArUcoTag#{t}" for t in colony.tags]
    names = track_names(out)
    # the first frames are never assigned (rolling window), the rest carry the decoded tags
    assert all(row == expected for f, row in names.items() if f >= 5)


def test_legacy_engine_from_decodes_matches_the_video_run(
    tmp_path, colony, slp_factory, video_factory
):
    slp = slp_factory(tmp_path / "bees.slp", colony, tracked=True)
    video = video_factory(tmp_path / "bees.avi", colony)
    decodes = decodes_of_colony(tmp_path / "bees.decodes.npz", colony)
    common = [
        *["--slp-path", str(slp), "--video-path", str(video), "--tag-node-name", "Tag"],
        *["--start-frame", "0", "--end-frame", "19", "--aruco-marker-set", "DICT_4X4_100"],
        *["--aruco-crop-size", "60"],
    ]  # fmt: skip
    naps_track.main([*common, "--output-path", str(tmp_path / "video.slp")])
    naps_track.main(
        [*common, "--output-path", str(tmp_path / "decodes.slp"), "--decodes", str(decodes)]
    )
    assert track_names(tmp_path / "decodes.slp") == track_names(tmp_path / "video.slp")


def test_decodes_for_another_frame_range_exit_with_a_message(cli_args):
    args, out, _, colony = cli_args
    i = args.index("--end-frame")
    args[i + 1] = str(colony.n_frames + 4)
    with pytest.raises(SystemExit) as exc:
        naps_track.main([*args, "--identity-engine", "joint", "--colony-size", "3"])
    assert isinstance(exc.value.code, str)
    assert exc.value.code.startswith("--decodes: ")
    assert "covers frames 0..19 but the run needs 0..24" in exc.value.code
    assert not out.exists()


def test_decodes_for_another_slp_exit_with_a_message(cli_args, tmp_path, colony, slp_factory):
    args, out, _, _ = cli_args
    other = colony.__class__(tags=colony.tags[:2], n_frames=colony.n_frames)
    args[args.index("--slp-path") + 1] = str(slp_factory(tmp_path / "two.slp", other))
    with pytest.raises(SystemExit) as exc:
        naps_track.main([*args, "--identity-engine", "joint", "--colony-size", "3"])
    assert "the SLEAP file has 2 instances, the decodes 3" in str(exc.value.code)
    assert not out.exists()


def test_decodes_cannot_be_combined_with_decoder_fast(cli_args):
    args, out, _, _ = cli_args
    with pytest.raises(SystemExit) as exc:
        naps_track.main(
            [*args, "--identity-engine", "joint", "--colony-size", "3", "--decoder", "fast"]
        )
    assert "mutually exclusive with decoding from the video" in str(exc.value.code)
    assert not out.exists()


def test_a_missing_decodes_file_exits_with_a_message(cli_args, tmp_path):
    args, _, _, _ = cli_args
    args[args.index("--decodes") + 1] = str(tmp_path / "missing.npz")
    with pytest.raises(SystemExit) as exc:
        naps_track.main(args)
    assert "no such file" in str(exc.value.code)


def test_build_decode_source_selects_the_precomputed_source(cli_args):
    args, _, _, _ = cli_args
    parsed = naps_track.build_parser().parse_args(args)
    assert isinstance(naps_track.build_decode_source(parsed), PrecomputedDecodeSource)
