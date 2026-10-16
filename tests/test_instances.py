"""``InstanceTable`` and ``table_from_labels``."""

import numpy as np
import pytest
import sleap_io as sio

from naps.instances import InstanceTable, table_from_labels

NODES = ["Tag", "Thorax", "head", "abdomen"]


def make_table(n_frames=4, n_slots=3, with_pose=True, first_frame=10):
    """A table whose every entry is distinguishable (xy encodes frame and slot)."""
    frame, slot = np.meshgrid(np.arange(n_frames), np.arange(n_slots), indexing="ij")
    xy = np.stack([frame * 100.0, slot * 10.0], axis=-1)
    return InstanceTable(
        first_frame=first_frame,
        xy=xy,
        tag_xy=xy + 1.0,
        score=frame + slot / 10.0,
        hint=slot.astype(np.int64),
        inst_index=slot.astype(np.int64),
        heading=slot * 0.5,
        pose=np.broadcast_to(xy[:, :, None, :], (n_frames, n_slots, 2, 2)).astype(np.float32)
        if with_pose
        else None,
    )


def test_basic_properties():
    table = make_table()
    assert (table.n_frames, table.n_slots) == (4, 3)
    assert table.valid.all()
    assert list(table.frames) == [10, 11, 12, 13]
    table.xy[1, 2] = np.nan
    assert not table.valid[1, 2]
    assert table.valid.sum() == 11


def test_valid_needs_both_coordinates():
    table = make_table()
    table.xy[0, 0, 1] = np.nan
    assert not table.valid[0, 0]


@pytest.mark.parametrize(
    "field", ["tag_xy", "score", "hint", "inst_index", "heading", "pose", "xy_rank", "xy_last"]
)
def test_shape_mismatch_raises(field):
    table = make_table()
    kwargs = {
        "first_frame": 0,
        "xy": table.xy,
        "tag_xy": table.tag_xy,
        "score": table.score,
        "hint": table.hint,
        "inst_index": table.inst_index,
        "heading": table.heading,
        "pose": table.pose,
    }
    if field == "xy_rank":
        kwargs["xy"] = table.xy[..., 0]
    elif field == "xy_last":
        kwargs["xy"] = np.zeros((4, 3, 3))
    elif field == "pose":
        kwargs["pose"] = table.pose[:2]
    else:
        kwargs[field] = kwargs[field][:2]
    with pytest.raises(ValueError, match="disagree in shape"):
        InstanceTable(**kwargs)


def test_pose_is_optional():
    table = make_table(with_pose=False)
    assert table.pose is None
    packed, _ = table.subset(np.ones((4, 3), dtype=bool))
    assert packed.pose is None


def test_subset_left_packs_in_original_order():
    table = make_table()
    member = np.zeros((4, 3), dtype=bool)
    member[0, [2]] = True
    member[1, [0, 2]] = True
    member[2, [1]] = True
    packed, source = table.subset(member)
    assert packed.n_slots == 2
    assert packed.first_frame == 10
    assert source.tolist() == [[2, -1], [0, 2], [1, -1], [-1, -1]]
    # the kept instances carry their own values, padding is blanked
    assert packed.xy[1, 0].tolist() == table.xy[1, 0].tolist()
    assert packed.xy[1, 1].tolist() == table.xy[1, 2].tolist()
    assert np.isnan(packed.xy[0, 1]).all()
    assert np.isnan(packed.tag_xy[0, 1]).all()
    assert np.isnan(packed.score[0, 1])
    assert packed.hint.tolist() == [[2, -1], [0, 2], [1, -1], [-1, -1]]
    assert packed.inst_index.tolist() == packed.hint.tolist()
    assert np.isnan(packed.heading[0, 1])
    assert packed.heading[1, 1] == table.heading[1, 2]
    assert packed.pose is not None
    assert packed.pose.dtype == np.float32
    assert packed.pose[1, 1, 0].tolist() == table.pose[1, 2, 0].tolist()
    assert np.isnan(packed.pose[3]).all()


def test_subset_ignores_invalid_instances_and_empty_member():
    table = make_table()
    table.xy[0, 1] = np.nan
    member = np.ones((4, 3), dtype=bool)
    packed, source = table.subset(member)
    assert packed.n_slots == 3
    assert source[0].tolist() == [0, 2, -1]
    empty, empty_source = table.subset(np.zeros((4, 3), dtype=bool))
    assert empty.n_slots == 1  # never zero slots
    assert (empty_source == -1).all()
    assert not empty.valid.any()
    assert empty.pose is not None
    assert empty.pose.shape[:2] == (4, 1)


def test_subset_keeps_dtypes():
    packed, _ = make_table().subset(np.ones((4, 3), dtype=bool))
    assert packed.hint.dtype == np.int64
    assert packed.inst_index.dtype == np.int64
    assert packed.xy.dtype == np.float64


# ------------------------------------------------------------------- table_from_labels
def skeleton(nodes=NODES):
    return sio.Skeleton(nodes)


def predicted(points, skel, track=None, score=0.9):
    return sio.PredictedInstance.from_numpy(
        np.asarray(points, dtype=float), skeleton=skel, score=score, track=track
    )


def labels_of(frames, skel, video=None):
    """``frames`` maps frame index -> list of instances."""
    video = video or sio.Video(filename="x.mp4", open_backend=False)
    lfs = [
        sio.LabeledFrame(video=video, frame_idx=idx, instances=insts)
        for idx, insts in frames.items()
    ]
    return sio.Labels(labeled_frames=lfs, videos=[video], skeletons=[skel])


BEE = [[10.0, 10.0], [12.0, 14.0], [20.0, 10.0], [4.0, 18.0]]  # Tag, Thorax, head, abdomen


def test_table_from_labels_basic():
    skel = skeleton()
    t3, t7 = sio.Track("track_3"), sio.Track("track_7")
    labels = labels_of(
        {
            0: [predicted(BEE, skel, t3, 0.8), predicted(np.array(BEE) + 100, skel, t7, 0.6)],
            2: [predicted(BEE, skel, None, 0.5)],
        },
        skel,
    )
    table = table_from_labels(labels, 0, 3, "Tag")
    assert (table.n_frames, table.n_slots) == (4, 2)
    assert table.first_frame == 0
    assert table.xy[0, 0].tolist() == [12.0, 14.0]  # Thorax by default
    assert table.tag_xy[0, 0].tolist() == [10.0, 10.0]
    assert table.hint[0].tolist() == [3, 7]
    assert table.hint[2].tolist() == [-1, -1]
    assert table.inst_index[0].tolist() == [0, 1]
    assert table.inst_index[1].tolist() == [-1, -1]  # frame without labels
    assert table.score[0, 0] == pytest.approx(0.8)
    assert table.heading[0, 0] == pytest.approx(np.arctan2(10 - 18, 20 - 4))
    assert table.pose is not None
    assert table.pose.shape == (4, 2, 4, 2)
    assert table.valid.sum() == 3
    assert not table.valid[1].any()


def test_table_from_labels_respects_the_frame_range():
    skel = skeleton()
    labels = labels_of({0: [predicted(BEE, skel)], 5: [predicted(BEE, skel)]}, skel)
    table = table_from_labels(labels, 4, 6, "Tag")
    assert table.first_frame == 4
    assert table.n_frames == 3
    assert table.valid.tolist() == [[False], [True], [False]]


def test_table_from_labels_empty_labels():
    table = table_from_labels(sio.Labels(), 0, 4, "Tag")
    assert (table.n_frames, table.n_slots) == (5, 1)
    assert not table.valid.any()
    assert table.pose is not None
    assert table.pose.shape == (5, 1, 0, 2)


def test_track_names_without_a_number_use_the_track_index():
    skel = skeleton()
    alpha, beta = sio.Track("alpha"), sio.Track("beta")
    labels = labels_of({0: [predicted(BEE, skel, alpha), predicted(BEE, skel, beta)]}, skel)
    table = table_from_labels(labels, 0, 0, "Tag")
    assert table.hint[0].tolist() == [labels.tracks.index(alpha), labels.tracks.index(beta)]


def test_track_unknown_to_the_labels_gets_minus_one():
    skel = skeleton()
    ghost = sio.Track("ghost")
    labels = labels_of({0: [predicted(BEE, skel, ghost)]}, skel)
    labels.tracks.clear()
    table = table_from_labels(labels, 0, 0, "Tag")
    assert table.hint[0, 0] == -1


def test_position_node_is_case_insensitive():
    skel = skeleton()
    labels = labels_of({0: [predicted(BEE, skel)]}, skel)
    table = table_from_labels(labels, 0, 0, "tag", position_node="THORAX")
    assert table.xy[0, 0].tolist() == [12.0, 14.0]
    assert table.tag_xy[0, 0].tolist() == [10.0, 10.0]


def test_position_falls_back_to_tag_node_then_to_the_mean():
    skel = skeleton()
    no_thorax = np.array(BEE)
    no_thorax[1] = np.nan
    no_thorax_no_tag = no_thorax.copy()
    no_thorax_no_tag[0] = np.nan
    labels = labels_of(
        {
            0: [
                predicted(no_thorax, skel),
                predicted(no_thorax_no_tag, skel),
            ]
        },
        skel,
    )
    table = table_from_labels(labels, 0, 0, "Tag")
    assert table.xy[0, 0].tolist() == [10.0, 10.0]  # tag node
    assert table.xy[0, 1].tolist() == pytest.approx([12.0, 14.0])  # mean of head and abdomen
    # a skeleton without the position node falls back as well
    other = sio.Skeleton(["Tag", "head"])
    labels2 = labels_of({0: [predicted([[1.0, 2.0], [5.0, 6.0]], other)]}, other)
    table2 = table_from_labels(labels2, 0, 0, "Tag", position_node="Thorax")
    assert table2.xy[0, 0].tolist() == [1.0, 2.0]
    assert np.isnan(table2.heading[0, 0])  # no abdomen node: no heading


def test_position_node_none_uses_the_tag_node():
    skel = skeleton()
    labels = labels_of({0: [predicted(BEE, skel)]}, skel)
    table = table_from_labels(labels, 0, 0, "Tag", position_node=None)
    assert table.xy[0, 0].tolist() == [10.0, 10.0]


def test_instance_without_any_finite_node_is_an_invalid_slot():
    skel = skeleton()
    nothing = np.full((4, 2), np.nan)
    labels = labels_of({0: [predicted(nothing, skel), predicted(BEE, skel)]}, skel)
    table = table_from_labels(labels, 0, 0, "Tag")
    assert table.valid[0].tolist() == [False, True]
    assert table.inst_index[0].tolist() == [0, 1]  # the slot still maps back to the instance
    assert np.isnan(table.tag_xy[0, 0]).all()
    assert np.isnan(table.heading[0, 0])


def test_heading_is_nan_when_a_node_is_missing():
    skel = skeleton()
    broken = np.array(BEE)
    broken[2] = np.nan  # head not visible
    labels = labels_of({0: [predicted(broken, skel)]}, skel)
    assert np.isnan(table_from_labels(labels, 0, 0, "Tag").heading[0, 0])


def test_heading_nodes_can_be_switched_off():
    skel = skeleton()
    labels = labels_of({0: [predicted(BEE, skel)]}, skel)
    table = table_from_labels(labels, 0, 0, "Tag", head_node=None, tail_node=None)
    assert np.isnan(table.heading[0, 0])


def test_keep_pose_false():
    skel = skeleton()
    labels = labels_of({0: [predicted(BEE, skel)]}, skel)
    assert table_from_labels(labels, 0, 0, "Tag", keep_pose=False).pose is None


def test_user_instance_without_score_gives_nan():
    skel = skeleton()
    inst = sio.Instance.from_numpy(np.array(BEE), skeleton=skel)
    labels = labels_of({0: [inst]}, skel)
    assert np.isnan(table_from_labels(labels, 0, 0, "Tag").score[0, 0])


def test_unknown_tag_node_raises():
    skel = skeleton()
    labels = labels_of({0: [predicted(BEE, skel)]}, skel)
    with pytest.raises(ValueError, match="tag node 'nope'"):
        table_from_labels(labels, 0, 0, "nope")


def test_unknown_tag_node_without_instances_is_not_checked():
    # nothing to look at: the empty table is returned
    assert table_from_labels(sio.Labels(), 0, 1, "nope").n_frames == 2
