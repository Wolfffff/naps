"""``apply_instance_tags``: writing per-instance tags (and scores) into a SLEAP file."""

import re

import numpy as np
import sleap_io as sio

from naps.sleap_utils import apply_instance_tags

TRACK_NAME = re.compile(r"^ArUcoTag#\d+$")


def make_labels(n_frames=6, n_instances=3, tracked=True):
    skeleton = sio.Skeleton(["Tag", "Thorax"])
    video = sio.Video(filename="synthetic.mp4", open_backend=False)
    tracks = [sio.Track(f"track_{k}") for k in range(n_instances)]
    frames = []
    for f in range(n_frames):
        insts = [
            sio.PredictedInstance.from_numpy(
                np.array([[10.0 * k + f, 5.0], [10.0 * k + f, 9.0]]),
                skeleton=skeleton,
                score=0.5 + 0.1 * k,
                track=tracks[k] if tracked else None,
            )
            for k in range(n_instances)
        ]
        frames.append(sio.LabeledFrame(video=video, frame_idx=f + 10, instances=insts))
    return sio.Labels(frames)


def table_arrays(n_frames=6, n_slots=3):
    """inst_index, slot_tag, slot_score with slot s holding instance s, tag 20 + s."""
    inst_index = np.tile(np.arange(n_slots), (n_frames, 1)).astype(np.int64)
    slot_tag = np.tile(20 + np.arange(n_slots), (n_frames, 1)).astype(np.int64)
    slot_score = np.full((n_frames, n_slots), 0.995, dtype=np.float32)
    return inst_index, slot_tag, slot_score


def round_trip(labels, tmp_path):
    path = tmp_path / "out.slp"
    sio.save_slp(labels, str(path))
    return sio.load_slp(str(path))


def test_tracks_are_named_with_integer_tags_only(tmp_path):
    labels = make_labels()
    inst_index, slot_tag, slot_score = table_arrays()
    out = apply_instance_tags(labels, 10, inst_index, slot_tag, slot_score)
    back = round_trip(out, tmp_path)
    assert {t.name for t in back.tracks} == {"ArUcoTag#20", "ArUcoTag#21", "ArUcoTag#22"}
    for lf in back.labeled_frames:
        assert [i.track.name for i in lf.instances] == [
            "ArUcoTag#20",
            "ArUcoTag#21",
            "ArUcoTag#22",
        ]
        assert all(TRACK_NAME.match(i.track.name) for i in lf.instances)


def test_scores_round_trip_and_identities_default_to_the_track_name(tmp_path):
    labels = make_labels()
    inst_index, slot_tag, slot_score = table_arrays()
    slot_score[0, 1] = 0.9999
    back = round_trip(apply_instance_tags(labels, 10, inst_index, slot_tag, slot_score), tmp_path)
    inst = back.labeled_frames[0].instances[1]
    assert inst.identity_score == np.float32(0.9999)
    assert inst.tracking_score == np.float32(0.9999)
    assert inst.identity is not None
    assert inst.identity.name == "ArUcoTag#21"


def test_identity_names_can_be_session_qualified(tmp_path):
    labels = make_labels()
    inst_index, slot_tag, slot_score = table_arrays()
    names = {20: "s1:ArUcoTag#20", 21: "s1:ArUcoTag#21"}
    back = round_trip(
        apply_instance_tags(labels, 10, inst_index, slot_tag, slot_score, identity_names=names),
        tmp_path,
    )
    inst = back.labeled_frames[0].instances
    assert inst[0].identity.name == "s1:ArUcoTag#20"
    assert inst[1].identity.name == "s1:ArUcoTag#21"
    assert inst[2].identity.name == "ArUcoTag#22"  # no entry: the track name
    assert inst[0].track.name == "ArUcoTag#20"  # tracks stay unqualified


def test_untracked_instances_have_no_track_and_no_identity(tmp_path):
    labels = make_labels(tracked=True)
    inst_index, slot_tag, slot_score = table_arrays()
    slot_tag[2, 1] = -1
    slot_tag[:, 2] = -1
    back = round_trip(apply_instance_tags(labels, 10, inst_index, slot_tag, slot_score), tmp_path)
    frame2 = back.labeled_frames[2].instances
    assert frame2[1].track is None
    assert frame2[1].identity is None
    assert frame2[1].identity_score is None
    assert all(
        lf.instances[2].track is None for lf in back.labeled_frames
    )  # the SLEAP track is gone
    assert back.labeled_frames[0].instances[0].track.name == "ArUcoTag#20"


def test_frames_outside_the_range_are_dropped_and_last_frame_limits(tmp_path):
    labels = make_labels(n_frames=6)
    inst_index, slot_tag, slot_score = table_arrays(n_frames=4)  # rows = frames 12..15
    out = apply_instance_tags(labels, 12, inst_index, slot_tag, slot_score)
    assert [lf.frame_idx for lf in out.labeled_frames] == [12, 13, 14, 15]
    limited = apply_instance_tags(labels, 12, inst_index, slot_tag, slot_score, last_frame=13)
    assert [lf.frame_idx for lf in limited.labeled_frames] == [12, 13]
    assert round_trip(limited, tmp_path).labeled_frames[0].frame_idx == 12
    # a last_frame beyond the arrays is clipped to them (no labels exist past the last row)
    beyond = apply_instance_tags(labels, 12, inst_index, slot_tag, slot_score, last_frame=500)
    assert [lf.frame_idx for lf in beyond.labeled_frames] == [12, 13, 14, 15]


def test_instance_order_and_points_are_preserved(tmp_path):
    labels = make_labels()
    inst_index, slot_tag, slot_score = table_arrays()
    out = apply_instance_tags(labels, 10, inst_index, slot_tag, slot_score)
    for before, after in zip(labels.labeled_frames, out.labeled_frames, strict=True):
        for a, b in zip(before.instances, after.instances, strict=True):
            np.testing.assert_array_equal(a.numpy(), b.numpy())
            assert a.score == b.score


def test_instance_index_mapping_with_gaps(tmp_path):
    # slots hold the instances in a different order and with an empty slot in between
    labels = make_labels(n_frames=2, n_instances=3)
    inst_index = np.array([[2, -1, 0, 1], [2, -1, 0, 1]], dtype=np.int64)
    slot_tag = np.array([[30, -1, 31, -1], [30, -1, 31, 32]], dtype=np.int64)
    slot_score = np.full((2, 4), 0.99, dtype=np.float32)
    out = apply_instance_tags(labels, 10, inst_index, slot_tag, slot_score)
    names = [[None if i.track is None else i.track.name for i in lf.instances] for lf in out]
    # instance 0 is in slot 2 (tag 31), instance 1 in slot 3 (tag -1 / 32), instance 2 in slot 0
    assert names[0] == ["ArUcoTag#31", None, "ArUcoTag#30"]
    assert names[1] == ["ArUcoTag#31", "ArUcoTag#32", "ArUcoTag#30"]


def test_an_instance_without_a_slot_is_left_untracked():
    labels = make_labels(n_frames=1, n_instances=3)
    inst_index = np.array([[0, 1, -1]], dtype=np.int64)  # instance 2 has no slot
    slot_tag = np.array([[40, 41, 42]], dtype=np.int64)
    out = apply_instance_tags(labels, 10, inst_index, slot_tag, np.ones((1, 3), np.float32))
    assert [i.track.name if i.track else None for i in out.labeled_frames[0].instances] == [
        "ArUcoTag#40",
        "ArUcoTag#41",
        None,
    ]


def test_all_untracked_writes_no_tracks():
    labels = make_labels(tracked=False)
    inst_index, slot_tag, slot_score = table_arrays()
    out = apply_instance_tags(labels, 10, inst_index, np.full_like(slot_tag, -1), slot_score)
    assert out.tracks == []
    assert all(i.track is None for lf in out for i in lf.instances)
