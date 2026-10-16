"""``naps-track`` on the real example (``tests/data``): golden output, closed set, report."""

import json
import re

import numpy as np
import pytest
import sleap_io as sio

from naps import naps_track
from naps.sleap_utils import load_tracks_from_slp
from tests.conftest import DATA_DIR

EXAMPLE_SLP = str(DATA_DIR / "example.slp")
EXAMPLE_VIDEO = str(DATA_DIR / "example.mp4")
GOLDEN = str(DATA_DIR / "example_naps_track_output.h5.slp")

# Golden-file tolerance (see test_naps_track_matches_golden_output).
#: Minimum fraction of instances whose tag equals the golden file's.
GOLDEN_MIN_AGREEMENT = 0.97
#: Maximum fraction of instances the golden file tags but the new run leaves untagged.
GOLDEN_MAX_LOST = 0.01
#: Minimum agreement between runs of different options wherever both assign a tag.
MIN_PAIRWISE_AGREEMENT = 0.95

CLOSED_SET = [25, 32, 1, 12]
BASE_ARGS = [
    "--slp-path", EXAMPLE_SLP,
    "--video-path", EXAMPLE_VIDEO,
    "--tag-node-name", "Tag",
    "--start-frame", "0",
    "--end-frame", "29",
    "--aruco-marker-set", "DICT_4X4_100",
    "--aruco-crop-size", "50",
]  # fmt: skip


def tag_id(track):
    """Tag id of an output track (``ArUcoTag#17`` or the legacy ``ArUcoTag#17.0``), or None."""
    return None if track is None else int(float(track.name.split("#")[1]))


def _tags_by_frame(path):
    """{frame: [tag or None per instance]} from a NAPS output file."""
    return {
        lf.frame_idx: [tag_id(inst.track) for inst in lf.instances]
        for lf in sio.load_slp(str(path))
    }


def _agreement(new_tags, reference_tags):
    """(instances tagged by both, of which with the same tag) between two {frame: tags}."""
    n_both = n_agree = 0
    for frame, frame_tags in new_tags.items():
        for new, ref in zip(frame_tags, reference_tags[frame], strict=True):
            if new is not None and ref is not None:
                n_both += 1
                n_agree += new == ref
    return n_both, n_agree


def _n_tagged(tags):
    return sum(t is not None for frame_tags in tags.values() for t in frame_tags)


@pytest.fixture(scope="module")
def plain_output(tmp_path_factory):
    """The example run once with default options (the slow part is shared by several tests)."""
    out = tmp_path_factory.mktemp("plain") / "plain.slp"
    naps_track.main([*BASE_ARGS, "--output-path", str(out)])
    return out


def test_naps_track_matches_golden_output(plain_output):
    """Compare the example run against the committed golden file.

    The golden file was produced with an older OpenCV/ArUco build, so identical tags are not
    expected: that build stored some tags as floats (``ArUcoTag#17.0``; compared here after
    normalisation) and decodes a few markers differently. Measured with OpenCV 5.0.0 on the
    30 example frames: 1380 of 1398 instances (98.7%) carry the golden tag; of the 1078
    instances both runs tag, 1063 (98.6%) agree; 3 instances are tagged only by the new run;
    none are tagged only by the golden run. The 15 mismatches are two short stretches on
    tracks 47 and 48 (a tag swap between two bees, 6-8 frames each) and one frame on track
    40; both builds have a comparable number of tag changes along SLEAP tracks (4 vs 3), so
    neither is clearly more plausible and the golden file is deliberately not regenerated.
    The thresholds ``GOLDEN_MIN_AGREEMENT`` / ``GOLDEN_MAX_LOST`` leave a margin below the
    measurement; geometry must be carried through exactly.
    """
    test_labels = sio.load_slp(str(plain_output))
    golden_labels = sio.load_slp(GOLDEN)

    assert [lf.frame_idx for lf in test_labels] == [lf.frame_idx for lf in golden_labels]
    n_total = n_agree = n_lost = 0
    for test_lf, golden_lf in zip(test_labels, golden_labels, strict=True):
        assert len(test_lf) == len(golden_lf)
        for test_inst, golden_inst in zip(test_lf.instances, golden_lf.instances, strict=True):
            # Geometry must be carried through unchanged
            assert np.array_equal(test_inst.numpy(), golden_inst.numpy(), equal_nan=True)
            n_total += 1
            n_agree += tag_id(test_inst.track) == tag_id(golden_inst.track)
            n_lost += tag_id(test_inst.track) is None and tag_id(golden_inst.track) is not None
    assert n_total > 0
    assert n_agree / n_total >= GOLDEN_MIN_AGREEMENT, (n_agree, n_total)
    assert n_lost / n_total <= GOLDEN_MAX_LOST, (n_lost, n_total)

    # The location matrix helper must still work on the output
    test_locations, test_node_names = load_tracks_from_slp(str(plain_output))
    golden_locations, golden_node_names = load_tracks_from_slp(GOLDEN)
    assert test_locations.shape[:3] == golden_locations.shape[:3]
    assert test_node_names == golden_node_names


def test_naps_track_h5_path_is_deprecated_and_never_read(plain_output, tmp_path, caplog):
    """``--h5-path`` is accepted (with a DeprecationWarning and a log line) but has no effect."""
    out = tmp_path / "with_h5.slp"
    bogus = str(tmp_path / "does_not_exist.h5")
    with pytest.warns(DeprecationWarning, match="--h5-path is deprecated"):
        naps_track.main([*BASE_ARGS, "--h5-path", bogus, "--output-path", str(out)])
    assert any("--h5-path is deprecated" in r.getMessage() for r in caplog.records)
    assert _tags_by_frame(out) == _tags_by_frame(plain_output)


def _write_idtracker_trajectories(path, perm, offset):
    """Synthetic idtracker.ai v6 trajectories of the closed-set bees, from the golden file.

    The animals are labelled in permuted order (animal ``a`` follows ``CLOSED_SET[perm[a]]``) and
    shifted by ``offset`` (a crop origin to undo).
    """
    traj = np.full((30, 4, 2), np.nan)
    prob = np.full((30, 4), np.nan)
    for lf in sio.load_slp(GOLDEN):
        for inst in lf.instances:
            tag = tag_id(inst.track)
            if tag in CLOSED_SET:
                a = perm.index(CLOSED_SET.index(tag))
                traj[lf.frame_idx, a] = inst.numpy()[0] - offset
                prob[lf.frame_idx, a] = 0.95
    np.save(path, {"trajectories": traj, "id_probabilities": prob})


def test_naps_track_closed_set_and_idtracker_evidence(plain_output, tmp_path, caplog):
    perm = [2, 0, 3, 1]  # idtracker animal a follows CLOSED_SET[perm[a]]
    trajectories = tmp_path / "trajectories.npy"
    _write_idtracker_trajectories(trajectories, perm, np.array([100.0, 50.0]))

    closed, fused = tmp_path / "closed.slp", tmp_path / "fused.slp"
    closed_set_args = ["--closed-set", *map(str, CLOSED_SET)]
    naps_track.main([*BASE_ARGS, *closed_set_args, "--output-path", str(closed)])
    caplog.set_level("INFO")
    fused_args = [
        *closed_set_args,
        *["--idtracker-trajectories", str(trajectories)],
        *["--idtracker-offset", "100", "50", "--idtracker-min-support", "10"],
        *["--output-path", str(fused)],
    ]
    naps_track.main([*BASE_ARGS, *fused_args])

    # The example file holds ~46 bees, far more than the 4-tag closed set: the closed set
    # warns, and elimination can only abstain (it needs the input to hold just the N animals)
    messages = [r.getMessage() for r in caplog.records]
    assert any("median of" in m for m in messages)
    assert any(
        "Closed-set elimination assigned 0" in m and "not_exactly_one" in m for m in messages
    )

    # The synthetic idtracker labels were aligned to the right tags (not silently refused)
    accepted = [m for m in messages if "accepted" in m]
    assert len(accepted) == 4
    for a in range(4):
        assert any(f"animal {a} -> tag {CLOSED_SET[perm[a]]} " in m for m in accepted), accepted

    plain_tags = _tags_by_frame(plain_output)
    closed_tags, fused_tags = _tags_by_frame(closed), _tags_by_frame(fused)

    # With a closed set, only closed-set tags (or unassigned) can appear, and none twice per frame
    for tags in (closed_tags, fused_tags):
        for frame_tags in tags.values():
            assigned = [t for t in frame_tags if t is not None]
            assert set(assigned) <= set(CLOSED_SET)
            assert len(assigned) == len(set(assigned))

    # Closed-set assignments agree with the legacy output wherever both assign
    for tags in (closed_tags, fused_tags):
        n_both, n_agree = _agreement(tags, plain_tags)
        assert n_both > 0
        assert n_agree / n_both >= MIN_PAIRWISE_AGREEMENT

    # Idtracker evidence is a pure addition: it never reduces what the closed set alone assigns
    assert _n_tagged(closed_tags) > 0
    assert _n_tagged(fused_tags) >= _n_tagged(closed_tags)

    # ... and the extra assignments agree with the golden labels
    n_both, n_agree = _agreement(fused_tags, _tags_by_frame(GOLDEN))
    assert n_agree / n_both >= MIN_PAIRWISE_AGREEMENT


def test_naps_track_idtracker_requires_closed_set(tmp_path):
    argv = [
        *BASE_ARGS,
        *["--idtracker-trajectories", str(tmp_path / "missing.npy")],
        *["--output-path", str(tmp_path / "x.slp")],
    ]
    with pytest.raises(SystemExit):
        naps_track.main(argv)


def test_naps_track_marker_subset(plain_output, tmp_path):
    subset = tmp_path / "subset.slp"
    naps_track.main(
        [*BASE_ARGS, "--aruco-marker-subset", *map(str, CLOSED_SET), "--output-path", str(subset)]
    )
    plain_tags, subset_tags = _tags_by_frame(plain_output), _tags_by_frame(subset)
    assigned = {t for frame_tags in subset_tags.values() for t in frame_tags if t is not None}
    assert assigned
    assert assigned <= set(CLOSED_SET)
    n_both, n_agree = _agreement(subset_tags, plain_tags)
    assert n_both > 0
    assert n_agree / n_both >= MIN_PAIRWISE_AGREEMENT


def test_naps_track_output_names_are_always_integer_tags(plain_output):
    names = {
        i.track.name for lf in sio.load_slp(str(plain_output)) for i in lf.instances if i.track
    }
    assert names
    assert all(re.fullmatch(r"ArUcoTag#\d+", n) for n in names), sorted(names)


def test_naps_track_quadrant_infer_and_report_on_the_real_example(plain_output, tmp_path):
    quad = tmp_path / "quad.slp"
    report = tmp_path / "report.json"
    infer_args = ["--closed-set-infer", "--closed-set-min-support", "10", "--closed-set-size", "4"]
    naps_track.main(
        [*BASE_ARGS, *infer_args, "--report-json", str(report), "--output-path", str(quad)]
    )
    rep = json.loads(report.read_text())
    # the example is one 3672 px arena with ~46 bees, not 4 colonies of 4: inference must not
    # pretend otherwise (it abstains, or only uses regions that really have 4 consistent tags)
    assert rep["mode"] == "quadrants"
    assert rep["frame_size"] == [3672, 3672]
    assert set(rep["regions"]) == {"top_left", "top_right", "bottom_left", "bottom_right"}
    inf = rep["inference"]
    assert set(inf["inferred"]) | set(inf["abstained"]) == set(rep["regions"])
    assert rep["n_instances"] == sum(len(lf.instances) for lf in sio.load_slp(str(plain_output)))
    # whatever it does, it never contradicts the legacy tags
    n_both, n_agree = _agreement(_tags_by_frame(quad), _tags_by_frame(plain_output))
    assert n_both == 0 or n_agree / n_both >= MIN_PAIRWISE_AGREEMENT
