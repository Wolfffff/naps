"""Window statuses, self-validation, the untrusted-window policy and the joint report."""

import json
from typing import Any

import numpy as np
import pytest

from naps import diagnostics as dg
from naps.consistency import ConsistencyParams
from naps.diagnostics import (
    DiagnosticsParams,
    WindowStatus,
    apply_policy,
    assess_result,
    assess_windows,
    build_report,
    completeness,
    self_validate,
    wilson,
    window_edges,
)
from naps.identity import RegionLabels, SlotFlag
from naps.joint import FLAT_REGION
from naps.naps_track import _json_default
from naps.sources import Source
from naps.tracklets import Tracklets
from tests.scenes import make_scene
from tests.test_joint import TAGS, flat_run


def sv(*args: Any, **kwargs: Any) -> Any:
    """``self_validate`` with a loose return type (the result is a nested JSON-like dict)."""
    return self_validate(*args, **kwargs)


def br(*args: Any, **kwargs: Any) -> Any:
    """``build_report`` with a loose return type."""
    return build_report(*args, **kwargs)


def scene_result(seed=1, closed_set=None, **kw):
    kw.setdefault("n_frames", 400)
    scene = make_scene(seed, **kw)
    result, _ = flat_run(scene, closed_set=closed_set)
    return scene, result


def tiny_labels(n_frames=20, n_slots=1, n_tags=2):
    """A hand-built RegionLabels with every slot unlabelled."""
    shape = (n_frames, n_slots)
    return RegionLabels(
        slot_tag=np.full(shape, -1, dtype=np.int16),
        slot_conf=np.full(shape, np.nan, dtype=np.float32),
        slot_source=np.zeros(shape, dtype=np.int8),
        slot_disputed=np.zeros(shape, dtype=bool),
        slot_flag=np.zeros(shape, dtype=np.int8),
        guess_tag=np.full(shape, -1, dtype=np.int16),
        guess_conf=np.zeros(shape, dtype=np.float32),
        tag_xy=np.full((n_frames, n_tags, 2), np.nan),
        tag_slot=np.full((n_frames, n_tags), -1, dtype=np.int64),
        tag_src=np.zeros((n_frames, n_tags), dtype=np.int8),
        tag_conf=np.full((n_frames, n_tags), np.nan, dtype=np.float32),
    )


# ------------------------------------------------------------------------ small helpers
def test_wilson_interval():
    assert wilson(0, 0) == [None, None]
    assert wilson(50, 100) == [0.4038, 0.5962]
    low, high = np.array(wilson(100, 100), dtype=float)
    assert high == 1.0
    assert 0.96 < low < 1.0
    low, high = np.array(wilson(0, 100), dtype=float)
    assert low == 0.0
    assert 0.0 < high < 0.04


def test_completeness_histogram_and_fractions():
    xy = np.full((4, 3, 2), np.nan)
    xy[0] = 1.0  # 3 tags
    xy[1, :2] = 1.0  # 2 tags
    xy[2, 0] = 1.0  # 1 tag
    out = completeness(xy)
    assert out["frames_with_n_tags"] == {"3": 1, "2": 1, "1": 1, "0": 1}
    assert out["bee_frame_completeness"] == round(6 / 12, 4)
    assert out["per_tag"] == [0.75, 0.5, 0.25]
    empty = completeness(np.zeros((0, 3, 2)))
    assert empty["bee_frame_completeness"] is None
    assert empty["per_tag"] == []


def test_window_edges():
    assert window_edges(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert window_edges(8, 4) == [(0, 4), (4, 8)]
    assert window_edges(3, 6000) == [(0, 3)]
    assert window_edges(0, 10) == []


def test_step_p99_needs_consecutive_masked_frames():
    xy = np.zeros((6, 1, 2))
    xy[:, 0, 0] = [0, 1, 2, 3, 100, 101]
    both = np.ones((6, 1), dtype=bool)
    assert dg._step_p99(xy, both) == pytest.approx(np.percentile([1, 1, 1, 97, 1], 99))
    none = np.zeros((6, 1), dtype=bool)
    assert dg._step_p99(xy, none) is None
    gappy = both.copy()
    gappy[4] = False
    assert dg._step_p99(xy, gappy) == pytest.approx(1.0)


def test_tracklet_median_length_edge_cases():
    labels = tiny_labels(20)
    assert dg._tracklet_median_len(labels, 0, 20) is None  # no tracklets at all
    labels.tracklets = Tracklets(np.full((20, 1), -1, dtype=np.int32))
    assert dg._tracklet_median_len(labels, 0, 20) is None  # n_tracklets == 0
    tid = np.full((20, 1), -1, dtype=np.int32)
    tid[:10, 0] = 0
    labels.tracklets = Tracklets(tid)
    assert dg._tracklet_median_len(labels, 0, 20) == 10.0
    assert dg._tracklet_median_len(labels, 10, 20) is None  # none present in the window


def test_impure_fraction_edge_cases():
    labels = tiny_labels(20)
    assert dg._impure_fraction(labels, 0, 20, 2) is None  # no tracklets / candidates
    tid = np.zeros((20, 1), dtype=np.int32)
    labels.tracklets = Tracklets(tid)
    labels.candidates = np.full((20, 1), -1, dtype=np.int16)
    assert dg._impure_fraction(labels, 0, 20, 2) is None  # no anchors
    labels.candidates[:3, 0] = 0
    assert dg._impure_fraction(labels, 0, 20, 2) is None  # fewer than 5 anchors
    labels.candidates[:, 0] = 0
    assert dg._impure_fraction(labels, 0, 20, 2) == 0.0
    labels.candidates[::2, 0] = 1
    assert dg._impure_fraction(labels, 0, 20, 2) == 1.0
    labels.tracklets = Tracklets(np.full((20, 1), -1, dtype=np.int32))
    assert dg._impure_fraction(labels, 0, 20, 2) is None


def test_slot_nn_is_none_without_entities():
    _, result = scene_result(2, n_frames=60)
    labels = result.regions[FLAT_REGION].labels
    nn = dg._slot_nn(labels)
    assert nn is not None
    assert nn.shape == labels.slot_tag.shape
    assert (nn[labels.valid] > 0).all()
    labels.entities = None
    assert dg._slot_nn(labels) is None


# --------------------------------------------------------------------------- statuses
def test_a_clean_window_is_unvalidated_without_self_validation():
    _, result = scene_result(3)
    run = result.regions[FLAT_REGION]
    (win,) = assess_windows(run, DiagnosticsParams())
    assert (win.start, win.stop) == (0, 400)
    assert win.status == "unvalidated"
    assert win.reasons == []
    assert set(win.checks) == {
        "median_tracklet_len_frame_weighted",
        "anchor_step_p99_px",
        "impure_tracklet_frac",
        "temporal_candidate_drop_fraction",
    }
    assert all(not c["fail"] for c in win.checks.values())
    wins = assess_windows(run, DiagnosticsParams(window_frames=150))
    assert [(w.start, w.stop) for w in wins] == [(0, 150), (150, 300), (300, 400)]


def test_unsolved_regions_have_no_windows():
    _, result = scene_result(3, closed_set=TAGS[:2], n_frames=30)
    assert assess_windows(result.regions[FLAT_REGION], DiagnosticsParams()) == []


@pytest.mark.parametrize(
    ("params", "check"),
    [
        (DiagnosticsParams(min_median_tracklet_len=1e9), "median_tracklet_len_frame_weighted"),
        (DiagnosticsParams(max_anchor_step_p99=0.001), "anchor_step_p99_px"),
    ],
)
def test_failed_data_checks_make_a_window_uncertain(params, check):
    _, result = scene_result(4)
    (win,) = assess_windows(result.regions[FLAT_REGION], params)
    assert win.status == "uncertain"
    assert win.checks[check]["fail"]
    assert win.reasons[0].startswith(check)


def test_informational_flags_never_fail_a_window():
    _, result = scene_result(5)
    run = result.regions[FLAT_REGION]
    run.labels.candidates = np.where(np.arange(400)[:, None] % 2 == 0, 0, 1) * np.ones(
        run.labels.candidates.shape, dtype=np.int16
    )
    (win,) = assess_windows(run, DiagnosticsParams(max_impure_tracklet_frac=0.0))
    assert win.checks["impure_tracklet_frac"]["fail"]
    assert win.status == "unvalidated"


def test_only_leave1_style_failures_in_a_well_decoded_window_need_no_repair():
    strict = {"max_temporal_drop_frac": -1.0, "min_temporal_candidates": 0}
    _, well = scene_result(6, decode_rate=0.97)
    (win,) = assess_windows(well.regions[FLAT_REGION], DiagnosticsParams(**strict))
    assert win.status == "no_repair_needed"
    assert win.checks["temporal_candidate_drop_fraction"]["fail"]
    _, poor = scene_result(6, decode_rate=0.5)
    (win,) = assess_windows(poor.regions[FLAT_REGION], DiagnosticsParams(**strict))
    assert win.status == "uncertain"


def test_a_few_candidates_are_not_judged():
    _, result = scene_result(7)
    params = DiagnosticsParams(max_temporal_drop_frac=-1.0, min_temporal_candidates=10**6)
    (win,) = assess_windows(result.regions[FLAT_REGION], params)
    assert not win.checks["temporal_candidate_drop_fraction"]["fail"]
    assert win.status == "unvalidated"


def test_missing_consistency_info_does_not_fail_the_drop_check():
    _, result = scene_result(7)
    run = result.regions[FLAT_REGION]
    run.labels.info = {}
    run.labels.consistency_candidates = run.labels.consistency_dropped = None
    (win,) = assess_windows(run, DiagnosticsParams())
    check = win.checks["temporal_candidate_drop_fraction"]
    assert check["value"] is None
    assert not check["fail"]
    run.labels.info = {"consistency": "not a dict"}
    (win,) = assess_windows(run, DiagnosticsParams())
    assert win.checks["temporal_candidate_drop_fraction"]["value"] is None


def test_the_drop_share_is_judged_per_window():
    _, result = scene_result(7)
    labels = result.regions[FLAT_REGION].labels
    labels.consistency_candidates = np.ones(400, dtype=np.int64)
    labels.consistency_dropped = np.r_[np.ones(200), np.zeros(200)].astype(np.int64)
    params = DiagnosticsParams(window_frames=200, min_temporal_candidates=10)
    first, second = assess_windows(result.regions[FLAT_REGION], params)
    bad, good = (w.checks["temporal_candidate_drop_fraction"] for w in (first, second))
    assert (bad["value"], bad["n_candidates"], bad["fail"]) == (1.0, 200, True)
    assert (good["value"], good["n_candidates"], good["fail"]) == (0.0, 200, False)
    assert first.status != second.status
    # a window without candidates has nothing to judge
    labels.consistency_candidates = np.zeros(400, dtype=np.int64)
    labels.consistency_dropped = np.zeros(400, dtype=np.int64)
    (win,) = assess_windows(result.regions[FLAT_REGION], DiagnosticsParams())
    check = win.checks["temporal_candidate_drop_fraction"]
    assert (check["value"], check["n_candidates"], check["fail"]) == (0.0, 0, False)


def test_without_per_frame_counts_the_region_wide_share_applies_to_every_window():
    _, result = scene_result(7)
    labels = result.regions[FLAT_REGION].labels
    labels.consistency_candidates = labels.consistency_dropped = None
    labels.info["consistency"] = {"n_candidates": 500, "dropped_fraction": 0.5}
    wins = assess_windows(result.regions[FLAT_REGION], DiagnosticsParams(window_frames=200))
    values = [w.checks["temporal_candidate_drop_fraction"]["value"] for w in wins]
    assert values == [0.5, 0.5]
    assert all(w.status == "uncertain" for w in wins)


# ----------------------------------------------------------------------- self-validation
def test_self_validation_scores_held_out_decodes():
    _, result = scene_result(8)
    run = result.regions[FLAT_REGION]
    windows = window_edges(400, 200)
    params = DiagnosticsParams(val_run_trials=3)
    val = sv(run, windows, params)
    assert set(val) == {"leave1", "leave1_per_tag", "run_masks"}
    leave1 = val["leave1"]
    assert leave1["n_heldout_bee_frames"] == int((run.labels.candidates >= 0).sum())
    assert leave1["coverage"] > 0.95
    assert leave1["precision"] > 0.98
    assert leave1["precision_ci95"][0] > 0.95
    assert set(leave1["by_nn_distance"]) == {"<100px", "100-150px", ">=150px"}
    assert len(leave1["by_window"]) == 2
    assert (
        sum(w["n_heldout_bee_frames"] for w in leave1["by_window"])
        == leave1["n_heldout_bee_frames"]
    )
    assert list(val["leave1_per_tag"]) == TAGS
    assert all(v["precision"] > 0.95 for v in val["leave1_per_tag"].values())
    masks = val["run_masks"]
    assert masks["n_heldout_bee_frames"] > 0
    assert masks["precision"] > 0.98
    # solving with held-out votes never changes the stored labels
    assert run.labels is result.regions[FLAT_REGION].labels


def test_self_validation_is_deterministic():
    _, result = scene_result(8, n_frames=200)
    run = result.regions[FLAT_REGION]
    params = DiagnosticsParams(val_run_trials=2)
    first = sv(run, window_edges(200, 200), params)
    second = sv(run, window_edges(200, 200), params)
    assert json.dumps(first, default=_json_default) == json.dumps(second, default=_json_default)


def test_self_validation_without_run_trials_or_labels():
    _, result = scene_result(9, n_frames=120)
    run = result.regions[FLAT_REGION]
    val = sv(run, [(0, 120)], DiagnosticsParams(val_run_trials=0))
    assert val["run_masks"] == {}
    assert val["leave1"]["n_heldout_bee_frames"] > 0
    labels = run.labels
    run.labels = None
    assert sv(run, [(0, 120)], DiagnosticsParams()) == {}
    run.labels = labels
    run.votes = None
    assert sv(run, [(0, 120)], DiagnosticsParams()) == {}


def test_self_validation_without_entities_uses_infinite_distances():
    _, result = scene_result(9, n_frames=120)
    run = result.regions[FLAT_REGION]
    run.labels.entities = None
    val = sv(run, [(0, 120)], DiagnosticsParams(val_run_trials=0))
    nn = val["leave1"]["by_nn_distance"]
    assert val["leave1"]["n_heldout_bee_frames"] > 0
    assert nn["<100px"]["n_heldout_bee_frames"] == 0
    assert nn["<100px"]["precision"] is None
    assert nn["<100px"]["coverage"] is None


@pytest.mark.parametrize(
    ("seed", "kwargs"),
    [
        (10, {"n_frames": 30}),  # shorter than most run lengths
        (11, {"n_frames": 400, "decode_rate": 1.0}),  # runs are placeable, then collide
        (12, {"n_frames": 400, "decode_rate": 0.05}),  # decodes too sparse for any run
        (13, {"n_frames": 600, "decode_rate": 0.9}),
    ],
)
def test_run_masks_on_short_sparse_and_dense_decodes(seed, kwargs):
    _, result = scene_result(seed, **kwargs)
    run = result.regions[FLAT_REGION]
    if run.status != "ok":
        pytest.skip("no data")
    val = sv(run, window_edges(kwargs["n_frames"], 6000), DiagnosticsParams(val_run_trials=6))
    assert "run_masks" in val
    runs = val["run_masks"]
    assert runs == {} or runs["n_heldout_bee_frames"] > 0


def test_leave1_criteria_enter_the_status():
    _, result = scene_result(14, decode_rate=0.97)
    run = result.regions[FLAT_REGION]
    windows = window_edges(400, 6000)
    val = sv(run, windows, DiagnosticsParams(val_run_trials=0))
    (trusted,) = assess_windows(run, DiagnosticsParams(), val)
    assert trusted.status == "trusted"
    assert {"leave1_precision", "leave1_worst_tag_precision"} <= set(trusted.checks)
    strict = DiagnosticsParams(min_leave1_precision=1.1, min_tag_leave1_precision=1.1)
    (win,) = assess_windows(run, strict, val)
    assert win.status == "no_repair_needed"  # decoded completeness >= 0.95
    assert win.checks["leave1_precision"]["fail"]
    _, poor = scene_result(14, decode_rate=0.5)
    poor_run = poor.regions[FLAT_REGION]
    poor_val = sv(poor_run, windows, DiagnosticsParams(val_run_trials=0))
    (win,) = assess_windows(poor_run, strict, poor_val)
    assert win.status == "uncertain"
    few = DiagnosticsParams(min_tag_n=10**6, min_tag_leave1_precision=1.1)
    (win,) = assess_windows(run, few, val)
    assert win.checks["leave1_worst_tag_precision"]["value"] is None
    assert not win.checks["leave1_worst_tag_precision"]["fail"]


def test_assess_result_validates_only_solved_regions():
    _, result = scene_result(15, n_frames=150)
    statuses, validation = assess_result(result, DiagnosticsParams())
    assert list(statuses) == [FLAT_REGION]
    assert validation == {}
    assert statuses[FLAT_REGION][0].status == "unvalidated"
    statuses, validation = assess_result(
        result, DiagnosticsParams(self_validate=True, val_run_trials=1)
    )
    assert statuses[FLAT_REGION][0].status == "trusted"
    assert set(validation[FLAT_REGION]) == {"leave1", "leave1_per_tag", "run_masks"}
    _, unsupported = scene_result(15, closed_set=TAGS[:2], n_frames=30)
    assert assess_result(unsupported, DiagnosticsParams(self_validate=True)) == ({}, {})


# ---------------------------------------------------------------------------- policy
def snapshot(result):
    return {
        "tag": result.slot_tag.copy(),
        "source": result.slot_source.copy(),
        "flag": result.slot_flag.copy(),
    }


@pytest.mark.parametrize("status", ["uncertain", "no_repair_needed"])
def test_withhold_removes_recovered_labels_and_keeps_decoded_ones(status):
    _, result = scene_result(16, absent={3: (50, 349)})
    before = snapshot(result)
    decoded = before["source"] == Source.DECODE
    recovered = (before["tag"] >= 0) & ~decoded
    assert recovered.any()
    statuses = {FLAT_REGION: [WindowStatus(0, 400, status, [], {})]}
    n = apply_policy(result, statuses, DiagnosticsParams())
    assert n == int(recovered.sum())
    np.testing.assert_array_equal(result.slot_tag[decoded], before["tag"][decoded])
    assert (result.slot_tag[recovered] == -1).all()
    assert (result.slot_flag[recovered] == SlotFlag.WITHHELD).all()
    assert (result.slot_source[recovered] == Source.NONE).all()
    labels = result.regions[FLAT_REGION].labels
    assert set(np.unique(labels.tag_src)) <= {Source.NONE, Source.DECODE}
    assert np.isnan(labels.tag_xy[labels.tag_src == Source.NONE]).all()
    assert (labels.tag_slot[labels.tag_src == Source.NONE] == -1).all()
    assert np.isnan(labels.tag_conf[labels.tag_src == Source.NONE]).all()


def test_withhold_only_touches_untrusted_windows():
    _, result = scene_result(16, absent={3: (50, 349)})
    before = snapshot(result)
    statuses = {
        FLAT_REGION: [
            WindowStatus(0, 200, "trusted", [], {}),
            WindowStatus(200, 400, "uncertain", [], {}),
        ]
    }
    apply_policy(result, statuses, DiagnosticsParams())
    np.testing.assert_array_equal(result.slot_tag[:200], before["tag"][:200])
    recovered = (before["tag"][200:] >= 0) & (before["source"][200:] != Source.DECODE)
    assert (result.slot_tag[200:][recovered] == -1).all()


def test_policy_is_a_no_op_for_trusted_windows_emit_and_unsolved_regions():
    _, result = scene_result(17, absent={3: (50, 349)})
    before = snapshot(result)
    trusted = {FLAT_REGION: [WindowStatus(0, 400, "unvalidated", [], {})]}
    assert apply_policy(result, trusted, DiagnosticsParams()) == 0
    untrusted = {FLAT_REGION: [WindowStatus(0, 400, "uncertain", [], {})]}
    assert apply_policy(result, untrusted, DiagnosticsParams(untrusted_policy="emit")) == 0
    np.testing.assert_array_equal(result.slot_tag, before["tag"])
    result.regions[FLAT_REGION].labels, labels = None, result.regions[FLAT_REGION].labels
    assert apply_policy(result, untrusted, DiagnosticsParams()) == 0
    result.regions[FLAT_REGION].labels = labels


# ---------------------------------------------------------------------------- report
def test_report_content_and_json():
    scene, result = scene_result(18, absent={3: (50, 349)}, miss_rate=0.05)
    params = DiagnosticsParams(self_validate=True, val_run_trials=1)
    statuses, validation = assess_result(result, params)
    apply_policy(result, statuses, params)
    cons = ConsistencyParams()
    report = br(result, statuses, validation, cons, extra={"video": "v.mp4"})
    json.dumps(report, default=_json_default)
    assert report["engine"] == "joint"
    assert report["video"] == "v.mp4"
    assert report["n_frames"] == 400
    assert report["first_frame"] == 0
    assert report["frame_size"] == [1800, 1800]
    assert report["n_instances"] == int(scene.table.valid.sum())
    assert report["n_instances_labelled"] == int((result.slot_tag >= 0).sum())
    assert report["decodes"]["decodes"] == len(scene.votes)
    assert report["region_status_counts"] == {"ok": 1}
    assert report["source_names"][1] == "decode"
    region = report["regions"][FLAT_REGION]
    assert region["tags"] == TAGS
    assert region["phantom_tags"] == 0
    assert region["status"] == "ok"
    assert region["n_tags"] == 4
    before, after = region["completeness_before"], region["completeness_after"]
    assert before["bee_frame_completeness"] < after["bee_frame_completeness"]
    assert after["bee_frame_completeness"] > 0.95
    assert region["sources"]["decode"] == int((result.slot_source == Source.DECODE).sum())
    assert set(region["abstentions"]) == set(SlotFlag.NAMES[1:])
    assert region["n_instances_labelled"] <= region["n_instances_in_inference"]
    assert report["sidecar_schema_version"] == 2
    assert report["reason_names"][:3] == ["none", "ghost_instance", "duplicate_within_dedupe_px"]
    unlabelled = int((scene.table.valid & (result.slot_tag < 0)).sum())
    assert sum(report["abstentions_by_reason"].values()) == unlabelled
    assert sum(region["abstentions_by_reason"].values()) == unlabelled
    assert set(region["abstentions_by_reason"]) <= set(report["reason_names"][1:])
    assert report["abstentions_by_reason"].get("untrusted_window_withheld", 0) == int(
        (result.slot_flag == SlotFlag.WITHHELD).sum()
    )
    assert report["n_tracklets"] == region["engine"]["n_tracklets"]
    assert set(region["demoted_decodes"]) == {
        "n_conflict",
        "n_stray",
        "n_relabelled",
        "n_confirmed",
    }
    assert region["confidence"]["threshold"] == 0.99
    assert region["confidence"]["min_recovered"] >= 0.99
    assert 0.99 <= region["confidence"]["mean_recovered"] <= 1.0
    assert set(region["temporal"]) == {
        "jumps",
        "recovered_runs",
        "duplicates_40px",
        "gate_violations",
        "anchors_contradicting_own_track",
    }
    assert region["temporal"]["jumps"]["rec_rec"]["gt300"] == 0
    assert region["engine"]["n_tracklets"] >= 4
    (window,) = region["windows"]
    assert window["status"] == "trusted"
    assert (window["start_row"], window["stop_row"]) == (0, 400)
    assert region["self_validation"]["leave1"]["precision"] > 0.98


def test_report_of_a_run_without_recovered_labels_or_optional_pieces():
    scene = make_scene(19, n_frames=100, decode_rate=1.0)
    result, source = flat_run(scene)
    source.frame_size = None
    run = result.regions[FLAT_REGION]
    run.config = None
    run.labels.valid = None
    result.frame_size = None
    report = br(result, {}, {}, ConsistencyParams())
    region = report["regions"][FLAT_REGION]
    assert report["frame_size"] is None
    assert "windows" in region
    assert region["windows"] == []
    assert region["confidence"] == {
        "mean_recovered": None,
        "min_recovered": None,
        "threshold": None,
    }
    assert region["n_instances_in_inference"] == run.labels.slot_tag.size
    assert "self_validation" not in region
    assert "video" not in report


def test_report_of_an_unsolved_region_has_only_the_basics():
    _, result = scene_result(20, closed_set=TAGS[:2], n_frames=30)
    report = br(result, {}, {}, ConsistencyParams())
    region = report["regions"][FLAT_REGION]
    assert set(region) == {"tags", "n_bees", "phantom_tags", "status", "reasons"}
    assert region["status"] == "unsupported_tagset"
    assert report["region_status_counts"] == {"unsupported_tagset": 1}
    assert report["n_instances_labelled"] == 0


def test_completeness_after_never_falls_below_before():
    _, result = scene_result(21, absent={0: (10, 300)})
    report = br(result, {}, {}, ConsistencyParams())
    region = report["regions"][FLAT_REGION]
    assert (
        region["completeness_after"]["bee_frame_completeness"]
        >= region["completeness_before"]["bee_frame_completeness"]
    )
    assert region["completeness_before"]["frames_with_n_tags"]["4"] < 400
