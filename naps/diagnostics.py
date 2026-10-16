"""Diagnostics of the joint identity engine: what was labelled, from what, and can it be trusted?

Per region the report answers

* **completeness** - frames with N/N-1/.../0 tags and bee-frame completeness, *before* (decoded
  anchors only, i.e. what the decoder alone resolves) and *after* the engine;
* **sources** - how many labels came from decodes, tracklet inference, elimination, appearance
  evidence and interpolation, and the confidence distribution;
* **abstentions** - why instances stay untracked (ghost, duplicate, excess instance, low
  confidence, untagged bee, blocked by an anchor, removed by the consistency stage, withheld);
* **disputes** - decoded labels the model confidently disagrees with (flagged, never changed);
* **temporal quality** - jump rates between consecutive frames by label class, recovered labels
  next to another tag, links beyond the physical gate;
* **window status** - for every window of ``window_frames`` frames of a *solved* region:
  ``trusted``, ``uncertain`` (recovered labels withheld unless ``untrusted_policy="emit"``),
  ``no_repair_needed`` or ``unvalidated`` (cheap checks pass, self-validation not run). A region
  that was not solved (``no_data``, ``unsupported_tagset``) has no windows; its status is the
  region's ``status``;
* **self-validation** (opt-in, ``self_validate``) - hold out the decodes of one tag for the whole
  video (leave-1) and of random runs, re-run the model and score the recovered labels against the
  held-out decodes (precision / coverage with Wilson intervals). Truth is the decode itself, so
  it is only an estimate: real dropouts are harder.

The criteria of the window status are the calibrated defaults of :class:`DiagnosticsParams`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from naps.consistency import (
    ConsistencyParams,
    anchors_off_chain,
    bridge_violations,
    duplicate_stats,
    jump_matrix,
    recovered_run_stats,
)
from naps.identity import RegionLabels, SlotFlag, Votes
from naps.joint import SIDECAR_SCHEMA_VERSION, STATUS_OK, JointResult, RegionRun, region_reasons
from naps.reasons import REASON_NAMES
from naps.sources import SOURCE_NAMES, Source

__all__ = [
    "DiagnosticsParams",
    "WindowStatus",
    "apply_policy",
    "assess_result",
    "assess_windows",
    "build_report",
    "completeness",
    "self_validate",
    "wilson",
    "window_edges",
]

logger = logging.getLogger(__name__)

#: Window statuses (the first three are decided from the data, the others from the region).
TRUSTED = "trusted"
UNCERTAIN = "uncertain"
NO_REPAIR_NEEDED = "no_repair_needed"
UNVALIDATED = "unvalidated"
# checks that, in a window the decoder already resolves (almost) completely, only say "the
# recovered labels would be unreliable": there is nothing to repair
_LEAVE1_CHECKS = (
    "leave1_precision",
    "leave1_worst_tag_precision",
    "temporal_candidate_drop_fraction",
)
_WILSON_Z = 1.96
_LENGTH_BINS = {"1-5": (1, 5), "6-25": (6, 25), "26-100": (26, 100), "101-500": (101, 500)}


@dataclass(frozen=True)
class DiagnosticsParams:
    """Criteria and options of the diagnostics (calibrated defaults, see the module docstring).

    Attributes:
        window_frames: Length of a status window in frames (6000 = 5 minutes at 20 fps).
        min_median_tracklet_len: Frame-weighted median tracklet length below which a window is
            a pile (tracklets fragment into ghost / jitter pieces).
        max_anchor_step_p99: 99th percentile of the frame-to-frame step of decoded positions
            above which the decoder itself is unreliable in the window (clean ~16 px, piles ~78).
        min_leave1_precision: Pooled leave-1 precision below which the window is uncertain.
        min_tag_leave1_precision: ... for the worst single tag with at least ``min_tag_n`` labels.
        min_tag_n: Labels a tag needs for the worst-tag check.
        max_impure_tracklet_frac: Informational: share of anchored tracklets with two tags.
        max_temporal_drop_frac: Share of the window's candidate labels (recovered labels the
            consistency stage examined) that it had to drop, above which the recovery is unstable
            (chosen, not calibrated). Computed per window.
        min_temporal_candidates: ... judged only if the window has at least this many candidates
            (chosen).
        no_repair_min_completeness: Completeness above which a window that only fails leave-1
            checks is ``no_repair_needed`` instead of ``uncertain``.
        untrusted_policy: ``"withhold"`` removes recovered labels from uncertain /
            no_repair_needed windows (decoded labels stay); ``"emit"`` keeps them.
        self_validate: Run the hold-out self-validation (slow: one solve per tag).
        val_run_trials: Number of random run-mask trials (each masks several runs at once).
        val_seed: Seed of the run masks.
    """

    window_frames: int = 6000
    min_median_tracklet_len: float = 20.0
    max_anchor_step_p99: float = 60.0
    min_leave1_precision: float = 0.98
    min_tag_leave1_precision: float = 0.95
    min_tag_n: int = 100
    max_impure_tracklet_frac: float = 0.10
    max_temporal_drop_frac: float = 0.25
    min_temporal_candidates: int = 200
    no_repair_min_completeness: float = 0.95
    untrusted_policy: Literal["withhold", "emit"] = "withhold"
    self_validate: bool = False
    val_run_trials: int = 3
    val_seed: int = 12345


@dataclass
class WindowStatus:
    """The assessment of one window of one region.

    Attributes:
        start: First frame row (0-based within the table).
        stop: One past the last frame row.
        status: One of the module-level status names.
        reasons: Failed criteria, human readable.
        checks: Every criterion with its value, threshold and verdict.
    """

    start: int
    stop: int
    status: str
    reasons: list[str]
    checks: dict[str, dict[str, object]]


# ------------------------------------------------------------------------- basic numbers
def wilson(k: int, n: int, z: float = _WILSON_Z) -> list[float | None]:
    """Wilson score interval of ``k`` successes in ``n`` trials (``[None, None]`` if ``n`` is 0)."""
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * float(np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def completeness(tag_xy: NDArray[np.float64]) -> dict[str, object]:
    """Frames with ``K..0`` tags and the share of present bee-frames.

    Args:
        tag_xy: ``(F, K, 2)`` positions, ``NaN`` = absent.
    """
    present = np.isfinite(tag_xy[..., 0])
    n_tags = present.sum(axis=1)
    k = present.shape[1]
    return {
        "frames_with_n_tags": {str(n): int((n_tags == n).sum()) for n in range(k, -1, -1)},
        "bee_frame_completeness": round(float(present.mean()), 4) if present.size else None,
        "per_tag": [round(float(v), 4) for v in present.mean(axis=0)] if present.size else [],
    }


def _step_p99(xy: NDArray[np.float64], mask: NDArray[np.bool_]) -> float | None:
    """99th percentile of the step between consecutive frames, over steps with both ends in mask."""
    both = mask[1:] & mask[:-1]
    step = np.linalg.norm(xy[1:] - xy[:-1], axis=-1)[both]
    step = step[np.isfinite(step)]
    return float(np.percentile(step, 99)) if len(step) else None


def _tracklet_median_len(labels: RegionLabels, start: int, stop: int) -> float | None:
    """Frame-weighted median length of the tracklets present in rows ``start..stop``."""
    tr = labels.tracklets
    if tr is None or tr.n_tracklets == 0:
        return None
    ids = np.unique(tr.tid[start:stop][tr.tid[start:stop] >= 0])
    if len(ids) == 0:
        return None
    lengths = np.sort(tr.length[ids])[::-1]
    cum = np.cumsum(lengths)
    return float(lengths[np.searchsorted(cum, cum[-1] / 2)])


def _impure_fraction(labels: RegionLabels, start: int, stop: int, n_tags: int) -> float | None:
    """Share of anchored tracklets that carry a second well-supported tag (informational)."""
    tr, cand = labels.tracklets, labels.candidates
    if tr is None or cand is None or tr.n_tracklets == 0:
        return None
    tid, tag = tr.tid[start:stop], cand[start:stop]
    rows, cols = np.nonzero((tid >= 0) & (tag >= 0))
    if len(rows) == 0:
        return None
    counts = np.zeros((tr.n_tracklets, n_tags))
    np.add.at(counts, (tid[rows, cols], tag[rows, cols]), 1)
    anchored = counts.sum(axis=1) >= 5
    if not anchored.any():
        return None
    ordered = np.sort(counts, axis=1)[:, ::-1]
    impure = anchored & (ordered[:, 1] >= 2) & (ordered[:, 1] >= 0.03 * ordered[:, 0])
    return float(impure.sum()) / float(anchored.sum())


# --------------------------------------------------------------------- self-validation
def _score_heldout(
    truth: NDArray[np.int16],
    held: NDArray[np.bool_],
    labels: RegionLabels,
    nn: NDArray[np.float64] | None,
) -> dict[str, NDArray]:
    """Per held-out bee-frame: was it labelled and was the label right?"""
    rows, cols = np.nonzero(held)
    pred = labels.slot_tag[rows, cols]
    emitted = pred >= 0
    return {
        "row": rows,
        "slot": cols,
        "emitted": emitted,
        "correct": emitted & (pred == truth[rows, cols]),
        "nn": nn[rows, cols] if nn is not None else np.full(len(rows), np.inf),
    }


def _slot_nn(labels: RegionLabels) -> NDArray[np.float64] | None:
    """``(F, S)`` distance of every kept instance to its nearest other kept instance."""
    ent = labels.entities
    if ent is None:
        return None
    out = np.full(ent.of_slot.shape, np.inf)
    rows, cols = np.nonzero(ent.of_slot >= 0)
    out[rows, cols] = ent.nn[rows, ent.of_slot[rows, cols]]
    return out


def _filter_votes(votes: Votes, drop: NDArray[np.bool_]) -> Votes:
    """The votes without the entries flagged in ``drop``."""
    keep = ~drop
    return Votes(votes.frame[keep], votes.slot[keep], votes.tag[keep], votes.weight[keep])


def _run_masks(
    candidates: NDArray[np.int16], n_tags: int, rng: np.random.Generator
) -> NDArray[np.bool_]:
    """Choose disjoint runs of decoded frames to hold out (4 per length bin).

    A run of length ``L`` of one tag is chosen where that tag is decoded in every frame, so the
    truth of every held-out frame is known; runs are kept 50 frames apart. (A run starts at a
    frame in ``0 .. n_frames - length - 1``, so none ends on the very last frame; the masks are
    seeded, and changing that would change every published self-validation number.)

    Returns:
        A ``(F, K)`` mask of held-out tag-frames.
    """
    n_frames = candidates.shape[0]
    present = np.zeros((n_frames, n_tags), dtype=bool)
    rows, cols = np.nonzero(candidates >= 0)
    present[rows, candidates[rows, cols]] = True
    mask = np.zeros_like(present)
    for lo, hi in _LENGTH_BINS.values():
        for _ in range(4):
            length = min(max(int(np.exp(rng.uniform(np.log(lo), np.log(hi + 1)))), lo), hi)
            tag = int(rng.integers(n_tags))
            if n_frames <= length:
                continue
            start = int(rng.integers(0, n_frames - length))
            near = slice(max(start - 50, 0), start + length + 50)
            if present[start : start + length, tag].all() and not mask[near].any():
                mask[start : start + length, tag] = True
    return mask


def _aggregate(parts: list[dict[str, NDArray]], windows: list[tuple[int, int]]) -> dict:
    """Precision / coverage pooled and per window / per nearest-neighbour class."""
    if not parts:
        return {}
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    n = len(cat["row"])

    def block(mask: NDArray[np.bool_]) -> dict[str, object]:
        n_m, e_m, c_m = (
            int(mask.sum()),
            int((cat["emitted"] & mask).sum()),
            int((cat["correct"] & mask).sum()),
        )
        return {
            "n_heldout_bee_frames": n_m,
            "n_emitted": e_m,
            "n_correct": c_m,
            "coverage": round(e_m / n_m, 4) if n_m else None,
            "precision": round(c_m / e_m, 4) if e_m else None,
            "precision_ci95": wilson(c_m, e_m),
        }

    out = block(np.ones(n, dtype=bool))
    out["by_nn_distance"] = {
        # an instance without any neighbour (nn = inf) belongs to the open-ended last class
        label: block(
            ((cat["nn"] >= lo) & (cat["nn"] < hi)) | (np.isposinf(cat["nn"]) & bool(np.isinf(hi)))
        )
        for label, (lo, hi) in {
            "<100px": (0, 100),
            "100-150px": (100, 150),
            ">=150px": (150, np.inf),
        }.items()
    }
    out["by_window"] = [block((cat["row"] >= a) & (cat["row"] < b)) for a, b in windows]
    return out


def self_validate(
    run: RegionRun, windows: list[tuple[int, int]], params: DiagnosticsParams
) -> dict[str, object]:
    """Hold out decodes, re-solve, and score the recovered labels against them.

    *Leave-1*: all decodes of one tag are removed for the whole region (the regime of almost
    every real dropout: one bee absent). *Run masks*: the decodes of a tag are removed over random
    runs of 1..500 frames. Truth is the clean decode that was removed, matched to its instance.

    Args:
        run: A solved region.
        windows: ``(start, stop)`` frame-row ranges for the per-window numbers.
        params: Options (number of run trials, seed).

    Returns:
        ``{"leave1": {...}, "run_masks": {...}}`` (precision / coverage with Wilson intervals,
        pooled, by nearest-neighbour distance and by window) and the per-tag leave-1 numbers.
    """
    labels, votes = run.labels, run.votes
    if labels is None or votes is None or labels.candidates is None:
        return {}
    n_tags = len(run.tags)
    truth = labels.candidates
    nn = _slot_nn(labels)
    leave1: list[dict[str, NDArray]] = []
    per_tag: dict[int, dict[str, object]] = {}
    for tag in range(n_tags):
        trial = run.solve(_filter_votes(votes, votes.tag == tag))
        part = _score_heldout(truth, truth == tag, trial, nn)
        leave1.append(part)
        per_tag[run.tags[tag]] = _aggregate([part], windows)
    out: dict[str, object] = {"leave1": _aggregate(leave1, windows), "leave1_per_tag": per_tag}
    rng = np.random.default_rng(params.val_seed)
    masked_parts: list[dict[str, NDArray]] = []
    for _ in range(params.val_run_trials):
        mask = _run_masks(truth, n_tags, rng)
        if not mask.any():
            continue
        trial = run.solve(_filter_votes(votes, mask[votes.frame, votes.tag]))
        held = np.zeros(truth.shape, dtype=bool)
        rows, cols = np.nonzero(truth >= 0)
        held[rows, cols] = mask[rows, truth[rows, cols]]
        masked_parts.append(_score_heldout(truth, held, trial, nn))
    out["run_masks"] = _aggregate(masked_parts, windows)
    return out


# ------------------------------------------------------------------------------ status
def _check(value: float | None, threshold: float, *, fail_if_above: bool, **extra: object) -> dict:
    """One criterion: value, threshold and verdict (a missing value never fails)."""
    failed = value is not None and (value > threshold if fail_if_above else value < threshold)
    return {"value": value, "threshold": threshold, "fail": bool(failed), **extra}


def window_edges(n_frames: int, size: int) -> list[tuple[int, int]]:
    """``(start, stop)`` frame-row ranges of consecutive windows of ``size`` frames."""
    return [(s, min(s + size, n_frames)) for s in range(0, n_frames, size)]


def _window_drop(labels: RegionLabels, window: slice) -> tuple[int, float | None]:
    """``(n_candidates, dropped share)`` of the consistency stage within the rows of a window.

    The per-frame counts of the stage are summed over the window. Labels without them (built by
    hand) fall back to the region-wide numbers in ``info["consistency"]``, which are then the same
    for every window.
    """
    if labels.consistency_candidates is not None and labels.consistency_dropped is not None:
        n_candidates = int(labels.consistency_candidates[window].sum())
        n_dropped = int(labels.consistency_dropped[window].sum())
        return n_candidates, (round(n_dropped / n_candidates, 4) if n_candidates else 0.0)
    cons = labels.info.get("consistency", {})
    if not isinstance(cons, dict):
        return 0, None
    drop = cons.get("dropped_fraction")
    return int(cons.get("n_candidates", 0)), drop if isinstance(drop, float) else None


def _data_checks(
    labels: RegionLabels, span: tuple[int, int], n_tags: int, params: DiagnosticsParams
) -> dict[str, dict[str, object]]:
    """The criteria that need no hold-out: tracklet length, decoder steps, purity, drop share."""
    start, stop = span
    sl = slice(start, stop)
    n_candidates, drop = _window_drop(labels, sl)
    drop_check = _check(
        drop,
        params.max_temporal_drop_frac,
        fail_if_above=True,
        n_candidates=n_candidates,
    )
    drop_check["fail"] = bool(drop_check["fail"]) and n_candidates >= params.min_temporal_candidates
    return {
        "median_tracklet_len_frame_weighted": _check(
            _tracklet_median_len(labels, start, stop),
            params.min_median_tracklet_len,
            fail_if_above=False,
        ),
        "anchor_step_p99_px": _check(
            _step_p99(labels.tag_xy[sl], labels.tag_src[sl] == Source.DECODE),
            params.max_anchor_step_p99,
            fail_if_above=True,
        ),
        "impure_tracklet_frac": _check(
            _impure_fraction(labels, start, stop, n_tags),
            params.max_impure_tracklet_frac,
            fail_if_above=True,
            informational=True,
        ),
        "temporal_candidate_drop_fraction": drop_check,
    }


def _validation_checks(
    validation: dict[str, Any], index: int, params: DiagnosticsParams
) -> dict[str, dict[str, object]]:
    """The leave-1 criteria of window ``index`` from :func:`self_validate`."""
    window = validation["leave1"]["by_window"][index]
    per_tag = [v["by_window"][index] for v in validation["leave1_per_tag"].values()]
    worst = [
        w["precision"]
        for w in per_tag
        if w["n_emitted"] >= params.min_tag_n and w["precision"] is not None
    ]
    return {
        "leave1_precision": _check(
            window["precision"],
            params.min_leave1_precision,
            fail_if_above=False,
            coverage=window["coverage"],
        ),
        "leave1_worst_tag_precision": _check(
            min(worst) if worst else None, params.min_tag_leave1_precision, fail_if_above=False
        ),
    }


def _decoded_completeness(labels: RegionLabels, span: tuple[int, int]) -> float:
    """Share of bee-frames of the window that carry a decode."""
    sl = slice(*span)
    return float((labels.tag_src[sl] == Source.DECODE).mean())


def assess_windows(
    run: RegionRun,
    params: DiagnosticsParams,
    validation: dict[str, Any] | None = None,
) -> list[WindowStatus]:
    """Assign a status to every window of a solved region.

    A failed (non-informational) criterion makes the window ``uncertain``; if only leave-1 type
    checks fail and the decoder already resolves >= ``no_repair_min_completeness`` of the window,
    it is ``no_repair_needed``. Without self-validation a window that passes the cheap checks is
    ``unvalidated``, with it ``trusted``.

    Args:
        run: A solved region.
        params: Criteria.
        validation: Output of :func:`self_validate` for the same windows (optional).

    Returns:
        One :class:`WindowStatus` per window, in order.
    """
    labels = run.labels
    if labels is None:
        return []
    out = []
    for i, span in enumerate(window_edges(labels.slot_tag.shape[0], params.window_frames)):
        checks = _data_checks(labels, span, len(run.tags), params)
        if validation and validation.get("leave1"):
            checks.update(_validation_checks(validation, i, params))
        failed = [k for k, v in checks.items() if v["fail"] and not v.get("informational")]
        reasons = [f"{k}: {checks[k]['value']:.3g} vs {checks[k]['threshold']}" for k in failed]
        if (
            failed
            and set(failed) <= set(_LEAVE1_CHECKS)
            and _decoded_completeness(labels, span) >= params.no_repair_min_completeness
        ):
            status = NO_REPAIR_NEEDED
        elif failed:
            status = UNCERTAIN
        else:
            status = TRUSTED if validation else UNVALIDATED
        out.append(WindowStatus(span[0], span[1], status, reasons, checks))
    return out


def apply_policy(
    result: JointResult, statuses: dict[str, list[WindowStatus]], params: DiagnosticsParams
) -> int:
    """Withhold recovered labels in untrusted windows (decoded labels are never touched).

    Args:
        result: The joint result; modified in place (labels, then ``refresh``).
        statuses: Window statuses by region.
        params: ``untrusted_policy`` decides whether anything is withheld.

    Returns:
        The number of instances whose label was withheld.
    """
    if params.untrusted_policy != "withhold":
        return 0
    n_withheld = 0
    for name, windows in statuses.items():
        labels = result.regions[name].labels
        if labels is None:
            continue
        for win in windows:
            if win.status not in (UNCERTAIN, NO_REPAIR_NEEDED):
                continue
            sl = slice(win.start, win.stop)
            recovered = (labels.slot_tag[sl] >= 0) & (labels.slot_source[sl] != Source.DECODE)
            labels.slot_flag[sl][recovered] = SlotFlag.WITHHELD
            labels.slot_tag[sl][recovered] = -1
            labels.slot_conf[sl][recovered] = np.nan
            labels.slot_source[sl][recovered] = Source.NONE
            drop = (labels.tag_src[sl] != Source.DECODE) & (labels.tag_src[sl] != Source.NONE)
            labels.tag_xy[sl][drop] = np.nan
            labels.tag_slot[sl][drop] = -1
            labels.tag_src[sl][drop] = Source.NONE
            labels.tag_conf[sl][drop] = np.nan
            n_withheld += int(recovered.sum())
    if n_withheld:
        result.refresh()
    return n_withheld


# ------------------------------------------------------------------------------ report
def _source_counts(src: NDArray[np.int8]) -> dict[str, int]:
    """Number of labels per source (``none`` excluded)."""
    return {SOURCE_NAMES[s]: int((src == s).sum()) for s in range(1, len(SOURCE_NAMES))}


def _region_report(
    run: RegionRun,
    statuses: list[WindowStatus],
    validation: dict[str, object] | None,
    consistency: ConsistencyParams,
) -> dict[str, object]:
    """The report block of one region."""
    labels = run.labels
    out: dict[str, object] = {
        "tags": run.tags,
        "n_bees": run.n_bees,
        "phantom_tags": run.n_bees - len(run.tags),
        "status": run.status,
        "reasons": run.reasons,
    }
    if labels is None:
        return out
    n_tags = len(run.tags)
    anchored = np.where((labels.tag_src == Source.DECODE)[..., None], labels.tag_xy, np.nan)
    off = anchors_off_chain(labels.tag_xy, labels.tag_src, consistency)
    valid = labels.valid if labels.valid is not None else np.ones_like(labels.slot_tag, dtype=bool)
    labelled = labels.slot_tag >= 0
    out.update(
        completeness_before=completeness(anchored),
        completeness_after=completeness(labels.tag_xy),
        before_definition="decoded anchors only (no inference, no carry-forward)",
        sources=_source_counts(labels.slot_source),
        n_instances_in_inference=int(valid.sum()),
        n_instances_labelled=int(labelled.sum()),
        abstentions={
            SlotFlag.NAMES[f]: int(((labels.slot_flag == f) & ~labelled).sum())
            for f in range(1, len(SlotFlag.NAMES))
        },
        abstentions_by_reason=_reason_counts(region_reasons(labels)),
        demoted_decodes={
            "n_conflict": labels.info.get("n_conflict_decodes_demoted"),
            "n_stray": labels.info.get("n_stray_anchors_demoted"),
            "n_relabelled": labels.info.get("n_demoted_decodes_relabelled"),
            "n_confirmed": labels.info.get("n_demoted_decodes_confirmed"),
        },
        disputed_anchors=int(labels.slot_disputed.sum()),
        confidence={
            "mean_recovered": _mean_conf(labels, recovered=True),
            "min_recovered": _min_conf(labels),
            "threshold": run.config.identity.threshold if run.config else None,
        },
        temporal={
            "jumps": jump_matrix(labels.tag_xy, labels.tag_src, off=off),
            "recovered_runs": recovered_run_stats(labels.tag_xy, labels.tag_src),
            "duplicates_40px": duplicate_stats(labels.tag_xy, labels.tag_src),
            "gate_violations": bridge_violations(labels.tag_xy, labels.tag_src, consistency, off),
            "anchors_contradicting_own_track": int(off.sum()),
        },
        engine=dict(labels.info),
        windows=[
            {
                "start_row": w.start,
                "stop_row": w.stop,
                "status": w.status,
                "reasons": w.reasons,
                "checks": w.checks,
            }
            for w in statuses
        ],
        n_tags=n_tags,
    )
    if validation:
        out["self_validation"] = validation
    return out


def _mean_conf(labels: RegionLabels, *, recovered: bool) -> float | None:
    """Mean confidence of the recovered (non-decoded) labels."""
    mask = (labels.slot_tag >= 0) & ((labels.slot_source != Source.DECODE) == recovered)
    return round(float(np.mean(labels.slot_conf[mask])), 4) if mask.any() else None


def _min_conf(labels: RegionLabels) -> float | None:
    """Smallest confidence among the recovered labels."""
    mask = (labels.slot_tag >= 0) & (labels.slot_source != Source.DECODE)
    return round(float(np.min(labels.slot_conf[mask])), 4) if mask.any() else None


def build_report(
    result: JointResult,
    statuses: dict[str, list[WindowStatus]],
    validation: dict[str, dict[str, object]],
    consistency: ConsistencyParams,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """The per-video report of the joint engine (written by ``--report-json``).

    Args:
        result: The joint result.
        statuses: Window statuses by region (from :func:`assess_windows`).
        validation: Self-validation by region (possibly empty).
        consistency: The consistency thresholds used.
        extra: Additional top-level entries (options, paths, runtime).

    Returns:
        A JSON-serialisable dictionary.
    """
    regions = {
        name: _region_report(run, statuses.get(name, []), validation.get(name), consistency)
        for name, run in result.regions.items()
    }
    t = result.table
    report: dict[str, object] = {
        "engine": "joint",
        "first_frame": t.first_frame,
        "n_frames": t.n_frames,
        "frame_size": list(result.frame_size) if result.frame_size else None,
        "decodes": dict(result.counters),
        "n_instances": int(t.valid.sum()),
        "n_instances_labelled": int((result.slot_tag >= 0).sum()),
        "regions": regions,
        "region_status_counts": _count([r["status"] for r in regions.values()]),
        "source_names": list(SOURCE_NAMES),
        "reason_names": list(REASON_NAMES),
        "abstentions_by_reason": _reason_counts(result.slot_reason),
        "n_tracklets": sum(
            run.labels.tracklets.n_tracklets
            for run in result.regions.values()
            if run.labels is not None and run.labels.tracklets is not None
        ),
        "sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
    }
    if extra:
        report.update(extra)
    return report


def _reason_counts(reason: NDArray[np.int8]) -> dict[str, int]:
    """Number of instances per reason name (reasons that do not occur are omitted)."""
    codes, counts = np.unique(reason[reason > 0], return_counts=True)
    return {REASON_NAMES[int(c)]: int(n) for c, n in zip(codes, counts, strict=True)}


def _count(items: list[object]) -> dict[str, int]:
    """Count the occurrences of every item."""
    out: dict[str, int] = {}
    for item in items:
        out[str(item)] = out.get(str(item), 0) + 1
    return out


def assess_result(
    result: JointResult, params: DiagnosticsParams
) -> tuple[dict[str, list[WindowStatus]], dict[str, dict[str, object]]]:
    """Self-validate (if enabled) and assess the windows of every solved region.

    Args:
        result: The joint result.
        params: Criteria and options.

    Returns:
        ``(statuses, validation)`` by region name.
    """
    statuses: dict[str, list[WindowStatus]] = {}
    validation: dict[str, dict[str, object]] = {}
    for name, run in result.regions.items():
        if run.status != STATUS_OK or run.labels is None:
            continue
        windows = window_edges(run.labels.slot_tag.shape[0], params.window_frames)
        val = self_validate(run, windows, params) if params.self_validate else None
        if val:
            validation[name] = val
        statuses[name] = assess_windows(run, params, val)
    return statuses, validation
