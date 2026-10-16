"""The joint identity engine as a pipeline stage: instances + video -> a label for every instance.

:class:`JointMatching` is to the joint engine what :class:`naps.matching.Matching` is to the legacy
one. It

1. decodes the tag around **every** instance (tracked or not) through a
   :class:`~naps.matching.DecodeSource`,
2. splits the instances into *regions* (the whole input for ``--closed-set``, quadrants for the
   quadrant modes), each with its own closed set of tags,
3. runs :func:`naps.identity.solve_region` once per region, and
4. scatters the labels back onto the instances of the input.

Regions the model cannot handle are reported, never guessed: a region without any decoded tag is
``no_data``; a region whose tag set is not 3..``colony_size`` tags is ``unsupported_tagset``. A
region of three tags is solved as a closed set of three tags plus one untagged bee (a *phantom*
tag that is never emitted).

**Sidecar schema** (:func:`write_sidecar`, ``<output>.identity.npz``, ``SIDECAR_SCHEMA_VERSION`` =
2; files without ``schema_version`` are version 1, which lacks the arrays marked *new*). Arrays
of shape ``(frames, slots)`` are indexed like the instance table (``inst_index`` maps a slot to the
position of the instance in its labeled frame):

* ``tag`` (tag id, -1 = untracked), ``confidence``, ``source`` (names: ``source_names``),
  ``disputed``, ``flag`` (names: ``flag_names``), ``region`` (index into ``region_names``);
* ``reason`` (*new*, int8, names: ``reason_names``, enum: :class:`naps.reasons.Reason`): why a
  valid instance has no tag; nonzero exactly where the slot holds an instance with a position and
  ``tag`` is -1. The codes are stable (appended, never renumbered);
* ``tracklet_id`` (*new*, int64, -1 = in no tracklet: duplicate, culled for excess, no position,
  or region not solved): the tracklet of the instance, unique across the regions of the file
  (``tracklet_offset__<r>`` is added to the region's own 0-based ids). Ghost instances stay in
  their tracklet (they are only denied a label);
* ``decode_demoted`` (*new*, bool): the instance had a clean decode that was demoted to weak
  evidence (the same tag on two instances of the frame, or a minority tag on its tracklet), so it
  is not an anchor and its ``tag`` may differ from the decode;
* ``schema_version`` (*new*), ``session``, ``meta`` (JSON), per region ``<r>``: ``tags__<r>``,
  ``tags_qualified__<r>``, ``status__<r>``, ``tracklet_offset__<r>`` (*new*; -1 for a region
  that was not solved), and for solved regions ``tag_xy__<r>``, ``tag_source__<r>``,
  ``tag_confidence__<r>``, ``tag_slot__<r>``.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from naps.consistency import ConsistencyParams
from naps.evidence import EvidenceSource
from naps.identity import (
    MAX_TAGS,
    IdentityParams,
    RegionLabels,
    SlotFlag,
    Votes,
    solve_region,
)
from naps.instances import InstanceTable
from naps.matching import Decodes, DecodeSource
from naps.quadrants import QuadrantScope, region_index
from naps.reasons import REASON_NAMES, Reason
from naps.sources import SOURCE_NAMES, Source
from naps.tags import normalize_tag, require_tag, tag_name
from naps.tracklets import TrackletMode, TrackletParams

__all__ = [
    "FLAT_REGION",
    "SIDECAR_SCHEMA_VERSION",
    "JointConfig",
    "JointMatching",
    "JointResult",
    "RegionRun",
    "qualify_tag",
    "write_sidecar",
]

logger = logging.getLogger(__name__)


# A region may hold this many more instances per frame (median) than its colony has bees
# (SLEAP emits duplicates / ghosts), but not more: that means other animals are in the region.
_CROWDING_SLACK = 2


def _median_count(member: NDArray[np.bool_]) -> float:
    """Median number of instances per frame over the frames that have any (``member.any()``)."""
    counts = member.sum(axis=1)
    return float(np.median(counts[counts > 0]))


def _decoded_slots(decodes: Decodes) -> Iterator[tuple[int, int, Sequence[Hashable | None]]]:
    """Yield ``(frame, slot, decoded tags)`` for every instance with a decode entry."""
    for frame, per_slot in decodes.items():
        for slot, entries in per_slot.items():
            yield frame, cast("int", slot), entries


#: Name of the single region of the flat ``--closed-set`` mode.
FLAT_REGION = "all"

#: Layout version of the identity sidecar (see the module docstring): 1 = the unversioned first
#: layout, 2 = adds ``reason``, ``tracklet_id``, ``decode_demoted`` and ``schema_version``.
SIDECAR_SCHEMA_VERSION = 2

# Region statuses decided before the model runs.
STATUS_OK = "ok"
STATUS_NO_DATA = "no_data"
STATUS_UNSUPPORTED = "unsupported_tagset"


@dataclass(frozen=True)
class JointConfig:
    """Everything that parametrises the joint engine.

    Attributes:
        identity: Model parameters (threshold, evidence weights, transition costs).
        consistency: Temporal-consistency thresholds.
        tracklets: Tracklet-linking parameters.
        tracklet_mode: ``"built"`` (own linking, SLEAP ids as a hint) or ``"sleap"`` (follow the
            SLEAP ids, split at jumps).
        colony_size: Bees per region. A region with fewer tags gets untagged-bee phantoms up to
            this number (3 tags + 1 untagged bee); more tags than this is unsupported.
        min_tags: Fewest tags a region may have.
    """

    identity: IdentityParams = field(default_factory=IdentityParams)
    consistency: ConsistencyParams = field(default_factory=ConsistencyParams)
    tracklets: TrackletParams = field(default_factory=TrackletParams)
    tracklet_mode: TrackletMode = "built"
    colony_size: int = 4
    min_tags: int = 3


@dataclass
class RegionRun:
    """One region: its tags, its instances and (once solved) its labels.

    Attributes:
        name: Region name (``"all"`` in the flat mode, ``top_left`` etc. for quadrants).
        tags: The region's tag ids, sorted (empty if unknown).
        n_bees: Bees in the region, tags plus untagged bees.
        status: ``ok``, ``no_data`` or ``unsupported_tagset`` (diagnostics refine ``ok``).
        reasons: Why the region is not ``ok``.
        member: ``(F, S)`` mask of the input's instances in this region.
        table: The region's packed instance table (``None`` unless solved).
        source_slot: ``(F, S_r)`` input slot of every region slot (``-1`` = padding).
        votes: Decode votes of the region (tag *indices*).
        extra: Soft votes from external evidence sources.
        labels: The labels (``None`` unless solved).
    """

    name: str
    tags: list[int]
    n_bees: int
    status: str
    reasons: list[str]
    member: NDArray[np.bool_]
    table: InstanceTable | None = None
    source_slot: NDArray[np.int64] | None = None
    votes: Votes | None = None
    extra: Votes | None = None
    labels: RegionLabels | None = None
    config: JointConfig | None = None

    @property
    def solved(self) -> bool:
        """Whether the model ran on this region."""
        return self.labels is not None

    def solve(self, votes: Votes | None = None) -> RegionLabels:
        """Run the model (again) on this region, optionally with replacement decode votes.

        Args:
            votes: Votes to use instead of the region's own (e.g. with held-out tags removed).

        Returns:
            The labels (not stored on the run unless ``votes`` is ``None``).

        Raises:
            RuntimeError: If the region has nothing to solve (``status`` is not ``ok``).
        """
        if self.table is None or self.votes is None or self.config is None:
            msg = f"region {self.name!r} cannot be solved (status {self.status})"
            raise RuntimeError(msg)
        cfg = self.config
        labels = solve_region(
            self.table,
            self.votes if votes is None else votes,
            len(self.tags),
            self.n_bees,
            cfg.identity,
            cfg.consistency,
            tracklet_params=cfg.tracklets,
            tracklet_mode=cfg.tracklet_mode,
            extra=self.extra,
        )
        if votes is None:
            self.labels = labels
        return labels


def region_reasons(labels: RegionLabels) -> NDArray[np.int8]:
    """The :class:`~naps.reasons.Reason` code of every slot of a solved region.

    The codes ``0..8`` coincide with :class:`~naps.identity.SlotFlag`; the unlabelled instances
    flagged ``low_confidence`` are refined by ``labels.low_conf_reason``.

    Args:
        labels: The labels of one region.

    Returns:
        ``(F, S)`` reason codes, 0 where the instance is labelled or the slot is empty.
    """
    reason = labels.slot_flag.astype(np.int8)
    refined = labels.low_conf_reason
    if refined is not None:
        use = (labels.slot_flag == SlotFlag.LOW_CONFIDENCE) & (refined > 0)
        reason = np.where(use, refined, reason)
    return np.where(labels.slot_tag >= 0, 0, reason).astype(np.int8)


@dataclass
class JointResult:
    """Labels of the whole input.

    Attributes:
        table: The input instances.
        regions: The regions by name.
        region_names: Region names; ``slot_region`` indexes into this list.
        slot_region: ``(F, S)`` region index of every instance (``-1`` = in no region).
        slot_tag: ``(F, S)`` tag id of every instance (``-1`` = untracked).
        slot_conf: ``(F, S)`` marginal of the label (``NaN`` where untracked).
        slot_source: ``(F, S)`` :class:`~naps.sources.Source` code.
        slot_disputed: ``(F, S)`` decoded instances whose tag the model confidently disputes.
        slot_flag: ``(F, S)`` :class:`~naps.identity.SlotFlag` of an unlabelled instance.
        frame_size: ``(width, height)`` of the video.
        counters: Decode bookkeeping (decodes seen / out of set / wrong region).
        slot_reason: ``(F, S)`` :class:`~naps.reasons.Reason` code of every valid instance
            without a tag (0 where the instance is labelled or the slot is empty).
        slot_tracklet: ``(F, S)`` tracklet id, unique across the regions (-1 = none).
        slot_demoted: ``(F, S)`` the instance's clean decode was demoted (not an anchor).
        tracklet_offset: Per region, the number added to its own 0-based tracklet ids.
    """

    table: InstanceTable
    regions: dict[str, RegionRun]
    region_names: list[str]
    slot_region: NDArray[np.int64]
    slot_tag: NDArray[np.int64]
    slot_conf: NDArray[np.float32]
    slot_source: NDArray[np.int8]
    slot_disputed: NDArray[np.bool_]
    slot_flag: NDArray[np.int8]
    frame_size: tuple[int, int] | None = None
    counters: Counter[str] = field(default_factory=Counter)
    slot_reason: NDArray[np.int8] = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int8))
    slot_tracklet: NDArray[np.int64] = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.int64)
    )
    slot_demoted: NDArray[np.bool_] = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    tracklet_offset: dict[str, int] = field(default_factory=dict)

    def refresh(self) -> None:
        """Rebuild the per-instance arrays from the regions' labels (after labels changed)."""
        shape = self.slot_tag.shape
        self.slot_tag = np.full(shape, -1, dtype=np.int64)
        self.slot_conf = np.full(shape, np.nan, dtype=np.float32)
        self.slot_source = np.zeros(shape, dtype=np.int8)
        self.slot_disputed = np.zeros(shape, dtype=bool)
        self.slot_flag = np.zeros(shape, dtype=np.int8)
        self.slot_reason = self._unplaced_reasons()
        self.slot_tracklet = np.full(shape, -1, dtype=np.int64)
        self.slot_demoted = np.zeros(shape, dtype=bool)
        self.tracklet_offset = {}
        offset = 0
        for name, run in self.regions.items():
            if run.labels is None or run.source_slot is None:
                continue
            rows, cols = np.nonzero(run.source_slot >= 0)
            dest = run.source_slot[rows, cols]
            lab = run.labels
            tag = lab.slot_tag[rows, cols]
            ids = np.array(run.tags + [-1] * max(run.n_bees - len(run.tags), 0))
            self.slot_tag[rows, dest] = np.where(tag >= 0, ids[np.maximum(tag, 0)], -1)
            self.slot_conf[rows, dest] = lab.slot_conf[rows, cols]
            self.slot_source[rows, dest] = lab.slot_source[rows, cols]
            self.slot_disputed[rows, dest] = lab.slot_disputed[rows, cols]
            self.slot_flag[rows, dest] = lab.slot_flag[rows, cols]
            self.slot_reason[rows, dest] = region_reasons(lab)[rows, cols]
            self.tracklet_offset[name] = offset
            if lab.tracklets is not None:
                tid = lab.tracklets.tid[rows, cols].astype(np.int64)
                self.slot_tracklet[rows, dest] = np.where(tid >= 0, tid + offset, -1)
                offset += lab.tracklets.n_tracklets
            if lab.demoted is not None:
                self.slot_demoted[rows, dest] = lab.demoted[rows, cols]

    def _unplaced_reasons(self) -> NDArray[np.int8]:
        """Reasons of the instances that no solved region labels or abstains on."""
        t = self.table
        reason = np.zeros(self.slot_tag.shape, dtype=np.int8)
        reason[(t.inst_index >= 0) & ~t.valid] = Reason.NO_POSITION
        reason[t.valid & (self.slot_region < 0)] = Reason.OUTSIDE_REGIONS
        for run in self.regions.values():
            if run.labels is None:
                unsupported = run.status == STATUS_UNSUPPORTED
                reason[run.member] = (
                    Reason.REGION_UNSUPPORTED_TAGSET if unsupported else Reason.REGION_NO_DATA
                )
        return reason


class JointMatching:
    """Decode, split into regions, solve.

    Args:
        table: Every instance of frames ``table.first_frame ..`` (see
            :func:`naps.instances.table_from_labels`).
        decode_source: Where the decodes come from.
        closed_set: Flat mode: the tags of the single region.
        quadrants: Quadrant mode: the scope (regions and tag sets, explicit or inferred).
        config: Engine configuration.
        evidence_sources: Extra (soft) evidence, e.g. idtracker.ai continuity.

    Raises:
        ValueError: If neither or both of ``closed_set`` and ``quadrants`` are given, or the
            colony size is outside 1..:data:`~naps.identity.MAX_TAGS`.
    """

    def __init__(  # noqa: PLR0917 (public API: mirrors Matching)
        self,
        table: InstanceTable,
        decode_source: DecodeSource,
        closed_set: Sequence[int] | None = None,
        quadrants: QuadrantScope | None = None,
        config: JointConfig | None = None,
        evidence_sources: Sequence[EvidenceSource] = (),
    ) -> None:
        """Validate the arguments; the work happens in :meth:`match`."""
        if (closed_set is None) == (quadrants is None):
            msg = "the joint engine needs exactly one of closed_set and quadrants"
            raise ValueError(msg)
        self.config = config or JointConfig()
        if not 1 <= self.config.colony_size <= MAX_TAGS:
            msg = f"colony_size must be 1..{MAX_TAGS}, got {self.config.colony_size}"
            raise ValueError(msg)
        self.table = table
        self.decode_source = decode_source
        self.closed_set = (
            None if closed_set is None else sorted({require_tag(t) for t in closed_set})
        )
        self.quadrants = quadrants
        self.evidence_sources = list(evidence_sources)

    # ------------------------------------------------------------------ stages
    def _coords(self) -> dict[int, dict[Hashable, Sequence[float]]]:
        """``coords[frame][slot]`` = tag-node position of every instance, for the decoder."""
        t = self.table
        coords: dict[int, dict[Hashable, Sequence[float]]] = {}
        for row, frame in enumerate(t.frames):
            coords[int(frame)] = {
                int(s): t.tag_xy[row, s] for s in np.flatnonzero(t.inst_index[row] >= 0)
            }
        return coords

    def _positions(self, which: str) -> dict[int, dict[Hashable, Sequence[float]]]:
        """``positions[frame][slot]`` of the body (``"xy"``) or the tag node (``"tag_xy"``)."""
        t = self.table
        array = t.xy if which == "xy" else t.tag_xy
        return {
            int(frame): {int(s): array[row, s] for s in np.flatnonzero(t.valid[row])}
            for row, frame in enumerate(t.frames)
        }

    def _plan(
        self, decodes: Decodes, frame_size: tuple[int, int] | None
    ) -> tuple[list[RegionRun], list[str], NDArray[np.int64]]:
        """Decide the regions, their tag sets and which of them the model can handle.

        Returns:
            ``(runs, region_names, slot_region)``.
        """
        valid = self.table.valid
        scope = self.quadrants
        if scope is None:
            flat = self._plan_region(FLAT_REGION, self.closed_set, valid, decodes)
            return [flat], [FLAT_REGION], np.where(valid, 0, -1).astype(np.int64)
        if frame_size is None and scope.rois is None:
            msg = "quadrant mode needs the video frame size, but no frame could be read"
            raise RuntimeError(msg)
        positions = self._positions("xy") if scope.tag_sets is None else {}
        scope.prepare(frame_size or (0, 0), positions)
        if scope.tag_sets is None:
            scope.infer_from_decodes(decodes)
        names, index = region_index(self.table.xy, scope.rois or {})
        runs = []
        for i, name in enumerate(names):
            tags = scope.tag_set(name)
            reasons = (
                [scope.inference.get("abstained", {}).get(name, "no tag set")]
                if tags is None
                else []
            )
            runs.append(self._plan_region(name, tags, (index == i) & valid, decodes, reasons))
        return runs, names, index

    def _plan_region(
        self,
        name: str,
        tags: Sequence[int] | None,
        member: NDArray[np.bool_],
        decodes: Decodes,
        reasons: list[str] | None = None,
    ) -> RegionRun:
        """Create the :class:`RegionRun` of a region and check that the model can handle it."""
        cfg = self.config
        reasons = list(reasons or [])
        n_tags = 0 if tags is None else len(tags)
        run = RegionRun(
            name, sorted(tags or []), max(n_tags, cfg.colony_size), STATUS_OK, reasons, member,
            config=cfg,
        )  # fmt: skip
        if not member.any():
            run.status, run.reasons = STATUS_NO_DATA, [*reasons, "no instances in the region"]
        elif tags is None:
            run.status = STATUS_UNSUPPORTED
        elif not cfg.min_tags <= n_tags <= cfg.colony_size:
            run.status = STATUS_UNSUPPORTED
            run.reasons.append(
                f"the joint model supports {cfg.min_tags}..{cfg.colony_size} tags per region, "
                f"found {n_tags}: {sorted(tags)}"
            )
        elif (median := _median_count(member)) > cfg.colony_size + _CROWDING_SLACK:
            run.status = STATUS_UNSUPPORTED
            run.reasons.append(
                f"the region holds a median of {median:g} instances per frame but the colony has "
                f"{cfg.colony_size} bees: the closed set assumes the region holds only these "
                "animals (e.g. a quadrant), not a full frame with other colonies"
            )
        elif not self._has_decodes(decodes, member, set(tags)):
            run.status, run.reasons = STATUS_NO_DATA, [*reasons, "no tag was decoded in the region"]
        return run

    def _has_decodes(self, decodes: Decodes, member: NDArray[np.bool_], tags: set[int]) -> bool:
        """Whether any decode of one of ``tags`` lies on an instance of ``member``."""
        first = self.table.first_frame
        return any(
            member[frame - first, slot]
            and any(normalize_tag(e, strict=True) in tags for e in entries)
            for frame, slot, entries in _decoded_slots(decodes)
        )

    def _votes_for(
        self,
        runs: list[RegionRun],
        decodes: Decodes,
        slot_region: NDArray[np.int64],
        counters: Counter[str],
    ) -> dict[str, list[tuple[int, int, int, float]]]:
        """Distribute the decodes over the regions as ``(row, slot, tag index, weight)`` rows."""
        first = self.table.first_frame
        owner = {t: r.name for r in runs for t in r.tags}
        index = {r.name: {t: i for i, t in enumerate(r.tags)} for r in runs}
        rows: dict[str, list[tuple[int, int, int, float]]] = {r.name: [] for r in runs}
        for frame, slot, entries in _decoded_slots(decodes):
            row = frame - first
            home = slot_region[row, slot]
            for entry in entries:
                tag = normalize_tag(entry, strict=True)
                if tag is None:
                    continue
                counters["decodes"] += 1
                if home < 0:
                    counters["decodes_outside_regions"] += 1
                elif tag in index[runs[home].name]:
                    rows[runs[home].name].append((row, slot, index[runs[home].name][tag], 1.0))
                elif tag in owner:
                    counters["decodes_wrong_region"] += 1
                else:
                    counters["decodes_out_of_set"] += 1
        return rows

    def _extra_votes(
        self, decodes: Decodes, runs: list[RegionRun], slot_region: NDArray[np.int64]
    ) -> dict[str, list[tuple[int, int, int, float]]]:
        """Soft votes of the external evidence sources, per region."""
        first = self.table.first_frame
        positions = self._positions("tag_xy")
        aruco: dict[int, dict[Hashable, list[tuple[int, float]]]] = {
            frame: {
                slot: [
                    (t, 1.0) for e in entries if (t := normalize_tag(e, strict=True)) is not None
                ]
                for slot, entries in per_slot.items()
            }
            for frame, per_slot in decodes.items()
        }
        out: dict[str, list[tuple[int, int, int, float]]] = {r.name: [] for r in runs}
        for source in self.evidence_sources:
            source.prepare(positions, aruco)
            for frame in decodes:
                row = frame - first
                votes = source.votes(frame, positions.get(frame, {}), aruco[frame])
                for slot, entries in votes.items():
                    home = slot_region[row, cast("int", slot)]
                    if home < 0:
                        continue
                    index = {t: i for i, t in enumerate(runs[home].tags)}
                    out[runs[home].name].extend(
                        (row, cast("int", slot), index[t], w) for t, w in entries if t in index
                    )
        return out

    @staticmethod
    def _to_votes(rows: list[tuple[int, int, int, float]], inverse: NDArray[np.int64]) -> Votes:
        """Pack vote rows, renumbering the slots into the region table."""
        if not rows:
            return Votes.empty()
        arr = np.array(rows, dtype=float)
        frame, slot = arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64)
        mapped = inverse[frame, slot]
        keep = mapped >= 0
        order = np.lexsort((mapped[keep], frame[keep]))
        return Votes(
            frame[keep][order],
            mapped[keep][order],
            arr[keep, 2].astype(np.int64)[order],
            arr[keep, 3][order],
        )

    def match(self) -> JointResult:
        """Decode the video, solve every region and scatter the labels onto the instances.

        Returns:
            The :class:`JointResult`.

        Raises:
            ValueError: If the instance table has no frames.
            RuntimeError: In quadrant mode, if no video frame could be read (the regions are cut
                from the frame size) and no ROI was given explicitly.
        """
        t = self.table
        if t.n_frames == 0:
            msg = "the instance table has no frames"
            raise ValueError(msg)
        first, last = int(t.frames[0]), int(t.frames[-1])
        decodes, frame_size = self.decode_source.decode(first, last, self._coords())
        runs, region_names, slot_region = self._plan(decodes, frame_size)
        counters: Counter[str] = Counter()
        rows = self._votes_for(runs, decodes, slot_region, counters)
        extra = self._extra_votes(decodes, runs, slot_region) if self.evidence_sources else {}
        for run in runs:
            if run.status != STATUS_OK:
                logger.warning("Region %s: %s (%s)", run.name, run.status, "; ".join(run.reasons))
                continue
            run.table, run.source_slot = t.subset(run.member)
            inverse = np.full(run.member.shape, -1, dtype=np.int64)
            r, c = np.nonzero(run.source_slot >= 0)
            inverse[r, run.source_slot[r, c]] = c
            run.votes = self._to_votes(rows[run.name], inverse)
            run.extra = self._to_votes(extra[run.name], inverse) if extra.get(run.name) else None
            logger.info(
                "Region %s: %s tags, %s bees, %s decodes",
                run.name, len(run.tags), run.n_bees, len(run.votes),
            )  # fmt: skip
            run.solve()
        shape = (t.n_frames, t.n_slots)
        result = JointResult(
            t,
            {r.name: r for r in runs},
            region_names,
            slot_region,
            np.full(shape, -1, dtype=np.int64),
            np.full(shape, np.nan, dtype=np.float32),
            np.zeros(shape, dtype=np.int8),
            np.zeros(shape, dtype=bool),
            np.zeros(shape, dtype=np.int8),
            frame_size,
            counters,
        )
        result.refresh()
        n_label = int((result.slot_tag >= 0).sum())
        logger.info(
            "Joint engine labelled %s of %s instances (decoded: %s)",
            n_label, int(t.valid.sum()), int((result.slot_source == Source.DECODE).sum()),
        )  # fmt: skip
        return result


def qualify_tag(session: str | None, tag: int) -> str:
    """The session-qualified name of a tag (``"<session>:ArUcoTag#<tag>"``).

    Tag ids are reused across colonies and trials, so the unqualified ``ArUcoTag#<int>`` track name
    is not unique across videos; the qualified name is.
    """
    return f"{session}:{tag_name(tag)}" if session else tag_name(tag)


def write_sidecar(
    path: str | os.PathLike[str],
    result: JointResult,
    session: str | None = None,
    meta: Mapping[str, object] | None = None,
) -> None:
    """Write the per-frame confidence / source sidecar of a joint run as a compressed ``.npz``.

    Per-instance arrays have shape ``(frames, slots)`` and are indexed like the instance table
    (``inst_index`` maps a slot to the instance's position in its labeled frame): ``tag`` (tag id,
    -1 = untracked), ``confidence``, ``source`` (see ``source_names``), ``disputed``, ``flag``
    (see ``flag_names``), ``region`` (index into ``region_names``), ``reason`` (why a valid
    instance has no tag, see ``reason_names``), ``tracklet_id`` (-1 = none; unique across
    regions) and ``decode_demoted``. Per region ``<r>`` there are ``tags__<r>``,
    ``tags_qualified__<r>``, ``status__<r>``, ``tracklet_offset__<r>``, and per tag and frame
    ``tag_xy__<r>`` (also the interpolated positions), ``tag_source__<r>``,
    ``tag_confidence__<r>`` and ``tag_slot__<r>`` (input slot carrying the tag, -1 = none or
    interpolated). ``schema_version`` is :data:`SIDECAR_SCHEMA_VERSION`; the full layout is in the
    module docstring.

    Args:
        path: Output file.
        result: The joint result.
        session: Session key that qualifies the tag ids (tag ids repeat across colonies/trials).
        meta: Extra JSON-serialisable metadata (options, versions).
    """
    t = result.table
    arrays: dict[str, NDArray] = {
        "frames": t.frames,
        "inst_index": t.inst_index,
        "tag": result.slot_tag,
        "confidence": result.slot_conf,
        "source": result.slot_source,
        "disputed": result.slot_disputed,
        "flag": result.slot_flag,
        "region": result.slot_region,
        "reason": result.slot_reason,
        "tracklet_id": result.slot_tracklet,
        "decode_demoted": result.slot_demoted,
        "region_names": np.array(result.region_names),
        "source_names": np.array(SOURCE_NAMES),
        "flag_names": np.array(SlotFlag.NAMES),
        "reason_names": np.array(REASON_NAMES),
        "schema_version": np.array(SIDECAR_SCHEMA_VERSION),
        "session": np.array(session or ""),
    }
    for name, run in result.regions.items():
        arrays[f"tags__{name}"] = np.array(run.tags, dtype=np.int64)
        arrays[f"tags_qualified__{name}"] = np.array([qualify_tag(session, x) for x in run.tags])
        arrays[f"status__{name}"] = np.array(run.status)
        arrays[f"tracklet_offset__{name}"] = np.array(result.tracklet_offset.get(name, -1))
        lab = run.labels
        if lab is None or run.source_slot is None:
            continue
        arrays[f"tag_xy__{name}"] = lab.tag_xy
        arrays[f"tag_source__{name}"] = lab.tag_src
        arrays[f"tag_confidence__{name}"] = lab.tag_conf
        in_slot = np.where(
            lab.tag_slot >= 0,
            np.take_along_axis(run.source_slot, np.maximum(lab.tag_slot, 0), axis=1),
            -1,
        )
        arrays[f"tag_slot__{name}"] = in_slot
    arrays["meta"] = np.array(json.dumps(dict(meta or {}), default=str))
    np.savez_compressed(path, **cast("dict[str, Any]", arrays))
