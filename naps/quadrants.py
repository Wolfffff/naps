"""Quadrant-aware closed-set identities.

Full-frame recordings can hold several independent colonies side by side (for
example four 1832x1832 tiles in a 3664x3664 frame, 4 tagged bees each). The flat
``--closed-set`` assumes the whole input is one colony. Here every tracked
instance is assigned to a region (quadrant) by position, each region has its own
tag set, and closed-set elimination runs per region:

* :class:`QuadrantScope` holds the regions (a grid of the frame size or an ROI
  list) and the per-region tag sets, given explicitly or inferred from the
  ArUco decodes of the video (:meth:`QuadrantScope.infer_from_decodes`).
* :meth:`QuadrantScope.vote_filter` drops votes for tags that do not belong to
  the region a track is in (a tag decoded in the wrong quadrant) and counts them.
* :func:`eliminate_by_quadrant` runs :func:`naps.closed_set.eliminate_identities`
  once per region on that region's tracks only.
* :func:`build_report` produces the per-video diagnostics written by
  ``--report-json``.

Everything is opt-in. A region without a tag set (explicit map that omits it, or
inference that abstained) is left in legacy behaviour: no elimination, and its
votes are only filtered for tags owned by *another* region (no tag-set check).
"""

import json
import logging
import math
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from naps.closed_set import EliminationResult, eliminate_identities
from naps.cost_matrix import MatchedDict
from naps.tags import normalize_tag, require_tag

__all__ = [
    "GRID_NAMES_2X2",
    "QuadrantElimination",
    "QuadrantScope",
    "build_report",
    "eliminate_by_quadrant",
    "locate",
    "make_grid",
    "parse_roi_args",
    "parse_tag_map",
    "region_index",
]

logger = logging.getLogger(__name__)

Roi = tuple[float, float, float, float]  # (x0, y0, x1, y1)
Position = Sequence[float]  # (x, y)
Decodes = Mapping[int, Mapping[Hashable, Sequence[Hashable | None]]]  # decodes[frame][track]
Assignments = Mapping[int, Mapping[Hashable, Hashable]]  # matched[frame][track] -> tag
Positions = Mapping[int, Mapping[Hashable, Position]]  # positions[frame][track] -> (x, y)
Instances = Mapping[int, Sequence[tuple[Hashable | None, Position | None]]]
Untracked = Mapping[int, Iterable[Position | None]]  # untracked[frame] -> positions

GRID_NAMES_2X2 = {
    (0, 0): "top_left",
    (0, 1): "top_right",
    (1, 0): "bottom_left",
    (1, 1): "bottom_right",
}
INF = float("inf")

# Tags per region the report histograms assume when the region has no tag set (the colonies here
# hold 4 bees); the histogram then covers the largest count actually observed if that is larger.
_DEFAULT_REGION_SIZE = 4
# Histograms of up to this many tags list every bin; larger ones list only non-empty bins.
_DENSE_HISTOGRAM_MAX = 8
# At most this many eliminated instances per region are listed in the report.
_ASSIGNED_SAMPLE_LIMIT = 100


class _ExcludedBase(TypedDict):
    reason: str  # multiple_quadrants | impure | low_support
    support: dict[str, int]  # region -> (frame, track) decodes of the tag there


class _Excluded(_ExcludedBase, total=False):
    purity: float  # only for reason == "impure"


# --------------------------------------------------------------------------- regions


def make_grid(width: float, height: float, rows: int = 2, cols: int = 2) -> dict[str, Roi]:
    """Return the regions ``name -> (x0, y0, x1, y1)`` of a ``rows`` x ``cols`` grid of the frame.

    Regions are half-open, ``x0 <= x < x1``. The outer edges are infinite so a
    prediction a few pixels outside the frame still falls in the nearest tile.
    A 2x2 grid uses the production names (top_left, top_right, bottom_left,
    bottom_right); otherwise regions are called ``r<row>c<col>``.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        rows: Number of grid rows.
        cols: Number of grid columns.

    Raises:
        ValueError: If ``rows`` or ``cols`` is less than 1.
    """
    if rows < 1 or cols < 1:
        msg = "grid needs at least 1 row and 1 column"
        raise ValueError(msg)
    xs = [-INF, *(width * j / cols for j in range(1, cols)), INF]
    ys = [-INF, *(height * i / rows for i in range(1, rows)), INF]
    rois = {}
    for i in range(rows):
        for j in range(cols):
            name = GRID_NAMES_2X2[(i, j)] if (rows, cols) == (2, 2) else f"r{i}c{j}"
            rois[name] = (xs[j], ys[i], xs[j + 1], ys[i + 1])
    return rois


def locate(pos: Position | None, rois: Mapping[str, Roi]) -> str | None:
    """Return the name of the region containing ``pos`` (x, y).

    Returns:
        The region name, or None if ``pos`` is missing/NaN/malformed, outside every region or
        inside several overlapping ones (ambiguous).
    """
    if pos is None:
        return None
    xy = np.asarray(pos, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        return None
    hits = [
        name for name, (x0, y0, x1, y1) in rois.items() if x0 <= xy[0] < x1 and y0 <= xy[1] < y1
    ]
    return hits[0] if len(hits) == 1 else None


def region_index(
    xy: NDArray[np.float64], rois: Mapping[str, Roi]
) -> tuple[list[str], NDArray[np.int64]]:
    """Vectorised :func:`locate` for an array of positions.

    Args:
        xy: ``(..., 2)`` positions (NaN positions belong to no region).
        rois: ``{region: (x0, y0, x1, y1)}``.

    Returns:
        ``(names, index)``: the region names (in ``rois`` order) and, for every position, the
        index of its region in ``names``, or -1 if it is outside every region, inside several
        overlapping ones, or not finite.
    """
    names = list(rois)
    finite = np.isfinite(xy).all(axis=-1)
    hits = np.zeros((*xy.shape[:-1], len(names)), dtype=bool)
    with np.errstate(invalid="ignore"):
        for i, (x0, y0, x1, y1) in enumerate(rois.values()):
            hits[..., i] = (xy[..., 0] >= x0) & (xy[..., 0] < x1)
            hits[..., i] &= (xy[..., 1] >= y0) & (xy[..., 1] < y1)
    unique = finite & (hits.sum(axis=-1) == 1)
    return names, np.where(unique, hits.argmax(axis=-1), -1).astype(np.int64)


# ------------------------------------------------------------------- parsing user input


def _parse_tag_list(values: Iterable[object], where: str) -> list[int]:
    """Parse tag ids, rejecting non-integral values and duplicates."""
    tags = []
    for v in values:
        tag = normalize_tag(v, strict=True)
        if tag is None:
            msg = f"{where}: {v!r} is not a tag id"
            raise ValueError(msg)
        tags.append(tag)
    if len(set(tags)) != len(tags):
        msg = f"{where}: duplicate tags in {tags}"
        raise ValueError(msg)
    return tags


def _is_file(spec: str) -> bool:
    """Whether ``spec`` names an existing file (False, not an error, for long/odd strings)."""
    try:
        return Path(spec).is_file()
    except (OSError, ValueError):  # e.g. an inline map longer than the file-name limit
        return False


def _read_map_file(spec: str) -> tuple[dict[str, list[int]], dict[str, Roi] | None]:
    """Read a ``.json`` / ``.yaml`` / ``.yml`` closed-set map."""
    path = Path(spec)
    with path.open() as handle:
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml  # noqa: PLC0415 (optional dependency, only needed for YAML maps)

            data = yaml.safe_load(handle)
        else:
            data = json.load(handle)
    if not isinstance(data, dict):
        msg = f"{spec}: expected a mapping of region -> tags"
        raise ValueError(msg)  # noqa: TRY004 (callers treat ValueError as a user-input error)
    rois: dict[str, Roi] | None = None
    if "tags" in data and isinstance(data["tags"], dict):
        raw = data["tags"]
        if "rois" in data:
            rois = {}
            for k, v in data["rois"].items():
                coords = tuple(float(x) for x in v)
                if len(coords) != 4:
                    msg = f"{spec}: roi {k} needs x0, y0, x1, y1"
                    raise ValueError(msg)
                rois[str(k)] = (coords[0], coords[1], coords[2], coords[3])
    else:
        raw = data
    return {str(k): _parse_tag_list(v, f"region {k}") for k, v in raw.items()}, rois


def _parse_inline_map(spec: str) -> dict[str, list[int]]:
    """Parse ``"top_left:45,46,47,48;top_right:..."``.

    Raises:
        ValueError: If an entry is not ``name:tag,tag,...``, a tag is invalid or repeated within a
            region, or a region name is given twice.
    """
    sets = {}
    for chunk in spec.split(";"):
        entry = chunk.strip()
        if not entry:
            continue
        if ":" not in entry:
            msg = (
                f"bad --closed-set-map entry {entry!r}: expected 'name:tag,tag,...' "
                "(or a path to an existing .json/.yaml file)"
            )
            raise ValueError(msg)
        name, tags = entry.split(":", 1)
        name = name.strip()
        if name in sets:
            msg = f"region {name!r} is given twice in --closed-set-map"
            raise ValueError(msg)
        tag_list = [t for t in tags.replace(" ", "").split(",") if t]
        sets[name] = _parse_tag_list(tag_list, f"region {name}")
    return sets


def parse_tag_map(spec: str) -> tuple[dict[str, list[int]], dict[str, Roi] | None]:
    """Parse ``--closed-set-map``.

    Args:
        spec: Either an inline string ``"top_left:45,46,47,48;top_right:..."`` or the path of a
            ``.json`` / ``.yaml`` / ``.yml`` file. The file holds either ``{name: [tags]}`` or
            ``{"tags": {name: [tags]}, "rois": {name: [x0, y0, x1, y1]}}``.

    Returns:
        ``(tag_sets, rois_or_None)``. Tag sets of different regions are disjoint.

    Raises:
        ValueError: If the map is empty or malformed, a tag is invalid or listed twice (within
            or across regions), an inline map names a region twice, or a roi does not have four
            coordinates.
    """
    if _is_file(spec):
        sets, rois = _read_map_file(spec)
    else:
        sets, rois = _parse_inline_map(spec), None
    if not sets:
        msg = "--closed-set-map is empty"
        raise ValueError(msg)
    seen: dict[int, str] = {}
    for name, tags in sets.items():
        for tag in tags:
            if tag in seen:
                msg = (
                    f"tag {tag} is listed in both {seen[tag]!r} and {name!r}: "
                    "a tag belongs to one region"
                )
                raise ValueError(msg)
            seen[tag] = name
    return sets, rois


def parse_roi_args(roi_args: Iterable[Sequence[str]] | None) -> dict[str, Roi] | None:
    """Parse ``--quadrant-roi NAME X0 Y0 X1 Y1`` (repeatable).

    Args:
        roi_args: One ``[name, x0, y0, x1, y1]`` list per occurrence of the flag.

    Returns:
        ``{name: (x0, y0, x1, y1)}``, or None if there are no arguments.

    Raises:
        ValueError: If a rectangle is empty or has other than four coordinates, or a name repeats.
    """
    if not roi_args:
        return None
    rois: dict[str, Roi] = {}
    for entry in roi_args:
        name, *coords = entry
        if len(coords) != 4:
            msg = f"roi {name}: expected X0 Y0 X1 Y1, got {len(coords)} values"
            raise ValueError(msg)
        x0, y0, x1, y1 = (float(c) for c in coords)
        if x1 <= x0 or y1 <= y0:
            msg = f"roi {name}: empty rectangle"
            raise ValueError(msg)
        if name in rois:
            msg = f"roi {name} given twice"
            raise ValueError(msg)
        rois[name] = (x0, y0, x1, y1)
    return rois


# ----------------------------------------------------------------------------- scope


@dataclass
class QuadrantScope:
    """Regions plus per-region tag sets for quadrant-aware closed-set identity.

    Args:
        tag_sets: Explicit ``{region: tags}``. Omit (and set ``infer``) to infer from data.
        rois: Explicit ``{region: (x0, y0, x1, y1)}``. Default: ``grid`` of the frame size.
        grid: ``(rows, cols)`` of the default grid.
        infer: Infer tag sets from the decodes (see :meth:`infer_from_decodes`).
        expected_size: Tags expected per region when inferring (4 for the colonies here).
        min_support: A tag is attributed to a region only if it was decoded in at least
            ``max(min_support, min_fraction * frames_with_instances)`` (frame, track) decodes of
            that region; a tag decoded on two tracks of the region in one frame counts twice.
        min_fraction: See ``min_support``.
        min_purity: Fraction of a tag's decode frames that must fall in its region.
        dominance_ratio: With more than ``expected_size`` candidates in a region, keep the top
            ``expected_size`` only if the weakest kept has this many times the support of the
            strongest rejected; otherwise abstain on the region.
        min_size: If given, a region with ``min_size..expected_size`` tags is accepted (default:
            exactly ``expected_size``). The joint engine uses 3: a quadrant with three tagged bees
            and one untagged bee is a closed set of three plus one.

    Attributes:
        active: ``{region: sorted tags}`` of the regions that currently have a tag set.
        frame_size: ``(width, height)`` given to :meth:`prepare`.
        positions: ``positions[frame][track] -> (x, y)`` given to :meth:`prepare`.
        inference: Diagnostics of the last :meth:`infer_from_decodes` (empty otherwise).
        rejected: Counts of votes rejected by :meth:`vote_filter`, by cause.
        wrong_quadrant: Counts of ``(tag, seen_in, belongs_to)`` for tags decoded in the wrong
            region.

    Raises:
        ValueError: If neither ``tag_sets`` nor ``infer`` is given, or ``tag_sets`` holds an
            invalid tag (``None``, NaN, non-integral, non-numeric; see
            :func:`naps.tags.require_tag`).
    """

    tag_sets: dict[str, list[int]] | None = None
    rois: dict[str, Roi] | None = None
    grid: tuple[int, int] = (2, 2)
    infer: bool = False
    expected_size: int = 4
    min_support: int = 30
    min_fraction: float = 0.01
    min_purity: float = 0.9
    dominance_ratio: float = 3.0
    min_size: int | None = None

    # filled in during a run
    frame_size: tuple[float, float] | None = field(default=None, init=False)
    positions: Positions | None = field(default=None, init=False, repr=False)
    inference: dict[str, Any] = field(default_factory=dict, init=False)
    rejected: Counter[str] = field(default_factory=Counter, init=False)
    wrong_quadrant: Counter[tuple[int, str, str]] = field(default_factory=Counter, init=False)

    def __post_init__(self) -> None:
        """Validate the arguments and activate explicit tag sets."""
        if self.tag_sets is None and not self.infer:
            msg = "QuadrantScope needs explicit tag_sets or infer=True"
            raise ValueError(msg)
        self._cache: dict[tuple[int, Hashable], str | None] = {}
        self.active: dict[str, list[int]] = {}
        self._owner: dict[int, str] = {}
        if self.tag_sets is not None:
            self._set_active({k: list(v) for k, v in self.tag_sets.items()})

    # -- setup
    def _set_active(self, sets: Mapping[str, Iterable[int]]) -> None:
        self.active = {k: sorted(require_tag(t) for t in v) for k, v in sets.items()}
        self._owner = {t: q for q, tags in self.active.items() for t in tags}

    def prepare(self, frame_size: Sequence[float], positions: Positions) -> None:
        """Fix the regions (from the frame size unless ROIs were given) and the positions.

        Args:
            frame_size: ``(width, height)`` of the video frames.
            positions: ``positions[frame][track] -> (x, y)`` used to place tracks in regions.

        Raises:
            ValueError: If an explicit tag set names a region that does not exist.
        """
        self.frame_size = (float(frame_size[0]), float(frame_size[1]))
        if self.rois is None:
            self.rois = make_grid(*self.frame_size, *self.grid)
        self.positions = positions
        self._cache.clear()
        if self.tag_sets is not None:
            unknown = sorted(set(self.tag_sets) - set(self.rois))
            if unknown:
                msg = (
                    f"closed-set regions {unknown} are not among the quadrant regions "
                    f"{sorted(self.rois)}"
                )
                raise ValueError(msg)

    # -- lookup
    @property
    def regions(self) -> list[str]:
        """Names of the regions (empty before :meth:`prepare` unless ROIs were given)."""
        return list(self.rois or {})

    def quadrant_of(self, frame: int, track: Hashable) -> str | None:
        """Return the region of ``track`` in ``frame`` (None if it cannot be placed).

        A track cannot be placed if it has no usable position, lies outside every region or
        inside overlapping ones, or :meth:`prepare` has not run yet.
        """
        if self.rois is None:
            return None
        key = (frame, track)
        if key not in self._cache:
            pos = (self.positions or {}).get(frame, {}).get(track)
            self._cache[key] = locate(pos, self.rois)
        return self._cache[key]

    def tag_set(self, quadrant: str) -> list[int] | None:
        """Return the tag set of ``quadrant`` (None if the region is in legacy mode)."""
        return self.active.get(quadrant)

    def owner(self, tag: int) -> str | None:
        """Return the region that owns ``tag`` (None if no region lists it)."""
        return self._owner.get(tag)

    def vote_filter(self, frame: int, track: Hashable, tag: Hashable) -> bool:
        """Return whether a vote for ``tag`` on ``track`` is consistent with its quadrant.

        Rejected votes are counted in :attr:`rejected` (and :attr:`wrong_quadrant`), except that an
        invalid tag (``None``, NaN, non-integral) is rejected without being counted. A track
        outside every region, a tag owned by another region and (in a region with a tag set) a
        tag outside that set are rejected. A region without a tag set only rejects tags owned by
        another region.
        """
        norm = normalize_tag(tag, strict=True)
        if norm is None:
            return False
        q = self.quadrant_of(frame, track)
        if q is None:
            self.rejected["track_outside_regions"] += 1
            return False
        tags = self.active.get(q)
        if tags is None:  # region left in legacy mode
            owner = self._owner.get(norm)
            if owner is not None and owner != q:
                self.rejected["wrong_quadrant"] += 1
                self.wrong_quadrant[(norm, q, owner)] += 1
                return False
            return True
        if norm in tags:
            return True
        owner = self._owner.get(norm)
        if owner is not None:
            self.rejected["wrong_quadrant"] += 1
            self.wrong_quadrant[(norm, q, owner)] += 1
        else:
            self.rejected["not_in_quadrant_set"] += 1
        return False

    # -- inference
    def _count_support(self, decodes: Decodes) -> tuple[dict[str, Counter[int]], Counter[str]]:
        """Count, per region, the decodes of each tag, and the frames with tracks.

        A tag counts once per (frame, track) in which it was decoded (not once per frame).
        """
        support: defaultdict[str, Counter[int]] = defaultdict(Counter)
        frames_with_instances: Counter[str] = Counter()
        for frame, tracks in decodes.items():
            seen_q = set()
            for track, tag_list in tracks.items():
                q = self.quadrant_of(frame, track)
                if q is None:
                    continue
                seen_q.add(q)
                normalised = (normalize_tag(t, strict=True) for t in tag_list)
                for tag in {t for t in normalised if t is not None}:
                    support[q][tag] += 1
            frames_with_instances.update(seen_q)
        return support, frames_with_instances

    def _classify_tags(
        self, support: Mapping[str, Counter[int]], thr: Mapping[str, float]
    ) -> tuple[dict[str, list[tuple[int, int]]], dict[int, _Excluded]]:
        """Split tags into per-region candidates and excluded ones (with the reason)."""
        total: Counter[int] = Counter()
        for q in support:
            total.update(support[q])

        candidates: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        excluded: dict[int, _Excluded] = {}
        for tag in sorted(total):
            strong = [q for q in support if support[q][tag] >= thr[q]]
            seen_in = {x: support[x][tag] for x in support if support[x][tag]}
            if len(strong) > 1:
                excluded[tag] = {
                    "reason": "multiple_quadrants",
                    "support": {q: support[q][tag] for q in strong},
                }
            elif len(strong) == 1:
                q = strong[0]
                purity = support[q][tag] / total[tag]
                if purity < self.min_purity:
                    excluded[tag] = {
                        "reason": "impure",
                        "purity": round(purity, 3),
                        "support": seen_in,
                    }
                else:
                    candidates[q].append((tag, support[q][tag]))
            else:
                excluded[tag] = {"reason": "low_support", "support": seen_in}
        return candidates, excluded

    def _decide_sets(
        self,
        candidates: Mapping[str, list[tuple[int, int]]],
        excluded: Mapping[int, _Excluded],
    ) -> tuple[dict[str, list[int]], dict[str, str]]:
        """Pick a tag set per region, or record why the region abstains."""
        sets: dict[str, list[int]] = {}
        abstained: dict[str, str] = {}
        n = self.expected_size
        n_min = n if self.min_size is None else self.min_size
        for q in self.regions:
            cand = sorted(candidates.get(q, []), key=lambda t: -t[1])
            tainted = sorted(
                t
                for t, e in excluded.items()
                if e["reason"] in ("multiple_quadrants", "impure") and q in e["support"]
            )
            blockers = sorted(
                t
                for t, e in excluded.items()
                if e["reason"] == "multiple_quadrants" and q in e["support"]
            )
            if blockers:
                # a well-supported tag that other quadrants also show: this region may hold
                # a bee we cannot attribute, so its closed set is not trustworthy
                abstained[q] = f"ambiguous_tags_in_multiple_quadrants {blockers}"
            elif n_min <= len(cand) <= n:
                sets[q] = [t for t, _ in cand]
            elif len(cand) < n_min:
                abstained[q] = f"fewer_than_{n_min}_tags (found {len(cand)}" + (
                    f"; ambiguous tags seen here: {tainted})" if tainted else ")"
                )
            elif cand[n - 1][1] >= self.dominance_ratio * cand[n][1]:
                sets[q] = [t for t, _ in cand[:n]]
            else:
                abstained[q] = f"more_than_{n}_tags_without_clear_{n}"
        return sets, abstained

    def infer_from_decodes(self, decodes: Decodes) -> dict[str, Any]:
        """Infer the per-region tag sets from raw ArUco decodes.

        For every tag we count the (frame, track) pairs in which it was decoded on a track located
        in each region (once per track, not once per frame). A tag is attributed to a region only
        if its support there reaches ``max(min_support, min_fraction * frames)``, it is not
        strongly supported in any other region (otherwise it is refused everywhere and flagged),
        and at least ``min_purity`` of its decodes lie in that region. A region gets a tag set only
        if it ends up with ``min_size..expected_size`` tags (``min_size`` defaults to
        ``expected_size``), or with more than ``expected_size`` candidates of which the top
        ``expected_size`` clearly dominate (then exactly ``expected_size`` tags are kept); fewer or
        ambiguous -> abstain, region stays in legacy mode.

        Call :meth:`prepare` first: before it no track can be placed in a region, so nothing is
        inferred (and without explicit ROIs there are no regions, hence not even an abstention
        is reported).

        Args:
            decodes: ``decodes[frame][track] -> [tag | None, ...]``.

        Returns:
            The diagnostics (also stored in :attr:`inference`): thresholds, per-region support,
            excluded tags with reasons, the inferred sets and the abstained regions.
        """
        support, frames_with_instances = self._count_support(decodes)
        thr = {
            q: max(self.min_support, self.min_fraction * frames_with_instances[q])
            for q in self.regions
        }
        candidates, excluded = self._classify_tags(support, thr)
        sets, abstained = self._decide_sets(candidates, excluded)
        self.inference = {
            "thresholds": {q: float(v) for q, v in thr.items()},
            "support": {
                q: {int(t): int(c) for t, c in sorted(s.items())} for q, s in support.items()
            },
            "excluded_tags": {int(t): e for t, e in excluded.items()},
            "inferred": {q: [int(t) for t in sorted(v)] for q, v in sets.items()},
            "abstained": abstained,
        }
        for q, reason in abstained.items():
            logger.warning("Tag inference abstained for %s: %s", q, reason)
        for q, tags in sets.items():
            logger.info("Inferred tag set for %s: %s", q, sorted(tags))
        # a tag excluded for multiple quadrants must not be re-used by the inferred sets
        self._set_active(sets)
        return self.inference

    def union_tags(self) -> set[int] | None:
        """Return all tags of all active regions (None if no region has a tag set)."""
        return set(self._owner) if self.active else None


# ---------------------------------------------------------------------- elimination


@dataclass
class QuadrantElimination:
    """Outcome of :func:`eliminate_by_quadrant`.

    Attributes:
        matched: ``matched[frame][track] -> tag`` after elimination.
        per_quadrant: The :class:`~naps.closed_set.EliminationResult` of every active region.
        n_assigned: Instances assigned by elimination, over all regions.
        abstained: Frames per abstention reason, over all regions.
        dropped_foreign: Tags dropped for not belonging to a region's set, over all regions.
        abstained_frames: ``{(frame, region): abstention reason}``.
    """

    matched: MatchedDict
    per_quadrant: dict[str, EliminationResult]
    n_assigned: int = 0
    abstained: Counter[str] = field(default_factory=Counter)
    dropped_foreign: int = 0
    abstained_frames: dict[tuple[int, str], str] = field(default_factory=dict)


def _require_rois(scope: QuadrantScope) -> dict[str, Roi]:
    """Return the regions of ``scope``, which must have been prepared."""
    if scope.rois is None:
        msg = "QuadrantScope.prepare() must be called before the regions are used"
        raise RuntimeError(msg)
    return scope.rois


def _bucket_untracked(
    scope: QuadrantScope,
    untracked_positions: Untracked | None,
    untracked_counts: Mapping[int, int] | None,
) -> dict[str, Counter[int]]:
    """Return ``{region: {frame: n}}``; an instance whose region is unknown blocks every region."""
    out: dict[str, Counter[int]] = {q: Counter() for q in scope.active}
    if untracked_positions is not None:
        rois = _require_rois(scope)
        for frame, plist in untracked_positions.items():
            for pos in plist:
                q = locate(pos, rois)
                for target in [q] if q is not None else list(scope.active):
                    if target in out:
                        out[target][frame] += 1
    elif untracked_counts:
        for frame, n in untracked_counts.items():
            for counter in out.values():
                counter[frame] += n
    return out


def _pass_through_unplaceable(
    scope: QuadrantScope,
    out: QuadrantElimination,
    untracked: Mapping[str, Counter[int]],
    *,
    frame: int,
    track: Hashable,
    raw_tag: Hashable,
) -> None:
    """Handle a tracked instance that cannot be placed in any region.

    Its tag (if it is one we know of) is kept as the cost matrix gave it, and elimination is
    blocked in this frame in every region: its tag would otherwise look missing to its own
    region and be handed to a different bee. A tag that no region owns is dropped (counted as
    foreign) unless some region is in legacy mode (has no tag set), in which case it is kept.
    """
    tag = normalize_tag(raw_tag, strict=True)
    if tag is not None:
        if scope.owner(tag) is not None or len(scope.active) < len(scope.regions):
            out.matched[frame][track] = tag
        else:
            out.dropped_foreign += 1
    for target in scope.active:
        untracked[target][frame] += 1


def _split_by_region(
    matched: Assignments,
    positions: Positions,
    scope: QuadrantScope,
    untracked: Mapping[str, Counter[int]],
    out: QuadrantElimination,
) -> tuple[
    dict[str, dict[int, dict[Hashable, Hashable]]],
    dict[str, dict[int, dict[Hashable, Position]]],
]:
    """Split the tracks of every frame by region; place the ones that bypass elimination.

    Tracks in a region with a tag set go to the returned per-region ``matched``/``positions``.
    Tracks in a legacy region, and tracks that cannot be placed, are written straight to
    ``out.matched`` (normalised); unplaceable tracks also block elimination in their frame.
    """
    sub_matched: dict[str, dict[int, dict[Hashable, Hashable]]] = {q: {} for q in scope.active}
    sub_pos: dict[str, dict[int, dict[Hashable, Position]]] = {q: {} for q in scope.active}
    for frame in sorted(set(matched) | set(positions)):
        m_row = matched.get(frame, {})
        p_row = positions.get(frame, {})
        for track in dict.fromkeys(chain(m_row, p_row)):  # first-seen order: deterministic
            q = scope.quadrant_of(frame, track)
            if q is None:  # no usable position, or overlapping ROIs
                _pass_through_unplaceable(
                    scope, out, untracked, frame=frame, track=track, raw_tag=m_row.get(track)
                )
                continue
            if q not in scope.active:  # legacy region: pass through, normalised
                tag = normalize_tag(m_row.get(track), strict=True)
                if tag is not None:
                    out.matched[frame][track] = tag
                continue
            if track in m_row:
                sub_matched[q].setdefault(frame, {})[track] = m_row[track]
            if track in p_row:
                sub_pos[q].setdefault(frame, {})[track] = p_row[track]
    return sub_matched, sub_pos


def eliminate_by_quadrant(  # noqa: PLR0917 (public API)
    matched: Assignments,
    positions: Positions,
    scope: QuadrantScope,
    max_jump: float = 120.0,
    max_look: int = 25,
    untracked_positions: Untracked | None = None,
    untracked_counts: Mapping[int, int] | None = None,
    max_distance: float | None = None,
) -> QuadrantElimination:
    """Run closed-set elimination per region.

    For every region with a tag set, only the tracks located in that region in a frame take
    part (so a missing tag is matched to an unassigned instance of the same colony). Tags not in
    the region's set are dropped from its tracks. Regions without a tag set (legacy mode) get no
    elimination and no tag-set filtering: their assignments are only re-normalised to plain
    ``int`` tags (``None`` and non-integral values are dropped).

    Args:
        matched: ``matched[frame][track] -> tag`` from the cost matrix. Not modified.
        positions: ``positions[frame][track] -> (x, y)``.
        scope: A prepared :class:`QuadrantScope`.
        max_jump: See :func:`naps.closed_set.eliminate_identities`.
        max_look: See :func:`naps.closed_set.eliminate_identities`.
        untracked_positions: ``{frame: [(x, y) | None]}`` of SLEAP instances without a track;
            takes precedence over ``untracked_counts``.
        untracked_counts: ``{frame: n}`` of SLEAP instances without a track, counted against
            every region (their region is unknown).
        max_distance: See :func:`naps.closed_set.eliminate_identities`.

    Returns:
        A :class:`QuadrantElimination`.

    Raises:
        RuntimeError: If ``untracked_positions`` is given but ``scope`` was never prepared.
    """
    result_matched: MatchedDict = defaultdict(lambda: defaultdict(str))
    out = QuadrantElimination(result_matched, {})
    untracked = _bucket_untracked(scope, untracked_positions, untracked_counts)
    sub_matched, sub_pos = _split_by_region(matched, positions, scope, untracked, out)

    for q, tags in scope.active.items():
        res = eliminate_identities(
            sub_matched[q],
            sub_pos[q],
            tags,
            max_jump=max_jump,
            max_look=max_look,
            untracked_counts=untracked.get(q, {}),
            max_distance=max_distance,
        )
        out.per_quadrant[q] = res
        out.n_assigned += len(res.assigned)
        out.dropped_foreign += res.dropped_foreign
        out.abstained.update(res.abstained)
        for frame, reason in res.abstained_frames.items():
            out.abstained_frames[(frame, q)] = reason
        for frame, row in res.matched.items():
            for track, tag in row.items():
                result_matched[frame][track] = tag
    return out


# ------------------------------------------------------------------------------ report


def _histogram(counter: Counter[int], n_max: int) -> dict[str, int]:
    """Histogram of ``counter`` over ``n_max..0`` (dense for small sets, else non-empty bins)."""
    return {
        str(k): int(counter.get(k, 0))
        for k in range(n_max, -1, -1)
        if n_max <= _DENSE_HISTOGRAM_MAX or counter.get(k, 0)
    }


class _ReportContext:
    """Region bookkeeping shared by the report builders."""

    def __init__(
        self,
        scope: QuadrantScope | None,
        closed_set: Iterable[int] | None,
        elimination: QuadrantElimination | EliminationResult | None,
    ) -> None:
        self.scope = scope
        self.elimination = elimination
        self.regions: list[str]
        self.tag_sets: dict[str, list[int] | None]
        if scope is not None:
            self.regions = scope.regions
            self.tag_sets = {q: scope.active.get(q) for q in self.regions}
        else:
            self.regions = ["all"]
            self.tag_sets = {
                "all": sorted(require_tag(t) for t in closed_set) if closed_set else None
            }

    def region_of(self, frame: int, track: Hashable | None, pos: Position | None) -> str | None:
        """Region of an instance: by its track if it has one, else by its position."""
        if self.scope is None:
            return "all"
        if track is not None:
            q = self.scope.quadrant_of(frame, track)
            if q is not None:
                return q
        return locate(pos, _require_rois(self.scope)) if pos is not None else None

    def abstain_reason(self, frame: int, q: str) -> str | None:
        """Why elimination abstained in ``frame`` for region ``q`` (None if it did not)."""
        if isinstance(self.elimination, QuadrantElimination):
            return self.elimination.abstained_frames.get((frame, q))
        if isinstance(self.elimination, EliminationResult):
            return self.elimination.abstained_frames.get(frame)
        return None

    def tags_in(self, matched: Assignments, frame: int, q: str) -> set[int]:
        """Distinct valid tags of region ``q`` in ``frame`` (restricted to its tag set if any)."""
        tag_set = self.tag_sets[q]
        tags: set[int] = set()
        for track, raw in matched.get(frame, {}).items():
            tag = normalize_tag(raw, strict=True)
            if tag is None:
                continue
            if self.scope is not None and self.scope.quadrant_of(frame, track) != q:
                continue
            if tag_set is not None and tag not in tag_set:
                continue
            tags.add(tag)
        return tags


def _region_histograms(
    ctx: _ReportContext,
    frames: range,
    instances: Instances,
    before: Assignments,
    after: Assignments,
) -> dict[str, dict[str, Any]]:
    """Per region: histograms of frames by number of tags (before/after) and of instances."""
    report = {}
    for q in ctx.regions:
        tag_set = ctx.tag_sets[q]
        counts = []
        for frame in frames:
            n_inst = sum(
                1
                for track, pos in instances.get(frame, [])
                if ctx.region_of(frame, track, pos) == q
            )
            counts.append(
                (len(ctx.tags_in(before, frame, q)), len(ctx.tags_in(after, frame, q)), n_inst)
            )
        # without a tag set the histogram covers the largest count actually observed
        n_max = len(tag_set) if tag_set else max([_DEFAULT_REGION_SIZE, *(max(c) for c in counts)])
        hist_before: Counter[int] = Counter()
        hist_after: Counter[int] = Counter()
        hist_instances: Counter[int] = Counter()
        for n_before, n_after, n_inst in counts:
            hist_before[min(n_before, n_max)] += 1
            hist_after[min(n_after, n_max)] += 1
            hist_instances[min(n_inst, n_max)] += 1
        report[q] = {
            "tag_set": tag_set,
            "frames_by_n_tags_before_elimination": _histogram(hist_before, n_max),
            "frames_by_n_tags_after_elimination": _histogram(hist_after, n_max),
            "frames_by_n_instances": _histogram(hist_instances, n_max),
        }
    return report


def _cause_of_missing_tag(  # noqa: PLR0911 (one return per cause keeps the decision table flat)
    ctx: _ReportContext,
    frame: int,
    track: Hashable | None,
    q: str | None,
    *,
    before_row: Mapping[Hashable, Hashable],
    after: Assignments,
) -> str:
    """Return why an instance has no tag after elimination."""
    if track is None:
        return "sleap_untracked_instance"
    if q is None:
        return "undeterminable_quadrant"
    if ctx.scope is not None and q not in ctx.scope.active:
        return "quadrant_without_tag_set"
    tag_set = ctx.tag_sets.get(q)
    raw = normalize_tag(before_row.get(track), strict=True)
    reason = ctx.abstain_reason(frame, q)
    if raw is not None and tag_set is not None and raw not in tag_set:
        return "wrong_quadrant_or_foreign_tag_dropped"
    if reason is not None:
        return f"elimination_abstained:{reason}"
    if tag_set is not None and len(ctx.tags_in(after, frame, q)) >= len(tag_set):
        return "extra_instance_all_tags_taken"
    return "no_decode_evidence"


def _untagged_instances(
    ctx: _ReportContext,
    frames: range,
    instances: Instances,
    before: Assignments,
    after: Assignments,
) -> tuple[int, int, int, Counter[str], dict[str, Counter[str]]]:
    """Count instances and those left without a tag, by cause.

    Returns:
        ``(n_instances, n_assigned_before, n_assigned_after, causes, causes_by_region)``.
        Instances whose region cannot be determined are listed under the region ``"none"``.
    """
    causes: Counter[str] = Counter()
    # pre-seeded so the regions keep their report order
    causes_by_region: defaultdict[str, Counter[str]] = defaultdict(
        Counter, {q: Counter() for q in ctx.regions}
    )
    n_inst_total = n_assigned_after = n_assigned_before = 0
    for frame in frames:
        row_b, row_a = before.get(frame, {}), after.get(frame, {})
        for track, pos in instances.get(frame, []):
            n_inst_total += 1
            q = ctx.region_of(frame, track, pos)
            tag_set = ctx.tag_sets.get(q) if q is not None else None
            tag_before = normalize_tag(row_b.get(track), strict=True) if track is not None else None
            tag_after = normalize_tag(row_a.get(track), strict=True) if track is not None else None
            if tag_set is not None and tag_before is not None and tag_before not in tag_set:
                tag_before = None
            if tag_before is not None:
                n_assigned_before += 1
            if tag_after is not None:
                n_assigned_after += 1
                continue
            why = _cause_of_missing_tag(ctx, frame, track, q, before_row=row_b, after=after)
            causes[why] += 1
            causes_by_region[q if q is not None else "none"][why] += 1
    return n_inst_total, n_assigned_before, n_assigned_after, causes, dict(causes_by_region)


def _elimination_summary(
    elimination: QuadrantElimination | EliminationResult | None,
) -> dict[str, Any] | None:
    """Return a JSON-ready summary of the elimination result.

    The eliminated samples of a :class:`QuadrantElimination` are written with ``int(track)``, so
    the track ids of the assignments must be integers (``naps-track`` uses the integer suffix of the
    SLEAP track name). The ``int()`` also keeps numpy integers JSON-serialisable; a non-numeric
    track id raises ``ValueError``/``TypeError``.
    """
    if isinstance(elimination, QuadrantElimination):
        return {
            "n_assigned": int(elimination.n_assigned),
            "abstention_reasons": dict(elimination.abstained),
            "dropped_foreign_tags": int(elimination.dropped_foreign),
            "per_quadrant": {
                q: {
                    "n_assigned": len(r.assigned),
                    "abstention_reasons": dict(r.abstained),
                    "dropped_foreign_tags": int(r.dropped_foreign),
                    # each sample: frame, track, tag, gap in frames, distance in pixels
                    "assigned_sample": [
                        [int(a[0]), int(cast("int", a[1])), int(a[2]), int(a[3]), round(a[4], 1)]
                        for a in r.assigned[:_ASSIGNED_SAMPLE_LIMIT]
                    ],
                }
                for q, r in elimination.per_quadrant.items()
            },
        }
    if isinstance(elimination, EliminationResult):
        return {
            "n_assigned": len(elimination.assigned),
            "abstention_reasons": dict(elimination.abstained),
            "dropped_foreign_tags": int(elimination.dropped_foreign),
        }
    return None


def build_report(  # noqa: PLR0917 (public API)
    scope: QuadrantScope | None,
    closed_set: Iterable[int] | None,
    first_frame: int,
    last_frame: int,
    before: Assignments,
    after: Assignments,
    instances: Instances,
    elimination: QuadrantElimination | EliminationResult | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the per-video diagnostics written by ``--report-json``.

    Args:
        scope: Quadrant scope, or None for a single region ("all").
        closed_set: Flat closed set (only without a scope).
        first_frame: First frame (inclusive) of the job.
        last_frame: Last frame (inclusive) of the job.
        before: ``{frame: {track: tag}}`` before elimination.
        after: ``{frame: {track: tag}}`` after elimination.
        instances: ``{frame: [(track_id or None, (x, y) or None), ...]}`` of every SLEAP instance.
        elimination: The elimination result, or None. Track ids in a
            :class:`QuadrantElimination` must be integers (see the eliminated samples).
        extra: Extra top-level entries merged into the report (they may override others).

    Returns:
        A JSON-serialisable dict (infinite ROI edges are ``None``).

    Raises:
        ValueError: If ``closed_set`` holds an invalid tag.
        RuntimeError: If ``scope`` was never prepared but an instance needs its regions.
    """
    frames = range(first_frame, last_frame + 1)
    ctx = _ReportContext(scope, closed_set, elimination)
    regions_report = _region_histograms(ctx, frames, instances, before, after)
    n_inst, n_before, n_after, causes, causes_by_region = _untagged_instances(
        ctx, frames, instances, before, after
    )

    if scope is not None:
        mode = "quadrants"
    elif closed_set:
        mode = "flat_closed_set"
    else:
        mode = "legacy"
    report: dict[str, Any] = {
        "frames": [int(first_frame), int(last_frame)],
        "frame_size": list(scope.frame_size) if scope is not None and scope.frame_size else None,
        "mode": mode,
        "regions": regions_report,
        "n_instances": int(n_inst),
        "n_instances_without_tag_before_elimination": int(n_inst - n_before),
        "n_instances_without_tag_after_elimination": int(n_inst - n_after),
        "n_none_instances_by_cause": dict(causes),
        "n_none_instances_by_region_and_cause": {
            q: dict(c) for q, c in causes_by_region.items() if c
        },
        "elimination": _elimination_summary(elimination),
    }
    if scope is not None:
        report["rois"] = {
            q: [None if math.isinf(v) else v for v in roi] for q, roi in (scope.rois or {}).items()
        }
        report["tag_sets_source"] = "inferred" if scope.tag_sets is None else "explicit"
        report["inference"] = scope.inference or None
        report["vote_filter_rejections"] = dict(scope.rejected)
        report["wrong_quadrant_votes"] = [
            {"tag": int(t), "seen_in": s, "belongs_to": b, "votes": int(n)}
            for (t, s, b), n in sorted(scope.wrong_quadrant.items())
        ]
    if extra:
        report.update(extra)
    return report
