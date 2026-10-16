"""Weighted identity evidence for NAPS.

NAPS decides identity by accumulating votes ``track -> tag`` over a rolling
window (see :mod:`naps.cost_matrix`). This module puts every source of votes
behind one small interface so that sources other than the ArUco decoder can
contribute, without changing how the votes are consumed:

* :class:`ArucoEvidence` wraps the ArUco decodes (weight 1 per decode, which is
  exactly the legacy behaviour) and can restrict votes to a closed set of tags.
* :class:`IdtrackerEvidence` turns an idtracker.ai trajectories file into
  *continuity* votes. ArUco stays the identity source: idtracker.ai labels are
  arbitrary per video, so they are first aligned to tags by co-occurrence with
  the ArUco decodes, and the source refuses to emit for any idtracker animal
  whose alignment is not well supported and pure.

Everything here is opt-in; :func:`collect_votes` is only called by
:class:`naps.matching.Matching` when evidence sources or a flat closed set are given, and always in
quadrant mode (there with ``allowed_tags=None`` and the quadrant vote filter). Without any of these
the legacy path passes the raw decodes to the cost matrix.
"""

import logging
import os
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from naps.tags import normalize_tag, require_tag

__all__ = [
    "AlignmentEntry",
    "ArucoEvidence",
    "EvidenceSource",
    "FrameVotes",
    "IdtrackerEvidence",
    "Vote",
    "collect_votes",
]

logger = logging.getLogger(__name__)

Vote = tuple[int, float]  # (tag, weight)
FrameVotes = dict[Hashable, list[Vote]]  # track -> votes in one frame
Decodes = Mapping[int, Mapping[Hashable, Sequence[Hashable | None]]]  # decodes[frame][track]
Positions = Mapping[int, Mapping[Hashable, Sequence[float]]]  # positions[frame][track] -> (x, y)
VoteFilter = Callable[[int, Hashable, int], bool]  # (frame, track, tag) -> keep?

# Cost assigned to an animal/track pair beyond ``max_dist`` in the join. It only has to exceed any
# real distance so the Hungarian solver never prefers it; pairs at this cost are discarded again.
_FORBIDDEN_JOIN_COST = 1e9


class EvidenceSource(ABC):
    """Yields weighted ``track -> tag`` votes for a frame."""

    name = "evidence"

    def prepare(self, positions: Positions, prior_votes: Mapping[int, FrameVotes]) -> None:  # noqa: B027 (optional hook: the default is deliberately a no-op)
        """Optional second-pass hook, called once before :meth:`votes`.

        Args:
            positions: ``positions[frame][track] -> (x, y)`` of the SLEAP tag node.
            prior_votes: ``prior_votes[frame][track] -> [(tag, weight)]``: the ArUco-only votes.
                :func:`collect_votes` hands the same mapping to every extra source, so a source
                never sees the votes of another extra source.
        """

    @abstractmethod
    def votes(
        self,
        frame: int,
        track_positions: Mapping[Hashable, Sequence[float]],
        prior: FrameVotes | None = None,
    ) -> FrameVotes:
        """Return the votes of one frame.

        Args:
            frame: Frame index.
            track_positions: ``track_positions[track] -> (x, y)`` of the tag nodes in this frame.
            prior: The ArUco-only votes of this frame (extra sources never see each other's
                votes).

        Returns:
            ``{track: [(tag, weight), ...]}``. A source abstains for a track by leaving it out.
        """


class ArucoEvidence(EvidenceSource):
    """ArUco decodes as an evidence source.

    Every track present in ``decodes[frame]`` keeps its key (with an empty list
    when nothing usable was decoded): the key is what the cost matrix treats as
    "track present in this frame".

    Args:
        decodes: ``decodes[frame][track] -> [tag | None, ...]`` as produced by
            ``MatchFrame.returnMarkerTags``.
        allowed_tags: If given, decodes of any other tag are dropped (closed set).
        weight: Weight of one decode. 1.0 reproduces the legacy behaviour.
        vote_filter: Optional ``(frame, track, tag) -> bool`` (e.g.
            ``QuadrantScope.vote_filter``); votes it rejects are dropped.

    Attributes:
        n_out_of_set: Number of decodes dropped because they were outside ``allowed_tags``.
    """

    name = "aruco"

    def __init__(
        self,
        decodes: Decodes,
        allowed_tags: Iterable[int] | None = None,
        weight: float = 1.0,
        vote_filter: VoteFilter | None = None,
    ) -> None:
        """Store the decodes and the filtering options."""
        self.decodes = decodes
        self.allowed_tags = None if allowed_tags is None else {require_tag(t) for t in allowed_tags}
        self.weight = float(weight)
        self.n_out_of_set = 0
        self.vote_filter = vote_filter

    def votes(
        self,
        frame: int,
        track_positions: Mapping[Hashable, Sequence[float]] | None = None,
        prior: FrameVotes | None = None,
    ) -> FrameVotes:
        """Return the usable ArUco decodes of ``frame`` as ``{track: [(tag, weight), ...]}``.

        Decodes that are ``None``/NaN/non-integral, outside ``allowed_tags`` or rejected by
        ``vote_filter`` are dropped (outside-set ones are counted in ``n_out_of_set``). Tags are
        normalised to plain ``int``. ``track_positions`` and ``prior`` are unused; unlike
        :meth:`EvidenceSource.votes` this override does not require ``track_positions``.
        """
        out: FrameVotes = {}
        for track, tag_list in self.decodes.get(frame, {}).items():
            kept: list[Vote] = []
            for raw_tag in tag_list:
                tag = normalize_tag(raw_tag, strict=True)
                if tag is None:
                    continue
                if self.allowed_tags is not None and tag not in self.allowed_tags:
                    self.n_out_of_set += 1
                    continue
                if self.vote_filter is not None and not self.vote_filter(frame, track, tag):
                    continue
                kept.append((tag, self.weight))
            out[track] = kept
        return out


@dataclass
class AlignmentEntry:
    """Alignment of one idtracker animal to an ArUco tag.

    Attributes:
        animal: The idtracker animal index.
        tag: Best tag by co-occurrence (None if no co-occurrence).
        support: Frames with a single, unambiguous ArUco decode.
        purity: Fraction of that support that agrees with ``tag``.
        accepted: Whether the animal may emit votes.
        reason: ``"ok"`` or why the animal was refused.
    """

    animal: int
    tag: int | None
    support: int
    purity: float
    accepted: bool
    reason: str


class IdtrackerEvidence(EvidenceSource):
    """Continuity votes from an idtracker.ai v6 ``trajectories.npy``.

    Per frame, idtracker animals (centroids + ``offset``) are joined to SLEAP
    tracks (tag node positions) with a distance-gated Hungarian assignment. A
    pair is kept only when it is unambiguous: the matched distance must be at
    most ``ambiguity_ratio`` times (and ``min_margin`` pixels less than) the
    distance to the next-nearest alternative on either side. Interpolated animals
    (finite position, NaN id_probability) take part in this competition, so they
    can never be stolen by a neighbour, but they never emit votes themselves.

    Idtracker labels are arbitrary per video. :meth:`prepare` estimates
    ``id_to_tag`` by counting, over joined frames in which the track has exactly
    one distinct ArUco decode, how often each animal co-occurs with each tag,
    solving the animal x tag assignment, and accepting an animal only if its
    support is at least ``min_support`` frames, its purity at least ``min_purity``, no other
    sufficiently supported animal claims the same best tag, and the one-to-one (Hungarian)
    assignment agrees with that tag (see :meth:`_refusal_reason`). Animals that are refused emit
    nothing.

    Args:
        trajectories: ``(frames, animals, 2)`` centroids (NaN = missing).
        id_probabilities: ``(frames, animals)`` or ``(frames, animals, 1)``;
            NaN marks an interpolated position, 0 an absent animal.
        offset: ``(x, y)`` added to the centroids to reach the coordinates of the
            SLEAP file (e.g. the quadrant crop origin).
        frame_offset: Trajectory index ``i`` corresponds to video frame
            ``i + frame_offset``.
        weight: Vote weight (scaled by ``id_probability`` if ``prob_weighted``).
            Keep below the ArUco weight of 1.
        prob_weighted: Scale the vote weight by the animal's id_probability.
        min_id_prob: Minimum id_probability for an animal to vote.
        max_dist: Maximum join distance in pixels.
        ambiguity_ratio: See above; lower is stricter.
        min_margin: The matched distance must also be at least this many pixels
            smaller than the next-nearest alternative. This must exceed the
            worst-case offset between an idtracker centroid and the SLEAP tag
            node, otherwise a near-crossing can pass the ratio test and swap.
        only_when_aruco_missing: Emit for a track only in frames where it has
            no usable ArUco decode (idtracker.ai bridges dropouts, never argues
            with a decode).
        min_support: Minimum number of unambiguous co-occurrence frames for an
            alignment to be accepted.
        min_purity: Minimum fraction of those frames that agree with the best tag.
        allowed_tags: Tags that animals may be aligned to (the closed set).
        id_to_tag: Fixed ``{animal: tag}`` mapping; skips the alignment step. The tags are
            normalised to plain ``int``.

    Attributes:
        id_to_tag: Accepted ``{animal: tag}`` alignment: empty until :meth:`prepare` ran, unless a
            fixed ``id_to_tag`` was given (then it holds that mapping from construction on).
        alignment: Per-animal diagnostics of the last :meth:`prepare`.

    Raises:
        ValueError: If the array shapes are inconsistent, or a tag in ``allowed_tags`` or
            ``id_to_tag`` is not a valid tag id.
    """

    name = "idtracker"

    def __init__(  # noqa: PLR0917 (public API: keyword-style tuning parameters)
        self,
        trajectories: NDArray[np.float64],
        id_probabilities: NDArray[np.float64],
        offset: Sequence[float] = (0.0, 0.0),
        frame_offset: int = 0,
        weight: float = 0.5,
        prob_weighted: bool = True,
        min_id_prob: float = 0.5,
        max_dist: float = 100.0,
        ambiguity_ratio: float = 0.5,
        min_margin: float = 20.0,
        only_when_aruco_missing: bool = True,
        min_support: int = 30,
        min_purity: float = 0.9,
        allowed_tags: Iterable[int] | None = None,
        id_to_tag: Mapping[int, int] | None = None,
    ) -> None:
        """Validate and store the trajectories and options (alignment happens in prepare)."""
        traj = np.asarray(trajectories, dtype=float)
        if traj.ndim != 3 or traj.shape[2] != 2:
            msg = f"trajectories must be (frames, animals, 2); got {traj.shape}"
            raise ValueError(msg)
        prob = np.asarray(id_probabilities, dtype=float)
        if prob.ndim == 3 and prob.shape[2] == 1:
            prob = prob[..., 0]
        if prob.shape != traj.shape[:2]:
            msg = f"id_probabilities {prob.shape} does not match trajectories {traj.shape}"
            raise ValueError(msg)
        self.trajectories = traj + np.asarray(offset, dtype=float)
        self.id_probabilities = prob
        self.frame_offset = int(frame_offset)
        self.weight = float(weight)
        self.prob_weighted = bool(prob_weighted)
        self.min_id_prob = float(min_id_prob)
        self.max_dist = float(max_dist)
        self.ambiguity_ratio = float(ambiguity_ratio)
        self.min_margin = float(min_margin)
        self.only_when_aruco_missing = bool(only_when_aruco_missing)
        self.min_support = int(min_support)
        self.min_purity = float(min_purity)
        self.allowed_tags = (
            None if allowed_tags is None else sorted({require_tag(t) for t in allowed_tags})
        )
        self.fixed_id_to_tag = (
            None if id_to_tag is None else {a: require_tag(t) for a, t in id_to_tag.items()}
        )
        self.id_to_tag: dict[int, int] = dict(self.fixed_id_to_tag or {})
        self.alignment: dict[int, AlignmentEntry] = {}
        self._join_cache: dict[int, list[tuple[int, Hashable, float]]] = {}

    @classmethod
    def from_file(cls, path: str | os.PathLike[str], **kwargs: object) -> "IdtrackerEvidence":
        """Load a v6 ``trajectories.npy`` (a pickled dict saved with ``np.save``).

        The file is unpickled, so only load files you trust.

        Args:
            path: Path of the ``trajectories.npy`` file.
            **kwargs: Passed to the constructor.

        Returns:
            The evidence source.

        Raises:
            ValueError: If the file lacks ``trajectories`` or ``id_probabilities``.
        """
        data = np.load(path, allow_pickle=True).item()
        if "trajectories" not in data:
            msg = f"{path} has no 'trajectories' entry"
            raise ValueError(msg)
        traj = data["trajectories"]
        prob = data.get("id_probabilities")
        if prob is None:
            msg = (
                f"{path} has no 'id_probabilities'; they are required to gate out "
                "interpolated positions"
            )
            raise ValueError(msg)
        return cls(traj, prob, **kwargs)  # ty: ignore[invalid-argument-type]  # kwargs are the constructor's own keyword arguments

    # ------------------------------------------------------------------ join
    def _confident(self, idx: int, animal: int) -> bool:
        """Whether the animal's id_probability at trajectory index ``idx`` allows it to vote."""
        p = self.id_probabilities[idx, animal]
        return bool(np.isfinite(p) and p >= self.min_id_prob and p > 0)

    def _join(
        self, frame: int, track_positions: Mapping[Hashable, Sequence[float]]
    ) -> list[tuple[int, Hashable, float]]:
        """Unambiguous ``(animal, track, distance)`` pairs for a frame.

        The result is cached per frame: the ``track_positions`` first seen for a frame win, so a
        source must always be given the same positions for the same frame.
        """
        if frame in self._join_cache:
            return self._join_cache[frame]
        idx = frame - self.frame_offset
        pairs: list[tuple[int, Hashable, float]] = []
        if 0 <= idx < self.trajectories.shape[0]:
            animals = np.where(np.isfinite(self.trajectories[idx]).all(axis=1))[0]
            tracks: list[Hashable] = []
            xy: list[NDArray[np.float64]] = []
            for track, pos in track_positions.items():
                arr = np.asarray(pos, dtype=float)
                if arr.shape == (2,) and np.isfinite(arr).all():
                    tracks.append(track)
                    xy.append(arr)
            if len(animals) and tracks:
                dist = np.linalg.norm(
                    self.trajectories[idx, animals][:, None, :] - np.asarray(xy)[None, :, :],
                    axis=2,
                )
                pairs = self._unambiguous_pairs(dist, animals, tracks)
        self._join_cache[frame] = pairs
        return pairs

    def _unambiguous_pairs(
        self, dist: NDArray[np.float64], animals: NDArray[np.intp], tracks: list[Hashable]
    ) -> list[tuple[int, Hashable, float]]:
        """Distance-gated Hungarian join, keeping only pairs that are clearly the best match.

        Args:
            dist: ``(n_animals, n_tracks)`` distances.
            animals: Idtracker animal index of each row of ``dist``.
            tracks: Track of each column of ``dist``.

        Returns:
            ``(animal, track, distance)`` for each pair within ``max_dist`` whose distance is at
            most ``ambiguity_ratio`` times, and at least ``min_margin`` pixels less than, the
            nearest alternative for either member. Ambiguous pairs (e.g. a crossing) are dropped.
        """
        cost = np.where(dist <= self.max_dist, dist, _FORBIDDEN_JOIN_COST)
        rows, cols = linear_sum_assignment(cost)
        pairs = []
        for r, c in zip(rows, cols, strict=True):
            d = dist[r, c]
            if d > self.max_dist:
                continue
            other_animals = np.delete(dist[:, c], r)
            other_tracks = np.delete(dist[r, :], c)
            alt = min(np.min(other_animals, initial=np.inf), np.min(other_tracks, initial=np.inf))
            # Ambiguous (e.g. a crossing): abstain rather than guess
            if d <= self.ambiguity_ratio * alt and alt - d >= self.min_margin:
                pairs.append((int(animals[r]), tracks[c], float(d)))
        return pairs

    # ------------------------------------------------------------- alignment
    def _count_cooccurrence(
        self, positions: Positions, prior_votes: Mapping[int, FrameVotes]
    ) -> defaultdict[int, Counter[int]]:
        """Count, per animal, how often it is joined to a track with a single distinct ArUco tag.

        Only frames in which the animal is confidently identified (:meth:`_confident`) count.
        """
        counts: defaultdict[int, Counter[int]] = defaultdict(Counter)
        for frame, votes in prior_votes.items():
            idx = frame - self.frame_offset
            if not 0 <= idx < self.trajectories.shape[0]:
                continue
            for animal, track, _ in self._join(frame, positions.get(frame, {})):
                if not self._confident(idx, animal):
                    continue
                tags = {tag for tag, _ in votes.get(track, [])}
                if len(tags) == 1:  # skip frames with no decode or a neighbour's tag too
                    counts[animal][next(iter(tags))] += 1
        return counts

    def prepare(self, positions: Positions, prior_votes: Mapping[int, FrameVotes]) -> None:
        """Align the idtracker animals to ArUco tags by co-occurrence.

        Fills :attr:`id_to_tag` (accepted animals only) and :attr:`alignment` (diagnostics for
        every animal). With a fixed ``id_to_tag`` nothing is estimated. If no animal ever
        co-occurs with a decode, or every animal is refused, the source emits nothing.

        Args:
            positions: ``positions[frame][track] -> (x, y)`` of the SLEAP tag node.
            prior_votes: ``prior_votes[frame][track] -> [(tag, weight)]`` from the ArUco source.
        """
        if self.fixed_id_to_tag is not None:
            self.id_to_tag = dict(self.fixed_id_to_tag)
            return
        n_animals = self.trajectories.shape[1]
        counts = self._count_cooccurrence(positions, prior_votes)

        tags = self.allowed_tags
        if tags is None:
            tags = sorted({t for c in counts.values() for t in c})
        self.alignment = {}
        self.id_to_tag = {}
        if not tags:
            for a in range(n_animals):
                self.alignment[a] = AlignmentEntry(a, None, 0, 0.0, False, "no co-occurrence")
            logger.warning(
                "idtracker alignment: no co-occurrence with ArUco decodes; emitting nothing"
            )
            return

        mat = np.array([[counts[a][t] for t in tags] for a in range(n_animals)], dtype=float)
        rows, cols = linear_sum_assignment(-mat)
        hungarian = {int(r): tags[c] for r, c in zip(rows, cols, strict=True)}
        tops = {a: (tags[int(mat[a].argmax())] if mat[a].sum() else None) for a in range(n_animals)}
        supported = [a for a in range(n_animals) if mat[a].sum() >= self.min_support]
        top_owner = Counter(tops[a] for a in supported)
        for a in range(n_animals):
            support = int(mat[a].sum())
            top = tops[a]
            purity = float(mat[a].max() / support) if support else 0.0
            reason = self._refusal_reason(
                support, purity, top_owner[top], hungarian.get(a) == top, top
            )
            if reason == "ok" and top is not None:
                self.alignment[a] = AlignmentEntry(a, top, support, purity, True, reason)
                self.id_to_tag[a] = top
            else:
                self.alignment[a] = AlignmentEntry(a, top, support, purity, False, reason)
                logger.info("idtracker animal %s refused: %s", a, reason)
        if not self.id_to_tag:
            logger.warning("idtracker alignment refused every animal; emitting nothing")

    def _refusal_reason(
        self, support: int, purity: float, n_claims: int, one_to_one: bool, top: int | None
    ) -> str:
        """Why an animal's alignment is refused, or ``"ok"``. Checks run in this order.

        Args:
            support: Number of co-occurrence frames of the animal.
            purity: Fraction of them agreeing with its best tag ``top``.
            n_claims: Number of sufficiently supported animals whose best tag is ``top``.
            one_to_one: Whether the one-to-one (Hungarian) assignment gave the animal ``top``.
            top: The animal's best tag.
        """
        if support < self.min_support:
            return f"support {support} < {self.min_support}"
        if purity < self.min_purity:
            return f"purity {purity:.2f} < {self.min_purity}"
        if top is None:  # only reachable with min_support <= 0 and min_purity <= 0
            return "no co-occurrence"
        if n_claims > 1:
            return f"tag {top} claimed by {n_claims} animals"
        if not one_to_one:
            return "one-to-one assignment disagrees with best tag"
        return "ok"

    # ----------------------------------------------------------------- votes
    def votes(
        self,
        frame: int,
        track_positions: Mapping[Hashable, Sequence[float]],
        prior: FrameVotes | None = None,
    ) -> FrameVotes:
        """Return the continuity votes of ``frame``.

        Only animals accepted by :meth:`prepare`, confidently identified and unambiguously joined
        to a track emit a vote; with ``only_when_aruco_missing`` tracks that have a usable ArUco
        decode in ``prior`` are skipped. Nothing is emitted before an alignment exists.
        """
        idx = frame - self.frame_offset
        out: FrameVotes = {}
        if not self.id_to_tag:
            return out
        for animal, track, _ in self._join(frame, track_positions):
            if animal not in self.id_to_tag or not self._confident(idx, animal):
                continue
            if self.only_when_aruco_missing and prior and prior.get(track):
                continue
            w = self.weight * (
                float(self.id_probabilities[idx, animal]) if self.prob_weighted else 1.0
            )
            out.setdefault(track, []).append((self.id_to_tag[animal], w))
        return out


def collect_votes(  # noqa: PLR0917 (public API)
    decodes: Decodes,
    positions: Positions,
    sources: Sequence[EvidenceSource] = (),
    allowed_tags: Iterable[int] | None = None,
    aruco_weight: float = 1.0,
    vote_filter: VoteFilter | None = None,
) -> dict[int, FrameVotes]:
    """Merge ArUco decodes and extra evidence into the dict the cost matrix consumes.

    Extra sources run after ArUco and see its votes as ``prior``. They can never create a track
    the video did not have.

    Args:
        decodes: ``decodes[frame][track] -> [tag | None, ...]``.
        positions: ``positions[frame][track] -> (x, y)`` of the tag nodes.
        sources: Extra evidence sources, run in order.
        allowed_tags: Closed set; ArUco decodes of other tags are dropped.
        aruco_weight: Weight of one ArUco decode.
        vote_filter: ``(frame, track, tag) -> bool``, applied to every vote of every source
            (quadrant-aware closed set).

    Returns:
        ``votes[frame][track] -> [(tag, weight), ...]`` with the same frame and track keys as
        ``decodes``.
    """
    aruco = ArucoEvidence(decodes, allowed_tags, aruco_weight, vote_filter=vote_filter)
    merged = {frame: aruco.votes(frame, positions.get(frame, {})) for frame in decodes}
    aruco_only = {f: {t: list(v) for t, v in tv.items()} for f, tv in merged.items()}
    if aruco.n_out_of_set:
        logger.info("Dropped %s ArUco decodes outside the closed set", aruco.n_out_of_set)
    for source in sources:
        source.prepare(positions, aruco_only)
        n_votes = 0
        for frame, frame_votes in merged.items():
            extra = source.votes(frame, positions.get(frame, {}), aruco_only[frame])
            for track, extra_votes in extra.items():
                if track in frame_votes:  # never create a track the video did not have
                    kept = [
                        v
                        for v in extra_votes
                        if vote_filter is None or vote_filter(frame, track, v[0])
                    ]
                    frame_votes[track].extend(kept)
                    n_votes += len(kept)
        logger.info("Evidence source %s contributed %s votes", source.name, n_votes)
    return merged
