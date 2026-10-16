"""Matching pipeline: decode ArUco tags around every SLEAP track and assign tags to tracks.

:class:`Matching` reads the video frame by frame, crops a window around each track's tag node
(:class:`MatchFrame`), decodes the crops with a marker detector (:class:`naps.aruco.ArUcoModel`),
optionally merges extra evidence and restricts the votes to a closed set, and finally lets
:class:`naps.cost_matrix.CostMatrix` pick the tag of every track from the votes in a rolling
window. Optional closed-set elimination (:mod:`naps.closed_set`, :mod:`naps.quadrants`) then
fills in a single missing identity where that is unambiguous and abstains otherwise.
"""

import logging
import warnings
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from naps.closed_set import EliminationResult, eliminate_identities
from naps.cost_matrix import CostMatrix, MatchedDict
from naps.decode import DecodeResult, TagDecoder
from naps.decode import decode_video as decode_tags
from naps.evidence import EvidenceSource, FrameVotes, collect_votes
from naps.parallel_decode import Backend, decode_in_chunks
from naps.quadrants import QuadrantElimination, QuadrantScope, eliminate_by_quadrant
from naps.tags import require_tag
from naps.video import FastVideoReader, open_capture

__all__ = [
    "ChunkedDecodeSource",
    "DecodeSource",
    "MatchFrame",
    "Matching",
    "OpenCVDecodeSource",
    "TagDecoderSource",
    "decode_video",
]

logger = logging.getLogger(__name__)

# ``decodes[frame][track] -> [tag | None, ...]`` as produced by ``MatchFrame.returnMarkerTags``
Decodes = dict[int, defaultdict[Hashable, list[Hashable | None]]]
# ``coords[track] -> (x, y)``: the tag node of every track in one frame
TrackCoords = Mapping[Hashable, Sequence[float]]
Image = NDArray[np.generic]  # what cv2 hands out (typically uint8, BGR)
MarkerDetector = Callable[[Image], Iterable[Hashable | None]]

# Warn when the median number of tracks per frame exceeds the closed set by more than this many.
_CLOSED_SET_TRACK_SLACK = 1


class DecodeSource(Protocol):
    """Where the tag decodes come from: reads the video and decodes tags at given positions.

    The joint identity engine only needs this small interface, so the decoder (OpenCV's
    ``VideoCapture`` plus the ArUco detector today, a faster reader / batched decoder later) can
    be swapped without touching the engine.
    """

    def decode(
        self,
        first_frame: int,
        last_frame: int,
        coords: Mapping[int, Mapping[Hashable, Sequence[float]]],
    ) -> tuple[Decodes, tuple[int, int] | None]:
        """Decode the tags at ``coords`` in frames ``first_frame..last_frame`` (inclusive).

        Args:
            first_frame: First frame (zero-based, inclusive).
            last_frame: Last frame (inclusive).
            coords: ``coords[frame][key] -> (x, y)`` of the tag node of every instance; a NaN
                coordinate means "no tag position" and yields ``[None]``. Every frame in the range
                must be present.

        Returns:
            ``(decodes, frame_size)``: ``decodes[frame][key] -> [tag | None, ...]`` and the
            ``(width, height)`` of the frames (``None`` if the first frame could not be read).
        """
        ...


def decode_video(  # noqa: PLR0917 (module-level form of the legacy decode loop)
    video_filename: str,
    first_frame: int,
    last_frame: int,
    coords: Mapping[int, Mapping[Hashable, Sequence[float]]],
    crop_size: int,
    marker_detector: MarkerDetector,
    decoder_threads: int | None = None,
) -> tuple[Decodes, tuple[int, int] | None]:
    """Read the video frame by frame and decode the tags around every position in ``coords``.

    Frames past the end of the video are tolerated while no key has a position there (they yield
    no votes); a key with a position in an unreadable frame raises.

    Args:
        video_filename: Path of the video.
        first_frame: First frame (zero-based, inclusive).
        last_frame: Last frame (inclusive).
        coords: ``coords[frame][key] -> (x, y)`` for every frame of the range.
        crop_size: Half-width in pixels of the crop around each position.
        marker_detector: Maps a cropped image to the decoded tags.
        decoder_threads: Decoder threads of the video backend (see
            :func:`naps.video.open_capture`); ``None``: OpenCV's default.

    Returns:
        ``(decodes, frame_size)`` as in :meth:`DecodeSource.decode`.

    Raises:
        RuntimeError: If a position lies in a frame that cannot be read.
        KeyError: If ``coords`` lacks a frame of the range.
    """
    logger.info("Processing frames %s to %s", first_frame, last_frame)
    decodes: Decodes = {}
    frame_size: tuple[int, int] | None = None
    current_frame = first_frame
    video = open_capture(video_filename, decoder_threads)
    try:
        # Zero-based start frame
        video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        frame = MatchFrame.fromCV2(*video.read())
        if frame.frame is not None and np.ndim(frame.frame) >= 2:
            frame_size = (frame.frame.shape[1], frame.frame.shape[0])
        while current_frame <= last_frame:
            try:
                frame.cropMarkerWithCoordsArray(coords[current_frame], crop_size)
            except RuntimeError as exc:
                msg = f"{video_filename}, frame {current_frame}: {exc}"
                raise RuntimeError(msg) from exc
            decodes[current_frame] = frame.returnMarkerTags(marker_detector)
            frame = MatchFrame.fromCV2(*video.read())
            current_frame += 1
    finally:
        video.release()
    return decodes, frame_size


class TagDecoderSource:
    """:class:`DecodeSource` backed by the decode stage (:func:`naps.decode.decode_video`).

    Same decodes as :class:`OpenCVDecodeSource` with the legacy settings, but with the fast
    grayscale frame reader and the configurable :class:`~naps.decode.TagDecoder` (closed-set
    decoding, position gate, second pass). The full :class:`~naps.decode.DecodeResult` of the last
    call (with corners and confidences) is kept in :attr:`last_result`.

    Args:
        video: Path of the video (or an open :class:`~naps.video.FastVideoReader`).
        decoder: The decoder.
        crop_half: Half-width in pixels of the crop around each tag node.
        legacy_pixels: Make the crops bit-identical to the legacy ``read()`` path.
        reader_threads: Decoder threads of the reader opened for a path (``None``: OpenCV's
            default; see :func:`naps.video.open_capture`).

    Attributes:
        last_result: The :class:`~naps.decode.DecodeResult` of the last :meth:`decode` call.
    """

    def __init__(
        self,
        video: str | Path | FastVideoReader,
        decoder: TagDecoder,
        crop_half: int = 50,
        legacy_pixels: bool = True,
        reader_threads: int | None = None,
    ) -> None:
        """Store the arguments; the video is opened by :meth:`decode`."""
        self.video = video
        self.decoder = decoder
        self.crop_half = crop_half
        self.legacy_pixels = legacy_pixels
        self.reader_threads = reader_threads
        self.last_result: DecodeResult | None = None

    def decode(
        self,
        first_frame: int,
        last_frame: int,
        coords: Mapping[int, Mapping[Hashable, Sequence[float]]],
    ) -> tuple[Decodes, tuple[int, int] | None]:
        """Decode ``coords`` with :func:`naps.decode.decode_video` (see :class:`DecodeSource`)."""
        result = decode_tags(
            self.video,
            coords,
            self.decoder,
            first_frame=first_frame,
            last_frame=last_frame,
            crop_half=self.crop_half,
            legacy_pixels=self.legacy_pixels,
            reader_threads=self.reader_threads,
        )
        self.last_result = result
        return result.votes, result.frame_size


class OpenCVDecodeSource:
    """:class:`DecodeSource` backed by ``cv2.VideoCapture`` and a marker detector (the legacy path).

    Args:
        video_filename: Path of the video.
        marker_detector: Maps a cropped image to the decoded tags (e.g. ``ArUcoModel.detect``).
        crop_size: Half-width in pixels of the crop around each tag node.
        decoder_threads: Decoder threads of the video backend (``None``: OpenCV's default; see
            :func:`naps.video.open_capture`).

    Raises:
        FileNotFoundError: If ``video_filename`` is not an existing file.
    """

    def __init__(
        self,
        video_filename: str,
        marker_detector: MarkerDetector,
        crop_size: int,
        decoder_threads: int | None = None,
    ) -> None:
        """Store the arguments; the video is opened by :meth:`decode`."""
        if not Path(video_filename).is_file():
            msg = f"{video_filename} does not exist"
            raise FileNotFoundError(msg)
        self.video_filename = video_filename
        self.marker_detector = marker_detector
        self.crop_size = crop_size
        self.decoder_threads = decoder_threads

    def decode(
        self,
        first_frame: int,
        last_frame: int,
        coords: Mapping[int, Mapping[Hashable, Sequence[float]]],
    ) -> tuple[Decodes, tuple[int, int] | None]:
        """Decode ``coords`` with :func:`decode_video` (see :meth:`DecodeSource.decode`)."""
        return decode_video(
            self.video_filename,
            first_frame,
            last_frame,
            coords,
            self.crop_size,
            self.marker_detector,
            self.decoder_threads,
        )


class ChunkedDecodeSource:
    """A :class:`DecodeSource` that decodes its inner source in parallel chunks of frames.

    The frame range is cut into contiguous chunks (:func:`naps.video.split_range`) that are
    decoded concurrently by calling ``inner.decode`` on each; the merged decodes are exactly the
    serial ones (see :mod:`naps.parallel_decode`). Build the inner source with
    ``decoder_threads=1`` / ``reader_threads=1`` so that each worker runs one decoder thread.
    The inner source's own bookkeeping (``last_result`` of :class:`TagDecoderSource`) is not
    available: the chunks run in other processes.

    Args:
        inner: The source that decodes one chunk (it must be safe to call from forked workers:
            :class:`OpenCVDecodeSource` and :class:`TagDecoderSource` are).
        workers: Parallel workers (1: the chunks run one after the other in this process).
        chunk_frames: Frames per chunk (default: balanced for ``workers``).
        align: Start chunks on frames ``b`` with ``b % align == offset % align`` (the keyframe
            spacing of the video makes the seek to a chunk cheap).
        offset: See ``align``.
        backend: ``"process"`` (fork) or ``"thread"`` workers.
        background: Optional job run while the chunks are decoded (see
            :func:`naps.parallel_decode.run_chunks`).
    """

    def __init__(
        self,
        inner: DecodeSource,
        *,
        workers: int = 1,
        chunk_frames: int | None = None,
        align: int = 1,
        offset: int = 0,
        backend: Backend = "process",
        background: Callable[[], None] | None = None,
    ) -> None:
        """Store the arguments."""
        self.inner = inner
        self.workers = workers
        self.chunk_frames = chunk_frames
        self.align = align
        self.offset = offset
        self.backend: Backend = backend
        self.background = background

    def decode(
        self,
        first_frame: int,
        last_frame: int,
        coords: Mapping[int, Mapping[Hashable, Sequence[float]]],
    ) -> tuple[Decodes, tuple[int, int] | None]:
        """Decode in chunks and merge (see :meth:`DecodeSource.decode`)."""
        return decode_in_chunks(
            lambda a, b: self.inner.decode(a, b, coords),
            first_frame,
            last_frame,
            workers=self.workers,
            chunk_frames=self.chunk_frames,
            align=self.align,
            offset=self.offset,
            backend=self.backend,
            background=self.background,
        )


class Matching:
    """Matching pipeline.

    Args:
        video_filename: Path of the video to read.
        video_first_frame: First frame (inclusive, zero-based) to match.
        video_last_frame: Last frame (inclusive, zero-based) to match.
        marker_detector: Function mapping a cropped image to a list of decoded tag ids
            (``None`` entries mean "no tag found").
        aruco_crop_size: Half-width in pixels of the crop around each tag node.
        half_rolling_window_size: Frames upstream and downstream of a frame whose votes are pooled
            by the cost matrix. The first ``half_rolling_window_size`` frames of a job are never
            assigned (see :mod:`naps.cost_matrix`).
        tag_node_dict: ``tag_node_dict[frame][track] -> (x, y)`` position of the tag node. Must
            hold every frame in ``video_first_frame..video_last_frame``.
        min_sleap_score: Deprecated and ignored; kept so positional callers do not shift.
        evidence_sources: Optional extra identity evidence (e.g. ``IdtrackerEvidence``) merged
            with the ArUco decodes. Default None: legacy behaviour.
        closed_set: Optional expected tags of the quadrant. Restricts ArUco votes to these tags
            and, if ``eliminate`` is set, runs closed-set elimination on the result.
        eliminate: Run closed-set elimination when ``closed_set`` (or ``quadrants``) is given.
        elimination_max_jump: Continuity gate (pixels per frame of gap) for elimination.
        elimination_max_look: Frames searched for the missing tag's reference position.
        elimination_max_distance: Optional absolute cap (pixels) on the continuity threshold.
        untracked_counts: ``{frame: n}`` SLEAP instances without a track (elimination abstains
            there).
        quadrants: Optional quadrant-aware closed set (mutually exclusive with ``closed_set``):
            tracks are assigned to regions by position, votes are filtered per region and
            elimination runs per region. Tag sets are explicit or inferred from this video's
            decodes.
        position_dict: ``{frame: {track: (x, y)}}`` used to locate tracks and by elimination in
            quadrant mode (default: ``tag_node_dict``; naps-track passes the tag node with a
            fallback node for NaN tags).
        untracked_positions: ``{frame: [(x, y) | None]}`` of SLEAP instances without a track.
        strict_carry_forward: Passed to :class:`~naps.cost_matrix.CostMatrix`.
        decode_source: Optional :class:`DecodeSource` serving the decodes of ``tag_node_dict``
            instead of reading ``video_filename`` (for example a
            :class:`~naps.precomputed.PrecomputedDecodeSource`); the video is then never opened
            and need not exist, and ``marker_detector`` is unused.
        **kwargs: Ignored (a warning names them); accepted for backward compatibility.

    Attributes:
        frame_size: ``(width, height)`` of the video frames, set by :meth:`match`.
        pre_elimination: The cost-matrix assignments before closed-set elimination, set by
            :meth:`match`.
        elimination_result: The elimination summary (``None`` if elimination did not run).

    Raises:
        FileNotFoundError: If ``video_filename`` is not an existing file.
        ValueError: If both ``closed_set`` and ``quadrants`` are given.
    """

    def __init__(  # noqa: PLR0917 (public API: many positional parameters kept for compatibility)
        self,
        video_filename: str,
        video_first_frame: int,
        video_last_frame: int,
        marker_detector: MarkerDetector,
        aruco_crop_size: int,
        half_rolling_window_size: int,
        tag_node_dict: Mapping[int, TrackCoords],
        min_sleap_score: float | None = None,
        evidence_sources: Sequence[EvidenceSource] | None = None,
        closed_set: Iterable[int] | None = None,
        eliminate: bool = True,
        elimination_max_jump: float = 120.0,
        elimination_max_look: int = 25,
        elimination_max_distance: float | None = None,
        untracked_counts: Mapping[int, int] | None = None,
        quadrants: QuadrantScope | None = None,
        position_dict: Mapping[int, Mapping[Hashable, Sequence[float]]] | None = None,
        untracked_positions: Mapping[int, Sequence[Sequence[float] | None]] | None = None,
        strict_carry_forward: bool = False,
        decode_source: DecodeSource | None = None,
        **kwargs: object,
    ) -> None:
        """Validate the arguments and store them; the work happens in :meth:`match`."""
        if decode_source is None and not Path(video_filename).is_file():
            msg = f"{video_filename} does not exist"
            raise FileNotFoundError(msg)
        if quadrants is not None and closed_set is not None:
            msg = "closed_set and quadrants are mutually exclusive"
            raise ValueError(msg)
        if min_sleap_score is not None:
            warnings.warn(
                "min_sleap_score is deprecated and ignored", DeprecationWarning, stacklevel=2
            )
        if kwargs:
            warnings.warn(f"Matching ignores the arguments {sorted(kwargs)}", stacklevel=2)

        self.strict_carry_forward = strict_carry_forward
        self.decode_source = decode_source

        self.video_filename = video_filename
        self.video_first_frame = video_first_frame
        self.video_last_frame = video_last_frame
        self.tag_node_dict = tag_node_dict

        self.half_rolling_window_size = half_rolling_window_size
        self.aruco_crop_size = aruco_crop_size
        self.marker_detector = marker_detector

        # Optional evidence / closed-set arguments (all off by default)
        self.evidence_sources = list(evidence_sources or [])
        self.closed_set = (
            None if closed_set is None else sorted({require_tag(t) for t in closed_set})
        )
        self.eliminate = eliminate
        self.elimination_max_jump = elimination_max_jump
        self.elimination_max_look = elimination_max_look
        self.elimination_max_distance = elimination_max_distance
        self.untracked_counts = untracked_counts or {}
        self.quadrants = quadrants
        self.position_dict = position_dict if position_dict is not None else tag_node_dict
        self.untracked_positions = untracked_positions

        self.elimination_result: EliminationResult | QuadrantElimination | None = None
        self.frame_size: tuple[int, int] | None = None
        self.pre_elimination: MatchedDict | None = None

    def match(self) -> MatchedDict:
        """Perform the matching.

        Returns:
            ``matched[frame][track] -> tag`` (plain ``int`` tags). A track without enough
            evidence is absent from its frame's row (abstention); a missing track reads as ``""``.
            Eliminated or foreign tags follow the rules of :mod:`naps.closed_set`.

        Raises:
            RuntimeError: If a frame with tracked instances cannot be read from the video (it is
                shorter than ``video_last_frame``), or quadrant mode could not read any frame.
            KeyError: If ``tag_node_dict`` lacks a frame in the requested range.
        """
        decodes = self._decode_video()
        scope = self.quadrants

        if scope is not None:
            votes = self._collect_quadrant_votes(scope, decodes)
        else:
            self._warn_if_closed_set_is_small(decodes)
            if self.evidence_sources or self.closed_set is not None:
                votes = collect_votes(
                    decodes,
                    self.tag_node_dict,
                    self.evidence_sources,
                    allowed_tags=self.closed_set,
                )
            else:
                votes = decodes

        job_cost_matrix = CostMatrix.fromDict(
            votes,
            self.video_first_frame,
            self.video_last_frame,
            self.half_rolling_window_size,
            strict_carry_forward=self.strict_carry_forward,
        )
        matched = job_cost_matrix.assignTrackTagPairs()
        self.pre_elimination = matched

        if scope is not None:
            if self.eliminate and scope.active:
                return self._eliminate_quadrants(scope, matched)
            return matched
        if self.closed_set is not None and self.eliminate:
            return self._eliminate_closed_set(self.closed_set, matched)
        return matched

    # ------------------------------------------------------------------ stages
    def _decode_video(self) -> Decodes:
        """Read the video and decode the tags around every track in every frame.

        Frames past the end of the video are tolerated while no track has a position there
        (they yield no votes); a track with a position in an unreadable frame raises. With a
        ``decode_source`` the decodes come from it and the video is not read.
        """
        if self.decode_source is not None:
            decodes, self.frame_size = self.decode_source.decode(
                self.video_first_frame, self.video_last_frame, self.tag_node_dict
            )
            return decodes
        decodes, self.frame_size = decode_video(
            self.video_filename,
            self.video_first_frame,
            self.video_last_frame,
            self.tag_node_dict,
            self.aruco_crop_size,
            self.marker_detector,
        )
        return decodes

    def _warn_if_closed_set_is_small(self, decodes: Decodes) -> None:
        """Warn when frames hold many more tracks than the closed set has tags."""
        if self.closed_set is None:
            return
        n_tracks = [len(tracks) for tracks in decodes.values()]
        median_tracks = float(np.median(n_tracks)) if n_tracks else 0.0
        if median_tracks > len(self.closed_set) + _CLOSED_SET_TRACK_SLACK:
            logger.warning(
                "closed_set has %s tags but frames have a median of %s tracks: decodes of "
                "all other tags are discarded and elimination will abstain. The closed set "
                "assumes the input holds only these animals (e.g. a per-quadrant crop).",
                len(self.closed_set),
                median_tracks,
            )

    def _collect_quadrant_votes(
        self, scope: QuadrantScope, decodes: Decodes
    ) -> dict[int, FrameVotes]:
        """Prepare the quadrant scope (infer tag sets if needed) and collect filtered votes."""
        if self.frame_size is None:
            msg = "quadrant mode needs the video frame size, but no frame could be read"
            raise RuntimeError(msg)
        scope.prepare(self.frame_size, self.position_dict)
        if scope.tag_sets is None:
            scope.infer_from_decodes(decodes)
        n_tracks: defaultdict[str, list[int]] = defaultdict(list)
        for frame, tracks in decodes.items():
            per_region: defaultdict[str | None, int] = defaultdict(int)
            for track in tracks:
                per_region[scope.quadrant_of(frame, track)] += 1
            for region in scope.regions:
                n_tracks[region].append(per_region[region])
        for region in scope.regions:
            logger.info(
                "Region %s: tags %s, median %s tracks per frame",
                region,
                scope.tag_set(region) or "none (legacy behaviour)",
                float(np.median(n_tracks[region])) if n_tracks[region] else 0.0,
            )
        votes = collect_votes(
            decodes,
            self.tag_node_dict,
            self.evidence_sources,
            allowed_tags=None,
            vote_filter=scope.vote_filter,
        )
        logger.info("Quadrant vote filter rejected %s", dict(scope.rejected) or "nothing")
        return votes

    def _eliminate_quadrants(self, scope: QuadrantScope, matched: MatchedDict) -> MatchedDict:
        """Run closed-set elimination once per region."""
        frames = range(self.video_first_frame, self.video_last_frame + 1)
        result = eliminate_by_quadrant(
            {f: matched[f] for f in frames if f in matched},
            {f: self.position_dict[f] for f in frames if f in self.position_dict},
            scope,
            max_jump=self.elimination_max_jump,
            max_look=self.elimination_max_look,
            max_distance=self.elimination_max_distance,
            untracked_positions=self.untracked_positions,
            untracked_counts=self.untracked_counts,
        )
        self.elimination_result = result
        logger.info(
            "Quadrant closed-set elimination assigned %s instances; abstained %s; "
            "dropped %s foreign tags",
            result.n_assigned,
            dict(result.abstained),
            result.dropped_foreign,
        )
        return result.matched

    def _eliminate_closed_set(self, closed_set: list[int], matched: MatchedDict) -> MatchedDict:
        """Run flat closed-set elimination as a fail-safe post-pass."""
        frames = range(self.video_first_frame, self.video_last_frame + 1)
        result = eliminate_identities(
            {f: matched[f] for f in frames if f in matched},
            {f: self.tag_node_dict[f] for f in frames if f in self.tag_node_dict},
            closed_set,
            max_jump=self.elimination_max_jump,
            max_look=self.elimination_max_look,
            untracked_counts=self.untracked_counts,
            max_distance=self.elimination_max_distance,
        )
        self.elimination_result = result
        return result.matched


class MatchFrame:
    """The image of one video frame plus the per-track marker crops taken from it.

    Args:
        frame_exists: Whether the frame was read correctly (``ret`` of ``VideoCapture.read``).
        frame: Value array of the frame image (``None`` if it could not be read).
        *args: Ignored; accepted for backward compatibility.
        **kwargs: Ignored; accepted for backward compatibility.

    Attributes:
        frame: Value array of the frame image.
        frame_exists: Boolean indicating if the frame was read correctly.
        frame_images: Cropped images by track (``None`` for a track without a usable position).
    """

    def __init__(
        self,
        frame_exists: bool,
        frame: Image | None,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Store the frame; crops are added by :meth:`cropMarkerWithCoordsArray`."""
        self.frame = frame
        self.frame_exists = frame_exists
        self.frame_images: dict[Hashable, Image | None] = {}

    @classmethod
    def fromCV2(  # noqa: N802 (public camelCase API)
        cls,
        frame_exists: bool,
        frame: Image | None,
        *args: object,
        **kwargs: object,
    ) -> "MatchFrame":
        """Create a MatchFrame from the output of ``cv2.VideoCapture.read()``.

        Args:
            frame_exists: Whether the frame was read correctly (``ret`` of ``read()``).
            frame: The frame image as returned by ``read()`` (``None`` on failure).
            *args: Ignored; accepted for backward compatibility.
            **kwargs: Ignored; accepted for backward compatibility.

        Returns:
            A MatchFrame object.
        """
        return cls(frame_exists, frame, *args, **kwargs)

    def cropMarkerWithCoordsArray(  # noqa: N802 (public camelCase API)
        self, coords_dict: TrackCoords, crop_size: int
    ) -> None:
        """Create a cropped image for each track in ``coords_dict`` (stored in ``frame_images``).

        Args:
            coords_dict: ``coords_dict[track] -> (x, y)`` marker coordinates of each track. A
                track with a NaN coordinate gets ``None`` instead of an image.
            crop_size: Half-width of the square crop in pixels (clipped to the frame).

        Raises:
            RuntimeError: If a track has a usable position but the frame image is missing.
        """
        for track, coords in coords_dict.items():
            if np.isnan(coords).any():
                self.frame_images[track] = None
                continue
            if self.frame is None:
                msg = (
                    "the video frame could not be read (is the last frame beyond the end of "
                    "the video?) but a track has a position in it"
                )
                raise RuntimeError(msg)

            y_min, y_max = self._cropped_coords(coords[1], crop_size, self.frame.shape[0])
            x_min, x_max = self._cropped_coords(coords[0], crop_size, self.frame.shape[1])
            self.frame_images[track] = self.frame[y_min:y_max, x_min:x_max, 0]

    @staticmethod
    def _cropped_coords(coord: float, crop_size: float, coord_max: int) -> tuple[int, int]:
        """Return the (min, max) pixel range of a crop centred on ``coord``, clipped to the frame.

        Args:
            coord: Coordinate to get the crop range of (truncated to an int).
            crop_size: Half-width of the crop.
            coord_max: Size of the frame along this axis (the range ends at ``coord_max - 1``).
        """
        return (
            int(np.maximum(int(coord) - crop_size, 0)),
            int(np.minimum(int(coord) + crop_size, coord_max - 1)),
        )

    def returnMarkerTags(  # noqa: N802 (public camelCase API)
        self, marker_detect: MarkerDetector
    ) -> defaultdict[Hashable, list[Hashable | None]]:
        """Detect marker tags in every track image with ``marker_detect``.

        Args:
            marker_detect: Marker detector function; returns a list of marker assignments
                for an image.

        Returns:
            ``tags[track] -> [tag | None, ...]`` for this frame. A track without an image gets
            ``[None]``.
        """
        track_tag_dict: defaultdict[Hashable, list[Hashable | None]] = defaultdict(list)
        for track, track_image in self.frame_images.items():
            if track_image is None:
                track_tag_dict[track].append(None)
                continue
            track_tag_dict[track].extend(marker_detect(track_image))
        return track_tag_dict
