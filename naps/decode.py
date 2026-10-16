"""The decode stage: read video frames, crop around each animal's tag node, decode the tags.

This module is the public API of the decode stage; :mod:`naps.matching` consumes its output. The
pieces:

* :class:`DecodeParams` and :class:`TagDecoder`: a configurable ArUco decoder for one crop (or a
  batch of crops) that returns :class:`~naps.aruco.Detection` objects (tag, corners, confidence)
  and can restrict the decoder to a closed set of tags, relax the error correction, preprocess the
  crop (contrast, upsampling), cascade a second detector pass for crops the first one missed, and
  accept only markers lying where the tag node says the tag is.
* :func:`crop_around`, :func:`decode_frame` and :func:`decode_video`: the legacy cropping rule,
  one frame of crops, and the whole video loop (``decode_video`` replaces
  ``Matching._decode_video``: it returns a :class:`DecodeResult` whose ``votes`` attribute is the
  ``votes[frame][track] -> [tag | None, ...]`` structure the cost matrix consumes).
* :class:`SkipPolicy`: an optional speed knob that skips crops of tracks whose tag has been
  decoded identically several times in a row and has not moved.
* :func:`detect_dictionary`: pick the ArUco dictionary from a few frames.

``TagDecoder()`` with the default :class:`DecodeParams` is the *lab production* detector (no
closed set, no position gate, no preprocessing); it is **not** the ``naps-track`` command-line
default (windows 10/30/12, constant 3, rate 1). Exact parity with a configured ``ArUcoModel`` is
obtained by passing ``model.detect`` itself as the legacy detector of :func:`decode_video`. The
closed-set preset and the fast CLI decoder additionally raise ``minMarkerPerimeterRate`` to 0.15
(:data:`FAST_MIN_MARKER_PERIMETER_RATE`), which leaves every decode unchanged and makes detection
faster. The measurements behind the presets (Beszel Trial2 camera0 only; re-measure on other
videos) are summarised in ``docs/decode_measurements.md``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from naps.aruco import ARUCO_TAG_SETS, ArUcoModel, Detection, closed_set_correction_bits
from naps.tags import require_tag
from naps.video import FastVideoReader, Frame, VideoReadError

__all__ = [
    "CLOSED_SET_PASSES",
    "FAST_MIN_MARKER_PERIMETER_RATE",
    "DecodeParams",
    "DecodeResult",
    "DictionaryGuess",
    "SkipPolicy",
    "TagDecoder",
    "crop_around",
    "decode_frame",
    "decode_video",
    "detect_dictionary",
    "guess_dictionary",
]

logger = logging.getLogger(__name__)

Votes = defaultdict[Hashable, list[Hashable | None]]
"""``votes[track] -> [tag | None, ...]`` of one frame (``[None]`` when nothing was decoded)."""
Coords = Mapping[Hashable, Sequence[float]]
"""``coords[track] -> (x, y)``: the tag node of every track in one frame."""
ClosedSets = (
    Iterable[int]
    | Mapping[Hashable, Iterable[int]]
    | Callable[[int, Hashable], Iterable[int] | None]
    | None
)
"""One closed set for every track, ``{track: closed set}``, ``(frame, track) -> closed set`` or
``None`` (no override; the decoder's own set applies). Iterables are read once and kept as tuples
(:func:`decode_frame` and :func:`decode_video` do that), so one-shot iterators are safe."""
LegacyDetector = Callable[[Frame], Iterable[Hashable | None]]
"""A legacy marker detector: crop in, list of tags (``None``: nothing found) out."""

Strategy = Literal["cascade", "union"]

_INTERPOLATION = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}
_MIN_WIN = 3
_STABLE_REPEATS = 3


@dataclass(frozen=True)
class DecodeParams:
    """Parameters of one detector pass.

    The defaults are the settings the lab's production pipeline passes to NAPS (its own
    configuration, not part of this repository: ``DICT_5X5_100``, adaptive threshold windows 3 to
    23 step 10, constant 7, error-correction rate 0.6, sub-pixel corners, margin 0.1), **not** the
    ``naps-track`` command-line defaults (windows 10/30/12, constant 3, rate 1). For exact parity
    with a given ``ArUcoModel`` pass ``model.detect`` itself as the legacy detector of
    :func:`decode_video`.

    Attributes:
        tag_set: OpenCV dictionary name (:data:`~naps.aruco.ARUCO_TAG_SETS`).
        closed_set: Tag ids the decoder may return (``None`` or empty: the whole dictionary).
            With a closed set most misreads are rejected by construction, which is what makes a
            higher ``error_correction_rate`` safe. Can be overridden per call.
        win_min: Smallest adaptive-threshold window (pixels, of the preprocessed image).
        win_max: Largest adaptive-threshold window.
        win_step: Window increment.
        thresh_const: Constant subtracted by the adaptive threshold (grey levels).
        margin: ``perspectiveRemoveIgnoredMarginPerCell``.
        error_correction_rate: OpenCV corrects ``int(max_bits * rate)`` bits, so with the 3 bits
            of ``DICT_5X5_100`` the rates 0.34-0.66 correct 1 bit (0.6 is production), 0.67-0.99
            2 bits and 1.0 3 bits.
        correction_bits: Bit capacity ``max_bits`` of the closed-set dictionary: ``None`` keeps
            the full dictionary's; ``"closed_set"`` uses the set's own (larger) capacity, see
            :func:`~naps.aruco.closed_set_correction_bits`; an ``int`` sets it.
        corner_refinement: ``"none"``, ``"subpix"`` (production), ``"contour"`` or ``"apriltag"``.
        scale: Resize factor applied to the crop before detecting (``1.0``: none); the threshold
            windows are scaled with it.
        interpolation: ``"cubic"`` or ``"linear"`` resampling when ``scale != 1``.
        clahe: ``(clip_limit, tile_grid)`` of a CLAHE contrast equalisation, or ``None``.
        overrides: Extra ``cv2.aruco.DetectorParameters`` fields as ``(name, value)`` pairs.
        min_marker_perimeter_rate: ``minMarkerPerimeterRate`` of the detector, a fraction of the
            larger side of the (resampled) image: OpenCV discards candidate contours whose perimeter
            is below ``rate * max(width, height)`` pixels. ``None`` (default) keeps OpenCV's 0.03,
            which is 3 px on a 100 px crop and so admits many tiny noise quads that the detector
            then has to reject one by one. Raising it to
            :data:`FAST_MIN_MARKER_PERIMETER_RATE` (0.15) makes detection up to 8x faster with
            the same decodes (measurements and the safe limit are in that constant); the
            closed-set preset (:data:`CLOSED_SET_PASSES`) and ``naps-track --decoder fast`` use
            it, while the plain ``DecodeParams()`` stays the legacy decoder. The rate is relative
            to the image, so it needs no rescaling when ``scale`` changes; for a different crop
            size keep ``rate * crop side`` below about 48 px. An explicit
            ``minMarkerPerimeterRate`` entry in ``overrides`` wins over this field. Must be
            positive.
    """

    tag_set: str = "DICT_5X5_100"
    closed_set: tuple[int, ...] | None = None
    win_min: int = 3
    win_max: int = 23
    win_step: int = 10
    thresh_const: float = 7.0
    margin: float = 0.1
    error_correction_rate: float = 0.6
    correction_bits: int | Literal["closed_set"] | None = None
    corner_refinement: str = "subpix"
    scale: float = 1.0
    interpolation: Literal["linear", "cubic"] = "cubic"
    clahe: tuple[float, int] | None = None
    overrides: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    min_marker_perimeter_rate: float | None = None


FAST_MIN_MARKER_PERIMETER_RATE = 0.15
"""``minMarkerPerimeterRate`` that speeds the detector up at identical decodes.

OpenCV discards candidate contours whose perimeter is below ``rate * max(width, height)`` pixels.
The default 0.03 is 3 px on a 100 px crop, so many tiny noise quads reach the (expensive) marker
test, while the tags are 24 to 41 px per side (96 to 164 px of perimeter) in the lab's
videos. Measured on 53,640 real crops from 61 windows of 12 colony-trials and both cameras (hours
0 to 112, separated and huddled bees; summary in ``docs/decode_measurements.md``): for the legacy
full-dictionary decoder, a single closed-set pass and the
3-pass preset, at crop half-widths 50, 64 and 100 and with x1.5 / x2 upsampling, any rate from
0.10 to 0.30 changed not one crop (0 lost, 0 gained, every vote list identical, wrong-in-set rate
unchanged; decoy false accepts identical at 0.15, at most 1-3 crops of 53,640 apart at other
rates), and 16,137 frame-by-track votes of ``decode_video`` on full frames were identical too.
Speed (one thread, per crop): the legacy decoder 6.8x (100 px crop) to 8.4x (128 px) faster, a
single closed-set pass 1.7x, the 3-pass preset about 2x, and about 1.0 to 1.6x for 200 px crops
or x2 upsampling, where the noise quads are a smaller share of the work.

The rate is relative to the image, so ``DecodeParams.scale`` needs no rescaling (the threshold
grows with the resampled image exactly as the tag does). The limit is physical: recall falls
once ``rate * max(crop side)`` exceeds the perimeter of the smallest tag (96 px here): 0.8 on a
128 px crop lost 0.4 to 0.8 points, 0.6 on a 200 px crop lost 7 to 17 points. Keep
``rate * max(crop side)`` below about 48 px (half of the smallest tag's perimeter): 0.15 is safe
for crops up to 320 px, while for larger crops lower the rate."""

_PIXELS_PER_CELL = (("perspectiveRemovePixelPerCell", 12),)
_FAST = FAST_MIN_MARKER_PERIMETER_RATE
CLOSED_SET_PASSES: tuple[DecodeParams, ...] = (
    DecodeParams(error_correction_rate=1.0, min_marker_perimeter_rate=_FAST),
    DecodeParams(
        error_correction_rate=1.0,
        margin=0.3,
        win_min=3,
        win_max=33,
        win_step=3,
        overrides=_PIXELS_PER_CELL,
        min_marker_perimeter_rate=_FAST,
    ),
    DecodeParams(
        error_correction_rate=1.0,
        margin=0.3,
        win_min=3,
        win_max=33,
        win_step=3,
        clahe=(2.0, 4),
        overrides=_PIXELS_PER_CELL,
        min_marker_perimeter_rate=_FAST,
    ),
)
"""Detector cascade for a *closed set* of a few tags (measured on the production videos).

Pass 1 is the production detector with the full error correction (3 bits for ``DICT_5X5_100``;
only safe because the closed set rejects almost every misread), pass 2 adds a finer threshold
sweep and a wider sampling margin, pass 3 adds CLAHE contrast equalisation for low-contrast tags.
A pass runs only on crops that the previous ones left empty. The closed set is
``DecodeParams.closed_set`` (or the ``closed_set`` argument of :meth:`TagDecoder.decode`)."""


class TagDecoder:
    """Decode ArUco tags in crops with one or more detector passes.

    Args:
        passes: One :class:`DecodeParams` or a sequence of them (tried in order).
        strategy: ``"cascade"`` (default) runs the next pass only on crops where the previous
            ones found nothing acceptable; ``"union"`` runs every pass and merges the results
            (one detection per tag, the most confident).
        position_radius: If set, only markers whose centre lies within this many pixels of the
            expected position (default: the crop centre; pass ``expected=`` to a decode call to
            say otherwise) are accepted. ``None`` (default) accepts every marker in the crop, as
            legacy NAPS does (a neighbour's tag inside the crop is then a vote for this track).
        min_confidence: Drop detections whose OpenCV confidence is below this value (``nan``
            confidences are kept). ``0.0`` keeps everything.
        legacy_lut: Optional 256-entry table applied to every crop before detection, to reproduce
            the intensities of the old ``read()`` path from luma crops (see
            :meth:`~naps.video.FastVideoReader.to_legacy`).

    Raises:
        ValueError: If ``passes`` is empty or a ``scale`` or ``min_marker_perimeter_rate`` is not
            positive.
    """

    def __init__(
        self,
        passes: DecodeParams | Sequence[DecodeParams] = DecodeParams(),  # noqa: B008 (frozen)
        *,
        strategy: Strategy = "cascade",
        position_radius: float | None = None,
        min_confidence: float = 0.0,
        legacy_lut: NDArray[np.uint8] | None = None,
    ) -> None:
        """Store the configuration; the OpenCV detectors are built lazily."""
        self.passes = (passes,) if isinstance(passes, DecodeParams) else tuple(passes)
        if not self.passes:
            msg = "at least one DecodeParams pass is required"
            raise ValueError(msg)
        if any(p.scale <= 0 for p in self.passes):
            msg = "scale must be positive"
            raise ValueError(msg)
        if any(
            p.min_marker_perimeter_rate is not None and not p.min_marker_perimeter_rate > 0
            for p in self.passes
        ):
            msg = "min_marker_perimeter_rate must be positive"
            raise ValueError(msg)
        self.strategy = strategy
        self.position_radius = position_radius
        self.min_confidence = min_confidence
        self.legacy_lut = legacy_lut
        self._models: dict[tuple[int, tuple[int, ...] | None], ArUcoModel] = {}
        self._clahe: dict[tuple[float, int], cv2.CLAHE] = {}

    def __getstate__(self) -> dict[str, object]:
        """Pickle the configuration only (OpenCV detectors are rebuilt lazily)."""
        state = self.__dict__.copy()
        state["_models"] = {}
        state["_clahe"] = {}
        return state

    # ------------------------------------------------------------------ one crop
    def decode(
        self,
        image: NDArray[np.generic] | None,
        *,
        closed_set: Iterable[int] | None = None,
        expected: Sequence[float] | None = None,
    ) -> list[Detection]:
        """Decode the markers in one crop.

        Args:
            image: 2-D grayscale crop (``None`` or an empty array gives no detections).
            closed_set: Tag ids to restrict the decoder to for this call (overrides the passes'
                ``closed_set``). ``None`` **or empty** means "no override": the decoder's own
                set applies (an empty answer never widens a restricted decoder to the whole
                dictionary).
            expected: ``(x, y)`` where the tag should be, in crop pixels, for the position gate
                (default: the crop centre).

        Returns:
            The accepted detections, in the pixels of ``image``, nearest to ``expected`` first
            (legacy detector order is not preserved).
        """
        if image is None or 0 in image.shape:
            return []
        override = _tag_tuple(closed_set)
        centre = (
            (image.shape[1] / 2.0, image.shape[0] / 2.0)
            if expected is None
            else (float(expected[0]), float(expected[1]))
        )
        accepted: dict[int, Detection] = {}
        for index, params in enumerate(self.passes):
            found = self._run_pass(index, params, image, override, centre)
            for detection in found:
                best = accepted.get(detection.tag)
                if best is None or _confidence(detection) > _confidence(best):
                    accepted[detection.tag] = detection
            if accepted and self.strategy == "cascade":
                break
        return sorted(accepted.values(), key=lambda d: _distance(d.center, centre))

    def tags(
        self,
        image: NDArray[np.generic] | None,
        *,
        closed_set: Iterable[int] | None = None,
        expected: Sequence[float] | None = None,
    ) -> list[int | None]:
        """Decode a crop into the legacy vote list: the tag ids, or ``[None]`` if none."""
        found = self.decode(image, closed_set=closed_set, expected=expected)
        return [d.tag for d in found] or [None]

    def __call__(self, image: NDArray[np.generic]) -> list[int | None]:
        """Marker-detector protocol of :class:`naps.matching.Matching`: crop in, votes out."""
        return self.tags(image)

    def decode_batch(
        self,
        crops: Mapping[Hashable, NDArray[np.generic] | None],
        *,
        closed_set: Iterable[int] | None = None,
        expected: Mapping[Hashable, Sequence[float]] | None = None,
    ) -> dict[Hashable, list[Detection]]:
        """Decode many crops (for example every track of one frame).

        Args:
            crops: ``crops[track] -> crop`` (``None`` for a track without a usable position).
            closed_set: As in :meth:`decode`, for every crop.
            expected: Optional ``expected[track] -> (x, y)`` in crop pixels for the position gate.

        Returns:
            ``detections[track]`` for every key of ``crops`` (empty for a ``None`` crop).
        """
        closed = None if closed_set is None else tuple(closed_set)
        return {
            track: self.decode(
                crop,
                closed_set=closed,
                expected=None if expected is None else expected.get(track),
            )
            for track, crop in crops.items()
        }

    # ------------------------------------------------------------------ internals
    def _run_pass(
        self,
        index: int,
        params: DecodeParams,
        image: NDArray[np.generic],
        override: tuple[int, ...] | None,
        centre: tuple[float, float],
    ) -> list[Detection]:
        """Run one detector pass and keep the acceptable detections (in input pixels)."""
        closed = override if override is not None else params.closed_set
        built = self._model(index, params, closed)
        prepared = self._preprocess(image, params)
        scale = params.scale
        found = []
        for detection in built.detect_detailed(prepared):
            corners = (detection.corners / scale).astype(np.float32)
            scaled = Detection(detection.tag, corners, detection.confidence)
            if self._accepts(scaled, centre):
                found.append(scaled)
        return found

    def _accepts(self, detection: Detection, centre: tuple[float, float]) -> bool:
        """Position and confidence gates."""
        if self.position_radius is not None and (
            _distance(detection.center, centre) > self.position_radius
        ):
            return False
        confidence = detection.confidence
        return not (confidence == confidence and confidence < self.min_confidence)  # noqa: PLR0124 (nan test)

    def _model(
        self, index: int, params: DecodeParams, closed: tuple[int, ...] | None
    ) -> ArUcoModel:
        """Build (once) the detector of pass ``index`` for the closed set ``closed``."""
        key = (index, closed)
        built = self._models.get(key)
        if built is None:
            window = [
                max(_MIN_WIN, round(w * params.scale))
                for w in (params.win_min, params.win_max, params.win_step)
            ]
            bits = params.correction_bits
            if bits == "closed_set":
                bits = closed_set_correction_bits(params.tag_set, closed) if closed else None
            model = ArUcoModel.withTagSet(
                params.tag_set,
                window[0],
                window[1],
                window[2],
                params.thresh_const,
                params.margin,
                params.error_correction_rate,
                closed,
                corner_refinement=params.corner_refinement,
                detector_overrides=_detector_overrides(params),
                subset_correction_bits=bits,
            )
            model.build()
            built = self._models[key] = model
        return built

    def _preprocess(self, image: NDArray[np.generic], params: DecodeParams) -> NDArray[np.uint8]:
        """Legacy intensity remap, contrast equalisation and resampling of a crop."""
        prepared = np.ascontiguousarray(image, dtype=np.uint8)
        if self.legacy_lut is not None:
            prepared = np.asarray(cv2.LUT(prepared, self.legacy_lut), dtype=np.uint8)
        if params.clahe is not None:
            clahe = self._clahe.get(params.clahe)
            if clahe is None:
                clip, tile = params.clahe
                clahe = self._clahe[params.clahe] = cv2.createCLAHE(
                    clipLimit=clip, tileGridSize=(tile, tile)
                )
            prepared = np.asarray(clahe.apply(prepared), dtype=np.uint8)
        if params.scale != 1.0:
            resized = cv2.resize(
                prepared,
                None,
                fx=params.scale,
                fy=params.scale,
                interpolation=_INTERPOLATION[params.interpolation],
            )
            prepared = np.asarray(resized, dtype=np.uint8)
        return prepared

    @classmethod
    def for_closed_set(
        cls,
        closed_set: Iterable[int],
        *,
        rescue_passes: int = len(CLOSED_SET_PASSES),
        tag_set: str = "DICT_5X5_100",
        position_radius: float | None = None,
        legacy_lut: NDArray[np.uint8] | None = None,
        min_marker_perimeter_rate: float | None = FAST_MIN_MARKER_PERIMETER_RATE,
    ) -> TagDecoder:
        """Build the measured-best decoder for a known small set of tags.

        Uses :data:`CLOSED_SET_PASSES` restricted to ``closed_set``. On the production videos
        (``DICT_5X5_100``, 4 tags, 111,700 crops) this finds the reference tag at the tag node
        in 97.5% of the crops against 91.6% for the legacy decoder and about 3x faster per crop,
        with no measurable increase in wrong in-set decodes (0.388% against 0.377%, both
        reference noise). It is not free of false accepts: with the three passes a decoder
        restricted to four tag ids that are not in the video accepted one of them in 8 of
        446,800 trials (0.0018%), and the unrestricted relaxed decoder returned an out-of-set id
        in 7 of 111,700 crops; use ``rescue_passes=2`` (0 of 446,800) to trade 1 point of recall
        for none observed.

        Args:
            closed_set: The colony's tag ids. Keep it small (the safety comes from the set being
                a small fraction of the dictionary); it must not be empty.
            rescue_passes: How many of the three passes to use (1 to 3).
            tag_set: Dictionary name.
            position_radius: Optional position gate, see the class docstring.
            legacy_lut: Optional intensity remap, see the class docstring.
            min_marker_perimeter_rate: ``minMarkerPerimeterRate`` of every pass (default
                :data:`FAST_MIN_MARKER_PERIMETER_RATE`: same decodes, 2 to 8x faster detection);
                ``None`` keeps OpenCV's default.

        Returns:
            The decoder.

        Raises:
            ValueError: If ``closed_set`` is empty, ``rescue_passes`` is not 1 to 3 or
                ``min_marker_perimeter_rate`` is not positive.
        """
        tags = tuple(sorted(require_tag(t) for t in closed_set))
        if not tags:
            msg = "a closed set must hold at least one tag"
            raise ValueError(msg)
        if not 1 <= rescue_passes <= len(CLOSED_SET_PASSES):
            msg = f"rescue_passes must be between 1 and {len(CLOSED_SET_PASSES)}"
            raise ValueError(msg)
        passes = [
            replace(
                p,
                tag_set=tag_set,
                closed_set=tags,
                min_marker_perimeter_rate=min_marker_perimeter_rate,
            )
            for p in CLOSED_SET_PASSES
        ]
        return cls(passes[:rescue_passes], position_radius=position_radius, legacy_lut=legacy_lut)

    def with_closed_set(self, closed_set: Iterable[int] | None) -> TagDecoder:
        """Return a copy of this decoder whose passes are restricted to ``closed_set``."""
        tags = _tag_tuple(closed_set)
        return TagDecoder(
            [replace(p, closed_set=tags) for p in self.passes],
            strategy=self.strategy,
            position_radius=self.position_radius,
            min_confidence=self.min_confidence,
            legacy_lut=self.legacy_lut,
        )

    def with_legacy_lut(self, legacy_lut: NDArray[np.uint8] | None) -> TagDecoder:
        """Return a copy of this decoder that remaps crops through ``legacy_lut`` first."""
        return TagDecoder(
            self.passes,
            strategy=self.strategy,
            position_radius=self.position_radius,
            min_confidence=self.min_confidence,
            legacy_lut=legacy_lut,
        )


def _detector_overrides(params: DecodeParams) -> dict[str, float]:
    """The ``DetectorParameters`` fields of a pass: its rate, then ``overrides`` (they win)."""
    fields: dict[str, float] = {}
    if params.min_marker_perimeter_rate is not None:
        fields["minMarkerPerimeterRate"] = params.min_marker_perimeter_rate
    fields.update(params.overrides)
    return fields


def _tag_tuple(closed_set: Iterable[int] | None) -> tuple[int, ...] | None:
    """Sorted tuple of a closed set; ``None`` for no set and for an empty one."""
    if closed_set is None:
        return None
    return tuple(sorted(require_tag(t) for t in closed_set)) or None


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance of two points."""
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _confidence(detection: Detection) -> float:
    """Detection confidence with ``nan`` treated as the lowest."""
    return -1.0 if detection.confidence != detection.confidence else detection.confidence


# ====================================================================== cropping and the video loop
def crop_around(frame: NDArray[np.generic], xy: Sequence[float], half: int) -> Frame | None:
    """Crop a square window around ``xy`` with the legacy NAPS rule.

    The window is ``[int(c) - half, int(c) + half)`` per axis, clipped to ``[0, size - 1)`` (the
    last pixel row/column is never included; kept for compatibility).

    Args:
        frame: 2-D frame.
        xy: ``(x, y)`` of the tag node.
        half: Half-width of the window in pixels.

    Returns:
        The crop (a view into ``frame`` when it is ``uint8``, else a converted copy); ``None`` if
        either coordinate is not finite.
    """
    if not np.isfinite(xy).all():
        return None
    height, width = frame.shape[:2]
    y0, y1 = max(int(xy[1]) - half, 0), min(int(xy[1]) + half, height - 1)
    x0, x1 = max(int(xy[0]) - half, 0), min(int(xy[0]) + half, width - 1)
    return np.asarray(frame[y0:y1, x0:x1], dtype=np.uint8)


def _crop_origin(xy: Sequence[float], half: int) -> tuple[int, int]:
    """``(x0, y0)`` of the window :func:`crop_around` takes (for expected positions)."""
    return max(int(xy[0]) - half, 0), max(int(xy[1]) - half, 0)


Decoder = TagDecoder | LegacyDetector
"""What :func:`decode_video` accepts: a :class:`TagDecoder` or a legacy crop to votes callable."""


@dataclass
class SkipPolicy:
    """Skip decoding tracks whose tag is already known and which have not moved.

    A track becomes skippable after ``confirmations`` consecutive decodes returned the same tag;
    it is then skipped for up to ``refresh - 1`` frames as long as its tag node stays within
    ``max_step`` pixels per frame of its last decoded position. The skipped frame's vote is the
    carried tag (``carry=True``) or empty. Any decode that disagrees resets the track.

    Attributes:
        confirmations: Consecutive identical decodes needed before skipping.
        refresh: A track is decoded again after this many frames.
        max_step: Largest allowed displacement (pixels per frame) since the last decode.
        carry: Whether a skipped frame votes for the carried tag.
    """

    confirmations: int = 3
    refresh: int = 3
    max_step: float = 3.0
    carry: bool = True
    _state: dict[Hashable, tuple[Hashable, int, float, float, int]] = field(
        default_factory=dict, repr=False
    )  # track -> (tag, streak, x, y, frame decoded)

    def skip_tag(self, track: Hashable, frame: int, xy: Sequence[float]) -> Hashable | None:
        """Return the tag to carry if ``track``'s crop at ``frame`` can be skipped, else None."""
        state = self._state.get(track)
        if state is None:
            return None
        tag, streak, x, y, last = state
        gap = frame - last
        if streak < self.confirmations or not 0 < gap < self.refresh:
            return None
        if _distance((xy[0], xy[1]), (x, y)) > self.max_step * gap:
            return None
        return tag

    def observe(
        self, track: Hashable, frame: int, xy: Sequence[float], tag: Hashable | None
    ) -> None:
        """Record the outcome of a decode of ``track`` at ``frame``."""
        state = self._state.get(track)
        if tag is None:
            self._state.pop(track, None)
            return
        streak = state[1] + 1 if state is not None and state[0] == tag else 1
        self._state[track] = (tag, streak, float(xy[0]), float(xy[1]), frame)

    def reset(self) -> None:
        """Forget every track."""
        self._state.clear()


@dataclass
class DecodeResult:
    """What :func:`decode_video` produces.

    Attributes:
        votes: ``votes[frame][track] -> [tag | None, ...]``, the structure
            ``Matching._decode_video`` returned (feed it to ``CostMatrix.fromDict``).
        detections: ``detections[frame][track] -> [Detection, ...]`` with corners in full-frame
            pixels (empty for skipped crops and tracks without a position).
        frame_size: ``(width, height)`` of the frames, or ``None`` if none could be read.
        n_decoded: Number of crops decoded.
        n_skipped: Number of crops skipped by the :class:`SkipPolicy`.
        truncated_at: First frame that could not be read because the video ended early.
    """

    votes: dict[int, Votes] = field(default_factory=dict)
    detections: dict[int, dict[Hashable, list[Detection]]] = field(default_factory=dict)
    frame_size: tuple[int, int] | None = None
    n_decoded: int = 0
    n_skipped: int = 0
    truncated_at: int | None = None


def decode_frame(
    frame: NDArray[np.generic],
    coords: Coords,
    decoder: TagDecoder,
    half: int,
    *,
    closed_set: ClosedSets = None,
    frame_index: int = 0,
) -> dict[Hashable, list[Detection]]:
    """Crop around every track's tag node and decode the crops of one frame.

    Args:
        frame: 2-D frame.
        coords: ``coords[track] -> (x, y)``; a NaN coordinate gives no detections.
        decoder: The decoder.
        half: Crop half-width (``--aruco-crop-size``).
        closed_set: Optional closed set for this frame (see :meth:`TagDecoder.decode`), a
            mapping ``track -> closed set`` (tracks missing from it are unrestricted) or a
            callable ``(frame_index, track) -> closed set | None``.
        frame_index: Index passed to a callable ``closed_set``.

    Returns:
        ``detections[track]`` with corners in full-frame pixels.
    """
    out: dict[Hashable, list[Detection]] = {}
    closed_set = _materialise(closed_set)
    for track, xy in coords.items():
        closed = _closed_for(closed_set, frame_index, track)
        out[track] = _decode_track(frame, xy, decoder, half, closed)
    return out


def _materialise(closed_set: ClosedSets) -> ClosedSets:
    """Closed sets that can be read any number of times.

    A plain iterable (for example a generator) or a mapping whose values are iterators would be
    used up by the first track, and every later track would silently be decoded against the
    whole dictionary (an empty set is no restriction). A callable is left alone: it answers
    afresh for every ``(frame, track)``.
    """
    if closed_set is None:
        return None
    if isinstance(closed_set, Mapping):
        # (a None entry means "no restriction", exactly as a missing track does)
        return {track: tuple(tags) for track, tags in closed_set.items() if tags is not None}
    if isinstance(closed_set, Iterable):
        return tuple(closed_set)
    return closed_set


def _closed_for(closed_set: ClosedSets, frame: int, track: Hashable) -> Iterable[int] | None:
    """The closed set that applies to ``track`` in ``frame``."""
    if isinstance(closed_set, Mapping):
        return closed_set.get(track)
    if closed_set is None or isinstance(closed_set, Iterable):
        return closed_set
    return closed_set(frame, track)


def _decode_track(
    frame: NDArray[np.generic],
    xy: Sequence[float],
    decoder: TagDecoder,
    half: int,
    closed_set: Iterable[int] | None,
) -> list[Detection]:
    """Decode one track's crop; detections in full-frame pixels."""
    crop = crop_around(frame, xy, half)
    if crop is None:
        return []
    x0, y0 = _crop_origin(xy, half)
    expected = (float(xy[0]) - x0, float(xy[1]) - y0)
    found = decoder.decode(crop, closed_set=closed_set, expected=expected)
    offset = np.array([x0, y0], dtype=np.float32)
    return [Detection(d.tag, d.corners + offset, d.confidence) for d in found]


class _LutDetector:
    """Wrap a legacy ``crop -> votes`` detector so it sees crops remapped through a table."""

    def __init__(self, detector: LegacyDetector, lut: NDArray[np.uint8]) -> None:
        self.detector = detector
        self.lut = lut

    def __call__(self, image: Frame) -> Iterable[Hashable | None]:
        """Remap ``image`` and detect."""
        return self.detector(np.asarray(cv2.LUT(image, self.lut), dtype=np.uint8))


def decode_video(
    video: FastVideoReader | str | Path,
    positions: Mapping[int, Coords],
    decoder: Decoder,
    *,
    first_frame: int,
    last_frame: int,
    crop_half: int = 50,
    closed_set: ClosedSets = None,
    skip: SkipPolicy | None = None,
    reader_mode: Literal["auto", "y", "bgr"] = "auto",
    legacy_pixels: bool = True,
    reader_threads: int | None = None,
) -> DecodeResult:
    """Decode the tags around every track in every frame of ``first_frame..last_frame``.

    Replacement for ``Matching._decode_video``. Frames past the end of the video are tolerated
    while no track has a position there (they yield empty rows); a track with a position in an
    unreadable frame raises.

    Args:
        video: An open :class:`~naps.video.FastVideoReader` (left open) or a video path.
        positions: ``positions[frame][track] -> (x, y)`` tag nodes; must hold every frame in the
            range.
        decoder: A :class:`TagDecoder`, or a legacy marker detector ``crop -> [tag | None, ...]``
            such as ``ArUcoModel.detect`` (then ``DecodeResult.detections`` stays empty and the
            per-call ``closed_set`` and the position gate do not apply).
        first_frame: First frame (inclusive, zero-based).
        last_frame: Last frame (inclusive).
        crop_half: Crop half-width in pixels (``naps-track`` passes ``--aruco-crop-size``,
            default 100; this function's own default is 50).
        closed_set: Optional closed set applied to every crop (overrides the decoder's), or a
            mapping ``track -> closed set`` for per-track (per-quadrant) sets, or a callable
            ``(frame, track) -> closed set | None`` (for example a quadrant lookup by position);
            a track missing from the mapping, or a ``None`` answer, uses the decoder's own set.
            Plain iterables and the values of a mapping may be one-shot iterators: they are
            turned into tuples once, before the first frame.
        skip: Optional :class:`SkipPolicy`.
        reader_mode: Mode of the :class:`~naps.video.FastVideoReader` opened for a path.
        legacy_pixels: Make the crops bit-identical to what the old ``read()`` path fed the
            decoder (default). With the fast luma reader this applies the measured intensity
            remap to each crop (microseconds); for colour video, where luma is not a pure remap
            of the blue channel, a path is read the old way instead. Pass ``False`` to decode the
            raw luma (the measured accuracy delta is in the decode benchmark report).
        reader_threads: Decoder threads of the :class:`~naps.video.FastVideoReader` opened for a
            path (``None``: OpenCV's default; 1 in each of several parallel workers).

    Returns:
        The :class:`DecodeResult`.

    Raises:
        KeyError: If ``positions`` lacks a frame of the range.
        VideoReadError: If a frame with tracked positions cannot be read, or a passed-in luma
            reader cannot reproduce the legacy pixels.
    """
    own = not isinstance(video, FastVideoReader)
    closed_set = _materialise(closed_set)  # a generator must serve every track and frame
    if isinstance(video, FastVideoReader):
        reader = video
    else:
        reader = FastVideoReader(video, mode=reader_mode, threads=reader_threads)
        if legacy_pixels and reader.luma_plane and reader.legacy_lut is None:
            # luma that is not a pure remap of the old pixels (colour video): read like legacy
            reader.close()
            reader = FastVideoReader(video, mode="bgr", threads=reader_threads)
    result = DecodeResult()
    try:
        if legacy_pixels and reader.luma_plane and reader.pixel_range != "full":
            lut = _legacy_lut(reader)
            if isinstance(decoder, TagDecoder):
                decoder = decoder.with_legacy_lut(lut)
            else:
                decoder = _LutDetector(decoder, lut)
        reader.seek(first_frame)
        for frame_index in range(first_frame, last_frame + 1):
            coords = positions[frame_index]
            image = reader.read()
            if reader.truncated_at is not None and result.truncated_at is None:
                result.truncated_at = reader.truncated_at
            if image is not None and result.frame_size is None:
                result.frame_size = (image.shape[1], image.shape[0])
            _decode_one_frame(
                result, frame_index, image, coords, decoder, crop_half, closed_set, skip, reader
            )
    finally:
        if own:
            reader.close()
    return result


def _legacy_lut(reader: FastVideoReader) -> NDArray[np.uint8]:
    """The reader's luma to ``read()`` table (raises if none is known)."""
    if reader.legacy_lut is None:
        msg = f"{reader.path}: no luma to read() remap is known; cannot reproduce legacy pixels"
        raise VideoReadError(msg)
    return reader.legacy_lut


def _decode_one_frame(  # noqa: PLR0917
    result: DecodeResult,
    frame_index: int,
    image: Frame | None,
    coords: Coords,
    decoder: Decoder,
    crop_half: int,
    closed_set: ClosedSets,
    skip: SkipPolicy | None,
    reader: FastVideoReader,
) -> None:
    """Decode one frame's crops into ``result``."""
    votes: Votes = defaultdict(list)
    detections: dict[Hashable, list[Detection]] = {}
    for track, xy in coords.items():
        if not np.isfinite(xy).all():
            votes[track].append(None)
            detections[track] = []
            continue
        if image is None:
            msg = (
                f"{reader.path}, frame {frame_index}: the video frame could not be read (is the "
                "last frame beyond the end of the video?) but a track has a position in it"
            )
            raise VideoReadError(msg)
        carried = None if skip is None else skip.skip_tag(track, frame_index, xy)
        if skip is not None and carried is not None:
            votes[track].append(carried if skip.carry else None)
            detections[track] = []
            result.n_skipped += 1
            continue
        if isinstance(decoder, TagDecoder):
            closed = _closed_for(closed_set, frame_index, track)
            found = _decode_track(image, xy, decoder, crop_half, closed)
            tags: list[Hashable | None] = [d.tag for d in found]
        else:
            found = []
            tags = list(decoder(cast("Frame", crop_around(image, xy, crop_half))))
        votes[track].extend(tags or [None])
        detections[track] = found
        result.n_decoded += 1
        if skip is not None:
            skip.observe(track, frame_index, xy, tags[0] if len(tags) == 1 else None)
    result.votes[frame_index] = votes
    result.detections[frame_index] = detections


# ====================================================================== dictionary detection
@dataclass(frozen=True)
class DictionaryGuess:
    """How well one ArUco dictionary explains a set of images.

    Attributes:
        tag_set: Dictionary name.
        n_decodes: Markers decoded (all ids).
        n_stable: Decodes whose id was decoded at least ``stable_repeats`` times in total (real
            tags recur; misreads scatter over many ids).
        tags: The ids decoded at least ``stable_repeats`` times, with their counts.
    """

    tag_set: str
    n_decodes: int
    n_stable: int
    tags: dict[int, int]


def detect_dictionary(
    images: Iterable[NDArray[np.generic]],
    candidates: Sequence[str] = ARUCO_TAG_SETS,
    *,
    stable_repeats: int = _STABLE_REPEATS,
    progress: Callable[[str], None] | None = None,
) -> list[DictionaryGuess]:
    """Rank ArUco dictionaries by how many stable (recurring) tag ids they decode in ``images``.

    Real tags are seen over and over, while a wrong dictionary decodes scattered random ids, so
    the count of decodes that belong to a recurring id separates them. Nested dictionaries (for
    example ``DICT_5X5_50`` inside ``DICT_5X5_100``) decode the same ids; ties are broken in
    favour of the smaller dictionary (the earlier entry of ``candidates``).

    Args:
        images: A few frames or crops (2-D grayscale); a few dozen showing tags is plenty.
        candidates: Dictionary names to try.
        stable_repeats: How many times an id must recur to count as a real tag.
        progress: Optional callback receiving each candidate name before it is tried.

    Returns:
        One :class:`DictionaryGuess` per candidate, best first.
    """
    frames = list(images)
    guesses = []
    for name in candidates:
        if progress is not None:
            progress(name)
        model = ArUcoModel.withTagSet(name, _MIN_WIN, 23, 10, 7.0, 0.1, 0.6)
        counts: dict[int, int] = {}
        for image in frames:
            for tag in model.detect(image):
                if tag is not None:
                    counts[tag] = counts.get(tag, 0) + 1
        stable = {t: c for t, c in counts.items() if c >= stable_repeats}
        guesses.append(DictionaryGuess(name, sum(counts.values()), sum(stable.values()), stable))
    order = {name: i for i, name in enumerate(candidates)}
    return sorted(guesses, key=lambda g: (-g.n_stable, order[g.tag_set]))


def guess_dictionary(
    images: Iterable[NDArray[np.generic]],
    candidates: Sequence[str] = ARUCO_TAG_SETS,
    *,
    min_stable: int = _STABLE_REPEATS,
) -> str | None:
    """Name the dictionary that decodes recurring tags in ``images``, or ``None``.

    Thin wrapper around :func:`detect_dictionary`. Feed it a few dozen crops or frames that show
    tags (on the production videos 12 random crops around tag nodes never sufficed, 24 sufficed in
    2 of 3 hours and 48 in all 3): with too little evidence no dictionary has recurring ids and
    the answer is ``None`` rather than an arbitrary first candidate.

    Args:
        images: Frames or crops (2-D grayscale).
        candidates: Dictionary names to try.
        min_stable: Minimum number of decodes of recurring ids the winner needs.

    Returns:
        The best dictionary name (smallest on ties between nested dictionaries) or ``None``.
    """
    ranking = detect_dictionary(images, candidates)
    if not ranking:  # no candidate dictionaries
        return None
    best = ranking[0]
    return best.tag_set if best.n_stable >= min_stable else None
