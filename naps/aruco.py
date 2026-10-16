"""Thin wrapper around ``cv2.aruco`` that decodes ArUco tags in image crops.

The wrapper hides the differences between the OpenCV 4.6 (function) and 4.7+/5.x
(``ArucoDetector``) ArUco APIs, optionally restricts the decoder to a subset of a dictionary
(``--aruco-marker-subset``) and always returns tag ids as plain Python ``int``.
"""

# camelCase parameter names mirror the OpenCV DetectorParameters fields and the CLI flags.
# ruff: noqa: N803

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import cv2.aruco
import numpy as np
from numpy.typing import NDArray

from naps.tags import require_tag

if TYPE_CHECKING:
    from cv2.typing import MatLike

__all__ = [
    "ARUCO_TAG_SETS",
    "CORNER_REFINEMENTS",
    "ArUcoModel",
    "Detection",
    "closed_set_correction_bits",
]

# Names of the ArUco dictionaries OpenCV supports (attributes of ``cv2.aruco``).
ARUCO_TAG_SETS = (
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_5X5_1000",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_6X6_1000",
    "DICT_7X7_50",
    "DICT_7X7_100",
    "DICT_7X7_250",
    "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_16h5",
    "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10",
    "DICT_APRILTAG_36h11",
)

# Error-correction bits of a custom dictionary under the pre-4.7 API (`custom_dictionary`).
_LEGACY_CUSTOM_DICT_MAX_CORRECTION_BITS = 1

# Corner refinement methods by name (``cv2.aruco.CORNER_REFINE_*``); NAPS has always used "subpix".
CORNER_REFINEMENTS = {
    "none": cv2.aruco.CORNER_REFINE_NONE,
    "subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
    "contour": cv2.aruco.CORNER_REFINE_CONTOUR,
    "apriltag": cv2.aruco.CORNER_REFINE_APRILTAG,
}

_N_CORNERS = 4
_N_ROTATIONS = 4


@dataclass(frozen=True, eq=False)
class Detection:
    """One decoded marker.

    Attributes:
        tag: The decoded tag id (a plain ``int``; the real id also when decoding with a subset).
        corners: Marker corners, shape ``(4, 2)`` (x, y), clockwise, in the pixels of the image
            handed to the detector.
        confidence: OpenCV's detection confidence in ``[0, 1]`` (1 minus the share of bits that
            had to be corrected / were uncertain); ``nan`` when the OpenCV build cannot compute it.
    """

    tag: int
    corners: NDArray[np.float32]
    confidence: float = math.nan

    @property
    def center(self) -> tuple[float, float]:
        """Mean of the four corners as ``(x, y)``."""
        x, y = np.asarray(self.corners, dtype=np.float64).reshape(-1, 2).mean(axis=0)
        return float(x), float(y)

    @property
    def side(self) -> float:
        """Mean side length of the marker in pixels."""
        quad = np.asarray(self.corners, dtype=np.float64).reshape(_N_CORNERS, 2)
        return float(np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1).mean())


def closed_set_correction_bits(tag_set: str, tags: Iterable[int]) -> int:
    """Return the number of bit errors a closed set of tags can correct unambiguously.

    A dictionary is built for ``min distance`` between all of its markers; a closed set of a few
    tags is usually much sparser, so it can safely correct more bits than the dictionary-wide
    ``maxCorrectionBits``. The result is ``(d - 1) // 2`` where ``d`` is the smallest Hamming
    distance between two different tags of the set over the four rotations, and between a tag
    and its own rotations.

    Args:
        tag_set: Name of the OpenCV dictionary, one of :data:`ARUCO_TAG_SETS`.
        tags: The closed set (ids in the dictionary).

    Returns:
        The correctable bit count (``0`` if the set is a single tag with a rotation-unique code).

    Raises:
        ValueError: If ``tag_set`` is unknown or an id is not in the dictionary.
    """
    if tag_set not in ARUCO_TAG_SETS:
        msg = f"Unable to assign tag set: {tag_set}"
        raise ValueError(msg)
    dictionary = _predefined_dictionary(tag_set)
    n_markers = dictionary.bytesList.shape[0]
    ids = [require_tag(t) for t in tags]
    bad = [t for t in ids if t < 0 or t >= n_markers]
    if bad:
        msg = f"Tag ids {bad} are not in {tag_set} ({n_markers} markers)"
        raise ValueError(msg)
    size = int(dictionary.markerSize)
    # ``bytesList[i]`` is (n_bytes, 4): the four rotations of marker ``i`` as 4 channels.
    bits = [
        [
            np.asarray(
                cv2.aruco.Dictionary.getBitsFromByteList(dictionary.bytesList[i][None], size, rot)
            ).astype(np.uint8)
            for rot in range(_N_ROTATIONS)
        ]
        for i in ids
    ]
    best = size * size
    for i, rotations in enumerate(bits):
        for j, other in enumerate(bits):
            for rot in range(_N_ROTATIONS):
                if i == j and rot == 0:
                    continue
                best = min(best, int(np.count_nonzero(rotations[0] != other[rot])))
    return (best - 1) // 2


def _predefined_dictionary(tag_set: str) -> cv2.aruco.Dictionary:
    """Return the predefined OpenCV dictionary ``tag_set`` (4.7+ or older API)."""
    dict_id = getattr(cv2.aruco, tag_set)
    try:
        return cv2.aruco.getPredefinedDictionary(dict_id)  # OpenCV >= 4.7
    except AttributeError:
        return cv2.aruco.Dictionary_get(dict_id)  # ty: ignore[unresolved-attribute]  # OpenCV < 4.7 API


class ArUcoModel:
    """Wrapper around the ``cv2.aruco`` detector.

    The OpenCV dictionary/parameter objects are created lazily by :meth:`build` (the first call
    to :meth:`detect` does it automatically) because they cannot be pickled and the model is
    handed to multiprocessing workers.

    The parameters are tuned for speed vs. recall: the thresholding parameters dramatically
    improve the detection rate at a large cost in speed. If detection is too slow, raise
    ``adaptiveThreshConstant``; if too few tags are found, lower it (diminishing returns). If
    false positives are a problem, lower ``errorCorrectionRate``.

    Args:
        tag_set: Name of the OpenCV dictionary, one of :data:`ARUCO_TAG_SETS`. Validated by
            :meth:`build`.
        adaptiveThreshWinSizeMin: Minimum window size of the adaptive thresholding.
        adaptiveThreshWinSizeMax: Maximum window size of the adaptive thresholding.
        adaptiveThreshWinSizeStep: Window size increment of the adaptive thresholding.
        adaptiveThreshConstant: Constant subtracted by the adaptive thresholding.
        perspectiveRemoveIgnoredMarginPerCell: Border margin ignored in each marker cell when
            reading the bits.
        errorCorrectionRate: Fraction of the dictionary's error-correction capacity to use.
        tag_subset_list: Optional closed set of tag ids the decoder may return. The decoder is
            restricted to these ids; decoded subset indices are mapped back to the real ids.
        corner_refinement: Corner refinement, a key of :data:`CORNER_REFINEMENTS` (default
            ``"subpix"``, what NAPS always used). Validated by :meth:`build`.
        detector_overrides: Further ``cv2.aruco.DetectorParameters`` fields to set by name (for
            example ``{"maxErroneousBitsInBorderRate": 0.5}``). Validated by :meth:`build`.
        subset_correction_bits: Only with a subset: the bit errors the subset dictionary may
            correct at ``errorCorrectionRate=1`` (default: the full dictionary's, which is
            conservative for a sparse subset; see :func:`closed_set_correction_bits`).
        **kwargs: Ignored; accepted for backward compatibility.

    Raises:
        ValueError: If ``tag_subset_list`` contains duplicate ids.
    """

    def __init__(  # noqa: PLR0917 (public API: parameters kept positional-capable)
        self,
        tag_set: str,
        adaptiveThreshWinSizeMin: int,
        adaptiveThreshWinSizeMax: int,
        adaptiveThreshWinSizeStep: int,
        adaptiveThreshConstant: float,
        perspectiveRemoveIgnoredMarginPerCell: float,
        errorCorrectionRate: float,
        tag_subset_list: Iterable[int] | None = None,
        *,
        corner_refinement: str = "subpix",
        detector_overrides: Mapping[str, float] | None = None,
        subset_correction_bits: int | None = None,
        **kwargs: object,
    ) -> None:
        """Store the parameters; the OpenCV objects are created by :meth:`build`."""
        self.tag_set = tag_set
        self.corner_refinement = corner_refinement
        self.detector_overrides = dict(detector_overrides or {})
        self.subset_correction_bits = subset_correction_bits

        self.adaptiveThreshWinSizeMin = adaptiveThreshWinSizeMin
        self.adaptiveThreshWinSizeMax = adaptiveThreshWinSizeMax
        self.adaptiveThreshWinSizeStep = adaptiveThreshWinSizeStep
        self.adaptiveThreshConstant = adaptiveThreshConstant
        self.errorCorrectionRate = errorCorrectionRate
        self.perspectiveRemoveIgnoredMarginPerCell = perspectiveRemoveIgnoredMarginPerCell

        # OpenCV objects: created by build().
        self.aruco_dict: cv2.aruco.Dictionary | None = None
        self.aruco_params: cv2.aruco.DetectorParameters | None = None
        self.aruco_detector: cv2.aruco.ArucoDetector | None = None
        self.model_built = False

        # Optional closed set of tag ids the decoder may return (``--aruco-marker-subset``)
        subset = [] if tag_subset_list is None else [require_tag(t) for t in tag_subset_list]
        if len(set(subset)) != len(subset):
            msg = "tag_subset_list contains duplicate tag ids"
            raise ValueError(msg)
        self.tag_subset_list: list[int] | None = subset or None
        self.subset_dict: dict[int, int] = {}  # subset-dictionary index -> real tag id

    @classmethod
    def withTagSet(  # noqa: N802, PLR0917 (public camelCase API, positional like the constructor)
        cls,
        tag_set: str,
        adaptiveThreshWinSizeMin: int,
        adaptiveThreshWinSizeMax: int,
        adaptiveThreshWinSizeStep: int,
        adaptiveThreshConstant: float,
        perspectiveRemoveIgnoredMarginPerCell: float,
        errorCorrectionRate: float,
        tag_subset_list: Iterable[int] | None = None,
        *,
        corner_refinement: str = "subpix",
        detector_overrides: Mapping[str, float] | None = None,
        subset_correction_bits: int | None = None,
        **kwargs: object,
    ) -> ArUcoModel:
        """Create a model for ``tag_set``; the arguments are those of the constructor."""
        return cls(
            tag_set,
            adaptiveThreshWinSizeMin,
            adaptiveThreshWinSizeMax,
            adaptiveThreshWinSizeStep,
            adaptiveThreshConstant,
            perspectiveRemoveIgnoredMarginPerCell,
            errorCorrectionRate,
            tag_subset_list,
            corner_refinement=corner_refinement,
            detector_overrides=detector_overrides,
            subset_correction_bits=subset_correction_bits,
            **kwargs,
        )

    def build(self) -> None:
        """Create the OpenCV dictionary, detector parameters and (if available) detector.

        Raises:
            ValueError: If ``tag_set`` is not one of :data:`ARUCO_TAG_SETS`, a requested subset
                id is not in the dictionary, ``corner_refinement`` is not a key of
                :data:`CORNER_REFINEMENTS`, or ``detector_overrides`` names a field that
                ``cv2.aruco.DetectorParameters`` does not have.
        """
        if self.corner_refinement not in CORNER_REFINEMENTS:
            msg = (
                f"Unknown corner refinement {self.corner_refinement!r}: "
                f"choose one of {sorted(CORNER_REFINEMENTS)}"
            )
            raise ValueError(msg)
        aruco_dict = self._assignArucoDict(self.tag_set)
        if self.tag_subset_list:
            aruco_dict = self._subsetArucoDict(aruco_dict, self.tag_subset_list)
        self.aruco_dict = aruco_dict

        params = self._new_detector_parameters()
        params.cornerRefinementMethod = CORNER_REFINEMENTS[self.corner_refinement]
        params.adaptiveThreshWinSizeMin = self.adaptiveThreshWinSizeMin
        params.adaptiveThreshWinSizeMax = self.adaptiveThreshWinSizeMax
        params.adaptiveThreshWinSizeStep = self.adaptiveThreshWinSizeStep
        params.adaptiveThreshConstant = self.adaptiveThreshConstant
        params.perspectiveRemoveIgnoredMarginPerCell = self.perspectiveRemoveIgnoredMarginPerCell
        params.errorCorrectionRate = self.errorCorrectionRate
        for name, value in self.detector_overrides.items():
            if not hasattr(params, name):
                msg = f"cv2.aruco.DetectorParameters has no field {name!r}"
                raise ValueError(msg)
            setattr(params, name, value)
        self.aruco_params = params

        # OpenCV >= 4.7 moved detection onto cv2.aruco.ArucoDetector (the module-level
        # cv2.aruco.detectMarkers is gone in 5.x); older builds keep the function API.
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.model_built = True

    def detect(self, img: NDArray[np.generic]) -> list[int | None]:
        """Decode the ArUco tags in ``img``.

        Args:
            img: Image (typically a crop around one animal's tag node), 2-D grayscale.

        Returns:
            The tag ids as plain Python ``int`` (with a subset, the real ids, not subset indices),
            in detector order. ``[None]`` if the image is empty (e.g. the crop fell outside the
            frame) or no tag was found, so a vote list is never empty.
        """
        if 0 in img.shape:
            return [None]

        corners, tags, _ = self._detect_raw(img, with_confidence=False)
        if len(corners) == 0:
            return [None]
        return [int(tag) for tag in self._real_ids(tags)]

    def detect_detailed(self, img: NDArray[np.generic]) -> list[Detection]:
        """Decode the ArUco tags in ``img`` and keep their geometry and confidence.

        Unlike :meth:`detect` this reports where each marker is, so the caller can tell the
        marker on the animal from a neighbour's that falls inside the crop.

        Args:
            img: Image (typically a crop around one animal's tag node), 2-D grayscale.

        Returns:
            One :class:`Detection` per decoded marker, in detector order; empty if the image is
            empty or no tag was found. The ``confidence`` is ``nan`` on OpenCV builds without
            ``ArucoDetector.detectMarkersWithConfidence``.
        """
        if 0 in img.shape:
            return []

        corners, tags, confidences = self._detect_raw(img, with_confidence=True)
        if len(corners) == 0:
            return []
        ids = self._real_ids(tags)
        scores = (
            [math.nan] * len(ids)
            if confidences is None
            else [float(c) for c in np.ravel(confidences)]
        )
        return [
            Detection(tag, np.asarray(quad, dtype=np.float32).reshape(_N_CORNERS, 2), score)
            for tag, quad, score in zip(ids, corners, scores, strict=True)
        ]

    def _detect_raw(
        self, img: NDArray[np.generic], *, with_confidence: bool
    ) -> tuple[Sequence[NDArray[np.generic]], NDArray[np.generic], NDArray[np.generic] | None]:
        """Run the OpenCV detector: ``(corners, ids, confidences)``; confidences may be None."""
        if not self.model_built:
            self.build()

        image = cast("MatLike", img)  # any numeric ndarray is a valid OpenCV Mat
        if self.aruco_detector is None:
            corners, tags, _ = cv2.aruco.detectMarkers(  # ty: ignore[unresolved-attribute]  # OpenCV < 4.7 API
                image, self.aruco_dict, parameters=self.aruco_params
            )
            return corners, tags, None
        if with_confidence and hasattr(self.aruco_detector, "detectMarkersWithConfidence"):
            corners, tags, confidences, _ = self.aruco_detector.detectMarkersWithConfidence(image)
            return corners, tags, confidences
        corners, tags, _ = self.aruco_detector.detectMarkers(image)
        return corners, tags, None

    def _real_ids(self, tags: NDArray[np.generic]) -> list[int]:
        """Flatten OpenCV's ids and map subset-dictionary indices back to the real tag ids."""
        # Ids are shaped (N, 1) in OpenCV 4.x but (N,) in 5.x, so flatten before indexing.
        indices = [int(np.ravel(marker_tag)[0]) for marker_tag in tags]
        if self.subset_dict:
            return [int(self.subset_dict[index]) for index in indices]
        return indices

    @staticmethod
    def _new_detector_parameters() -> cv2.aruco.DetectorParameters:
        """Create detector parameters with the new (4.7+) or old (< 4.7) OpenCV API."""
        try:
            return cv2.aruco.DetectorParameters()
        except AttributeError:
            return cv2.aruco.DetectorParameters_create()  # ty: ignore[unresolved-attribute]  # OpenCV < 4.7 API

    def _subsetArucoDict(  # noqa: N802 (private, mirrors _assignArucoDict)
        self, aruco_dict: cv2.aruco.Dictionary, tag_subset_list: list[int]
    ) -> cv2.aruco.Dictionary:
        """Restrict a dictionary to ``tag_subset_list``.

        The decoder can then only return those tags; indices of the subset dictionary are mapped
        back to tag ids in :meth:`detect` (via ``self.subset_dict``, set here). The correction
        capacity of the full dictionary is kept (conservative for a subset).

        Raises:
            ValueError: If an id is negative or not in ``aruco_dict``.
        """
        n_markers = aruco_dict.bytesList.shape[0]
        bad = [t for t in tag_subset_list if t < 0 or t >= n_markers]
        if bad:
            msg = f"Tag ids {bad} are not in {self.tag_set} ({n_markers} markers)"
            raise ValueError(msg)
        sub_bytes = np.ascontiguousarray(np.take(aruco_dict.bytesList, tag_subset_list, axis=0))
        max_bits = (
            int(aruco_dict.maxCorrectionBits)
            if self.subset_correction_bits is None
            else int(self.subset_correction_bits)
        )
        try:
            # New API (OpenCV 4.7 and later)
            subset = cv2.aruco.Dictionary(sub_bytes, int(aruco_dict.markerSize), max_bits)
        except (AttributeError, TypeError):
            # custom_dictionary was removed in 4.7+
            subset = cv2.aruco.custom_dictionary(  # ty: ignore[unresolved-attribute]  # OpenCV < 4.7 API
                0, aruco_dict.markerSize, _LEGACY_CUSTOM_DICT_MAX_CORRECTION_BITS
            )
            subset.bytesList = sub_bytes
        self.subset_dict = dict(enumerate(tag_subset_list))
        return subset

    def _assignArucoDict(self, tag_set: str) -> cv2.aruco.Dictionary:  # noqa: N802 (name kept)
        """Return the predefined OpenCV dictionary called ``tag_set``.

        Raises:
            ValueError: If ``tag_set`` is not one of :data:`ARUCO_TAG_SETS`.
        """
        if tag_set not in ARUCO_TAG_SETS:
            msg = f"Unable to assign tag set: {tag_set}"
            raise ValueError(msg)
        return _predefined_dictionary(tag_set)
