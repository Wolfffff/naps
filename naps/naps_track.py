#!/usr/bin/env python
"""``naps-track``: hybrid tracking with SLEAP poses and ArUco tags.

Exit status: 0 on success; 1 for invalid option combinations (a message is printed to
stderr); 2 for command-line syntax errors (argparse). Any other failure raises and exits
with a traceback.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import sleap_io as sio

from naps.aruco import ArUcoModel
from naps.consistency import ConsistencyParams
from naps.decode import FAST_MIN_MARKER_PERIMETER_RATE, DecodeParams, TagDecoder
from naps.diagnostics import (
    DiagnosticsParams,
    apply_policy,
    assess_result,
)
from naps.diagnostics import (
    build_report as build_joint_report,
)
from naps.evidence import EvidenceSource, IdtrackerEvidence
from naps.identity import MAX_TAGS, IdentityParams
from naps.instances import table_from_labels
from naps.joint import JointConfig, JointMatching, JointResult, qualify_tag, write_sidecar
from naps.matching import (
    ChunkedDecodeSource,
    DecodeSource,
    Matching,
    OpenCVDecodeSource,
    TagDecoderSource,
)
from naps.parallel_decode import available_cpus
from naps.precomputed import DecodesError, PrecomputedDecodeSource
from naps.quadrants import (
    QuadrantScope,
    build_report,
    make_grid,
    parse_roi_args,
    parse_tag_map,
)
from naps.sleap_utils import apply_instance_tags, update_labeled_frames
from naps.slp_fast import save_labels
from naps.slp_tracks import InstanceRow, UnsupportedSlpError, iter_window_instances
from naps.tracklets import TrackletParams
from naps.utils.cli import add_verbosity_arguments, log_verbosity

logger = logging.getLogger("NAPS Logger")

#: Default cap (pixels) on the elimination continuity threshold in the quadrant modes.
QUADRANT_MAX_DISTANCE = 400.0

#: Deprecated options, accepted but ignored: ``dest -> explanation``.
DEPRECATED_OPTIONS = {
    "h5_path": (
        "--h5-path is deprecated and ignored: NAPS always reads the tracks from "
        "--slp-path (the option will be removed in a future release)"
    ),
}


# --------------------------------------------------------------------------- helpers


def _instance_position(
    points: np.ndarray, tag_idx: int, node_names: Sequence[str], fallback_name: str
) -> np.ndarray | None:
    """(x, y) of the tag node, else the fallback node, else the mean of the finite nodes.

    Returns ``None`` if no node is finite.
    """
    candidates = [points[tag_idx]]
    if fallback_name in node_names:
        candidates.append(points[node_names.index(fallback_name)])
    for xy in candidates:
        if np.isfinite(xy).all():
            return np.asarray(xy, dtype=float)
    finite = points[np.isfinite(points).all(axis=1)]
    return finite.mean(axis=0) if len(finite) else None


def _json_default(obj: object) -> object:
    """``json.dump`` hook for the numpy scalars, sets and tuples in the report."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    """Build the ``naps-track`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="naps-track",
        description="NAPS -- Hybrid tracking using SLEAP and ArUco tags",
    )
    _add_io_arguments(parser)
    _add_aruco_arguments(parser)
    _add_closed_set_arguments(parser)
    _add_quadrant_arguments(parser)
    _add_idtracker_arguments(parser)
    _add_joint_arguments(parser)
    _add_diagnostic_arguments(parser)
    _add_performance_arguments(parser)
    add_verbosity_arguments(parser)
    return parser


def _add_io_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("input and output")
    group.add_argument(
        "--slp-path",
        help=(
            "The filepath of the SLEAP (.slp) file to pull coordinates from. This "
            "should correspond with the input video file. The SLEAP file contains pose "
            "estimation data required for NAPS."
        ),
        type=str,
        required=True,
    )
    group.add_argument(
        "--h5-path",
        help=(
            "DEPRECATED and ignored (a warning is logged): NAPS always reads the "
            "tracks from --slp-path."
        ),
        type=str,
        default=None,
    )
    group.add_argument(
        "--video-path",
        help=(
            "The filepath of the video used with SLEAP. This should be the video file "
            "that SLEAP was run on."
        ),
        type=str,
        required=True,
    )
    group.add_argument(
        "--decodes",
        help=(
            "Precomputed tag decodes (.npz, e.g. written by a GPU pipeline that decoded the "
            "video once for SLEAP and the tags; the layout is in naps.precomputed) for "
            "--slp-path. Replaces reading --video-path and decoding the tags: the video is "
            "never opened (--video-path still names it and the session), and the --aruco-* "
            "decoding options are not used. Mutually exclusive with --decoder fast. Works with "
            "both identity engines. Exits with an error if the file does not match the SLEAP "
            "file or the frame range."
        ),
        type=str,
        default=None,
    )
    group.add_argument(
        "--tag-node-name",
        help=(
            "The ArUco tag SLEAP node name. This should correspond to the node name of "
            "the ArUco tag in the SLEAP data. 'tag' is set as the default value."
        ),
        type=str,
        default="tag",
    )
    group.add_argument(
        "--start-frame",
        help=(
            "The zero-based fully-closed frame to begin NAPS assignment. This value "
            "allows you to specify from which frame to start the processing."
        ),
        type=int,
        required=True,
    )
    group.add_argument(
        "--end-frame",
        help=(
            "The zero-based fully-closed frame to stop NAPS assignment. This allows "
            "you to specify at which frame to end the processing."
        ),
        type=int,
        required=True,
    )
    group.add_argument(
        "--half-rolling-window-size",
        help=(
            "Specifies the number of flanking frames (prior and subsequent) required "
            "in the rolling window for Hungarian matching a frame. The larger this "
            "window, the more frames will be used for matching. This can result in "
            "more robust tracks but in the event of an identity swap, it may take "
            "longer to correct the identity. The default is set to 5, resulting in an "
            "11 frame window. You should change this depending on your intended use "
            "and your recording setup. "
        ),
        type=int,
        default=5,
    )
    group.add_argument(
        "--output-path",
        help=(
            "Output path of the resulting SLEAP h5 analysis file. The default value is "
            "'naps_output.analysis.h5', storing the output in the current working "
            "directory. If a different location or file name is preferred, it can be "
            "specified here."
        ),
        type=str,
        default="naps_output.analysis.h5",
    )


def _positive_float(text: str) -> float:
    """``argparse`` type of a strictly positive finite float."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not 0 < value < float("inf"):
        raise argparse.ArgumentTypeError(f"{text!r} must be a positive number")
    return value


def _add_aruco_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "ArUco detection",
        "Most of these are passed to OpenCV's ArUco detector; see the OpenCV documentation.",
    )
    group.add_argument(
        "--aruco-marker-set",
        help=(
            "The ArUco markers used in the video. This must match the specific set of "
            "ArUco markers used in the video for accurate detection and tracking. An "
            "example would be DICT_5X5_50, which is a set of 50 5x5 markers. More "
            "information can be found in the OpenCV documentation."
        ),
        type=str,
        required=True,
    )
    group.add_argument(
        "--aruco-crop-size",
        help=(
            "The number of pixels horizontally and vertically around the ArUco SLEAP "
            "node to identify the marker. The cropping area should be large enough to "
            "include the whole marker for accurate detection, but not much larger, to "
            "keep the processing efficient and avoid capturing multiple tags. This is "
            "data set specific. Default 100 (the value used in the NAPS paper); "
            "without a default the run crashed in croppedCoords with a None crop size."
        ),
        type=int,
        default=100,
    )
    group.add_argument(
        "--aruco-adaptive-thresh-win-size-min",
        dest="adaptiveThreshWinSizeMin",
        help=(
            "Specifies the value for adaptiveThreshWinSizeMin used in adaptive "
            "thresholding. This parameter affects the adaptive thresholding in the "
            "ArUco marker detection, which can impact the robustness of marker "
            "detection. The default value is 10, a commonly used value. More "
            "information can be found in the OpenCV documentation."
        ),
        type=int,
        default=10,
    )
    group.add_argument(
        "--aruco-adaptive-thresh-win-size-max",
        dest="adaptiveThreshWinSizeMax",
        help=(
            "Specifies the value for adaptiveThreshWinSizeMax used in adaptive "
            "thresholding. This parameter, similar to adaptiveThreshWinSizeMin, "
            "influences the adaptive thresholding. The default value is 30, providing "
            "a larger window size for the thresholding process, improving marker "
            "detection under diverse lighting conditions. More information can be "
            "found in the OpenCV documentation."
        ),
        type=int,
        default=30,
    )
    group.add_argument(
        "--aruco-adaptive-thresh-win-size-step",
        dest="adaptiveThreshWinSizeStep",
        help=(
            "Specifies the value for adaptiveThreshWinSizeStep used in adaptive "
            "thresholding. This parameter determines the step size for the window in "
            "adaptive thresholding, affecting the granularity of the process. The "
            "default value is 12, offering a balanced choice between processing speed "
            "and thresholding precision. More information can be found in the OpenCV "
            "documentation."
        ),
        type=int,
        default=12,
    )
    group.add_argument(
        "--aruco-adaptive-thresh-constant",
        dest="adaptiveThreshConstant",
        help=(
            "Specifies the value for adaptiveThreshConstant used in adaptive "
            "thresholding. This parameter is a constant subtracted from the mean or "
            "weighted sum of the neighbourhood pixels. The default value is 3, which "
            "works well in most scenarios, but can be adjusted based on specific "
            "lighting conditions. More information can be found in the OpenCV "
            "documentation."
        ),
        type=float,
        default=3,
    )
    group.add_argument(
        "--aruco-perspective-rm-ignored-margin",
        dest="perspectiveRemoveIgnoredMarginPerCell",
        help=(
            "Specifies the value for perspectiveRemoveIgnoredMarginPerCell. This "
            "parameter is used in the perspective removal of the marker. The default "
            "value is 0.1, which is a reasonable value for many situations. More "
            "information can be found in the OpenCV documentation."
        ),
        type=float,
        default=0.1,
    )
    group.add_argument(
        "--aruco-error-correction-rate",
        dest="errorCorrectionRate",
        help=(
            "Specifies the value for errorCorrectionRate. This parameter is used for "
            "error correction when decoding ArUco tags. The default value is 1 is to "
            "fully utilize the error capability for each tag dictionary. Adjust this "
            "parameter can be necessary for improving the robustness of marker "
            "detection, especially in the presence of camera noise or occlusions. "
            "More information can be found in the OpenCV documentation."
        ),
        type=float,
        default=1,
    )
    group.add_argument(
        "--aruco-marker-subset",
        help=(
            "Restrict the ArUco decoder to these marker ids (e.g. the tags used in "
            "the colony). Off by default."
        ),
        type=int,
        nargs="+",
        default=None,
    )
    group.add_argument(
        "--aruco-min-marker-perimeter-rate",
        dest="aruco_min_marker_perimeter_rate",
        help=(
            "minMarkerPerimeterRate of the ArUco detector: candidate contours whose perimeter "
            "is below RATE x the larger side of the crop (in pixels) are discarded. OpenCV's "
            "default 0.03 (3 px on a 100 px crop) lets many tiny noise quads through and is the "
            f"main cost of detection; {FAST_MIN_MARKER_PERIMETER_RATE} gives identical decodes "
            "up to 8x faster (53,640 real crops, 12 colonies/cameras: no crop changed). Recall "
            "only drops when RATE x crop side exceeds the smallest tag's perimeter (about 96 px "
            "here), so keep RATE x crop side below about 48 px. Default: OpenCV's rate for the "
            f"legacy decoder (unchanged), {FAST_MIN_MARKER_PERIMETER_RATE} for --decoder fast; "
            "pass 0.03 to restore OpenCV's rate there."
        ),
        type=_positive_float,
        default=None,
    )
    group.add_argument(
        "--strict-carry-forward",
        help=(
            "Only carry the previous frame's tag into tracks the linear assignment "
            "left unmatched (a fresh assignment is never overridden). Off by default: "
            "on real data this flipped tags 2-8x more often along a SLEAP track than "
            "the legacy sticky carry-forward."
        ),
        action="store_true",
    )


def _add_closed_set_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "closed set of tags (opt-in)",
        "--closed-set (one colony) is mutually exclusive with --closed-set-map and "
        "--closed-set-infer (one tag set per quadrant), which are mutually exclusive "
        "with each other.",
    )
    group.add_argument(
        "--closed-set",
        help=(
            "The N expected tag ids of this video/quadrant. Drops ArUco votes for any "
            "other tag and runs closed-set elimination (if exactly one expected tag "
            "and one tracked instance are unassigned in a frame, assign it). Off by "
            "default."
        ),
        type=int,
        nargs="+",
        default=None,
    )
    group.add_argument(
        "--closed-set-map",
        help=(
            "Quadrant-aware closed set: tag ids per region, either inline, e.g. "
            '"top_left:45,46,47,48;top_right:49,50,51,52;...", or the path of a '
            ".json/.yaml file ({region: [tags]} or {tags: {...}, rois: {region: "
            "[x0, y0, x1, y1]}}). Every tracked instance is assigned to a region by "
            "position; votes for tags that do not belong to its region are dropped "
            "and closed-set elimination runs per region. Regions not listed are left "
            "untouched. Mutually exclusive with --closed-set. Off by default."
        ),
        type=str,
        default=None,
    )
    group.add_argument(
        "--closed-set-infer",
        help=(
            "Quadrant-aware closed set with the tag sets inferred from this video's "
            "ArUco decodes (tags decoded consistently in one region with enough "
            "support; ambiguous regions/tags abstain and stay in legacy behaviour). "
            "Mutually exclusive with --closed-set/--closed-set-map."
        ),
        action="store_true",
    )
    group.add_argument(
        "--closed-set-size",
        help=(
            "Tags expected per region when inferring. A region with fewer is not used. Default 4."
        ),
        type=int,
        default=4,
    )
    group.add_argument(
        "--closed-set-min-support",
        help=(
            "Inference: minimum number of frames a tag must be decoded in a region "
            "(also at least --closed-set-min-fraction of the frames with instances "
            "there). Default 30."
        ),
        type=int,
        default=30,
    )
    group.add_argument(
        "--closed-set-min-fraction",
        help=(
            "Inference: minimum fraction of a region's frames a tag must be decoded "
            "in. Default 0.01."
        ),
        type=float,
        default=0.01,
    )
    group.add_argument(
        "--closed-set-min-purity",
        help=(
            "Inference: fraction of a tag's decode frames that must lie in its region. Default 0.9."
        ),
        type=float,
        default=0.9,
    )
    group.add_argument(
        "--no-closed-set-elimination",
        dest="closed_set_elimination",
        help="With --closed-set, only filter votes; do not run the elimination step.",
        action="store_false",
    )
    group.add_argument(
        "--closed-set-max-jump",
        help=(
            "Elimination continuity gate in pixels per frame of gap between the "
            "instance and where the missing tag was last/next seen. Default 120."
        ),
        type=float,
        default=120.0,
    )
    group.add_argument(
        "--closed-set-max-distance",
        help=(
            "Absolute cap in pixels on the elimination continuity threshold "
            "(otherwise --closed-set-max-jump times the gap in frames, which reaches "
            "thousands of pixels). Default: none for --closed-set, 400 for the "
            "quadrant modes."
        ),
        type=float,
        default=None,
    )


def _add_quadrant_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "quadrants (opt-in, with --closed-set-map or --closed-set-infer)"
    )
    group.add_argument(
        "--quadrant-grid",
        help=(
            "ROWS COLS of the grid of the video frame size that defines the regions. "
            "Default 2 2 (top_left, top_right, bottom_left, bottom_right)."
        ),
        type=int,
        nargs=2,
        default=(2, 2),
        metavar=("ROWS", "COLS"),
    )
    group.add_argument(
        "--quadrant-roi",
        help=(
            "Explicit region, repeatable; replaces the grid: NAME X0 Y0 X1 Y1 (pixels, half-open)."
        ),
        nargs=5,
        action="append",
        default=None,
        metavar=("NAME", "X0", "Y0", "X1", "Y1"),
    )
    group.add_argument(
        "--position-node-name",
        help=(
            "Node used as the instance position when the tag node is NaN (quadrant "
            "mode: locating the instance and the elimination continuity check). Falls "
            "back to the mean of the finite nodes. Default 'Thorax'."
        ),
        type=str,
        default="Thorax",
    )


def _add_idtracker_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "idtracker.ai continuity evidence (opt-in, requires a closed set)"
    )
    group.add_argument(
        "--idtracker-trajectories",
        help=(
            "Path to an idtracker.ai v6 trajectories.npy. Used ONLY as continuity "
            "evidence to bridge ArUco dropouts; ArUco decodes stay the identity "
            "source. Requires --closed-set (or --closed-set-map/--closed-set-infer). "
            "Off by default."
        ),
        type=str,
        default=None,
    )
    group.add_argument(
        "--idtracker-offset",
        help=(
            "X Y offset (pixels) added to idtracker.ai coordinates to reach SLEAP "
            "coordinates, e.g. the origin of the quadrant crop idtracker.ai ran on. "
            "Default 0 0."
        ),
        type=float,
        nargs=2,
        default=(0.0, 0.0),
    )
    group.add_argument(
        "--idtracker-frame-offset",
        help=(
            "Trajectory row i corresponds to video frame i + this value (use when "
            "idtracker.ai ran on a segment). Default 0."
        ),
        type=int,
        default=0,
    )
    group.add_argument(
        "--idtracker-weight",
        help=(
            "Weight of one idtracker.ai vote (scaled by its id_probability); an ArUco "
            "decode has weight 1. Default 0.5."
        ),
        type=float,
        default=0.5,
    )
    group.add_argument(
        "--idtracker-min-prob",
        help=(
            "Minimum id_probability for an idtracker.ai position to vote (NaN, i.e. "
            "interpolated, never votes). Default 0.5."
        ),
        type=float,
        default=0.5,
    )
    group.add_argument(
        "--idtracker-max-dist",
        help=(
            "Maximum distance in pixels between an idtracker.ai centroid and a SLEAP "
            "tag node to join them. Default 100."
        ),
        type=float,
        default=100.0,
    )
    group.add_argument(
        "--idtracker-min-margin",
        help=(
            "A join is only used if the next-nearest alternative is at least this "
            "many pixels farther away (abstains at crossings). Should exceed the "
            "offset between idtracker.ai centroids and the SLEAP tag node. Default 20."
        ),
        type=float,
        default=20.0,
    )
    group.add_argument(
        "--idtracker-min-support",
        help=(
            "Minimum number of unambiguous ArUco co-occurrences for an idtracker.ai "
            "animal to be aligned to a tag. Default 30."
        ),
        type=int,
        default=30,
    )
    group.add_argument(
        "--idtracker-min-purity",
        help=(
            "Minimum fraction of an animal's co-occurrences that must agree on one "
            "tag; otherwise that animal emits nothing. Default 0.9."
        ),
        type=float,
        default=0.9,
    )


def _add_joint_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "joint identity engine (opt-in; requires --closed-set, --closed-set-map or "
        "--closed-set-infer)",
        "The joint engine builds tracklets from the SLEAP instances itself, decodes the tag "
        "around every instance and infers the tag of every instance for the whole video with an "
        "exact closed-set model. Clean, unique decodes are anchors and are never overridden "
        "(a contradicted decode is only flagged 'disputed'); isolated misreads (the same tag on "
        "two instances of a frame, or a tiny minority tag on a tracklet) are demoted to weak "
        "evidence, counted in the report, and the instance may be relabelled. It labels the "
        "first and last frames, recovers bees whose tag was not decoded, and abstains (with a "
        "reason code in the sidecar) where it cannot be sure. The default engine stays "
        "'legacy'.",
    )
    group.add_argument(
        "--identity-engine",
        help="Identity assignment engine. Default: legacy (rolling-window Hungarian).",
        choices=("legacy", "joint"),
        default="legacy",
    )
    group.add_argument(
        "--tracklets",
        help=(
            "joint engine: 'built' links the SLEAP instances by position/velocity/pose and uses "
            "SLEAP track ids only as a hint (works without a SLEAP tracker); 'sleap' follows the "
            "SLEAP track ids but still splits them at implausible jumps. In 'sleap' mode a link "
            "needs equal ids on both sides, so instances without a SLEAP track never continue a "
            "tracklet (use 'built' for untracked input). Default: built."
        ),
        choices=("built", "sleap"),
        default="built",
    )
    group.add_argument(
        "--min-confidence",
        help=(
            "joint engine: a recovered tag is written only if its posterior is at least this. "
            "On 38 held-out windows (396k bee-frames, scored against NAPS's own labels, not "
            "independent ground truth) the posterior was well calibrated (ECE 0.0028), so a lower "
            "value trades precision for coverage rather than hitting a miscalibrated band: 0.98 "
            "passed a pre-registered rule with --untrusted-policy emit (correct minus wrong "
            "+1.4 percentage points); 0.95 and 0.9 recover more labels at more wrong ones "
            "(+1.0 and +1.9 per 1000 bee-frames). Default 0.99."
        ),
        type=float,
        default=0.99,
    )
    group.add_argument(
        "--consistency",
        help=(
            "joint engine: temporal-consistency stage (ghost instances, duplicate tags, chain "
            "veto, gated interpolation). --no-consistency switches all of it off, including the "
            "gap interpolation. Default on."
        ),
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    group.add_argument(
        "--interpolate-max-gap",
        help=(
            "joint engine: longest gap (frames) between two positions of a tag that is "
            "interpolated (only where plausible; sidecar only, no instances are invented). "
            "0 switches interpolation off; so does --no-consistency. Default 20."
        ),
        type=int,
        default=20,
    )
    group.add_argument(
        "--colony-size",
        help=(
            "joint engine: bees per colony/quadrant. A region with fewer tags gets untagged-bee "
            "placeholders (3 tags + 1 untagged bee); a region with more tags is unsupported. "
            f"3..{MAX_TAGS}, or 2 for a pair with an explicit two-tag --closed-set. Default 4."
        ),
        type=int,
        default=4,
    )
    group.add_argument(
        "--decoder",
        help=(
            "joint engine: 'legacy' reads the video with cv2.VideoCapture and decodes with the "
            "ArUco model of the --aruco-* options (as the legacy engine does); 'fast' uses the "
            "decode stage (grayscale luma reader, same detector settings and decodes, faster). "
            "Default legacy."
        ),
        choices=("legacy", "fast"),
        default="legacy",
    )
    group.add_argument(
        "--untrusted-policy",
        help=(
            "joint engine: 'withhold' (default) writes only decoded labels in windows that fail "
            "the quality criteria; 'emit' keeps the recovered labels there too."
        ),
        choices=("withhold", "emit"),
        default="withhold",
    )
    group.add_argument(
        "--self-validate",
        help=(
            "joint engine: hold out decodes (leave-1 per tag, random runs), re-run the model and "
            "report precision/coverage per window; slow (one extra solve per tag plus 3 solves "
            "with random held-out runs). Off by default."
        ),
        action="store_true",
    )
    group.add_argument(
        "--window-frames",
        help="joint engine: length of a status window in frames. Default 6000.",
        type=int,
        default=6000,
    )
    group.add_argument(
        "--identity-sidecar",
        help=(
            "joint engine: path of the per-frame sidecar (.npz: confidence, source, flag, "
            "reason code of every unlabelled instance, tracklet id, positions). Default: the "
            "output path with the suffix '.identity.npz'."
        ),
        type=str,
        default=None,
    )
    group.add_argument(
        "--session",
        help=(
            "joint engine: key that qualifies tag ids in the outputs (tag ids are reused across "
            "colonies and trials). Default: the name of the directory holding the video."
        ),
        type=str,
        default=None,
    )


def _add_performance_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "performance", "Speed options; none of them changes the result."
    )
    group.add_argument(
        "--no-fast-read",
        dest="fast_read",
        help=(
            "Load the whole .slp with sleap-io before tracking instead of reading the tag-node "
            "positions straight from its tables (same positions, much slower)."
        ),
        action="store_false",
    )
    group.add_argument(
        "--no-fast-write",
        dest="fast_write",
        help=(
            "Write the output .slp with sleap-io's own frame writer instead of the vectorised "
            "one (same datasets, much slower)."
        ),
        action="store_false",
    )
    group.add_argument(
        "--decode-workers",
        help=(
            "Decode the video (read, crop, ArUco) in this many parallel workers over contiguous "
            "frame ranges. The tags are identical to the serial run. Default 1 (serial); 0 uses "
            "all CPUs available to the process. Not used with --decodes."
        ),
        type=int,
        default=1,
    )
    group.add_argument(
        "--decode-chunk-frames",
        help="Frames per decode chunk (default: balanced for the number of workers).",
        type=int,
        default=None,
    )
    group.add_argument(
        "--decode-gop",
        help=(
            "Keyframe spacing of the video in frames (250 for the lab's hourly HEVC videos). "
            "Parallel decode chunks then start near keyframes, which makes the seek to a chunk "
            "cheap. 0 (default): do not align."
        ),
        type=int,
        default=0,
    )
    group.add_argument(
        "--decode-gop-offset",
        help=(
            "With --decode-gop: chunks start this many frames after a keyframe (OpenCV's seek "
            "lands on the keyframe BEFORE a target less than about a second after a keyframe, so "
            "a small positive offset seeks fastest). Default 25."
        ),
        type=int,
        default=25,
    )


def _add_diagnostic_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("diagnostics")
    group.add_argument(
        "--report-json",
        help=(
            "Write per-video diagnostics as JSON: frames with N..0 tags per region "
            "before and after elimination, instances left without a tag by cause, "
            "eliminated assignments, abstention reasons (the joint engine writes its own "
            "report: completeness, sources, abstentions by reason, window status)."
        ),
        type=str,
        default=None,
    )


# --------------------------------------------------------------------------- validation


def warn_deprecated_options(args: argparse.Namespace) -> None:
    """Warn (``DeprecationWarning`` and a log line) about each deprecated option given."""
    for dest, message in DEPRECATED_OPTIONS.items():
        if getattr(args, dest, None) is not None:
            warnings.warn(message, DeprecationWarning, stacklevel=3)
            logger.warning(message)


def is_quadrant_mode(args: argparse.Namespace) -> bool:
    """Whether ``--closed-set-map`` or ``--closed-set-infer`` is on."""
    return bool(args.closed_set_map) or args.closed_set_infer


def validate_args(args: argparse.Namespace) -> None:
    """Check the option combinations argparse cannot express.

    Raises:
        SystemExit: With an explanatory message (exit status 1) for an invalid combination.
    """
    quadrant_mode = is_quadrant_mode(args)
    if args.decode_workers < 0 or args.decode_gop < 0 or args.decode_gop_offset < 0:
        raise SystemExit(
            "--decode-workers, --decode-gop and --decode-gop-offset must not be negative"
        )
    if args.decode_chunk_frames is not None and args.decode_chunk_frames < 1:
        raise SystemExit("--decode-chunk-frames must be at least 1")
    if args.decodes:
        if args.decoder != "legacy":
            raise SystemExit(
                "--decodes (precomputed decodes) is mutually exclusive with decoding from the "
                f"video: drop --decoder {args.decoder}"
            )
        if not Path(args.decodes).is_file():
            raise SystemExit(f"--decodes {args.decodes}: no such file")
    if quadrant_mode and args.closed_set:
        raise SystemExit(
            "--closed-set (one colony) cannot be combined with "
            "--closed-set-map/--closed-set-infer (per quadrant)"
        )
    if args.closed_set_map and args.closed_set_infer:
        raise SystemExit("use either --closed-set-map or --closed-set-infer, not both")
    if args.idtracker_trajectories and not args.closed_set and not quadrant_mode:
        raise SystemExit(
            "--idtracker-trajectories requires --closed-set "
            "(or --closed-set-map/--closed-set-infer)"
        )
    if args.identity_engine == "joint":
        _validate_joint_args(args, quadrant_mode)


def _validate_joint_args(args: argparse.Namespace, quadrant_mode: bool) -> None:
    """Check the options of ``--identity-engine joint``.

    Raises:
        SystemExit: With an explanatory message (exit status 1) for an invalid combination.
    """
    if not args.closed_set and not quadrant_mode:
        raise SystemExit(
            "--identity-engine joint needs the closed set of tags: pass --closed-set, "
            "--closed-set-map or --closed-set-infer"
        )
    # A pair (2 tags) is only allowed with an explicit --closed-set of 2 tags: inferred tag sets
    # keep the 3-tag minimum, so their behaviour is unchanged.
    pair = bool(args.closed_set) and len(set(args.closed_set)) == 2
    if not (2 if pair else 3) <= args.colony_size <= MAX_TAGS:
        raise SystemExit(
            f"--colony-size must be between 3 and {MAX_TAGS} "
            "(2 only with an explicit two-tag --closed-set)"
        )
    if not 0.5 < args.min_confidence <= 1.0:
        raise SystemExit("--min-confidence must be in (0.5, 1]")
    if args.interpolate_max_gap < 0 or args.window_frames < 1:
        raise SystemExit("--interpolate-max-gap must be >= 0 and --window-frames >= 1")
    if args.closed_set and not 2 <= len(set(args.closed_set)) <= args.colony_size:
        raise SystemExit(
            f"--closed-set needs between 2 and --colony-size ({args.colony_size}) tags for the "
            "joint engine"
        )


# --------------------------------------------------------------------------- pipeline steps


@dataclass
class SleapTracks:
    """What :func:`load_sleap_tracks` reads from the SLEAP file, keyed by frame index."""

    #: ``{frame: {track id: tag-node (x, y)}}`` (a nested defaultdict, as ``Matching`` expects).
    tag_locations: defaultdict = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(tuple))
    )
    #: ``{frame: {track id: position}}``: the tag node, or the fallback node where it is NaN.
    positions: defaultdict = field(default_factory=lambda: defaultdict(dict))
    #: ``{frame: [(track id | None, position | None)]}`` of every instance, for the report.
    instances_by_frame: defaultdict = field(default_factory=lambda: defaultdict(list))
    #: ``{frame: number of instances without a SLEAP track}``.
    untracked_counts: defaultdict = field(default_factory=lambda: defaultdict(int))
    #: ``{frame: [position | None]}`` of the instances without a SLEAP track.
    untracked_positions: defaultdict = field(default_factory=lambda: defaultdict(list))


def _eager_instances(
    labels: sio.Labels,
    start_frame: int,
    end_frame: int,
    tag_node_name: str,
    position_node_name: str,
) -> Iterator[InstanceRow]:
    """The instances of the window from a loaded :py:class:`sleap_io.Labels` (the reference)."""
    for lf in labels.labeled_frames:
        if lf.frame_idx < start_frame or lf.frame_idx > end_frame:
            continue
        for instance in lf.instances:
            node_names = instance.skeleton.node_names
            try:
                tag_idx = node_names.index(tag_node_name)
            except ValueError as exc:
                raise ValueError(
                    f"tag node {tag_node_name!r} is not a node of the SLEAP skeleton {node_names}"
                ) from exc
            points = instance.numpy()
            position = _instance_position(points, tag_idx, node_names, position_node_name)
            track_id = None if instance.track is None else int(instance.track.name.split("_")[-1])
            yield lf.frame_idx, track_id, points[tag_idx], position


def load_sleap_tracks(
    slp_path: str | Path | sio.Labels,
    start_frame: int,
    end_frame: int,
    tag_node_name: str,
    position_node_name: str,
    *,
    fast: bool = True,
) -> SleapTracks:
    """Read the tag-node locations of every instance in ``[start_frame, end_frame]``.

    Args:
        slp_path: SLEAP ``.slp`` file, or the already loaded :py:class:`sleap_io.Labels`.
        start_frame: First frame (inclusive).
        end_frame: Last frame (inclusive).
        tag_node_name: Name of the skeleton node that sits on the ArUco tag.
        position_node_name: Fallback node for the instance position when the tag node
            is NaN.
        fast: Read the tables of a ``.slp`` path directly (:mod:`naps.slp_tracks`, same result,
            an order of magnitude faster than building every instance); if the file has an
            unexpected layout the Labels are loaded instead.

    Raises:
        ValueError: If the skeleton has no node called ``tag_node_name``.
    """
    tracks = SleapTracks()
    rows: Iterator[InstanceRow] | None = None
    if fast and not isinstance(slp_path, sio.Labels):
        try:
            rows = iter_window_instances(
                slp_path,
                start_frame,
                end_frame,
                tag_node_name,
                position_node_name,
                _instance_position,
            )
            first = next(rows, None)  # surface an UnsupportedSlpError now, not mid-loop
            rows = chain([first], rows) if first is not None else iter(())
        except UnsupportedSlpError as exc:
            logger.info("Fast .slp reader not used (%s); loading the labels", exc)
            rows = None
    if rows is None:
        labels = slp_path if isinstance(slp_path, sio.Labels) else sio.load_slp(str(slp_path))
        rows = _eager_instances(labels, start_frame, end_frame, tag_node_name, position_node_name)
    for frame_idx, track_id, tag_xy, position in rows:
        if track_id is None:
            logger.debug("Frame %s: skipping instance without a SLEAP track.", frame_idx)
            tracks.untracked_counts[frame_idx] += 1
            tracks.untracked_positions[frame_idx].append(position)
            tracks.instances_by_frame[frame_idx].append((None, position))
            continue
        tracks.tag_locations[frame_idx][track_id] = tag_xy
        if position is not None:
            tracks.positions[frame_idx][track_id] = position
        tracks.instances_by_frame[frame_idx].append((track_id, position))
    return tracks


def build_aruco_model(args: argparse.Namespace) -> ArUcoModel:
    """The ArUco detector for the ``--aruco-*`` options."""
    return ArUcoModel.withTagSet(
        args.aruco_marker_set,
        adaptiveThreshWinSizeMin=args.adaptiveThreshWinSizeMin,
        adaptiveThreshWinSizeMax=args.adaptiveThreshWinSizeMax,
        adaptiveThreshWinSizeStep=args.adaptiveThreshWinSizeStep,
        adaptiveThreshConstant=args.adaptiveThreshConstant,
        perspectiveRemoveIgnoredMarginPerCell=args.perspectiveRemoveIgnoredMarginPerCell,
        errorCorrectionRate=args.errorCorrectionRate,
        tag_subset_list=args.aruco_marker_subset,
        detector_overrides=(
            None
            if args.aruco_min_marker_perimeter_rate is None
            else {"minMarkerPerimeterRate": args.aruco_min_marker_perimeter_rate}
        ),
    )


def _quadrant_scope_from_args(args: argparse.Namespace) -> QuadrantScope:
    """Build the scope; raises ``ValueError`` for an inconsistent map, ROIs or grid."""
    tag_sets, map_rois = parse_tag_map(args.closed_set_map) if args.closed_set_map else (None, None)
    rois = parse_roi_args(args.quadrant_roi) or map_rois
    region_names = set(rois) if rois else set(make_grid(1, 1, *args.quadrant_grid))
    unknown = sorted(set(tag_sets or {}) - region_names)
    if unknown:
        raise ValueError(f"regions {unknown} are not among the regions {sorted(region_names)}")
    joint = getattr(args, "identity_engine", "legacy") == "joint"
    return QuadrantScope(
        tag_sets=tag_sets,
        rois=rois,
        grid=tuple(args.quadrant_grid),
        infer=args.closed_set_infer,
        # the joint engine also takes a quadrant of three tags plus one untagged bee
        expected_size=args.colony_size if joint else args.closed_set_size,
        min_support=args.closed_set_min_support,
        min_fraction=args.closed_set_min_fraction,
        min_purity=args.closed_set_min_purity,
        min_size=3 if joint else None,
    )


def build_quadrant_scope(args: argparse.Namespace) -> QuadrantScope | None:
    """The quadrant-aware closed-set scope, or ``None`` outside the quadrant modes.

    Raises:
        SystemExit: If the map, ROIs or grid are inconsistent.
    """
    if not is_quadrant_mode(args):
        return None
    try:
        return _quadrant_scope_from_args(args)
    except ValueError as exc:
        raise SystemExit(f"invalid quadrant closed-set configuration: {exc}") from exc


def build_evidence_sources(args: argparse.Namespace) -> list[EvidenceSource]:
    """Extra identity evidence (currently idtracker.ai trajectories); empty by default."""
    if not args.idtracker_trajectories:
        return []
    return [
        IdtrackerEvidence.from_file(
            args.idtracker_trajectories,
            offset=args.idtracker_offset,
            frame_offset=args.idtracker_frame_offset,
            weight=args.idtracker_weight,
            min_id_prob=args.idtracker_min_prob,
            max_dist=args.idtracker_max_dist,
            min_margin=args.idtracker_min_margin,
            min_support=args.idtracker_min_support,
            min_purity=args.idtracker_min_purity,
            allowed_tags=args.closed_set,
        )
    ]


def run_matching(
    args: argparse.Namespace,
    tracks: SleapTracks,
    aruco_model: ArUcoModel | None,
    evidence_sources: list[EvidenceSource],
    scope: QuadrantScope | None,
    *,
    decode_source: DecodeSource | None = None,
) -> tuple[Matching, dict]:
    """Match the SLEAP tracks to ArUco tags frame by frame.

    Args:
        args: The parsed options.
        tracks: The SLEAP tracks.
        aruco_model: The tag detector; ``None`` if ``decode_source`` serves the decodes.
        evidence_sources: Extra identity evidence.
        scope: The quadrant scope, if any.
        decode_source: Optional precomputed decodes (``--decodes``) instead of the video.

    Returns:
        The :class:`Matching` (for its diagnostics) and its ``{frame: {track: tag}}`` result.
    """
    if args.closed_set_max_distance is not None:
        max_distance = args.closed_set_max_distance
    else:
        max_distance = QUADRANT_MAX_DISTANCE if scope is not None else None
    matching = Matching(
        args.video_path,
        args.start_frame,
        args.end_frame,
        marker_detector=aruco_model.detect if aruco_model is not None else (lambda _: []),
        decode_source=decode_source,
        aruco_crop_size=args.aruco_crop_size,
        half_rolling_window_size=args.half_rolling_window_size,
        tag_node_dict=tracks.tag_locations,
        evidence_sources=evidence_sources,
        closed_set=args.closed_set,
        eliminate=args.closed_set_elimination,
        strict_carry_forward=args.strict_carry_forward,
        elimination_max_jump=args.closed_set_max_jump,
        elimination_max_distance=max_distance,
        untracked_counts=tracks.untracked_counts,
        quadrants=scope,
        position_dict=tracks.positions if scope is not None else None,
        untracked_positions=tracks.untracked_positions if scope is not None else None,
    )
    return matching, matching.match()


def report_diagnostics(
    args: argparse.Namespace,
    tracks: SleapTracks,
    matching: Matching,
    matching_dict: dict,
    scope: QuadrantScope | None,
) -> None:
    """Log the diagnostics (always in the opt-in modes); write ``--report-json`` on request."""
    if not (args.report_json or scope is not None or args.closed_set):
        return
    before = matching.pre_elimination
    if before is None:
        raise RuntimeError("Matching.match() did not record the assignments before elimination")
    report = build_report(
        scope,
        args.closed_set,
        args.start_frame,
        args.end_frame,
        before,
        matching_dict,
        tracks.instances_by_frame,
        elimination=matching.elimination_result,
        extra={
            "video": args.video_path,
            "slp": args.slp_path,
            "frame_size": list(matching.frame_size) if matching.frame_size else None,
            # the rolling window never assigns the first half-window frames (the tail is
            # carried forward)
            "cost_matrix_unassigned_head_frames": args.half_rolling_window_size,
            "options": {
                k: v
                for k, v in vars(args).items()
                if k.startswith(("closed_set", "quadrant", "position_node"))
            },
        },
    )
    logger.info(
        "Instances without a tag: %s of %s before elimination, %s after; by cause: %s",
        report["n_instances_without_tag_before_elimination"],
        report["n_instances"],
        report["n_instances_without_tag_after_elimination"],
        report["n_none_instances_by_cause"],
    )
    if args.report_json:
        with Path(args.report_json).open("w") as handle:
            json.dump(report, handle, indent=2, default=_json_default)
        logger.info("Wrote report to %s", args.report_json)


def log_evidence_alignment(evidence_sources: Sequence[Any]) -> None:
    """Log how each idtracker.ai animal was aligned to a tag (accepted or refused and why)."""
    for source in evidence_sources:
        for animal, entry in getattr(source, "alignment", {}).items():
            logger.info(
                "idtracker animal %s -> tag %s (support %s, purity %.3f): %s",
                animal,
                entry.tag,
                entry.support,
                entry.purity,
                "accepted" if entry.accepted else f"refused ({entry.reason})",
            )


def write_output(
    args: argparse.Namespace, matching_dict: dict, labels: sio.Labels | None = None
) -> None:
    """Re-track the SLEAP file with the matched tags and save it to ``--output-path``.

    Args:
        args: The parsed command line.
        matching_dict: ``{frame: {track: tag}}`` from the matching.
        labels: The already loaded ``--slp-path`` file (loaded again from disk if omitted).
    """
    # The reconstruction assumes each track has a single tag assigned to it; generalising
    # it so a track can switch tags over time is future work.
    source = args.slp_path if labels is None else labels
    retracked = update_labeled_frames(source, matching_dict, args.start_frame, args.end_frame)
    save_labels(retracked, args.output_path, fast=getattr(args, "fast_write", True))


def build_decode_source(args: argparse.Namespace) -> DecodeSource:
    """The :class:`~naps.matching.DecodeSource` selected by ``--decodes`` / ``--decoder``.

    ``--decodes`` serves precomputed votes and never opens the video. Otherwise both decoders read
    the video and decode the tag around every instance with the ``--aruco-*`` settings (and the
    ``--aruco-marker-subset`` as closed set of the dictionary, if given).
    """
    if getattr(args, "decodes", None):
        return PrecomputedDecodeSource(args.decodes, video=args.video_path)
    if args.decoder == "fast":
        params = DecodeParams(
            tag_set=args.aruco_marker_set,
            closed_set=None
            if args.aruco_marker_subset is None
            else tuple(args.aruco_marker_subset),
            win_min=args.adaptiveThreshWinSizeMin,
            win_max=args.adaptiveThreshWinSizeMax,
            win_step=args.adaptiveThreshWinSizeStep,
            thresh_const=args.adaptiveThreshConstant,
            margin=args.perspectiveRemoveIgnoredMarginPerCell,
            error_correction_rate=args.errorCorrectionRate,
            min_marker_perimeter_rate=(
                FAST_MIN_MARKER_PERIMETER_RATE
                if args.aruco_min_marker_perimeter_rate is None
                else args.aruco_min_marker_perimeter_rate
            ),
        )
        fast = TagDecoderSource(
            args.video_path,
            TagDecoder(params),
            crop_half=args.aruco_crop_size,
            reader_threads=decoder_threads(args),
        )
        return chunked_decode_source(args, fast)
    aruco_model = build_aruco_model(args)
    legacy = OpenCVDecodeSource(
        args.video_path, aruco_model.detect, args.aruco_crop_size, decoder_threads(args)
    )
    return chunked_decode_source(args, legacy)


def decode_workers(args: argparse.Namespace) -> int:
    """Number of parallel decode workers asked for (``--decode-workers``; 0 = all CPUs)."""
    workers = getattr(args, "decode_workers", 1)
    return available_cpus() if workers == 0 else workers


def parallel_decode_requested(args: argparse.Namespace) -> bool:
    """Whether ``--decode-workers`` (more than one) or ``--decode-chunk-frames`` were given."""
    return decode_workers(args) > 1 or getattr(args, "decode_chunk_frames", None) is not None


def decoder_threads(args: argparse.Namespace) -> int | None:
    """Decoder threads of each video reader: 1 if several workers decode, else OpenCV's default."""
    return 1 if decode_workers(args) > 1 else None


def chunked_decode_source(
    args: argparse.Namespace, inner: DecodeSource, background: Callable[[], None] | None = None
) -> DecodeSource:
    """``inner`` decoded in parallel chunks if ``--decode-workers`` / ``--decode-chunk-frames`` ask.

    Without those options ``inner`` itself is returned, so the serial path is untouched.

    Args:
        args: The parsed options.
        inner: The source that decodes one chunk.
        background: Optional job run while the chunks are decoded.
    """
    if not parallel_decode_requested(args):
        return inner
    return ChunkedDecodeSource(
        inner,
        workers=decode_workers(args),
        chunk_frames=getattr(args, "decode_chunk_frames", None),
        align=max(getattr(args, "decode_gop", 0), 1),
        offset=getattr(args, "decode_gop_offset", 25),
        background=background,
    )


def build_joint_config(args: argparse.Namespace) -> JointConfig:
    """The :class:`~naps.joint.JointConfig` for the joint-engine options of ``args``."""
    return JointConfig(
        identity=IdentityParams(threshold=args.min_confidence),
        consistency=ConsistencyParams(enabled=args.consistency, gap_max=args.interpolate_max_gap),
        tracklets=TrackletParams(),
        tracklet_mode=args.tracklets,
        colony_size=args.colony_size,
        # an explicit two-tag closed set (a pair of animals) lowers the region minimum to 2
        min_tags=2 if args.closed_set and len(set(args.closed_set)) == 2 else 3,
    )


def _sidecar_path(args: argparse.Namespace) -> Path:
    """Where the identity sidecar goes (``--identity-sidecar`` or next to the output file)."""
    if args.identity_sidecar:
        return Path(args.identity_sidecar)
    out = Path(args.output_path)
    stem = out.name.removesuffix(".h5").removesuffix(".analysis").removesuffix(".slp")
    return out.with_name(f"{stem}.identity.npz")


def run_joint(args: argparse.Namespace) -> JointResult:
    """Run the joint identity engine: decode every instance, infer, diagnose, write.

    Args:
        args: Validated command-line arguments (``--identity-engine joint``).

    Returns:
        The :class:`~naps.joint.JointResult` after the untrusted-window policy was applied.
    """
    t0 = time.perf_counter()
    logger.info("Loading predictions...")
    labels = sio.load_slp(args.slp_path)
    table = table_from_labels(
        labels,
        args.start_frame,
        args.end_frame,
        args.tag_node_name,
        position_node=args.position_node_name,
    )
    decode_source = build_decode_source(args)
    config = build_joint_config(args)
    evidence_sources = build_evidence_sources(args)
    scope = build_quadrant_scope(args)
    logger.info("Starting joint matching...")
    result = JointMatching(
        table,
        decode_source,
        closed_set=args.closed_set,
        quadrants=scope,
        config=config,
        evidence_sources=evidence_sources,
    ).match()
    params = DiagnosticsParams(
        window_frames=args.window_frames,
        untrusted_policy=args.untrusted_policy,
        self_validate=args.self_validate,
    )
    statuses, validation = assess_result(result, params)
    n_withheld = apply_policy(result, statuses, params)
    session = args.session or Path(args.video_path).resolve().parent.name
    options = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    sidecar = _sidecar_path(args)
    write_sidecar(sidecar, result, session, {"options": options})
    used_tags = {t for run in result.regions.values() for t in run.tags}
    new_labels = apply_instance_tags(
        labels,
        table.first_frame,
        table.inst_index,
        result.slot_tag,
        result.slot_conf,
        identity_names={t: qualify_tag(session, t) for t in used_tags},
    )
    new_labels.save(args.output_path)
    if args.report_json:
        report = build_joint_report(
            result,
            statuses,
            validation,
            config.consistency,
            extra={
                "video": args.video_path,
                "slp": args.slp_path,
                "session": session,
                "sidecar": str(sidecar),
                "n_instances_withheld": n_withheld,
                "options": options,
                "runtime_s": round(time.perf_counter() - t0, 2),
            },
        )
        with Path(args.report_json).open("w") as handle:
            json.dump(report, handle, indent=2, default=_json_default)
        logger.info("Wrote report to %s", args.report_json)
    logger.info("Joint engine finished in %.2f seconds.", time.perf_counter() - t0)
    return result


def _legacy_instance_keys(tracks: SleapTracks) -> dict[int, dict[Hashable, int]]:
    """``{frame: {track id: index in the labeled frame}}`` of the tracked instances."""
    return {
        frame: {track: index for index, (track, _) in enumerate(entries) if track is not None}
        for frame, entries in tracks.instances_by_frame.items()
    }


def run(args: argparse.Namespace) -> None:
    """Run the whole pipeline for validated ``args``."""
    if getattr(args, "identity_engine", "legacy") == "joint":
        run_joint(args)
        return
    t0_total = time.perf_counter()

    logger.info("Loading predictions...")
    loaded: dict[str, sio.Labels] = {}

    def load_labels() -> None:
        loaded["labels"] = sio.load_slp(str(args.slp_path))

    fast_read = getattr(args, "fast_read", True)
    if not fast_read:
        load_labels()  # the slow reference path: the whole file first
    tracks = load_sleap_tracks(
        loaded.get("labels", args.slp_path),
        args.start_frame,
        args.end_frame,
        args.tag_node_name,
        args.position_node_name,
        fast=fast_read,
    )

    aruco_model: ArUcoModel | None = None
    decode_source: DecodeSource | None = None
    if getattr(args, "decodes", None):
        logger.info("Using the precomputed decodes %s (the video is not read).", args.decodes)
        decode_source = PrecomputedDecodeSource(
            args.decodes, video=args.video_path, keys=_legacy_instance_keys(tracks)
        )
    else:
        logger.info("Create ArUco model...")
        t0 = time.perf_counter()
        aruco_model = build_aruco_model(args)
        logger.info("ArUco model built in %.2f seconds.", time.perf_counter() - t0)
        # parallel decode (opt-in): the Labels, needed again to write the output, are loaded
        # while the video is decoded
        if parallel_decode_requested(args):
            decode_source = chunked_decode_source(
                args,
                OpenCVDecodeSource(
                    args.video_path, aruco_model.detect, args.aruco_crop_size, decoder_threads(args)
                ),
                None if loaded else load_labels,
            )

    # Opt-in evidence sources (none by default)
    scope = build_quadrant_scope(args)
    evidence_sources = build_evidence_sources(args)

    logger.info("Starting matching...")
    t0 = time.perf_counter()
    matching, matching_dict = run_matching(
        args, tracks, aruco_model, evidence_sources, scope, decode_source=decode_source
    )
    logger.info("Done matching in %.2f seconds.", time.perf_counter() - t0)

    report_diagnostics(args, tracks, matching, matching_dict, scope)
    log_evidence_alignment(evidence_sources)

    logger.info("Reconstructing SLEAP file...")
    t0 = time.perf_counter()
    if not loaded:
        load_labels()
    write_output(args, matching_dict, loaded["labels"])
    logger.info("Done reconstructing SLEAP file in %.2f seconds.", time.perf_counter() - t0)

    logger.info("Complete NAPS runtime: %.2f", time.perf_counter() - t0_total)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``naps-track``.

    Args:
        argv: Command-line arguments (default: ``sys.argv[1:]``).

    Returns:
        The process exit status (0 on success).

    Raises:
        SystemExit: On invalid options (status 1 with a message, or 2 from argparse).
    """
    args = build_parser().parse_args(argv)
    with log_verbosity(args.verbose, args.quiet):
        warn_deprecated_options(args)
        validate_args(args)
        try:
            run(args)
        except DecodesError as exc:
            raise SystemExit(f"--decodes: {exc}") from exc
    return 0


if __name__ == "__main__":
    sys.exit(main())
