"""General helper functions for interfacing with SLEAP files (via ``sleap_io``)."""

import logging
import os
import pathlib
from collections.abc import Mapping, Sequence

import attrs
import h5py
import numpy as np
import sleap_io as sio
from numpy.typing import NDArray

from naps.tags import normalize_tag, tag_name

__all__ = [
    "apply_instance_tags",
    "get_location_matrix",
    "load_tracks_from_slp",
    "update_labeled_frames",
]

logger = logging.getLogger(__name__)

# Separator of the numeric suffix in SLEAP track names (``track_0``, ``track_1``, ...).
_TRACK_NAME_SEPARATOR = "_"


# Pulled from SLEAP's codebase
def get_location_matrix(
    labels: sio.Labels, all_frames: bool, video: sio.Video | None = None
) -> tuple[NDArray[np.float64], list[sio.Node]]:
    """Build a numpy matrix with point location data.

    This function assumes either all instances have tracks or no instances have tracks
    (instances without a track are ignored when the project has tracks). User-labeled instances
    are preferred over predicted ones. If one track has several instances in a frame, the last
    one wins and a warning is logged.

    Args:
        labels: The :py:class:`sleap_io.Labels` from which to get data.
        all_frames: If True, include NaN rows so that the frame index lines up with the rows of
            the output. Otherwise, there are only rows for the frames between the first and last
            frames with labeling data.
        video: The :py:class:`sleap_io.Video` from which to get data. Defaults to the first video
            of ``labels``.

    Returns:
        ``(locations, nodes)``: the point locations with shape (frames, nodes, 2, tracks) (NaN
        where nothing was labeled) and the nodes of the skeleton.

    Raises:
        IndexError: If ``labels`` has no videos, or the video has no labeled frames.
    """
    track_count = len(labels.tracks) or 1
    node_count = len(labels.skeletons[0].nodes)

    # Retrieve frames from current video only
    if video is None:
        if not labels.videos:
            msg = "There are no videos in this project. No occupancy matrix to return."
            raise IndexError(msg)
        video = labels.videos[0]
    labeled_frames = labels.find(video)

    frame_idxs = sorted(lf.frame_idx for lf in labeled_frames)
    if not frame_idxs:
        msg = f"No labeled frames in {video.filename}. No occupancy matrix to return."
        raise IndexError(msg)
    first_frame_idx = 0 if all_frames else frame_idxs[0]
    frame_count = frame_idxs[-1] - first_frame_idx + 1  # count includes unlabeled frames

    # "tracks" matrix: frames * nodes * 2 * tracks
    locations_matrix = np.full((frame_count, node_count, 2, track_count), np.nan, dtype=float)

    lfs_instances, n_collisions = _select_instances(labels, labeled_frames)
    if n_collisions:
        logger.warning(
            "There is more than one instance per track on a single frame in %s (frame, track) "
            "pairs; the last instance overwrites the others.",
            n_collisions,
        )

    for lf, inst in lfs_instances:
        frame_i = lf.frame_idx - first_frame_idx
        # Without tracks we assume a single instance per frame (we cannot use
        # lf.instances.index(inst): that would need the max number of instances in any frame).
        track_i = 0 if inst.track is None else labels.tracks.index(inst.track)
        locations_matrix[frame_i, ..., track_i] = inst.numpy()
    return locations_matrix, labels.skeletons[0].nodes


def _select_instances(
    labels: sio.Labels, labeled_frames: Sequence[sio.LabeledFrame]
) -> tuple[list[tuple[sio.LabeledFrame, sio.Instance | sio.PredictedInstance]], int]:
    """Pick, per frame and track, the user instances if any, else the predicted ones.

    Returns:
        ``(lf, instance)`` pairs, and the number of (frame, track) pairs with more than one
        instance.
    """
    tracks = labels.tracks or [None]  # None: project without tracks
    lfs_instances = []
    n_collisions = 0
    for lf in labeled_frames:
        for track in tracks:
            track_instances = [inst for inst in lf.user_instances if inst.track == track]
            if not track_instances:
                track_instances = [inst for inst in lf.predicted_instances if inst.track == track]
            lfs_instances.extend((lf, inst) for inst in track_instances)
            if track is not None and len(track_instances) > 1:
                n_collisions += 1
    return lfs_instances, n_collisions


def load_tracks_from_slp(slp_path: str | os.PathLike[str]) -> tuple[NDArray[np.float64], list[str]]:
    """Load tracks from a SLEAP project file or analysis h5.

    Args:
        slp_path: Path to a SLEAP project file (``.slp``) or analysis h5 (``.h5``). The h5 is
            typically generated using ``sleap-convert``. Analysis h5s are much faster to load
            and thus preferable.

    Returns:
        ``(locations, node_names)`` where ``locations`` has shape (frames, nodes, 2, tracks) and
        ``node_names`` is a list of node names.

    Raises:
        ValueError: If the suffix of ``slp_path`` is neither ``.slp`` nor ``.h5``.
    """
    suffix = pathlib.Path(slp_path).suffix
    if suffix == ".slp":
        dset = sio.load_slp(slp_path)
        locations, nodes = get_location_matrix(dset, all_frames=True)
        return locations, [node.name for node in nodes]
    if suffix == ".h5":
        with h5py.File(slp_path, "r") as f:
            locations = f["tracks"][:].T
            node_names = [n.decode() for n in f["node_names"][:]]
        return locations, node_names
    msg = f"{slp_path}: expected a SLEAP project (.slp) or analysis (.h5) file, got {suffix!r}"
    raise ValueError(msg)


def _clean_matching_dict(
    matching_dict: Mapping[int, Mapping[int, object]],
) -> dict[int, dict[int, int]]:
    """Normalise all tags of ``matching_dict`` to ``int`` and drop invalid ones.

    Tags are normalised so a track is always named ``ArUcoTag#<int>`` (never ``ArUcoTag#21.0``,
    ``ArUcoTag#None`` or ``ArUcoTag#nan``). Entries that are not valid tags leave the instance
    untracked; those other than ``""``/``None`` are counted and logged.
    """
    skipped_invalid = 0
    clean: dict[int, dict[int, int]] = {}
    for frame_idx, track_dict in matching_dict.items():
        row = {}
        for track, tag in track_dict.items():
            value = normalize_tag(tag, strict=True)
            if value is None:
                if tag not in ("", None):
                    skipped_invalid += 1
                continue
            row[track] = value
        clean[frame_idx] = row
    if skipped_invalid:
        logger.warning("Ignored %s invalid (non-integer/NaN) tag assignments", skipped_invalid)
    return clean


def _tag_of_instance(
    inst: sio.Instance | sio.PredictedInstance, row: Mapping[int, int], frame_idx: int
) -> int | None:
    """Tag of ``inst`` from the frame's ``{sleap track number: tag}`` row, or None."""
    if inst.track is None:
        logger.debug("Frame %s: instance has no SLEAP track, leaving untracked.", frame_idx)
        return None
    try:
        return row[int(inst.track.name.split(_TRACK_NAME_SEPARATOR)[-1])]
    except (KeyError, ValueError) as e:
        logger.debug(
            "Frame %s: could not map track %r to a tag (%s: %s), leaving untracked.",
            frame_idx,
            inst.track.name,
            type(e).__name__,
            e,
        )
        return None


def _keep_last_instance_per_tag(tags: list[int | None]) -> int:
    """Set all but the last instance of each tag to None (in place); return how many were reset."""
    seen: set[int] = set()
    n_reset = 0
    for i in reversed(range(len(tags))):
        tag = tags[i]
        if tag is None:
            continue
        if tag in seen:
            tags[i] = None
            n_reset += 1
        else:
            seen.add(tag)
    return n_reset


def update_labeled_frames(
    slp_path: str | os.PathLike[str] | sio.Labels,
    matching_dict: Mapping[int, Mapping[int, object]],
    first_frame_idx: int,
    last_frame_idx: int,
) -> sio.Labels:
    """Generate a :py:class:`sleap_io.Labels` from a SLEAP file with tracks from ``matching_dict``.

    Frames are selected and looked up by ``LabeledFrame.frame_idx``, never by position in the
    list of labeled frames. An instance is left untracked (``track=None``) if it had no SLEAP
    track, if its SLEAP track number (the ``_<n>`` suffix of its name) has no valid tag in its
    frame, or if another instance of the same frame already carries that tag. In the last case
    the last instance of the frame keeps the tag: it is the one whose position NAPS decoded
    (``naps-track`` lets the last instance of a SLEAP track win), so a tag is never given to two
    instances of one frame.

    Args:
        slp_path: Path to the SLEAP project file (``.slp``), or the already loaded
            :py:class:`sleap_io.Labels` of it (it is not modified: instances are copied).
        matching_dict: ``matching_dict[frame][track] -> tag``. Tags are normalised to ``int``;
            invalid ones (``""``, ``None``, NaN, non-integral) leave the instance untracked.
        first_frame_idx: First frame index (inclusive) to include in the reconstructed file.
        last_frame_idx: Last frame index (inclusive) to include in the reconstructed file.

    Returns:
        Labels containing the frames with ``first_frame_idx <= frame_idx <= last_frame_idx``,
        with instances re-tracked as ``ArUcoTag#{tag}``.
    """
    labels = slp_path if isinstance(slp_path, sio.Labels) else sio.load_slp(slp_path)
    clean = _clean_matching_dict(matching_dict)

    all_tags = {tag for row in clean.values() for tag in row.values()}
    logger.info("Total tags: %s", len(all_tags))
    new_tracks = {tag: sio.Track(name=tag_name(tag)) for tag in sorted(all_tags)}

    new_lfs = []
    n_duplicates = 0
    for lf in labels.labeled_frames:
        if lf.frame_idx < first_frame_idx or lf.frame_idx > last_frame_idx:
            continue
        row = clean.get(lf.frame_idx, {})
        tags = [_tag_of_instance(inst, row, lf.frame_idx) for inst in lf.instances]
        n_duplicates += _keep_last_instance_per_tag(tags)
        tracked_instances = [
            attrs.evolve(inst, track=None if tag is None else new_tracks[tag])
            for inst, tag in zip(lf.instances, tags, strict=True)
        ]
        new_lfs.append(
            sio.LabeledFrame(video=lf.video, frame_idx=lf.frame_idx, instances=tracked_instances)
        )
    if n_duplicates:
        logger.warning(
            "%s instances shared a tag with a later instance of the same frame and were left "
            "untracked",
            n_duplicates,
        )
    return sio.Labels(labeled_frames=new_lfs, videos=labels.videos, skeletons=labels.skeletons)


def apply_instance_tags(  # noqa: PLR0917 (every argument is one array of the instance table)
    labels: sio.Labels,
    first_frame: int,
    inst_index: NDArray[np.int64],
    slot_tag: NDArray[np.int64],
    slot_score: NDArray[np.float32],
    last_frame: int | None = None,
    identity_names: Mapping[int, str] | None = None,
) -> sio.Labels:
    """Re-track a SLEAP file with per-instance tags (no SLEAP track ids needed).

    The joint identity engine labels *instances*, tracked or not. ``inst_index[row, slot]`` is the
    position of an instance in ``LabeledFrame.instances`` (``-1`` for an empty slot) and
    ``slot_tag[row, slot]`` its tag (``-1`` = leave untracked). Row ``r`` is video frame
    ``first_frame + r``. Instances that carry a tag get the track ``ArUcoTag#<tag>`` (``slot_tag``
    is an integer array, so the name is always ``ArUcoTag#<int>``), ``tracking_score`` and
    ``identity_score`` set to ``slot_score``; every other instance is written without a track.

    SLEAP stores ``identity_score`` only together with an :class:`sleap_io.Identity`, so every
    tagged instance also gets the identity of its tag (named ``identity_names[tag]``, e.g. the
    session-qualified ``"<session>:ArUcoTag#<tag>"``, default the track name).

    Args:
        labels: The loaded SLEAP labels (the same object the instance table was built from).
        first_frame: Video frame number of row 0 of the arrays.
        inst_index: ``(F, S)`` instance position in its labeled frame, -1 = none.
        slot_tag: ``(F, S)`` tag id, -1 = untracked.
        slot_score: ``(F, S)`` confidence of the label.
        last_frame: Last frame to write (default: the last row of the arrays; a larger value is
            clipped to it, the arrays hold no labels beyond).
        identity_names: Optional ``{tag: identity name}``.

    Returns:
        Labels with the frames ``first_frame..last_frame`` and re-tracked instances.
    """
    last = first_frame + len(slot_tag) - 1
    if last_frame is not None:
        last = min(last, last_frame)
    tags = sorted({int(t) for t in np.unique(slot_tag) if t >= 0})
    new_tracks = {tag: sio.Track(name=tag_name(tag)) for tag in tags}
    names = identity_names or {}
    identities = {tag: sio.Identity(name=names.get(tag, tag_name(tag))) for tag in tags}
    new_lfs = []
    for lf in labels.labeled_frames:
        if lf.frame_idx < first_frame or lf.frame_idx > last:
            continue
        row = lf.frame_idx - first_frame
        by_instance = {
            int(inst_index[row, s]): s
            for s in range(inst_index.shape[1])
            if inst_index[row, s] >= 0
        }
        instances = []
        for i, inst in enumerate(lf.instances):
            slot = by_instance.get(i)
            tag = int(slot_tag[row, slot]) if slot is not None else -1
            if tag < 0:
                instances.append(attrs.evolve(inst, track=None, identity=None, identity_score=None))
                continue
            score = float(slot_score[row, slot])
            instances.append(
                attrs.evolve(
                    inst,
                    track=new_tracks[tag],
                    tracking_score=score,
                    identity=identities[tag],
                    identity_score=score,
                )
            )
        new_lfs.append(
            sio.LabeledFrame(video=lf.video, frame_idx=lf.frame_idx, instances=instances)
        )
    return sio.Labels(labeled_frames=new_lfs, videos=labels.videos, skeletons=labels.skeletons)
