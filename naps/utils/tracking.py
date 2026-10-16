#!/usr/bin/env python
"""Array of tracked animal coordinates, with interaction assignment and plotting."""

from __future__ import annotations

import csv
import importlib
import logging
from collections.abc import Iterator, Sequence
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pandas as pd
import sleap_io as sio
import yaml

from naps.sleap_utils import load_tracks_from_slp
from naps.utils.interactor import Interactor

logger = logging.getLogger("NAPS Logger")

#: Columns of the interactions TSV written by :meth:`TrackingArray.assign`.
INTERACTION_COLUMNS = [
    "Origin interactor",
    "Destination interactor",
    "Interaction Name",
    "Interaction Frame",
    "Origin Area",
    "Destination Area",
]


class TrackingArray:
    """Coordinates of all tracks as ``(frames, nodes, 2, tracks)`` plus the interactor model.

    Use :meth:`fromSLEAP` or :meth:`fromH5` to build one from a file.
    """

    def __init__(  # noqa: PLR0917 (public API: positional order kept)
        self,
        frames_coords_array: np.ndarray | str | None = None,
        tracks: Sequence[str] | None = None,
        nodes: Sequence[str] | None = None,
        interactor_model_dict: dict | None = None,
        chunk_size: int = 5000,
        threads: int = 1,
        **kwargs: object,  # extra keyword arguments are accepted and ignored
    ) -> None:
        """Store the coordinates and processing options.

        Args:
            frames_coords_array: ``(frames, nodes, 2, tracks)`` array. ``None`` (or the legacy empty
                string) gives an empty array.
            tracks: Track names, one per last-axis entry of ``frames_coords_array``.
            nodes: Node names, one per second-axis entry.
            interactor_model_dict: ``Skeleton`` / ``Models`` / ``Interactions`` model dictionary.
            chunk_size: Number of frames processed at a time by :meth:`assign`.
            threads: Number of worker processes used by :meth:`assign`.
            **kwargs: Ignored.
        """
        if frames_coords_array is None or isinstance(frames_coords_array, str):
            frames_coords_array = np.empty((0, 0, 2, 0))
        self._frames_coords_array: np.ndarray = frames_coords_array
        self._tracks: list[str] = list(tracks) if tracks is not None else []
        self._nodes: list[str] = list(nodes) if nodes is not None else []
        self._interactor_model_dict: dict = (
            interactor_model_dict if interactor_model_dict is not None else {}
        )
        self._chunk_size = chunk_size
        self._threads = threads

        # State of the chunk currently being processed
        self._process_frame_start = 0
        self._process_frame_end = 0
        self._process_coords_array: np.ndarray | None = None

    def _track_pairs(self) -> Iterator[tuple[int, int]]:
        """Every pair of track indices."""
        yield from combinations(range(len(self._tracks)), 2)

    def assign(self, out_file: str | Path, frame_start: int = 0, frame_end: int = -1) -> None:
        """Write the interactions between every pair of tracks to a tab-separated file.

        Args:
            out_file: Output TSV (columns :data:`INTERACTION_COLUMNS`).
            frame_start: First frame to process.
            frame_end: End frame (exclusive); ``-1`` means all frames.
        """
        frame_end = frame_end if frame_end != -1 else self._frames_coords_array.shape[0]

        with Path(out_file).open("w", newline="") as outfile:
            csv_writer = csv.DictWriter(outfile, fieldnames=INTERACTION_COLUMNS, delimiter="\t")
            csv_writer.writeheader()

            for chunk_frame_start in range(frame_start, frame_end, self._chunk_size):
                # Slice out the frames of this chunk
                self._process_frame_start = chunk_frame_start
                chunk_end = chunk_frame_start + self._chunk_size
                self._process_frame_end = min(frame_end, chunk_end)
                self._process_coords_array = self._frames_coords_array[
                    self._process_frame_start : self._process_frame_end, :, :, :
                ]

                if self._threads > 1:
                    with Pool(processes=self._threads) as pool:
                        for report_list in pool.starmap(
                            self._check_for_interactions, self._track_pairs()
                        ):
                            csv_writer.writerows(report_list)
                else:
                    for track_a, track_b in self._track_pairs():
                        csv_writer.writerows(self._check_for_interactions(track_a, track_b))

    def plot(  # noqa: PLR0917 (public API: positional order kept)
        self,
        interactions_filename: str | Path,
        models_to_plot: Sequence[str],
        video_filename: str | Path,
        out_file: str | Path,
        frame_start: int = 0,
        frame_end: int = -1,
        limit_interactions: Sequence[str] | None = None,
        plot_color: str | Sequence[float] | None = None,
        line_width: int = 2,
    ) -> None:
        """Draw the interacting animals' models onto the video and save the result.

        Args:
            interactions_filename: TSV written by :meth:`assign`.
            models_to_plot: Names of the interactor models to draw for every interacting animal.
            video_filename: Input video.
            out_file: Output video (mp4v).
            frame_start: First frame to draw.
            frame_end: Last frame to draw (inclusive); ``-1`` means the last frame.
            limit_interactions: Only plot these interaction names (default: all).
            plot_color: ``None`` (red), an RGB triple (0-255) or a matplotlib colour name.
            line_width: Line width in pixels.

        Raises:
            ValueError: If no interaction lies within the frame range, or ``plot_color`` is not a
                triple.
        """
        n_frames = self._frames_coords_array.shape[0]
        frame_end = frame_end if frame_end != -1 else n_frames
        frame_end = frame_end - 1 if frame_end >= n_frames else frame_end

        # DataFrame of the instances to plot per frame
        frame_instances = self.processInteractionsFile(
            interactions_filename, frame_start, frame_end, limit_interactions=limit_interactions
        )

        # OpenCV draws in BGR
        color = self._resolve_color(plot_color)[::-1]

        video = cv2.VideoCapture(str(video_filename))
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

        video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = int(video.get(cv2.CAP_PROP_FPS))

        out_video = cv2.VideoWriter(
            str(out_file),
            cv2.VideoWriter.fourcc(*"mp4v"),
            video_fps,
            (video_width, video_height),
        )

        current_frame = frame_start
        frame_read, frame_img = video.read()
        while frame_read and current_frame <= frame_end:
            instances_to_plot = frame_instances[
                frame_instances["Interaction Frame"] == current_frame
            ]["Instances"].values
            # processInteractionsFile groups by frame: at most one row per frame
            instances = instances_to_plot[0] if len(instances_to_plot) > 0 else []

            for instance_name in instances:
                instance_interactor = Interactor.withModelDict(
                    instance_name,
                    current_frame,
                    self._nodes,
                    self._frames_coords_array[
                        current_frame, :, :, self._tracks.index(instance_name)
                    ],
                    self._interactor_model_dict,
                )
                for model_name in models_to_plot:
                    polygon = instance_interactor._models[model_name]
                    model_coords = np.array(polygon.exterior.coords).round().astype(np.int32)
                    cv2.polylines(frame_img, [model_coords], True, color, line_width)

            out_video.write(frame_img)

            frame_read, frame_img = video.read()
            current_frame += 1

        video.release()
        out_video.release()

    @staticmethod
    def _resolve_color(plot_color: str | Sequence[float] | None) -> tuple[float, ...]:
        """RGB triple (0-255) for ``plot_color``: ``None`` is red, a str is a matplotlib colour."""
        if plot_color is None:
            return (255, 0, 0)
        if isinstance(plot_color, str):
            # matplotlib is an optional dependency, only needed to resolve colour names
            colors = importlib.import_module("matplotlib.colors")
            return tuple(v * 255 for v in colors.to_rgb(plot_color))
        if len(plot_color) != 3:
            raise ValueError(
                f"Unable to assign plot color: {plot_color}. Iterables must be of length 3"
            )
        return tuple(plot_color)

    def _check_for_interactions(self, track_a: int, track_b: int) -> list[dict]:
        """Interactions between two tracks over the frames of the current chunk."""
        if self._process_coords_array is None:
            raise RuntimeError("no chunk is being processed; call assign() instead")
        interaction_list: list[dict] = []

        # Subset the array to the two tracks
        filtered_coords_array = np.take(self._process_coords_array, [track_a, track_b], 3)

        for frame, row in enumerate(filtered_coords_array):
            # Skip frames with missing data for either track
            if np.isnan(row).any():
                continue

            # Frame number in the whole array
            adj_frame = self._process_frame_start + frame

            interactor1 = Interactor.withModelDict(
                self._tracks[track_a],
                adj_frame,
                self._nodes,
                row[:, :, 0],
                self._interactor_model_dict,
            )
            interactor2 = Interactor.withModelDict(
                self._tracks[track_b],
                adj_frame,
                self._nodes,
                row[:, :, 1],
                self._interactor_model_dict,
            )
            interaction_list.extend(interactor1.interacts(interactor2))

        return interaction_list

    @staticmethod
    def _select_model(model_dict: dict | None, model_yaml: str | Path) -> dict:
        """The interactor model from exactly one of a dict or a YAML file."""
        if model_dict and model_yaml:
            raise ValueError("Only a single Interactor model may be specified")
        if model_yaml:
            return TrackingArray.loadYAML(model_yaml)
        if model_dict:
            return model_dict
        raise ValueError("An Interactor model is required: give model_dict or model_yaml")

    @classmethod
    def fromSLEAP(  # noqa: N802 (public camelCase API)
        cls,
        sleap_file: str | Path,
        model_dict: dict | None = None,
        model_yaml: str | Path = "",
        **kwargs: Any,  # noqa: ANN401 (forwarded verbatim to __init__)
    ) -> TrackingArray:
        """Build a :class:`TrackingArray` from a SLEAP ``.slp`` file.

        Args:
            sleap_file: SLEAP project file.
            model_dict: Interactor model dictionary (exclusive with ``model_yaml``).
            model_yaml: Path of a YAML interactor model (exclusive with ``model_dict``).
            **kwargs: Passed to the constructor (e.g. ``threads``, ``chunk_size``).

        Raises:
            ValueError: If both or neither of ``model_dict`` / ``model_yaml`` are given.
        """
        tracks = [track.name for track in sio.load_slp(str(sleap_file)).tracks]

        # Array layout: frames, nodes, (x, y), tracks
        frames_coords_array, nodes = load_tracks_from_slp(str(sleap_file))

        model = cls._select_model(model_dict, model_yaml)
        return cls(
            frames_coords_array=frames_coords_array,
            tracks=tracks,
            nodes=nodes,
            interactor_model_dict=model,
            **kwargs,
        )

    @classmethod
    def fromH5(  # noqa: N802 (public camelCase API)
        cls,
        h5_file: str | Path,
        model_dict: dict | None = None,
        model_yaml: str | Path = "",
        **kwargs: Any,  # noqa: ANN401 (forwarded verbatim to __init__)
    ) -> TrackingArray:
        """Build a :class:`TrackingArray` from a SLEAP analysis ``.h5`` file.

        Args:
            h5_file: Analysis file with the datasets ``tracks``, ``track_names`` and ``node_names``.
            model_dict: Interactor model dictionary (exclusive with ``model_yaml``).
            model_yaml: Path of a YAML interactor model (exclusive with ``model_dict``).
            **kwargs: Passed to the constructor (e.g. ``threads``, ``chunk_size``).

        Raises:
            ValueError: If both or neither of ``model_dict`` / ``model_yaml`` are given.
        """
        with h5py.File(h5_file, "r") as handle:
            tracks = [name.decode("utf-8") for name in list(handle["track_names"][:])]
            logger.debug("Tracks: %s", tracks)
            # Array layout: frames, nodes, (x, y), tracks
            frames_coords_array = handle["tracks"][:].T
            nodes = [name.decode() for name in handle["node_names"][:]]

        model = cls._select_model(model_dict, model_yaml)
        return cls(
            frames_coords_array=frames_coords_array,
            tracks=tracks,
            nodes=nodes,
            interactor_model_dict=model,
            **kwargs,
        )

    @staticmethod
    def loadYAML(yaml_filename: str | Path) -> dict:  # noqa: N802 (public camelCase API)
        """Read a YAML file holding a mapping into a dict.

        Raises:
            ValueError: If the file is not valid YAML, is empty, or does not hold a mapping at the
                top level.
        """
        with Path(yaml_filename).open() as yaml_file:
            try:
                data = yaml.safe_load(yaml_file)
            except yaml.YAMLError as err:
                raise ValueError(f"Unable to parse {yaml_filename}: {err}") from err
        if not isinstance(data, dict):
            msg = f"Expected a mapping in {yaml_filename}, found {type(data).__name__}"
            raise ValueError(msg)  # noqa: TRY004 (a bad file is a user-input error)
        return data

    @staticmethod
    def processInteractionsFile(  # noqa: N802 (public camelCase API)
        interactions_file: str | Path,
        frame_start: int,
        frame_end: int,
        limit_interactions: Sequence[str] | None = None,
        expected_cols: Sequence[str] = tuple(INTERACTION_COLUMNS),
    ) -> pd.DataFrame:
        """Instances to plot per frame, from an interactions TSV.

        Args:
            interactions_file: TSV written by :meth:`assign`.
            frame_start: First frame (inclusive).
            frame_end: Last frame (inclusive).
            limit_interactions: Keep only these interaction names (default: all).
            expected_cols: Columns that must be present.

        Returns:
            DataFrame with the columns ``Interaction Frame`` and ``Instances`` (the list of instance
            names, origin and destination of every interaction, in that frame).

        Raises:
            KeyError: If ``interactions_file`` lacks one of ``expected_cols``.
            ValueError: If no interaction lies within the frame range.
        """
        origin = pd.read_csv(interactions_file, sep="\t")[list(expected_cols)]
        if limit_interactions:
            origin = origin[origin["Interaction Name"].isin(limit_interactions)]
        destination = origin.copy()

        # Origin and destination instances of each interaction, one row each
        origin = origin.drop(columns=["Destination interactor"]).rename(
            columns={"Origin interactor": "Instance"}
        )
        destination = destination.drop(columns=["Origin interactor"]).rename(
            columns={"Destination interactor": "Instance"}
        )

        instances = pd.concat([origin, destination], ignore_index=True)
        instances = instances[instances["Interaction Frame"].between(frame_start, frame_end)]

        if instances.shape[0] == 0:
            raise ValueError(f"No interactions found between frames {frame_start} to {frame_end}")

        return (
            instances.groupby("Interaction Frame")["Instance"]
            .apply(list)
            .reset_index(name="Instances")
        )

    @staticmethod
    def interpolate(coords_array: np.ndarray) -> np.ndarray:
        """Linearly interpolate gaps per track and coordinate, then blank implausible jumps.

        Interior gaps are filled linearly; trailing NaNs are held at the last valid value (flat
        extension, pandas' default ``limit_direction="forward"``) and leading NaNs stay NaN.
        Afterwards, values that differ from the previous frame by 5 or more (in the units of the
        coordinates, typically pixels, per frame; after interpolation) are set back to NaN. The
        array is modified in place and returned.

        Args:
            coords_array: ``(frames, nodes, 2, tracks)`` array.
        """
        for track in range(coords_array.shape[3]):
            for axis in range(2):
                coords_df = pd.DataFrame(coords_array[:, :, axis, track])
                coords_df = coords_df.interpolate(method="linear", axis=0)
                jumps = coords_df.diff().abs().ge(5)
                coords_array[:, :, axis, track] = coords_df.mask(jumps).to_numpy()
        return coords_array
