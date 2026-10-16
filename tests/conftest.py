"""Shared pytest fixtures: synthetic videos, markers, SLEAP/analysis files and interactor models.

Everything is built under ``tmp_path`` from a seeded RNG, so tests are deterministic, fast
and never write inside the repository or touch the network. The fixtures are additive:
individual test modules may keep their own helpers (a local fixture of the same name simply
takes precedence).

Fixtures
--------
``data_dir``               ``tests/data`` (the committed example files), as a ``Path``.
``rng``                    ``numpy.random.Generator`` with a fixed seed.
``marker_image``           factory ``(tag, size=40, border=8) -> image`` of a DICT_4X4_100 marker.
``colony``                 the default :class:`SyntheticColony` (3 static bees, tags 5/6/7).
``slp_factory``            factory writing a small tracked ``.slp`` of a colony.
``h5_factory``             factory writing a SLEAP-style analysis ``.h5``.
``video_factory``          factory writing a tiny MJPG video that shows the colony's markers.
``interactor_model_dict``  a valid ``Skeleton`` / ``Models`` / ``Interactions`` dict.
``interactor_model_yaml``  the same model written to a YAML file.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest
import sleap_io as sio

DATA_DIR = Path(__file__).resolve().parent / "data"

#: ArUco dictionary used by every synthetic scene (``--aruco-marker-set DICT_4X4_100``).
ARUCO_SET = "DICT_4X4_100"
#: Tag/position node names of the synthetic skeleton.
TAG_NODE = "Tag"
THORAX_NODE = "Thorax"


def render_marker(tag: int, size: int = 40, border: int = 8) -> np.ndarray:
    """Grayscale image of ArUco marker ``tag`` with a white quiet zone of ``border`` px."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    img = cv2.aruco.generateImageMarker(dictionary, tag, size)
    return cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, value=255)


@dataclass
class SyntheticColony:
    """A few static bees, each carrying one ArUco tag, seen for ``n_frames`` frames.

    Bee ``k`` is SLEAP track ``track_<k>`` (the ``track_<int>`` naming NAPS expects), sits at
    ``positions[k]`` (x, y in pixels) and wears tag ``tags[k]``. Its skeleton is ``Tag`` (on the
    marker) and ``Thorax`` (12 px below it).
    """

    tags: tuple[int, ...] = (5, 6, 7)
    n_frames: int = 20
    frame_size: int = 240
    positions: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.positions:
            # Well separated so each ArUco crop contains exactly one marker.
            grid = [(60.0, 60.0), (170.0, 60.0), (60.0, 170.0), (170.0, 170.0), (115.0, 115.0)]
            self.positions = grid[: len(self.tags)]

    @property
    def n_bees(self) -> int:
        return len(self.tags)

    def frame_image(self) -> np.ndarray:
        """BGR frame with every marker drawn (identical for every frame: the bees do not move)."""
        img = np.full((self.frame_size, self.frame_size, 3), 120, dtype=np.uint8)
        for tag, (x, y) in zip(self.tags, self.positions, strict=True):
            marker = render_marker(tag)
            half = marker.shape[0] // 2
            x0, y0 = int(x) - half, int(y) - half
            img[y0 : y0 + marker.shape[0], x0 : x0 + marker.shape[1]] = marker[:, :, None]
        return img


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Directory holding the committed example data files."""
    return DATA_DIR


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator (fresh for every test)."""
    return np.random.default_rng(20261008)


@pytest.fixture
def marker_image():
    """Factory ``(tag, size=40, border=8) -> image`` of an ArUco DICT_4X4_100 marker."""
    return render_marker


@pytest.fixture
def colony() -> SyntheticColony:
    """The default three-bee synthetic colony (tags 5, 6, 7; 20 frames)."""
    return SyntheticColony()


@pytest.fixture
def slp_factory():
    """Factory ``(path, colony, *, tracked=True, untracked=(), video_name=...) -> path``.

    Writes a ``.slp`` of the colony; ``untracked`` lists bee indices that get no SLEAP track.
    """

    def make(
        path: Path,
        colony: SyntheticColony | None = None,
        *,
        tracked: bool = True,
        untracked: Sequence[int] = (),
        video_name: str = "synthetic.mp4",
    ) -> Path:
        colony = colony or SyntheticColony()
        skeleton = sio.Skeleton([TAG_NODE, THORAX_NODE])
        video = sio.Video(filename=video_name, open_backend=False)
        tracks = [sio.Track(f"track_{k}") for k in range(colony.n_bees)]
        frames = []
        for frame_idx in range(colony.n_frames):
            instances = []
            for k, (x, y) in enumerate(colony.positions):
                points = np.array([[x, y], [x, y + 12.0]])
                instances.append(
                    sio.PredictedInstance.from_numpy(
                        points,
                        skeleton=skeleton,
                        score=1.0,
                        track=tracks[k] if tracked and k not in untracked else None,
                    )
                )
            frames.append(sio.LabeledFrame(video=video, frame_idx=frame_idx, instances=instances))
        sio.Labels(frames).save(str(path), verbose=False)
        return Path(path)

    return make


@pytest.fixture
def h5_factory():
    """Factory writing a SLEAP analysis ``.h5`` (``tracks`` as (tracks, 2, nodes, frames))."""

    def make(
        path: Path,
        coords: np.ndarray,
        track_names: list[str],
        node_names: list[str],
    ) -> Path:
        """``coords`` has the in-memory shape (frames, nodes, 2, tracks)."""
        with h5py.File(path, "w") as handle:
            handle.create_dataset("tracks", data=np.asarray(coords).T)
            handle.create_dataset("track_names", data=np.array([t.encode() for t in track_names]))
            handle.create_dataset("node_names", data=np.array([n.encode() for n in node_names]))
        return Path(path)

    return make


@pytest.fixture
def video_factory():
    """Factory ``(path, colony) -> path`` writing a real (MJPG) video of the colony's markers."""

    def make(path: Path, colony: SyntheticColony | None = None) -> Path:
        colony = colony or SyntheticColony()
        size = colony.frame_size
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 10.0, (size, size))
        if not writer.isOpened():  # pragma: no cover - environment without an MJPG encoder
            pytest.skip("OpenCV build cannot write MJPG video")
        frame = colony.frame_image()
        for _ in range(colony.n_frames):
            writer.write(frame)
        writer.release()
        return Path(path)

    return make


@pytest.fixture
def interactor_model_dict() -> dict:
    """A valid interactor model over the nodes Head, Thorax and Abdomen.

    ``head`` is a buffered point, ``body`` a buffered line; the ``Head_w_antennae`` model is
    the head disc, ``Body`` the body minus the head. ``head_to_body`` is non-symmetrical
    (reported in both directions), ``head_to_head`` symmetrical.
    """
    return {
        "Skeleton": [
            {"name": "head", "shape": "point", "nodes": ["Head"], "buffer": 5},
            {"name": "body", "shape": "line", "nodes": ["Head", "Thorax", "Abdomen"], "buffer": 3},
        ],
        "Models": [
            {"name": "Head_w_antennae", "includes": ["head"]},
            {"name": "Body", "includes": ["body"], "excludes": ["head"]},
        ],
        "Interactions": [
            {"name": "head_to_body", "origin": "Head_w_antennae", "dest": "Body"},
            {"name": "head_to_head", "origin": "Head_w_antennae", "dest": "Head_w_antennae",
             "symmetical": True},
        ],
    }  # fmt: skip


@pytest.fixture
def interactor_model_yaml(tmp_path: Path, interactor_model_dict: dict) -> Path:
    """:func:`interactor_model_dict` written to ``tmp_path/model.yaml``."""
    yaml = pytest.importorskip("yaml")  # PyYAML is only needed by the interactor tooling
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(interactor_model_dict))
    return path
