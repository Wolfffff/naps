import numpy as np

from naps.sleap_utils import load_tracks_from_slp
from tests.conftest import DATA_DIR


def test_load_tracks_from_slp():
    locations, node_names = load_tracks_from_slp(str(DATA_DIR / "example.slp"))
    assert isinstance(locations, np.ndarray)
    assert locations.ndim == 4  # (frames, nodes, 2, tracks)
    assert locations.shape[1] == len(node_names)
    assert locations.shape[2] == 2
