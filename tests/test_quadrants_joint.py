"""``region_index`` and ``QuadrantScope(min_size=...)``, the quadrant pieces of the joint engine."""

import numpy as np

from naps.quadrants import QuadrantScope, locate, make_grid, region_index

NAMES = ["top_left", "top_right", "bottom_left", "bottom_right"]
ORIGIN = {
    "top_left": (0, 0),
    "top_right": (1832, 0),
    "bottom_left": (0, 1832),
    "bottom_right": (1832, 1832),
}
SIZE = (3664, 3664)


def track(q, k):
    return 10 * NAMES.index(q) + k


def positions(n=60):
    out = {}
    for f in range(n):
        out[f] = {}
        for q in NAMES:
            ox, oy = ORIGIN[q]
            for k in range(4):
                out[f][track(q, k)] = np.array(
                    [ox + 300 + 600 * (k % 2), oy + 300 + 600 * (k // 2)]
                )
    return out


def decodes_with(sizes, n=60):
    """Decodes where quadrant ``q`` has ``sizes[q]`` well-supported tags (tag = 10*quadrant+k)."""
    return {
        f: {
            track(q, k): [10 * NAMES.index(q) + k] if k < sizes[q] else [None]
            for q in NAMES
            for k in range(4)
        }
        for f in range(n)
    }


def infer(decodes, **kw):
    scope = QuadrantScope(infer=True, min_support=10, **kw)
    scope.prepare(SIZE, positions(len(decodes)))
    scope.infer_from_decodes(decodes)
    return scope


# ------------------------------------------------------------------------ region_index
def test_region_index_matches_the_grid():
    rois = make_grid(*SIZE)
    xy = np.array(
        [
            [[100.0, 100.0], [3000.0, 100.0]],
            [[100.0, 3000.0], [3000.0, 3000.0]],
        ]
    )
    names, index = region_index(xy, rois)
    assert names == NAMES
    assert index.shape == (2, 2)
    assert [names[i] for i in index.ravel()] == NAMES


def test_region_index_outer_edges_are_open():
    names, index = region_index(np.array([[-50.0, -50.0], [4000.0, 4000.0]]), make_grid(*SIZE))
    assert [names[i] for i in index] == ["top_left", "bottom_right"]


def test_region_index_is_minus_one_for_nan_inf_outside_and_overlap():
    rois = {"a": (0.0, 0.0, 100.0, 100.0), "b": (50.0, 50.0, 200.0, 200.0)}
    xy = np.array(
        [
            [10.0, 10.0],  # a only
            [150.0, 150.0],  # b only
            [75.0, 75.0],  # both: ambiguous
            [500.0, 500.0],  # neither
            [np.nan, 10.0],
            [10.0, np.inf],
        ]
    )
    names, index = region_index(xy, rois)
    assert names == ["a", "b"]
    assert index.tolist() == [0, 1, -1, -1, -1, -1]


def test_region_index_agrees_with_locate_on_random_points():
    rng = np.random.default_rng(3)
    rois = make_grid(*SIZE)
    xy = rng.uniform(-100, 3800, size=(200, 2))
    names, index = region_index(xy, rois)
    for point, i in zip(xy, index, strict=True):
        assert locate(point, rois) == (None if i < 0 else names[i])


# ------------------------------------------------------------------------- min_size
def test_default_requires_exactly_the_expected_number_of_tags():
    scope = infer(decodes_with(dict.fromkeys(NAMES, 3)))
    assert scope.active == {}
    assert scope.inference["abstained"]["top_left"].startswith("fewer_than_4_tags (found 3")


def test_min_size_accepts_a_quadrant_of_three_tags():
    sizes = {"top_left": 3, "top_right": 4, "bottom_left": 3, "bottom_right": 4}
    scope = infer(decodes_with(sizes), min_size=3)
    assert set(scope.active) == set(NAMES)
    assert len(scope.active["top_left"]) == 3
    assert len(scope.active["top_right"]) == 4
    assert not scope.inference["abstained"]


def test_min_size_still_abstains_below_the_minimum():
    sizes = {"top_left": 2, "top_right": 4, "bottom_left": 3, "bottom_right": 4}
    scope = infer(decodes_with(sizes), min_size=3)
    assert "top_left" not in scope.active
    assert scope.inference["abstained"]["top_left"].startswith("fewer_than_3_tags (found 2")
    assert set(scope.active) == {"top_right", "bottom_left", "bottom_right"}


def test_min_size_still_abstains_above_the_expected_size_without_a_dominant_group():
    d = decodes_with(dict.fromkeys(NAMES, 4))
    for f in d:  # a fifth tag, equally supported
        d[f][track("top_left", 0)].append(99)
    scope = infer(d, min_size=3)
    assert "top_left" in scope.inference["abstained"]
    assert scope.inference["abstained"]["top_left"].startswith("more_than_4_tags")


def test_min_size_keeps_the_dominant_four_among_five():
    d = decodes_with(dict.fromkeys(NAMES, 4))
    for f in range(12):  # a fifth, much weaker tag
        d[f][track("top_left", 0)].append(99)
    scope = infer(d, min_size=3, dominance_ratio=3.0)
    assert scope.active["top_left"] == [0, 1, 2, 3]
