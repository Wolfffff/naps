"""Geometry and validation of :class:`naps.utils.interactor.Interactor` (synthetic coordinates)."""

import copy

import numpy as np
import pytest

pytest.importorskip("shapely")

from naps.utils.interactor import Interactor

NODES = ["Head", "Thorax", "Abdomen"]


def bee(model, name, head, thorax, abdomen, *, frame=7):
    """An Interactor with the given node coordinates (rows follow NODES)."""
    coords = np.array([head, thorax, abdomen], dtype=float)
    return Interactor.withModelDict(name, frame, NODES, coords, model)


def test_no_interaction_when_far_apart(interactor_model_dict):
    a = bee(interactor_model_dict, "A", (0, 0), (20, 0), (40, 0))
    b = bee(interactor_model_dict, "B", (0, 200), (20, 200), (40, 200))
    assert a.interacts(b) == []


def test_head_touching_the_other_body_is_one_directed_row(interactor_model_dict):
    a = bee(interactor_model_dict, "A", (0, 0), (20, 0), (40, 0))
    # B's body crosses A's head; B's head is far from A's body and head
    b = bee(interactor_model_dict, "B", (0, 30), (0, 10), (0, -10))
    rows = a.interacts(b)
    assert len(rows) == 1
    row = rows[0]
    assert row["Origin interactor"] == "A"
    assert row["Destination interactor"] == "B"
    assert row["Interaction Name"] == "head_to_body"
    assert row["Interaction Frame"] == 7
    assert set(row) == {
        "Origin interactor", "Destination interactor", "Interaction Name",
        "Interaction Frame", "Origin Area", "Destination Area",
    }  # fmt: skip
    # head = a disc of radius 5 (a 64-gon: slightly below pi r^2)
    assert row["Origin Area"] == pytest.approx(np.pi * 25, rel=0.01)
    assert row["Destination Area"] > 0


def test_non_symmetrical_rule_reports_both_directions(interactor_model_dict):
    a = bee(interactor_model_dict, "A", (0, 0), (20, 0), (40, 0))
    b = bee(interactor_model_dict, "B", (30, 6), (0, 6), (-30, 6))
    rows = [r for r in a.interacts(b) if r["Interaction Name"] == "head_to_body"]
    assert [(r["Origin interactor"], r["Destination interactor"]) for r in rows] == [
        ("A", "B"),
        ("B", "A"),
    ]
    # the frame of both rows is the frame of the interactor that evaluates the rule
    assert {r["Interaction Frame"] for r in rows} == {7}


def test_symmetrical_rule_reports_a_single_row(interactor_model_dict):
    a = bee(interactor_model_dict, "A", (0, 0), (20, 0), (40, 0))
    b = bee(interactor_model_dict, "B", (6, 0), (26, 0), (46, 0))
    rows = [r for r in a.interacts(b) if r["Interaction Name"] == "head_to_head"]
    assert len(rows) == 1
    assert (rows[0]["Origin interactor"], rows[0]["Destination interactor"]) == ("A", "B")


@pytest.mark.parametrize(
    ("shape", "nodes", "buffer"),
    [
        ("line", ["Head", "Thorax"], 2),
        ("reduced-line", ["Head", "Thorax"], 2),
        ("reduced-line", ["Head", "Thorax", "Abdomen"], 2),
        ("hull", ["Head", "Thorax", "Abdomen"], 1),
        ("point", ["Thorax"], 4),
    ],
)
def test_every_skeleton_shape_builds_a_polygon(shape, nodes, buffer):
    model = {
        "Skeleton": [{"name": "s", "shape": shape, "nodes": nodes, "buffer": buffer}],
        "Models": [{"name": "m", "includes": ["s"]}],
        "Interactions": [],
    }
    a = bee(model, "A", (0, 0), (20, 0), (40, 10))
    assert a._models["m"].area > 0
    assert a.interacts(a) == []  # no rules, no rows


def test_exclude_nodes_use_every_other_node(interactor_model_dict):
    model = copy.deepcopy(interactor_model_dict)
    model["Skeleton"] = [
        {
            "name": "front",
            "shape": "line",
            "buffer": 0.5,
            "nodes": ["Head"],
            "exclude_nodes": ["Abdomen"],
        },
    ]
    model["Models"] = [{"name": "Head_w_antennae", "includes": ["front"]}]
    model["Interactions"] = []
    a = bee(model, "A", (0, 0), (20, 0), (400, 0))
    # nodes = Head, Thorax (the Abdomen at x=400 is excluded)
    assert a._models["Head_w_antennae"].bounds[2] == pytest.approx(20.5)


def test_models_subtract_their_excludes(interactor_model_dict):
    a = bee(interactor_model_dict, "A", (0, 0), (20, 0), (40, 0))
    assert not a._models["Body"].intersects(a._skeleton["head"].buffer(-0.1))


def test_extra_constructor_keys_and_return_int_are_accepted(interactor_model_dict):
    model = copy.deepcopy(interactor_model_dict)
    for rule in model["Interactions"]:
        rule["return_int"] = True
    coords = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    a = Interactor.withModelDict("A", 1, NODES, coords, model, unexpected_extra="ignored")
    b = Interactor.withModelDict("B", 1, NODES, coords + np.array([2.0, 0.0]), model)
    assert "head_to_head" in {r["Interaction Name"] for r in a.interacts(b)}


def test_default_interactor_is_empty():
    empty = Interactor()
    assert (empty.name, empty.frame) == ("", -1)
    assert empty.interacts(Interactor()) == []


def _model(skeleton=None, models=None, interactions=None):
    return {
        "Skeleton": skeleton
        if skeleton is not None
        else [{"name": "head", "shape": "point", "nodes": ["Head"], "buffer": 5}],
        "Models": models if models is not None else [{"name": "m", "includes": ["head"]}],
        "Interactions": interactions
        if interactions is not None
        else [{"name": "rule", "origin": "m", "dest": "m"}],
    }


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (_model(skeleton=[{"shape": "point", "nodes": ["Head"]}]), "No name given for skeleton"),
        (
            _model(skeleton=[{"name": "s", "shape": "blob", "nodes": ["Head"]}]),
            r"The shape \(blob\) is not supported",
        ),
        (
            _model(skeleton=[{"name": "s", "shape": "line", "nodes": ["Head"]}]),
            "requires two or more nodes, found 1 nodes",
        ),
        (
            _model(skeleton=[{"name": "s", "shape": "reduced-line", "nodes": []}]),
            "requires two or more nodes, found 0 nodes",
        ),
        (
            _model(skeleton=[{"name": "s", "shape": "hull", "nodes": ["Head"]}]),
            "requires two or more nodes, found 1 nodes",
        ),
        (
            _model(skeleton=[{"name": "s", "shape": "point", "nodes": ["Head", "Thorax"]}]),
            "requires a single node, found 2 nodes",
        ),
        (_model(models=[{"includes": ["head"]}]), "No name given for interaction model"),
    ],
)
def test_invalid_model_definitions_are_rejected(model, message):
    with pytest.raises(ValueError, match=message):
        bee(model, "A", (0, 0), (1, 0), (2, 0))


@pytest.mark.parametrize(
    ("rule", "exception", "message"),
    [
        ({"origin": "m", "dest": "m"}, ValueError, "No name given for interaction"),
        ({"name": "r", "dest": "m"}, ValueError, "No origin given"),
        ({"name": "r", "origin": "m"}, ValueError, "No dest given"),
        ({"name": "r", "origin": "nope", "dest": "m"}, ValueError, r"origin \(nope\) not found"),
        ({"name": "r", "origin": "m", "dest": "nope"}, ValueError, r"dest \(nope\) not found"),
        ({"name": "r", "origin": "m", "dest": "m", "symmetical": 1}, TypeError, "must be boolean"),
        ({"name": "r", "origin": "m", "dest": "m", "mystery": 1}, TypeError, "mystery"),
    ],
)
def test_invalid_interaction_rules_are_rejected(rule, exception, message):
    model = _model(interactions=[rule])
    a = bee(model, "A", (0, 0), (1, 0), (2, 0))
    b = bee(model, "B", (0, 0), (1, 0), (2, 0))
    with pytest.raises(exception, match=message):
        a.interacts(b)
