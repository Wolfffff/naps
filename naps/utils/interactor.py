#!/usr/bin/env python
"""Geometric interaction model of one tracked animal in one frame (shapely based)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from shapely import area
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import unary_union

#: Skeleton shapes accepted by :class:`Interactor`.
SKELETON_SHAPES = ("line", "reduced-line", "point", "hull")


class Interactor:
    """One animal at one frame: its skeleton shapes, interaction models and interaction rules.

    The geometry is described by a model dictionary (usually loaded from YAML) with three keys:

    * ``Skeleton``: shapes built from the node coordinates, each ``{name, shape, nodes, buffer,
      exclude_nodes}`` with ``shape`` one of :data:`SKELETON_SHAPES`. A non-empty
      ``exclude_nodes`` takes precedence over ``nodes``: the shape then uses every node of the
      animal except those listed (``nodes`` is ignored).
    * ``Models``: named regions, each ``{name, includes, excludes}`` combining skeleton shapes.
    * ``Interactions``: rules ``{name, origin, dest, symmetical, return_int}`` testing whether
      the ``origin`` model of one animal intersects the ``dest`` model of another.
      (``symmetical`` is the historical spelling of the YAML key and is kept. ``return_int``
      is accepted for compatibility and has no effect on the reported rows.)
    """

    def __init__(  # noqa: PLR0917 (public API: positional order kept)
        self,
        name: str = "",
        frame: int = -1,
        nodes: Sequence[str] | None = None,
        coords_array: np.ndarray | None = None,
        skeleton: Sequence[dict] | None = None,
        model: Sequence[dict] | None = None,
        interactions: Sequence[dict] | None = None,
        **kwargs: object,  # extra keys of a model dict are accepted and ignored
    ) -> None:
        """Build the skeleton shapes and models of one animal.

        Args:
            name: Name of the animal (its track name).
            frame: Frame the coordinates belong to.
            nodes: Node names, in the order of the rows of ``coords_array``.
            coords_array: ``(nodes, 2)`` array of x, y coordinates.
            skeleton: Skeleton shape specifications (``Skeleton`` of the model dict).
            model: Model specifications (``Models`` of the model dict).
            interactions: Interaction rules (``Interactions`` of the model dict).
            **kwargs: Ignored.
        """
        self.name = name
        self.frame = frame

        # The node names are required to map the skeleton nodes onto coords_array rows
        self._nodes = list(nodes) if nodes is not None else []

        self._skeleton: dict[str, Any] = {}
        self._models: dict[str, Any] = {}
        self._interactions = interactions if interactions is not None else []

        coords = coords_array if coords_array is not None else np.array([])

        # Build the skeleton shapes from the coordinates, then the models from the shapes
        for spec in skeleton or []:
            self._update_skeleton(coords, **spec)
        for spec in model or []:
            self._update_model(**spec)

    def _update_skeleton(
        self,
        coords_array: np.ndarray,
        *,
        name: str = "",
        shape: str = "",
        nodes: Sequence[str] | None = None,
        buffer: float = 0,
        exclude_nodes: Sequence[str] | None = None,
    ) -> None:
        """Add one skeleton shape (validated) to ``self._skeleton``.

        A non-empty ``exclude_nodes`` overrides ``nodes`` (all other nodes are used).
        """
        nodes = list(nodes) if nodes is not None else []
        if exclude_nodes:
            nodes = [n for n in self._nodes if n not in exclude_nodes]

        if not name:
            raise ValueError(
                f"No name given for skeleton with shape ({shape}) and nodes ({', '.join(nodes)})"
            )
        if shape not in SKELETON_SHAPES:
            raise ValueError(f"The shape ({shape}) is not supported")

        if shape in ("line", "reduced-line", "hull") and len(nodes) <= 1:
            raise ValueError(
                f"The shape ({shape}) requires two or more nodes, found {len(nodes)} nodes."
            )
        if shape == "point" and len(nodes) != 1:
            raise ValueError(
                f"The shape ({shape}) requires a single node, found {len(nodes)} nodes."
            )

        node_coords: list[Any] = [coords_array[self._nodes.index(n)] for n in nodes]

        if shape == "line":
            self._skeleton[name] = LineString(node_coords).buffer(buffer)
        elif shape == "reduced-line":
            # Shorten the line by ``buffer`` at both ends (first and last segment)
            first_segment = LineString(node_coords[:2])
            node_coords[0] = first_segment.interpolate(buffer)
            if len(node_coords) == 2:
                node_coords[-1] = first_segment.interpolate(first_segment.length - buffer)
            else:
                last_segment = LineString(node_coords[-2:])
                node_coords[-1] = last_segment.interpolate(last_segment.length - buffer)
            self._skeleton[name] = LineString(node_coords).buffer(buffer)
        elif shape == "hull":
            self._skeleton[name] = MultiPoint(node_coords).convex_hull.buffer(buffer)
        else:  # point
            self._skeleton[name] = Point(node_coords[0]).buffer(buffer)

    def _update_model(
        self,
        name: str = "",
        includes: Sequence[str] = (),
        excludes: Sequence[str] = (),
    ) -> None:
        """Add one model (union of ``includes`` minus union of ``excludes``) to ``self._models``."""
        if not name:
            raise ValueError(
                f"No name given for interaction model with includes ({', '.join(includes)}) "
                f"and excludes ({', '.join(excludes)})"
            )

        interaction = unary_union([self._skeleton[inc] for inc in includes])
        if len(excludes):
            exclusion = unary_union([self._skeleton[exc] for exc in excludes])
            interaction = interaction.difference(exclusion)
        self._models[name] = interaction

    def interacts(self, other_interactor: Interactor) -> list[dict]:
        """Evaluate every interaction rule of this animal against ``other_interactor``.

        Returns:
            One row per detected interaction, with the keys ``Origin interactor``,
            ``Destination interactor``, ``Interaction Name``, ``Interaction Frame``,
            ``Origin Area`` and ``Destination Area``. A non-symmetrical rule is tested in both
            directions and can yield two rows.
        """
        interaction_list: list[dict] = []
        for interaction in self._interactions:
            interaction_list.extend(self._evaluate_rule(other_interactor, **interaction))
        return interaction_list

    def _report(
        self,
        name: str,
        origin: str,
        dest: str,
        origin_interactor: Interactor,
        dest_interactor: Interactor,
    ) -> dict:
        """One reported interaction row."""
        return {
            "Origin interactor": origin_interactor.name,
            "Destination interactor": dest_interactor.name,
            "Interaction Name": name,
            "Interaction Frame": self.frame,
            "Origin Area": area(origin_interactor._models[origin]),
            "Destination Area": area(dest_interactor._models[dest]),
        }

    def _evaluate_rule(
        self,
        other_interactor: Interactor,
        *,
        name: str = "",
        origin: str = "",
        dest: str = "",
        symmetical: bool = False,
        return_int: bool = False,  # accepted for compatibility, no effect on the rows
    ) -> list[dict]:
        """Validate one rule and return its interaction rows (empty if nothing intersects)."""
        if not name:
            raise ValueError(
                f"No name given for interaction with origin ({origin}) and dest ({dest})"
            )
        if not origin:
            raise ValueError(f"No origin given for interaction ({name})")
        if not dest:
            raise ValueError(f"No dest given for interaction ({name})")
        if origin not in self._models:
            raise ValueError(f"Given origin ({origin}) not found among interaction models")
        if dest not in self._models:
            raise ValueError(f"Given dest ({dest}) not found among interaction models")
        if not isinstance(symmetical, bool):
            raise TypeError("symmetical must be boolean")

        def intersects(origin_interactor: Interactor, dest_interactor: Interactor) -> bool:
            return bool(origin_interactor._models[origin].intersects(dest_interactor._models[dest]))

        if symmetical:
            if not intersects(self, other_interactor):
                return []
            return [self._report(name, origin, dest, self, other_interactor)]

        rows = []
        if intersects(self, other_interactor):
            rows.append(self._report(name, origin, dest, self, other_interactor))
        # The inverse direction: the other animal's origin model against this animal's dest model
        if intersects(other_interactor, self):
            rows.append(self._report(name, origin, dest, other_interactor, self))
        return rows

    @classmethod
    def withModelDict(  # noqa: N802 (public camelCase API)
        cls,
        name: str,
        frame: int,
        nodes: Sequence[str],
        coords_array: np.ndarray,
        model_dict: dict,
        **kwargs: object,
    ) -> Interactor:
        """Build an :class:`Interactor` from a model dictionary.

        Args:
            name: Name of the animal (its track name).
            frame: Frame the coordinates belong to.
            nodes: Node names, in the order of the rows of ``coords_array``.
            coords_array: ``(nodes, 2)`` array of x, y coordinates.
            model_dict: Dictionary with the keys ``Skeleton``, ``Models`` and ``Interactions``.
            **kwargs: Passed to the constructor.
        """
        return cls(
            name=name,
            frame=frame,
            nodes=nodes,
            coords_array=coords_array,
            skeleton=model_dict["Skeleton"],
            model=model_dict["Models"],
            interactions=model_dict["Interactions"],
            **kwargs,
        )
