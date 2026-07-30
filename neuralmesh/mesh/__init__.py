"""Mesh generation and the mesh-to-graph conversion."""

from .geometry import (
    TriMesh,
    annulus_mesh,
    boundary_nodes_of,
    rectangle_mesh,
    refine,
    strip_mesh,
    unit_square_mesh,
)
from .graph import (
    EDGE_FEATURES,
    NODE_FEATURES,
    MeshGraph,
    build_graph,
    cell_to_node,
    graph_diameter,
    graph_from_solution,
)

__all__ = [
    "EDGE_FEATURES",
    "NODE_FEATURES",
    "MeshGraph",
    "TriMesh",
    "annulus_mesh",
    "boundary_nodes_of",
    "build_graph",
    "cell_to_node",
    "graph_diameter",
    "graph_from_solution",
    "rectangle_mesh",
    "refine",
    "strip_mesh",
    "unit_square_mesh",
]
