"""Converting a finite element mesh into a graph a network can consume.

Design decisions that matter, because they determine what the model is able to learn:

*Edges carry relative geometry, not absolute.* Edge features are the displacement
vector between endpoints and its length. Nothing in the message-passing path sees an
absolute coordinate, so the learned operator is translation invariant by
construction rather than by hoping the data covers every position.

*Boundary data is a node feature, not a post-hoc mask.* The Dirichlet value lives on
the node alongside a boundary indicator. A model that has to propagate boundary
information inward is exactly the model that reveals under-reaching, which is the
phenomenon this repository is built to measure.

*Conductivity is averaged from cells to nodes.* The FEM solver holds conductivity per
cell, but message passing runs on nodes, so cell values are area-weighted onto
vertices. This loses the sharp jump at a material interface, and that lossiness is
deliberate and documented rather than hidden: it is part of why the surrogate cannot
be perfect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import TriMesh

NODE_FEATURES = ("is_boundary", "boundary_value", "source", "conductivity")
EDGE_FEATURES = ("dx", "dy", "length")


@dataclass
class MeshGraph:
    """Graph view of a mesh, with features ready for a network.

    Attributes
    ----------
    node_features:
        ``(n_nodes, 4)`` array of ``NODE_FEATURES``.
    edge_index:
        ``(2, n_directed_edges)`` source and destination indices. Every mesh edge
        appears in both directions so message passing is symmetric.
    edge_features:
        ``(n_directed_edges, 3)`` array of ``EDGE_FEATURES``.
    positions:
        ``(n_nodes, 2)`` coordinates, kept separate from node features so attention
        blocks can use them for positional encoding without leaking absolute
        position into the message-passing path.
    target:
        ``(n_nodes,)`` ground-truth field, or ``None`` for inference-only graphs.
    """

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    positions: np.ndarray
    target: np.ndarray | None = None

    @property
    def n_nodes(self) -> int:
        return len(self.node_features)

    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[1]

    def __post_init__(self) -> None:
        if self.edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be (2, E), got {self.edge_index.shape}")
        if len(self.edge_features) != self.n_edges:
            raise ValueError("edge_features and edge_index disagree on edge count")
        if len(self.positions) != self.n_nodes:
            raise ValueError("positions and node_features disagree on node count")
        if self.target is not None and len(self.target) != self.n_nodes:
            raise ValueError("target and node_features disagree on node count")

    def summary(self) -> str:
        return (
            f"MeshGraph(nodes={self.n_nodes}, directed_edges={self.n_edges}, "
            f"node_dim={self.node_features.shape[1]}, "
            f"edge_dim={self.edge_features.shape[1]})"
        )


def cell_to_node(mesh: TriMesh, cell_values: np.ndarray) -> np.ndarray:
    """Area-weighted average of a per-cell field onto nodes."""
    vals = np.broadcast_to(np.asarray(cell_values, dtype=np.float64), (mesh.n_cells,))
    areas = mesh.cell_areas()
    num = np.zeros(mesh.n_nodes, dtype=np.float64)
    den = np.zeros(mesh.n_nodes, dtype=np.float64)
    for corner in range(3):
        idx = mesh.triangles[:, corner]
        np.add.at(num, idx, vals * areas)
        np.add.at(den, idx, areas)
    # A node always belongs to at least one cell in a valid mesh, but guard anyway.
    den[den == 0.0] = 1.0
    return num / den


def build_graph(
    mesh: TriMesh,
    *,
    source: np.ndarray,
    conductivity: np.ndarray | float = 1.0,
    dirichlet_nodes: np.ndarray | None = None,
    dirichlet_values: np.ndarray | float = 0.0,
    target: np.ndarray | None = None,
) -> MeshGraph:
    """Assemble a :class:`MeshGraph` from a mesh and its problem data."""
    if dirichlet_nodes is None:
        dirichlet_nodes = mesh.boundary_nodes
    dirichlet_nodes = np.asarray(dirichlet_nodes, dtype=np.int64)

    is_boundary = np.zeros(mesh.n_nodes, dtype=np.float64)
    is_boundary[dirichlet_nodes] = 1.0

    # Accept boundary data as a scalar, one value per constrained node, or a field over
    # every node. solve_poisson takes the whole-field form under the near-identical name
    # dirichlet_value, and silently reinterpreting one as the other is a bug that has
    # already happened once in this codebase, so both are handled explicitly here.
    bvals = np.zeros(mesh.n_nodes, dtype=np.float64)
    dv = np.asarray(dirichlet_values, dtype=np.float64)
    if dv.ndim == 0:
        bvals[dirichlet_nodes] = float(dv)
    elif dv.shape == (mesh.n_nodes,):
        bvals[dirichlet_nodes] = dv[dirichlet_nodes]
    elif dv.shape == dirichlet_nodes.shape:
        bvals[dirichlet_nodes] = dv
    else:
        raise ValueError(
            f"dirichlet_values must be a scalar, ({len(dirichlet_nodes)},) over the "
            f"constrained nodes, or ({mesh.n_nodes},) over every node; got {dv.shape}"
        )

    src = np.asarray(source, dtype=np.float64)
    if src.shape != (mesh.n_nodes,):
        raise ValueError(f"source must be ({mesh.n_nodes},), got {src.shape}")

    k_node = cell_to_node(mesh, conductivity)

    node_features = np.column_stack([is_boundary, bvals, src, k_node])

    undirected = mesh.edges()
    both = np.vstack([undirected, undirected[:, ::-1]])
    edge_index = both.T.astype(np.int64)

    d = mesh.points[edge_index[1]] - mesh.points[edge_index[0]]
    length = np.linalg.norm(d, axis=1, keepdims=True)
    edge_features = np.hstack([d, length])

    return MeshGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        positions=mesh.points.copy(),
        target=None if target is None else np.asarray(target, dtype=np.float64),
    )


def graph_from_solution(solution) -> MeshGraph:
    """Build a supervised graph directly from a :class:`PoissonSolution`."""
    return build_graph(
        solution.mesh,
        source=solution.f_node,
        conductivity=solution.k_cell,
        dirichlet_nodes=solution.dirichlet_nodes,
        dirichlet_values=solution.dirichlet_values,
        target=solution.u,
    )


def graph_diameter(graph: MeshGraph, *, max_nodes: int = 4000) -> int:
    """Unweighted graph diameter by repeated breadth-first search.

    This is the number that determines how many message-passing steps a purely
    local architecture needs before information from one side of the domain can
    influence the other. Reported by the under-reaching benchmark.

    For meshes above ``max_nodes`` the exact diameter is expensive, so a
    double-sweep lower bound is returned instead: BFS from an arbitrary node, then
    BFS from the furthest node found. That is exact on trees and tight in practice
    on mesh graphs.
    """
    n = graph.n_nodes
    adj = _adjacency_list(graph)

    if n <= max_nodes:
        return max(_bfs_depth(adj, s) for s in range(n))

    far = _bfs_argmax(adj, 0)
    return _bfs_depth(adj, far)


def _adjacency_list(graph: MeshGraph) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(graph.n_nodes)]
    src, dst = graph.edge_index
    for a, b in zip(src.tolist(), dst.tolist()):
        adj[a].append(b)
    return adj


def _bfs_dist(adj: list[list[int]], start: int) -> np.ndarray:
    dist = np.full(len(adj), -1, dtype=np.int64)
    dist[start] = 0
    frontier = [start]
    while frontier:
        nxt: list[int] = []
        for u in frontier:
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return dist


def _bfs_depth(adj: list[list[int]], start: int) -> int:
    return int(_bfs_dist(adj, start).max())


def _bfs_argmax(adj: list[list[int]], start: int) -> int:
    return int(np.argmax(_bfs_dist(adj, start)))
