r"""Graph view of a tetrahedral mesh and its nonlinear diffusion problem.

This mirrors :mod:`neuralmesh.mesh.graph` but for 3D and for the nonlinear operator,
and it reuses :class:`~neuralmesh.mesh.graph.MeshGraph` unchanged: that container never
assumed two dimensions, so the models, the trainer and ``graph_diameter`` all work here
without modification.

One decision here is load-bearing and easy to get wrong.

**The conductivity feature must not be** :math:`k(u)`. In the linear 2D problem the
conductivity is data, so handing it to the network is fair. Here :math:`k = k_0(1 +
\alpha u^2)` depends on the solution, so a node feature carrying :math:`k(u)` would carry
:math:`u` itself, and the network could invert it almost exactly. Every metric would
improve and the benchmark would be measuring nothing. The features are therefore
:math:`k_0` and :math:`\alpha`, which are the actual inputs a designer specifies, and
:func:`leakage_report` exists so the claim can be checked rather than asserted.

Node features, in order:

0. boundary indicator, 1 on a constrained node
1. boundary value, zero away from the boundary
2. source term
3. :math:`k_0`, the linear part of the conductivity
4. :math:`\alpha`, the strength of the nonlinearity

Edge features are the displacement and its length, ``[dx, dy, dz, |d|]``.
"""

from __future__ import annotations

import numpy as np

from ..fem.mesh3d import TetMesh
from .graph import MeshGraph

NODE_FEATURES_3D = ("is_boundary", "boundary_value", "source", "k0", "alpha")
EDGE_FEATURES_3D = ("dx", "dy", "dz", "length")

NODE_DIM_3D = len(NODE_FEATURES_3D)
EDGE_DIM_3D = len(EDGE_FEATURES_3D)


def build_graph_3d(
    mesh: TetMesh,
    *,
    source: np.ndarray,
    k0: float = 1.0,
    alpha: float = 1.0,
    dirichlet_nodes: np.ndarray | None = None,
    dirichlet_values: np.ndarray | float = 0.0,
    target: np.ndarray | None = None,
) -> MeshGraph:
    """Assemble a :class:`MeshGraph` from a tetrahedral mesh and its problem data."""
    if dirichlet_nodes is None:
        dirichlet_nodes = mesh.boundary_nodes
    dirichlet_nodes = np.asarray(dirichlet_nodes, dtype=np.int64)

    is_boundary = np.zeros(mesh.n_nodes, dtype=np.float64)
    is_boundary[dirichlet_nodes] = 1.0

    # Same three accepted shapes as the 2D builder, for the same reason: the singular and
    # plural parameter names differ by one character across the codebase.
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

    node_features = np.column_stack(
        [
            is_boundary,
            bvals,
            src,
            np.full(mesh.n_nodes, float(k0)),
            np.full(mesh.n_nodes, float(alpha)),
        ]
    )

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


def graph_from_solution_3d(solution) -> MeshGraph:
    """Build a supervised graph from a :class:`~neuralmesh.fem.nonlinear3d.Solution3D`."""
    return build_graph_3d(
        solution.mesh,
        source=solution.f_node,
        k0=solution.law.k0,
        alpha=solution.law.alpha,
        dirichlet_nodes=solution.dirichlet_nodes,
        dirichlet_values=solution.dirichlet_values,
        target=solution.u,
    )


def leakage_report(graph: MeshGraph) -> dict[str, float]:
    r"""How much of the target each input feature explains on its own.

    Returns the squared Pearson correlation between every node feature and the target,
    over interior nodes only. Boundary nodes are excluded because their target *is* their
    boundary-value feature by construction, which is not leakage, it is the boundary
    condition.

    The value to watch is ``conductivity_like``, the largest score among the material
    features. If :math:`k(u)` were being passed in instead of :math:`k_0` and
    :math:`\alpha`, that score would be close to 1 and the benchmark would be measuring
    the network's ability to invert a monotone function rather than to solve a PDE.
    """
    if graph.target is None:
        raise ValueError("leakage can only be assessed on a supervised graph")

    interior = graph.node_features[:, 0] == 0.0
    if interior.sum() < 3:
        raise ValueError("not enough interior nodes to assess leakage")

    y = graph.target[interior]
    out: dict[str, float] = {}
    for i, name in enumerate(NODE_FEATURES_3D):
        x = graph.node_features[interior, i]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            out[name] = 0.0
        else:
            out[name] = float(np.corrcoef(x, y)[0, 1] ** 2)
    out["conductivity_like"] = max(out["k0"], out["alpha"])
    return out
