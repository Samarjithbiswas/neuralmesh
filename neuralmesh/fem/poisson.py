r"""P1 finite element solver for the steady diffusion equation.

Solves

.. math::
    -\nabla\cdot(k\,\nabla u) = f \quad\text{in }\Omega,
    \qquad u = g \quad\text{on }\partial\Omega

on an unstructured triangular mesh with linear (P1) Lagrange elements.

This is the ground-truth generator for the learned surrogates. It is deliberately
written from the weak form rather than wrapped around a library, both so the element
matrices are inspectable and so :mod:`neuralmesh.fem.verify` can check the
discretisation against exact solutions and recover the expected convergence rate.

Weak form
---------
Multiplying by a test function :math:`v` vanishing on the Dirichlet boundary and
integrating by parts gives

.. math::
    \int_\Omega k\,\nabla u\cdot\nabla v \,\mathrm{d}\Omega
    = \int_\Omega f\,v\,\mathrm{d}\Omega .

On a P1 triangle the shape-function gradients are constant, so with area :math:`A`
and gradient coefficient vectors :math:`\mathbf{b}, \mathbf{c}` the element
stiffness is

.. math::
    \mathbf{K}_e = k A\,(\mathbf{b}\mathbf{b}^{\mathsf T}
                        + \mathbf{c}\mathbf{c}^{\mathsf T}),

and lumping the source over the three vertices gives
:math:`\mathbf{f}_e = f A / 3`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..mesh.geometry import TriMesh

ScalarField = Callable[[np.ndarray], np.ndarray]


@dataclass
class PoissonSolution:
    """Result of a diffusion solve."""

    mesh: TriMesh
    u: np.ndarray
    k_cell: np.ndarray
    f_node: np.ndarray
    dirichlet_nodes: np.ndarray
    dirichlet_values: np.ndarray

    def l2_error(self, exact: ScalarField) -> float:
        """Mass-matrix-weighted :math:`L^2` error against an exact solution.

        Using the consistent mass matrix rather than a plain vector norm makes the
        number mesh-independent, which is what lets the convergence test measure a
        rate instead of a coincidence.
        """
        u_ex = np.asarray(exact(self.mesh.points), dtype=np.float64)
        err = self.u - u_ex
        mass = assemble_mass(self.mesh)
        return float(np.sqrt(max(err @ (mass @ err), 0.0)))

    def max_error(self, exact: ScalarField) -> float:
        u_ex = np.asarray(exact(self.mesh.points), dtype=np.float64)
        return float(np.abs(self.u - u_ex).max())


def shape_gradients(mesh: TriMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell P1 shape-function gradients and areas.

    Returns ``(b, c, area)`` where ``b[i]`` and ``c[i]`` are length-3 vectors of
    :math:`\\partial N/\\partial x` and :math:`\\partial N/\\partial y` for cell ``i``.
    """
    p = mesh.points[mesh.triangles]
    x1, y1 = p[:, 0, 0], p[:, 0, 1]
    x2, y2 = p[:, 1, 0], p[:, 1, 1]
    x3, y3 = p[:, 2, 0], p[:, 2, 1]

    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if np.any(np.abs(det) < 1e-14):
        bad = int(np.argmin(np.abs(det)))
        raise ValueError(f"degenerate cell {bad}: |det| = {abs(det[bad]):.3e}")

    b = np.column_stack([y2 - y3, y3 - y1, y1 - y2]) / det[:, None]
    c = np.column_stack([x3 - x2, x1 - x3, x2 - x1]) / det[:, None]
    return b, c, 0.5 * det


def assemble_stiffness(mesh: TriMesh, k_cell: np.ndarray | float = 1.0) -> sp.csr_matrix:
    """Global stiffness matrix for conductivity ``k_cell`` (per cell or scalar)."""
    b, c, area = shape_gradients(mesh)
    k = np.broadcast_to(np.asarray(k_cell, dtype=np.float64), (mesh.n_cells,))

    # (n_cells, 3, 3) element blocks, vectorised outer products
    ke = (k * area)[:, None, None] * (
        b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]
    )

    tri = mesh.triangles
    rows = np.repeat(tri, 3, axis=1).ravel()
    cols = np.tile(tri, (1, 3)).ravel()
    return sp.coo_matrix(
        (ke.ravel(), (rows, cols)), shape=(mesh.n_nodes, mesh.n_nodes)
    ).tocsr()


def assemble_mass(mesh: TriMesh) -> sp.csr_matrix:
    r"""Consistent mass matrix.

    For a P1 triangle, :math:`\int N_i N_j = \frac{A}{12}(1 + \delta_{ij})`, giving
    the classic ``[[2,1,1],[1,2,1],[1,1,2]] * A/12`` block.
    """
    _, _, area = shape_gradients(mesh)
    local = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]) / 12.0
    me = area[:, None, None] * local[None, :, :]

    tri = mesh.triangles
    rows = np.repeat(tri, 3, axis=1).ravel()
    cols = np.tile(tri, (1, 3)).ravel()
    return sp.coo_matrix(
        (me.ravel(), (rows, cols)), shape=(mesh.n_nodes, mesh.n_nodes)
    ).tocsr()


def assemble_load(mesh: TriMesh, f_node: np.ndarray) -> np.ndarray:
    """Load vector from nodal source values, using the consistent mass matrix.

    ``M @ f`` is exact for a source that is itself P1, and is a better
    approximation than area lumping for smooth sources.
    """
    f = np.asarray(f_node, dtype=np.float64)
    if f.shape != (mesh.n_nodes,):
        raise ValueError(f"f_node must be ({mesh.n_nodes},), got {f.shape}")
    return assemble_mass(mesh) @ f


def apply_dirichlet(
    K: sp.csr_matrix,
    rhs: np.ndarray,
    nodes: np.ndarray,
    values: np.ndarray,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Impose Dirichlet conditions by row replacement with load correction.

    The constrained columns are eliminated into the right-hand side first, so the
    reduced system stays symmetric. Replacing rows without that correction is a
    common and quiet source of wrong answers on non-zero boundary data.
    """
    nodes = np.asarray(nodes, dtype=np.int64)
    values = np.broadcast_to(np.asarray(values, dtype=np.float64), nodes.shape)

    rhs = rhs.astype(np.float64, copy=True)
    K = K.tolil(copy=True)

    g = np.zeros(K.shape[0], dtype=np.float64)
    g[nodes] = values
    rhs -= K.tocsr() @ g

    free = np.ones(K.shape[0], dtype=bool)
    free[nodes] = False
    for i in nodes:
        K.rows[i] = [int(i)]
        K.data[i] = [1.0]
    Kc = K.tocsr()
    Kc = Kc.T.tolil()
    for i in nodes:
        Kc.rows[i] = [int(i)]
        Kc.data[i] = [1.0]
    Kc = Kc.T.tocsr()

    rhs[nodes] = values
    return Kc, rhs


def solve_poisson(
    mesh: TriMesh,
    *,
    source: ScalarField | np.ndarray | float = 1.0,
    conductivity: np.ndarray | float = 1.0,
    dirichlet_value: ScalarField | np.ndarray | float = 0.0,
    dirichlet_nodes: np.ndarray | None = None,
) -> PoissonSolution:
    """Solve the steady diffusion problem on ``mesh``.

    Parameters
    ----------
    source:
        Callable evaluated at node coordinates, an array of nodal values, or a
        constant.
    conductivity:
        Per-cell array or a constant. Piecewise-constant conductivity is what makes
        the learned surrogate task non-trivial.
    dirichlet_value:
        Boundary data, in the same three forms as ``source``.
    dirichlet_nodes:
        Constrained nodes. Defaults to the whole boundary.
    """
    if dirichlet_nodes is None:
        dirichlet_nodes = mesh.boundary_nodes
    dirichlet_nodes = np.asarray(dirichlet_nodes, dtype=np.int64)
    if dirichlet_nodes.size == 0:
        raise ValueError("pure Neumann problem is singular; constrain at least one node")

    f_node = _to_nodal(source, mesh)
    g_node = _to_nodal(dirichlet_value, mesh)
    g_vals = g_node[dirichlet_nodes]

    k_cell = np.broadcast_to(
        np.asarray(conductivity, dtype=np.float64), (mesh.n_cells,)
    ).copy()
    if np.any(k_cell <= 0.0):
        raise ValueError("conductivity must be positive everywhere")

    K = assemble_stiffness(mesh, k_cell)
    rhs = assemble_load(mesh, f_node)
    Kc, rhs_c = apply_dirichlet(K, rhs, dirichlet_nodes, g_vals)

    u = spla.spsolve(Kc.tocsc(), rhs_c)
    if not np.all(np.isfinite(u)):
        raise RuntimeError("linear solve produced non-finite values")

    return PoissonSolution(
        mesh=mesh,
        u=np.asarray(u, dtype=np.float64),
        k_cell=k_cell,
        f_node=f_node,
        dirichlet_nodes=dirichlet_nodes,
        dirichlet_values=g_vals,
    )


def _to_nodal(spec: ScalarField | np.ndarray | float, mesh: TriMesh) -> np.ndarray:
    if callable(spec):
        out = np.asarray(spec(mesh.points), dtype=np.float64)
    else:
        out = np.broadcast_to(np.asarray(spec, dtype=np.float64), (mesh.n_nodes,)).astype(
            np.float64
        )
    if out.shape != (mesh.n_nodes,):
        raise ValueError(f"expected ({mesh.n_nodes},) nodal values, got {out.shape}")
    return out
