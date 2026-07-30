r"""Nonlinear diffusion in three dimensions, solved by Newton-Raphson.

The problem is quasilinear and elliptic:

.. math::
    -\nabla\cdot\!\big(k(u)\,\nabla u\big) = f \quad\text{in }\Omega,
    \qquad u = g \quad\text{on }\partial\Omega

with a solution-dependent conductivity, by default :math:`k(u) = k_0(1 + \alpha u^2)`.
This is a genuine nonlinearity rather than a linear problem with awkward coefficients:
the operator applied to :math:`2u` is not twice the operator applied to :math:`u`, so
there is no superposition and no single linear solve that gets the answer.

Why this problem for a learned-simulation benchmark:

* It stays elliptic, so the solution at every interior point still depends on every
  boundary value. That is the property under-reaching is about, and it survives the move
  to 3D and to nonlinearity.
* It is nonlinear in a way that has a physical reading (temperature-dependent
  conductivity, saturating magnetics, pressure-dependent permeability), so the benchmark
  is not an artificial construction.
* It admits an exact manufactured solution, so it can be verified rather than trusted.

The Newton tangent is *consistent*, meaning it includes the
:math:`\mathrm{d}k/\mathrm{d}u` term rather than only the :math:`k(u)` term. A tangent
that drops it still converges, but linearly instead of quadratically. Quadratic
convergence is therefore a verification signal in its own right: if the observed
residual does not square each iteration, the tangent is wrong, and
:func:`newton_convergence_orders` measures exactly that.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mesh3d import TetMesh

Field = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------- constitutive law
@dataclass(frozen=True)
class PowerLawConductivity:
    r""":math:`k(u) = k_0\,(1 + \alpha u^{2})`, with :math:`k'(u) = 2 k_0 \alpha u`.

    Squared rather than linear in :math:`u` so that :math:`k` stays positive for any
    sign of the solution. A law that can cross zero turns an elliptic problem into
    something with no unique solution, and Newton will happily march toward it.
    """

    k0: float = 1.0
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.k0 <= 0.0:
            raise ValueError("k0 must be positive for the problem to stay elliptic")
        if self.alpha < 0.0:
            raise ValueError("alpha must be non-negative to keep k(u) positive")

    def k(self, u: np.ndarray) -> np.ndarray:
        return self.k0 * (1.0 + self.alpha * u**2)

    def dk(self, u: np.ndarray) -> np.ndarray:
        return 2.0 * self.k0 * self.alpha * u


@dataclass
class NewtonHistory:
    residual_norms: list[float] = field(default_factory=list)
    step_norms: list[float] = field(default_factory=list)

    @property
    def iterations(self) -> int:
        return len(self.step_norms)

    def convergence_orders(self) -> list[float]:
        """Observed order of convergence between consecutive residuals."""
        return newton_convergence_orders(self.residual_norms)


@dataclass
class Solution3D:
    """Result of a nonlinear 3D solve."""

    mesh: TetMesh
    u: np.ndarray
    law: PowerLawConductivity
    f_node: np.ndarray
    dirichlet_nodes: np.ndarray
    dirichlet_values: np.ndarray
    history: NewtonHistory
    converged: bool

    def l2_error(self, exact: Field) -> float:
        """Mass-weighted :math:`L^2` error, so the number is mesh independent."""
        u_ex = np.asarray(exact(self.mesh.points), dtype=np.float64)
        e = self.u - u_ex
        M = assemble_mass(self.mesh)
        return float(np.sqrt(max(e @ (M @ e), 0.0)))

    def max_error(self, exact: Field) -> float:
        u_ex = np.asarray(exact(self.mesh.points), dtype=np.float64)
        return float(np.abs(self.u - u_ex).max())

    def energy(self) -> float:
        r"""Dirichlet-type energy :math:`\tfrac12\int k(u)|\nabla u|^2`."""
        grads, vol = self.mesh.shape_gradients()
        ue = self.u[self.mesh.tets]
        gu = np.einsum("cn,cnd->cd", ue, grads)
        k = self.law.k(ue.mean(axis=1))
        return float(0.5 * np.sum(k * vol * np.einsum("cd,cd->c", gu, gu)))


def newton_convergence_orders(norms: list[float]) -> list[float]:
    r"""Estimate :math:`p` in :math:`\|r_{n+1}\| \sim \|r_n\|^p`.

    Two is the signature of a consistent tangent. One means the tangent is missing a
    term, which is the most common silent error in a nonlinear solver because the
    iteration still converges and the answer still looks right.
    """
    out: list[float] = []
    for i in range(len(norms) - 2):
        a, b, c = norms[i], norms[i + 1], norms[i + 2]
        if min(a, b, c) <= 0.0 or a == b:
            continue
        # guard against the machine-precision floor, where the ratio is meaningless
        if c < 1e-14 or b < 1e-13:
            continue
        out.append(float(np.log(c / b) / np.log(b / a)))
    return out


# ------------------------------------------------------------------------------ assembly
def assemble_mass(mesh: TetMesh) -> sp.csr_matrix:
    r"""Consistent P1 mass matrix on tetrahedra.

    The exact element matrix is :math:`\frac{V}{20}(1 + \delta_{ij})`, so diagonal
    entries are :math:`V/10` and off-diagonals :math:`V/20`. Total mass equals the
    domain volume exactly, which is the cheapest available check on the assembly.
    """
    vol = mesh.volumes()
    local = (np.ones((4, 4)) + np.eye(4)) / 20.0
    ke = vol[:, None, None] * local[None, :, :]
    return _scatter(mesh, ke)


def _scatter(mesh: TetMesh, ke: np.ndarray) -> sp.csr_matrix:
    """Scatter per-element 4x4 matrices into a global sparse matrix."""
    t = mesh.tets
    rows = np.repeat(t, 4, axis=1).ravel()
    cols = np.tile(t, (1, 4)).ravel()
    return sp.csr_matrix((ke.ravel(), (rows, cols)), shape=(mesh.n_nodes, mesh.n_nodes))


def assemble_load(mesh: TetMesh, f_node: np.ndarray) -> np.ndarray:
    """Consistent load vector, ``M f`` rather than a lumped approximation."""
    return assemble_mass(mesh) @ np.asarray(f_node, dtype=np.float64)


def residual_and_tangent(
    mesh: TetMesh,
    u: np.ndarray,
    law: PowerLawConductivity,
    load: np.ndarray,
) -> tuple[np.ndarray, sp.csr_matrix]:
    r"""Nonlinear residual and its consistent Jacobian.

    With :math:`u_e` the element mean of :math:`u`,

    .. math::
        r_i = \sum_e k(u_e)\,V_e\,\nabla u\cdot\nabla N_i - F_i,
        \qquad
        \frac{\partial r_i}{\partial u_j} = \sum_e
        \Big[ k(u_e) V_e \nabla N_i\cdot\nabla N_j
        + \tfrac14 k'(u_e) V_e\,(\nabla u\cdot\nabla N_i) \Big]

    The second term is the one that is easy to omit. Without it Newton still converges,
    but linearly, and :func:`newton_convergence_orders` will report about 1 instead of 2.
    """
    grads, vol = mesh.shape_gradients()  # (C,4,3), (C,)
    ue = u[mesh.tets]  # (C,4)
    u_elem = ue.mean(axis=1)  # (C,)
    gu = np.einsum("cn,cnd->cd", ue, grads)  # (C,3) gradient of u per element

    k = law.k(u_elem)
    dk = law.dk(u_elem)

    # residual: k * V * (grad u . grad N_i)
    gu_dot_gN = np.einsum("cd,cnd->cn", gu, grads)  # (C,4)
    contrib = (k * vol)[:, None] * gu_dot_gN  # (C,4)
    r = np.zeros(mesh.n_nodes, dtype=np.float64)
    np.add.at(r, mesh.tets, contrib)
    r -= load

    # tangent: symmetric part plus the rank-one dk/du part
    sym = (k * vol)[:, None, None] * np.einsum("cnd,cmd->cnm", grads, grads)
    rank1 = 0.25 * (dk * vol)[:, None, None] * gu_dot_gN[:, :, None] * np.ones((1, 1, 4))
    return r, _scatter(mesh, sym + rank1)


def _to_nodal(spec: Field | np.ndarray | float, mesh: TetMesh) -> np.ndarray:
    if callable(spec):
        return np.asarray(spec(mesh.points), dtype=np.float64).reshape(mesh.n_nodes)
    arr = np.asarray(spec, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(mesh.n_nodes, float(arr))
    if arr.shape != (mesh.n_nodes,):
        raise ValueError(f"expected ({mesh.n_nodes},), got {arr.shape}")
    return arr


def solve_nonlinear(
    mesh: TetMesh,
    *,
    source: Field | np.ndarray | float = 1.0,
    law: PowerLawConductivity | None = None,
    dirichlet_value: Field | np.ndarray | float = 0.0,
    dirichlet_nodes: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 40,
    verbose: bool = False,
) -> Solution3D:
    """Solve the nonlinear diffusion problem by Newton-Raphson.

    Convergence is judged on the residual norm restricted to free degrees of freedom.
    Judging it on the step size instead is a common mistake: a step can be tiny while
    the residual is still large, which reports success on a stalled iteration.
    """
    law = law or PowerLawConductivity()
    f_node = _to_nodal(source, mesh)
    g_node = _to_nodal(dirichlet_value, mesh)
    bnd = (
        mesh.boundary_nodes
        if dirichlet_nodes is None
        else np.asarray(dirichlet_nodes, dtype=np.int64)
    )
    free = np.setdiff1d(np.arange(mesh.n_nodes), bnd)
    if free.size == 0:
        raise ValueError("every node is constrained; nothing to solve for")

    load = assemble_load(mesh, f_node)

    # start from the boundary data extended by zero, which already satisfies the
    # constraints exactly, so every Newton step can keep them satisfied
    u = np.zeros(mesh.n_nodes, dtype=np.float64)
    u[bnd] = g_node[bnd]

    hist = NewtonHistory()
    converged = False

    for it in range(max_iter):
        r, J = residual_and_tangent(mesh, u, law, load)
        rn = float(np.linalg.norm(r[free]))
        hist.residual_norms.append(rn)
        if verbose:
            print(f"  newton {it:3d}  |r| = {rn:.6e}")
        if rn < tol:
            converged = True
            break

        Jff = J[free][:, free].tocsc()
        du = spla.spsolve(Jff, -r[free])
        if not np.all(np.isfinite(du)):
            break
        u[free] += du
        hist.step_norms.append(float(np.linalg.norm(du)))
    else:
        r, _ = residual_and_tangent(mesh, u, law, load)
        rn = float(np.linalg.norm(r[free]))
        hist.residual_norms.append(rn)
        converged = rn < tol

    return Solution3D(
        mesh=mesh,
        u=u,
        law=law,
        f_node=f_node,
        dirichlet_nodes=bnd,
        dirichlet_values=g_node,
        history=hist,
        converged=converged,
    )


# ------------------------------------------------------------------ manufactured solution
def manufactured(law: PowerLawConductivity | None = None) -> tuple[Field, Field]:
    r"""An exact solution and its matching source, for verification.

    Take :math:`u = \sin(\pi x)\sin(\pi y)\sin(\pi z)`, which vanishes on the boundary
    of the unit cube. For :math:`k(u) = k_0(1+\alpha u^2)`,

    .. math::
        f = -\nabla\cdot(k\nabla u)
          = -k(u)\,\nabla^2 u - k'(u)\,|\nabla u|^2
          = 3\pi^2 k(u)\, u - k'(u)\,|\nabla u|^2

    using :math:`\nabla^2 u = -3\pi^2 u`. Both terms are needed: dropping the second
    gives a source that is wrong by an amount proportional to alpha, which is precisely
    the nonlinear part, so a solver bug and a source bug would cancel and the
    convergence test would still pass. Returned together for that reason.
    """
    law = law or PowerLawConductivity()
    pi = np.pi

    def exact(p: np.ndarray) -> np.ndarray:
        return np.sin(pi * p[:, 0]) * np.sin(pi * p[:, 1]) * np.sin(pi * p[:, 2])

    def source(p: np.ndarray) -> np.ndarray:
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        sx, sy, sz = np.sin(pi * x), np.sin(pi * y), np.sin(pi * z)
        cx, cy, cz = np.cos(pi * x), np.cos(pi * y), np.cos(pi * z)
        u = sx * sy * sz
        grad_sq = (pi**2) * ((cx * sy * sz) ** 2 + (sx * cy * sz) ** 2 + (sx * sy * cz) ** 2)
        return 3.0 * pi**2 * law.k(u) * u - law.dk(u) * grad_sq

    return exact, source
