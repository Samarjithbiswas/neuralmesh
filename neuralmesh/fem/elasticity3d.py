r"""Linear elastostatics on tetrahedra, as a second physics.

Everything else in this repository is scalar diffusion. That is a real limitation: a
finding measured on one operator might be a fact about elliptic problems or a fact about
the Laplacian, and there is no way to tell from inside a single physics. Linear
elasticity is the cheapest honest second case. It is still elliptic, so the interior
still depends on every boundary value and the reach question is unchanged, but it is
**vector valued** with three coupled unknowns per node rather than one.

.. math::
    -\nabla\cdot\boldsymbol{\sigma} = \mathbf{f}

.. math::
    \boldsymbol{\sigma} = 2\mu\,\boldsymbol{\varepsilon}
    + \lambda\,\mathrm{tr}(\boldsymbol{\varepsilon})\,\mathbf{I}

.. math::
    \boldsymbol{\varepsilon} = \tfrac{1}{2}
    \left(\nabla\mathbf{u} + \nabla\mathbf{u}^{\mathsf T}\right)

The coupling is the point. In diffusion, one scalar at a node responds to one scalar at
its neighbours. Here a displacement in :math:`x` generates stress that drives
displacement in :math:`y` and :math:`z` through Poisson's ratio, so a learned model has
to represent a genuinely multi-component field rather than three independent copies of
the scalar problem.

Element formulation is standard constant-strain tetrahedra: P1 shape functions give a
constant strain per element, so the element stiffness is

.. math:: \mathbf{K}^e = V_e\, \mathbf{B}^{\mathsf T}\mathbf{D}\,\mathbf{B}

with no quadrature loop, which keeps thousands of solves affordable in pure NumPy.

Verification here is stronger than for the scalar problem, because elasticity has
structure that a wrong assembly cannot fake. The stiffness matrix must have **exactly six
zero eigenvalues**, corresponding to three translations and three rotations, and a rigid
rotation must produce zero strain. Getting the six right is a much sharper test than
symmetry alone: an assembly with a transposed index or a wrong shear term typically
produces five or seven.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mesh3d import TetMesh

VectorField = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class IsotropicMaterial:
    r"""Isotropic linear elastic material, stored as Lame parameters.

    Constructed from Young's modulus and Poisson's ratio because those are what a
    datasheet gives, and converted once. Poisson's ratio is bounded away from 0.5
    deliberately: at exactly 0.5 the material is incompressible, :math:`\lambda` diverges,
    and constant-strain tetrahedra lock, producing a spuriously stiff answer that still
    converges and still looks plausible. Refusing the input is better than returning it.
    """

    E: float = 1.0
    nu: float = 0.3

    def __post_init__(self) -> None:
        if self.E <= 0.0:
            raise ValueError("Young's modulus must be positive")
        if not (-1.0 < self.nu < 0.5):
            raise ValueError(
                f"Poisson's ratio must lie in (-1, 0.5); got {self.nu}. At 0.5 the material "
                "is incompressible and constant-strain tetrahedra lock."
            )
        if self.nu > 0.49:
            # Not fatal, but the user should know the element is near its failure mode.
            import warnings

            warnings.warn(
                f"nu = {self.nu} is close to incompressible; P1 tetrahedra lock and will "
                "report a stiffer response than the true solution.",
                RuntimeWarning,
                stacklevel=2,
            )

    @property
    def lam(self) -> float:
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    @property
    def mu(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    def constitutive(self) -> np.ndarray:
        """The 6x6 matrix D in Voigt order (xx, yy, zz, xy, yz, zx)."""
        lam, mu = self.lam, self.mu
        D = np.zeros((6, 6), dtype=np.float64)
        D[:3, :3] = lam
        D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * mu
        D[3, 3] = D[4, 4] = D[5, 5] = mu
        return D


@dataclass
class ElasticSolution:
    """Result of an elastostatic solve. Displacement is ``(n_nodes, 3)``."""

    mesh: TetMesh
    u: np.ndarray
    material: IsotropicMaterial
    f_node: np.ndarray
    dirichlet_nodes: np.ndarray
    dirichlet_values: np.ndarray

    def l2_error(self, exact: VectorField) -> float:
        """Volume-weighted L2 error over all three components."""
        u_ex = np.asarray(exact(self.mesh.points), dtype=np.float64)
        err = self.u - u_ex
        M = assemble_scalar_mass(self.mesh)
        return float(np.sqrt(max(sum(err[:, k] @ (M @ err[:, k]) for k in range(3)), 0.0)))

    def max_error(self, exact: VectorField) -> float:
        u_ex = np.asarray(exact(self.mesh.points), dtype=np.float64)
        return float(np.abs(self.u - u_ex).max())

    def strains(self) -> np.ndarray:
        """``(n_cells, 6)`` constant strain per element, Voigt order."""
        B = strain_operators(self.mesh)
        ue = self.u[self.mesh.tets].reshape(self.mesh.n_cells, 12)
        return np.einsum("cij,cj->ci", B, ue)

    def stresses(self) -> np.ndarray:
        return self.strains() @ self.material.constitutive().T

    def von_mises(self) -> np.ndarray:
        s = self.stresses()
        sx, sy, sz, txy, tyz, tzx = (s[:, i] for i in range(6))
        return np.sqrt(
            0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
            + 3.0 * (txy**2 + tyz**2 + tzx**2)
        )

    def strain_energy(self) -> float:
        vol = self.mesh.volumes()
        return float(
            0.5 * np.sum(vol * np.einsum("ci,ci->c", self.strains(), self.stresses()))
        )


def strain_operators(mesh: TetMesh) -> np.ndarray:
    """``(n_cells, 6, 12)`` strain-displacement matrices B, Voigt order.

    Rows are the six strain components, columns the twelve element degrees of freedom
    ordered node-major as ``(u0x, u0y, u0z, u1x, ...)``.
    """
    grads, _ = mesh.shape_gradients()  # (C, 4, 3)
    C = mesh.n_cells
    B = np.zeros((C, 6, 12), dtype=np.float64)
    for i in range(4):
        gx, gy, gz = grads[:, i, 0], grads[:, i, 1], grads[:, i, 2]
        c = 3 * i
        B[:, 0, c + 0] = gx
        B[:, 1, c + 1] = gy
        B[:, 2, c + 2] = gz
        B[:, 3, c + 0] = gy
        B[:, 3, c + 1] = gx
        B[:, 4, c + 1] = gz
        B[:, 4, c + 2] = gy
        B[:, 5, c + 0] = gz
        B[:, 5, c + 2] = gx
    return B


def assemble_stiffness(mesh: TetMesh, material: IsotropicMaterial) -> sp.csr_matrix:
    r"""Global stiffness.

    :math:`\mathbf{K}^e = V_e\,\mathbf{B}^{\mathsf T}\mathbf{D}\mathbf{B}`
    """
    B = strain_operators(mesh)
    D = material.constitutive()
    vol = mesh.volumes()
    ke = vol[:, None, None] * np.einsum("cki,kl,clj->cij", B, D, B)

    dofs = (3 * mesh.tets[:, :, None] + np.arange(3)[None, None, :]).reshape(mesh.n_cells, 12)
    rows = np.repeat(dofs, 12, axis=1).ravel()
    cols = np.tile(dofs, (1, 12)).ravel()
    n = 3 * mesh.n_nodes
    return sp.csr_matrix((ke.ravel(), (rows, cols)), shape=(n, n))


def assemble_scalar_mass(mesh: TetMesh) -> sp.csr_matrix:
    """P1 mass matrix on the scalar space, used for error norms."""
    vol = mesh.volumes()
    local = (np.ones((4, 4)) + np.eye(4)) / 20.0
    ke = vol[:, None, None] * local[None, :, :]
    t = mesh.tets
    rows = np.repeat(t, 4, axis=1).ravel()
    cols = np.tile(t, (1, 4)).ravel()
    return sp.csr_matrix((ke.ravel(), (rows, cols)), shape=(mesh.n_nodes, mesh.n_nodes))


def assemble_load(mesh: TetMesh, f_node: np.ndarray) -> np.ndarray:
    """Consistent load vector for a nodal body force ``(n_nodes, 3)``."""
    M = assemble_scalar_mass(mesh)
    return np.column_stack([M @ f_node[:, k] for k in range(3)]).ravel()


def rigid_body_modes(mesh: TetMesh) -> np.ndarray:
    """The six-dimensional null space of an unconstrained stiffness matrix.

    Three translations and three infinitesimal rotations. Returned orthonormalised so it
    can be used directly to test that ``K`` annihilates exactly this space and nothing
    more.
    """
    p = mesh.points
    n = mesh.n_nodes
    modes = np.zeros((3 * n, 6), dtype=np.float64)
    for k in range(3):
        modes[k::3, k] = 1.0
    # rotations about x, y, z
    rot = [(1, 2), (2, 0), (0, 1)]
    for j, (a, b) in enumerate(rot):
        modes[a::3, 3 + j] = -p[:, b]
        modes[b::3, 3 + j] = p[:, a]
    q, _ = np.linalg.qr(modes)
    return q


def _to_vector_field(
    spec: VectorField | np.ndarray | float, mesh: TetMesh, name: str
) -> np.ndarray:
    """Normalise a vector input to ``(n_nodes, 3)``, with a useful error if it cannot be.

    Accepts a callable evaluated at the nodes, a single scalar applied to all components,
    one vector of length three applied to every node, or a full nodal field.

    The shape is validated *before* any broadcast. Letting NumPy broadcast first and
    checking afterwards means a wrongly shaped input dies with "operands could not be
    broadcast together", which tells the caller nothing about which argument was wrong.
    """
    if callable(spec):
        out = np.asarray(spec(mesh.points), dtype=np.float64)
    else:
        arr = np.asarray(spec, dtype=np.float64)
        if arr.ndim == 0:
            out = np.full((mesh.n_nodes, 3), float(arr))
        elif arr.shape == (3,):
            out = np.tile(arr, (mesh.n_nodes, 1))
        else:
            out = arr
    if out.shape != (mesh.n_nodes, 3):
        raise ValueError(
            f"{name} must be a scalar, a length-3 vector, or an array of shape "
            f"({mesh.n_nodes}, 3); got {out.shape}"
        )
    return out


def solve_elasticity(
    mesh: TetMesh,
    *,
    body_force: VectorField | np.ndarray | float = 0.0,
    material: IsotropicMaterial | None = None,
    dirichlet_value: VectorField | np.ndarray | float = 0.0,
    dirichlet_nodes: np.ndarray | None = None,
) -> ElasticSolution:
    """Solve the linear elastostatic problem with Dirichlet displacement data."""
    material = material or IsotropicMaterial()

    f_node = _to_vector_field(body_force, mesh, "body force")
    g_node = _to_vector_field(dirichlet_value, mesh, "boundary data")

    bnd = (
        mesh.boundary_nodes
        if dirichlet_nodes is None
        else np.asarray(dirichlet_nodes, dtype=np.int64)
    )
    fixed = (3 * bnd[:, None] + np.arange(3)[None, :]).ravel()
    free = np.setdiff1d(np.arange(3 * mesh.n_nodes), fixed)
    if free.size == 0:
        raise ValueError("every degree of freedom is constrained; nothing to solve for")

    K = assemble_stiffness(mesh, material)
    rhs = assemble_load(mesh, f_node)

    u = np.zeros(3 * mesh.n_nodes, dtype=np.float64)
    u[fixed] = g_node.ravel()[fixed]
    rhs = rhs - K @ u

    Kff = K[free][:, free].tocsc()
    u[free] = spla.spsolve(Kff, rhs[free])

    return ElasticSolution(
        mesh=mesh,
        u=u.reshape(mesh.n_nodes, 3),
        material=material,
        f_node=f_node,
        dirichlet_nodes=bnd,
        dirichlet_values=g_node,
    )


def manufactured(material: IsotropicMaterial | None = None) -> tuple[VectorField, VectorField]:
    r"""Exact displacement and the body force that produces it.

        Take :math:`\mathbf{u} = (\sin\pi x\,\sin\pi y\,\sin\pi z,\;0,\;0)`, which vanishes on
        the boundary of the unit cube. Using the Navier form
        :math:`
    abla\cdotoldsymbol\sigma = (\lambda+\mu)
    abla(
    abla\cdot\mathbf u)
        + \mu
    abla^2\mathbf u`,

        .. math::
            f_x = \pi^2(\lambda + 4\mu)\,s_x s_y s_z, \quad
            f_y = -\pi^2(\lambda+\mu)\,c_x c_y s_z, \quad
            f_z = -\pi^2(\lambda+\mu)\,c_x s_y c_z

        Note that :math:`f_y` and :math:`f_z` are non-zero even though the displacement has
        only an :math:`x` component. That cross-coupling is exactly the structure a scalar
        problem does not have, and a manufactured source that omitted it would let a
        decoupled, wrong solver pass the convergence test.
    """
    material = material or IsotropicMaterial()
    lam, mu = material.lam, material.mu
    pi = np.pi

    def exact(p: np.ndarray) -> np.ndarray:
        s = np.sin(pi * p[:, 0]) * np.sin(pi * p[:, 1]) * np.sin(pi * p[:, 2])
        return np.column_stack([s, np.zeros_like(s), np.zeros_like(s)])

    def force(p: np.ndarray) -> np.ndarray:
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        sx, sy, sz = np.sin(pi * x), np.sin(pi * y), np.sin(pi * z)
        cx, cy, cz = np.cos(pi * x), np.cos(pi * y), np.cos(pi * z)
        fx = pi**2 * (lam + 4.0 * mu) * sx * sy * sz
        fy = -(pi**2) * (lam + mu) * cx * cy * sz
        fz = -(pi**2) * (lam + mu) * cx * sy * cz
        return np.column_stack([fx, fy, fz])

    return exact, force
