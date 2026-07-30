"""Finite element solvers used as ground truth, in 2D and 3D.

Verified rather than assumed: see ``neuralmesh verify`` and ``tests/test_fem.py``.
"""

from .poisson import (
    PoissonSolution,
    apply_dirichlet,
    assemble_load,
    assemble_mass,
    assemble_stiffness,
    shape_gradients,
    solve_poisson,
)

__all__ = [
    "PoissonSolution",
    "apply_dirichlet",
    "assemble_load",
    "assemble_mass",
    "assemble_stiffness",
    "shape_gradients",
    "solve_poisson",
]

from .mesh3d import TetMesh, bar_mesh, box_mesh
from .nonlinear3d import (
    NewtonHistory,
    PowerLawConductivity,
    Solution3D,
    manufactured,
    newton_convergence_orders,
    residual_and_tangent,
    solve_nonlinear,
)

__all__ += [
    "NewtonHistory",
    "PowerLawConductivity",
    "Solution3D",
    "TetMesh",
    "bar_mesh",
    "box_mesh",
    "manufactured",
    "newton_convergence_orders",
    "residual_and_tangent",
    "solve_nonlinear",
]
