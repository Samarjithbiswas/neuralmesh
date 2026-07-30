"""P1 finite element solver used as ground truth.

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
