"""Verification tests for the finite element solver.

These are the tests that matter most in the whole suite. Everything downstream
treats this solver's output as ground truth, so if the assembly is wrong then every
learned result is measuring the wrong thing and still looks fine.

Each test checks a property with a known answer rather than a stored number, so a
regression cannot be papered over by updating a fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralmesh import (
    assemble_mass,
    assemble_stiffness,
    rectangle_mesh,
    refine,
    solve_poisson,
    unit_square_mesh,
)


def test_stiffness_is_symmetric():
    """Reciprocity: pushing i and measuring at j equals the reverse."""
    K = assemble_stiffness(unit_square_mesh(9, jitter=0.15)).toarray()
    assert np.allclose(K, K.T, atol=1e-12)


def test_stiffness_annihilates_constants():
    """A constant field has zero gradient, so it must be in the null space of K.

    This single check catches most sign and indexing errors in the assembly, because
    almost any mistake breaks the row sums.
    """
    K = assemble_stiffness(unit_square_mesh(11, jitter=0.2))
    ones = np.ones(K.shape[0])
    assert np.abs(K @ ones).max() < 1e-10


def test_mass_matrix_sums_to_area():
    """Total mass equals the domain area, exactly, for a consistent mass matrix."""
    mesh = rectangle_mesh(10, 8, width=3.0, height=2.0, jitter=0.0)
    assert assemble_mass(mesh).sum() == pytest.approx(6.0, rel=1e-12)


def test_mass_matrix_is_positive_definite():
    mesh = unit_square_mesh(8, jitter=0.1)
    eig = np.linalg.eigvalsh(assemble_mass(mesh).toarray())
    assert eig.min() > 0.0


def test_harmonic_field_is_reproduced_exactly():
    r"""A P1 space contains all linear functions, so a linear solution is exact.

    With zero source and boundary data taken from :math:`u = 2x - 3y + 1`, the
    discrete solution should match to round-off, not to discretisation error. This is
    the strongest single check available without a convergence study.
    """
    mesh = unit_square_mesh(12, jitter=0.25, seed=3)

    def exact(p):
        return 2.0 * p[:, 0] - 3.0 * p[:, 1] + 1.0

    sol = solve_poisson(mesh, source=0.0, dirichlet_value=exact)
    assert sol.max_error(exact) < 1e-12


def test_dirichlet_values_are_imposed_exactly():
    mesh = unit_square_mesh(9, jitter=0.1)

    def g(p):
        return np.sin(p[:, 0]) + p[:, 1]

    sol = solve_poisson(mesh, source=2.0, dirichlet_value=g)
    b = mesh.boundary_nodes
    assert np.allclose(sol.u[b], g(mesh.points)[b], atol=1e-12)


def test_solution_scales_linearly_with_source():
    """The problem is linear, so doubling the load must double the answer."""
    mesh = unit_square_mesh(10, jitter=0.15)
    a = solve_poisson(mesh, source=1.0, dirichlet_value=0.0)
    b = solve_poisson(mesh, source=2.0, dirichlet_value=0.0)
    assert np.allclose(2.0 * a.u, b.u, rtol=1e-9, atol=1e-12)


def test_positive_source_gives_positive_interior():
    """Maximum principle: a positive source with zero boundary data lifts the interior."""
    mesh = unit_square_mesh(12, jitter=0.1)
    sol = solve_poisson(mesh, source=1.0, dirichlet_value=0.0)
    interior = np.setdiff1d(np.arange(mesh.n_nodes), mesh.boundary_nodes)
    assert sol.u[interior].min() > 0.0


def test_higher_conductivity_reduces_response():
    """Stiffer medium, smaller deflection, for identical loading."""
    mesh = unit_square_mesh(11, jitter=0.0)
    soft = solve_poisson(mesh, source=1.0, conductivity=1.0, dirichlet_value=0.0)
    hard = solve_poisson(mesh, source=1.0, conductivity=10.0, dirichlet_value=0.0)
    assert hard.u.max() < soft.u.max()


@pytest.mark.slow
def test_second_order_convergence():
    r"""Measured error rate must match the theoretical :math:`O(h^2)`.

    A manufactured solution :math:`u = \sin(\pi x)\sin(\pi y)` with the matching
    source :math:`f = 2\pi^2 u`. Halving the element size should cut the
    mass-weighted :math:`L^2` error by about four.

    This is a real verification test: the theory predicts a specific number, and a
    bug in the stiffness matrix almost always breaks the rate even when it leaves the
    solution looking plausible.
    """

    def exact(p):
        return np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])

    def source(p):
        return 2.0 * np.pi**2 * exact(p)

    errors, sizes = [], []
    for n in (9, 17, 33):
        mesh = unit_square_mesh(n, jitter=0.0)
        sol = solve_poisson(mesh, source=source, dirichlet_value=exact)
        errors.append(sol.l2_error(exact))
        sizes.append(1.0 / (n - 1))

    rates = [
        np.log(errors[i] / errors[i + 1]) / np.log(sizes[i] / sizes[i + 1])
        for i in range(len(errors) - 1)
    ]
    assert all(e > 0 for e in errors)
    # Tolerant band: asymptotic order 2, but coarse meshes carry higher-order terms.
    assert min(rates) > 1.7, f"observed rates {rates}"
    assert max(rates) < 2.4, f"observed rates {rates}"


def test_refine_increases_resolution_and_keeps_accuracy():
    def exact(p):
        return p[:, 0] - 0.5 * p[:, 1]

    coarse = unit_square_mesh(7, jitter=0.0)
    fine = refine(coarse)
    assert fine.n_nodes > coarse.n_nodes
    assert len(fine.triangles) == 4 * len(coarse.triangles)
    sol = solve_poisson(fine, source=0.0, dirichlet_value=exact)
    assert sol.max_error(exact) < 1e-11
