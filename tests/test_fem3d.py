"""Verification tests for the 3D nonlinear solver.

The most valuable test here is ``test_tangent_matches_finite_differences``. A Newton
tangent that is wrong in the nonlinear term still converges, just linearly instead of
quadratically, and the final answer still looks right. Comparing the analytic tangent
against a finite-difference Jacobian column by column is the only way to be sure the
derivative is actually the derivative of the residual being solved.

The second most valuable is ``test_nonlinearity_is_real``. A "nonlinear" solver that
quietly behaves linearly would pass every convergence test in this file.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralmesh.fem.mesh3d import bar_mesh, boundary_nodes_of, box_mesh
from neuralmesh.fem.nonlinear3d import (
    PowerLawConductivity,
    assemble_mass,
    manufactured,
    newton_convergence_orders,
    residual_and_tangent,
    solve_nonlinear,
)


# ---------------------------------------------------------------------------- geometry
def test_box_volume_is_exact():
    m = box_mesh(6, 5, 4, lx=2.0, ly=1.5, lz=0.5, jitter=0.0)
    assert m.volumes().sum() == pytest.approx(2.0 * 1.5 * 0.5, rel=1e-12)


def test_jitter_preserves_volume_because_boundary_is_fixed():
    """Interior jitter must not change the domain, or Dirichlet data lands off-face."""
    m = box_mesh(7, 7, 7, jitter=0.4, seed=1)
    assert m.volumes().sum() == pytest.approx(1.0, rel=1e-12)


def test_all_tetrahedra_are_positively_oriented():
    m = box_mesh(6, 6, 6, jitter=0.3, seed=2)
    assert m.signed_volumes().min() > 0.0


def test_element_count_matches_kuhn_subdivision():
    nx, ny, nz = 5, 4, 3
    m = box_mesh(nx, ny, nz, jitter=0.0)
    assert m.n_cells == 6 * (nx - 1) * (ny - 1) * (nz - 1)
    assert m.n_nodes == nx * ny * nz


def test_topological_boundary_matches_the_geometric_faces():
    """The boundary is found by face counting, so check it against the box geometry."""
    m = box_mesh(6, 6, 6, jitter=0.0)
    p = m.points
    on_face = (
        np.isclose(p[:, 0], 0)
        | np.isclose(p[:, 0], 1)
        | np.isclose(p[:, 1], 0)
        | np.isclose(p[:, 1], 1)
        | np.isclose(p[:, 2], 0)
        | np.isclose(p[:, 2], 1)
    )
    assert set(m.boundary_nodes) == set(np.flatnonzero(on_face))


def test_boundary_of_a_single_tetrahedron_is_all_four_nodes():
    tets = np.array([[0, 1, 2, 3]])
    assert set(boundary_nodes_of(tets)) == {0, 1, 2, 3}


def test_shape_gradients_sum_to_zero():
    """The four P1 gradients must cancel, since the basis sums to one everywhere."""
    grads, _ = box_mesh(5, 5, 5, jitter=0.25, seed=3).shape_gradients()
    assert np.abs(grads.sum(axis=1)).max() < 1e-10


def test_shape_gradients_reproduce_a_linear_field_exactly():
    """Interpolating a linear field must give its exact constant gradient."""
    m = box_mesh(5, 5, 5, jitter=0.2, seed=4)
    a = np.array([2.0, -3.0, 0.5])
    u = m.points @ a
    grads, _ = m.shape_gradients()
    gu = np.einsum("cn,cnd->cd", u[m.tets], grads)
    assert np.abs(gu - a).max() < 1e-10


def test_quality_of_a_uniform_mesh_is_uniform_and_sane():
    q = box_mesh(6, 6, 6, jitter=0.0).quality()
    assert q.min() > 0.5
    assert q.max() <= 1.0 + 1e-9
    assert np.allclose(q, q[0], atol=1e-9), "a uniform Kuhn mesh should be uniform"


def test_bar_mesh_is_long_and_thin():
    m = bar_mesh(length=8.0, n_long=25, n_cross=4, jitter=0.1)
    ext = m.points.max(axis=0) - m.points.min(axis=0)
    assert ext[0] == pytest.approx(8.0, rel=1e-9)
    assert ext[1] == pytest.approx(1.0, rel=1e-9)


def test_malformed_input_is_rejected():
    from neuralmesh.fem.mesh3d import TetMesh

    with pytest.raises(ValueError):
        TetMesh(points=np.zeros((4, 2)), tets=np.array([[0, 1, 2, 3]]))
    with pytest.raises(ValueError):
        TetMesh(points=np.zeros((4, 3)), tets=np.array([[0, 1, 2]]))
    with pytest.raises(ValueError):
        TetMesh(points=np.zeros((4, 3)), tets=np.array([[0, 1, 2, 9]]))
    with pytest.raises(ValueError):
        box_mesh(1, 5, 5)


# ------------------------------------------------------------------------------ algebra
def test_mass_matrix_integrates_to_the_volume():
    m = box_mesh(6, 5, 4, lx=2.0, ly=1.0, lz=0.5, jitter=0.2, seed=5)
    assert assemble_mass(m).sum() == pytest.approx(1.0, rel=1e-12)


def test_mass_matrix_is_symmetric_and_positive_definite():
    M = assemble_mass(box_mesh(5, 5, 5, jitter=0.1)).toarray()
    assert np.abs(M - M.T).max() < 1e-14
    assert np.linalg.eigvalsh(M).min() > 0.0


def test_tangent_is_symmetric_in_the_linear_case():
    """With alpha = 0 the problem is linear and the tangent must be symmetric."""
    m = box_mesh(5, 5, 5, jitter=0.1, seed=6)
    law = PowerLawConductivity(k0=2.0, alpha=0.0)
    u = np.random.default_rng(0).normal(size=m.n_nodes)
    _, J = residual_and_tangent(m, u, law, np.zeros(m.n_nodes))
    A = J.toarray()
    assert np.abs(A - A.T).max() < 1e-12


def test_tangent_matches_finite_differences():
    """The strongest check available on the Newton tangent.

    A tangent that omits the dk/du term still converges, only linearly, and the
    converged answer still looks correct. Differencing the residual is the only way to
    confirm the analytic Jacobian is the Jacobian of the residual actually being solved.
    """
    m = box_mesh(4, 4, 4, jitter=0.15, seed=7)
    law = PowerLawConductivity(k0=1.3, alpha=2.0)
    rng = np.random.default_rng(1)
    u = 0.4 * rng.normal(size=m.n_nodes)
    load = rng.normal(size=m.n_nodes) * 0.1

    r0, J = residual_and_tangent(m, u, law, load)
    A = J.toarray()

    eps = 1e-7
    probe = rng.choice(m.n_nodes, size=min(12, m.n_nodes), replace=False)
    for j in probe:
        up = u.copy()
        up[j] += eps
        rp, _ = residual_and_tangent(m, up, law, load)
        fd = (rp - r0) / eps
        err = np.abs(fd - A[:, j]).max()
        scale = max(np.abs(A[:, j]).max(), 1.0)
        assert err / scale < 2e-4, (
            f"column {j}: analytic tangent disagrees with finite differences by "
            f"{err:.3e} (relative {err / scale:.2e})"
        )


def test_law_rejects_non_elliptic_parameters():
    with pytest.raises(ValueError):
        PowerLawConductivity(k0=0.0)
    with pytest.raises(ValueError):
        PowerLawConductivity(alpha=-1.0)


def test_law_derivative_matches_finite_difference():
    law = PowerLawConductivity(k0=1.7, alpha=0.9)
    u = np.linspace(-2.0, 2.0, 21)
    eps = 1e-6
    fd = (law.k(u + eps) - law.k(u - eps)) / (2 * eps)
    assert np.abs(fd - law.dk(u)).max() < 1e-6


# ------------------------------------------------------------------------------ solving
def test_patch_test_linear_problem_reproduces_a_linear_field():
    """With alpha = 0 the operator is linear, so a linear field must be exact."""
    m = box_mesh(6, 6, 6, jitter=0.25, seed=8)
    law = PowerLawConductivity(k0=1.0, alpha=0.0)

    def exact(p):
        return 1.0 + 2.0 * p[:, 0] - 0.5 * p[:, 1] + 0.25 * p[:, 2]

    sol = solve_nonlinear(m, source=0.0, law=law, dirichlet_value=exact, tol=1e-12)
    assert sol.converged
    assert sol.max_error(exact) < 1e-10


def test_dirichlet_data_is_imposed_exactly():
    m = box_mesh(5, 5, 5, jitter=0.1)

    def g(p):
        return np.sin(p[:, 0]) + p[:, 2]

    sol = solve_nonlinear(m, source=1.0, dirichlet_value=g)
    b = m.boundary_nodes
    assert np.abs(sol.u[b] - g(m.points)[b]).max() < 1e-13


def test_nonlinearity_is_real():
    """u(2f) must not equal 2 u(f). A linear solver in disguise would pass everything else.

    The source has to be large enough that the nonlinearity actually engages. For
    :math:`-\\nabla^2 u = f` on the unit cube the peak is only about :math:`0.074 f`, so
    with :math:`f = 1` and :math:`\\alpha = 5` the term :math:`\\alpha u^2` is around
    0.005 and the problem is nearly linear by construction. That is a property of the
    amplitude, not of the solver, so the test drives it to :math:`u \\sim 1`.
    """
    m = box_mesh(5, 5, 5, jitter=0.0)
    law = PowerLawConductivity(k0=1.0, alpha=5.0)
    a = solve_nonlinear(m, source=20.0, law=law, dirichlet_value=0.0)
    b = solve_nonlinear(m, source=40.0, law=law, dirichlet_value=0.0)
    assert a.converged and b.converged
    assert np.abs(a.u).max() > 0.4, (
        f"solution peak {np.abs(a.u).max():.3f} is too small for the nonlinearity to bite"
    )
    dev = float(np.abs(2.0 * a.u - b.u).max() / np.abs(b.u).max())
    assert dev > 0.05, f"superposition held to {dev:.4f}; the problem is behaving linearly"


def test_linear_case_does_obey_superposition():
    """The mirror of the previous test: with alpha = 0 superposition must hold."""
    m = box_mesh(5, 5, 5, jitter=0.0)
    law = PowerLawConductivity(alpha=0.0)
    a = solve_nonlinear(m, source=1.0, law=law, dirichlet_value=0.0)
    b = solve_nonlinear(m, source=2.0, law=law, dirichlet_value=0.0)
    assert np.abs(2.0 * a.u - b.u).max() < 1e-9


def test_maximum_principle_for_a_positive_source():
    m = box_mesh(6, 6, 6, jitter=0.1)
    sol = solve_nonlinear(m, source=1.0, dirichlet_value=0.0)
    interior = np.setdiff1d(np.arange(m.n_nodes), m.boundary_nodes)
    assert sol.u[interior].min() > 0.0


def test_higher_conductivity_reduces_the_response():
    m = box_mesh(5, 5, 5, jitter=0.0)
    soft = solve_nonlinear(
        m, source=1.0, law=PowerLawConductivity(k0=1.0, alpha=0.0), dirichlet_value=0.0
    )
    hard = solve_nonlinear(
        m, source=1.0, law=PowerLawConductivity(k0=6.0, alpha=0.0), dirichlet_value=0.0
    )
    assert hard.u.max() < soft.u.max()


def test_energy_is_positive_and_grows_with_the_source():
    m = box_mesh(5, 5, 5, jitter=0.0)
    a = solve_nonlinear(m, source=1.0, dirichlet_value=0.0)
    b = solve_nonlinear(m, source=3.0, dirichlet_value=0.0)
    assert 0.0 < a.energy() < b.energy()


def test_fully_constrained_problem_is_rejected():
    m = box_mesh(2, 2, 2, jitter=0.0)
    with pytest.raises(ValueError):
        solve_nonlinear(m, source=1.0, dirichlet_nodes=np.arange(m.n_nodes))


# -------------------------------------------------------------------- Newton behaviour
def test_newton_converges_quadratically():
    """Quadratic order is the signature of a consistent tangent."""
    m = box_mesh(7, 7, 7, jitter=0.0)
    law = PowerLawConductivity(k0=1.0, alpha=1.0)
    exact, src = manufactured(law)
    sol = solve_nonlinear(m, source=src, law=law, dirichlet_value=exact, tol=1e-12)
    assert sol.converged
    orders = sol.history.convergence_orders()
    assert orders, "not enough usable residuals to estimate an order"
    assert max(orders) > 1.7, f"observed orders {orders} suggest an inconsistent tangent"


def test_convergence_order_helper_recovers_a_known_rate():
    # a synthetic quadratic sequence: r_{n+1} = r_n^2
    seq = [1e-1]
    for _ in range(4):
        seq.append(seq[-1] ** 2)
    orders = newton_convergence_orders(seq)
    assert orders and all(abs(o - 2.0) < 0.05 for o in orders)


def test_linear_problem_converges_in_one_step():
    """With alpha = 0 Newton is exact, so one solve must clear the residual."""
    m = box_mesh(5, 5, 5, jitter=0.0)
    sol = solve_nonlinear(
        m, source=1.0, law=PowerLawConductivity(alpha=0.0), dirichlet_value=0.0, tol=1e-11
    )
    assert sol.converged
    assert sol.history.iterations == 1, (
        f"a linear problem took {sol.history.iterations} Newton steps"
    )


@pytest.mark.slow
def test_manufactured_solution_converges_at_second_order():
    law = PowerLawConductivity(k0=1.0, alpha=1.0)
    exact, src = manufactured(law)
    errors, hs = [], []
    for n in (5, 9, 17):
        m = box_mesh(n, n, n, jitter=0.0)
        sol = solve_nonlinear(m, source=src, law=law, dirichlet_value=exact, tol=1e-11)
        assert sol.converged
        errors.append(sol.l2_error(exact))
        hs.append(1.0 / (n - 1))
    rates = [
        float(np.log(errors[i] / errors[i + 1]) / np.log(hs[i] / hs[i + 1]))
        for i in range(len(errors) - 1)
    ]
    assert all(e > 0 for e in errors)
    assert max(rates) > 1.8, f"observed rates {rates}, expected approaching 2"
    assert min(rates) > 1.5, f"observed rates {rates}"


def test_manufactured_source_includes_the_nonlinear_term():
    """Guard against a source that silently drops the dk/du contribution.

    If the manufactured source omitted that term, a solver bug in the same term would
    cancel against it and the convergence test would still pass. So check the source
    actually depends on alpha.
    """
    p = np.array([[0.3, 0.4, 0.5], [0.7, 0.2, 0.6]])
    _, s0 = manufactured(PowerLawConductivity(k0=1.0, alpha=0.0))
    _, s1 = manufactured(PowerLawConductivity(k0=1.0, alpha=2.0))
    assert np.abs(s0(p) - s1(p)).max() > 1e-3
