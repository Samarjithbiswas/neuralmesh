"""Verification tests for the 3D linear elasticity solver.

Elasticity has structure that a wrong assembly cannot fake, which makes it easier to
verify than the scalar problem. Two tests carry most of the weight.

``test_stiffness_has_exactly_six_zero_modes`` is the sharpest check available. An
unconstrained elastic body can translate three ways and rotate three ways, and nothing
else, so the stiffness matrix must have exactly six zero eigenvalues. A transposed index
or a wrong shear term in the strain-displacement matrix typically produces five or seven,
and both are caught here while a symmetry check would pass.

``test_manufactured_force_has_transverse_components`` guards the convergence test. The
manufactured displacement points only along x, but the body force that produces it has
non-zero y and z components because of Poisson coupling. A solver that treated the three
displacement components as independent would be wrong, and would still pass a convergence
test built on a source that omitted those terms.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralmesh.fem.elasticity3d import (
    ElasticSolution,
    IsotropicMaterial,
    assemble_load,
    assemble_scalar_mass,
    assemble_stiffness,
    manufactured,
    rigid_body_modes,
    solve_elasticity,
    strain_operators,
)
from neuralmesh.fem.mesh3d import bar_mesh, box_mesh

STEEL = IsotropicMaterial(E=1.0, nu=0.3)


# ------------------------------------------------------------------------- material
def test_lame_parameters_match_the_textbook_conversion():
    m = IsotropicMaterial(E=210.0, nu=0.3)
    assert m.mu == pytest.approx(210.0 / (2 * 1.3))
    assert m.lam == pytest.approx(210.0 * 0.3 / (1.3 * 0.4))


def test_material_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        IsotropicMaterial(E=0.0)
    with pytest.raises(ValueError):
        IsotropicMaterial(E=-1.0)
    with pytest.raises(ValueError, match="incompressible"):
        IsotropicMaterial(nu=0.5)
    with pytest.raises(ValueError):
        IsotropicMaterial(nu=-1.5)


def test_near_incompressible_material_warns_about_locking():
    """P1 tetrahedra lock as nu approaches 0.5 and report a stiffer answer than the truth."""
    with pytest.warns(RuntimeWarning, match="lock"):
        IsotropicMaterial(nu=0.495)


def test_constitutive_matrix_is_symmetric_and_positive_definite():
    D = STEEL.constitutive()
    assert np.allclose(D, D.T)
    assert np.linalg.eigvalsh(D).min() > 0.0


def test_constitutive_matrix_has_the_expected_block_structure():
    D = IsotropicMaterial(E=1.0, nu=0.25).constitutive()
    lam, mu = IsotropicMaterial(E=1.0, nu=0.25).lam, IsotropicMaterial(E=1.0, nu=0.25).mu
    assert D[0, 0] == pytest.approx(lam + 2 * mu)
    assert D[0, 1] == pytest.approx(lam)
    assert D[3, 3] == pytest.approx(mu)
    # no coupling between normal and shear for an isotropic material
    assert np.allclose(D[:3, 3:], 0.0)


# -------------------------------------------------------------------- strain operator
def test_rigid_translation_produces_zero_strain():
    mesh = box_mesh(5, 5, 5, jitter=0.2, seed=1)
    B = strain_operators(mesh)
    u = np.tile(np.array([0.3, -0.2, 0.7]), (mesh.n_nodes, 1))
    ue = u[mesh.tets].reshape(mesh.n_cells, 12)
    assert np.abs(np.einsum("cij,cj->ci", B, ue)).max() < 1e-12


def test_rigid_rotation_produces_zero_strain():
    """Infinitesimal rotation is the harder of the two rigid-body checks."""
    mesh = box_mesh(5, 5, 5, jitter=0.2, seed=2)
    B = strain_operators(mesh)
    p = mesh.points
    omega = np.array([0.01, -0.02, 0.015])
    u = np.cross(np.tile(omega, (mesh.n_nodes, 1)), p)
    ue = u[mesh.tets].reshape(mesh.n_cells, 12)
    assert np.abs(np.einsum("cij,cj->ci", B, ue)).max() < 1e-12


def test_uniform_stretch_gives_the_exact_strain():
    mesh = box_mesh(5, 5, 5, jitter=0.15, seed=3)
    B = strain_operators(mesh)
    eps = 0.004
    u = np.column_stack(
        [eps * mesh.points[:, 0], np.zeros(mesh.n_nodes), np.zeros(mesh.n_nodes)]
    )
    ue = u[mesh.tets].reshape(mesh.n_cells, 12)
    strain = np.einsum("cij,cj->ci", B, ue)
    assert np.allclose(strain[:, 0], eps, atol=1e-12)
    assert np.abs(strain[:, 1:]).max() < 1e-12


def test_pure_shear_lands_in_the_shear_component():
    mesh = box_mesh(5, 5, 5, jitter=0.0)
    B = strain_operators(mesh)
    g = 0.003
    u = np.column_stack(
        [g * mesh.points[:, 1], np.zeros(mesh.n_nodes), np.zeros(mesh.n_nodes)]
    )
    ue = u[mesh.tets].reshape(mesh.n_cells, 12)
    strain = np.einsum("cij,cj->ci", B, ue)
    assert np.allclose(strain[:, 3], g, atol=1e-12)  # engineering shear gamma_xy
    assert np.abs(strain[:, :3]).max() < 1e-12


# ----------------------------------------------------------------------- stiffness
def test_stiffness_is_symmetric():
    K = assemble_stiffness(box_mesh(5, 5, 5, jitter=0.15, seed=4), STEEL).toarray()
    assert np.abs(K - K.T).max() < 1e-12


def test_stiffness_has_exactly_six_zero_modes():
    """Three translations and three rotations, no more and no fewer.

    This is the sharpest single check on the assembly. A wrong shear term or a transposed
    index usually produces five or seven zero modes, and a symmetry check would miss both.
    """
    mesh = box_mesh(5, 5, 5, jitter=0.15, seed=5)
    K = assemble_stiffness(mesh, STEEL).toarray()
    eig = np.linalg.eigvalsh(K)
    scale = max(abs(eig).max(), 1.0)
    n_zero = int((np.abs(eig) < 1e-9 * scale).sum())
    assert n_zero == 6, f"expected 6 rigid-body modes, found {n_zero}"
    assert eig.min() > -1e-9 * scale, "stiffness is not positive semidefinite"


def test_stiffness_annihilates_the_rigid_body_modes():
    mesh = box_mesh(5, 5, 5, jitter=0.2, seed=6)
    K = assemble_stiffness(mesh, STEEL)
    R = rigid_body_modes(mesh)
    assert np.abs(K @ R).max() < 1e-10


def test_rigid_body_modes_are_orthonormal_and_six_dimensional():
    R = rigid_body_modes(box_mesh(4, 4, 4, jitter=0.0))
    assert R.shape[1] == 6
    assert np.allclose(R.T @ R, np.eye(6), atol=1e-10)


def test_stiffness_scales_linearly_with_youngs_modulus():
    mesh = box_mesh(4, 4, 4, jitter=0.0)
    a = assemble_stiffness(mesh, IsotropicMaterial(E=1.0, nu=0.3)).toarray()
    b = assemble_stiffness(mesh, IsotropicMaterial(E=3.0, nu=0.3)).toarray()
    assert np.allclose(3.0 * a, b, atol=1e-10)


def test_scalar_mass_integrates_to_the_volume():
    mesh = box_mesh(6, 5, 4, lx=2.0, ly=1.0, lz=0.5, jitter=0.2, seed=7)
    assert assemble_scalar_mass(mesh).sum() == pytest.approx(1.0, rel=1e-12)


def test_load_vector_has_three_components_per_node():
    mesh = box_mesh(4, 4, 4, jitter=0.0)
    f = np.tile(np.array([1.0, 0.0, 0.0]), (mesh.n_nodes, 1))
    rhs = assemble_load(mesh, f)
    assert rhs.shape == (3 * mesh.n_nodes,)
    # total force in x equals volume times unit density, y and z vanish
    assert rhs[0::3].sum() == pytest.approx(1.0, rel=1e-10)
    assert abs(rhs[1::3].sum()) < 1e-12


# -------------------------------------------------------------------------- solving
def test_patch_test_linear_displacement_is_exact():
    """A P1 space contains every linear field, so a linear displacement must be exact."""
    mesh = box_mesh(6, 6, 6, jitter=0.25, seed=8)
    A = np.array([[0.02, -0.01, 0.005], [0.003, 0.015, -0.002], [-0.004, 0.006, 0.011]])

    def exact(p):
        return p @ A.T

    sol = solve_elasticity(mesh, body_force=0.0, material=STEEL, dirichlet_value=exact)
    assert sol.max_error(exact) < 1e-11


def test_dirichlet_data_is_imposed_exactly():
    mesh = box_mesh(5, 5, 5, jitter=0.1)

    def g(p):
        return np.column_stack([0.01 * p[:, 1], 0.02 * p[:, 2], -0.01 * p[:, 0]])

    sol = solve_elasticity(mesh, body_force=0.0, material=STEEL, dirichlet_value=g)
    b = sol.dirichlet_nodes
    assert np.abs(sol.u[b] - g(mesh.points)[b]).max() < 1e-12


def test_response_scales_linearly_with_load():
    mesh = box_mesh(5, 5, 5, jitter=0.0)
    f1 = np.tile([0.0, 0.0, -1.0], (mesh.n_nodes, 1))
    a = solve_elasticity(mesh, body_force=f1, material=STEEL)
    b = solve_elasticity(mesh, body_force=2.0 * f1, material=STEEL)
    assert np.allclose(2.0 * a.u, b.u, rtol=1e-8, atol=1e-14)


def test_stiffer_material_deflects_less():
    mesh = box_mesh(5, 5, 5, jitter=0.0)
    f = np.tile([0.0, 0.0, -1.0], (mesh.n_nodes, 1))
    soft = solve_elasticity(mesh, body_force=f, material=IsotropicMaterial(E=1.0, nu=0.3))
    hard = solve_elasticity(mesh, body_force=f, material=IsotropicMaterial(E=10.0, nu=0.3))
    assert np.abs(hard.u).max() < np.abs(soft.u).max()


def test_poisson_coupling_produces_transverse_motion():
    """Stretch along x and the body must contract in y and z. Scalar problems cannot do this.

    This is the property that makes elasticity a genuine second physics rather than three
    copies of the diffusion problem.
    """
    mesh = box_mesh(6, 6, 6, jitter=0.0)
    p = mesh.points
    x0, x1 = p[:, 0].min(), p[:, 0].max()
    ends = np.flatnonzero(np.isclose(p[:, 0], x0) | np.isclose(p[:, 0], x1))

    g = np.zeros((mesh.n_nodes, 3))
    g[np.isclose(p[:, 0], x1), 0] = 0.01  # pull the far face along x

    sol = solve_elasticity(
        mesh,
        body_force=0.0,
        material=IsotropicMaterial(E=1.0, nu=0.35),
        dirichlet_value=g,
        dirichlet_nodes=ends,
    )
    interior = np.setdiff1d(np.arange(mesh.n_nodes), ends)
    transverse = np.abs(sol.u[interior, 1:]).max()
    assert transverse > 1e-5, (
        f"stretching along x produced only {transverse:.2e} transverse motion; "
        "the components are not coupled"
    )


def test_zero_poisson_ratio_gives_no_transverse_motion():
    """The mirror of the previous test. With nu = 0 the components decouple."""
    mesh = box_mesh(6, 6, 6, jitter=0.0)
    p = mesh.points
    x0, x1 = p[:, 0].min(), p[:, 0].max()
    ends = np.flatnonzero(np.isclose(p[:, 0], x0) | np.isclose(p[:, 0], x1))
    g = np.zeros((mesh.n_nodes, 3))
    g[np.isclose(p[:, 0], x1), 0] = 0.01

    sol = solve_elasticity(
        mesh,
        body_force=0.0,
        material=IsotropicMaterial(E=1.0, nu=0.0),
        dirichlet_value=g,
        dirichlet_nodes=ends,
    )
    interior = np.setdiff1d(np.arange(mesh.n_nodes), ends)
    assert np.abs(sol.u[interior, 1:]).max() < 1e-9


def test_strain_energy_and_von_mises_are_sane():
    mesh = box_mesh(5, 5, 5, jitter=0.0)
    f = np.tile([0.0, 0.0, -1.0], (mesh.n_nodes, 1))
    sol = solve_elasticity(mesh, body_force=f, material=STEEL)
    assert sol.strain_energy() > 0.0
    vm = sol.von_mises()
    assert vm.shape == (mesh.n_cells,)
    assert vm.min() >= 0.0


def test_fully_constrained_problem_is_rejected():
    mesh = box_mesh(3, 3, 3, jitter=0.0)
    with pytest.raises(ValueError):
        solve_elasticity(mesh, dirichlet_nodes=np.arange(mesh.n_nodes))


def test_badly_shaped_inputs_are_rejected():
    mesh = box_mesh(4, 4, 4, jitter=0.0)
    with pytest.raises(ValueError, match="body force"):
        solve_elasticity(mesh, body_force=np.zeros((mesh.n_nodes, 2)))
    with pytest.raises(ValueError, match="boundary data"):
        solve_elasticity(mesh, dirichlet_value=np.zeros((mesh.n_nodes, 7)))


def test_bar_mesh_solves_and_returns_the_right_shape():
    mesh = bar_mesh(length=4.0, n_long=13, n_cross=4, jitter=0.1)
    f = np.tile([0.0, 0.0, -1.0], (mesh.n_nodes, 1))
    sol = solve_elasticity(mesh, body_force=f, material=STEEL)
    assert isinstance(sol, ElasticSolution)
    assert sol.u.shape == (mesh.n_nodes, 3)
    assert np.isfinite(sol.u).all()


# ------------------------------------------------------------------ manufactured case
def test_manufactured_force_has_transverse_components():
    """The displacement points only along x, but Poisson coupling drives y and z force.

    A manufactured source missing those terms would let a decoupled, wrong solver pass the
    convergence test below, so the coupling is asserted directly.
    """
    _, force = manufactured(STEEL)
    p = np.array([[0.3, 0.4, 0.5], [0.7, 0.2, 0.6], [0.25, 0.75, 0.35]])
    f = force(p)
    assert np.abs(f[:, 1]).max() > 1e-3, "no y component: the source is decoupled"
    assert np.abs(f[:, 2]).max() > 1e-3, "no z component: the source is decoupled"


def test_manufactured_displacement_vanishes_on_the_boundary():
    exact, _ = manufactured(STEEL)
    p = np.array([[0.0, 0.4, 0.5], [1.0, 0.2, 0.6], [0.3, 0.0, 0.35], [0.3, 0.5, 1.0]])
    assert np.abs(exact(p)).max() < 1e-12


@pytest.mark.slow
def test_manufactured_solution_converges_at_second_order():
    exact, force = manufactured(STEEL)
    errors, hs = [], []
    for n in (5, 9, 17):
        mesh = box_mesh(n, n, n, jitter=0.0)
        sol = solve_elasticity(mesh, body_force=force, material=STEEL, dirichlet_value=exact)
        errors.append(sol.l2_error(exact))
        hs.append(1.0 / (n - 1))
    rates = [
        float(np.log(errors[i] / errors[i + 1]) / np.log(hs[i] / hs[i + 1]))
        for i in range(len(errors) - 1)
    ]
    assert all(e > 0 for e in errors)
    assert max(rates) > 1.7, f"observed rates {rates}, expected approaching 2"
    assert min(rates) > 1.4, f"observed rates {rates}"
