"""A verification suite for the finite element solver.

Run it with ``neuralmesh verify`` (add ``--full`` for the slower checks).

Every learned number in this package is measured against this solver, so the solver is
the thing that has to be right. Each check below has an answer known in advance from
theory, not a stored fixture, so a regression cannot be hidden by updating a golden
file. Checks are grouped by what kind of error they catch:

*Algebraic properties* catch sign errors, transposed indices and bad assembly. They are
cheap and they fail loudly. The stiffness matrix must be symmetric because the
underlying bilinear form is; it must annihilate constants because a constant field has
no gradient; the mass matrix must integrate to the exact domain area.

*Consistency checks* catch a solver that assembles correctly but solves the wrong
problem. Linearity, the maximum principle, and exact reproduction of fields that lie in
the discrete space.

*Convergence* is the only check that verifies the discretisation itself. Theory predicts
a specific number, and almost any bug in the stiffness matrix breaks the rate even when
the solution still looks plausible. Two independent manufactured solutions are used, so
a coincidence in one does not pass the suite.

*An independent reference* closes the loop: the solver is compared against a truncated
analytical series for a case where one exists, so agreement does not depend on any other
part of this codebase being correct.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fem.poisson import assemble_mass, assemble_stiffness, solve_poisson
from .mesh.geometry import unit_square_mesh


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    group: str

    @property
    def mark(self) -> str:
        return "PASS" if self.passed else "FAIL"


# --------------------------------------------------------------- algebraic properties
def check_symmetry() -> Check:
    K = assemble_stiffness(unit_square_mesh(11, jitter=0.15)).toarray()
    err = float(np.abs(K - K.T).max())
    return Check(
        "stiffness matrix is symmetric",
        err < 1e-12,
        f"max |K - K^T| = {err:.2e}",
        "algebraic",
    )


def check_constant_nullspace() -> Check:
    K = assemble_stiffness(unit_square_mesh(13, jitter=0.2))
    err = float(np.abs(K @ np.ones(K.shape[0])).max())
    return Check(
        "stiffness annihilates constants",
        err < 1e-10,
        f"max |K 1| = {err:.2e}, a constant field has zero gradient",
        "algebraic",
    )


def check_mass_area() -> Check:
    from .mesh.geometry import rectangle_mesh

    mesh = rectangle_mesh(12, 9, width=3.0, height=2.0, jitter=0.0)
    total = float(assemble_mass(mesh).sum())
    err = abs(total - 6.0)
    return Check(
        "mass matrix integrates to the domain area",
        err < 1e-10,
        f"sum M = {total:.10f} against exact 6.0, error {err:.2e}",
        "algebraic",
    )


def check_mass_positive_definite() -> Check:
    M = assemble_mass(unit_square_mesh(9, jitter=0.1)).toarray()
    lo = float(np.linalg.eigvalsh(M).min())
    return Check(
        "mass matrix is positive definite",
        lo > 0.0,
        f"smallest eigenvalue = {lo:.3e}",
        "algebraic",
    )


def check_stiffness_semidefinite() -> Check:
    K = assemble_stiffness(unit_square_mesh(9, jitter=0.1)).toarray()
    eig = np.linalg.eigvalsh(K)
    return Check(
        "stiffness is positive semidefinite",
        float(eig.min()) > -1e-10,
        f"smallest eigenvalue = {eig.min():.3e}, one zero mode for the constant",
        "algebraic",
    )


# ------------------------------------------------------------------ consistency checks
def check_patch_test() -> Check:
    """A P1 space contains every linear function, so a linear field must be exact."""
    mesh = unit_square_mesh(12, jitter=0.25, seed=3)

    def exact(p):
        return 2.0 * p[:, 0] - 3.0 * p[:, 1] + 1.0

    err = solve_poisson(mesh, source=0.0, dirichlet_value=exact).max_error(exact)
    return Check(
        "patch test: linear field reproduced exactly",
        err < 1e-11,
        f"max error {err:.2e}, expected round-off not discretisation error",
        "consistency",
    )


def check_linearity() -> Check:
    mesh = unit_square_mesh(11, jitter=0.15)
    a = solve_poisson(mesh, source=1.0, dirichlet_value=0.0)
    b = solve_poisson(mesh, source=3.0, dirichlet_value=0.0)
    err = float(np.abs(3.0 * a.u - b.u).max())
    return Check(
        "solution scales linearly with the source",
        err < 1e-9,
        f"max |3 u(f) - u(3f)| = {err:.2e}",
        "consistency",
    )


def check_maximum_principle() -> Check:
    mesh = unit_square_mesh(13, jitter=0.1)
    sol = solve_poisson(mesh, source=1.0, dirichlet_value=0.0)
    interior = np.setdiff1d(np.arange(mesh.n_nodes), mesh.boundary_nodes)
    lo = float(sol.u[interior].min())
    return Check(
        "maximum principle: positive source lifts the interior",
        lo > 0.0,
        f"minimum interior value = {lo:.5f}, must be strictly positive",
        "consistency",
    )


def check_boundary_imposed() -> Check:
    mesh = unit_square_mesh(11, jitter=0.12)

    def g(p):
        return np.sin(2.0 * p[:, 0]) + 0.5 * p[:, 1]

    sol = solve_poisson(mesh, source=2.0, dirichlet_value=g)
    b = mesh.boundary_nodes
    err = float(np.abs(sol.u[b] - g(mesh.points)[b]).max())
    return Check(
        "Dirichlet data is imposed exactly",
        err < 1e-12,
        f"max boundary error {err:.2e}",
        "consistency",
    )


def check_conductivity_monotone() -> Check:
    mesh = unit_square_mesh(11, jitter=0.0)
    soft = solve_poisson(mesh, source=1.0, conductivity=1.0, dirichlet_value=0.0)
    hard = solve_poisson(mesh, source=1.0, conductivity=8.0, dirichlet_value=0.0)
    lo, hi = float(hard.u.max()), float(soft.u.max())
    # Guard the division. A genuinely broken solver can return a zero or negative peak,
    # and dividing by it emits a NaN warning that obscures the failure it should report.
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(lo) < 1e-300:
        return Check(
            "higher conductivity reduces the response",
            False,
            f"degenerate peaks: soft {hi:.3e}, hard {lo:.3e}",
            "consistency",
        )
    ratio = hi / lo
    return Check(
        "higher conductivity reduces the response",
        lo < hi,
        f"peak ratio soft/hard = {ratio:.3f}, expected near the conductivity ratio 8",
        "consistency",
    )


# ------------------------------------------------------------------------- convergence
def _rate(exact, source, sizes=(9, 17, 33)) -> tuple[list[float], list[float]]:
    errors, h = [], []
    for n in sizes:
        mesh = unit_square_mesh(n, jitter=0.0)
        sol = solve_poisson(mesh, source=source, dirichlet_value=exact)
        errors.append(sol.l2_error(exact))
        h.append(1.0 / (n - 1))
    rates = [
        float(np.log(errors[i] / errors[i + 1]) / np.log(h[i] / h[i + 1]))
        for i in range(len(errors) - 1)
    ]
    return errors, rates


def check_convergence_trig(verbose: bool = False) -> Check:
    r"""Manufactured solution :math:`\sin(\pi x)\sin(\pi y)`, source :math:`2\pi^2 u`."""

    def exact(p):
        return np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])

    def source(p):
        return 2.0 * np.pi**2 * exact(p)

    errors, rates = _rate(exact, source)
    if verbose:
        for h, e in zip((1 / 8, 1 / 16, 1 / 32), errors):
            print(f"      h = {h:.4f}   L2 error {e:.4e}")
    ok = all(1.7 < r < 2.4 for r in rates)
    return Check(
        "second-order convergence, trigonometric solution",
        ok,
        f"rates {[round(r, 3) for r in rates]} against theory 2",
        "convergence",
    )


def check_convergence_poly(verbose: bool = False) -> Check:
    r"""A second manufactured solution, so one lucky rate cannot pass the suite.

    :math:`u = x(1-x)y(1-y)`, which vanishes on the boundary, with the exact source
    :math:`f = 2[y(1-y) + x(1-x)]`.
    """

    def exact(p):
        x, y = p[:, 0], p[:, 1]
        return x * (1.0 - x) * y * (1.0 - y)

    def source(p):
        x, y = p[:, 0], p[:, 1]
        return 2.0 * (y * (1.0 - y) + x * (1.0 - x))

    errors, rates = _rate(exact, source)
    if verbose:
        for h, e in zip((1 / 8, 1 / 16, 1 / 32), errors):
            print(f"      h = {h:.4f}   L2 error {e:.4e}")
    ok = all(1.7 < r < 2.4 for r in rates)
    return Check(
        "second-order convergence, polynomial solution",
        ok,
        f"rates {[round(r, 3) for r in rates]} against theory 2",
        "convergence",
    )


# --------------------------------------------------------------- independent reference
def check_series_reference() -> Check:
    r"""Compare against a truncated analytical series, computed independently.

    For :math:`-\nabla^2 u = 1` on the unit square with :math:`u = 0` on the boundary,
    separation of variables gives

    .. math::
        u(x,y) = \sum_{m,n \text{ odd}}
        \frac{16}{\pi^4 m n (m^2 + n^2)} \sin(m\pi x)\sin(n\pi y)

    This is the strongest check in the suite, because it does not rely on any other part
    of this codebase being correct. The series is evaluated here in closed form and the
    solver never sees it.
    """
    mesh = unit_square_mesh(41, jitter=0.0)
    sol = solve_poisson(mesh, source=1.0, dirichlet_value=0.0)

    x, y = mesh.points[:, 0], mesh.points[:, 1]
    u = np.zeros_like(x)
    for m in range(1, 80, 2):
        for n in range(1, 80, 2):
            coeff = 16.0 / (np.pi**4 * m * n * (m * m + n * n))
            u += coeff * np.sin(m * np.pi * x) * np.sin(n * np.pi * y)

    peak = float(u.max())
    err = float(np.abs(sol.u - u).max())
    rel = err / peak
    return Check(
        "matches an independent analytical series",
        rel < 5e-3,
        f"max |FEM - series| = {err:.3e}, {100 * rel:.3f}% of the peak {peak:.6f}",
        "reference",
    )


# ------------------------------------------------------------------ 3D nonlinear solver
def check_3d_volume() -> Check:
    from .fem.mesh3d import box_mesh

    m = box_mesh(6, 5, 4, lx=2.0, ly=1.5, lz=0.5, jitter=0.3, seed=1)
    got = float(m.volumes().sum())
    err = abs(got - 1.5)
    return Check(
        "3D mesh volume is exact under interior jitter",
        err < 1e-12,
        f"sum of tet volumes {got:.12f} against exact 1.5, error {err:.2e}",
        "nonlinear3d",
    )


def check_3d_orientation() -> Check:
    from .fem.mesh3d import box_mesh

    lo = float(box_mesh(6, 6, 6, jitter=0.35, seed=2).signed_volumes().min())
    return Check(
        "every tetrahedron is positively oriented",
        lo > 0.0,
        f"smallest signed volume {lo:.3e}, a negative one flips every element integral",
        "nonlinear3d",
    )


def check_3d_patch_test() -> Check:
    from .fem.mesh3d import box_mesh
    from .fem.nonlinear3d import PowerLawConductivity, solve_nonlinear

    m = box_mesh(6, 6, 6, jitter=0.25, seed=3)

    def exact(p):
        return 1.0 + 2.0 * p[:, 0] - 0.5 * p[:, 1] + 0.25 * p[:, 2]

    sol = solve_nonlinear(
        m,
        source=0.0,
        law=PowerLawConductivity(alpha=0.0),
        dirichlet_value=exact,
        tol=1e-12,
    )
    err = sol.max_error(exact)
    return Check(
        "3D patch test: linear field reproduced exactly",
        sol.converged and err < 1e-10,
        f"max error {err:.2e} on a jittered tetrahedral mesh",
        "nonlinear3d",
    )


def check_3d_nonlinearity_is_real() -> Check:
    from .fem.mesh3d import box_mesh
    from .fem.nonlinear3d import PowerLawConductivity, solve_nonlinear

    m = box_mesh(5, 5, 5, jitter=0.0)
    law = PowerLawConductivity(k0=1.0, alpha=5.0)
    a = solve_nonlinear(m, source=20.0, law=law, dirichlet_value=0.0)
    b = solve_nonlinear(m, source=40.0, law=law, dirichlet_value=0.0)
    dev = float(np.abs(2.0 * a.u - b.u).max() / max(np.abs(b.u).max(), 1e-300))
    return Check(
        "the problem is genuinely nonlinear",
        dev > 0.05,
        f"superposition violated by {100 * dev:.1f}%, so u(2f) is not 2u(f)",
        "nonlinear3d",
    )


def check_3d_tangent() -> Check:
    """Finite-difference the residual to confirm the analytic Jacobian is correct."""
    from .fem.mesh3d import box_mesh
    from .fem.nonlinear3d import PowerLawConductivity, residual_and_tangent

    m = box_mesh(4, 4, 4, jitter=0.15, seed=7)
    law = PowerLawConductivity(k0=1.3, alpha=2.0)
    rng = np.random.default_rng(1)
    u = 0.4 * rng.normal(size=m.n_nodes)
    load = 0.1 * rng.normal(size=m.n_nodes)

    r0, J = residual_and_tangent(m, u, law, load)
    A = J.toarray()
    eps = 1e-7
    worst = 0.0
    for j in rng.choice(m.n_nodes, size=min(10, m.n_nodes), replace=False):
        up = u.copy()
        up[j] += eps
        rp, _ = residual_and_tangent(m, up, law, load)
        fd = (rp - r0) / eps
        scale = max(float(np.abs(A[:, j]).max()), 1.0)
        worst = max(worst, float(np.abs(fd - A[:, j]).max()) / scale)
    return Check(
        "Newton tangent matches finite differences",
        worst < 2e-4,
        f"worst relative column error {worst:.2e}; a dropped dk/du term shows up here",
        "nonlinear3d",
    )


def check_3d_newton_order(verbose: bool = False) -> Check:
    from .fem.mesh3d import box_mesh
    from .fem.nonlinear3d import PowerLawConductivity, manufactured, solve_nonlinear

    law = PowerLawConductivity(k0=1.0, alpha=1.0)
    exact, src = manufactured(law)
    sol = solve_nonlinear(
        box_mesh(7, 7, 7, jitter=0.0),
        source=src,
        law=law,
        dirichlet_value=exact,
        tol=1e-12,
        verbose=verbose,
    )
    orders = sol.history.convergence_orders()
    ok = bool(orders) and max(orders) > 1.7
    return Check(
        "Newton converges quadratically",
        sol.converged and ok,
        f"observed orders {[round(o, 2) for o in orders]} in "
        f"{sol.history.iterations} iterations; 2 means the tangent is consistent",
        "nonlinear3d",
    )


def check_3d_convergence(verbose: bool = False) -> Check:
    from .fem.mesh3d import box_mesh
    from .fem.nonlinear3d import PowerLawConductivity, manufactured, solve_nonlinear

    law = PowerLawConductivity(k0=1.0, alpha=1.0)
    exact, src = manufactured(law)
    errors, hs = [], []
    for n in (5, 9, 17):
        m = box_mesh(n, n, n, jitter=0.0)
        sol = solve_nonlinear(m, source=src, law=law, dirichlet_value=exact, tol=1e-11)
        errors.append(sol.l2_error(exact))
        hs.append(1.0 / (n - 1))
        if verbose:
            print(f"      h = {hs[-1]:.4f}  nodes {m.n_nodes:6d}  L2 = {errors[-1]:.4e}")
    rates = [
        float(np.log(errors[i] / errors[i + 1]) / np.log(hs[i] / hs[i + 1]))
        for i in range(len(errors) - 1)
    ]
    return Check(
        "3D nonlinear problem converges at second order",
        bool(rates) and max(rates) > 1.8 and min(rates) > 1.5,
        f"rates {[round(r, 3) for r in rates]} against theory 2, manufactured solution",
        "nonlinear3d",
    )


FAST = [
    check_symmetry,
    check_constant_nullspace,
    check_mass_area,
    check_mass_positive_definite,
    check_stiffness_semidefinite,
    check_patch_test,
    check_linearity,
    check_maximum_principle,
    check_boundary_imposed,
    check_conductivity_monotone,
    check_3d_volume,
    check_3d_orientation,
    check_3d_patch_test,
    check_3d_nonlinearity_is_real,
    check_3d_tangent,
]

SLOW = [
    check_convergence_trig,
    check_convergence_poly,
    check_series_reference,
    check_3d_newton_order,
    check_3d_convergence,
]

GROUP_TITLE = {
    "algebraic": "Algebraic properties  (catch sign, index and assembly errors)",
    "consistency": "Consistency  (catch solving the wrong problem correctly)",
    "convergence": "Convergence  (verify the discretisation against theory)",
    "reference": "Independent reference  (does not rely on this codebase)",
    "nonlinear3d": "3D nonlinear solver  (tetrahedra, k(u), Newton-Raphson)",
}


def run(full: bool = True, verbose: bool = False) -> list[Check]:
    checks: list[Check] = [f() for f in FAST]
    if full:
        for f in SLOW:
            try:
                checks.append(f(verbose=verbose))  # type: ignore[call-arg]
            except TypeError:
                checks.append(f())
    return checks


def report(checks: list[Check], width: int = 78) -> bool:
    print("=" * width)
    print("neuralmesh finite element verification")
    print("=" * width)

    for group in ("algebraic", "consistency", "convergence", "reference", "nonlinear3d"):
        rows = [c for c in checks if c.group == group]
        if not rows:
            continue
        print(f"\n{GROUP_TITLE[group]}")
        print("-" * width)
        for c in rows:
            print(f"  {c.mark}  {c.name}")
            print(f"        {c.detail}")

    passed = sum(c.passed for c in checks)
    total = len(checks)
    print("\n" + "=" * width)
    if passed == total:
        print(f"solver verified: {passed}/{total} checks passed")
    else:
        print(f"VERIFICATION FAILED: {passed}/{total} passed")
        for c in checks:
            if not c.passed:
                print(f"  failed: {c.name}  ({c.detail})")
    print("=" * width)
    return passed == total
