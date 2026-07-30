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
]

SLOW = [
    check_convergence_trig,
    check_convergence_poly,
    check_series_reference,
]

GROUP_TITLE = {
    "algebraic": "Algebraic properties  (catch sign, index and assembly errors)",
    "consistency": "Consistency  (catch solving the wrong problem correctly)",
    "convergence": "Convergence  (verify the discretisation against theory)",
    "reference": "Independent reference  (does not rely on this codebase)",
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

    for group in ("algebraic", "consistency", "convergence", "reference"):
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
