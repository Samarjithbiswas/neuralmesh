"""Tests for the verification suite itself.

A verification suite that always passes is worse than none, because it manufactures
confidence. So these tests check two separate things: that the suite passes on the real
solver, and that it would actually fail if the solver were wrong.

The second is the one that matters. ``test_suite_detects_a_broken_solver`` monkeypatches
a sign error into the stiffness assembly and asserts that the suite notices.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralmesh import verify as V


def test_fast_checks_all_pass():
    checks = V.run(full=False)
    failed = [c.name for c in checks if not c.passed]
    assert not failed, f"fast checks failed: {failed}"
    assert len(checks) == len(V.FAST)


@pytest.mark.slow
def test_full_suite_passes():
    checks = V.run(full=True)
    failed = [c.name for c in checks if not c.passed]
    assert not failed, f"checks failed: {failed}"
    assert len(checks) == len(V.FAST) + len(V.SLOW)


def test_every_check_declares_a_known_group():
    for c in V.run(full=False):
        assert c.group in V.GROUP_TITLE
        assert c.detail, f"{c.name} reported no detail"
        assert c.mark in {"PASS", "FAIL"}


@pytest.mark.slow
def test_independent_series_reference_is_tight():
    """The analytical comparison should agree far better than its own tolerance."""
    c = V.check_series_reference()
    assert c.passed
    # the printed detail carries the percentage; parse it back rather than trusting prose
    pct = float(c.detail.split(",")[1].strip().split("%")[0])
    assert pct < 0.5, f"series agreement degraded to {pct}%"


@pytest.mark.slow
def test_both_manufactured_solutions_give_second_order():
    for c in (V.check_convergence_trig(), V.check_convergence_poly()):
        assert c.passed, c.detail
        rates = [float(x) for x in c.detail.split("[")[1].split("]")[0].split(",")]
        assert all(1.7 < r < 2.4 for r in rates), c.detail


def test_suite_detects_a_broken_solver(monkeypatch):
    """Inject a sign error into the assembly and confirm the suite catches it.

    Without this test, a suite of thirteen passing checks proves only that the checks
    run. A dropped minus sign in the stiffness matrix is the classic FEM bug, and it is
    exactly the kind that leaves the solution looking plausible.
    """
    import neuralmesh.fem.poisson as P

    real = P.assemble_stiffness

    def broken(mesh, k_cell=1.0):
        return -real(mesh, k_cell)

    monkeypatch.setattr(P, "assemble_stiffness", broken)
    monkeypatch.setattr(V, "assemble_stiffness", broken)

    checks = V.run(full=False)
    assert any(not c.passed for c in checks), (
        "the suite passed a solver with a flipped stiffness sign, so it is not "
        "actually verifying anything"
    )


def test_report_returns_false_when_a_check_fails(capsys):
    fake = [
        V.Check("a good one", True, "fine", "algebraic"),
        V.Check("a bad one", False, "not fine", "consistency"),
    ]
    assert V.report(fake) is False
    out = capsys.readouterr().out
    assert "VERIFICATION FAILED" in out
    assert "a bad one" in out


def test_report_returns_true_when_all_pass(capsys):
    fake = [V.Check("ok", True, "fine", "algebraic")]
    assert V.report(fake) is True
    assert "solver verified" in capsys.readouterr().out


def test_mass_area_check_is_exact_not_approximate():
    """The domain-area check should be exact to round-off, not merely close."""
    c = V.check_mass_area()
    assert c.passed
    err = float(c.detail.split("error")[1].strip())
    assert err < 1e-12, f"mass integration error {err} is larger than round-off"


def test_conductivity_ratio_matches_theory():
    """For this linear problem the peak should scale as one over conductivity."""
    c = V.check_conductivity_monotone()
    assert c.passed
    ratio = float(c.detail.split("=")[1].split(",")[0].strip())
    assert np.isclose(ratio, 8.0, rtol=1e-3), (
        f"peak ratio {ratio} should equal the conductivity ratio 8 for a linear problem"
    )
