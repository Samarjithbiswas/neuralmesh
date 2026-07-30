"""Tests for the 3D graph construction and nonlinear dataset.

The one to read is ``test_leakage_report_would_catch_a_real_leak``. Everything else
confirms the pipeline is wired correctly; that test confirms the guard against the
mistake that would silently invalidate the whole 3D benchmark.

In the linear 2D problem the conductivity is data, so handing it to the network is fair.
In the nonlinear problem k depends on u, so a node feature carrying k(u) carries u. Every
metric would improve and the benchmark would be measuring the network's ability to invert
a monotone function. Asserting that we do not do this is worth little; demonstrating that
the check would fire if we did is worth something.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralmesh.data.dataset import Normaliser
from neuralmesh.data.dataset3d import (
    PARAM_BOUNDS_3D,
    bar_dataset_3d,
    make_case_3d,
)
from neuralmesh.fem.mesh3d import bar_mesh, box_mesh
from neuralmesh.mesh.graph import MeshGraph, graph_diameter
from neuralmesh.mesh.graph3d import (
    EDGE_DIM_3D,
    NODE_DIM_3D,
    build_graph_3d,
    graph_from_solution_3d,
    leakage_report,
)


@pytest.fixture(scope="module")
def solved():
    mesh = bar_mesh(length=4.0, n_long=13, n_cross=4, jitter=0.1, seed=0)
    params = PARAM_BOUNDS_3D.mean(axis=1)
    return make_case_3d(mesh, params)


@pytest.fixture(scope="module")
def graph(solved):
    return graph_from_solution_3d(solved)


# ------------------------------------------------------------------------ structure
def test_feature_dimensions(graph):
    assert graph.node_features.shape[1] == NODE_DIM_3D == 5
    assert graph.edge_features.shape[1] == EDGE_DIM_3D == 4
    assert graph.positions.shape[1] == 3


def test_edges_are_symmetric(graph):
    pairs = set(zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()))
    assert all((b, a) in pairs for a, b in pairs)


def test_no_self_loops(graph):
    assert not (graph.edge_index[0] == graph.edge_index[1]).any()


def test_edge_length_feature_matches_the_displacement(graph):
    d = graph.edge_features[:, :3]
    length = graph.edge_features[:, 3]
    assert np.allclose(np.linalg.norm(d, axis=1), length)


def test_boundary_indicator_matches_the_mesh(solved, graph):
    flag = graph.node_features[:, 0]
    assert set(np.unique(flag)).issubset({0.0, 1.0})
    assert np.array_equal(np.flatnonzero(flag == 1.0), solved.mesh.boundary_nodes)


def test_target_equals_the_solve(solved, graph):
    assert graph.target is not None
    assert np.allclose(graph.target, solved.u)


def test_boundary_values_are_carried_and_interior_stays_zero(solved, graph):
    b = solved.mesh.boundary_nodes
    interior = np.setdiff1d(np.arange(solved.mesh.n_nodes), b)
    assert np.allclose(graph.node_features[b, 1], solved.dirichlet_values[b])
    assert np.allclose(graph.node_features[interior, 1], 0.0)


def test_material_features_are_the_inputs_not_the_state(solved, graph):
    """k0 and alpha, which a designer specifies, rather than k(u), which encodes u."""
    assert np.allclose(graph.node_features[:, 3], solved.law.k0)
    assert np.allclose(graph.node_features[:, 4], solved.law.alpha)


def test_three_boundary_data_shapes_agree():
    mesh = box_mesh(4, 4, 4, jitter=0.0)
    src = np.ones(mesh.n_nodes)
    b = mesh.boundary_nodes
    a = build_graph_3d(mesh, source=src, dirichlet_values=1.5)
    c = build_graph_3d(mesh, source=src, dirichlet_values=np.full(b.shape, 1.5))
    d = build_graph_3d(mesh, source=src, dirichlet_values=np.full(mesh.n_nodes, 1.5))
    assert np.allclose(a.node_features[:, 1], c.node_features[:, 1])
    assert np.allclose(a.node_features[:, 1], d.node_features[:, 1])


def test_bad_shapes_are_rejected():
    mesh = box_mesh(4, 4, 4, jitter=0.0)
    with pytest.raises(ValueError, match="dirichlet_values"):
        build_graph_3d(mesh, source=np.ones(mesh.n_nodes), dirichlet_values=np.zeros(3))
    with pytest.raises(ValueError, match="source"):
        build_graph_3d(mesh, source=np.ones(3))


def test_diameter_grows_with_aspect_ratio():
    short = bar_mesh(length=3.0, n_long=10, n_cross=4, jitter=0.0)
    long_ = bar_mesh(length=12.0, n_long=37, n_cross=4, jitter=0.0)
    gs = build_graph_3d(short, source=np.zeros(short.n_nodes))
    gl = build_graph_3d(long_, source=np.zeros(long_.n_nodes))
    assert graph_diameter(gl) > 2 * graph_diameter(gs)


# -------------------------------------------------------------------------- leakage
def test_leakage_report_is_clean_on_the_real_features(graph):
    rep = leakage_report(graph)
    assert rep["conductivity_like"] < 0.05
    assert rep["is_boundary"] < 0.05


def test_leakage_report_would_catch_a_real_leak(solved, graph):
    """Deliberately inject k(u) as a node feature and confirm the check fires.

    Without this, ``leakage_report`` returning small numbers proves only that it runs.
    """
    leaked = graph.node_features.copy()
    k_of_u = solved.law.k(solved.u)  # this is the mistake being guarded against
    leaked[:, 3] = k_of_u

    poisoned = MeshGraph(
        node_features=leaked,
        edge_index=graph.edge_index,
        edge_features=graph.edge_features,
        positions=graph.positions,
        target=graph.target,
    )
    rep = leakage_report(poisoned)
    assert rep["conductivity_like"] > 0.5, (
        f"a deliberately leaked k(u) scored only {rep['conductivity_like']:.3f}; "
        "the guard is not sensitive enough to be worth trusting"
    )
    # and the honest version must score far lower on the same graph
    assert leakage_report(graph)["conductivity_like"] < 0.05


def test_leakage_report_needs_a_target():
    mesh = box_mesh(4, 4, 4, jitter=0.0)
    g = build_graph_3d(mesh, source=np.ones(mesh.n_nodes))
    with pytest.raises(ValueError):
        leakage_report(g)


# -------------------------------------------------------------------------- dataset
def test_make_case_3d_validates_its_parameters():
    mesh = box_mesh(4, 4, 4, jitter=0.0)
    with pytest.raises(ValueError):
        make_case_3d(mesh, np.zeros(3))


def test_make_case_3d_imposes_the_end_faces():
    mesh = bar_mesh(length=4.0, n_long=13, n_cross=4, jitter=0.0)
    p = PARAM_BOUNDS_3D.mean(axis=1).copy()
    p[5], p[6] = -0.8, 0.9  # left and right face values
    sol = make_case_3d(mesh, p)
    x = mesh.points[:, 0]
    left = np.isclose(x, x.min())
    right = np.isclose(x, x.max())
    assert np.allclose(sol.u[left], -0.8, atol=1e-10)
    assert np.allclose(sol.u[right], 0.9, atol=1e-10)


@pytest.fixture(scope="module")
def small3d():
    return bar_dataset_3d(12, aspect_ratio=4.0, n_cross=4, seed=0)


def test_dataset_splits_are_populated(small3d):
    assert len(small3d.train) and len(small3d.val) and len(small3d.test)


def test_every_case_has_its_own_mesh(small3d):
    coords = [g.positions.tobytes() for g in small3d.train]
    assert len(set(coords)) == len(coords), "meshes were reused across samples"


def test_targets_vary_between_cases(small3d):
    spread = float(np.std([g.target.mean() for g in small3d.train]))
    assert spread > 1e-6


def test_normaliser_is_fitted_on_training_data_only(small3d):
    train_only = Normaliser.fit(small3d.train)
    everything = Normaliser.fit(small3d.train + small3d.val + small3d.test)
    assert not np.isclose(train_only.target_mean, everything.target_mean, rtol=1e-12)
    assert np.allclose(train_only.node_mean, small3d.normaliser.node_mean)


def test_normalised_training_features_are_standardised(small3d):
    stacked = np.vstack([g.node_features for g in small3d.normalised().train])
    assert np.allclose(stacked.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(stacked.std(axis=0), 1.0, atol=1e-6)


def test_dataset_is_reproducible(small3d):
    again = bar_dataset_3d(12, aspect_ratio=4.0, n_cross=4, seed=0)
    assert np.allclose(small3d.train[0].target, again.train[0].target)


def test_too_few_samples_is_rejected():
    with pytest.raises(ValueError):
        bar_dataset_3d(2, aspect_ratio=4.0, n_cross=4, seed=0)


@pytest.mark.slow
def test_3d_study_runs_end_to_end():
    from neuralmesh.evaluate.underreach3d import run_underreach_study_3d

    r = run_underreach_study_3d(
        aspect_ratio=4.0,
        n_samples=12,
        epochs=4,
        shallow_blocks=2,
        deep_blocks=4,
        hidden_dim=16,
        verbose=False,
    )
    assert len(r.results) == 4
    assert r.graph_diameter > 0
    assert r.n_edges > 0, "n_edges was reported as zero, which was a real bug once"
    for a in r.results:
        assert np.isfinite(a.test_rel_l2)
        assert len(a.rel_l2_by_band) == 4
    labels = {a.label: a for a in r.results}
    trans = next(v for k, v in labels.items() if "Transformer" in k)
    shallow = next(v for k, v in labels.items() if k.startswith("MeshGraphNet L=2"))
    assert trans.receptive_hops == -1 and shallow.receptive_hops == 2
    assert abs(trans.n_parameters - shallow.n_parameters) / shallow.n_parameters < 0.25
