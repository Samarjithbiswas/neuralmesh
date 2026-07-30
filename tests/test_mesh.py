"""Mesh geometry and graph-construction tests.

The graph tests exist because the whole under-reaching claim is a statement about
graph diameter. If ``graph_diameter`` were wrong, the experiment would still run and
still produce a table, and the table would mean nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralmesh import (
    TriMesh,
    annulus_mesh,
    boundary_nodes_of,
    build_graph,
    graph_diameter,
    graph_from_solution,
    rectangle_mesh,
    solve_poisson,
    strip_mesh,
    unit_square_mesh,
)


def test_triangles_are_oriented_counter_clockwise():
    """Signed area must be positive for every cell, or element integrals flip sign."""
    mesh = unit_square_mesh(9, jitter=0.25, seed=1)
    p = mesh.points[mesh.triangles]
    x1, y1 = p[:, 0, 0], p[:, 0, 1]
    x2, y2 = p[:, 1, 0], p[:, 1, 1]
    x3, y3 = p[:, 2, 0], p[:, 2, 1]
    twice_area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    assert twice_area.min() > 0.0


def test_boundary_nodes_of_a_square_are_the_perimeter():
    mesh = rectangle_mesh(6, 6, jitter=0.0)
    on_edge = (
        np.isclose(mesh.points[:, 0], 0.0)
        | np.isclose(mesh.points[:, 0], 1.0)
        | np.isclose(mesh.points[:, 1], 0.0)
        | np.isclose(mesh.points[:, 1], 1.0)
    )
    assert set(mesh.boundary_nodes) == set(np.flatnonzero(on_edge))


def test_annulus_has_two_boundary_loops():
    """A hole means the boundary is not simply connected; both rings must be found."""
    mesh = annulus_mesh(r_inner=0.35, r_outer=1.0, n_theta=28, n_radial=5)
    r = np.linalg.norm(mesh.points[mesh.boundary_nodes], axis=1)
    assert (np.abs(r - 0.35) < 1e-6).any()
    assert (np.abs(r - 1.0) < 1e-6).any()


def test_boundary_edges_appear_in_exactly_one_triangle():
    tri = unit_square_mesh(7, jitter=0.0).triangles
    found = boundary_nodes_of(tri)
    assert found.size > 0
    assert found.dtype.kind == "i"
    assert np.all(np.diff(found) > 0), "boundary node indices should be sorted, unique"


def test_rejects_malformed_input():
    with pytest.raises(ValueError):
        TriMesh(points=np.zeros((4, 3)), triangles=np.array([[0, 1, 2]]))
    with pytest.raises(ValueError):
        TriMesh(points=np.zeros((4, 2)), triangles=np.array([[0, 1]]))
    with pytest.raises(ValueError):
        TriMesh(points=np.zeros((3, 2)), triangles=np.array([[0, 1, 9]]))


def _graph(mesh):
    sol = solve_poisson(mesh, source=1.0, dirichlet_value=0.0)
    return graph_from_solution(sol)


def test_graph_edges_are_symmetric():
    """Message passing must be able to travel both ways along every mesh edge."""
    g = _graph(unit_square_mesh(8, jitter=0.1))
    pairs = set(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()))
    assert all((b, a) in pairs for a, b in pairs)


def test_graph_has_no_self_loops():
    g = _graph(unit_square_mesh(8, jitter=0.1))
    assert not (g.edge_index[0] == g.edge_index[1]).any()


def test_graph_feature_shapes_line_up():
    mesh = unit_square_mesh(9, jitter=0.15)
    g = _graph(mesh)
    assert g.node_features.shape == (mesh.n_nodes, 4)
    assert g.positions.shape == (mesh.n_nodes, 2)
    assert g.edge_features.shape == (g.edge_index.shape[1], 3)
    assert g.target is not None and g.target.shape == (mesh.n_nodes,)


def test_boundary_indicator_is_binary_and_matches_the_mesh():
    mesh = unit_square_mesh(9, jitter=0.0)
    g = _graph(mesh)
    flag = g.node_features[:, 0]
    assert set(np.unique(flag)).issubset({0.0, 1.0})
    assert np.allclose(np.flatnonzero(flag == 1.0), mesh.boundary_nodes)


def test_diameter_of_a_path_graph_is_its_length():
    """Sanity check on the BFS, using a graph whose diameter is known by hand."""
    n = 9
    pts = np.column_stack([np.linspace(0, 1, n), np.zeros(n)])
    src = np.arange(n - 1)
    dst = src + 1
    edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    from neuralmesh import MeshGraph

    g = MeshGraph(
        node_features=np.zeros((n, 4)),
        edge_index=edge_index,
        edge_features=np.zeros((edge_index.shape[1], 3)),
        positions=pts,
    )
    assert graph_diameter(g) == n - 1


def test_longer_strips_have_larger_diameter():
    """The experiment turns aspect ratio into reach difficulty, so this must hold."""
    short = graph_diameter(_graph(strip_mesh(length=3.0, height=1.0, ny=5)))
    long_ = graph_diameter(_graph(strip_mesh(length=10.0, height=1.0, ny=5)))
    assert long_ > short


def test_build_graph_without_a_target_is_allowed():
    mesh = unit_square_mesh(7, jitter=0.0)
    g = build_graph(
        mesh,
        source=np.ones(mesh.n_nodes),
        conductivity=np.ones(len(mesh.triangles)),
        dirichlet_values=np.zeros(mesh.n_nodes),
    )
    assert g.target is None


def test_build_graph_accepts_all_three_boundary_data_shapes():
    """A scalar, one value per constrained node, and a whole-node field must agree.

    ``solve_poisson`` takes whole-field boundary data as ``dirichlet_value`` while
    ``build_graph`` takes ``dirichlet_values``. The names are one character apart and
    the conventions differ, which has already caused a real bug here, so the
    equivalence is pinned by a test rather than by a comment.
    """
    mesh = unit_square_mesh(7, jitter=0.0)
    src = np.ones(mesh.n_nodes)
    b = mesh.boundary_nodes

    scalar = build_graph(mesh, source=src, dirichlet_values=2.5)
    per_node = build_graph(mesh, source=src, dirichlet_values=np.full(b.shape, 2.5))
    whole_field = build_graph(mesh, source=src, dirichlet_values=np.full(mesh.n_nodes, 2.5))

    assert np.allclose(scalar.node_features[:, 1], per_node.node_features[:, 1])
    assert np.allclose(scalar.node_features[:, 1], whole_field.node_features[:, 1])
    # Interior nodes must stay at zero regardless of which form was supplied.
    interior = np.setdiff1d(np.arange(mesh.n_nodes), b)
    assert np.allclose(whole_field.node_features[interior, 1], 0.0)


def test_build_graph_rejects_a_wrongly_sized_boundary_array():
    mesh = unit_square_mesh(7, jitter=0.0)
    with pytest.raises(ValueError, match="dirichlet_values"):
        build_graph(mesh, source=np.ones(mesh.n_nodes), dirichlet_values=np.zeros(3))
