"""Unstructured triangular mesh generation.

Meshes are Delaunay triangulations of scattered points, which gives genuinely
unstructured connectivity rather than a reindexed grid. That matters here: a graph
network trained on structured connectivity learns the grid, not the physics, and the
resolution-generalisation experiment in :mod:`neuralmesh.evaluate` would be vacuous.

Every mesh carries the boundary information the FEM solvers need, computed from the
triangulation itself (an edge on exactly one triangle is a boundary edge) rather than
from the generating geometry, so it stays correct for domains with holes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import Delaunay


@dataclass
class TriMesh:
    """A 2D triangular mesh.

    Attributes
    ----------
    points:
        ``(n_nodes, 2)`` vertex coordinates.
    triangles:
        ``(n_cells, 3)`` vertex indices, counter-clockwise.
    boundary_nodes:
        Sorted indices of nodes lying on the domain boundary.
    """

    points: np.ndarray
    triangles: np.ndarray
    boundary_nodes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    def __post_init__(self) -> None:
        self.points = np.ascontiguousarray(self.points, dtype=np.float64)
        self.triangles = np.ascontiguousarray(self.triangles, dtype=np.int64)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError(f"points must be (n, 2), got {self.points.shape}")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError(f"triangles must be (m, 3), got {self.triangles.shape}")
        if self.triangles.size and self.triangles.max() >= len(self.points):
            raise ValueError("triangle references a node index that does not exist")
        self._orient_ccw()
        if self.boundary_nodes.size == 0:
            self.boundary_nodes = boundary_nodes_of(self.triangles)

    @property
    def n_nodes(self) -> int:
        return len(self.points)

    @property
    def n_cells(self) -> int:
        return len(self.triangles)

    @property
    def interior_nodes(self) -> np.ndarray:
        mask = np.ones(self.n_nodes, dtype=bool)
        mask[self.boundary_nodes] = False
        return np.flatnonzero(mask)

    def _orient_ccw(self) -> None:
        """Flip any clockwise triangle so signed areas are positive.

        The FEM assembly divides by the signed area determinant; a clockwise cell
        would contribute a negative-definite block and silently corrupt the solve.
        """
        p = self.points[self.triangles]
        det = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (
            p[:, 2, 0] - p[:, 0, 0]
        ) * (p[:, 1, 1] - p[:, 0, 1])
        flip = det < 0.0
        if np.any(flip):
            self.triangles[flip] = self.triangles[flip][:, [0, 2, 1]]

    def cell_areas(self) -> np.ndarray:
        p = self.points[self.triangles]
        det = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (
            p[:, 2, 0] - p[:, 0, 0]
        ) * (p[:, 1, 1] - p[:, 0, 1])
        return 0.5 * det

    def area(self) -> float:
        return float(self.cell_areas().sum())

    def characteristic_length(self) -> float:
        """Representative element size ``h``, used for convergence-rate checks."""
        return float(np.sqrt(2.0 * self.cell_areas().mean()))

    def edges(self) -> np.ndarray:
        """Unique undirected edges as a sorted ``(n_edges, 2)`` array."""
        t = self.triangles
        raw = np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
        raw = np.sort(raw, axis=1)
        return np.unique(raw, axis=0)

    def summary(self) -> str:
        return (
            f"TriMesh(nodes={self.n_nodes}, cells={self.n_cells}, "
            f"edges={len(self.edges())}, boundary={len(self.boundary_nodes)}, "
            f"h={self.characteristic_length():.4f})"
        )


def boundary_nodes_of(triangles: np.ndarray) -> np.ndarray:
    """Nodes on the boundary, found from edge-to-cell incidence.

    An interior edge is shared by exactly two triangles; a boundary edge by one.
    This is topological, so it handles holes and concave domains without knowing
    anything about the geometry that produced the mesh.
    """
    t = np.asarray(triangles, dtype=np.int64)
    raw = np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    raw = np.sort(raw, axis=1)
    uniq, counts = np.unique(raw, axis=0, return_counts=True)
    border = uniq[counts == 1]
    return np.unique(border.reshape(-1))


def _lattice_with_jitter(
    nx: int, ny: int, width: float, height: float, jitter: float, rng: np.random.Generator
) -> np.ndarray:
    """Perturbed lattice interior points plus an exact boundary loop.

    Boundary points are left unjittered so the domain outline stays a clean
    rectangle; interior jitter is what makes the connectivity irregular.
    """
    xs = np.linspace(0.0, width, nx)
    ys = np.linspace(0.0, height, ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel()])

    on_edge = (
        np.isclose(pts[:, 0], 0.0)
        | np.isclose(pts[:, 0], width)
        | np.isclose(pts[:, 1], 0.0)
        | np.isclose(pts[:, 1], height)
    )
    hx = width / max(nx - 1, 1)
    hy = height / max(ny - 1, 1)
    amp = jitter * np.array([hx, hy])
    noise = rng.uniform(-1.0, 1.0, size=pts.shape) * amp
    pts[~on_edge] += noise[~on_edge]
    return pts


def unit_square_mesh(
    n: int = 12,
    *,
    jitter: float = 0.25,
    seed: int | None = 0,
) -> TriMesh:
    """Unstructured mesh of the unit square with ``n`` points per side."""
    return rectangle_mesh(n, n, width=1.0, height=1.0, jitter=jitter, seed=seed)


def rectangle_mesh(
    nx: int = 12,
    ny: int = 12,
    *,
    width: float = 1.0,
    height: float = 1.0,
    jitter: float = 0.25,
    seed: int | None = 0,
) -> TriMesh:
    """Unstructured mesh of an axis-aligned rectangle.

    Parameters
    ----------
    nx, ny:
        Points along x and y before jitter. Must both be at least 2.
    jitter:
        Interior perturbation as a fraction of nominal spacing. 0 gives a
        structured triangulation; values above about 0.35 risk sliver cells.
    """
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must both be >= 2")
    if not 0.0 <= jitter < 0.5:
        raise ValueError("jitter must be in [0, 0.5)")
    rng = np.random.default_rng(seed)
    pts = _lattice_with_jitter(nx, ny, width, height, jitter, rng)
    tri = Delaunay(pts)
    return TriMesh(points=pts, triangles=tri.simplices)


def strip_mesh(
    length: float = 8.0,
    height: float = 1.0,
    *,
    nx: int = 40,
    ny: int = 6,
    jitter: float = 0.2,
    seed: int | None = 0,
) -> TriMesh:
    """A long thin domain, used for the under-reaching experiment.

    Graph diameter grows with ``nx``, so information from one short edge needs many
    message-passing hops to reach the other. That is precisely the regime where a
    fixed-depth MeshGraphNet under-reaches and global attention does not.
    """
    return rectangle_mesh(nx, ny, width=length, height=height, jitter=jitter, seed=seed)


def annulus_mesh(
    r_inner: float = 0.3,
    r_outer: float = 1.0,
    *,
    n_radial: int = 8,
    n_theta: int = 40,
    jitter: float = 0.15,
    seed: int | None = 0,
) -> TriMesh:
    """Mesh of an annulus, giving a domain with an interior hole.

    Included because a hole exercises the topological boundary detection and
    produces a non-convex graph, which a lattice never does. Delaunay of an
    annular point set fills the hole, so cells whose centroid falls inside the
    inner radius are removed afterwards.
    """
    if not 0.0 < r_inner < r_outer:
        raise ValueError("require 0 < r_inner < r_outer")
    rng = np.random.default_rng(seed)
    radii = np.linspace(r_inner, r_outer, n_radial)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    rr, tt = np.meshgrid(radii, thetas, indexing="ij")
    rr = rr.ravel()
    tt = tt.ravel()

    on_ring = np.isclose(rr, r_inner) | np.isclose(rr, r_outer)
    dr = (r_outer - r_inner) / max(n_radial - 1, 1)
    dt = 2.0 * np.pi / n_theta
    rr[~on_ring] += rng.uniform(-1.0, 1.0, size=(~on_ring).sum()) * jitter * dr
    tt[~on_ring] += rng.uniform(-1.0, 1.0, size=(~on_ring).sum()) * jitter * dt

    pts = np.column_stack([rr * np.cos(tt), rr * np.sin(tt)])
    tri = Delaunay(pts)
    cent = pts[tri.simplices].mean(axis=1)
    keep = np.linalg.norm(cent, axis=1) > r_inner
    return TriMesh(points=pts, triangles=tri.simplices[keep])


def refine(mesh: TriMesh) -> TriMesh:
    """Uniform 1-to-4 refinement by edge bisection.

    Each triangle becomes four similar triangles sharing new midside nodes. Node
    positions of the parent mesh are preserved and appear first in the refined
    point array, which lets :mod:`neuralmesh.evaluate` compare a coarse-trained
    model against a fine solve at coincident points.
    """
    pts = mesh.points
    tris = mesh.triangles

    edges = mesh.edges()
    mid_index: dict[tuple[int, int], int] = {}
    mids = np.empty((len(edges), 2), dtype=np.float64)
    for k, (a, b) in enumerate(edges):
        mid_index[(int(a), int(b))] = len(pts) + k
        mids[k] = 0.5 * (pts[a] + pts[b])

    def mid(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        return mid_index[key]

    new_pts = np.vstack([pts, mids])
    new_tris = np.empty((4 * len(tris), 3), dtype=np.int64)
    for i, (a, b, c) in enumerate(tris):
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        new_tris[4 * i + 0] = (a, ab, ca)
        new_tris[4 * i + 1] = (ab, b, bc)
        new_tris[4 * i + 2] = (ca, bc, c)
        new_tris[4 * i + 3] = (ab, bc, ca)
    return TriMesh(points=new_pts, triangles=new_tris)
