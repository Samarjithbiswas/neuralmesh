"""Tetrahedral meshes in three dimensions.

Why 3D at all: the 2D scalar study in this repository isolates under-reaching cleanly,
and that is exactly why it is a weak headline. Reviewers of learned-simulation work
reasonably ask whether a result survives a problem with real dimensionality and real
nonlinearity. This module is the first half of answering that; :mod:`neuralmesh.fem.
nonlinear3d` is the second.

Design notes that matter downstream:

*Kuhn subdivision, six tetrahedra per cube.* It is the standard subdivision that tiles
without introducing hanging nodes, and it produces tetrahedra of equal volume on a
uniform grid, which keeps element quality predictable when jitter is added.

*Shape-function gradients are constant per element.* For P1 on a tetrahedron the
gradient of the solution is constant inside the element, so assembly needs one 3x3
inverse per element and no quadrature loop. That is what makes a pure NumPy solver fast
enough to generate thousands of training cases.

*Orientation is fixed at construction.* A tetrahedron with a negative signed volume has
its nodes in a left-handed order, and every element integral built from it carries the
wrong sign. Rather than take absolute values later and hide the problem, the connectivity
is corrected once here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# The six tetrahedra of a Kuhn subdivision of the unit cube, in terms of the eight
# corner indices ordered as (i, j, k) with k fastest.
_CUBE_TETS = np.array(
    [
        [0, 1, 3, 7],
        [0, 1, 7, 5],
        [0, 5, 7, 4],
        [0, 3, 2, 7],
        [0, 6, 4, 7],
        [0, 2, 6, 7],
    ],
    dtype=np.int64,
)


@dataclass
class TetMesh:
    """A 3D tetrahedral mesh.

    Attributes
    ----------
    points:
        ``(n_nodes, 3)`` vertex coordinates.
    tets:
        ``(n_cells, 4)`` vertex indices, oriented to positive signed volume.
    boundary_nodes:
        Sorted indices of nodes on the domain boundary.
    """

    points: np.ndarray
    tets: np.ndarray
    boundary_nodes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    def __post_init__(self) -> None:
        self.points = np.ascontiguousarray(self.points, dtype=np.float64)
        self.tets = np.ascontiguousarray(self.tets, dtype=np.int64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must be (n, 3), got {self.points.shape}")
        if self.tets.ndim != 2 or self.tets.shape[1] != 4:
            raise ValueError(f"tets must be (m, 4), got {self.tets.shape}")
        if self.tets.size and self.tets.max() >= len(self.points):
            raise ValueError("a tetrahedron references a node index that does not exist")
        self._orient()
        if self.boundary_nodes.size == 0:
            self.boundary_nodes = boundary_nodes_of(self.tets)

    @property
    def n_nodes(self) -> int:
        return len(self.points)

    @property
    def n_cells(self) -> int:
        return len(self.tets)

    def _orient(self) -> None:
        """Swap two nodes wherever the signed volume is negative."""
        vol = self.signed_volumes()
        bad = vol < 0.0
        if bad.any():
            self.tets[bad] = self.tets[bad][:, [0, 2, 1, 3]]

    def jacobians(self) -> np.ndarray:
        """``(n_cells, 3, 3)`` edge matrices ``[p1-p0, p2-p0, p3-p0]`` as columns."""
        p = self.points[self.tets]
        return np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=-1)

    def signed_volumes(self) -> np.ndarray:
        return np.linalg.det(self.jacobians()) / 6.0

    def volumes(self) -> np.ndarray:
        return np.abs(self.signed_volumes())

    def shape_gradients(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Constant per-element gradients of the four P1 basis functions.

        Returns ``(grads, volumes)`` where ``grads`` is ``(n_cells, 4, 3)``.

        With reference coordinates :math:`(\xi,\eta,\zeta)` and
        :math:`N_1=\xi, N_2=\eta, N_3=\zeta, N_0=1-\xi-\eta-\zeta`, the chain rule gives
        :math:`\nabla N_i = J^{-\mathsf T} \hat\nabla N_i`, so the gradients of
        :math:`N_1,N_2,N_3` are the rows of :math:`J^{-1}` and
        :math:`\nabla N_0 = -\sum_{i>0}\nabla N_i`.
        """
        J = self.jacobians()
        Jinv = np.linalg.inv(J)
        grads = np.empty((self.n_cells, 4, 3), dtype=np.float64)
        # rows of J^{-1} are the physical gradients of N1, N2, N3
        grads[:, 1:, :] = Jinv
        grads[:, 0, :] = -Jinv.sum(axis=1)
        return grads, self.volumes()

    def cell_centroids(self) -> np.ndarray:
        return self.points[self.tets].mean(axis=1)

    def edges(self) -> np.ndarray:
        """``(n_undirected_edges, 2)`` unique node pairs, each row sorted ascending."""
        t = self.tets
        pairs = np.vstack(
            [
                t[:, [0, 1]],
                t[:, [0, 2]],
                t[:, [0, 3]],
                t[:, [1, 2]],
                t[:, [1, 3]],
                t[:, [2, 3]],
            ]
        )
        pairs = np.sort(pairs, axis=1)
        return np.unique(pairs, axis=0)

    def quality(self) -> np.ndarray:
        r"""Radius ratio, normalised so a regular tetrahedron scores 1.

        Defined as :math:`3 r_{\text{in}} / r_{\text{circ}}`. Values near zero mean
        sliver elements, which wreck the condition number long before they change the
        mesh statistics anyone usually reports.
        """
        p = self.points[self.tets]
        vol = self.volumes()
        # face areas via cross products of the four triangular faces
        faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        area = np.zeros(self.n_cells)
        for a, b, c in faces:
            area += 0.5 * np.linalg.norm(
                np.cross(p[:, b] - p[:, a], p[:, c] - p[:, a]), axis=1
            )
        r_in = 3.0 * vol / np.maximum(area, 1e-300)

        # circumradius from the standard determinant formula
        a2 = np.linalg.norm(p[:, 1] - p[:, 0], axis=1) ** 2
        b2 = np.linalg.norm(p[:, 2] - p[:, 0], axis=1) ** 2
        c2 = np.linalg.norm(p[:, 3] - p[:, 0], axis=1) ** 2
        rhs = np.stack([a2, b2, c2], axis=-1)[:, :, None]
        J = self.jacobians()
        centre = 0.5 * np.linalg.solve(np.swapaxes(J, 1, 2), rhs)[:, :, 0]
        r_circ = np.linalg.norm(centre, axis=1)
        return 3.0 * r_in / np.maximum(r_circ, 1e-300)

    def summary(self) -> str:
        q = self.quality()
        return (
            f"TetMesh(nodes={self.n_nodes}, tets={self.n_cells}, "
            f"boundary={len(self.boundary_nodes)}, "
            f"volume={self.volumes().sum():.6f}, "
            f"quality min={q.min():.3f} mean={q.mean():.3f})"
        )


def boundary_nodes_of(tets: np.ndarray) -> np.ndarray:
    """Nodes on faces that belong to exactly one tetrahedron.

    Topological rather than geometric, so it works on any domain shape rather than only
    on boxes where the faces can be found by coordinate comparison.
    """
    t = np.asarray(tets, dtype=np.int64)
    faces = np.vstack([t[:, [0, 1, 2]], t[:, [0, 1, 3]], t[:, [0, 2, 3]], t[:, [1, 2, 3]]])
    keys = np.sort(faces, axis=1)
    uniq, counts = np.unique(keys, axis=0, return_counts=True)
    return np.unique(uniq[counts == 1]).astype(np.int64)


def box_mesh(
    nx: int = 7,
    ny: int = 7,
    nz: int = 7,
    *,
    lx: float = 1.0,
    ly: float = 1.0,
    lz: float = 1.0,
    jitter: float = 0.0,
    seed: int | None = 0,
) -> TetMesh:
    """Structured tetrahedral mesh of a box, six tetrahedra per cube.

    ``jitter`` perturbs interior nodes by that fraction of the local spacing. Boundary
    nodes are never moved, so the domain stays exactly the requested box and Dirichlet
    data can still be imposed by coordinate.
    """
    if min(nx, ny, nz) < 2:
        raise ValueError("each direction needs at least 2 nodes")

    xs = np.linspace(0.0, lx, nx)
    ys = np.linspace(0.0, ly, ny)
    zs = np.linspace(0.0, lz, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        h = np.array([lx / (nx - 1), ly / (ny - 1), lz / (nz - 1)])
        interior = (
            (pts[:, 0] > 1e-12)
            & (pts[:, 0] < lx - 1e-12)
            & (pts[:, 1] > 1e-12)
            & (pts[:, 1] < ly - 1e-12)
            & (pts[:, 2] > 1e-12)
            & (pts[:, 2] < lz - 1e-12)
        )
        noise = (rng.random((int(interior.sum()), 3)) - 0.5) * jitter * h
        pts[interior] += noise

    def nid(i, j, k):
        return (i * ny + j) * nz + k

    cells = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                corners = np.array(
                    [
                        nid(i, j, k),
                        nid(i, j, k + 1),
                        nid(i, j + 1, k),
                        nid(i, j + 1, k + 1),
                        nid(i + 1, j, k),
                        nid(i + 1, j, k + 1),
                        nid(i + 1, j + 1, k),
                        nid(i + 1, j + 1, k + 1),
                    ],
                    dtype=np.int64,
                )
                cells.append(corners[_CUBE_TETS])
    tets = np.vstack(cells)
    return TetMesh(points=pts, tets=tets)


def bar_mesh(
    length: float = 8.0,
    *,
    n_long: int = 33,
    n_cross: int = 4,
    jitter: float = 0.15,
    seed: int | None = 0,
) -> TetMesh:
    """A long thin bar, which is the 3D analogue of the 2D strip.

    Aspect ratio drives graph diameter up while node count stays affordable, which is
    what makes under-reaching measurable rather than merely arguable.
    """
    return box_mesh(
        n_long, n_cross, n_cross, lx=length, ly=1.0, lz=1.0, jitter=jitter, seed=seed
    )
