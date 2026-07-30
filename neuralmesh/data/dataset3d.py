r"""Dataset generation for the 3D nonlinear benchmark.

Each case is a solve of :math:`-\nabla\cdot(k(u)\nabla u) = f` on a long thin bar, with

* a Gaussian source at a sampled position and width,
* a sampled conductivity :math:`k_0` and nonlinearity strength :math:`\alpha`,
* different Dirichlet values on the two end faces.

The end-face asymmetry is the whole point. The other four faces are held at zero, so the
only driver that has to travel the length of the bar is the difference between the two
ends. A model that cannot propagate information that far is wrong in the middle, and
wrong there specifically rather than uniformly.

Sampling is Latin hypercube for the same reason as in 2D: every solve costs real time, so
a stratified design covers the parameter space far more evenly than random draws at the
same budget.

Normalisation statistics are fitted on the training split only, and the mesh is reseeded
per sample so no model can memorise a node ordering. Both are enforced by tests rather
than left to discipline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..fem.mesh3d import TetMesh, bar_mesh
from ..fem.nonlinear3d import PowerLawConductivity, solve_nonlinear
from ..mesh.graph import MeshGraph
from ..mesh.graph3d import graph_from_solution_3d
from .dataset import Normaliser, latin_hypercube, scale_to_bounds

# 0  source amplitude
# 1  source centre, as a fraction of bar length
# 2  source width, as a fraction of the cross-section
# 3  log10 of k0
# 4  alpha, the nonlinearity strength
# 5  value on the x = 0 face
# 6  value on the x = L face
PARAM_BOUNDS_3D = np.array(
    [
        [2.0, 12.0],
        [0.20, 0.80],
        [0.30, 1.20],
        [-0.4, 0.4],
        [0.0, 3.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
    ]
)
PARAM_NAMES_3D = (
    "source_amp",
    "source_cx",
    "source_width",
    "log10_k0",
    "alpha",
    "left_value",
    "right_value",
)


def make_case_3d(mesh: TetMesh, params: np.ndarray):
    """Turn one parameter vector into a solved nonlinear problem."""
    p = np.asarray(params, dtype=np.float64)
    if p.shape != (len(PARAM_BOUNDS_3D),):
        raise ValueError(f"expected {len(PARAM_BOUNDS_3D)} parameters, got {p.shape}")
    amp, cx, width, log_k0, alpha, left, right = p

    xyz = mesh.points
    x0, x1 = xyz[:, 0].min(), xyz[:, 0].max()
    length = x1 - x0
    cross = max(xyz[:, 1].max() - xyz[:, 1].min(), xyz[:, 2].max() - xyz[:, 2].min(), 1e-12)

    centre = np.array([x0 + cx * length, xyz[:, 1].mean(), xyz[:, 2].mean()], dtype=np.float64)
    sigma = max(width * cross, 1e-6)
    r2 = ((xyz - centre) ** 2).sum(axis=1)
    source = amp * np.exp(-r2 / (2.0 * sigma**2))

    tol = 1e-9 * max(length, 1.0)
    on_left = np.isclose(xyz[:, 0], x0, atol=tol)
    on_right = np.isclose(xyz[:, 0], x1, atol=tol)

    g = np.zeros(mesh.n_nodes, dtype=np.float64)
    g[on_left] = left
    g[on_right] = right

    return solve_nonlinear(
        mesh,
        source=source,
        law=PowerLawConductivity(k0=float(10.0**log_k0), alpha=float(alpha)),
        dirichlet_value=g,
        dirichlet_nodes=mesh.boundary_nodes,
        tol=1e-9,
        max_iter=40,
    )


@dataclass
class Dataset3D:
    """Train, validation and test split of solved 3D graphs."""

    train: list[MeshGraph]
    val: list[MeshGraph]
    test: list[MeshGraph]
    normaliser: Normaliser
    params: np.ndarray
    meta: dict = field(default_factory=dict)

    def normalised(self) -> Dataset3D:
        return Dataset3D(
            train=[self.normaliser.apply(g) for g in self.train],
            val=[self.normaliser.apply(g) for g in self.val],
            test=[self.normaliser.apply(g) for g in self.test],
            normaliser=self.normaliser,
            params=self.params,
            meta={**self.meta, "normalised": True},
        )

    def summary(self) -> str:
        n = self.train[0].n_nodes if self.train else 0
        return (
            f"Dataset3D(train={len(self.train)}, val={len(self.val)}, "
            f"test={len(self.test)}, nodes_per_graph={n})"
        )


def bar_dataset_3d(
    n_samples: int = 40,
    *,
    aspect_ratio: float = 8.0,
    n_cross: int = 4,
    jitter: float = 0.15,
    seed: int = 0,
    splits: tuple[float, float] = (0.7, 0.15),
    verbose: bool = False,
) -> Dataset3D:
    """Solve ``n_samples`` nonlinear cases on bars of the given aspect ratio.

    ``n_long`` is chosen so element size is roughly isotropic, which keeps graph diameter
    proportional to aspect ratio without producing sliver tetrahedra.
    """
    n_long = int(round(aspect_ratio * (n_cross - 1))) + 1
    unit = latin_hypercube(n_samples, len(PARAM_BOUNDS_3D), seed=seed)
    params = scale_to_bounds(unit, PARAM_BOUNDS_3D)

    graphs: list[MeshGraph] = []
    n_failed = 0
    for i in range(n_samples):
        mesh = bar_mesh(
            length=aspect_ratio,
            n_long=n_long,
            n_cross=n_cross,
            jitter=jitter,
            seed=seed + 5000 + i,
        )
        sol = make_case_3d(mesh, params[i])
        if not sol.converged:
            # A non-converged solve is not ground truth. Dropping it is honest; keeping it
            # would quietly poison the labels with whatever Newton happened to reach.
            n_failed += 1
            continue
        graphs.append(graph_from_solution_3d(sol))
        if verbose and (i + 1) % 10 == 0:
            print(f"  solved {i + 1}/{n_samples}")

    if n_failed:
        print(f"  warning: dropped {n_failed} case(s) where Newton did not converge")
    if len(graphs) < 3:
        raise ValueError(f"only {len(graphs)} usable cases; increase n_samples")

    n_train = int(round(splits[0] * len(graphs)))
    n_val = max(int(round(splits[1] * len(graphs))), 1)
    train, val, test = (
        graphs[:n_train],
        graphs[n_train : n_train + n_val],
        graphs[n_train + n_val :],
    )
    if not train or not val or not test:
        raise ValueError(f"splits {splits} on {len(graphs)} cases left a partition empty")

    return Dataset3D(
        train=train,
        val=val,
        test=test,
        normaliser=Normaliser.fit(train),
        params=params,
        meta={
            "aspect_ratio": aspect_ratio,
            "n_long": n_long,
            "n_cross": n_cross,
            "jitter": jitter,
            "seed": seed,
            "n_failed": n_failed,
            "param_names": list(PARAM_NAMES_3D),
        },
    )
