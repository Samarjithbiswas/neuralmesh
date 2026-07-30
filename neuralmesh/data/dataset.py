"""Dataset generation: sample a parameter space, solve, cache.

Sampling uses Latin hypercube rather than uniform random draws. Every solve costs
real time, so for a fixed budget stratified sampling covers the parameter space far
more evenly, and the surrogate's validity region is exactly the region that was
sampled.

Normalisation statistics are computed on the training split only and then applied to
validation and test. Fitting them on everything leaks distributional information and
flatters the reported error, which is the most common silent mistake in surrogate
papers.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..fem.poisson import solve_poisson
from ..mesh.geometry import TriMesh, rectangle_mesh
from ..mesh.graph import MeshGraph, graph_from_solution


def latin_hypercube(n_samples: int, n_dims: int, *, seed: int | None = 0) -> np.ndarray:
    """Latin hypercube sample on the unit cube.

    Each dimension is split into ``n_samples`` equal-probability strata, one point is
    drawn per stratum, and the strata are independently permuted across dimensions.
    The result is a design where every one-dimensional projection is uniform by
    construction, which pure random sampling only achieves in expectation.
    """
    if n_samples < 1 or n_dims < 1:
        raise ValueError("n_samples and n_dims must both be >= 1")
    rng = np.random.default_rng(seed)
    cut = np.linspace(0.0, 1.0, n_samples + 1)
    lower, upper = cut[:-1], cut[1:]
    out = np.empty((n_samples, n_dims), dtype=np.float64)
    for d in range(n_dims):
        pts = lower + rng.random(n_samples) * (upper - lower)
        out[:, d] = rng.permutation(pts)
    return out


def scale_to_bounds(unit: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Map unit-cube samples onto ``bounds`` given as ``(n_dims, 2)``."""
    b = np.asarray(bounds, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError(f"bounds must be (n_dims, 2), got {b.shape}")
    if b.shape[0] != unit.shape[1]:
        raise ValueError("bounds and sample dimensionality disagree")
    if np.any(b[:, 1] <= b[:, 0]):
        raise ValueError("every bound must satisfy low < high")
    return b[:, 0] + unit * (b[:, 1] - b[:, 0])


# Parameter space for the diffusion benchmark.
#   0  source amplitude
#   1  source x-centre        (as a fraction of width)
#   2  source y-centre        (as a fraction of height)
#   3  source width           (as a fraction of the shorter side)
#   4  conductivity contrast  (log10 of the ratio between two material regions)
#   5  material split         (x-fraction where the two regions meet)
#   6  left boundary value
#   7  right boundary value
PARAM_BOUNDS = np.array(
    [
        [0.5, 4.0],
        [0.15, 0.85],
        [0.15, 0.85],
        [0.08, 0.35],
        [-1.0, 1.0],
        [0.25, 0.75],
        [-1.0, 1.0],
        [-1.0, 1.0],
    ]
)
PARAM_NAMES = (
    "source_amp",
    "source_cx",
    "source_cy",
    "source_width",
    "log_contrast",
    "split_x",
    "left_value",
    "right_value",
)


def make_case(mesh: TriMesh, params: np.ndarray):
    """Turn one parameter vector into a solved diffusion problem.

    A Gaussian source at a sampled location, a two-region conductivity field with a
    sampled contrast, and different Dirichlet values on the left and right edges. The
    left-right boundary asymmetry is what forces information to travel the length of
    the domain, which the under-reaching benchmark depends on.
    """
    p = np.asarray(params, dtype=np.float64)
    if p.shape != (len(PARAM_BOUNDS),):
        raise ValueError(f"expected {len(PARAM_BOUNDS)} parameters, got {p.shape}")
    amp, cx, cy, sw, log_k, split, left, right = p

    xy = mesh.points
    x0, x1 = xy[:, 0].min(), xy[:, 0].max()
    y0, y1 = xy[:, 1].min(), xy[:, 1].max()
    w, h = x1 - x0, y1 - y0
    short = min(w, h)

    centre = np.array([x0 + cx * w, y0 + cy * h])
    r2 = ((xy - centre) ** 2).sum(axis=1)
    sigma = max(sw * short, 1e-6)
    source = amp * np.exp(-r2 / (2.0 * sigma**2))

    cent_x = mesh.points[mesh.triangles][:, :, 0].mean(axis=1)
    ratio = 10.0**log_k
    conductivity = np.where(cent_x < x0 + split * w, 1.0, ratio)

    tol = 1e-9 * max(w, 1.0)
    on_left = np.isclose(xy[:, 0], x0, atol=tol)
    on_right = np.isclose(xy[:, 0], x1, atol=tol)

    # solve_poisson takes boundary data as a field over all nodes and reads off the
    # constrained ones, so build the full nodal array here. Top and bottom edges stay
    # at zero, which leaves the left-right difference as the only long-range driver.
    g_node = np.zeros(mesh.n_nodes, dtype=np.float64)
    g_node[on_left] = left
    g_node[on_right] = right

    return solve_poisson(
        mesh,
        source=source,
        conductivity=conductivity,
        dirichlet_nodes=mesh.boundary_nodes,
        dirichlet_value=g_node,
    )


@dataclass
class Normaliser:
    """Per-feature standardisation, fitted on training data only."""

    node_mean: np.ndarray
    node_std: np.ndarray
    edge_mean: np.ndarray
    edge_std: np.ndarray
    target_mean: float
    target_std: float

    @staticmethod
    def fit(graphs: list[MeshGraph], *, eps: float = 1e-8) -> Normaliser:
        nf = np.vstack([g.node_features for g in graphs])
        ef = np.vstack([g.edge_features for g in graphs])
        tg = np.concatenate([g.target for g in graphs if g.target is not None])
        return Normaliser(
            node_mean=nf.mean(axis=0),
            node_std=nf.std(axis=0) + eps,
            edge_mean=ef.mean(axis=0),
            edge_std=ef.std(axis=0) + eps,
            target_mean=float(tg.mean()),
            target_std=float(tg.std() + eps),
        )

    def apply(self, graph: MeshGraph) -> MeshGraph:
        return MeshGraph(
            node_features=(graph.node_features - self.node_mean) / self.node_std,
            edge_index=graph.edge_index,
            edge_features=(graph.edge_features - self.edge_mean) / self.edge_std,
            positions=graph.positions,
            target=(
                None
                if graph.target is None
                else (graph.target - self.target_mean) / self.target_std
            ),
        )

    def invert_target(self, y: np.ndarray) -> np.ndarray:
        return y * self.target_std + self.target_mean

    def to_dict(self) -> dict:
        return {
            "node_mean": self.node_mean.tolist(),
            "node_std": self.node_std.tolist(),
            "edge_mean": self.edge_mean.tolist(),
            "edge_std": self.edge_std.tolist(),
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }


@dataclass
class Dataset:
    """Train/validation/test split of solved graphs, plus fitted normalisation."""

    train: list[MeshGraph]
    val: list[MeshGraph]
    test: list[MeshGraph]
    normaliser: Normaliser
    params: np.ndarray
    meta: dict = field(default_factory=dict)

    def normalised(self) -> Dataset:
        return Dataset(
            train=[self.normaliser.apply(g) for g in self.train],
            val=[self.normaliser.apply(g) for g in self.val],
            test=[self.normaliser.apply(g) for g in self.test],
            normaliser=self.normaliser,
            params=self.params,
            meta={**self.meta, "normalised": True},
        )

    def summary(self) -> str:
        nodes = self.train[0].n_nodes if self.train else 0
        return (
            f"Dataset(train={len(self.train)}, val={len(self.val)}, "
            f"test={len(self.test)}, nodes_per_graph={nodes})"
        )


def generate_dataset(
    n_samples: int = 120,
    *,
    nx: int = 14,
    ny: int = 14,
    width: float = 1.0,
    height: float = 1.0,
    jitter: float = 0.22,
    splits: tuple[float, float] = (0.7, 0.15),
    seed: int = 0,
    vary_mesh: bool = True,
) -> Dataset:
    """Sample the parameter space, solve each case, and split.

    Parameters
    ----------
    vary_mesh:
        When true, each sample gets its own mesh seed, so the model never sees the
        same connectivity twice and cannot memorise a node ordering. This is the
        honest setting and the default.
    """
    unit = latin_hypercube(n_samples, len(PARAM_BOUNDS), seed=seed)
    params = scale_to_bounds(unit, PARAM_BOUNDS)

    graphs: list[MeshGraph] = []
    for i in range(n_samples):
        mesh = rectangle_mesh(
            nx,
            ny,
            width=width,
            height=height,
            jitter=jitter,
            seed=(seed + 1000 + i) if vary_mesh else seed,
        )
        graphs.append(graph_from_solution(make_case(mesh, params[i])))

    n_train = int(round(splits[0] * n_samples))
    n_val = int(round(splits[1] * n_samples))
    train = graphs[:n_train]
    val = graphs[n_train : n_train + n_val]
    test = graphs[n_train + n_val :]
    if not train or not val or not test:
        raise ValueError(f"splits {splits} on {n_samples} samples left an empty partition")

    return Dataset(
        train=train,
        val=val,
        test=test,
        normaliser=Normaliser.fit(train),
        params=params,
        meta={
            "n_samples": n_samples,
            "nx": nx,
            "ny": ny,
            "width": width,
            "height": height,
            "jitter": jitter,
            "seed": seed,
            "vary_mesh": vary_mesh,
            "param_names": list(PARAM_NAMES),
        },
    )


def dataset_fingerprint(**kwargs) -> str:
    """Stable short hash of generation arguments, used for cache filenames."""
    blob = json.dumps(kwargs, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def load_or_generate(cache_dir: str | Path = ".cache", **kwargs) -> Dataset:
    """Return a cached dataset if the arguments match, otherwise generate and cache."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"diffusion_{dataset_fingerprint(**kwargs)}.pkl"
    if path.exists():
        with path.open("rb") as fh:
            return pickle.load(fh)
    ds = generate_dataset(**kwargs)
    with path.open("wb") as fh:
        pickle.dump(ds, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return ds
