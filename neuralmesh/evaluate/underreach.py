"""The under-reaching experiment.

Claim under test
----------------
A MeshGraphNet with :math:`L` message-passing blocks has a receptive field of exactly
:math:`L` hops. For an elliptic problem the solution at an interior node depends on
*every* boundary value, so when the graph diameter :math:`D` exceeds :math:`L` the
architecture is structurally unable to represent the solution, no matter how long it
trains or how wide it is. This is *under-reaching*.

Interleaving global attention removes the dependence of receptive field on depth, so
the same :math:`L` should recover most of the loss.

Experimental design
-------------------
Long thin strip domains, where diameter grows with aspect ratio while node count stays
manageable. The only long-range driver is the difference between the left and right
Dirichlet values, so a model that cannot propagate information along the strip cannot
get the interior right.

Three controls make the result mean something:

1. **Parameter count is matched.** Attention adds parameters, so the transformer's
   width is reduced until its parameter count is within a few percent of the baseline.
   Otherwise the experiment measures capacity, not reach.
2. **A no-communication control is included.** ``node_mlp`` cannot see any other node.
   Any error level it also reaches is not evidence about message passing.
3. **Error is reported against distance from the driven boundary.** The prediction of
   under-reaching is not merely worse error, it is worse error *in the middle*, where
   hop distance exceeds the receptive field. A uniform degradation would be a
   different phenomenon.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..data.dataset import (
    PARAM_BOUNDS,
    Dataset,
    Normaliser,
    latin_hypercube,
    make_case,
    scale_to_bounds,
)
from ..mesh.geometry import rectangle_mesh
from ..mesh.graph import graph_diameter, graph_from_solution
from ..models.architectures import (
    ModelConfig,
    build_model,
    count_parameters,
    match_capacity,
)
from ..train.trainer import TensorGraph, TrainConfig, evaluate, train_model


@dataclass
class ArchResult:
    name: str
    label: str
    n_parameters: int
    hidden_dim: int
    n_blocks: int
    receptive_hops: int
    test_mse: float
    test_rel_l2: float
    rel_l2_by_band: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UnderReachResult:
    aspect_ratio: float
    n_nodes: int
    graph_diameter: int
    n_train: int
    epochs: int
    band_edges: list[float]
    results: list[ArchResult]

    def to_dict(self) -> dict:
        return {
            "aspect_ratio": self.aspect_ratio,
            "n_nodes": self.n_nodes,
            "graph_diameter": self.graph_diameter,
            "n_train": self.n_train,
            "epochs": self.epochs,
            "band_edges": self.band_edges,
            "results": [r.to_dict() for r in self.results],
        }

    def table(self) -> str:
        head = (
            f"aspect {self.aspect_ratio:g}  nodes {self.n_nodes}  "
            f"diameter {self.graph_diameter}\n"
            f"{'architecture':<26}{'params':>9}{'hops':>6}{'relL2':>9}   by distance band\n"
            f"{'-' * 78}"
        )
        rows = []
        for r in self.results:
            hops = "global" if r.receptive_hops < 0 else str(r.receptive_hops)
            bands = "  ".join(f"{b:.3f}" for b in r.rel_l2_by_band)
            rows.append(
                f"{r.label:<26}{r.n_parameters:>9,}{hops:>6}{r.test_rel_l2:>9.4f}   {bands}"
            )
        return head + "\n" + "\n".join(rows)


def strip_dataset(
    n_samples: int,
    *,
    aspect_ratio: float = 8.0,
    ny: int = 6,
    jitter: float = 0.2,
    seed: int = 0,
    splits: tuple[float, float] = (0.7, 0.15),
) -> Dataset:
    """Diffusion cases on strip meshes of the given aspect ratio."""
    nx = int(round(aspect_ratio * (ny - 1))) + 1
    unit = latin_hypercube(n_samples, len(PARAM_BOUNDS), seed=seed)
    params = scale_to_bounds(unit, PARAM_BOUNDS)

    graphs = []
    for i in range(n_samples):
        mesh = rectangle_mesh(
            nx, ny, width=aspect_ratio, height=1.0, jitter=jitter, seed=seed + 500 + i
        )
        graphs.append(graph_from_solution(make_case(mesh, params[i])))

    n_train = int(round(splits[0] * n_samples))
    n_val = max(int(round(splits[1] * n_samples)), 1)
    train, val, test = (
        graphs[:n_train],
        graphs[n_train : n_train + n_val],
        graphs[n_train + n_val :],
    )
    if not test:
        raise ValueError("no test samples left; increase n_samples")
    return Dataset(
        train=train,
        val=val,
        test=test,
        normaliser=Normaliser.fit(train),
        params=params,
        meta={"aspect_ratio": aspect_ratio, "nx": nx, "ny": ny},
    )


def _rel_l2_by_band(model, graphs: list[TensorGraph], band_edges: np.ndarray) -> list[float]:
    """Relative L2 within bands of normalised distance from the nearest short edge.

    Band 0 is nearest the driven boundaries, the last band is the middle of the strip.
    Under-reaching should show as error growing across the bands.
    """
    n_bands = len(band_edges) - 1
    num = np.zeros(n_bands)
    den = np.zeros(n_bands)

    model.eval()
    with torch.no_grad():
        for g in graphs:
            pred = g.forward(model).cpu().numpy()
            targ = g.target.cpu().numpy()
            interior = g.interior.cpu().numpy()
            x = g.positions[:, 0].cpu().numpy()

            span = x.max() - x.min()
            if span <= 0:
                continue
            # normalised distance from nearest short edge, 0 at an edge, 0.5 mid-strip
            frac = np.minimum(x - x.min(), x.max() - x) / span

            for b in range(n_bands):
                sel = interior & (frac >= band_edges[b]) & (frac < band_edges[b + 1])
                if not sel.any():
                    continue
                num[b] += float(((pred[sel] - targ[sel]) ** 2).sum())
                den[b] += float((targ[sel] ** 2).sum())

    out = []
    for b in range(n_bands):
        out.append(float(np.sqrt(num[b] / den[b])) if den[b] > 0 else float("nan"))
    return out


def run_underreach_study(
    *,
    aspect_ratio: float = 8.0,
    n_samples: int = 60,
    ny: int = 6,
    epochs: int = 80,
    shallow_blocks: int = 4,
    deep_blocks: int = 16,
    hidden_dim: int = 64,
    seed: int = 0,
    n_bands: int = 4,
    verbose: bool = True,
) -> UnderReachResult:
    """Train the architectures on a strip domain and compare reach-limited error.

    Configurations compared:

    * ``node_mlp`` -- no communication, the control
    * ``meshgraphnet`` with ``shallow_blocks`` -- receptive field smaller than diameter
    * ``meshgraphnet`` with ``deep_blocks`` -- receptive field closer to diameter
    * ``mesh_graph_transformer`` with ``shallow_blocks`` -- shallow but global,
      parameter-matched to the shallow baseline
    """
    ds = strip_dataset(n_samples, aspect_ratio=aspect_ratio, ny=ny, seed=seed).normalised()
    diameter = graph_diameter(ds.train[0])
    n_nodes = ds.train[0].n_nodes
    test = [TensorGraph(g) for g in ds.test]
    band_edges = np.linspace(0.0, 0.5, n_bands + 1)

    shallow_cfg = ModelConfig(hidden_dim=hidden_dim, n_blocks=shallow_blocks)
    baseline_params = count_parameters(build_model("meshgraphnet", shallow_cfg))
    tcfg = match_capacity("mesh_graph_transformer", baseline_params, shallow_cfg)

    plan = [
        ("node_mlp", "node MLP (no comms)", shallow_cfg),
        ("meshgraphnet", f"MeshGraphNet L={shallow_blocks}", shallow_cfg),
        (
            "meshgraphnet",
            f"MeshGraphNet L={deep_blocks}",
            ModelConfig(hidden_dim=hidden_dim, n_blocks=deep_blocks),
        ),
        ("mesh_graph_transformer", f"MGN-Transformer L={shallow_blocks}", tcfg),
    ]

    if verbose:
        print(
            f"strip aspect {aspect_ratio:g}: {n_nodes} nodes, graph diameter {diameter}, "
            f"{len(ds.train)} train / {len(ds.test)} test"
        )

    results: list[ArchResult] = []
    for name, label, cfg in plan:
        model = build_model(name, cfg)
        if verbose:
            print(f"\n{label}  ({count_parameters(model):,} params)")
        model, _ = train_model(
            model,
            ds.train,
            ds.val,
            TrainConfig(epochs=epochs, seed=seed, log_every=max(epochs // 3, 1)),
            verbose=verbose,
        )
        mse, rel = evaluate(model, test)
        results.append(
            ArchResult(
                name=name,
                label=label,
                n_parameters=count_parameters(model),
                hidden_dim=cfg.hidden_dim,
                n_blocks=cfg.n_blocks,
                receptive_hops=model.receptive_hops,
                test_mse=mse,
                test_rel_l2=rel,
                rel_l2_by_band=_rel_l2_by_band(model, test, band_edges),
            )
        )

    return UnderReachResult(
        aspect_ratio=aspect_ratio,
        n_nodes=n_nodes,
        graph_diameter=diameter,
        n_train=len(ds.train),
        epochs=epochs,
        band_edges=band_edges.tolist(),
        results=results,
    )


def save_result(result: UnderReachResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path
