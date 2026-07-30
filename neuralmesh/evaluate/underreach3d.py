r"""The under-reaching experiment, on the 3D nonlinear problem.

This is the test that matters. The 2D result is on scalar steady diffusion, which is a
deliberately clean setting, and the obvious objection is that the finding is an artefact
of that cleanliness. Here the same protocol runs on
:math:`-\nabla\cdot(k(u)\nabla u) = f` on tetrahedra, where the operator is nonlinear and
the geometry is three dimensional.

The protocol is unchanged on purpose, because changing it alongside the problem would
make any difference in outcome uninterpretable:

1. parameter counts matched across architectures, so the comparison measures design
   rather than capacity
2. a no-communication control included, so any error level it also reaches is not
   evidence about message passing
3. error resolved by distance from the driven ends, because the prediction of
   under-reaching is worse error *in the middle*, not uniformly worse

The result is allowed to come out negative. If the distance-resolved signature does not
survive nonlinearity, that is the finding and it gets reported as such.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..data.dataset3d import bar_dataset_3d
from ..mesh.graph import graph_diameter
from ..mesh.graph3d import EDGE_DIM_3D, NODE_DIM_3D
from ..models.architectures import ModelConfig, build_model, count_parameters, match_capacity
from ..train.trainer import TensorGraph, TrainConfig, evaluate, train_model


@dataclass
class ArchResult3D:
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
class UnderReach3DResult:
    aspect_ratio: float
    n_nodes: int
    n_edges: int
    graph_diameter: int
    n_train: int
    epochs: int
    band_edges: list[float]
    results: list[ArchResult3D]

    def to_dict(self) -> dict:
        return {
            "problem": "3D nonlinear diffusion, k(u) = k0 (1 + alpha u^2)",
            "aspect_ratio": self.aspect_ratio,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "graph_diameter": self.graph_diameter,
            "n_train": self.n_train,
            "epochs": self.epochs,
            "band_edges": self.band_edges,
            "results": [r.to_dict() for r in self.results],
        }

    def table(self) -> str:
        head = (
            f"3D nonlinear  aspect {self.aspect_ratio:g}  nodes {self.n_nodes}  "
            f"edges {self.n_edges}  diameter {self.graph_diameter}\n"
            f"{'architecture':<26}{'params':>9}{'hops':>6}{'relL2':>9}   by distance band\n"
            f"{'-' * 80}"
        )
        rows = []
        for r in self.results:
            hops = "global" if r.receptive_hops < 0 else str(r.receptive_hops)
            bands = "  ".join(f"{b:.3f}" for b in r.rel_l2_by_band)
            rows.append(
                f"{r.label:<26}{r.n_parameters:>9,}{hops:>6}{r.test_rel_l2:>9.4f}   {bands}"
            )
        return head + "\n" + "\n".join(rows)


def _rel_l2_by_band(model, graphs: list[TensorGraph], band_edges: np.ndarray) -> list[float]:
    """Relative L2 within bands of normalised distance from the nearest driven end.

    Band 0 sits against the driven faces, the last band is the middle of the bar. The
    prediction under test is that error grows across the bands for a reach-limited model
    and stays flat for a global one.
    """
    n_bands = len(band_edges) - 1
    num = np.zeros(n_bands)
    den = np.zeros(n_bands)

    model.eval()
    import torch

    with torch.no_grad():
        for g in graphs:
            pred = g.forward(model).cpu().numpy()
            targ = g.target.cpu().numpy()
            interior = g.interior.cpu().numpy()
            x = g.positions[:, 0].cpu().numpy()

            span = x.max() - x.min()
            if span <= 0:
                continue
            frac = np.minimum(x - x.min(), x.max() - x) / span

            for b in range(n_bands):
                sel = interior & (frac >= band_edges[b]) & (frac < band_edges[b + 1])
                if not sel.any():
                    continue
                num[b] += float(((pred[sel] - targ[sel]) ** 2).sum())
                den[b] += float((targ[sel] ** 2).sum())

    return [
        float(np.sqrt(num[b] / den[b])) if den[b] > 0 else float("nan") for b in range(n_bands)
    ]


def run_underreach_study_3d(
    *,
    aspect_ratio: float = 8.0,
    n_samples: int = 40,
    n_cross: int = 4,
    epochs: int = 80,
    shallow_blocks: int = 4,
    deep_blocks: int = 16,
    hidden_dim: int = 64,
    seed: int = 0,
    n_bands: int = 4,
    verbose: bool = True,
) -> UnderReach3DResult:
    """Train the four configurations on 3D nonlinear cases and compare."""
    if verbose:
        print(f"generating {n_samples} nonlinear 3D solves, aspect {aspect_ratio:g} ...")
    ds = bar_dataset_3d(
        n_samples,
        aspect_ratio=aspect_ratio,
        n_cross=n_cross,
        seed=seed,
        verbose=verbose,
    ).normalised()

    diameter = graph_diameter(ds.train[0])
    n_nodes = ds.train[0].n_nodes
    n_edges = ds.train[0].edge_index.shape[1]
    test = [TensorGraph(g) for g in ds.test]
    band_edges = np.linspace(0.0, 0.5, n_bands + 1)

    base = ModelConfig(
        node_dim=NODE_DIM_3D,
        edge_dim=EDGE_DIM_3D,
        hidden_dim=hidden_dim,
        n_blocks=shallow_blocks,
    )
    baseline_params = count_parameters(build_model("meshgraphnet", base))
    tcfg = match_capacity("mesh_graph_transformer", baseline_params, base)

    plan = [
        ("node_mlp", "node MLP (no comms)", base),
        ("meshgraphnet", f"MeshGraphNet L={shallow_blocks}", base),
        (
            "meshgraphnet",
            f"MeshGraphNet L={deep_blocks}",
            ModelConfig(**{**base.to_dict(), "n_blocks": deep_blocks}),
        ),
        ("mesh_graph_transformer", f"MGN-Transformer L={shallow_blocks}", tcfg),
    ]

    if verbose:
        print(
            f"bar aspect {aspect_ratio:g}: {n_nodes} nodes, {n_edges} directed edges, "
            f"diameter {diameter}, {len(ds.train)} train / {len(ds.test)} test"
        )

    results: list[ArchResult3D] = []
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
            ArchResult3D(
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

    return UnderReach3DResult(
        aspect_ratio=aspect_ratio,
        n_nodes=n_nodes,
        n_edges=n_edges,
        graph_diameter=diameter,
        n_train=len(ds.train),
        epochs=epochs,
        band_edges=band_edges.tolist(),
        results=results,
    )


def save_result_3d(result: UnderReach3DResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path
