"""Training loop, losses and metrics.

Two things here are deliberate rather than incidental.

*Boundary nodes are excluded from the loss.* Their values are inputs, not
predictions. Including them inflates every metric, because the model learns to copy a
feature it was handed, and on a fine mesh the boundary can be a third of the nodes.

*A physics residual is available but off by default.* Weighting a PDE residual against
a data term is a real hyperparameter with real failure modes, and reporting a number
obtained with an untuned physics weight would be misleading. The term is implemented,
tested, and opt-in.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from ..mesh.graph import MeshGraph


@dataclass
class TrainConfig:
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    physics_weight: float = 0.0
    scheduler: str = "cosine"
    min_lr_factor: float = 0.02
    seed: int = 0
    log_every: int = 10
    device: str = "cpu"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_rel_l2: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)

    def best_epoch(self) -> int:
        return int(np.argmin(self.val_loss)) if self.val_loss else -1

    def to_dict(self) -> dict:
        return asdict(self)


class TensorGraph:
    """A :class:`MeshGraph` moved onto a device as tensors."""

    __slots__ = (
        "edge_features",
        "edge_index",
        "interior",
        "node_features",
        "positions",
        "target",
    )

    def __init__(self, graph: MeshGraph, device: str = "cpu") -> None:
        f32 = torch.float32
        self.node_features = torch.as_tensor(graph.node_features, dtype=f32, device=device)
        self.edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long, device=device)
        self.edge_features = torch.as_tensor(graph.edge_features, dtype=f32, device=device)
        self.positions = torch.as_tensor(graph.positions, dtype=f32, device=device)
        self.target = (
            None
            if graph.target is None
            else torch.as_tensor(graph.target, dtype=f32, device=device)
        )
        # Column 0 of node features is the boundary indicator, but after
        # normalisation it is no longer 0/1, so recover interior nodes by taking the
        # lower of the two distinct values present.
        col = self.node_features[:, 0]
        self.interior = col == col.min()

    def forward(self, model: nn.Module) -> Tensor:
        return model(self.node_features, self.edge_index, self.edge_features, self.positions)


def masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean squared error over masked nodes only."""
    if mask.sum() == 0:
        raise ValueError("mask selects no nodes")
    return ((pred[mask] - target[mask]) ** 2).mean()


def relative_l2(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> float:
    r"""Relative :math:`L^2` error, :math:`\|p - t\|_2 / \|t\|_2`.

    Reported instead of raw MSE because it is dimensionless and comparable across
    cases whose solution magnitudes differ by orders of magnitude.
    """
    p, t = (pred, target) if mask is None else (pred[mask], target[mask])
    denom = torch.linalg.vector_norm(t)
    if float(denom) == 0.0:
        return float("nan")
    return float(torch.linalg.vector_norm(p - t) / denom)


def dirichlet_residual(pred: Tensor, graph: TensorGraph) -> Tensor:
    r"""Graph Dirichlet energy of the prediction, a smoothness prior.

    .. math:: R = \frac{1}{|E|}\sum_{(i,j)\in E} \frac{(u_i - u_j)^2}{\|x_i-x_j\|^2}

    Not a PDE residual: it does not know the source term, so minimising it alone
    drives the field toward a constant. Used only as a weak regulariser, which is
    why ``physics_weight`` defaults to zero.
    """
    src, dst = graph.edge_index[0], graph.edge_index[1]
    du = pred[dst] - pred[src]
    dx2 = (graph.edge_features[:, :2] ** 2).sum(dim=1).clamp_min(1e-12)
    return (du**2 / dx2).mean()


def evaluate(model: nn.Module, graphs: list[TensorGraph]) -> tuple[float, float]:
    """Mean masked MSE and mean relative L2 over a list of graphs."""
    model.eval()
    losses: list[float] = []
    rels: list[float] = []
    with torch.no_grad():
        for g in graphs:
            pred = g.forward(model)
            losses.append(float(masked_mse(pred, g.target, g.interior)))
            rels.append(relative_l2(pred, g.target, g.interior))
    return float(np.mean(losses)), float(np.mean(rels))


def train_model(
    model: nn.Module,
    train_graphs: list[MeshGraph],
    val_graphs: list[MeshGraph],
    cfg: TrainConfig | None = None,
    *,
    verbose: bool = True,
) -> tuple[nn.Module, History]:
    """Train one model, returning it restored to its best validation weights.

    Restoring the best checkpoint rather than the last epoch matters on small
    datasets, where the final epoch is frequently not the best one and reporting it
    understates the architecture.
    """
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = cfg.device
    model = model.to(device)
    tr = [TensorGraph(g, device) for g in train_graphs]
    va = [TensorGraph(g, device) for g in val_graphs]
    if not tr or not va:
        raise ValueError("training and validation sets must both be non-empty")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.epochs, eta_min=cfg.lr * cfg.min_lr_factor
        )
    elif cfg.scheduler == "none":
        sched = None
    else:
        raise ValueError(f"unknown scheduler {cfg.scheduler!r}")

    history = History()
    best = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    order = np.arange(len(tr))
    rng = np.random.default_rng(cfg.seed)

    for epoch in range(cfg.epochs):
        t0 = time.perf_counter()
        model.train()
        rng.shuffle(order)
        running = 0.0

        for i in order:
            g = tr[i]
            opt.zero_grad(set_to_none=True)
            pred = g.forward(model)
            loss = masked_mse(pred, g.target, g.interior)
            if cfg.physics_weight > 0.0:
                loss = loss + cfg.physics_weight * dirichlet_residual(pred, g)
            loss.backward()
            if cfg.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += float(loss.detach())

        val_loss, val_rel = evaluate(model, va)
        history.train_loss.append(running / len(tr))
        history.val_loss.append(val_loss)
        history.val_rel_l2.append(val_rel)
        history.lr.append(opt.param_groups[0]["lr"])
        history.epoch_seconds.append(time.perf_counter() - t0)

        if val_loss < best:
            best = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if sched is not None:
            sched.step()

        if verbose and (epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1):
            print(
                f"  epoch {epoch:4d}  train {history.train_loss[-1]:.4e}  "
                f"val {val_loss:.4e}  relL2 {val_rel:.4f}"
            )

    model.load_state_dict(best_state)
    return model, history


def save_run(
    path: str | Path,
    model: nn.Module,
    history: History,
    model_cfg: dict,
    train_cfg: dict,
    extra: dict | None = None,
) -> Path:
    """Write weights and a JSON sidecar recording exactly how the run was configured."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path.with_suffix(".pt"))
    meta = {
        "model": model_cfg,
        "train": train_cfg,
        "history": history.to_dict(),
        "best_epoch": history.best_epoch(),
        **(extra or {}),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path.with_suffix(".pt")
