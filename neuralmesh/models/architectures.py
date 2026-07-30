"""The three architectures this repository compares.

* :class:`MeshGraphNet` -- encode, process with local message passing, decode. The
  established baseline for mesh-based learned simulation.
* :class:`MeshGraphTransformer` -- the same, with global physics-attention
  interleaved between message-passing blocks, addressing under-reaching.
* :class:`NodeMLP` -- a per-node MLP with no communication at all. Included as a
  control: any result that this model also achieves was never about the graph.

All three share the encode-process-decode skeleton so a comparison isolates the
processor, and all three take an identical :class:`~neuralmesh.mesh.graph.MeshGraph`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .blocks import MLP, MessagePassingBlock, PhysicsAttention


@dataclass
class ModelConfig:
    """Hyperparameters shared by the architectures."""

    node_dim: int = 4
    edge_dim: int = 3
    hidden_dim: int = 64
    out_dim: int = 1
    n_blocks: int = 8
    n_heads: int = 4
    attention_every: int = 2
    dropout: float = 0.0
    mlp_layers: int = 2

    # Operator-baseline settings. Kept on the shared config so every architecture is
    # built the same way and parameter matching works across all six without special
    # cases at the call site.
    spatial_dim: int = 2
    grid_resolution: int = 32
    fourier_modes: int = 8
    basis_size: int = 64
    gno_radius: float = 0.25
    max_neighbours: int = 24

    def to_dict(self) -> dict:
        return asdict(self)


class _EncodeProcessDecode(nn.Module):
    """Shared skeleton. Subclasses supply the processor."""

    #: Whether the processor consumes edge state. A subclass that ignores edges must
    #: say so, otherwise it carries an edge encoder that never receives a gradient.
    #: Those dead parameters would still be counted by ``count_parameters``, which
    #: would overstate the capacity of the no-communication control and quietly
    #: weaken the parameter-matched comparison the whole study rests on.
    uses_edges: bool = True

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.node_encoder = MLP(
            cfg.node_dim, cfg.hidden_dim, cfg.hidden_dim, n_layers=cfg.mlp_layers
        )
        self.edge_encoder = (
            MLP(cfg.edge_dim, cfg.hidden_dim, cfg.hidden_dim, n_layers=cfg.mlp_layers)
            if self.uses_edges
            else None
        )
        # No LayerNorm on the decoder: it would rescale the physical output.
        self.decoder = MLP(
            cfg.hidden_dim,
            cfg.hidden_dim,
            cfg.out_dim,
            n_layers=cfg.mlp_layers,
            layer_norm=False,
        )

    def process(self, h: Tensor, e: Tensor, edge_index: Tensor, pos: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        positions: Tensor,
    ) -> Tensor:
        h = self.node_encoder(node_features)
        e = None if self.edge_encoder is None else self.edge_encoder(edge_features)
        h = self.process(h, e, edge_index, positions)
        return self.decoder(h).squeeze(-1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        return f"{type(self).__name__}(params={self.n_parameters():,})"


class NodeMLP(_EncodeProcessDecode):
    """Control model: no message passing, no attention.

    Each node is mapped independently from its own features to its own output. It
    cannot represent any solution where a node's value depends on a distant boundary
    condition, which for an elliptic PDE is every solution. If this model scores well
    on a benchmark, the benchmark is not measuring what it claims to.
    """

    uses_edges = False

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.trunk = MLP(
            cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, n_layers=cfg.mlp_layers
        )

    def process(self, h: Tensor, e: Tensor, edge_index: Tensor, pos: Tensor) -> Tensor:
        return self.trunk(h)

    @property
    def receptive_hops(self) -> int:
        """Zero: a node never sees any other node."""
        return 0


class MeshGraphNet(_EncodeProcessDecode):
    """Local message passing only, in the style of Pfaff et al.

    Receptive field after ``n_blocks`` steps is the ``n_blocks``-hop neighbourhood.
    Accurate and cheap when the graph diameter is smaller than that; structurally
    unable to solve the problem when it is not.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.blocks = nn.ModuleList(
            MessagePassingBlock(cfg.hidden_dim, n_layers=cfg.mlp_layers)
            for _ in range(cfg.n_blocks)
        )

    def process(self, h: Tensor, e: Tensor, edge_index: Tensor, pos: Tensor) -> Tensor:
        for block in self.blocks:
            h, e = block(h, e, edge_index)
        return h

    @property
    def receptive_hops(self) -> int:
        return self.cfg.n_blocks


class MeshGraphTransformer(_EncodeProcessDecode):
    """Message passing interleaved with global physics attention.

    A block index ``i`` gets an attention layer when
    ``(i + 1) % attention_every == 0``, so with the default of 2 the processor
    alternates local and global steps. Local blocks keep the geometric inductive
    bias that makes mesh graph networks sample-efficient; attention removes the
    dependence of receptive field on depth.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        if cfg.attention_every < 1:
            raise ValueError("attention_every must be >= 1")
        self.blocks = nn.ModuleList(
            MessagePassingBlock(cfg.hidden_dim, n_layers=cfg.mlp_layers)
            for _ in range(cfg.n_blocks)
        )
        self.attn_at = {i for i in range(cfg.n_blocks) if (i + 1) % cfg.attention_every == 0}
        self.attn = nn.ModuleDict(
            {
                str(i): PhysicsAttention(
                    cfg.hidden_dim, n_heads=cfg.n_heads, dropout=cfg.dropout
                )
                for i in sorted(self.attn_at)
            }
        )

    def process(self, h: Tensor, e: Tensor, edge_index: Tensor, pos: Tensor) -> Tensor:
        for i, block in enumerate(self.blocks):
            h, e = block(h, e, edge_index)
            if i in self.attn_at:
                h = self.attn[str(i)](h, pos)
        return h

    @property
    def receptive_hops(self) -> int:
        """Global reach once any attention layer is present."""
        return -1 if self.attn_at else self.cfg.n_blocks


from .operators import OPERATORS  # noqa: E402  (placed here to avoid a cycle)

#: Every architecture the study can build. The first three are written here; the last
#: three are the published operator families a reviewer will ask about.
ARCHITECTURES: dict[str, type[nn.Module]] = {
    "node_mlp": NodeMLP,
    "meshgraphnet": MeshGraphNet,
    "mesh_graph_transformer": MeshGraphTransformer,
    **OPERATORS,
}

#: Which entries are reimplementations of published work rather than this repository's own.
PUBLISHED_BASELINES = tuple(OPERATORS)


def build_model(name: str, cfg: ModelConfig | None = None) -> nn.Module:
    """Instantiate an architecture by name."""
    if name not in ARCHITECTURES:
        raise KeyError(f"unknown architecture {name!r}; choose from {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[name](cfg or ModelConfig())


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def match_capacity(
    name: str, target_params: int, cfg: ModelConfig, *, tol: float = 0.08
) -> ModelConfig:
    """Adjust ``hidden_dim`` so an architecture lands near ``target_params``.

    A comparison between architectures at different parameter counts measures
    capacity as much as design. This searches multiples of ``n_heads`` for the width
    whose parameter count is closest to the target, so the benchmark can hold model
    size roughly fixed and vary only the processor.
    """
    best: tuple[float, int] | None = None
    step = max(cfg.n_heads, 1)
    for width in range(step, 33 * step, step):
        trial = ModelConfig(**{**cfg.to_dict(), "hidden_dim": width})
        try:
            count = count_parameters(ARCHITECTURES[name](trial))
        except ValueError:
            continue
        rel = abs(count - target_params) / max(target_params, 1)
        if best is None or rel < best[0]:
            best = (rel, width)
        if rel < tol:
            break
    if best is None:
        raise RuntimeError(f"could not size {name} to {target_params} parameters")
    return ModelConfig(**{**cfg.to_dict(), "hidden_dim": best[1]})
