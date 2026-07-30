"""Building blocks: MLPs, message passing, and physics attention.

Scatter operations use ``Tensor.index_add_`` rather than ``torch_scatter`` so the
package installs with nothing beyond PyTorch. On the mesh sizes here that is not the
bottleneck, and it keeps the dependency surface small enough to audit.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class MLP(nn.Module):
    """Multilayer perceptron with optional output LayerNorm.

    LayerNorm rather than BatchNorm throughout. Graphs arrive one at a time with
    varying node counts, so batch statistics are meaningless here and BatchNorm
    fails outright on a single-sample batch.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        n_layers: int = 2,
        layer_norm: bool = True,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
        if layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def scatter_sum(src: Tensor, index: Tensor, n_out: int) -> Tensor:
    """Sum ``src`` rows into ``n_out`` buckets given by ``index``."""
    out = src.new_zeros((n_out, src.shape[-1]))
    return out.index_add_(0, index, src)


def scatter_mean(src: Tensor, index: Tensor, n_out: int) -> Tensor:
    """Mean of ``src`` rows per bucket, with empty buckets left at zero."""
    total = scatter_sum(src, index, n_out)
    ones = src.new_ones((src.shape[0], 1))
    count = scatter_sum(ones, index, n_out).clamp_min(1.0)
    return total / count


class MessagePassingBlock(nn.Module):
    r"""One residual message-passing step over an unstructured graph.

    Edges update from their endpoints, then nodes aggregate incoming edges:

    .. math::
        \mathbf{e}_{ij} \leftarrow \mathbf{e}_{ij}
            + \varphi\!\left([\mathbf{e}_{ij}, \mathbf{h}_i, \mathbf{h}_j]\right),
        \qquad
        \mathbf{h}_i \leftarrow \mathbf{h}_i
            + \gamma\!\left(\left[\mathbf{h}_i,
              \textstyle\sum_{j\in\mathcal{N}(i)} \mathbf{e}_{ji}\right]\right)

    Residual form matters: without it, stacking the fifteen-plus blocks needed to
    cover a long mesh drives activations toward a fixed point and training stalls.
    Sum aggregation rather than mean, because the physical quantity being propagated
    is a flux, and sums are what conserve it.
    """

    def __init__(self, hidden_dim: int, *, n_layers: int = 2) -> None:
        super().__init__()
        self.edge_mlp = MLP(3 * hidden_dim, hidden_dim, hidden_dim, n_layers=n_layers)
        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim, n_layers=n_layers)

    def forward(self, h: Tensor, e: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        src, dst = edge_index[0], edge_index[1]
        e_in = torch.cat([e, h[src], h[dst]], dim=-1)
        e = e + self.edge_mlp(e_in)

        agg = scatter_sum(e, dst, h.shape[0])
        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        return h, e


class PhysicsAttention(nn.Module):
    r"""Global multi-head self-attention over mesh nodes, with geometric bias.

    Message passing is local: after :math:`L` steps a node has seen only its
    :math:`L`-hop neighbourhood. When the graph diameter exceeds :math:`L`, boundary
    information physically cannot reach the interior. This is the *under-reaching*
    problem, and it is the reason a purely local architecture degrades as meshes grow.

    Attention has a receptive field of the entire mesh in one step:

    .. math::
        \mathrm{Attn}(\mathbf{Q},\mathbf{K},\mathbf{V}) =
        \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^{\mathsf T}}{\sqrt{d_h}}
        + \mathbf{B}\right)\mathbf{V}

    The bias :math:`\mathbf{B}` is a learned function of pairwise distance. Without
    it, attention is permutation invariant and blind to geometry, so a node cannot
    distinguish a near neighbour from a far one. With it, the block keeps the
    geometric inductive bias that makes MeshGraphNet effective while gaining global
    reach.

    Complexity is :math:`O(N^2)` in node count. For the mesh sizes in this
    repository that is affordable; ``chunk_size`` bounds peak memory by computing
    attention in row blocks.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        n_heads: int = 4,
        dropout: float = 0.0,
        chunk_size: int | None = 1024,
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.chunk_size = chunk_size

        self.norm = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

        # Distance -> per-head additive bias. Small on purpose: this is a modulation
        # of attention, not a second network.
        self.dist_bias = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, n_heads))

        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = MLP(hidden_dim, 2 * hidden_dim, hidden_dim, layer_norm=False)

    def forward(self, h: Tensor, positions: Tensor) -> Tensor:
        n = h.shape[0]
        x = self.norm(h)
        qkv = self.qkv(x).reshape(n, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (n, heads, head_dim)

        q = q.transpose(0, 1)  # (heads, n, head_dim)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        chunk = self.chunk_size or n
        pieces: list[Tensor] = []
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            scores = torch.matmul(q[:, start:stop], k.transpose(-2, -1))
            scores = scores / math.sqrt(self.head_dim)

            d = torch.cdist(positions[start:stop], positions).unsqueeze(-1)
            bias = self.dist_bias(d).permute(2, 0, 1)  # (heads, chunk, n)
            scores = scores + bias

            attn = torch.softmax(scores, dim=-1)
            pieces.append(torch.matmul(attn, v))

        ctx = torch.cat(pieces, dim=1).transpose(0, 1).reshape(n, -1)
        h = h + self.drop(self.out(ctx))
        h = h + self.ffn(self.ffn_norm(h))
        return h
