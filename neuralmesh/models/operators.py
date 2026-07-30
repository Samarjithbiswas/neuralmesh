r"""Published neural-operator baselines, on the same interface as the graph models.

Comparing three architectures written by one author establishes an internal ranking, not
a contribution to the literature. These are the families a reviewer will ask about, each
attacking global coupling by a different mechanism:

* :class:`FNO` (Li et al., 2021) mixes in Fourier space. One spectral layer touches the
  whole domain, so there is no under-reaching by construction.
* :class:`DeepONet` (Lu et al., 2021) splits into a branch network reading the input
  function and a trunk reading the query location, combined by a dot product. Global
  because the branch sees a pooled summary of the entire input.
* :class:`GNO` (Li et al., 2020) is a kernel integral over a *radius* neighbourhood built
  from positions, not from mesh connectivity. Reach grows with the radius rather than
  with depth.

All three take the identical ``(node_features, edge_index, edge_features, positions)``
signature as the graph models and return one value per node, so the existing trainer,
evaluator and parameter-matching machinery work unchanged. That is what makes the
comparison fair: same data, same optimiser, same budget, same metric.

**The honest caveat on FNO.** The FFT needs values on a regular grid, and an unstructured
mesh has none. This implementation resamples node values onto a uniform grid, runs the
spectral layers there, and interpolates back. That resampling is a real source of error
and it is exactly why geometry-aware FNO variants exist. Reporting FNO on unstructured
geometry without saying this would be misleading, so :attr:`FNO.resample_loss` measures
the round-trip error directly and a test asserts it is tracked rather than assumed small.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .blocks import MLP, scatter_mean


# --------------------------------------------------------------------------- resampling
class GridResampler(nn.Module):
    """Move node values onto a uniform grid and back.

    Scatter uses a mean over the nodes falling in each cell; gather uses bilinear or
    trilinear interpolation via ``grid_sample``, which is differentiable and works
    identically in 2D and 3D.

    Empty cells are a genuine problem on a long thin domain: a grid fine enough to
    resolve the short direction leaves most cells along the long direction empty. Those
    cells are filled by nearest-occupied propagation rather than left at zero, because
    zeros would be read by the spectral layers as real field values.
    """

    def __init__(self, resolution: int = 32) -> None:
        super().__init__()
        if resolution < 4:
            raise ValueError("resolution must be at least 4 for the FFT to be meaningful")
        self.resolution = resolution

    @staticmethod
    def _normalise(positions: Tensor) -> Tensor:
        lo = positions.min(dim=0).values
        hi = positions.max(dim=0).values
        span = (hi - lo).clamp_min(1e-12)
        return (positions - lo) / span

    def scatter(self, values: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        """``(n, c)`` node values to ``(1, c, R, R[, R])``, plus the occupancy mask."""
        n, c = values.shape
        d = positions.shape[1]
        R = self.resolution
        unit = self._normalise(positions)

        idx = (unit * (R - 1)).round().long().clamp(0, R - 1)
        flat = idx[:, 0]
        for k in range(1, d):
            flat = flat * R + idx[:, k]

        cells = R**d
        grid = scatter_mean(values, flat, cells)
        occupied = torch.zeros(cells, 1, device=values.device, dtype=values.dtype)
        occupied.index_add_(
            0, flat, torch.ones(n, 1, device=values.device, dtype=values.dtype)
        )
        mask = (occupied > 0).to(values.dtype)

        shape = (1, c) + (R,) * d
        grid = grid.T.reshape(shape)
        mask_g = mask.T.reshape((1, 1) + (R,) * d)

        # Fill empty cells by dilating occupied values, so the FFT never sees fabricated
        # zeros. A handful of passes is enough for the aspect ratios used here.
        pool = F.max_pool2d if d == 2 else F.max_pool3d
        for _ in range(3):
            if bool((mask_g > 0).all()):
                break
            spread = pool(grid * mask_g, kernel_size=3, stride=1, padding=1)
            spread_m = pool(mask_g, kernel_size=3, stride=1, padding=1)
            take = (mask_g < 0.5) & (spread_m > 0.5)
            grid = torch.where(take, spread, grid)
            mask_g = torch.where(take, torch.ones_like(mask_g), mask_g)
        return grid, mask_g

    def gather(self, grid: Tensor, positions: Tensor) -> Tensor:
        """``(1, c, R, ...)`` back to ``(n, c)`` by bilinear/trilinear interpolation."""
        d = positions.shape[1]
        unit = self._normalise(positions)
        # grid_sample expects coordinates in [-1, 1] and reversed axis order
        coords = (unit * 2.0 - 1.0).flip(-1)
        samp = coords.view(1, -1, 1, 2) if d == 2 else coords.view(1, -1, 1, 1, 3)
        out = F.grid_sample(grid, samp, mode="bilinear", align_corners=True)
        return out.reshape(grid.shape[1], -1).T

    def round_trip_error(self, values: Tensor, positions: Tensor) -> Tensor:
        """Relative error of scatter-then-gather. The cost of using a grid at all."""
        grid, _ = self.scatter(values, positions)
        back = self.gather(grid, positions)
        denom = values.norm().clamp_min(1e-12)
        return (back - values).norm() / denom


# ------------------------------------------------------------------------ spectral layer
class SpectralConv(nn.Module):
    r"""Fourier layer: transform, keep and mix low modes, transform back.

    .. math:: (\mathcal{K}v)(x) = \mathcal{F}^{-1}\big(R_\phi\cdot\mathcal{F}(v)\big)(x)

    Truncating to ``modes`` low frequencies is both a smoothing prior and what keeps the
    parameter count finite. Global reach comes free because every Fourier mode is
    supported over the whole domain, so touching one mode touches everywhere at once.
    """

    def __init__(self, width: int, modes: int, ndim: int) -> None:
        super().__init__()
        if ndim not in (2, 3):
            raise ValueError("only 2D and 3D are supported")
        self.width = width
        self.modes = modes
        self.ndim = ndim
        shape = (width, width) + (modes,) * ndim
        scale = 1.0 / (width * width)
        self.weight_real = nn.Parameter(scale * torch.randn(*shape))
        self.weight_imag = nn.Parameter(scale * torch.randn(*shape))

    def forward(self, x: Tensor) -> Tensor:
        dims = tuple(range(2, 2 + self.ndim))
        ft = torch.fft.rfftn(x, dim=dims)

        m = self.modes
        sizes = [ft.shape[d] for d in dims]
        m_eff = [min(m, s) for s in sizes]

        weight = torch.complex(self.weight_real, self.weight_imag)
        out = torch.zeros_like(ft)
        if self.ndim == 2:
            a, b = m_eff
            out[:, :, :a, :b] = torch.einsum(
                "bixy,ioxy->boxy", ft[:, :, :a, :b], weight[:, :, :a, :b]
            )
        else:
            a, b, c = m_eff
            out[:, :, :a, :b, :c] = torch.einsum(
                "bixyz,ioxyz->boxyz", ft[:, :, :a, :b, :c], weight[:, :, :a, :b, :c]
            )
        return torch.fft.irfftn(out, s=x.shape[2:], dim=dims)


# ------------------------------------------------------------------------------- models
class FNO(nn.Module):
    """Fourier Neural Operator, resampled onto a grid to reach unstructured meshes."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        width = cfg.hidden_dim
        self.ndim = getattr(cfg, "spatial_dim", 2)
        self.resampler = GridResampler(getattr(cfg, "grid_resolution", 32))
        modes = getattr(cfg, "fourier_modes", 8)

        self.lift = nn.Linear(cfg.node_dim + self.ndim, width)
        self.spectral = nn.ModuleList(
            SpectralConv(width, modes, self.ndim) for _ in range(cfg.n_blocks)
        )
        conv = nn.Conv2d if self.ndim == 2 else nn.Conv3d
        self.local = nn.ModuleList(conv(width, width, 1) for _ in range(cfg.n_blocks))
        self.project = MLP(width, width, cfg.out_dim, n_layers=2, layer_norm=False)
        #: relative error of the last scatter-gather round trip, for reporting
        self.resample_loss: float = float("nan")

    @property
    def receptive_hops(self) -> int:
        """Global: a spectral layer couples the entire domain in one step."""
        return -1

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        positions: Tensor,
    ) -> Tensor:
        pos = positions[:, : self.ndim]
        x = torch.cat([node_features, self.resampler._normalise(pos)], dim=-1)
        x = self.lift(x)

        with torch.no_grad():
            self.resample_loss = float(self.resampler.round_trip_error(x.detach(), pos))

        grid, _ = self.resampler.scatter(x, pos)
        for spec, loc in zip(self.spectral, self.local):
            grid = F.gelu(spec(grid) + loc(grid))
        back = self.resampler.gather(grid, pos)
        return self.project(back).squeeze(-1)


class DeepONet(nn.Module):
    r"""Branch reads the input function, trunk reads the query point.

    .. math:: \mathcal{G}(a)(y) \approx \sum_{k=1}^{p} b_k(a)\, t_k(y)

    Worth noticing what this structure is: the trunk outputs are a learned basis and the
    branch outputs are coefficients. That is modal superposition with the basis learned
    from data rather than computed from an eigenvalue problem.

    The branch here pools node features over the whole mesh, which is what makes the model
    global: every query point sees a summary of the entire input field.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        p = getattr(cfg, "basis_size", 64)
        width = cfg.hidden_dim
        self.ndim = getattr(cfg, "spatial_dim", 2)

        self.encode = MLP(cfg.node_dim, width, width, n_layers=2)
        # mean and max pooling together: the mean carries the average state, the max
        # carries extremes such as a localised source that a mean would wash out
        self.branch = MLP(2 * width, width, p, n_layers=3, layer_norm=False)
        self.trunk = MLP(self.ndim, width, p, n_layers=3, layer_norm=False)
        self.bias = nn.Parameter(torch.zeros(1))

    @property
    def receptive_hops(self) -> int:
        return -1

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        positions: Tensor,
    ) -> Tensor:
        h = self.encode(node_features)
        pooled = torch.cat([h.mean(dim=0), h.max(dim=0).values], dim=-1)
        coeff = self.branch(pooled)

        pos = positions[:, : self.ndim]
        lo, hi = pos.min(dim=0).values, pos.max(dim=0).values
        unit = (pos - lo) / (hi - lo).clamp_min(1e-12)
        basis = self.trunk(unit)
        return (basis * coeff).sum(dim=-1) + self.bias


class GNO(nn.Module):
    r"""Graph Neural Operator: kernel integral over a radius neighbourhood.

    .. math::
        h_i \leftarrow \sigma\Big(W h_i + \frac{1}{|\mathcal{N}(i)|}
        \sum_{j\in\mathcal{N}(i)} \kappa(x_i, x_j)\, h_j\Big)

    The distinguishing feature against MeshGraphNet is where the neighbourhood comes from.
    MeshGraphNet uses the mesh connectivity, so reach is fixed at one hop per layer. GNO
    builds a *radius* graph from positions, so a larger radius buys reach without buying
    depth. That is a different way of attacking the same problem and it is why the model
    belongs in this comparison rather than being a relabelled baseline.

    Neighbourhoods are capped at ``max_neighbours`` by random subsampling, which is the
    Nystrom approximation from the original work and keeps cost linear rather than
    quadratic in node count.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        width = cfg.hidden_dim
        self.ndim = getattr(cfg, "spatial_dim", 2)
        self.radius = getattr(cfg, "gno_radius", 0.25)
        self.max_neighbours = getattr(cfg, "max_neighbours", 24)

        self.encode = MLP(cfg.node_dim, width, width, n_layers=2)
        self.kernel = nn.ModuleList(
            MLP(self.ndim + 1, width, width, n_layers=2, layer_norm=False)
            for _ in range(cfg.n_blocks)
        )
        self.selfmap = nn.ModuleList(nn.Linear(width, width) for _ in range(cfg.n_blocks))
        self.decode = MLP(width, width, cfg.out_dim, n_layers=2, layer_norm=False)

    @property
    def receptive_hops(self) -> int:
        """Global in the sense that reach is set by radius, not by depth."""
        return -1

    def _radius_graph(self, positions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        pos = positions[:, : self.ndim]
        lo, hi = pos.min(dim=0).values, pos.max(dim=0).values
        unit = (pos - lo) / (hi - lo).clamp_min(1e-12)

        n = unit.shape[0]
        d = torch.cdist(unit, unit)
        within = d <= self.radius
        within.fill_diagonal_(False)

        src, dst = within.nonzero(as_tuple=True)
        if src.numel() == 0:  # radius too small: fall back to nearest neighbour
            nearest = d.masked_fill(torch.eye(n, dtype=torch.bool, device=d.device), 1e9)
            dst = torch.arange(n, device=d.device)
            src = nearest.argmin(dim=1)

        if self.max_neighbours > 0 and src.numel() > n * self.max_neighbours:
            keep = torch.randperm(src.numel(), device=src.device)[: n * self.max_neighbours]
            src, dst = src[keep], dst[keep]

        rel = unit[src] - unit[dst]
        feat = torch.cat([rel, rel.norm(dim=-1, keepdim=True)], dim=-1)
        return src, dst, feat

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        positions: Tensor,
    ) -> Tensor:
        src, dst, kfeat = self._radius_graph(positions)
        h = self.encode(node_features)
        n = h.shape[0]
        for kern, selfw in zip(self.kernel, self.selfmap):
            messages = kern(kfeat) * h[src]
            agg = scatter_mean(messages, dst, n)
            h = F.gelu(selfw(h) + agg)
        return self.decode(h).squeeze(-1)


OPERATORS = {"fno": FNO, "deeponet": DeepONet, "gno": GNO}


def spectral_radius_note() -> str:
    """One-line statement of the FNO limitation, for printing in result tables."""
    return (
        "FNO requires a regular grid; node values are resampled onto one and "
        "interpolated back, which is a real error source, reported as resample_loss."
    )
