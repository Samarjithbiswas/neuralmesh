"""Tests for the published operator baselines.

Two groups matter more than the rest.

``test_every_global_model_actually_reaches_globally`` measures reach empirically for each
baseline instead of trusting the ``receptive_hops`` label. A baseline that claims global
coupling but does not deliver it would make the comparison meaningless in the direction
that flatters this repository's own model, so it is checked rather than asserted.

``test_fno_resample_error_is_measured`` exists because FNO cannot run on an unstructured
mesh without resampling to a grid, and that resampling is a real error source. Quoting an
FNO number without it would be misleading.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neuralmesh import (
    ModelConfig,
    TensorGraph,
    build_model,
    count_parameters,
    graph_from_solution,
    match_capacity,
    solve_poisson,
    unit_square_mesh,
)
from neuralmesh.models.architectures import ARCHITECTURES, PUBLISHED_BASELINES
from neuralmesh.models.operators import GridResampler, SpectralConv, spectral_radius_note

ALL_NAMES = sorted(ARCHITECTURES)


@pytest.fixture(scope="module")
def graph():
    mesh = unit_square_mesh(11, jitter=0.15, seed=2)
    sol = solve_poisson(mesh, source=1.0, dirichlet_value=0.0)
    return TensorGraph(graph_from_solution(sol))


def _cfg(**kw):
    base = {"hidden_dim": 32, "n_blocks": 3, "grid_resolution": 16, "fourier_modes": 6}
    base.update(kw)
    return ModelConfig(**base)


# ------------------------------------------------------------------------ registration
def test_published_baselines_are_registered():
    assert set(PUBLISHED_BASELINES) == {"fno", "deeponet", "gno"}
    for name in PUBLISHED_BASELINES:
        assert name in ARCHITECTURES


@pytest.mark.parametrize("name", ALL_NAMES)
def test_forward_shape_and_finiteness(graph, name):
    model = build_model(name, _cfg())
    out = graph.forward(model)
    assert out.shape == (graph.node_features.shape[0],)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", ALL_NAMES)
def test_gradients_reach_every_parameter(graph, name):
    model = build_model(name, _cfg())
    graph.forward(model).sum().backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"{name}: no gradient reached {missing}"


@pytest.mark.parametrize("name", PUBLISHED_BASELINES)
def test_baselines_report_global_reach(name):
    assert build_model(name, _cfg()).receptive_hops == -1


# ------------------------------------------------------------------------------- reach
@pytest.mark.parametrize("name", ["fno", "deeponet", "gno", "mesh_graph_transformer"])
def test_every_global_model_actually_reaches_globally(name):
    """Perturb one end of a long strip; the far end must respond.

    A local model fails this by construction. A model that merely *claims* to be global
    would also fail it, which is the case worth catching, because an under-powered
    baseline would make this repository's own model look better than it is.
    """
    n = 40
    pts = np.column_stack([np.linspace(0.0, 1.0, n), np.zeros(n)])
    a = np.arange(n - 1)
    edge_index = torch.tensor(
        np.vstack([np.concatenate([a, a + 1]), np.concatenate([a + 1, a])]), dtype=torch.long
    )
    edge_features = torch.zeros(edge_index.shape[1], 3)
    positions = torch.tensor(pts, dtype=torch.float32)

    torch.manual_seed(0)
    cfg = _cfg(n_blocks=1, attention_every=1, gno_radius=0.9, grid_resolution=16)
    model = build_model(name, cfg).eval()

    nf = torch.zeros(n, 4)
    with torch.no_grad():
        base = model(nf, edge_index, edge_features, positions)
        bumped = nf.clone()
        bumped[0, 2] = 1.0
        after = model(bumped, edge_index, edge_features, positions)

    moved = (after - base).abs()
    assert moved[-1] > 1e-9, (
        f"{name} claims global reach but perturbing node 0 did not move node {n - 1}"
    )


def test_local_model_still_fails_the_same_reach_probe():
    """The control on the control: the probe must distinguish local from global."""
    n = 40
    pts = np.column_stack([np.linspace(0.0, 1.0, n), np.zeros(n)])
    a = np.arange(n - 1)
    edge_index = torch.tensor(
        np.vstack([np.concatenate([a, a + 1]), np.concatenate([a + 1, a])]), dtype=torch.long
    )
    edge_features = torch.zeros(edge_index.shape[1], 3)
    positions = torch.tensor(pts, dtype=torch.float32)

    torch.manual_seed(0)
    model = build_model("meshgraphnet", _cfg(n_blocks=2)).eval()
    nf = torch.zeros(n, 4)
    with torch.no_grad():
        base = model(nf, edge_index, edge_features, positions)
        bumped = nf.clone()
        bumped[0, 2] = 1.0
        after = model(bumped, edge_index, edge_features, positions)
    assert (after - base).abs()[-1] < 1e-9, "the probe cannot tell local from global"


# ---------------------------------------------------------------------------- resampler
def test_resampler_round_trip_is_reasonable_on_a_smooth_field():
    torch.manual_seed(0)
    pos = torch.rand(400, 2)
    values = torch.sin(3 * pos[:, :1]) + pos[:, 1:]
    err = float(GridResampler(32).round_trip_error(values, pos))
    assert 0.0 <= err < 0.35, f"round-trip error {err:.3f} is implausible"


def test_resampler_fills_empty_cells_rather_than_leaving_zeros():
    """A long thin domain leaves most grid cells empty; zeros would be read as data.

    The property under test is that the fill materially increases occupancy, so it is
    measured against the raw scatter rather than against a hardcoded fraction.
    """
    torch.manual_seed(0)
    R = 32
    # A genuinely thin strip. The cross-section spans a fraction of one grid cell, which
    # is exactly the regime where a naive scatter leaves most of the grid at zero.
    x = torch.rand(200)
    y = 0.5 + 0.01 * (torch.rand(200) - 0.5)
    pos = torch.stack([x, y], dim=1)
    values = torch.ones(200, 3)

    resampler = GridResampler(R)
    unit = resampler._normalise(pos)
    idx = (unit * (R - 1)).round().long().clamp(0, R - 1)
    raw_cells = len({(int(a), int(b)) for a, b in idx})
    raw_fraction = raw_cells / (R * R)

    grid, mask = resampler.scatter(values, pos)
    filled_fraction = float(mask.mean())

    assert filled_fraction > 3.0 * raw_fraction, (
        f"fill barely helped: {raw_fraction:.3f} -> {filled_fraction:.3f}"
    )
    occupied_vals = grid[mask.expand_as(grid) > 0.5]
    assert torch.isfinite(occupied_vals).all()
    assert float(occupied_vals.min()) > 0.0, "a filled cell kept a fabricated zero"


def test_resampler_rejects_a_useless_resolution():
    with pytest.raises(ValueError):
        GridResampler(2)


def test_resampler_works_in_three_dimensions():
    torch.manual_seed(1)
    pos = torch.rand(300, 3)
    values = torch.rand(300, 4)
    grid, _ = GridResampler(8).scatter(values, pos)
    assert grid.shape == (1, 4, 8, 8, 8)
    back = GridResampler(8).gather(grid, pos)
    assert back.shape == (300, 4)


def test_spectral_conv_preserves_shape_and_is_real():
    torch.manual_seed(0)
    x = torch.randn(1, 8, 16, 16)
    out = SpectralConv(8, 5, 2)(x)
    assert out.shape == x.shape
    assert not out.is_complex()


def test_spectral_conv_rejects_unsupported_dimension():
    with pytest.raises(ValueError):
        SpectralConv(8, 4, 4)


def test_fno_resample_error_is_measured(graph):
    """The limitation has to be reported, so it has to be recorded."""
    model = build_model("fno", _cfg())
    graph.forward(model)
    assert np.isfinite(model.resample_loss)
    assert model.resample_loss >= 0.0
    assert "resample" in spectral_radius_note()


# ------------------------------------------------------------------------- invariances
@pytest.mark.parametrize("name", ["deeponet", "gno"])
def test_mesh_free_baselines_are_permutation_invariant(graph, name):
    """Relabelling nodes must not change the physics."""
    torch.manual_seed(0)
    model = build_model(name, _cfg()).eval()
    n = graph.node_features.shape[0]
    perm = torch.randperm(n)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n)

    with torch.no_grad():
        base = model(
            graph.node_features, graph.edge_index, graph.edge_features, graph.positions
        )
        shuffled = model(
            graph.node_features[perm],
            inv[graph.edge_index],
            graph.edge_features,
            graph.positions[perm],
        )
    assert torch.allclose(base, shuffled[inv], atol=1e-4)


def test_deeponet_is_a_learned_basis_expansion():
    """Trunk gives a basis, branch gives coefficients. Check the shapes really are that."""
    cfg = _cfg(basis_size=17)
    model = build_model("deeponet", cfg)
    assert model.trunk.net[-1].out_features == 17
    assert model.branch.net[-1].out_features == 17


def test_gno_radius_controls_neighbourhood_size():
    """Reach should come from the radius, which is the whole point of GNO."""
    torch.manual_seed(0)
    pos = torch.rand(120, 2)
    small = build_model("gno", _cfg(gno_radius=0.08, max_neighbours=0))
    large = build_model("gno", _cfg(gno_radius=0.5, max_neighbours=0))
    n_small = small._radius_graph(pos)[0].numel()
    n_large = large._radius_graph(pos)[0].numel()
    assert n_large > 3 * n_small, f"radius had little effect: {n_small} -> {n_large}"


def test_gno_falls_back_when_the_radius_is_too_small():
    """A radius smaller than the closest pair must not produce an empty graph."""
    torch.manual_seed(0)
    pos = torch.rand(50, 2)
    model = build_model("gno", _cfg(gno_radius=1e-6))
    src, dst, feat = model._radius_graph(pos)
    assert src.numel() == 50 and dst.numel() == 50
    assert feat.shape == (50, 3)


# -------------------------------------------------------------------- fair comparison
@pytest.mark.parametrize("name", PUBLISHED_BASELINES)
def test_capacity_matching_works_for_the_baselines(name):
    """Without this the comparison measures width rather than design."""
    base = _cfg(hidden_dim=64, n_blocks=3)
    target = count_parameters(build_model("meshgraphnet", base))
    matched = match_capacity(name, target, base)
    got = count_parameters(build_model(name, matched))
    assert abs(got - target) / target < 0.5, (
        f"{name} could not be sized near {target:,}; got {got:,}"
    )


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_architecture_trains_for_a_few_steps(name):
    """End-to-end: the shared trainer must drive every model, not just the graph ones."""
    from neuralmesh import TrainConfig, evaluate, generate_dataset, train_model

    ds = generate_dataset(n_samples=8, nx=6, ny=6, seed=1).normalised()
    model = build_model(name, _cfg(hidden_dim=16, n_blocks=2, grid_resolution=8))
    before, _ = evaluate(model, [TensorGraph(g) for g in ds.test])
    model, _ = train_model(
        model, ds.train, ds.val, TrainConfig(epochs=5, seed=0), verbose=False
    )
    after, _ = evaluate(model, [TensorGraph(g) for g in ds.test])
    assert np.isfinite(after)
    assert after < before * 3.0, f"{name} diverged: {before:.3e} -> {after:.3e}"
