"""Architecture tests, including the ones that make the experiment meaningful.

Two properties here are load-bearing for the whole repository:

* ``test_receptive_field_is_exactly_n_blocks`` measures reach empirically instead of
  trusting the ``receptive_hops`` property. If message passing actually travelled
  further or less far than claimed, the under-reaching result would be measuring
  something other than what it says.
* ``test_capacity_matching_lands_close`` guards the fairness of the comparison. An
  unmatched comparison measures capacity, not architecture.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neuralmesh import (
    MeshGraphNet,
    MeshGraphTransformer,
    ModelConfig,
    NodeMLP,
    TensorGraph,
    build_model,
    count_parameters,
    graph_from_solution,
    match_capacity,
    solve_poisson,
    unit_square_mesh,
)
from neuralmesh.models.blocks import PhysicsAttention, scatter_mean, scatter_sum

ARCH_NAMES = ["node_mlp", "meshgraphnet", "mesh_graph_transformer"]


@pytest.fixture(scope="module")
def graph():
    mesh = unit_square_mesh(9, jitter=0.15, seed=2)
    sol = solve_poisson(mesh, source=1.0, dirichlet_value=0.0)
    return TensorGraph(graph_from_solution(sol))


def test_scatter_sum_matches_a_loop():
    src = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    idx = torch.tensor([0, 0, 1, 2, 2, 2])
    got = scatter_sum(src, idx, 3)
    want = torch.zeros(3, 2)
    for i, j in enumerate(idx.tolist()):
        want[j] += src[i]
    assert torch.allclose(got, want)


def test_scatter_mean_leaves_empty_buckets_at_zero():
    src = torch.ones(3, 2)
    idx = torch.tensor([0, 0, 2])
    got = scatter_mean(src, idx, 4)
    assert torch.allclose(got[0], torch.ones(2))
    assert torch.allclose(got[1], torch.zeros(2))
    assert torch.allclose(got[3], torch.zeros(2))


@pytest.mark.parametrize("name", ARCH_NAMES)
def test_forward_pass_shape_and_finiteness(graph, name):
    model = build_model(name, ModelConfig(hidden_dim=32, n_blocks=3))
    out = graph.forward(model)
    assert out.shape == (graph.node_features.shape[0],)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", ARCH_NAMES)
def test_gradients_reach_every_parameter(graph, name):
    """A parameter that never receives a gradient is dead weight or a wiring bug."""
    model = build_model(name, ModelConfig(hidden_dim=32, n_blocks=2))
    graph.forward(model).sum().backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


@pytest.mark.parametrize("name", ARCH_NAMES)
def test_prediction_is_invariant_to_node_permutation(graph, name):
    """Relabelling nodes must not change the physics.

    A model that fails this has learned something about index order, which will not
    survive a new mesh.
    """
    torch.manual_seed(0)
    model = build_model(name, ModelConfig(hidden_dim=32, n_blocks=2)).eval()
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
    assert torch.allclose(base, shuffled[inv], atol=1e-5)


def test_receptive_hops_property_reports_the_design():
    assert NodeMLP(ModelConfig()).receptive_hops == 0
    assert MeshGraphNet(ModelConfig(n_blocks=7)).receptive_hops == 7
    # negative means global: attention removes the dependence on depth
    assert MeshGraphTransformer(ModelConfig(n_blocks=4)).receptive_hops == -1


@pytest.mark.parametrize("n_blocks", [1, 2, 4])
def test_receptive_field_is_exactly_n_blocks(n_blocks):
    """Measure reach instead of trusting the label.

    Perturb one node's input on a long path graph and find how far the output moves.
    A pure message-passing model must influence exactly ``n_blocks`` hops and no more.
    """
    n = 24
    pts = np.column_stack([np.linspace(0.0, 1.0, n), np.zeros(n)])
    a = np.arange(n - 1)
    edge_index = torch.tensor(
        np.vstack([np.concatenate([a, a + 1]), np.concatenate([a + 1, a])]), dtype=torch.long
    )
    d = pts[edge_index[1].numpy()] - pts[edge_index[0].numpy()]
    edge_features = torch.tensor(
        np.hstack([d, np.linalg.norm(d, axis=1, keepdims=True)]), dtype=torch.float32
    )
    positions = torch.tensor(pts, dtype=torch.float32)

    torch.manual_seed(0)
    model = build_model("meshgraphnet", ModelConfig(hidden_dim=24, n_blocks=n_blocks)).eval()

    nf = torch.zeros(n, 4)
    with torch.no_grad():
        base = model(nf, edge_index, edge_features, positions)
        bumped = nf.clone()
        bumped[0, 2] = 1.0  # perturb the source term at one end
        after = model(bumped, edge_index, edge_features, positions)

    moved = (after - base).abs() > 1e-6
    reach = int(moved.nonzero().max()) if moved.any() else -1
    assert reach == n_blocks, f"expected reach {n_blocks} hops, measured {reach}"


def test_node_mlp_has_no_reach_at_all():
    """The control model must be genuinely blind to its neighbours."""
    n = 12
    pts = np.column_stack([np.linspace(0.0, 1.0, n), np.zeros(n)])
    a = np.arange(n - 1)
    edge_index = torch.tensor(
        np.vstack([np.concatenate([a, a + 1]), np.concatenate([a + 1, a])]), dtype=torch.long
    )
    edge_features = torch.zeros(edge_index.shape[1], 3)
    positions = torch.tensor(pts, dtype=torch.float32)

    torch.manual_seed(0)
    model = build_model("node_mlp", ModelConfig(hidden_dim=16, n_blocks=3)).eval()
    nf = torch.zeros(n, 4)
    with torch.no_grad():
        base = model(nf, edge_index, edge_features, positions)
        bumped = nf.clone()
        bumped[0, 2] = 1.0
        after = model(bumped, edge_index, edge_features, positions)

    changed = (after - base).abs() > 1e-6
    assert changed[0], "the perturbed node itself should change"
    assert not changed[1:].any(), "a no-communication model must not affect any neighbour"


def test_transformer_reaches_globally_in_one_block():
    """Attention must move information across the whole graph immediately."""
    n = 30
    pts = np.column_stack([np.linspace(0.0, 1.0, n), np.zeros(n)])
    a = np.arange(n - 1)
    edge_index = torch.tensor(
        np.vstack([np.concatenate([a, a + 1]), np.concatenate([a + 1, a])]), dtype=torch.long
    )
    edge_features = torch.zeros(edge_index.shape[1], 3)
    positions = torch.tensor(pts, dtype=torch.float32)

    torch.manual_seed(0)
    cfg = ModelConfig(hidden_dim=32, n_blocks=1, attention_every=1)
    model = build_model("mesh_graph_transformer", cfg).eval()
    nf = torch.zeros(n, 4)
    with torch.no_grad():
        base = model(nf, edge_index, edge_features, positions)
        bumped = nf.clone()
        bumped[0, 2] = 1.0
        after = model(bumped, edge_index, edge_features, positions)

    moved = (after - base).abs() > 1e-8
    assert moved[-1], "one attention block should already reach the far end of the graph"


def test_attention_chunking_does_not_change_the_answer():
    """Chunking bounds memory; it must not alter the result."""
    torch.manual_seed(0)
    h = torch.randn(70, 32)
    pos = torch.randn(70, 2)
    block = PhysicsAttention(32, n_heads=4).eval()
    with torch.no_grad():
        block.chunk_size = None
        whole = block(h, pos)
        block.chunk_size = 16
        chunked = block(h, pos)
    assert torch.allclose(whole, chunked, atol=1e-5)


def test_attention_rejects_indivisible_head_count():
    with pytest.raises(ValueError):
        PhysicsAttention(30, n_heads=4)


def test_capacity_matching_lands_close():
    """Without this, the headline comparison measures width rather than design."""
    base_cfg = ModelConfig(hidden_dim=64, n_blocks=3)
    target = count_parameters(build_model("meshgraphnet", base_cfg))
    matched = match_capacity("mesh_graph_transformer", target, base_cfg)
    got = count_parameters(build_model("mesh_graph_transformer", matched))
    assert abs(got - target) / target < 0.15, f"target {target}, got {got}"


def test_deeper_meshgraphnet_has_more_parameters():
    small = count_parameters(build_model("meshgraphnet", ModelConfig(n_blocks=2)))
    large = count_parameters(build_model("meshgraphnet", ModelConfig(n_blocks=8)))
    assert large > small


def test_unknown_architecture_name_is_rejected():
    with pytest.raises(KeyError):
        build_model("transformer_but_spelled_wrong")
