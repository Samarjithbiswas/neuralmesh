"""Training, dataset and study tests.

The dataset tests are the ones worth reading. Two of them check for the mistakes that
make a surrogate paper wrong while making its numbers look better:

* normalisation statistics must be fitted on the training split only
* every case must get its own mesh, so a model cannot memorise one node ordering
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neuralmesh import (
    PARAM_BOUNDS,
    Dataset,
    ModelConfig,
    Normaliser,
    TensorGraph,
    TrainConfig,
    build_model,
    evaluate,
    generate_dataset,
    latin_hypercube,
    masked_mse,
    relative_l2,
    scale_to_bounds,
    train_model,
)
from neuralmesh.evaluate.underreach import run_underreach_study, strip_dataset


# --------------------------------------------------------------------- sampling
def test_latin_hypercube_is_stratified_in_every_dimension():
    """One sample per stratum per dimension, which random sampling only achieves on average."""
    n, d = 40, 5
    u = latin_hypercube(n, d, seed=1)
    assert u.shape == (n, d)
    assert u.min() >= 0.0 and u.max() <= 1.0
    for j in range(d):
        strata = np.floor(u[:, j] * n).astype(int)
        assert len(np.unique(strata)) == n, f"dimension {j} is not stratified"


def test_latin_hypercube_is_reproducible_and_seed_sensitive():
    assert np.allclose(latin_hypercube(12, 3, seed=7), latin_hypercube(12, 3, seed=7))
    assert not np.allclose(latin_hypercube(12, 3, seed=7), latin_hypercube(12, 3, seed=8))


def test_latin_hypercube_rejects_degenerate_sizes():
    with pytest.raises(ValueError):
        latin_hypercube(0, 3)
    with pytest.raises(ValueError):
        latin_hypercube(5, 0)


def test_scale_to_bounds_respects_the_box():
    u = latin_hypercube(30, len(PARAM_BOUNDS), seed=2)
    p = scale_to_bounds(u, PARAM_BOUNDS)
    assert (p >= PARAM_BOUNDS[:, 0]).all()
    assert (p <= PARAM_BOUNDS[:, 1]).all()


def test_scale_to_bounds_rejects_inverted_or_mismatched_bounds():
    u = latin_hypercube(5, 2, seed=0)
    with pytest.raises(ValueError):
        scale_to_bounds(u, np.array([[1.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        scale_to_bounds(u, np.array([[0.0, 1.0]]))


# ---------------------------------------------------------------------- dataset
@pytest.fixture(scope="module")
def small_dataset():
    return generate_dataset(n_samples=14, nx=7, ny=7, jitter=0.15, seed=0)


def test_splits_are_non_empty_and_disjoint_in_size(small_dataset):
    ds = small_dataset
    assert len(ds.train) and len(ds.val) and len(ds.test)
    assert len(ds.train) + len(ds.val) + len(ds.test) == 14


def test_every_case_gets_its_own_mesh(small_dataset):
    """Varying the mesh is what stops the model memorising a node ordering."""
    coords = [g.positions.tobytes() for g in small_dataset.train]
    assert len(set(coords)) == len(coords), "meshes were reused across samples"


def test_targets_actually_vary_between_cases(small_dataset):
    """If the parameter sweep produced near-identical fields there is nothing to learn."""
    spread = np.std([g.target.mean() for g in small_dataset.train])
    assert spread > 1e-6


def test_normaliser_is_fitted_on_training_data_only(small_dataset):
    """Fitting on everything leaks distribution information and flatters the error.

    The check: statistics computed from the training split alone must differ from
    statistics computed over all three splits. If they matched, the split would not be
    doing anything.
    """
    ds = small_dataset
    train_only = Normaliser.fit(ds.train)
    everything = Normaliser.fit(ds.train + ds.val + ds.test)
    assert not np.isclose(train_only.target_mean, everything.target_mean, rtol=1e-12)
    assert np.allclose(train_only.node_mean, ds.normaliser.node_mean)


def test_normalised_training_features_are_standardised(small_dataset):
    nd = small_dataset.normalised()
    stacked = np.vstack([g.node_features for g in nd.train])
    assert np.allclose(stacked.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(stacked.std(axis=0), 1.0, atol=1e-6)


def test_normalisation_round_trips(small_dataset):
    ds = small_dataset
    g = ds.train[0]
    norm = ds.normaliser.apply(g)
    assert np.allclose(ds.normaliser.invert_target(norm.target), g.target, atol=1e-9)


def test_generate_dataset_rejects_a_split_that_empties_a_partition():
    with pytest.raises(ValueError):
        generate_dataset(n_samples=2, nx=5, ny=5, splits=(1.0, 0.0))


def test_dataset_is_reproducible_for_a_fixed_seed():
    a = generate_dataset(n_samples=6, nx=5, ny=5, seed=3)
    b = generate_dataset(n_samples=6, nx=5, ny=5, seed=3)
    assert np.allclose(a.train[0].target, b.train[0].target)


# ----------------------------------------------------------------------- losses
def test_masked_mse_ignores_unmasked_nodes():
    pred = torch.tensor([0.0, 5.0, 0.0])
    targ = torch.tensor([0.0, 0.0, 0.0])
    mask = torch.tensor([True, False, True])
    assert masked_mse(pred, targ, mask).item() == pytest.approx(0.0)


def test_masked_mse_rejects_an_empty_mask():
    with pytest.raises(ValueError):
        masked_mse(torch.zeros(3), torch.zeros(3), torch.zeros(3, dtype=torch.bool))


def test_relative_l2_is_scale_invariant():
    """Dimensionless by construction, which is why it is comparable across cases."""
    targ = torch.tensor([1.0, 2.0, 3.0])
    pred = torch.tensor([1.1, 1.9, 3.2])
    a = relative_l2(pred, targ)
    b = relative_l2(pred * 1000.0, targ * 1000.0)
    assert a == pytest.approx(b, rel=1e-6)


def test_relative_l2_is_zero_for_a_perfect_prediction():
    t = torch.tensor([1.0, -2.0, 0.5])
    assert relative_l2(t, t) == pytest.approx(0.0, abs=1e-7)


def test_boundary_nodes_are_excluded_from_the_loss(small_dataset):
    """Boundary values are inputs. Scoring them inflates every metric."""
    tg = TensorGraph(small_dataset.normalised().train[0])
    assert tg.interior.sum() < tg.node_features.shape[0]
    assert tg.interior.sum() > 0


# ---------------------------------------------------------------------- training
def test_training_reduces_the_loss_and_restores_the_best_checkpoint():
    ds = generate_dataset(n_samples=12, nx=6, ny=6, seed=1).normalised()
    # Seed before constructing the model. train_model seeds internally, but weight
    # initialisation happens first and draws from the global RNG, so without this the
    # starting point depends on whichever test ran previously and the assertion below
    # becomes order-dependent.
    torch.manual_seed(0)
    model = build_model("meshgraphnet", ModelConfig(hidden_dim=24, n_blocks=2))
    before, _ = evaluate(model, [TensorGraph(g) for g in ds.test])
    model, history = train_model(
        model, ds.train, ds.val, TrainConfig(epochs=30, seed=0), verbose=False
    )
    after, _ = evaluate(model, [TensorGraph(g) for g in ds.test])
    assert after < before, f"loss did not fall: {before:.4e} -> {after:.4e}"
    assert len(history.val_loss) == 30
    # the restored weights must be the best epoch, not merely the last one
    assert history.val_loss[history.best_epoch()] == pytest.approx(min(history.val_loss))


def test_physics_term_runs_and_changes_the_result():
    """The residual is opt-in, and a weight of zero must be a genuine no-op path."""
    ds = generate_dataset(n_samples=10, nx=6, ny=6, seed=2).normalised()
    outs = []
    for weight in (0.0, 0.05):
        model = build_model("meshgraphnet", ModelConfig(hidden_dim=16, n_blocks=2))
        torch.manual_seed(0)
        model, _ = train_model(
            model,
            ds.train,
            ds.val,
            TrainConfig(epochs=6, seed=0, physics_weight=weight),
            verbose=False,
        )
        outs.append(evaluate(model, [TensorGraph(g) for g in ds.test])[0])
    assert outs[0] != outs[1]


def test_training_requires_both_splits():
    ds = generate_dataset(n_samples=8, nx=5, ny=5, seed=0).normalised()
    model = build_model("node_mlp", ModelConfig(hidden_dim=8, n_blocks=1))
    with pytest.raises(ValueError):
        train_model(model, ds.train, [], TrainConfig(epochs=1), verbose=False)


def test_unknown_scheduler_is_rejected():
    ds = generate_dataset(n_samples=8, nx=5, ny=5, seed=0).normalised()
    model = build_model("node_mlp", ModelConfig(hidden_dim=8, n_blocks=1))
    with pytest.raises(ValueError):
        train_model(
            model,
            ds.train,
            ds.val,
            TrainConfig(epochs=1, scheduler="cosine_but_wrong"),
            verbose=False,
        )


# ------------------------------------------------------------------- the study
def test_strip_dataset_diameter_grows_with_aspect_ratio():
    from neuralmesh import graph_diameter

    short = strip_dataset(10, aspect_ratio=3.0, ny=5, seed=0)
    long_ = strip_dataset(10, aspect_ratio=12.0, ny=5, seed=0)
    assert graph_diameter(long_.train[0]) > graph_diameter(short.train[0])


def test_strip_dataset_refuses_to_leave_an_empty_test_split():
    with pytest.raises(ValueError):
        strip_dataset(1, aspect_ratio=4.0, ny=5)


@pytest.mark.slow
def test_study_runs_and_reports_all_four_configurations():
    """A cheap end-to-end pass. It checks plumbing, not the scientific claim."""
    result = run_underreach_study(
        aspect_ratio=5.0,
        n_samples=14,
        ny=5,
        epochs=6,
        shallow_blocks=2,
        deep_blocks=5,
        hidden_dim=16,
        verbose=False,
    )
    assert len(result.results) == 4
    assert result.graph_diameter > 0
    labels = [r.label for r in result.results]
    assert any("no comms" in x for x in labels)
    assert any("Transformer" in x for x in labels)

    for r in result.results:
        assert np.isfinite(r.test_rel_l2)
        assert len(r.rel_l2_by_band) == 4

    # the transformer is parameter-matched to the shallow baseline, not merely bigger
    by_label = {r.label: r for r in result.results}
    shallow = next(v for k, v in by_label.items() if k.startswith("MeshGraphNet L=2"))
    trans = next(v for k, v in by_label.items() if "Transformer" in k)
    assert abs(trans.n_parameters - shallow.n_parameters) / shallow.n_parameters < 0.25
    assert trans.receptive_hops == -1 and shallow.receptive_hops == 2

    assert "relL2" in result.table()
    assert isinstance(Dataset, type)
