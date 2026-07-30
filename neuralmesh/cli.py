"""Command line interface.

    neuralmesh verify                 check the solver against known answers
    neuralmesh solve                  solve one case and print a summary
    neuralmesh dataset --samples 200  generate and cache a dataset
    neuralmesh train --arch ...       train a single architecture
    neuralmesh study                  run the under-reaching comparison

``verify`` is first on purpose. Every learned number in this package is measured
against this solver's output, so the solver is the thing that has to be right, and it
should be possible to check that in one command without reading any code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _verify(args: argparse.Namespace) -> int:
    """Convergence rate and an exactness check, printed as a pass or fail table."""
    from .fem.poisson import solve_poisson
    from .mesh.geometry import unit_square_mesh

    print("finite element verification")
    print("-" * 62)

    def exact(p):
        return np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])

    def source(p):
        return 2.0 * np.pi**2 * exact(p)

    errors, sizes = [], []
    for n in (9, 17, 33):
        mesh = unit_square_mesh(n, jitter=0.0)
        sol = solve_poisson(mesh, source=source, dirichlet_value=exact)
        err = sol.l2_error(exact)
        errors.append(err)
        sizes.append(1.0 / (n - 1))
        print(f"  n={n:<4d} nodes={mesh.n_nodes:<6d} h={sizes[-1]:.4f}   L2 error {err:.3e}")

    rates = [
        float(np.log(errors[i] / errors[i + 1]) / np.log(sizes[i] / sizes[i + 1]))
        for i in range(len(errors) - 1)
    ]
    print(f"  observed convergence rates: {[round(r, 3) for r in rates]}")
    ok_rate = all(1.7 < r < 2.4 for r in rates)
    print(f"  {'PASS' if ok_rate else 'FAIL'}  second-order convergence (theory: 2)")

    def linear(p):
        return 2.0 * p[:, 0] - 3.0 * p[:, 1] + 1.0

    mesh = unit_square_mesh(12, jitter=0.25, seed=3)
    exact_err = solve_poisson(mesh, source=0.0, dirichlet_value=linear).max_error(linear)
    ok_exact = exact_err < 1e-12
    print(
        f"  {'PASS' if ok_exact else 'FAIL'}  linear field reproduced exactly "
        f"(max error {exact_err:.2e})"
    )

    print("-" * 62)
    both = ok_rate and ok_exact
    print("solver verified" if both else "SOLVER VERIFICATION FAILED")
    return 0 if both else 1


def _solve(args: argparse.Namespace) -> int:
    from .fem.poisson import solve_poisson
    from .mesh.geometry import rectangle_mesh
    from .mesh.graph import graph_diameter, graph_from_solution

    mesh = rectangle_mesh(args.nx, args.ny, width=args.width, jitter=args.jitter)
    sol = solve_poisson(mesh, source=args.source, dirichlet_value=0.0)
    graph = graph_from_solution(sol)
    interior = np.setdiff1d(np.arange(mesh.n_nodes), mesh.boundary_nodes)

    print(f"nodes            {mesh.n_nodes}")
    print(f"cells            {len(mesh.triangles)}")
    print(f"boundary nodes   {len(mesh.boundary_nodes)}")
    print(f"directed edges   {graph.edge_index.shape[1]}")
    print(f"graph diameter   {graph_diameter(graph)}")
    print(f"u range          [{sol.u.min():.5f}, {sol.u.max():.5f}]")
    print(f"interior mean    {sol.u[interior].mean():.5f}")
    return 0


def _dataset(args: argparse.Namespace) -> int:
    from .data.dataset import load_or_generate

    ds = load_or_generate(
        cache_dir=args.cache,
        n_samples=args.samples,
        nx=args.nx,
        ny=args.ny,
        jitter=args.jitter,
        seed=args.seed,
    )
    print(ds.summary())
    print(f"nodes per graph  {ds.train[0].n_nodes}")
    print(f"cached under     {args.cache}")
    return 0


def _train(args: argparse.Namespace) -> int:
    from .data.dataset import load_or_generate
    from .models.architectures import ModelConfig, build_model, count_parameters
    from .train.trainer import TensorGraph, TrainConfig, evaluate, save_run, train_model

    ds = load_or_generate(
        cache_dir=args.cache, n_samples=args.samples, nx=args.nx, ny=args.ny, seed=args.seed
    ).normalised()

    cfg = ModelConfig(hidden_dim=args.hidden, n_blocks=args.blocks)
    model = build_model(args.arch, cfg)
    print(
        f"{args.arch}: {count_parameters(model):,} parameters, "
        f"receptive field {model.receptive_hops} hops"
    )

    tcfg = TrainConfig(
        epochs=args.epochs, lr=args.lr, physics_weight=args.physics_weight, seed=args.seed
    )
    model, history = train_model(model, ds.train, ds.val, tcfg)
    mse, rel = evaluate(model, [TensorGraph(g) for g in ds.test])
    print(f"\ntest masked MSE      {mse:.4e}")
    print(f"test relative L2     {rel:.4f}")

    if args.save:
        path = save_run(
            args.save,
            model,
            history,
            cfg.to_dict(),
            tcfg.to_dict(),
            extra={"architecture": args.arch, "test_mse": mse, "test_rel_l2": rel},
        )
        print(f"saved {path} and its json sidecar")
    return 0


def _study(args: argparse.Namespace) -> int:
    from .evaluate.underreach import run_underreach_study, save_result

    result = run_underreach_study(
        aspect_ratio=args.aspect,
        n_samples=args.samples,
        ny=args.ny,
        epochs=args.epochs,
        shallow_blocks=args.shallow,
        deep_blocks=args.deep,
        hidden_dim=args.hidden,
        seed=args.seed,
        verbose=not args.quiet,
    )
    print()
    print(result.table())
    if args.out:
        print(f"\nwrote {save_result(result, args.out)}")
    else:
        print()
        print(json.dumps(result.to_dict()["results"][-1], indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="neuralmesh", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("verify", help="check the solver against known answers")
    p.set_defaults(func=_verify)

    p = sub.add_parser("solve", help="solve one diffusion case and print a summary")
    p.add_argument("--nx", type=int, default=16)
    p.add_argument("--ny", type=int, default=16)
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--jitter", type=float, default=0.2)
    p.add_argument("--source", type=float, default=1.0)
    p.set_defaults(func=_solve)

    p = sub.add_parser("dataset", help="generate and cache a dataset")
    p.add_argument("--samples", type=int, default=120)
    p.add_argument("--nx", type=int, default=14)
    p.add_argument("--ny", type=int, default=14)
    p.add_argument("--jitter", type=float, default=0.22)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", type=Path, default=Path(".cache"))
    p.set_defaults(func=_dataset)

    p = sub.add_parser("train", help="train one architecture")
    p.add_argument(
        "--arch",
        default="mesh_graph_transformer",
        choices=["node_mlp", "meshgraphnet", "mesh_graph_transformer"],
    )
    p.add_argument("--samples", type=int, default=120)
    p.add_argument("--nx", type=int, default=14)
    p.add_argument("--ny", type=int, default=14)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--physics-weight",
        type=float,
        default=0.0,
        help="weight on the graph Dirichlet-energy regulariser; off by default",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", type=Path, default=Path(".cache"))
    p.add_argument("--save", type=Path, default=None)
    p.set_defaults(func=_train)

    p = sub.add_parser("study", help="run the under-reaching comparison")
    p.add_argument("--aspect", type=float, default=8.0)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--ny", type=int, default=6)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--shallow", type=int, default=4)
    p.add_argument("--deep", type=int, default=16)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=_study)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
