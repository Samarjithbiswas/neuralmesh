"""Reproduce the under-reaching study that the README reports.

    python examples/run_underreach.py                  # the headline configuration
    python examples/run_underreach.py --quick          # a few minutes, for a smoke test
    python examples/run_underreach.py --sweep 4 8 16   # error against graph diameter

Everything runs on CPU. The headline configuration takes on the order of an hour on a
laptop; ``--quick`` exists so the code path can be checked without waiting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuralmesh.evaluate.underreach import run_underreach_study, save_result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aspect", type=float, default=8.0, help="strip aspect ratio")
    ap.add_argument("--samples", type=int, default=100, help="number of solved cases")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--ny", type=int, default=6, help="nodes across the strip")
    ap.add_argument("--shallow", type=int, default=4, help="blocks in the shallow models")
    ap.add_argument("--deep", type=int, default=16, help="blocks in the deep baseline")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--quick", action="store_true", help="tiny run, for checking plumbing")
    ap.add_argument(
        "--sweep",
        type=float,
        nargs="+",
        default=None,
        metavar="ASPECT",
        help="run several aspect ratios and report error against diameter",
    )
    args = ap.parse_args(argv)

    if args.quick:
        args.samples, args.epochs, args.deep = 30, 20, 8

    aspects = args.sweep if args.sweep else [args.aspect]
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []

    for aspect in aspects:
        result = run_underreach_study(
            aspect_ratio=aspect,
            n_samples=args.samples,
            ny=args.ny,
            epochs=args.epochs,
            shallow_blocks=args.shallow,
            deep_blocks=args.deep,
            hidden_dim=args.hidden,
            seed=args.seed,
            verbose=True,
        )
        print()
        print(result.table())
        print()

        path = save_result(result, args.out / f"underreach_aspect{aspect:g}.json")
        print(f"wrote {path}")
        summary.append(
            {
                "aspect_ratio": aspect,
                "n_nodes": result.n_nodes,
                "graph_diameter": result.graph_diameter,
                "rel_l2": {r.label: r.test_rel_l2 for r in result.results},
            }
        )

    if len(summary) > 1:
        (args.out / "sweep.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\ndiameter sweep")
        print(f"{'aspect':>8}{'nodes':>8}{'diam':>7}   relative L2 by architecture")
        for row in summary:
            cells = "  ".join(f"{k.split()[0][:6]}={v:.3f}" for k, v in row["rel_l2"].items())
            head = f"{row['aspect_ratio']:>8g}{row['n_nodes']:>8}{row['graph_diameter']:>7}"
            print(f"{head}   {cells}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
