"""Run the under-reaching study on the 3D nonlinear problem.

    python examples/run_underreach3d.py --sweep 4 8 12

This is the experiment that decides whether the 2D finding was an artefact of a
deliberately clean setting. Same protocol, harder problem.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuralmesh.evaluate.underreach3d import run_underreach_study_3d, save_result_3d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep", type=float, nargs="+", default=[4.0, 8.0, 12.0])
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--cross", type=int, default=4)
    ap.add_argument("--shallow", type=int, default=4)
    ap.add_argument("--deep", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for aspect in args.sweep:
        r = run_underreach_study_3d(
            aspect_ratio=aspect,
            n_samples=args.samples,
            n_cross=args.cross,
            epochs=args.epochs,
            shallow_blocks=args.shallow,
            deep_blocks=args.deep,
            hidden_dim=args.hidden,
            seed=args.seed,
            verbose=False,
        )
        print()
        print(r.table())
        save_result_3d(r, args.out / f"underreach3d_aspect{aspect:g}.json")
        rows.append(
            {
                "aspect": aspect,
                "diameter": r.graph_diameter,
                "nodes": r.n_nodes,
                "rel_l2": {a.label: a.test_rel_l2 for a in r.results},
                "deep_band": {a.label: a.rel_l2_by_band[-1] for a in r.results},
            }
        )

    (args.out / "sweep3d.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("aggregate relative L2 against graph diameter")
    print("=" * 78)
    labels = list(rows[0]["rel_l2"].keys())
    print(f"{'diameter':>9}" + "".join(f"{lab.split()[0][:12]:>14}" for lab in labels))
    for row in rows:
        print(
            f"{row['diameter']:>9}" + "".join(f"{row['rel_l2'][lab]:>14.4f}" for lab in labels)
        )

    print("\ndeepest band only, the middle of the bar")
    print(f"{'diameter':>9}" + "".join(f"{lab.split()[0][:12]:>14}" for lab in labels))
    for row in rows:
        print(
            f"{row['diameter']:>9}"
            + "".join(f"{row['deep_band'][lab]:>14.4f}" for lab in labels)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
