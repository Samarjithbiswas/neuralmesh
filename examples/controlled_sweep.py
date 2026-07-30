r"""Diameter sweep with node count held fixed.

    python examples/controlled_sweep.py --nodes 420

Why this script exists, and why the earlier sweep is not enough.

The obvious way to vary graph diameter is to make the strip longer at fixed
cross-section. That is what ``run_underreach.py`` does, and it has a confound that
invalidates any claim about a diameter *trend*: a longer strip has more nodes, so as
distance-to-travel goes up, the number of supervised nodes per training graph goes up
with it. More training signal and more required reach arrive together, and no amount of
staring at the resulting curve separates them.

Worse, the two effects push in opposite directions, so a flat curve is not evidence of
no effect. It is equally consistent with under-reaching being exactly cancelled by the
extra supervision.

The fix is to trade width for length. Choosing ``(aspect_ratio, ny)`` pairs that keep
``nx * ny`` roughly constant makes the strip longer *and* thinner at the same time, so
node count stays fixed while diameter grows. Now the only thing changing is the distance
information has to travel, which is the variable the hypothesis is actually about.

The cost is that the strip gets genuinely thinner, so the cross-sectional resolution
drops. That is a real limitation and it is reported rather than hidden: at ``ny = 4``
there are only two interior nodes across the strip.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuralmesh.evaluate.underreach import run_underreach_study, save_result


def pairs_for(target_nodes: int, ny_values: tuple[int, ...]) -> list[tuple[float, int, int]]:
    """Choose ``(aspect_ratio, ny, predicted_nodes)`` holding ``nx * ny`` near target."""
    out = []
    for ny in ny_values:
        nx = max(int(round(target_nodes / ny)), ny + 1)
        aspect = (nx - 1) / (ny - 1)
        out.append((round(aspect, 4), ny, nx * ny))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nodes", type=int, default=420, help="node count to hold fixed")
    ap.add_argument(
        "--ny",
        type=int,
        nargs="+",
        default=[12, 10, 8, 7, 6, 5, 4],
        help="cross-stream node counts; fewer means longer and thinner",
    )
    ap.add_argument("--samples", type=int, default=70)
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--shallow", type=int, default=4)
    ap.add_argument("--deep", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results_controlled"))
    args = ap.parse_args(argv)

    plan = pairs_for(args.nodes, tuple(args.ny))
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"holding node count near {args.nodes}, trading width for length")
    print(f"{'ny':>4}{'aspect':>10}{'predicted nodes':>17}")
    for aspect, ny, n in plan:
        print(f"{ny:>4}{aspect:>10.3f}{n:>17}")
    print(flush=True)

    rows = []
    for aspect, ny, _ in plan:
        r = run_underreach_study(
            aspect_ratio=aspect,
            ny=ny,
            n_samples=args.samples,
            epochs=args.epochs,
            shallow_blocks=args.shallow,
            deep_blocks=args.deep,
            hidden_dim=args.hidden,
            seed=args.seed,
            verbose=False,
        )
        print(r.table(), flush=True)
        print(flush=True)
        save_result(r, args.out / f"controlled_ny{ny}.json")
        rows.append(
            {
                "ny": ny,
                "aspect_ratio": aspect,
                "diameter": r.graph_diameter,
                "nodes": r.n_nodes,
                "rel_l2": {a.label: a.test_rel_l2 for a in r.results},
                "deep_band": {a.label: a.rel_l2_by_band[-1] for a in r.results},
                "bands": {a.label: a.rel_l2_by_band for a in r.results},
            }
        )

    (args.out / "controlled.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    labels = list(rows[0]["rel_l2"].keys())
    spread = max(r["nodes"] for r in rows) - min(r["nodes"] for r in rows)
    print("=" * 84)
    print(
        f"node count held between {min(r['nodes'] for r in rows)} and "
        f"{max(r['nodes'] for r in rows)}, a spread of {spread}"
    )
    print("=" * 84)

    for title, key in (
        ("aggregate relative L2", "rel_l2"),
        ("deep band, mid-strip", "deep_band"),
    ):
        print(f"\n{title}")
        print(f"{'diam':>6}{'nodes':>7}" + "".join(f"{x.split()[0][:11]:>13}" for x in labels))
        for row in rows:
            print(
                f"{row['diameter']:>6}{row['nodes']:>7}"
                + "".join(f"{row[key][x]:>13.4f}" for x in labels)
            )

    print("\nchange from smallest to largest diameter, at fixed node count")
    for x in labels:
        a, b = rows[0]["rel_l2"][x], rows[-1]["rel_l2"][x]
        da, db = rows[0]["deep_band"][x], rows[-1]["deep_band"][x]
        print(
            f"  {x:26s} aggregate {100 * (b - a) / a:+7.1f}%"
            f"   deep band {100 * (db - da) / da:+7.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
