"""Repeat the comparison over several seeds, so the headline gets an error bar.

    python examples/seed_study.py --seeds 5 --aspect 8

The single-seed result in the README is suggestive and nothing more. A difference
between two architectures is only worth reporting if it survives reseeding, because
weight initialisation and data ordering both move the number.

This runs the same four configurations at several seeds and reports, for each
architecture, the mean and sample standard deviation, plus a paired comparison against
the parameter-matched baseline. Paired rather than unpaired: every seed trains all four
models on the same data in the same order, so the seed-to-seed variation is shared and
differencing within a seed removes it. With five seeds an unpaired test would have
almost no power; the paired one has a chance.

No p-value is printed. At five seeds a normal-theory p-value would be theatre. What is
printed instead is the per-seed difference, its mean, its standard deviation, and the
count of seeds where the sign agrees, which is the honest summary at this sample size.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from neuralmesh.evaluate.underreach import run_underreach_study

BASELINE_PREFIX = "MeshGraphNet L="
TRANSFORMER = "MGN-Transformer"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--aspect", type=float, default=8.0)
    ap.add_argument("--samples", type=int, default=70)
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--ny", type=int, default=6)
    ap.add_argument("--shallow", type=int, default=4)
    ap.add_argument("--deep", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("results/seeds.json"))
    args = ap.parse_args(argv)

    per_seed: list[dict] = []
    for seed in range(args.seeds):
        print(f"\n=== seed {seed} " + "=" * 52)
        r = run_underreach_study(
            aspect_ratio=args.aspect,
            n_samples=args.samples,
            ny=args.ny,
            epochs=args.epochs,
            shallow_blocks=args.shallow,
            deep_blocks=args.deep,
            hidden_dim=args.hidden,
            seed=seed,
            verbose=False,
        )
        print(r.table())
        per_seed.append(
            {
                "seed": seed,
                "graph_diameter": r.graph_diameter,
                "n_nodes": r.n_nodes,
                "rel_l2": {a.label: a.test_rel_l2 for a in r.results},
                "deep_band": {a.label: a.rel_l2_by_band[-1] for a in r.results},
                "params": {a.label: a.n_parameters for a in r.results},
            }
        )

    labels = list(per_seed[0]["rel_l2"].keys())

    print("\n" + "=" * 74)
    print(
        f"{args.seeds} seeds, aspect {args.aspect:g}, diameter {per_seed[0]['graph_diameter']}"
    )
    print("=" * 74)
    print(f"{'architecture':<26}{'params':>9}{'mean relL2':>12}{'std':>9}{'min':>9}{'max':>9}")
    print("-" * 74)

    summary: dict[str, dict] = {}
    for lab in labels:
        vals = [s["rel_l2"][lab] for s in per_seed]
        deep = [s["deep_band"][lab] for s in per_seed]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary[lab] = {
            "params": per_seed[0]["params"][lab],
            "mean": statistics.fmean(vals),
            "std": sd,
            "min": min(vals),
            "max": max(vals),
            "values": vals,
            "deep_mean": statistics.fmean(deep),
            "deep_std": statistics.stdev(deep) if len(deep) > 1 else 0.0,
            "deep_values": deep,
        }
        print(
            f"{lab:<26}{summary[lab]['params']:>9,}{summary[lab]['mean']:>12.4f}"
            f"{sd:>9.4f}{min(vals):>9.4f}{max(vals):>9.4f}"
        )

    # paired comparison: transformer against the shallow parameter-matched baseline
    shallow = next(
        (x for x in labels if x.startswith(BASELINE_PREFIX) and f"={args.shallow}" in x), None
    )
    trans = next((x for x in labels if TRANSFORMER in x), None)

    paired = None
    if shallow and trans:
        diffs = [
            s["rel_l2"][shallow] - s["rel_l2"][trans] for s in per_seed
        ]  # positive means the transformer is better
        rel = [100.0 * d / s["rel_l2"][shallow] for d, s in zip(diffs, per_seed)]
        agree = sum(1 for d in diffs if d > 0)
        paired = {
            "baseline": shallow,
            "candidate": trans,
            "per_seed_improvement_pct": rel,
            "mean_improvement_pct": statistics.fmean(rel),
            "std_improvement_pct": statistics.stdev(rel) if len(rel) > 1 else 0.0,
            "seeds_favouring_candidate": agree,
            "n_seeds": len(diffs),
        }
        print("\npaired comparison, within each seed")
        print(f"  {trans} against {shallow}")
        print("  per-seed improvement: " + ", ".join(f"{x:+.1f}%" for x in rel))
        print(
            f"  mean {paired['mean_improvement_pct']:+.1f}%  "
            f"std {paired['std_improvement_pct']:.1f}%  "
            f"sign agrees on {agree}/{len(diffs)} seeds"
        )
        if agree != len(diffs):
            print(
                "  NOTE: the sign does not agree on every seed. The effect is not "
                "robust at this budget."
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "config": vars(args) | {"out": str(args.out)},
                "per_seed": per_seed,
                "summary": summary,
                "paired": paired,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
