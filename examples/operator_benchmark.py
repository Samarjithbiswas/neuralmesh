r"""All six architectures, parameter-matched, at two graph diameters.

    python examples/operator_benchmark.py

This is the experiment the controlled sweep made interesting.

The original question was whether global attention fixes under-reaching. The controlled
sweep answered no: extra receptive field stops buying anything as diameter grows
(MeshGraphNet L=16 beats L=4 by 59 percent at diameter 34 and by 2 percent at diameter
104), so reach is not the operative variable. Attention still wins, but by a margin that
*shrinks* with diameter rather than growing.

That leaves a better question. If reach is not what attention is buying, what is, and do
the other global mechanisms buy the same thing? The three published baselines each get
global coupling by a different route:

* FNO gets it from the spectral transform, where every mode spans the domain.
* DeepONet gets it from pooling the whole input into a coefficient vector.
* GNO gets it from a radius neighbourhood, so reach scales with radius rather than depth.

If all four global models track each other across diameter, the shared property is
globality and the mechanism does not matter much. If they diverge, the divergence
identifies what actually matters, and that is a more specific and more publishable claim
than the one this repository started with.

Two diameters, chosen from the controlled sweep: 34, where attention's advantage is
largest, and 104, where it has nearly evaporated. Node count is held near 420 at both,
so the comparison is not re-confounded.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from neuralmesh.evaluate.underreach import strip_dataset
from neuralmesh.mesh.graph import graph_diameter
from neuralmesh.models.architectures import (
    ModelConfig,
    build_model,
    count_parameters,
    match_capacity,
)
from neuralmesh.train.trainer import TensorGraph, TrainConfig, evaluate, train_model

# (ny, aspect) pairs holding nodes near 420, taken from the controlled sweep
CONFIGS = [(12, 3.0909), (4, 34.6667)]

PLAN = [
    ("node_mlp", "control (no comms)"),
    ("meshgraphnet", "MeshGraphNet L=4"),
    ("mesh_graph_transformer", "MGN-Transformer L=4"),
    ("fno", "FNO"),
    ("deeponet", "DeepONet"),
    ("gno", "GNO"),
]


def run_one(ny: int, aspect: float, args, seed: int) -> dict:
    ds = strip_dataset(args.samples, aspect_ratio=aspect, ny=ny, seed=seed).normalised()
    diameter = graph_diameter(ds.train[0])
    n_nodes = ds.train[0].n_nodes
    test = [TensorGraph(g) for g in ds.test]

    base = ModelConfig(hidden_dim=args.hidden, n_blocks=args.blocks)
    target = count_parameters(build_model("meshgraphnet", base))

    row = {"ny": ny, "diameter": diameter, "nodes": n_nodes, "seed": seed, "models": {}}
    for name, label in PLAN:
        cfg = (
            base
            if name in ("node_mlp", "meshgraphnet")
            else match_capacity(name, target, base)
        )
        model = build_model(name, cfg)
        params = count_parameters(model)
        model, _ = train_model(
            model, ds.train, ds.val, TrainConfig(epochs=args.epochs, seed=seed), verbose=False
        )
        mse, rel = evaluate(model, test)
        entry = {"params": params, "rel_l2": rel, "mse": mse}
        if name == "fno":
            entry["resample_loss"] = getattr(model, "resample_loss", float("nan"))
        row["models"][label] = entry
        print(
            f"    {label:24s} {params:>9,}  relL2 {rel:.4f}"
            + (f"  (grid resample err {entry['resample_loss']:.3f})" if name == "fno" else ""),
            flush=True,
        )
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=int, default=70)
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("results_operators"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for ny, aspect in CONFIGS:
        for seed in range(args.seeds):
            print(f"\n=== ny {ny}, aspect {aspect:g}, seed {seed} ===", flush=True)
            rows.append(run_one(ny, aspect, args, seed))
    (args.out / "operators.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    labels = [lab for _, lab in PLAN]
    print("\n" + "=" * 86)
    print(f"all six architectures, {args.seeds} seeds, node count held near 420")
    print("=" * 86)
    print(
        f"{'architecture':<26}{'params':>10}"
        + "".join(f"{'d=' + str(d):>16}" for d in sorted({r["diameter"] for r in rows}))
    )

    diams = sorted({r["diameter"] for r in rows})
    summary = {}
    for lab in labels:
        cells = []
        for d in diams:
            vals = [r["models"][lab]["rel_l2"] for r in rows if r["diameter"] == d]
            m = statistics.fmean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0.0
            cells.append(f"{m:.4f}+/-{s:.3f}")
            summary.setdefault(lab, {})[d] = {"mean": m, "std": s, "values": vals}
        p = rows[0]["models"][lab]["params"]
        print(f"{lab:<26}{p:>10,}" + "".join(f"{c:>16}" for c in cells))

    print("\nadvantage over MeshGraphNet L=4, by diameter")
    basel = "MeshGraphNet L=4"
    for lab in labels:
        if lab == basel:
            continue
        parts = []
        for d in diams:
            b = summary[basel][d]["mean"]
            v = summary[lab][d]["mean"]
            parts.append(f"d={d}: {100 * (b - v) / b:+6.1f}%")
        print(f"  {lab:24s} " + "   ".join(parts))

    print(
        "\nIf the global models track each other, globality is the shared property.\n"
        "If they diverge, the divergence is the finding."
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
