"""Plot the committed benchmark output.

    python examples/plot_results.py                 # reads benchmarks/, writes docs/
    python examples/plot_results.py --results results --out docs

Two panels, because the paper-style headline and the honest caveat are different plots:

left    aggregate relative L2 against graph diameter, one line per architecture
right   deep-interior error against graph diameter

The point of showing both is that the left panel does *not* show the reach-limited
models degrading with diameter, and the right panel does. Reporting only the left would
hide the effect; reporting only the right would overstate how visible it is.

Needs matplotlib, which is an optional dependency:

    pip install -e ".[plot]"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BG = "#0b1016"
PANEL = "#0e151d"
INK = "#e6edf3"
MUTED = "#8b949e"
DIM = "#6e7f91"

STYLE = {
    "node MLP": ("#8b949e", ":", "o"),
    "MeshGraphNet L=4": ("#60a5fa", "-", "s"),
    "MeshGraphNet L=16": ("#a78bfa", "-", "^"),
    "MGN-Transformer": ("#34d399", "-", "D"),
}


def key_for(label: str) -> str:
    for k in STYLE:
        if label.startswith(k.split()[0]) and (k.split("=")[-1] in label or "=" not in k):
            if k.startswith("MeshGraphNet") and f"L={k.split('=')[-1]}" not in label:
                continue
            return k
    return label


def load(results: Path) -> list[dict]:
    runs = []
    for f in sorted(results.glob("underreach_aspect*.json")):
        runs.append(json.loads(f.read_text(encoding="utf-8")))
    runs.sort(key=lambda d: d["graph_diameter"])
    if not runs:
        raise SystemExit(f"no underreach_aspect*.json found in {results}")
    return runs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("benchmarks"))
    ap.add_argument("--out", type=Path, default=Path("docs"))
    args = ap.parse_args(argv)

    try:
        import matplotlib
    except ImportError:
        raise SystemExit('matplotlib is needed: pip install -e ".[plot]"') from None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "savefig.facecolor": BG,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": "#26333f",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "grid.color": "#1b2630",
            "legend.frameon": False,
        }
    )

    runs = load(args.results)
    diameters = [r["graph_diameter"] for r in runs]

    series_agg: dict[str, list[float]] = {}
    series_deep: dict[str, list[float]] = {}
    for r in runs:
        for a in r["results"]:
            k = key_for(a["label"])
            series_agg.setdefault(k, []).append(a["test_rel_l2"])
            series_deep.setdefault(k, []).append(a["rel_l2_by_band"][-1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.2), dpi=180)

    for ax, series, title, note, drop_control in (
        (
            ax1,
            series_agg,
            "Aggregate error: no diameter trend",
            "the reach-limited models do NOT degrade here",
            False,
        ),
        (
            ax2,
            series_deep,
            "Deep-interior error: the effect appears",
            "control omitted so the trends are legible; it sits at 0.44 to 0.73",
            True,
        ),
    ):
        ax.grid(True, linewidth=0.6, alpha=0.5)
        for k, ys in series.items():
            # The control is a floor, not a trend. Left in the right-hand panel it
            # stretches the axis by a factor of three and flattens everything that the
            # panel exists to show.
            if drop_control and k.startswith("node MLP"):
                continue
            col, ls, mk = STYLE.get(k, (MUTED, "-", "o"))
            ax.plot(
                diameters,
                ys,
                color=col,
                linestyle=ls,
                marker=mk,
                linewidth=1.9,
                markersize=5.5,
                label=k,
            )
        ax.set_xlabel("graph diameter (hops across the mesh)")
        ax.set_ylabel(r"relative $L^2$ error")
        ax.set_xticks(diameters)
        ax.set_title(title, color=INK)
        ax.text(0.5, -0.30, note, transform=ax.transAxes, ha="center", color=DIM, fontsize=9.5)

    ax1.set_yscale("log")
    ax1.legend(loc="center right", labelcolor=MUTED, fontsize=9)
    ax2.legend(loc="lower right", labelcolor=MUTED, fontsize=9)

    fig.suptitle(
        "Under-reaching is invisible in the aggregate metric and visible by distance "
        "from the boundary",
        color=INK,
        fontsize=12.5,
        fontweight="bold",
        y=1.03,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "underreach.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"wrote {path}  {path.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
