r"""Plot direct-vs-composite hypervolume traces for one benchmark's results
directory (as written by `run_ablation.py --out-dir ...`): one line per
solver, SOLID for the direct variant and DASHED for its composite
counterpart, mean over trials with a shaded standard-error band -- the
same visual convention `tau315/composite-mobo`'s own `benchmark_common.py`
plotting uses (solid/dashed here in place of its two-color scheme, since
this plots every method pair for a benchmark on one axes rather than one
pair per panel).

Usage:
    python -m composite_ablation.plot_ablation composite_ablation/results/dtlz2_2obj_6d
    python -m composite_ablation.plot_ablation composite_ablation/results/dtlz2_2obj_6d \
        --output composite_ablation/results/dtlz2_2obj_6d/plot.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# One fixed color per solver family so the same method always reads the
# same color across every benchmark's plot; direct=solid, composite=dashed
# of that same color (not a second color) so the direct/composite contrast
# is always linestyle, never confusable with a different-method contrast.
_COLORS = {
    "standard_mobo / composite_mobo": "#1f77b4",
    "chebyshev_bo / composite_chebyshev_bo": "#ff7f0e",
    "spherical_chebyshev_bo / composite_spherical_chebyshev_bo": "#2ca02c",
}


def _mean_and_sem(traces: np.ndarray) -> tuple:
    # Ragged if a mid-pair timeout left trials of different lengths --
    # trim to the shortest completed trace so mean/SEM stay well-defined.
    # `traces` was saved with dtype=object (needed since a mid-run save can
    # itself be ragged); cast each row back to float64 explicitly, since an
    # object-dtype array of boxed np.float64 scalars fails np.sqrt/std.
    min_len = min(len(t) for t in traces)
    stacked = np.stack([np.asarray(t[:min_len], dtype=np.float64) for t in traces], axis=0)
    mean = stacked.mean(axis=0)
    sem = (
        stacked.std(axis=0, ddof=1) / np.sqrt(len(stacked))
        if len(stacked) > 1
        else np.zeros_like(mean)
    )
    return mean, sem


def plot_benchmark_dir(results_dir: Path, output: Path, title: str = None) -> None:
    npz_files = sorted(results_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"no .npz result files found in {results_dir}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    plotted_any = False
    for npz_path in npz_files:
        # "_-_" is the only substitution `run_ablation.py`'s
        # `name.replace(" ", "_").replace("/", "-")` introduces beyond
        # underscores already present in method names (standard_mobo,
        # chebyshev_bo, ...) -- reversing only that substring, not every
        # underscore, recovers the exact original pair name.
        name = npz_path.stem.replace("_-_", " / ")
        # Prefer the sibling per-pair summary.json when present (newer
        # runs only -- older ones wrote one combined summary.json instead).
        summary_path = npz_path.with_name(npz_path.stem + "_summary.json")
        if summary_path.exists():
            name = json.loads(summary_path.read_text())["name"]
        color = _COLORS.get(name, None)

        data = np.load(npz_path, allow_pickle=True)
        direct_traces = data["direct_traces"]
        composite_traces = data["composite_traces"]

        if len(direct_traces) == 0:
            continue
        mean, sem = _mean_and_sem(direct_traces)
        evals = np.arange(1, len(mean) + 1)
        ax.plot(evals, mean, linestyle="-", color=color, linewidth=2.0, label=f"{name.split(' / ')[0]} (direct)")
        ax.fill_between(evals, mean - sem, mean + sem, color=color, alpha=0.15, linewidth=0)
        plotted_any = True

        if len(composite_traces) == 0:
            continue
        mean_c, sem_c = _mean_and_sem(composite_traces)
        evals_c = np.arange(1, len(mean_c) + 1)
        ax.plot(
            evals_c, mean_c, linestyle="--", color=color, linewidth=2.0,
            label=f"{name.split(' / ')[-1]} (composite)",
        )
        ax.fill_between(evals_c, mean_c - sem_c, mean_c + sem_c, color=color, alpha=0.15, linewidth=0)

    if not plotted_any:
        raise ValueError(f"every result file in {results_dir} had zero completed trials")

    ax.set_xlabel("Total function evaluations")
    ax.set_ylabel("Dominated hypervolume")
    ax.set_title(title or results_dir.name)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Saved: {output.resolve()}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    output = args.output or (args.results_dir / "plot.png")
    plot_benchmark_dir(args.results_dir, output, title=args.title)


if __name__ == "__main__":
    main()
