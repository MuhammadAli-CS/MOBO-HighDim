r"""Direct-vs-composite ablation: does adding known composite structure
help each of three solver families (qLogEHVI, Chebyshev scalarization,
spherical-linear Chebyshev scalarization)?

Motivation: this project already has its own direct-vs-composite
comparison for MORBO (`morbo/problems/composite_dtlz2*.py`,
`experiments/composite_curve_dtlz2_100d/`, `experiments/
correlation_ablation_dtlz2curve/`) -- but not for the other solver
families a collaborator's repo implements
(https://github.com/tau315/composite-mobo). This script runs THOSE
solvers (vendored in `composite_ablation/solvers.py`) against either this
project's own `composite_dtlz2` benchmark (`--source ours`, from
`plug_and_play/benchmarks.py`) or that collaborator's own six-benchmark
suite (`--source tau`, vendored in `composite_ablation/tau_benchmarks.py`),
so the composite-structure question gets asked consistently across solver
families and benchmark suites, not just for MORBO on our own benchmark.

Usage:
    python -m composite_ablation.run_ablation --source ours --benchmark composite_dtlz2 \
        --dim 6 --num-objectives 5 --trials 10
    python -m composite_ablation.run_ablation --source tau --benchmark dtlz2_2obj_6d --trials 20
    python -m composite_ablation.run_ablation --source tau --benchmark projected_langermann_2obj_500d \
        --trials 20   # "high"-suite: runs the spherical pair only, see below

For `--source tau`, the solver pair(s) run are chosen automatically from
that benchmark's own `suite` field (mirroring the collaborator repo's own
low/high split): "low" runs standard_mobo/composite_mobo AND
chebyshev_bo/composite_chebyshev_bo; "high" runs
spherical_chebyshev_bo/composite_spherical_chebyshev_bo only (that pair is
the one actually meant for high-dimensional inputs; the low-dim pairs
would also technically run but are not a fair/intended test at e.g. 500D).
`--source ours` always runs every pair `--include-spherical` allows,
since this project's own benchmarks are picked per-call rather than
grouped into a fixed suite.

Only benchmarks exposing composite structure (`--source ours`:
`Benchmark.raw_eval_fn` set, currently just `composite_dtlz2`;
`--source tau`: all six always do) can run the composite half of each
pair -- others only report the direct solver's hypervolume.

Threading: caps `torch`'s intra-/inter-op thread pools (`--num-threads`/
`--num-interop-threads`, defaults 8/4) at the top of `main()`. Without
this, torch/MKL spawn threads sized to the NODE's full core count rather
than this job's actual `--cpus-per-task` allocation -- on a shared node
running several jobs at once this causes severe over-subscription
contention, not just slower-than-expected runs but outright failures:
three cluster jobs sharing a node with five others were observed to
either OOM-kill or have per-trial time balloon 10x over a run before
timing out entirely, while running alone on a node was fine. The exact
same class of problem (`torch`/MKL saturating far more threads than the
SLURM allocation) was already hit and fixed once before in this project,
in `run_comparison.py`, for one specific experiment -- this carries that
fix into this newer entry point instead of rediscovering it per-script.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from composite_ablation.adapters import MinimizeConventionProblem, to_minimize_convention
from composite_ablation.solvers import (
    chebyshev_bo,
    composite_chebyshev_bo,
    composite_mobo,
    composite_spherical_chebyshev_bo,
    simplex_weights,
    spherical_chebyshev_bo,
    standard_mobo,
)
from composite_ablation.tau_benchmarks import get_tau_benchmark
from plug_and_play.benchmarks import get_benchmark


def dominated_hypervolume_trace(Y_min: torch.Tensor, ref_point_min: torch.Tensor) -> np.ndarray:
    """Prefix dominated hypervolume for minimization-convention `Y_min`.

    Vendored from https://github.com/tau315/composite-mobo's
    `benchmark_common.py` (same repo `composite_ablation/solvers.py` is
    vendored from), unchanged, so traces are directly comparable to that
    repo's own plots.
    """
    values = -Y_min.detach().double().cpu()
    ref = -ref_point_min.detach().double().cpu()
    hypervolume = Hypervolume(ref_point=ref)
    trace = np.zeros(len(values), dtype=np.float64)
    for end in range(1, len(values) + 1):
        prefix = values[:end]
        valid = (prefix > ref).all(dim=-1)
        if valid.any():
            front = prefix[valid]
            front = front[is_non_dominated(front)]
            trace[end - 1] = float(hypervolume.compute(front))
    return trace


def _paired_summary(direct: np.ndarray, composite: np.ndarray) -> str:
    from scipy.stats import ttest_rel, wilcoxon

    delta = composite - direct
    rel = delta / np.clip(np.abs(direct), 1e-12, None)
    mean_rel = rel.mean() * 100
    win_rate = f"{(delta > 0).sum()}/{len(delta)}"
    try:
        _, wp = wilcoxon(direct, composite)
    except ValueError:
        wp = float("nan")
    _, tp = ttest_rel(composite, direct)
    return (
        f"composite - direct mean delta: {mean_rel:+.1f}%  win-rate {win_rate}  "
        f"Wilcoxon p={wp:.4g}  paired-t p={tp:.4g}"
    )


def run_pair(
    name: str,
    direct_fn,
    composite_fn,
    problem: MinimizeConventionProblem,
    *,
    trials: int,
    seed0: int,
    solver_kwargs: dict,
    out_dir: Path = None,
) -> dict:
    direct_finals, composite_finals = [], []
    direct_traces, composite_traces = [], []
    for trial in range(trials):
        seed = seed0 + 10_007 * trial
        started = time.perf_counter()
        if "weights" in solver_kwargs:
            kw = {k: v for k, v in solver_kwargs.items() if k != "weights"}
            direct_result = direct_fn(
                problem.evaluate, problem.dim, solver_kwargs["weights"], problem.ideal,
                seed=seed, **kw,
            )
        else:
            direct_result = direct_fn(
                problem.evaluate, problem.dim, problem.ref_point, seed=seed, **solver_kwargs
            )
        direct_trace = dominated_hypervolume_trace(direct_result.Y, problem.ref_point)
        direct_finals.append(direct_trace[-1])
        direct_traces.append(direct_trace)

        if problem.evaluate_components is None:
            print(f"{name:<28} trial={trial + 1:02d}/{trials:02d} "
                  f"direct HV={direct_trace[-1]:.6f} (no composite structure available)")
            continue

        if "weights" in solver_kwargs:
            kw = {k: v for k, v in solver_kwargs.items() if k != "weights"}
            composite_result = composite_fn(
                problem.evaluate, problem.evaluate_components, problem.compose,
                problem.dim, solver_kwargs["weights"], problem.ideal, seed=seed, **kw,
            )
        else:
            composite_result = composite_fn(
                problem.evaluate, problem.evaluate_components, problem.compose,
                problem.dim, problem.ref_point, seed=seed, **solver_kwargs,
            )
        composite_trace = dominated_hypervolume_trace(composite_result.Y, problem.ref_point)
        composite_finals.append(composite_trace[-1])
        composite_traces.append(composite_trace)
        elapsed = time.perf_counter() - started
        print(
            f"{name:<28} trial={trial + 1:02d}/{trials:02d} "
            f"direct HV={direct_trace[-1]:.6f}  composite HV={composite_trace[-1]:.6f}  "
            f"time={elapsed:.1f}s", flush=True,
        )
        del direct_result, composite_result
        gc.collect()

        if out_dir is not None:
            _save_traces(out_dir, name, direct_traces, composite_traces)

    summary = {"name": name, "direct_finals": direct_finals, "composite_finals": composite_finals}
    if composite_finals:
        line = _paired_summary(np.array(direct_finals), np.array(composite_finals))
        print(f"=== {name}: {line} ===\n", flush=True)
        summary["summary"] = line

    if out_dir is not None:
        _save_traces(out_dir, name, direct_traces, composite_traces)
        # Written per-pair (not one shared summary.json) so concurrent jobs
        # for different pairs of the SAME benchmark (now the norm -- see
        # --pair) never race on the same file.
        safe_name = name.replace(" ", "_").replace("/", "-")
        with open(out_dir / f"{safe_name}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    return summary


def _save_traces(out_dir: Path, name: str, direct_traces: list, composite_traces: list) -> None:
    """Write whatever trials have completed so far -- called after EVERY
    trial (not just at the end of a pair) so a mid-pair SLURM timeout still
    leaves usable partial results instead of losing the whole pair (this
    happened for real: the first `composite_ablation` cluster run allotted
    too little wall-clock time per job, and 4 of 7 jobs were killed
    mid-pair with nothing saved -- see cluster/submit_composite_ablation.sh's
    per-pair time budgets, since bumped up in response)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace(" ", "_").replace("/", "-")
    np.savez(
        out_dir / f"{safe_name}.npz",
        direct_traces=np.array(direct_traces, dtype=object),
        composite_traces=np.array(composite_traces, dtype=object),
    )


def _build_problem(args) -> tuple:
    """Returns (problem, num_objectives, suites_to_run: list[str])."""
    if args.source == "ours":
        bench = get_benchmark(args.benchmark, dim=args.dim, num_objectives=args.num_objectives)
        problem = to_minimize_convention(bench)
        return problem, bench.num_objectives, ["low"] + (["high"] if args.include_spherical else [])

    tau_bench = get_tau_benchmark(args.benchmark)
    problem = MinimizeConventionProblem(
        evaluate=tau_bench.evaluate,
        dim=tau_bench.dim,
        ref_point=tau_bench.ref_point,
        ideal=tau_bench.ideal,
        evaluate_components=tau_bench.evaluate_components,
        compose=tau_bench.compose,
    )
    suites = [tau_bench.suite] if not args.include_spherical else [tau_bench.suite, "high"]
    return problem, tau_bench.num_objectives, sorted(set(suites))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["ours", "tau"], default="ours")
    parser.add_argument("--benchmark", default="composite_dtlz2")
    parser.add_argument("--dim", type=int, default=6, help="--source ours only")
    parser.add_argument("--num-objectives", type=int, default=5, help="--source ours only")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=40)
    parser.add_argument("--weights", type=int, default=8)
    parser.add_argument("--per-weight", type=int, default=5)
    parser.add_argument(
        "--include-spherical", action="store_true",
        help="also run spherical_chebyshev_bo / composite_spherical_chebyshev_bo "
        "(the high-dimensional pair; slower, meant for dim >= ~20) even for a "
        "'low'-suite (--source tau) or generic (--source ours) benchmark.",
    )
    parser.add_argument("--quick", action="store_true", help="tiny smoke-test-scale run")
    parser.add_argument("--out-dir", type=Path, default=None, help="save raw HV traces here (.npz per pair)")
    parser.add_argument(
        "--pair", choices=["all", "standard", "chebyshev", "spherical"], default="all",
        help="run only one solver pair instead of every pair its suite allows -- lets each "
        "pair run as its own SLURM job (real per-trial costs turned out to be 300-550s, "
        "so 'all' in one job routinely blew past a 2h budget; see cluster/"
        "submit_composite_ablation.sh).",
    )
    parser.add_argument(
        "--num-threads", type=int, default=8,
        help="torch intra-op thread cap -- see module docstring's Threading section. "
        "Should not exceed this job's actual --cpus-per-task.",
    )
    parser.add_argument("--num-interop-threads", type=int, default=4, help="torch inter-op thread cap")
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(args.num_interop_threads)

    if args.quick:
        args.trials, args.n_init, args.n_iter = 1, 3, 2
        args.weights, args.per_weight = 2, 1

    problem, num_objectives, suites = _build_problem(args)
    print(
        f"source={args.source} benchmark={args.benchmark} dim={problem.dim} "
        f"num_objectives={num_objectives} suites={suites} pair={args.pair} "
        f"composite structure available: {problem.evaluate_components is not None}",
        flush=True,
    )

    all_summaries = []
    if "low" in suites and args.pair in ("all", "standard"):
        all_summaries.append(run_pair(
            "standard_mobo / composite_mobo",
            standard_mobo, composite_mobo, problem,
            trials=args.trials, seed0=args.seed,
            solver_kwargs=dict(n_init=args.n_init, n_iter=args.n_iter),
            out_dir=args.out_dir,
        ))

    if "low" in suites and args.pair in ("all", "chebyshev"):
        weights = simplex_weights(args.weights, num_objectives, seed=314159)
        all_summaries.append(run_pair(
            "chebyshev_bo / composite_chebyshev_bo",
            chebyshev_bo, composite_chebyshev_bo, problem,
            trials=args.trials, seed0=args.seed,
            solver_kwargs=dict(weights=weights, n_init=args.n_init, n_per_scalarization=args.per_weight),
            out_dir=args.out_dir,
        ))

    if "high" in suites and args.pair in ("all", "spherical"):
        weights = simplex_weights(args.weights, num_objectives, seed=314159)
        all_summaries.append(run_pair(
            "spherical_chebyshev_bo / composite_spherical_chebyshev_bo",
            spherical_chebyshev_bo, composite_spherical_chebyshev_bo, problem,
            trials=args.trials, seed0=args.seed,
            solver_kwargs=dict(weights=weights, n_init=args.n_init, n_per_scalarization=args.per_weight),
            out_dir=args.out_dir,
        ))


if __name__ == "__main__":
    main()
